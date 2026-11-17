# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""An echo state network that generates the next joint angles from the current ones.

The ESN is trained by teacher forcing: its input is a demonstrated joint-angle
trajectory, and its target is the same trajectory one step ahead. With several
demonstrations, each one runs from a reset reservoir, and one readout is fitted
on all of them together. Run autonomously, the ESN's output is fed back as its
next input, so the trained ESN becomes a dynamical system that generates a
trajectory by itself.

Before training on each demonstration and before every run, the ESN is driven
for a warm-up period by the start posture held still. This brings the reservoir
from its reset state to a state that reflects the start posture. The warm-up
samples are excluded from training, so the ESN learns only to move from the
start posture, never to stay there.

With noisy teacher forcing (``teacher_noise_deg``), each training trajectory is fed
in several times (``teacher_copies``), each time with fresh Gaussian noise added to
the input, while the target stays the clean trajectory one step ahead: from a
slightly wrong posture, the ESN learns to produce the right next one. This is the
noise immunization of Lukoševičius's practical guide to ESNs (2012, section 5.2.1),
which makes teacher-forced feedback stable. Every copy adds to the data the ridge
regularization is weighed against, so K copies weaken a given ridge K times.

The noise is white unless ``teacher_noise_exponent`` colors it: its power spectrum
is then proportional to 1/f^β, with β the exponent, such as 1 for pink noise and 2
for brown noise, which drifts like a random walk, or -1 for blue noise and -2 for
violet noise, whose power rises with the frequency. Colored noise is drawn white,
shaped in the frequency domain, without its mean, and scaled so that over each
copy of a trajectory its standard deviation in each joint is ``teacher_noise_deg``.

Before they enter the ESN, joint angles are normalized by the range each joint
covers in the training data, so that the training data span [-1, 1], and the
outputs are converted back. Unlike the mean and standard deviation, the range does
not depend on how long the demonstrations hold still.

An ESN can be trained on the demonstrations of several tasks, such as reaches to
different targets, or a reach and a periodic motion (``contexts`` of
:meth:`ReachingEsn.fit`). Its input then holds, after the joint angles, one value
per task, the context: 1 for the task of the demonstration being fed in and 0 for
the others (a one-hot vector), times ``context_scaling``. Nothing else tells the
ESN which task a demonstration belongs to, and its output is still the next joint
angles alone. Run on its own or on a robot, the ESN is given the context of the
task it should do (:meth:`ReachingEsn.set_context`), which can change at any step.
Without contexts, the input is the joint angles alone, as before.
"""

from __future__ import annotations

import itertools
import tomllib
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rclib
import tomli_w
from numpy.typing import ArrayLike, NDArray


def colored_noise(rng: np.random.Generator, shape: tuple[int, ...], exponent: float) -> NDArray[np.float64]:
    """Gaussian noise along the first axis of ``shape``, with a power spectrum proportional to 1/f^``exponent``.

    With the exponent 0, the noise is white, drawn with a standard deviation of 1.
    Otherwise it is white noise shaped in the frequency domain, without its mean
    (the frequency 0), and scaled to a standard deviation of 1 along the first axis.
    """
    white = rng.normal(size=shape)
    if exponent == 0:
        return white
    frequency = np.fft.rfftfreq(shape[0])
    gain = np.zeros_like(frequency)
    gain[1:] = frequency[1:] ** (-exponent / 2)
    gain = gain.reshape((-1,) + (1,) * (len(shape) - 1))
    noise = np.fft.irfft(np.fft.rfft(white, axis=0) * gain, n=shape[0], axis=0)
    return noise / noise.std(axis=0)


@dataclass(frozen=True)
class EsnConfig:
    """ESN hyperparameters, read from the ``[esn]`` table of a configuration file."""

    dt: float  # sampling period of the ESN (s)
    warmup: float  # how long the held start posture drives the ESN before it moves (s)
    n_neurons: int  # number of reservoir neurons
    spectral_radius: float  # largest absolute eigenvalue of the reservoir weight matrix
    sparsity: float  # fraction of nonzero reservoir connections (rclib's name for it)
    leak_rate: float  # fraction of a neuron's state replaced at each step, in (0, 1]
    input_scaling: float  # scale of the random input weights
    bias: bool  # whether each neuron receives a random constant input
    ridge: float  # ridge-regression regularization of the readout
    seed: int  # random seed of the reservoir and input weights, and of the teacher noise
    teacher_noise_deg: float = 0.0  # standard deviation of the noise on the teacher-forced input (deg)
    teacher_copies: int = 1  # how many times each training trajectory is fed in, each with its own noise
    teacher_noise_exponent: float = 0.0  # its power spectrum goes as 1/f to this: 0 white, 1 pink, 2 brown, -1 blue
    bias_scaling: float = 1.0  # each neuron's bias is drawn from [-1, 1] and multiplied by this
    context_scaling: float = 1.0  # the context input's value for the task being done (the others' is 0)

    @property
    def warmup_steps(self) -> int:
        """The warm-up period in samples."""
        return round(self.warmup / self.dt)


class ReachingEsn:
    """An ESN trained on joint-angle trajectories and run autonomously.

    Parameters
    ----------
    config : EsnConfig
        The ESN hyperparameters.
    model : rclib.ESN, optional
        An rclib model built with ``config``, such as one loaded by :meth:`load`;
        by default a new, untrained one is built.
    """

    def __init__(self, config: EsnConfig, model: rclib.ESN | None = None) -> None:
        self.config = config
        self.contexts: tuple[str, ...] = ()  # the names of the tasks the ESN was trained on; none without a context
        self.context = np.zeros(0)  # the context now: one weight per task, 1 for the task being done
        if model is not None:
            self.model = model
            self.center = np.zeros(0)  # middle of each joint's range in the training data
            self.half_range = np.ones(0)  # half of that range
            return
        self.model = rclib.ESN()
        self.model.add_reservoir(
            rclib.reservoirs.RandomSparse(
                n_neurons=config.n_neurons,
                spectral_radius=config.spectral_radius,
                sparsity=config.sparsity,
                leak_rate=config.leak_rate,
                input_scaling=config.input_scaling,
                include_bias=config.bias,
                seed=config.seed,
                bias_scaling=config.bias_scaling,
            )
        )
        self.model.set_readout(rclib.readouts.Ridge(alpha=config.ridge, include_bias=True))
        self.center = np.zeros(0)  # middle of each joint's range in the training data
        self.half_range = np.ones(0)  # half of that range

    def fit(
        self,
        trajectories: Sequence[NDArray[np.float64]],
        contexts: Sequence[int] | None = None,
        names: Sequence[str] | None = None,
    ) -> None:
        """Train the readout on joint-angle trajectories, each shaped ``(n_samples, n_joints)``.

        ``contexts`` gives the task of each trajectory, as an index from 0, for an ESN
        of several tasks (see the module); ``names`` names the tasks, in the order of
        their indices (by default ``"0"``, ``"1"``, ...). Afterwards the ESN's context
        is the first task's.
        """
        if contexts is not None and len(contexts) != len(trajectories):
            msg = f"{len(trajectories)} trajectories need {len(trajectories)} contexts, not {len(contexts)}"
            raise ValueError(msg)
        if contexts is None:
            self.contexts = ()
        else:
            n_tasks = max(contexts) + 1 if names is None else len(names)
            if min(contexts) < 0 or max(contexts) >= n_tasks:
                msg = f"the contexts must be task indices from 0 to {n_tasks - 1}"
                raise ValueError(msg)
            self.contexts = tuple(str(i) for i in range(n_tasks)) if names is None else tuple(names)
        self.context = self.context_vector(0) if self.contexts else np.zeros(0)
        all_samples = np.vstack(trajectories)
        low, high = all_samples.min(axis=0), all_samples.max(axis=0)
        self.center = (low + high) / 2
        # A joint that never moves in the training data is only centered.
        self.half_range = np.where(high > low, (high - low) / 2, 1.0)
        sequences = [self._with_warmup(self._normalize(q)) for q in trajectories]
        # Each input predicts the next sample; every sequence's warm-up is washed out.
        # The teacher noise, if any, is on the input's joint angles only: the target stays clean.
        rng = np.random.default_rng(self.config.seed)
        noise_std = np.radians(self.config.teacher_noise_deg) / self.half_range  # in normalized units
        inputs, targets = [], []
        for i, sequence in enumerate(sequences):
            context = None if contexts is None else self.context_vector(contexts[i])
            for _ in range(self.config.teacher_copies):
                noise = (
                    colored_noise(rng, sequence[:-1].shape, self.config.teacher_noise_exponent) * noise_std
                    if self.config.teacher_noise_deg
                    else 0.0
                )
                inputs.append(self._with_context(sequence[:-1] + noise, context))
                targets.append(sequence[1:])
        self.model.fit_sequences(inputs, targets, washout_len=self.config.warmup_steps)

    def context_vector(self, task: int | str | ArrayLike) -> NDArray[np.float64]:
        """The context of ``task``: one weight per task, 1 for the task and 0 for the others.

        ``task`` is a task's index or name, or the weights themselves, one per task,
        such as a context between two tasks.
        """
        if not self.contexts:
            msg = "the ESN was trained without contexts"
            raise ValueError(msg)
        if isinstance(task, str):
            if task not in self.contexts:
                msg = f"unknown task {task!r}; the ESN's tasks are {', '.join(self.contexts)}"
                raise ValueError(msg)
            task = self.contexts.index(task)
        if isinstance(task, (int, np.integer)):
            if not 0 <= task < len(self.contexts):
                msg = f"the task's index must be from 0 to {len(self.contexts) - 1}, not {task}"
                raise ValueError(msg)
            return np.eye(len(self.contexts))[int(task)]
        weights = np.asarray(task, dtype=np.float64)
        if weights.shape != (len(self.contexts),):
            msg = f"a context has one weight per task ({len(self.contexts)}), not the shape {weights.shape}"
            raise ValueError(msg)
        return weights.copy()

    def set_context(self, task: int | str | ArrayLike) -> None:
        """Give the ESN the context of ``task`` from its next step on (see :meth:`context_vector`).

        The reservoir keeps its state: the context can change in the middle of a run.
        """
        self.context = self.context_vector(task)

    def one_step_predictions(self, q: NDArray[np.float64], context: int | str | None = None) -> NDArray[np.float64]:
        """Predict each next sample of ``q`` from the samples up to it (teacher forcing).

        Returns the predictions of ``q[1:]``, for checking how well the readout fits.
        An ESN of several tasks is given the ``context`` of ``q``'s task (by default,
        its current context).
        """
        sequence = self._with_warmup(self._normalize(q))
        weights = None if not self.contexts else self.context if context is None else self.context_vector(context)
        predictions = self.model.predict(self._with_context(sequence[:-1], weights), reset_state_before_predict=True)
        return self._denormalize(predictions[self.config.warmup_steps :])

    def generate(self, start_q: ArrayLike, n_steps: int) -> NDArray[np.float64]:
        """Run the ESN autonomously from the start posture ``start_q`` for ``n_steps`` steps.

        Returns the trajectory, shaped ``(n_steps + 1, n_joints)``, beginning with ``start_q``.
        An ESN of several tasks runs with its current context (:meth:`set_context`).
        """
        if self.contexts:  # rclib feeds the output back as the whole input, which here has the context too
            steps = itertools.islice(self.stream(start_q), self.config.warmup_steps, None)
            return np.array(list(itertools.islice(steps, n_steps + 1)))
        start = self._normalize(np.asarray(start_q, dtype=np.float64)[np.newaxis, :])
        self.model.reset_reservoirs()
        # Prime with the held start posture, exactly as in training, then feed outputs back.
        generated = self.model.predict_generative(self._with_warmup(start), n_steps)
        return self._denormalize(np.vstack([start, generated]))

    def stream(self, start_q: ArrayLike) -> Iterator[NDArray[np.float64]]:
        """Run the ESN autonomously from ``start_q`` one step at a time, without end.

        Yields the posture at every step, as :meth:`generate` would compute it: the
        start posture held through the warm-up (``warmup_steps`` times, at times
        before 0), the start posture once more (time 0), and then each generated
        posture. Only the next step is computed, so a caller can run the ESN live.
        An ESN of several tasks is given its context at each step, as it is then
        (:meth:`set_context`), so the caller can change it while the ESN runs.
        """
        start = self._normalize(np.asarray(start_q, dtype=np.float64)[np.newaxis, :])
        held = self._denormalize(start)[0]
        self.model.reset_reservoirs()
        for _ in range(self.config.warmup_steps):
            self.model.predict_online(self._with_context(start, self.context))
            yield held
        # The start posture's own step gives the first output.
        output = self.model.predict_online(self._with_context(start, self.context))
        yield held
        while True:
            yield self._denormalize(output)[0]
            output = self.model.predict_online(self._with_context(output, self.context))

    def reset(self) -> None:
        """Reset the reservoir, as before every run."""
        self.model.reset_reservoirs()

    def step(self, q: ArrayLike) -> NDArray[np.float64]:
        """Drive the ESN one step with the joint angles ``q``, and return the next joint angles it gives.

        On a robot, ``q`` is the measured posture. After :meth:`reset`, the start
        posture drives the ESN ``warmup_steps`` times for the warm-up, and once more
        for its first output, as in training; fed its own outputs from then on, the
        ESN generates its autonomous run (:meth:`generate`). An ESN of several tasks
        is given its current context (:meth:`set_context`).
        """
        u = self._normalize(np.asarray(q, dtype=np.float64)[np.newaxis, :])
        return self._denormalize(self.model.predict_online(self._with_context(u, self.context)))[0]

    def state(self) -> NDArray[np.float64]:
        """The reservoir's current state, one value per neuron (a copy)."""
        return np.array(self.model.get_reservoir(0).getState(), dtype=np.float64).ravel()

    def save(self, path: str | Path) -> None:
        """Save the trained ESN to ``path``, a TOML file, and the rclib model beside it.

        The TOML file holds the hyperparameters and the normalization, readable by
        people. The rclib model file has the same name with the suffix ``.rclib`` and
        holds the reservoir, the readout weights, and the reservoir state.
        """
        path = Path(path)
        model_path = path.with_suffix(".rclib")
        self.model.save(model_path)
        record = {
            "model": model_path.name,
            "esn": asdict(self.config),
            "normalization": {"center_rad": self.center.tolist(), "half_range_rad": self.half_range.tolist()},
        }
        if self.contexts:
            record["contexts"] = list(self.contexts)  # the tasks' names, in the order of the context input
        header = "# A trained ESN of arm_esn_ctrl; load it with arm_esn_ctrl.esn.ReachingEsn.load(<this file>).\n"
        path.write_text(header + tomli_w.dumps(record), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ReachingEsn:
        """Load an ESN saved by :meth:`save` from its TOML file."""
        path = Path(path)
        with path.open("rb") as f:
            record = tomllib.load(f)
        esn = cls(EsnConfig(**record["esn"]), model=rclib.ESN.load(path.parent / record["model"]))
        esn.center = np.asarray(record["normalization"]["center_rad"], dtype=np.float64)
        esn.half_range = np.asarray(record["normalization"]["half_range_rad"], dtype=np.float64)
        esn.contexts = tuple(record.get("contexts", ()))
        if esn.contexts:
            esn.context = esn.context_vector(0)
        return esn

    def _with_warmup(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Prepend the first sample, held for the warm-up period."""
        return np.vstack([np.repeat(u[:1], self.config.warmup_steps, axis=0), u])

    def _with_context(self, u: NDArray[np.float64], context: NDArray[np.float64] | None) -> NDArray[np.float64]:
        """The ESN's input: the normalized joint angles ``u``, then the ``context`` scaled, on every row."""
        if not self.contexts or context is None:
            return u
        return np.hstack([u, np.tile(self.config.context_scaling * context, (len(u), 1))])

    def _normalize(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return (q - self.center) / self.half_range

    def _denormalize(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        return u * self.half_range + self.center
