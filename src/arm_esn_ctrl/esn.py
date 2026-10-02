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

Before they enter the ESN, joint angles are normalized by the range each joint
covers in the training data, so that the training data span [-1, 1], and the
outputs are converted back. Unlike the mean and standard deviation, the range does
not depend on how long the demonstrations hold still.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rclib
import tomli_w
from numpy.typing import ArrayLike, NDArray


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
    seed: int  # random seed of the reservoir and input weights

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
            )
        )
        self.model.set_readout(rclib.readouts.Ridge(alpha=config.ridge, include_bias=True))
        self.center = np.zeros(0)  # middle of each joint's range in the training data
        self.half_range = np.ones(0)  # half of that range

    def fit(self, trajectories: Sequence[NDArray[np.float64]]) -> None:
        """Train the readout on joint-angle trajectories, each shaped ``(n_samples, n_joints)``."""
        all_samples = np.vstack(trajectories)
        low, high = all_samples.min(axis=0), all_samples.max(axis=0)
        self.center = (low + high) / 2
        # A joint that never moves in the training data is only centered.
        self.half_range = np.where(high > low, (high - low) / 2, 1.0)
        sequences = [self._with_warmup(self._normalize(q)) for q in trajectories]
        # Each input predicts the next sample; every sequence's warm-up is washed out.
        self.model.fit_sequences(
            [sequence[:-1] for sequence in sequences],
            [sequence[1:] for sequence in sequences],
            washout_len=self.config.warmup_steps,
        )

    def one_step_predictions(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict each next sample of ``q`` from the samples up to it (teacher forcing).

        Returns the predictions of ``q[1:]``, for checking how well the readout fits.
        """
        sequence = self._with_warmup(self._normalize(q))
        predictions = self.model.predict(sequence[:-1], reset_state_before_predict=True)
        return self._denormalize(predictions[self.config.warmup_steps :])

    def generate(self, start_q: ArrayLike, n_steps: int) -> NDArray[np.float64]:
        """Run the ESN autonomously from the start posture ``start_q`` for ``n_steps`` steps.

        Returns the trajectory, shaped ``(n_steps + 1, n_joints)``, beginning with ``start_q``.
        """
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
        """
        start = self._normalize(np.asarray(start_q, dtype=np.float64)[np.newaxis, :])
        held = self._denormalize(start)[0]
        self.model.reset_reservoirs()
        for _ in range(self.config.warmup_steps):
            self.model.predict_online(start)
            yield held
        output = self.model.predict_online(start)  # the start posture's own step gives the first output
        yield held
        while True:
            yield self._denormalize(output)[0]
            output = self.model.predict_online(output)

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
        return esn

    def _with_warmup(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Prepend the first sample, held for the warm-up period."""
        return np.vstack([np.repeat(u[:1], self.config.warmup_steps, axis=0), u])

    def _normalize(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return (q - self.center) / self.half_range

    def _denormalize(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        return u * self.half_range + self.center
