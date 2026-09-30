# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""An echo state network that generates the next joint angles from the current ones.

The ESN is trained by teacher forcing: its input is a demonstrated joint-angle
trajectory, and its target is the same trajectory one step ahead. Run
autonomously, its output is fed back as its next input, so the trained ESN
becomes a dynamical system that generates a trajectory by itself.

Before training and before every run, the ESN is driven for a warm-up period by
the start posture held still. This brings the reservoir from its reset state to
a state that reflects the start posture. The warm-up samples are excluded from
training, so the ESN learns only to move from the start posture, never to stay
there.

Joint angles are normalized with the mean and standard deviation of the
training data before they enter the ESN, and the outputs are converted back.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rclib
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
    """An ESN trained on a joint-angle trajectory and run autonomously.

    Parameters
    ----------
    config : EsnConfig
        The ESN hyperparameters.
    """

    def __init__(self, config: EsnConfig) -> None:
        self.config = config
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
        self.mean = np.zeros(0)
        self.std = np.ones(0)

    def fit(self, q: NDArray[np.float64]) -> None:
        """Train the readout on one joint-angle trajectory ``q`` (shape ``(n_samples, n_joints)``)."""
        self.mean = q.mean(axis=0)
        self.std = q.std(axis=0)
        sequence = self._with_warmup(self._normalize(q))
        # Each input predicts the next sample; the warm-up is washed out.
        self.model.fit(sequence[:-1], sequence[1:], washout_len=self.config.warmup_steps)

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

    def _with_warmup(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Prepend the first sample, held for the warm-up period."""
        return np.vstack([np.repeat(u[:1], self.config.warmup_steps, axis=0), u])

    def _normalize(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        return (q - self.mean) / self.std

    def _denormalize(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        return u * self.std + self.mean
