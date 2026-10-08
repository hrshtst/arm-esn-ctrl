# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""A trained ESN's reservoir states, recomputed rather than stored.

From the reset state, each step of the reservoir depends only on its previous
state and its input, and a trained ESN saves its weights (``esn.rclib``). So the
states of a run follow from its inputs, and the runs need not store them:

- :func:`teacher_forced_states`: while a demonstration is fed in after the held
  start posture, as in training;
- :func:`run_states`: in a robot run driven by the ESN (``experiments/robot_esn.py``),
  from its log, which keeps every posture the ESN was given.

Times are on the task clock: the warm-up runs at negative times, and the state at
time t has seen the input up to t.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from skelarm import StateLog

    from arm_esn_ctrl.esn import ReachingEsn


def teacher_forced_states(esn: ReachingEsn, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """The reservoir states while the demonstration ``q`` is fed in after the warm-up, as in training.

    One state per input: the held start posture for the warm-up, then every sample.
    """
    esn.reset()
    states = []
    for posture in [q[0]] * esn.config.warmup_steps + list(q):
        esn.step(posture)
        states.append(esn.state())
    return np.array(states)


def run_states(
    esn: ReachingEsn, log: StateLog, tolerance: float = 1e-8
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The reservoir states of a robot run that ``esn`` drove, recomputed from the run's ``log``.

    In such a run, the ESN is stepped at every reference instant ``k dt`` (its
    period), from the warm-up at ``k = -warmup_steps``, with the arm's measured
    posture, which the log keeps. Fed those postures from the reset state, the ESN
    goes through the run's states again. Its outputs must reproduce the logged
    reference: from the task's start, the reference reaches the output for instant
    ``k + 1`` at that instant. A log that misses an instant, or whose reference
    differs by more than ``tolerance`` (rad), was not driven by this ESN, and is
    refused with a ValueError.

    Returns the instants' times and the states, one row per instant.
    """
    dt, warmup = esn.config.dt, esn.config.warmup_steps
    times = log.times
    k = np.arange(-warmup, int(np.ceil(times[-1] / dt - 1e-9)))  # the instants before the log's end
    at = np.searchsorted(times, k * dt - 1e-9)
    if at[-1] >= len(times) or not np.allclose(times[at], k * dt, rtol=0.0, atol=1e-9):
        msg = f"the log does not sample every instant of the ESN's period {dt:g} s, from the warm-up on"
        raise ValueError(msg)
    q = log.channel("q").reshape(len(times), -1)
    q_ref = log.channel("q_ref").reshape(len(times), -1)
    esn.reset()
    outputs, states = [], []
    for i in at:
        outputs.append(esn.step(q[i]))
        states.append(esn.state())
    error = float(np.abs(np.array(outputs)[warmup:-1] - q_ref[at[warmup + 1 :]]).max())
    if error > tolerance:
        msg = f"the ESN's outputs differ from the logged reference by up to {error:.2g} rad: it did not drive this run"
        raise ValueError(msg)
    return (k * dt).astype(np.float64), np.array(states)
