# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the ESN trained on one trajectory and run autonomously."""

import dataclasses

import numpy as np
import pytest
from numpy.typing import NDArray

from arm_esn_ctrl.esn import EsnConfig, ReachingEsn

DT = 0.01
CONFIG = EsnConfig(
    dt=DT,
    warmup=1.0,
    n_neurons=300,
    spectral_radius=0.5,
    sparsity=0.1,
    leak_rate=0.3,
    input_scaling=1.0,
    bias=True,
    ridge=1.0,
    seed=0,
)


def joint_reach(start=(0.3, 2.0), goal=(0.8, 1.6)) -> NDArray[np.float64]:
    """A two-joint minimum-jerk reach lasting 1 s, then 1 s of holding still."""
    tau = np.clip(np.arange(0.0, 2.0 + DT / 2, DT) / 1.0, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    start_q = np.asarray(start, dtype=np.float64)
    return start_q + np.outer(s, np.asarray(goal, dtype=np.float64) - start_q)


def test_warmup_is_converted_to_samples():
    assert CONFIG.warmup_steps == 100


def test_autonomous_run_reproduces_the_training_trajectory():
    q = joint_reach()
    esn = ReachingEsn(CONFIG)
    esn.fit([q])

    generated = esn.generate(q[0], len(q) - 1)

    assert generated.shape == q.shape
    assert generated[0] == pytest.approx(q[0])
    assert np.degrees(np.sqrt(np.mean((generated - q) ** 2))) < 2.0
    assert np.degrees(np.abs(generated[-1] - q[-1])).max() < 0.5


def test_autonomous_run_stays_at_the_goal_after_the_trajectory_ends():
    q = joint_reach()
    esn = ReachingEsn(CONFIG)
    esn.fit([q])

    generated = esn.generate(q[0], 3 * (len(q) - 1))

    assert np.degrees(np.abs(generated[len(q) :] - q[-1])).max() < 0.5


def test_runs_restart_from_a_reset_reservoir():
    q = joint_reach()
    esn = ReachingEsn(CONFIG)
    esn.fit([q])

    assert esn.generate(q[0], 50) == pytest.approx(esn.generate(q[0], 50))


def test_each_training_trajectory_is_reproduced_from_its_own_start():
    trajectories = [joint_reach(start) for start in ((0.3, 2.0), (1.0, 1.3), (0.2, 1.4))]
    esn = ReachingEsn(CONFIG)
    esn.fit(trajectories)

    for q in trajectories:
        generated = esn.generate(q[0], len(q) - 1)
        assert np.degrees(np.sqrt(np.mean((generated - q) ** 2))) < 2.0
        assert np.degrees(np.abs(generated[-1] - q[-1])).max() < 0.5


def test_normalization_does_not_depend_on_how_long_the_hold_lasts():
    q = joint_reach()
    longer_hold = np.vstack([q, np.repeat(q[-1:], 200, axis=0)])
    short, long = ReachingEsn(CONFIG), ReachingEsn(CONFIG)
    short.fit([q])
    long.fit([longer_hold])

    assert long.center == pytest.approx(short.center)
    assert long.half_range == pytest.approx(short.half_range)
    normalized = (q - short.center) / short.half_range
    assert normalized.min(axis=0) == pytest.approx([-1.0, -1.0])
    assert normalized.max(axis=0) == pytest.approx([1.0, 1.0])


def test_stream_yields_what_generate_computes():
    q = joint_reach()
    esn = ReachingEsn(CONFIG)
    esn.fit([q])
    n = 150

    stream = esn.stream(q[0])
    streamed = np.array([next(stream) for _ in range(CONFIG.warmup_steps + 1 + n)])

    assert streamed[: CONFIG.warmup_steps + 1] == pytest.approx(np.repeat(q[:1], CONFIG.warmup_steps + 1, axis=0))
    assert streamed[CONFIG.warmup_steps :] == pytest.approx(esn.generate(q[0], n), abs=1e-12)


def test_stepping_on_its_own_outputs_generates_the_autonomous_run():
    q = joint_reach()
    esn = ReachingEsn(CONFIG)
    esn.fit([q])
    n = 150

    esn.reset()
    for _ in range(CONFIG.warmup_steps):  # the warm-up: the held start posture
        esn.step(q[0])
    stepped = [q[0], esn.step(q[0])]  # the start posture's own step gives the first posture
    for _ in range(n - 1):
        stepped.append(esn.step(stepped[-1]))

    assert np.array(stepped) == pytest.approx(esn.generate(q[0], n), abs=1e-12)


def test_a_saved_esn_loads_and_runs_identically(tmp_path):
    q = joint_reach()
    esn = ReachingEsn(CONFIG)
    esn.fit([q])

    esn.save(tmp_path / "esn.toml")
    loaded = ReachingEsn.load(tmp_path / "esn.toml")

    assert (tmp_path / "esn.rclib").exists()
    assert loaded.config == CONFIG
    assert np.array_equal(loaded.center, esn.center)
    assert np.array_equal(loaded.half_range, esn.half_range)
    assert np.array_equal(loaded.generate(q[0], 200), esn.generate(q[0], 200))


def test_the_reservoir_state_is_readable_at_every_step():
    q = joint_reach()
    esn = ReachingEsn(CONFIG)
    esn.fit([q])

    esn.reset()
    reset = esn.state()
    esn.step(q[0])
    first = esn.state()
    esn.step(q[0])

    assert reset.shape == (CONFIG.n_neurons,) and np.all(reset == 0.0)
    assert np.linalg.norm(first) > 0.0
    assert not np.allclose(esn.state(), first)  # a copy: it does not change with later steps


def test_scaling_the_normalization_is_scaling_the_input():
    """Normalizing the training data to [-s, s] instead of [-1, 1] gives the runs of an s times larger input scaling.

    The input weights are linear, the reservoir's bias depends on neither, and the
    ridge readout scales with its target, so the normalization's scale is no
    hyperparameter of its own.
    """
    q = joint_reach()
    span = 0.3
    scaled = ReachingEsn(CONFIG)
    scaled.fit([q])
    # Refit with the training data normalized to [-span, span].
    scaled.half_range = scaled.half_range / span
    sequences = [scaled._with_warmup(scaled._normalize(q))]
    scaled.model.fit_sequences(
        [sequence[:-1] for sequence in sequences],
        [sequence[1:] for sequence in sequences],
        washout_len=CONFIG.warmup_steps,
    )
    plain = ReachingEsn(dataclasses.replace(CONFIG, input_scaling=CONFIG.input_scaling * span))
    plain.fit([q])

    start = q[0] + np.radians([5.0, -5.0])
    assert np.abs(scaled.generate(start, 100) - plain.generate(start, 100)).max() < 1e-9


def trained(config: EsnConfig, q: NDArray[np.float64]) -> ReachingEsn:
    esn = ReachingEsn(config)
    esn.fit([q])
    return esn


NOISY = dataclasses.replace(CONFIG, teacher_noise_deg=2.0, teacher_copies=4)


def test_teacher_noise_is_off_unless_asked_for():
    assert (CONFIG.teacher_noise_deg, CONFIG.teacher_copies) == (0.0, 1)


def test_noisy_teacher_forcing_is_reproducible_and_changes_the_readout():
    q = joint_reach()
    first, second = trained(NOISY, q).generate(q[0], 150), trained(NOISY, q).generate(q[0], 150)
    np.testing.assert_array_equal(first, second)
    assert not np.allclose(first, trained(CONFIG, q).generate(q[0], 150))


def test_noisy_teacher_forcing_steers_an_offset_input_back_toward_the_trajectory():
    """Fed the trajectory 2 deg off in joint 1, the next output keeps less of the offset after noisy training."""
    q = joint_reach()
    offset = np.radians([2.0, 0.0])

    def kept(esn: ReachingEsn) -> float:
        return float(np.median((esn.one_step_predictions(q + offset) - q[1:])[:, 0]) / offset[0])

    assert kept(trained(NOISY, q)) < 0.75 * kept(trained(CONFIG, q))


def test_an_esn_saved_before_the_teacher_noise_loads_without_it(tmp_path):
    q = joint_reach()
    path = tmp_path / "esn.toml"
    trained(CONFIG, q).save(path)
    text = path.read_text()
    path.write_text("".join(line for line in text.splitlines(keepends=True) if not line.startswith("teacher_")))
    loaded = ReachingEsn.load(path)
    assert (loaded.config.teacher_noise_deg, loaded.config.teacher_copies) == (0.0, 1)
    np.testing.assert_array_equal(loaded.generate(q[0], 150), trained(CONFIG, q).generate(q[0], 150))
