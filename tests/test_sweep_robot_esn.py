# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the ESN sweep on the robot with one task and with several."""

import sys
import tomllib

import numpy as np
import pytest
import tomli_w
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.sweeps import score_settings
from arm_esn_ctrl.tracking import tracking_gains

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from sweep_robot_esn import sweep_combination, sweep_context

CONFIG = REPO_ROOT / "experiments/demonstrations/reach_tvs.toml"
START = [18.2, 119.9]  # where every task's take starts (deg)
GOALS = {"near": [48.6, 97.2], "far": [10.0, 80.0]}  # where each task's take ends (deg), its target the hand there
ESN = {
    "dt": 0.01,
    "warmup": 0.5,
    "n_neurons": 200,
    "spectral_radius": 1.3,
    "sparsity": 0.1,
    "leak_rate": 0.05,
    "input_scaling": 0.1,
    "bias": True,
    "ridge": 1e-6,
    "seed": 0,
}
TRACKER = {"laws": ["computed_torque"], "omegas": [40.0], "acceleration_filter": 0.02}
EVALUATION = {"duration": 2.5, "hold": 0.5, "start_offsets_deg": [[0.0, 0.0]]}


def joint_reach(start, goal, n=200):
    """A take sampled every 10 ms: a smooth reach in joint space over 1 s, then the goal held."""
    tau = np.clip(np.linspace(0.0, 2.0, n), 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return np.radians(start) + np.outer(s, np.radians(goal) - np.radians(start))


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """A scratch storage root with a run of one take taught by hand for each task, each run with its own target."""
    with CONFIG.open("rb") as f:
        demo_config = tomllib.load(f)
    del demo_config["controller"]
    skeleton = Skeleton.from_toml(CONFIG)
    for name, goal in GOALS.items():
        q = joint_reach(START, goal)
        target = endpoint_positions(skeleton, q[-1:])[0]
        demo_config["task"]["target"] = {"pos": target.tolist(), "tolerance": 0.02}
        run = tmp_path / "results" / "demonstrations" / name
        run.mkdir(parents=True)
        (run / "config.toml").write_text(tomli_w.dumps(demo_config))
        log = StateLog(skeleton)
        for k, posture in enumerate(q):
            skeleton.q = posture
            log.record_skeleton(skeleton, 0.01 * k)
        log.save(run / "demo_00.sklog.npz")
    monkeypatch.setenv("ARM_ESN_CTRL_STORAGE_ROOT", str(tmp_path))
    return tmp_path


def sweep_config(*tasks):
    """A sweep's configuration: a [[demonstrations]] table per task, or with one task, a [demonstrations] table."""
    tables = [
        {"run": f"results/demonstrations/{name}", "train": ["demo_00.sklog.npz"], "context": name} for name in tasks
    ]
    demonstrations = tables if len(tasks) > 1 else {key: tables[0][key] for key in ("run", "train")}
    return {"demonstrations": demonstrations, "esn": ESN, "tracker": TRACKER, "evaluation": EVALUATION}


def test_a_sweep_of_two_tasks_trains_one_esn_on_both_and_runs_each_toward_its_own_target(storage):
    context = sweep_context(sweep_config("near", "far"))

    row, runs = sweep_combination(context, {"leak_rate": 0.05})

    # Both takes start from the same posture: only the context tells the ESN which target to go to.
    assert [run["task"] for run in runs] == ["near", "far"]
    assert list(runs[0])[:4] == ["task", "setting", "start", "origin"]
    assert all(run["success"] and run["final_distance_m"] < 0.02 for run in runs)
    assert row["runs"] == 2
    assert row["failures"] == 0


def test_the_context_scaling_is_swept_as_any_other_hyperparameter(storage):
    context = sweep_context(sweep_config("near", "far"))

    (_, scaled), (_, unscaled) = (sweep_combination(context, {"context_scaling": value}) for value in (3.0, 1.0))

    assert scaled[1]["taught_path_rmse_m"] != unscaled[1]["taught_path_rmse_m"]


def test_a_sweep_of_two_tasks_sets_joint_pds_gains_where_the_takes_of_both_end(storage):
    config = sweep_config("near", "far") | {"tracker": {"laws": ["pd"], "omegas": [20.0], "acceleration_filter": 0.02}}

    context = sweep_context(config)

    tracker_config, gains = context.trackers["pd_w20"]
    # The mean of the takes' final postures.
    posture = np.radians(np.mean(list(GOALS.values()), axis=0))
    for got, expected in zip(gains, tracking_gains(tracker_config, context.setups[0].skeleton, posture), strict=True):
        np.testing.assert_allclose(got, expected)


def test_a_single_task_sweep_has_no_task_column_and_no_scores_of_each_task(storage):
    context = sweep_context(sweep_config("near"))

    row, runs = sweep_combination(context, {"leak_rate": 0.05})

    assert context.names == ()
    assert list(runs[0])[:3] == ["setting", "start", "origin"]
    assert "task" not in runs[0]
    assert row["runs"] == 1
    (score,) = score_settings([{"leak_rate": 0.05} | run for run in runs], ["leak_rate"])
    assert list(score) == ["leak_rate", "runs", "holding_runs", "median_detour_ratio"]
