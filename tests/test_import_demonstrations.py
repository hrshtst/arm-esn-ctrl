# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for importing takes taught by hand into demonstration runs, of one target or of several."""

import csv
import sys
import tomllib

import numpy as np
import pytest
import tomli_w
from skelarm import Skeleton, StateLog, Task, compute_inverse_kinematics

from arm_esn_ctrl.autonomous import load_setups
from arm_esn_ctrl.demonstrations import endpoint_positions, smooth_take
from arm_esn_ctrl.storage import ENV_VAR, REPO_ROOT

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import import_demonstrations

TARGETS = REPO_ROOT / "experiments/demonstrations/reach_manual_v2_targets.toml"
ONE_TARGET = REPO_ROOT / "experiments/demonstrations/reach_manual_v2_no_limits.toml"
TAKES = "data/reach_manual_v2_targets"  # where the takes of TARGETS are, under the storage root


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def target(index):
    """The position of a target of TARGETS (m)."""
    return load(TARGETS)["task"]["targets"][index]["pos"]


def recorded_task(active, **changes):
    """The [task] that a take records, as the recorder saves it: TARGETS's, with the target active in the take."""
    return Task.from_dict(load(TARGETS)["task"]).to_dict() | {"active": active} | changes


def taught_take(start_deg, hand, task):
    """A take as the recorder saves it: at rest at the start, a minimum-jerk reach that puts the hand at ``hand``,
    then still for 3 s; sampled near 100 Hz. ``task`` is the [task] it records.
    """
    skeleton = Skeleton.from_toml(TARGETS)
    start = np.radians(start_deg)
    skeleton.q = start
    end = compute_inverse_kinematics(skeleton.clone(), tuple(hand)).q
    rng = np.random.default_rng(0)
    times = np.cumsum(np.concatenate([[0.0], rng.uniform(0.009, 0.012, 500)]))
    times = times[times <= 4.5]
    tau = np.clip(times - 0.5, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    log = StateLog(skeleton, extra={"playback": {"task": task}})
    for t, q in zip(times, start + np.outer(s, end - start), strict=True):
        skeleton.q = q
        log.record_skeleton(skeleton, float(t))
    return log


@pytest.fixture
def storage(tmp_path, monkeypatch):
    """A scratch storage root, with the directory of TARGETS's takes."""
    root = tmp_path / "store"
    (root / TAKES).mkdir(parents=True)
    monkeypatch.setenv(ENV_VAR, str(root))
    return root


def import_takes(config, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["import_demonstrations.py", str(config)])
    import_demonstrations.main()


def runs(storage):
    return sorted((storage / "results" / "demonstrations").iterdir())


def metrics(run):
    with (run / "metrics.csv").open() as f:
        return list(csv.DictReader(f))


def joint_angles(path):
    """The joint angles of a demonstration's file (rad)."""
    return import_demonstrations.joint_angles(StateLog.load(path))


def test_takes_of_two_targets_found_by_the_pattern_are_imported_into_one_run_per_target(storage, monkeypatch, capsys):
    starts = [[13.0, 71.0], [133.0, 71.0], [73.0, 71.0]]
    for i, (start, active) in enumerate(zip(starts, [0, 1, 0], strict=True), start=1):
        taught_take(start, target(active), recorded_task(active)).save(storage / TAKES / f"take_{i:03d}.sklog.npz")
    (storage / TAKES / "notes.txt").write_text("")  # not a take: the pattern leaves it out

    import_takes(TARGETS, monkeypatch)

    run_a, run_b = runs(storage)
    assert run_a.name.endswith("-reach_manual_v2_targets_a")
    assert run_b.name.endswith("-reach_manual_v2_targets_b")
    config, a, b = load(TARGETS), load(run_a / "config.toml"), load(run_b / "config.toml")
    # Each run's configuration is one of its target alone, with its takes and where they begin.
    assert a["task"] == {"type": "reaching", "target": config["task"]["targets"][0], "duration": 4.0}
    assert b["task"] == {"type": "reaching", "target": config["task"]["targets"][1], "duration": 4.0}
    assert a["recording"]["takes"] == [f"{TAKES}/take_001.sklog.npz", f"{TAKES}/take_003.sklog.npz"]
    assert b["recording"]["takes"] == [f"{TAKES}/take_002.sklog.npz"]
    np.testing.assert_allclose(a["demonstrations"]["start_q"], [starts[0], starts[2]], atol=1e-9)
    np.testing.assert_allclose(b["demonstrations"]["start_q"], [starts[1]], atol=1e-9)
    for key in ("skeleton", "initial", "simulator"):
        assert a[key] == b[key] == config[key]
    assert {key: a["recording"][key] for key in ("hold", "max_tip_speed", "filter")} == {
        key: config["recording"][key] for key in ("hold", "max_tip_speed", "filter")
    }
    assert load(run_a / "run.toml")["config"] == "experiments/demonstrations/reach_manual_v2_targets.toml"
    # The takes, in the order of their names, as recorded and filtered, with their measurements and figure.
    assert sorted(path.name for path in run_a.glob("demo_*")) == [
        "demo_00.sklog.npz",
        "demo_00_filtered.sklog.npz",
        "demo_01.sklog.npz",
        "demo_01_filtered.sklog.npz",
    ]
    assert (run_a / "demo_01.sklog.npz").read_bytes() == (storage / TAKES / "take_003.sklog.npz").read_bytes()
    rows = metrics(run_a)
    assert [row["demo"] for row in rows] == ["demo_00", "demo_00_filtered", "demo_01", "demo_01_filtered"]
    assert [row["take"] for row in rows] == [f"{TAKES}/take_001.sklog.npz"] * 2 + [f"{TAKES}/take_003.sklog.npz"] * 2
    assert (run_a / "demonstrations.png").exists()
    out = capsys.readouterr().out
    assert f"Imported 2 takes of target A to {run_a}" in out
    assert f"Imported 1 take of target B to {run_b}" in out


def test_a_takes_target_is_the_one_its_log_records_whatever_the_configurations_active(tmp_path, storage, monkeypatch):
    config = load(TARGETS)
    config["task"]["active"] = 2  # C
    path = tmp_path / "demonstrations" / TARGETS.name
    path.parent.mkdir()
    path.write_text(tomli_w.dumps(config))
    taught_take([13.0, 71.0], target(1), recorded_task(1)).save(storage / TAKES / "take_001.sklog.npz")

    import_takes(path, monkeypatch)

    (run,) = runs(storage)
    assert run.name.endswith("-reach_manual_v2_targets_b")
    assert load(run / "config.toml")["task"]["target"] == config["task"]["targets"][1]


def test_a_take_that_records_no_target_or_other_targets_is_not_imported_and_says_why(storage, monkeypatch, capsys):
    targets = load(TARGETS)["task"]["targets"]
    moved = [targets[0], targets[1] | {"pos": [0.0, 0.69]}, targets[2]]
    wider = [targets[0] | {"tolerance": 0.03}, targets[1], targets[2]]
    one_target = {"type": "reaching", "target": targets[0], "duration": 4.0}  # as a take of stage 1 records
    for i, task in enumerate([one_target, recorded_task(1, targets=moved), recorded_task(0, targets=wider)], start=1):
        taught_take([13.0, 71.0], target(task.get("active", 0)), task).save(storage / TAKES / f"take_{i:03d}.sklog.npz")
    taught_take([13.0, 71.0], target(0), recorded_task(0)).save(storage / TAKES / "take_004.sklog.npz")

    with pytest.raises(SystemExit):
        import_takes(TARGETS, monkeypatch)

    assert not (storage / "results").exists()
    problems = [line for line in capsys.readouterr().out.splitlines() if line.startswith("  - ")]
    assert problems == [
        "  - take_001.sklog.npz: the take records no multi-target [task], so not which target it reaches",
        "  - take_002.sklog.npz: the take records other targets than the configuration's [task]",
        "  - take_003.sklog.npz: the take records other targets than the configuration's [task]",
    ]


def test_one_failing_take_keeps_every_take_from_being_imported(storage, monkeypatch, capsys):
    taught_take([13.0, 71.0], target(0), recorded_task(0)).save(storage / TAKES / "take_001.sklog.npz")
    short = np.array(target(1)) - [0.0, 0.05]  # the hand stops 5 cm short of B
    taught_take([133.0, 71.0], short, recorded_task(1)).save(storage / TAKES / "take_002.sklog.npz")

    with pytest.raises(SystemExit):
        import_takes(TARGETS, monkeypatch)

    assert not (storage / "results").exists()
    out = capsys.readouterr().out
    assert out.startswith("Not imported: ")
    problems = [line for line in out.splitlines() if line.startswith("  - ")]
    assert problems == [
        "  - demo_00 of target B (take_002.sklog.npz): the hand never comes within 0.02 m of the target",
        "  - demo_00_filtered of target B (take_002.sklog.npz): the hand never comes within 0.02 m of the target",
    ]


def test_a_pattern_that_no_take_matches_is_an_error(storage, monkeypatch):
    (storage / TAKES / "take.sklog.npz").write_text("")  # the recorder's --output, which it numbers

    with pytest.raises(FileNotFoundError, match="no take"):
        import_takes(TARGETS, monkeypatch)

    assert not (storage / "results").exists()


def test_the_runs_of_two_targets_load_as_two_tasks_each_with_its_own_target_and_start(storage, monkeypatch):
    taught_take([13.0, 71.0], target(0), recorded_task(0)).save(storage / TAKES / "take_001.sklog.npz")
    taught_take([133.0, 71.0], target(1), recorded_task(1)).save(storage / TAKES / "take_002.sklog.npz")
    import_takes(TARGETS, monkeypatch)
    run_a, run_b = runs(storage)

    setups = load_setups(
        {
            "demonstrations": [
                {"run": f"results/demonstrations/{run.name}", "train": ["demo_00.sklog.npz"], "context": context}
                for run, context in ((run_a, "a"), (run_b, "b"))
            ],
            "esn": {"dt": 0.01},
            "evaluation": {"duration": 1.0, "hold": 0.5, "start_offsets_deg": [[0.0, 0.0]]},
        }
    )

    assert [setup.target.tolist() for setup in setups] == [target(0), target(1)]
    assert [setup.radius for setup in setups] == [0.02, 0.02]
    assert [setup.task["target"]["label"] for setup in setups] == ["A", "B"]
    assert all(setup.q_refs is None for setup in setups)  # taught by hand: no demonstrator
    starts = [np.degrees(setup.starts[0].q) for setup in setups]
    np.testing.assert_allclose(starts, [[13.0, 71.0], [133.0, 71.0]], atol=1e-9)


def test_a_targets_run_imports_again_from_its_own_configuration_into_the_same_demonstrations(
    tmp_path, storage, monkeypatch
):
    """Each target's run is what a configuration of that target alone, with its takes and starts, imports."""
    taught_take([13.0, 71.0], target(0), recorded_task(0)).save(storage / TAKES / "take_001.sklog.npz")
    taught_take([73.0, 71.0], target(0), recorded_task(0)).save(storage / TAKES / "take_002.sklog.npz")
    import_takes(TARGETS, monkeypatch)
    (run,) = runs(storage)
    again = tmp_path / "demonstrations" / "again.toml"
    again.parent.mkdir()
    again.write_text((run / "config.toml").read_text())

    import_takes(again, monkeypatch)

    (other,) = [each for each in runs(storage) if each != run]
    assert other.name.endswith("-again")
    assert metrics(other) == metrics(run)
    for name in ("demo_00_filtered", "demo_01_filtered"):
        first, second = StateLog.load(run / f"{name}.sklog.npz"), StateLog.load(other / f"{name}.sklog.npz")
        np.testing.assert_array_equal(first.channel("q"), second.channel("q"))


def test_a_take_recorded_a_turn_up_in_one_joint_is_imported_within_a_turn_with_its_hand_path_and_its_move(
    storage, monkeypatch, capsys
):
    take = taught_take([13.0, 71.0 + 360.0], target(0), recorded_task(0))
    take.save(storage / TAKES / "take_001.sklog.npz")

    import_takes(TARGETS, monkeypatch)

    (run,) = runs(storage)
    np.testing.assert_allclose(load(run / "config.toml")["demonstrations"]["start_q"], [[13.0, 71.0]], atol=1e-9)
    # The take as recorded, a turn down in joint 2 at every sample: the same velocities and hand path.
    recorded, imported = import_demonstrations.joint_angles(take), joint_angles(run / "demo_00.sklog.npz")
    np.testing.assert_allclose(imported, recorded - [0.0, 2 * np.pi], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(StateLog.load(run / "demo_00.sklog.npz").channel("dq"), take.channel("dq"))
    skeleton = Skeleton.from_toml(TARGETS)
    hand = endpoint_positions(skeleton, recorded)
    np.testing.assert_allclose(endpoint_positions(skeleton, imported), hand, rtol=0, atol=1e-12)
    # The filtered demonstration is the take filtered, a turn down in joint 2, along the same hand path.
    smoothed = import_demonstrations.joint_angles(smooth_take(take, skeleton, {}, load(TARGETS)["recording"]["filter"]))
    filtered = joint_angles(run / "demo_00_filtered.sklog.npz")
    np.testing.assert_allclose(filtered, smoothed - [0.0, 2 * np.pi], rtol=0, atol=1e-9)
    np.testing.assert_allclose(
        endpoint_positions(skeleton, filtered), endpoint_positions(skeleton, smoothed), rtol=0, atol=1e-9
    )
    # Its move in the measurements, which still name and hash the take's own file.
    rows = metrics(run)
    assert [(row["turns_q1"], row["turns_q2"]) for row in rows] == [("0", "1")] * 2
    assert [float(row["start_q2_deg"]) for row in rows] == pytest.approx([71.0] * 2)
    assert {row["take_sha256"] for row in rows} == {
        import_demonstrations.sha256_of(storage / TAKES / "take_001.sklog.npz")
    }
    assert "The take (take_001.sklog.npz) is moved by -360 deg in joint 2" in capsys.readouterr().out


def test_two_takes_of_the_same_motion_a_turn_apart_are_imported_as_the_same_joint_angles(storage, monkeypatch):
    taught_take([13.0, 71.0], target(0), recorded_task(0)).save(storage / TAKES / "take_001.sklog.npz")
    taught_take([13.0 - 360.0, 71.0 + 360.0], target(0), recorded_task(0)).save(storage / TAKES / "take_002.sklog.npz")

    import_takes(TARGETS, monkeypatch)

    (run,) = runs(storage)
    for first, second in (("demo_00", "demo_01"), ("demo_00_filtered", "demo_01_filtered")):
        np.testing.assert_allclose(
            joint_angles(run / f"{first}.sklog.npz"), joint_angles(run / f"{second}.sklog.npz"), rtol=0, atol=1e-9
        )
    np.testing.assert_allclose(load(run / "config.toml")["demonstrations"]["start_q"], [[13.0, 71.0]] * 2, atol=1e-9)
    assert [(row["turns_q1"], row["turns_q2"]) for row in metrics(run)] == [("0", "0")] * 2 + [("-1", "1")] * 2


def test_without_the_key_a_take_a_turn_up_is_imported_as_recorded_and_its_demonstration_is_a_copy_of_the_take(
    tmp_path, storage, monkeypatch, capsys
):
    config = load(TARGETS)
    del config["recording"]["start_within_a_turn"]
    path = tmp_path / "demonstrations" / TARGETS.name
    path.parent.mkdir()
    path.write_text(tomli_w.dumps(config))
    taught_take([13.0, 71.0 + 360.0], target(0), recorded_task(0)).save(storage / TAKES / "take_001.sklog.npz")

    import_takes(path, monkeypatch)

    (run,) = runs(storage)
    text = (run / "config.toml").read_text()
    assert "begins (deg), as recorded.\n" in text
    np.testing.assert_allclose(tomllib.loads(text)["demonstrations"]["start_q"], [[13.0, 431.0]], atol=1e-9)
    assert (run / "demo_00.sklog.npz").read_bytes() == (storage / TAKES / "take_001.sklog.npz").read_bytes()
    assert np.degrees(joint_angles(run / "demo_00_filtered.sklog.npz")[0]) == pytest.approx([13.0, 431.0], abs=0.01)
    assert not any(key.startswith("turns_q") for key in metrics(run)[0])
    assert "is moved by" not in capsys.readouterr().out


def test_a_take_that_already_starts_within_a_turn_is_not_changed_by_the_key(storage, monkeypatch, capsys):
    taught_take([100.0, -71.0], target(1), recorded_task(1)).save(storage / TAKES / "take_001.sklog.npz")

    import_takes(TARGETS, monkeypatch)

    (run,) = runs(storage)
    np.testing.assert_allclose(load(run / "config.toml")["demonstrations"]["start_q"], [[100.0, -71.0]], atol=1e-9)
    assert (run / "demo_00.sklog.npz").read_bytes() == (storage / TAKES / "take_001.sklog.npz").read_bytes()
    assert [(row["turns_q1"], row["turns_q2"]) for row in metrics(run)] == [("0", "0")] * 2
    assert "is moved by" not in capsys.readouterr().out


def test_a_configuration_that_lists_its_takes_and_starts_imports_them_into_one_run_as_before(
    tmp_path, storage, monkeypatch, capsys
):
    config = load(ONE_TARGET)
    config["recording"]["takes"] = ["data/listed/take_00.sklog.npz", "data/listed/take_01.sklog.npz"]
    config["demonstrations"]["start_q"] = [[13.0, 71.0], [73.0, 71.0]]
    path = tmp_path / "demonstrations" / "listed.toml"
    path.parent.mkdir()
    text = "# Two takes of one target.\n" + tomli_w.dumps(config)
    path.write_text(text)
    (storage / "data/listed").mkdir()
    for take, start in zip(config["recording"]["takes"], config["demonstrations"]["start_q"], strict=True):
        taught_take(start, config["task"]["target"]["pos"], config["task"]).save(storage / take)

    import_takes(path, monkeypatch)

    (run,) = runs(storage)
    assert run.name.endswith("-listed")
    assert (run / "config.toml").read_text() == text
    assert [row["take"] for row in metrics(run)] == ["data/listed/take_00.sklog.npz"] * 2 + [
        "data/listed/take_01.sklog.npz"
    ] * 2
    assert f"Imported 2 takes to {run}" in capsys.readouterr().out
