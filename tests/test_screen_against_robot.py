# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the robot outcomes the screen is set against: report 011's rule, and the offset starts counted."""

import csv
import sys

from arm_esn_ctrl.storage import REPO_ROOT

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from screen_against_robot import OFFSET_COLUMNS, outcomes, robot_runs

RADIUS = 0.02


def run(**changes: str) -> dict[str, str]:
    """A sweep's row, as read from ``runs.csv``: it arrives, holds, and ends at the target, and follows by 0.5."""
    row = {
        "setting": "pd_w20_z0.4",
        "start": "0",
        "spectral_radius": "1.2",
        "input_scaling": "1.5",
        "seed": "8",
        "arrived": "True",
        "arrival_time_s": "9.5",
        "success": "True",
        "hold_observed_s": "2.0",
        "final_distance_m": "0.003",
        "following_gain": "0.5",
    }
    for i in (1, 2, 3):
        row |= {f"return_time_s_{i}": "1.0", f"following_gain_{i}": "0.5"}
    return row | changes


def scenarios() -> dict[str, dict[str, str]]:
    return {"nominal": run(), "block": run(), "pushes": run()}


def test_without_the_offset_starts_the_rule_is_report_011s():
    outcome = outcomes(scenarios(), RADIUS)

    assert outcome["passes"]
    assert "offset_failures" not in outcome


def test_an_esn_failing_from_one_offset_start_fails():
    offsets = [run(start=str(k)) for k in range(48)] + [run(start="48", success="False", hold_observed_s="1.0")]

    outcome = outcomes(scenarios(), RADIUS, offsets)

    assert not outcome["passes"]
    assert outcome["offset_failures"] == 1
    assert not outcome["never_arrives"]


def test_never_arriving_from_an_offset_start_is_never_arriving():
    offsets = [run(), run(start="1", arrived="False", arrival_time_s="nan", hold_observed_s="nan")]

    outcome = outcomes(scenarios(), RADIUS, offsets)

    assert outcome["never_arrives"]
    assert outcome["offset_failures"] == 1


def test_leaving_the_goal_under_the_pushes_is_excused_but_not_from_an_offset_start():
    runs = scenarios() | {"pushes": run(success="False", hold_observed_s="0.5")}

    assert outcomes(runs, RADIUS)["passes"]
    assert not outcomes(runs, RADIUS, [run(success="False", hold_observed_s="0.5")])["passes"]


def sweep(tmp_path, name: str, evaluation: str, disturbance: str, rows: list[dict[str, str]]) -> str:
    """A sweep's run directory at leak rate 0.03, holding its configuration and its ``runs.csv``."""
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "config.toml").write_text(f"[esn]\nleak_rate = 0.03\n\n[evaluation]\n{evaluation}\n{disturbance}")
    with (run_dir / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(run_dir)


def test_an_offset_sweep_gives_every_start_of_every_esn_with_the_rules_columns(tmp_path):
    block = '[disturbance]\ntype = "block"\nonset = 4.5\nrelease = 5.5\n'
    pushes = '[[disturbance]]\ntype = "push"\nonset = 3.0\n[[disturbance]]\ntype = "push"\nonset = 7.0\n'
    grid = "[evaluation.start_grid]\nspan_deg = 15.0\nstep_deg = 5.0\n"
    references = [
        sweep(tmp_path, "nominal", "duration = 24.0", "", [run()]),
        sweep(tmp_path, "block", "duration = 24.0", block, [run()]),
        sweep(tmp_path, "pushes", "duration = 24.0", pushes, [run()]),
        sweep(tmp_path, "offsets", "duration = 24.0\n" + grid, "", [run(start=str(k)) for k in range(3)]),
    ]

    robot, offsets = robot_runs(references)

    key = (0.03, 1.2, 1.5, 8)
    assert set(robot[key]) == {"nominal", "block", "pushes"}
    assert len(offsets[key]) == 3
    assert set(offsets[key][0]) == set(OFFSET_COLUMNS)


def test_a_hold_short_of_the_window_by_rounding_alone_holds():
    offsets = [run(hold_observed_s="1.9999999999999982")]

    assert outcomes(scenarios(), RADIUS, offsets)["passes"]
