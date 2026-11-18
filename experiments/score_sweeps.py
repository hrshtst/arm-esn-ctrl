# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Score the settings of an ESN search by its undisturbed runs and its pushed runs together (Stage 2).

    uv run python experiments/score_sweeps.py experiments/<experiment>/<score>.toml

The configuration's ``[runs]`` names two runs of ``experiments/sweep_robot_esn.py``,
relative to the storage root, that ran the same ESNs with the same tracker:
``undisturbed`` and ``pushes``. The settings are those of the undisturbed run, each
tried with its seeds if the seed was swept; the pushed run must hold every one of
their ESNs, and its other ESNs are left out.

1. **An undisturbed run passes** if it holds: the hand arrives within the goal
   radius and stays within it for the hold window (the sweep's ``failed``).
2. **A pushed run passes** if the hand arrives and ends within the goal radius.
   Leaving the goal during the hold is excused: a push can carry the arm into the
   goal early.
3. **A setting's score** is how many of its runs pass, the most first, then the
   median detour ratio of its undisturbed runs that arrive, the smallest first
   (see :func:`arm_esn_ctrl.sweeps.score_settings`). Its typical seed is, of its
   seeds with the most passing runs, the one whose own median detour ratio is
   nearest the setting's.

The run directory receives ``scores.csv``: one row per setting, the best first: its
hyperparameters, its runs and how many pass (``runs``, ``passing_runs``), apart for
the undisturbed and the pushed (``undisturbed_runs``, ``holding_runs``,
``pushed_runs``, ``passing_pushed_runs``), its median detour ratio, and its seeds
and typical seed.
"""

from __future__ import annotations

import argparse
import csv
import math
import tomllib
from pathlib import Path
from typing import Any

from arm_esn_ctrl.storage import resolve_run_path, start_run
from arm_esn_ctrl.sweeps import score_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="scoring configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    undisturbed_dir = resolve_run_path(config["runs"]["undisturbed"])
    pushed_dir = resolve_run_path(config["runs"]["pushes"])
    sweep = read_toml(undisturbed_dir / "config.toml")
    swept = [*sweep.get("settings", [{}])[0], *sweep.get("sweep", {})]
    take = read_toml(resolve_run_path(sweep["demonstrations"]["run"]) / "config.toml")
    radius = float(take["task"]["target"]["tolerance"])

    scores = score_runs(read_runs(undisturbed_dir / "runs.csv"), read_runs(pushed_dir / "runs.csv"), swept, radius)
    with (run_dir / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)

    names = [name for name in swept if name != "seed"]
    print("The most runs that pass, then the smallest median detour ratio of the undisturbed runs that arrive:")
    for score in scores[:10]:
        values = ", ".join(f"{name}={score[name]:g}" for name in names)
        seed = f"; typical seed {score['typical_seed']}" if "typical_seed" in score else ""
        print(
            f"  {values}: {score['holding_runs']} of {score['undisturbed_runs']} hold undisturbed,"
            f" {score['passing_pushed_runs']} of {score['pushed_runs']} pass the pushes,"
            f" median detour ratio {score['median_detour_ratio']:.4f}{seed}"
        )
    print(f"\nWrote the results to {run_dir}")


def score_runs(
    undisturbed: list[dict[str, Any]], pushed: list[dict[str, Any]], swept: list[str], radius: float
) -> list[dict[str, Any]]:
    """Score each setting over its undisturbed and its pushed runs, the best first (see the module docstring).

    ``undisturbed`` and ``pushed`` are the rows of the two sweeps' ``runs.csv``,
    ``swept`` the hyperparameters that differ between the ESNs (``seed`` among them
    if it was swept), and ``radius`` the goal radius (m).
    """
    names = [name for name in swept if name != "seed"]

    def esn(run: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(run[name] for name in swept)

    esns = {esn(run) for run in undisturbed}
    pushed = [run for run in pushed if esn(run) in esns]
    missing = esns - {esn(run) for run in pushed}
    if missing:
        msg = f"{len(missing)} ESNs of the undisturbed run are not in the pushed run, such as {sorted(missing)[0]}"
        raise ValueError(msg)
    # A pushed run counts by where it ends, and its path, which the pushes lengthen, does not count as a detour.
    pushed = [
        run | {"failed": not (run["arrived"] and run["final_distance_m"] <= radius), "detour_ratio": math.nan}
        for run in pushed
    ]

    def setting(run: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(run[name] for name in names)

    scores = []
    for score in score_settings(undisturbed + pushed, names):
        of_setting = [[run for run in runs if setting(run) == setting(score)] for runs in (undisturbed, pushed)]
        passing = [sum(not run["failed"] for run in runs) for runs in of_setting]
        scores.append(
            {name: score[name] for name in names}
            | {
                "runs": score["runs"],
                "passing_runs": score["holding_runs"],
                "undisturbed_runs": len(of_setting[0]),
                "holding_runs": passing[0],
                "pushed_runs": len(of_setting[1]),
                "passing_pushed_runs": passing[1],
                "median_detour_ratio": score["median_detour_ratio"],
            }
            | {name: score[name] for name in ("seeds", "typical_seed") if name in score}
        )
    return scores


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def read_runs(path: Path) -> list[dict[str, Any]]:
    """The rows of a sweep's ``runs.csv``: booleans, whole seeds, and numbers as such."""
    with path.open(newline="") as f:
        return [{name: value(name, text) for name, text in row.items()} for row in csv.DictReader(f)]


def value(name: str, text: str) -> Any:
    if text in ("True", "False"):
        return text == "True"
    if name == "seed":
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


if __name__ == "__main__":
    main()
