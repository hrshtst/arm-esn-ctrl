# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Sweep ESN hyperparameters by how the robot follows the taught path with each ESN (Stage 2).

    uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_tracking/<sweep>.toml

The configuration holds the ``[demonstrations]`` and ``[esn]`` tables of
``experiments/autonomous_esn.py``; a ``[sweep]`` table, which lists values for one
to three of the ``[esn]`` hyperparameters, as in ``experiments/sweep_esn.py``; the
``[tracker]`` table of ``experiments/robot_esn.py``; and an ``[evaluation]`` table:
the run's duration, the hold, and the start postures. Every combination of the
swept values is trained on the demonstrations and generates the reference of the
simulated arm from every start posture, with every tracker setting, undisturbed,
as the ESN arm of ``robot_esn.py`` does.

The combinations are ranked by

1. their failed runs: runs that never arrive, that leave the goal during the hold
   window, or that end before the whole window (the fewest first);
2. the worst path RMSE of their runs from the taught motion, regardless of timing
   (``taught_path_rmse_m``; see :func:`arm_esn_ctrl.metrics.path_rmse`), the
   smallest first: arriving late or moving slowly costs nothing.

The run directory receives:

- ``sweep.csv``: one row per combination: its hyperparameters, its failed runs,
  the worst and the mean path RMSE of its runs and their latest arrival, then, for
  each tracker setting from the first start posture, the path RMSE of the arm and
  of its reference (what the ESN generates), the RMS distance of the hand from the
  taught hand at equal times, the arrival time, the first step, the detour beyond
  the start, and the peak torque;
- ``runs.csv``: the metrics of every run (see ``robot_esn.arm_metrics``);
- ``sweep.png``, with two or three swept hyperparameters: heatmaps of the worst
  path RMSE, the failed runs, and the latest arrival.

The runs are not saved. To look at one combination, copy its values into a
configuration for ``autonomous_esn.py`` and run its ESN with ``robot_esn.py``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from robot_esn import Setting, arm_metrics, posed, tracker_settings
from sweep_esn import plot_sweep

from arm_esn_ctrl.autonomous import load_setup
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.metrics import distances_to_path
from arm_esn_ctrl.storage import resolve_run_path, start_run
from arm_esn_ctrl.tracking import (
    EsnSource,
    TrackerConfig,
    nearest_demonstration,
    task_joint_angles,
    track,
    tracking_gains,
)

# The heatmaps: metric column in sweep.csv, title, factor to its display unit, and logarithmic colors.
MAPS = [
    ("worst_path_rmse_m", "Worst path RMSE from the taught motion (mm)", 1000.0, True),
    ("failures", "Failed runs: never arrive, leave the goal, or hold too short", 1.0, False),
    ("latest_arrival_s", "Latest arrival (s)", 1.0, False),
]
# What sweep.csv gives for each tracker setting, from the first start posture: its column, and the run's metric.
PER_SETTING = {
    "path_rmse_m": "taught_path_rmse_m",
    "reference_path_rmse_m": "reference_path_rmse_m",
    "tip_error_m": "taught_tip_error_m",
    "arrival_time_s": "arrival_time_s",
    "first_step_m": "first_step_m",
    "detour_m": "detour_m",
    "peak_torque_nm": "peak_torque_nm",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="sweep configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    if "disturbance" in config:
        msg = "sweep_robot_esn.py runs undisturbed; run disturbances with robot_esn.py"
        raise ValueError(msg)
    sweep: dict[str, list[Any]] = config["sweep"]
    if len(sweep) not in (1, 2, 3):
        msg = f"[sweep] must list one to three hyperparameters, not {len(sweep)}"
        raise ValueError(msg)
    evaluation = config["evaluation"]
    setup = load_setup(
        {"demonstrations": config["demonstrations"], "esn": {"dt": config["esn"]["dt"]}, "evaluation": evaluation}
    )
    with (resolve_run_path(config["demonstrations"]["run"]) / "config.toml").open("rb") as f:
        simulator = tomllib.load(f)["simulator"]
    tracker = config["tracker"]
    settings = tracker_settings(tracker)
    end_posture = np.mean([q[-1] for q in setup.demos.values()], axis=0)  # where the reaches end
    trackers = {}
    for setting in settings:
        tracker_config = TrackerConfig(
            setting.law,
            setting.omega,
            tracker["acceleration_filter"],
            damping=setting.damping,
            reference_velocity=tracker.get("reference_velocity", True),
        )
        trackers[setting.name] = (tracker_config, tracking_gains(tracker_config, setup.skeleton, end_posture))
    duration = float(evaluation["duration"])
    print(
        f"{len(setup.starts)} start postures, {len(settings)} tracker settings"
        f" ({', '.join(setting.label for setting in settings)}), {duration:g} s runs"
    )

    combinations = [dict(zip(sweep, values, strict=True)) for values in itertools.product(*sweep.values())]
    rows, run_rows = [], []
    for k, combination in enumerate(combinations):
        esn = ReachingEsn(EsnConfig(**(config["esn"] | combination)))
        esn.fit(list(setup.demos.values()))
        runs = []
        for setting in settings:
            tracker_config, gains = trackers[setting.name]
            for i, start in enumerate(setup.starts):
                log = track(
                    posed(setup.skeleton, start.q),
                    EsnSource(esn),
                    tracker_config,
                    gains,
                    period=esn.config.dt,
                    warmup_steps=esn.config.warmup_steps,
                    duration=duration,
                    dt=simulator["dt"],
                    enforce_limits=simulator.get("enforce_limits", True),
                )
                metrics = arm_metrics(log, i, setup, esn.config.dt, [0.0, duration], [])
                hand = endpoint_positions(setup.skeleton, task_joint_angles(log, esn.config.dt, duration))
                runs.append(
                    {"setting": setting.name, "start": i, "origin": start.origin}
                    | metrics
                    | {"detour_m": detour(hand, setup.demo_hands[nearest_demonstration(start.q, setup.demos)])}
                    | {"failed": not metrics["success"] or metrics["hold_observed_s"] < setup.hold - 1e-9}
                )
        row = combination | summarize(runs, settings)
        rows.append(row)
        run_rows += [combination | run for run in runs]
        values = ", ".join(f"{name}={value:g}" for name, value in combination.items())
        arrivals = ", ".join(f"{row[f'{setting.name}_arrival_time_s']:.2f} s" for setting in settings)
        print(
            f"[{k + 1:3d}/{len(combinations)}] {values}: {row['failures']} of {row['runs']} runs fail;"
            f" worst path RMSE {1000 * row['worst_path_rmse_m']:6.1f} mm; arrival {arrivals}"
        )

    for name, table in (("sweep.csv", rows), ("runs.csv", run_rows)):
        with (run_dir / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print_ranking(rows, list(sweep), settings)
    if len(sweep) > 1:
        plot_sweep(rows, sweep, MAPS, f"ESN sweep on the robot: {args.config.stem}").savefig(
            run_dir / "sweep.png", dpi=150
        )
    print(f"\nWrote the results to {run_dir}")


def detour(hand: NDArray[np.float64], taught_hand: NDArray[np.float64], every: int = 5) -> float:
    """How much farther from the taught hand path the hand strays than where it started (m).

    As ``sweep_esn.detour``: both paths are compared every ``every`` samples.
    """
    distances = distances_to_path(hand[::every], taught_hand[::every])
    return float(distances.max() - distances[0])


def summarize(runs: list[dict[str, Any]], settings: list[Setting]) -> dict[str, Any]:
    """A combination's failed runs, worst and mean path RMSE, and latest arrival; then each setting's first run."""
    path_rmse = [run["taught_path_rmse_m"] for run in runs]
    arrivals = [run["arrival_time_s"] for run in runs]
    summary: dict[str, Any] = {
        "runs": len(runs),
        "failures": sum(run["failed"] for run in runs),
        "worst_path_rmse_m": float(np.max(path_rmse)),
        "mean_path_rmse_m": float(np.mean(path_rmse)),
        "latest_arrival_s": float(np.max(arrivals)) if np.all(np.isfinite(arrivals)) else float("nan"),
    }
    for setting in settings:
        first = next(run for run in runs if run["setting"] == setting.name and run["start"] == 0)
        summary |= {f"{setting.name}_{column}": first[metric] for column, metric in PER_SETTING.items()}
    return summary


def print_ranking(rows: list[dict[str, Any]], names: list[str], settings: list[Setting], count: int = 10) -> None:
    """Print the combinations with the fewest failed runs, then the smallest worst path RMSE."""
    ranked = sorted(rows, key=lambda r: (r["failures"], r["worst_path_rmse_m"]))
    succeeding = sum(r["failures"] == 0 for r in rows)
    print(f"\n{succeeding} of {len(rows)} combinations arrive and hold in every run.")
    print("The fewest failed runs, then the smallest worst path RMSE from the taught motion:")
    for r in ranked[:count]:
        values = ", ".join(f"{name}={r[name]:g}" for name in names)
        details = "; ".join(
            f"{setting.label}: path RMSE {1000 * r[f'{setting.name}_path_rmse_m']:.1f} mm"
            f" (reference {1000 * r[f'{setting.name}_reference_path_rmse_m']:.1f} mm),"
            f" tip error {1000 * r[f'{setting.name}_tip_error_m']:.0f} mm,"
            f" arrival {r[f'{setting.name}_arrival_time_s']:.2f} s"
            for setting in settings
        )
        print(
            f"  {values}: {r['failures']} of {r['runs']} fail, worst path RMSE {1000 * r['worst_path_rmse_m']:.1f} mm"
            f" | {details}"
        )


if __name__ == "__main__":
    main()
