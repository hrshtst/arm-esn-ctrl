# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Sweep ESN hyperparameters by how the robot follows the taught path with each ESN (Stage 2).

    uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_tracking/<sweep>.toml [--workers 30]

The configuration holds the ``[demonstrations]`` and ``[esn]`` tables of
``experiments/autonomous_esn.py``; a ``[sweep]`` table, which lists values for one
to three of the ``[esn]`` hyperparameters, as in ``experiments/sweep_esn.py``; the
``[tracker]`` table of ``experiments/robot_esn.py``; and an ``[evaluation]`` table:
the run's duration, the hold, and the start postures. Every combination of the
swept values is trained on the demonstrations and generates the reference of the
simulated arm from every start posture, with every tracker setting, as the ESN arm
of ``robot_esn.py`` does: undisturbed, or pushed and blocked by the optional
``[disturbance]`` table or ``[[disturbance]]`` tables of ``robot_esn.py``, with the
``effort_window``, the ``return_tolerance``, the ringing's cutoff and window, and the
``following_window`` of its ``[evaluation]``.

The combinations are ranked by

1. their failed runs: runs that never arrive, that leave the goal during the hold
   window, or that end before the whole window (the fewest first);
2. the worst path RMSE of their runs from the taught motion, regardless of timing
   (``taught_path_rmse_m``; see :func:`arm_esn_ctrl.metrics.path_rmse`), the
   smallest first: arriving late or moving slowly costs nothing.

The run directory receives:

- ``sweep.csv``: one row per combination: its hyperparameters, its failed runs,
  the worst and the mean path RMSE of its runs, their latest arrival, and the
  hand's largest distance from the target at the end of a run, then, for
  each tracker setting from the first start posture, the path RMSE of the arm and
  of its reference (what the ESN generates), the RMS distance of the hand from the
  taught hand at equal times, the arrival time, the first step, the detour beyond
  the start, and the peak torque;
- with disturbances, also in ``sweep.csv``: the runs whose hand is not back on the
  taught hand path after a disturbance (within the return tolerance, before the
  next disturbance or the end of the run), the latest return over the runs and
  disturbances, and the hand's farthest distance from the path after one; then,
  for each tracker setting from the first start posture, each disturbance's
  ``reference_lead``, ``off_path_m``, ``return_time_s``, ``end_gap_deg``,
  ``ringing_m``, and ``following_gain`` (see ``robot_esn.arm_metrics``), and the peak
  disturbance force;
- ``runs.csv``: the metrics of every run (see ``robot_esn.arm_metrics``);
- ``sweep.png``, with two or three swept hyperparameters: heatmaps of the worst
  path RMSE, the failed runs, and the latest arrival, and with disturbances, of the
  runs not back on the path, the latest return, and the farthest from the path.

With ``seed`` among the swept keys, the other swept values make the settings, each
tried with every seed (a random reservoir; see :mod:`arm_esn_ctrl.sweeps`), and the
run directory also receives:

- ``seeds.csv``: one row per setting, over its seeds: the seeds with a failed run,
  the median, mean, standard deviation, smallest, and largest worst path RMSE, the
  median latest arrival, and the largest distance from the target at the end;
  with disturbances, also the seeds with a run not back on the path, the median
  latest return, and the median farthest distance from the path;
- ``seeds.png``, with two other swept hyperparameters: heatmaps of the median
  worst path RMSE, the seeds with a failed run, the largest final distance, and
  the median latest arrival.

With ``--workers``, the combinations are spread over that many processes, each
computing with one thread; their results are the same as in one process.

The runs are not saved. To look at one combination, copy its values into a
configuration for ``autonomous_esn.py`` and run its ESN with ``robot_esn.py``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing
import os
import tomllib
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from robot_esn import (
    FOLLOWING_WINDOW,
    RETURN_TOLERANCE,
    Setting,
    arm_metrics,
    per_disturbance,
    posed,
    ringing_settings,
    tracker_settings,
)
from sweep_esn import plot_sweep

from arm_esn_ctrl.autonomous import Setup, load_setup
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.disturbances import (
    check_recorded_arm,
    disturbance_spans,
    keep_recorded_logs,
    make_disturbance,
    recorded_evaluation,
)
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.metrics import distances_to_path
from arm_esn_ctrl.storage import resolve_run_path, start_run
from arm_esn_ctrl.sweeps import summarize_seeds
from arm_esn_ctrl.tracking import (
    EsnSource,
    TrackerConfig,
    gain_posture,
    nearest_demonstration,
    task_joint_angles,
    track,
)

# The heatmaps: metric column in sweep.csv, title, factor to its display unit, and logarithmic colors.
MAPS = [
    ("worst_path_rmse_m", "Worst path RMSE from the taught motion (mm)", 1000.0, True),
    ("failures", "Failed runs: never arrive, leave the goal, or hold too short", 1.0, False),
    ("latest_arrival_s", "Latest arrival (s)", 1.0, False),
]
# More heatmaps with disturbances, as MAPS.
DISTURBANCE_MAPS = [
    ("unreturned", "Runs not back on the path after a disturbance (of {runs})", 1.0, False),
    ("latest_return_s", "Latest return to the path after a disturbance (s)", 1.0, False),
    ("worst_off_path_m", "Farthest from the path after a disturbance (mm)", 1000.0, True),
]
# The heatmaps of seeds.png, as MAPS: each setting over its seeds.
SEED_MAPS = [
    ("median_worst_path_rmse_m", "Median over the seeds of the worst path RMSE (mm)", 1000.0, True),
    ("failing_seeds", "Seeds with a failed run (of {seeds})", 1.0, False),
    ("worst_final_distance_m", "Largest distance from the target at the end (mm)", 1000.0, True),
    ("median_latest_arrival_s", "Median over the seeds of the latest arrival (s)", 1.0, False),
]
# More heatmaps of seeds.png with disturbances, as MAPS.
SEED_DISTURBANCE_MAPS = [
    ("unreturned_seeds", "Seeds with a run not back on the path (of {seeds})", 1.0, False),
    ("median_latest_return_s", "Median over the seeds of the latest return (s)", 1.0, False),
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


@dataclass(frozen=True)
class Context:
    """What every combination's runs need, loaded once per process."""

    config: dict[str, Any]
    setup: Setup
    simulator: dict[str, Any]
    settings: list[Setting]
    trackers: dict[str, tuple[TrackerConfig, tuple[NDArray[np.float64], NDArray[np.float64]]]]
    duration: float
    window: list[float]
    spans: list[tuple[float, float]]
    tolerance: float
    ringing: tuple[float, float]
    following: float


_context: Context | None = None  # this process's, set by init_worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="sweep configuration file (TOML)")
    parser.add_argument(
        "--workers", type=int, default=1, help="worker processes, of one thread each (default 1: in this process)"
    )
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    sweep: dict[str, list[Any]] = config["sweep"]
    if len(sweep) not in (1, 2, 3):
        msg = f"[sweep] must list one to three hyperparameters, not {len(sweep)}"
        raise ValueError(msg)
    context = sweep_context(config)
    disturbance = config.get("disturbance")
    check_recorded_arm(disturbance, context.setup.skeleton)
    keep_recorded_logs(disturbance, run_dir)
    settings, spans = context.settings, context.spans
    print(
        f"{len(context.setup.starts)} start postures, {len(settings)} tracker settings"
        f" ({', '.join(setting.label for setting in settings)}), {context.duration:g} s runs"
    )
    print(f"Disturbance: {disturbance or 'none'}")

    combinations = [dict(zip(sweep, values, strict=True)) for values in itertools.product(*sweep.values())]
    rows, run_rows = [], []
    if args.workers == 1:
        results: Iterable[tuple[dict[str, Any], list[dict[str, Any]]]] = (
            sweep_combination(context, combination) for combination in combinations
        )
        pool = None
    else:
        # Many processes of one thread each, rather than threads that the processes would contend for.
        os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
        pool = ProcessPoolExecutor(
            args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(config,)
        )
        results = pool.map(sweep_job, combinations)
    for k, (combination, (row, runs)) in enumerate(zip(combinations, results, strict=True)):
        rows.append(row)
        run_rows += [combination | run for run in runs]
        values = ", ".join(f"{name}={value:g}" for name, value in combination.items())
        arrivals = ", ".join(f"{row[f'{setting.name}_arrival_time_s']:.2f} s" for setting in settings)
        returns = f"; {row['unreturned']} not back, latest return {row['latest_return_s']:.2f} s" if spans else ""
        print(
            f"[{k + 1:3d}/{len(combinations)}] {values}: {row['failures']} of {row['runs']} runs fail;"
            f" worst path RMSE {1000 * row['worst_path_rmse_m']:6.1f} mm; arrival {arrivals}{returns}",
            flush=True,
        )
    if pool is not None:
        pool.shutdown()

    tables = {"sweep.csv": rows, "runs.csv": run_rows}
    names = [name for name in sweep if name != "seed"]  # the settings, each tried with every seed
    if "seed" in sweep:
        tables["seeds.csv"] = summarize_seeds(rows, names)
    for name, table in tables.items():
        with (run_dir / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print_ranking(rows, list(sweep), settings)
    if len(sweep) > 1:
        maps = [
            (metric, label.format(runs=rows[0]["runs"]), factor, log)
            for metric, label, factor, log in MAPS + (DISTURBANCE_MAPS if spans else [])
        ]
        tops = {"unreturned": float(rows[0]["runs"])}
        title = f"ESN sweep on the robot: {args.config.stem}"
        plot_sweep(rows, sweep, maps, title, tops).savefig(run_dir / "sweep.png", dpi=150)
    if "seed" in sweep:
        print_seed_ranking(tables["seeds.csv"], names)
        if len(names) > 1:
            seeds = len(sweep["seed"])
            maps = [
                (metric, label.format(seeds=seeds), factor, log)
                for metric, label, factor, log in SEED_MAPS + (SEED_DISTURBANCE_MAPS if spans else [])
            ]
            title = f"ESN sweep on the robot over {seeds} seeds: {args.config.stem}"
            tops = {"failing_seeds": float(seeds), "unreturned_seeds": float(seeds)}
            plot_sweep(tables["seeds.csv"], {name: sweep[name] for name in names}, maps, title, tops).savefig(
                run_dir / "seeds.png", dpi=150
            )
    print(f"\nWrote the results to {run_dir}")


def sweep_context(config: dict[str, Any]) -> Context:
    """What every combination's runs need, from the sweep's configuration."""
    disturbance = config.get("disturbance")
    # Under a recorded disturbance, the runs start from its log's start posture, with an arm like its log's.
    evaluation = recorded_evaluation(config["evaluation"], disturbance)
    setup = load_setup(
        {"demonstrations": config["demonstrations"], "esn": {"dt": config["esn"]["dt"]}, "evaluation": evaluation}
    )
    with (resolve_run_path(config["demonstrations"]["run"]) / "config.toml").open("rb") as f:
        simulator = tomllib.load(f)["simulator"]
    tracker = config["tracker"]
    settings = tracker_settings(tracker)
    end_posture = gain_posture(setup.demos.values())  # where the reaches end, for joint PD's gains
    trackers = {}
    for setting in settings:
        tracker_config = TrackerConfig(
            setting.law,
            setting.omega,
            tracker["acceleration_filter"],
            damping=setting.damping,
            reference_velocity=tracker.get("reference_velocity", True),
            control_period=tracker.get("control_period"),
        )
        trackers[setting.name] = (tracker_config, setting.gains(tracker_config, setup.skeleton, end_posture))
    duration = float(evaluation["duration"])
    return Context(
        config,
        setup,
        simulator,
        settings,
        trackers,
        duration,
        evaluation.get("effort_window", [0.0, duration]),
        disturbance_spans(disturbance),
        evaluation.get("return_tolerance", RETURN_TOLERANCE),
        ringing_settings(evaluation),
        evaluation.get("following_window", FOLLOWING_WINDOW),
    )


def init_worker(config: dict[str, Any]) -> None:
    """Load this process's context."""
    global _context
    _context = sweep_context(config)


def sweep_job(combination: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one combination in this worker."""
    assert _context is not None
    return sweep_combination(_context, combination)


def sweep_combination(context: Context, combination: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Train the ESN of ``combination`` and run it from every start posture with every tracker setting.

    Returns its row of ``sweep.csv`` and its runs' metrics, one dict a run.
    """
    setup, simulator, settings = context.setup, context.simulator, context.settings
    disturbance = context.config.get("disturbance")
    esn = ReachingEsn(EsnConfig(**(context.config["esn"] | combination)))
    esn.fit(list(setup.demos.values()))
    runs = []
    for setting in settings:
        tracker_config, gains = context.trackers[setting.name]
        for i, start in enumerate(setup.starts):
            log = track(
                posed(setup.skeleton, start.q),
                EsnSource(esn),
                tracker_config,
                gains,
                period=esn.config.dt,
                warmup_steps=esn.config.warmup_steps,
                duration=context.duration,
                dt=simulator["dt"],
                enforce_limits=simulator.get("enforce_limits", True),
                external_force=make_disturbance(
                    disturbance,
                    endpoint_positions(setup.skeleton, start.q[np.newaxis])[0],
                    setup.target,
                    dt=simulator["dt"],
                ),
            )
            metrics = arm_metrics(
                log,
                i,
                setup,
                esn.config.dt,
                context.window,
                context.spans,
                context.tolerance,
                context.ringing,
                context.following,
            )
            hand = endpoint_positions(setup.skeleton, task_joint_angles(log, esn.config.dt, context.duration))
            runs.append(
                {"setting": setting.name, "start": i, "origin": start.origin}
                | metrics
                | {"detour_m": detour(hand, setup.demo_hands[nearest_demonstration(start.q, setup.demos)])}
                | {"failed": not metrics["success"] or metrics["hold_observed_s"] < setup.hold - 1e-9}
            )
    return combination | summarize(runs, settings, len(context.spans)), runs


def detour(hand: NDArray[np.float64], taught_hand: NDArray[np.float64], every: int = 5) -> float:
    """How much farther from the taught hand path the hand strays than where it started (m).

    As ``sweep_esn.detour``: both paths are compared every ``every`` samples.
    """
    distances = distances_to_path(hand[::every], taught_hand[::every])
    return float(distances.max() - distances[0])


def summarize(runs: list[dict[str, Any]], settings: list[Setting], disturbances: int) -> dict[str, Any]:
    """A combination's failed runs, worst and mean path RMSE, and latest arrival; then each setting's first run.

    With ``disturbances`` (how many), also its runs not back on the path after one,
    the latest return, and the farthest from the path, and each setting's first run
    under each disturbance.
    """
    path_rmse = [run["taught_path_rmse_m"] for run in runs]
    arrivals = [run["arrival_time_s"] for run in runs]
    summary: dict[str, Any] = {
        "runs": len(runs),
        "failures": sum(run["failed"] for run in runs),
        "worst_path_rmse_m": float(np.max(path_rmse)),
        "mean_path_rmse_m": float(np.mean(path_rmse)),
        "latest_arrival_s": float(np.max(arrivals)) if np.all(np.isfinite(arrivals)) else float("nan"),
        "worst_final_distance_m": float(np.max([run["final_distance_m"] for run in runs])),
    }
    columns = dict(PER_SETTING)
    if disturbances:
        returns = np.array([[run[name] for name in per_disturbance("return_time_s", disturbances)] for run in runs])
        farthest = [run[name] for run in runs for name in per_disturbance("off_path_m", disturbances)]
        summary |= {
            "unreturned": int(np.sum(~np.all(np.isfinite(returns), axis=1))),
            "latest_return_s": float(returns.max()) if np.all(np.isfinite(returns)) else float("nan"),
            "worst_off_path_m": float(np.max(farthest)),
        }
        for metric in ("reference_lead", "off_path_m", "return_time_s", "end_gap_deg", "ringing_m", "following_gain"):
            columns |= {name: name for name in per_disturbance(metric, disturbances)}
        columns["peak_external_force_n"] = "peak_external_force_n"
    for setting in settings:
        first = next(run for run in runs if run["setting"] == setting.name and run["start"] == 0)
        summary |= {f"{setting.name}_{column}": first[metric] for column, metric in columns.items()}
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


def print_seed_ranking(rows: list[dict[str, Any]], names: list[str], count: int = 10) -> None:
    """Print the settings whose every seed arrives and holds, by their median worst path RMSE over the seeds."""
    holding = sorted((r for r in rows if r["failing_seeds"] == 0), key=lambda r: r["median_worst_path_rmse_m"])
    print(f"\n{len(holding)} of {len(rows)} settings arrive and hold in every run with all {rows[0]['seeds']} seeds.")
    print("The smallest median worst path RMSE over the seeds:")
    for r in holding[:count]:
        values = ", ".join(f"{name}={r[name]:g}" for name in names)
        returns = ""
        if "unreturned_seeds" in r:
            returns = (
                f"; {r['unreturned_seeds']} seeds not back on the path,"
                f" median latest return {r['median_latest_return_s']:.2f} s"
            )
        print(
            f"  {values}: median {1000 * r['median_worst_path_rmse_m']:.2f} mm,"
            f" mean {1000 * r['mean_worst_path_rmse_m']:.2f} +- {1000 * r['sd_worst_path_rmse_m']:.2f} mm,"
            f" {1000 * r['min_worst_path_rmse_m']:.2f} to {1000 * r['max_worst_path_rmse_m']:.2f} mm;"
            f" median latest arrival {r['median_latest_arrival_s']:.2f} s;"
            f" farthest from the target at the end {1000 * r['worst_final_distance_m']:.1f} mm{returns}"
        )


if __name__ == "__main__":
    main()
