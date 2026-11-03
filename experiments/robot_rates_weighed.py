# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The rates along the robot's run, with the arm's joint angles and velocities weighed otherwise.

    uv run python experiments/robot_rates_weighed.py \\
        experiments/manual_demonstration_v2_consistent_robot_loop/robot_rates_weighed.toml --workers 12

``robot_fixed_points.py`` measures the rates along the robot's run in a state whose
values are of three kinds, the reservoir's, the joint angles, and the joint velocities,
weighed as :attr:`arm_esn_ctrl.dynamics.RobotLoop.scale` weighs them. Over windows of
seconds, the rates depend on that. Here each ESN of the configuration's pulses' run
(``[runs]``, as ``robot_fixed_points.py`` reads it) runs on the robot from the take's
start, and its rates are measured over the windows of ``[rates]`` with the joint angles
and the joint velocities weighed by each pair of ``factors`` more (``[weighings]``).

The run directory receives ``weighings.csv``: per ESN, weighing, and window, the
largest rate across the run, its median over the windows and its range. The weighings
are spread over ``--workers`` processes, each computing with one thread.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import multiprocessing
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
from push_pulses import load_context
from robot_esn import posed

from arm_esn_ctrl.dynamics import ClosedLoop, RobotLoop, StateRecorder, local_rates
from arm_esn_ctrl.storage import resolve_run_path, start_run
from arm_esn_ctrl.tracking import track

_config: dict[str, Any] = {}  # this process's, set by init_worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="configuration file (TOML)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="worker processes")
    args = parser.parse_args()
    config, run_dir = start_run(args.config)
    with (resolve_run_path(config["runs"]["pulses"]) / "config.toml").open("rb") as f:
        keys = [table["key"] for table in tomllib.load(f)["esn"]]
    jobs = [(key, (float(a), float(v))) for key in keys for a, v in config["weighings"]["factors"]]
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(config,)
    ) as pool:
        for result in pool.map(weighed_job, jobs):
            rows += result
    with (run_dir / "weighings.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"  {row['esn']}, joint angles x{row['angles']:g}, velocities x{row['velocities']:g},"
            f" {row['window_s']:g} s: the largest rate across the run, median {row['median_per_s']:+.3f} /s"
            f" ({row['min_per_s']:+.2f} to {row['max_per_s']:+.2f})"
        )
    print(f"\nWrote the results to {run_dir}")


def init_worker(config: dict[str, Any]) -> None:
    _config.update(config)


def weighed_job(job: tuple[str, tuple[float, float]]) -> list[dict[str, Any]]:
    """One ESN's rates along the robot's run, with one weighing, in this worker."""
    key, weights = job
    with (resolve_run_path(_config["runs"]["pulses"]) / "config.toml").open("rb") as f:
        context = load_context(tomllib.load(f))
    esn, setup = context.esns[key], context.setup
    take = next(iter(setup.demos.values()))
    loop = ClosedLoop.from_esn(esn)
    robot = RobotLoop.from_tracker(
        loop,
        setup.skeleton,
        context.tracker,
        context.gains,
        context.simulator["dt"],
        enforce_limits=context.simulator.get("enforce_limits", True),
        stops=context.simulator.get("stops", "projected"),
    )
    robot = dataclasses.replace(robot, weights=weights)
    recorder = StateRecorder(esn)
    log = track(
        posed(setup.skeleton, setup.starts[0].q),
        recorder,
        context.tracker,
        context.gains,
        period=loop.dt,
        warmup_steps=esn.config.warmup_steps,
        duration=len(take) * loop.dt,
        dt=context.simulator["dt"],
        enforce_limits=context.simulator.get("enforce_limits", True),
        stops=context.simulator.get("stops", "projected"),
    )
    along = robot.run_states(log, recorder, esn.config.warmup_steps)[1 : len(take)]  # as robot_fixed_points.py
    rates = _config["rates"]
    rows = []
    for seconds in rates["windows"]:
        r = local_rates(robot, along, round(seconds / loop.dt), rates["stride"], 1)
        across = r.across[:, 0]
        rows.append(
            {"esn": key, "angles": weights[0], "velocities": weights[1], "window_s": float(seconds)}
            | {"median_per_s": float(np.median(across)), "min_per_s": float(across.min())}
            | {"max_per_s": float(across.max())}
        )
    return rows


if __name__ == "__main__":
    main()
