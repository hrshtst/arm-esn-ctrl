# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Release the arm at the ESNs' unstable fixed points again, nudged off them, on the robot and on their own.

    uv run python experiments/nudge_releases.py \\
        experiments/manual_demonstration_v2_consistent_robot_loop/nudge_releases.toml --workers 30

The configuration names a run of ``release_robot.py`` (``[runs]``). Each of its releases
runs again as that run ran it, its ESN trained as its screen trained it, but from the
fixed point's posture moved by ``deg`` in each joint (``[nudge]``): the arm at rest
there, the ESN held there, then let go, on the robot with the release run's tracker, and
the ESN on its own. Held long enough at a fixed point, the reservoir can come to rest on
it to the last bit, and a simulation, which adds nothing, leaves it there however fast
it grows; nudged, it leaves an unstable fixed point as fast as that grows.

The run directory receives ``release.csv``: the release run's rows, each with, nudged,
how long the arm stays within the release run's ``stay_deg`` of the fixed point, when
the robot's hand arrives, whether it ends within the goal, and when the ESN on its own
arrives. The ESNs are spread over ``--workers`` processes, each computing with one
thread.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import tomllib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import release_robot
from screen_esns import run_outcomes

from arm_esn_ctrl.dynamics import ClosedLoop
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.screening import autonomous_runs
from arm_esn_ctrl.storage import resolve_run_path, start_run

SWEPT = ("leak_rate", "spectral_radius", "input_scaling", "seed")
Key = tuple[float, float, float, int]

_nudge = 0.0  # this process's, set by init_worker (deg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="configuration file (TOML)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="worker processes")
    args = parser.parse_args()
    config, run_dir = start_run(args.config)
    release_dir = resolve_run_path(config["runs"]["release"])
    with (release_dir / "config.toml").open("rb") as f:
        release_config = tomllib.load(f)
    by_esn: dict[Key, list[dict[str, str]]] = defaultdict(list)
    with (release_dir / "release.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            by_esn[esn_key(row)].append(row)
    jobs = sorted(by_esn.items())
    print(
        f"{sum(len(rows) for _, rows in jobs)} releases of {len(jobs)} ESNs, nudged by {config['nudge']['deg']:g} deg"
    )

    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(release_config, float(config["nudge"]["deg"])),
    ) as pool:
        for k, result in enumerate(pool.map(esn_job, jobs)):
            rows += result
            if (k + 1) % 100 == 0 or k + 1 == len(jobs):
                print(f"  [{k + 1:4d}/{len(jobs)}]", flush=True)
    with (run_dir / "release.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows)
    print(f"\nWrote the results to {run_dir}")


def init_worker(release_config: dict[str, Any], nudge: float) -> None:
    """Load the release run's context in this process."""
    global _nudge
    release_robot.init_worker(release_config)
    _nudge = nudge


def esn_job(job: tuple[Key, list[dict[str, str]]]) -> list[dict[str, Any]]:
    """One ESN's releases, nudged, in this worker."""
    context = release_robot._context
    assert context is not None
    key, releases = job
    combination: dict[str, Any] = dict(zip(SWEPT, key, strict=True))
    esn = ReachingEsn(EsnConfig(**(context.esn | combination)))
    esn.fit([context.take])
    loop = ClosedLoop.from_esn(esn)
    table, dt = context.config["release"], esn.config.dt
    hold, n_steps = round(table["hold"] / dt), round(table["duration"] / dt)
    rows = []
    for release in releases:
        q0 = np.radians([float(release["q1_deg"]), float(release["q2_deg"])])
        start = q0 + np.radians(_nudge)
        q = release_robot.robot_run(context, esn, start, hold, table["duration"], None)
        robot = run_outcomes(context.setup, q[np.newaxis], dt)
        own = run_outcomes(context.setup, autonomous_runs(loop, start[np.newaxis], hold, n_steps)[0], dt)
        moved = np.flatnonzero(np.degrees(np.abs(q - q0)).max(axis=1) > table["stay_deg"])
        rows.append(
            dict(release)
            | {
                "nudged_stays_s": float(moved[0] * dt) if len(moved) else np.inf,
                "nudged_arrival_s": float(robot["arrival"][0]),
                "nudged_ends_at_target": bool(robot["at_target"][0]),
                "nudged_own_arrival_s": float(own["arrival"][0]),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> None:
    """Print, by leak rate group, how the nudged releases compare with the release run's."""
    lr = np.array([float(r["leak_rate"]) for r in rows])
    for name, group in (("all", lr > 0), ("up to 0.1", lr <= 0.1), ("from 0.15", lr >= 0.15)):
        sub = [r for r, g in zip(rows, group, strict=True) if g]
        stays = np.array([float(r["stays_s"]) for r in sub])
        nudged = np.array([r["nudged_stays_s"] for r in sub])
        robot, own = (np.array([r[c] for r in sub]) for c in ("nudged_arrival_s", "nudged_own_arrival_s"))
        both = np.isfinite(robot) & np.isfinite(own)
        print(
            f"  {name}: {len(sub)} releases; staying the whole release {np.isinf(stays).sum()} as run,"
            f" {np.isinf(nudged).sum()} nudged; nudged, both arrive at {both.sum()}, only one at"
            f" {(np.isfinite(robot) != np.isfinite(own)).sum()}, the robot a median"
            f" {np.median(robot[both] - own[both]) if both.any() else float('nan'):+.2f} s later"
        )


def esn_key(row: dict[str, str]) -> Key:
    return (
        float(row["leak_rate"]),
        float(row["spectral_radius"]),
        float(row["input_scaling"]),
        int(float(row["seed"])),
    )


if __name__ == "__main__":
    main()
