# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the take of a periodic motion by a script, as the recorder would save one taught by hand.

    uv run python experiments/make_periodic_take.py experiments/demonstrations/ellipse_scripted_v2_no_limits.toml

The configuration's ``[task]`` is a skelarm ``periodic_curve`` task: its curve, once
around every ``period`` seconds. The take is the joint angles that put the hand on
the curve, the curve's angle growing at an even rate from 0, for the task's
``duration``, sampled every ``[script]`` ``dt`` seconds. They are solved by skelarm's
inverse kinematics sample by sample, each seeded with the one before and the first
with the start posture of ``[demonstrations]`` ``start_q``, so that the joint angles
are continuous and keep the start's elbow side.

The take is written to the one file of ``[recording]`` ``takes``, under the storage
root, as a skelarm state log with the sampling rates the recorder records, which
``experiments/import_demonstrations.py`` then checks, measures, and imports as a take
recorded by hand. A file that exists is never overwritten, as the recorder never
overwrites one: to make the take again, remove its file first.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from skelarm import PeriodicTaskReference, Skeleton, build_curve, ik_joint_reference

from arm_esn_ctrl.demonstrations import joint_trajectory_log
from arm_esn_ctrl.storage import resolve_run_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="demonstration configuration file of a periodic task (TOML)")
    args = parser.parse_args()
    with args.config.open("rb") as f:
        config = tomllib.load(f)
    (path,) = config["recording"]["takes"]
    take = resolve_run_path(path)
    if take.exists():
        msg = f"Not made: {take} exists. Remove it first to make the take again."
        raise SystemExit(msg)

    skeleton = Skeleton.from_toml(args.config)
    (start_deg,) = config["demonstrations"]["start_q"]
    times, q = periodic_take(skeleton, config["task"], np.radians(start_deg), config["script"]["dt"])
    log = joint_trajectory_log(skeleton, times, q, config["task"], producer="make_periodic_take")
    # The rates the recorder records of a take, here met exactly.
    rate = 1.0 / config["script"]["dt"]
    log.extra["acquisition"] = {"requested_rate_hz": rate, "achieved_rate_hz": (len(times) - 1) / times[-1]}
    take.parent.mkdir(parents=True, exist_ok=True)
    log.save(take)
    print(f"Made a take of {times[-1]:g} s, {len(times)} samples, to {take}")


def periodic_take(
    skeleton: Skeleton, task: dict[str, Any], start_q: NDArray[np.float64], dt: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The times and joint angles (rad) that put the hand on the curve of a ``periodic_curve`` ``task``.

    The curve's angle grows at an even rate, a whole turn every ``period`` seconds,
    for the task's ``duration``, sampled every ``dt``. Each sample is solved by
    inverse kinematics, seeded with the one before, the first with ``start_q``.
    """
    seeded = skeleton.clone()
    seeded.q = start_q
    curve = PeriodicTaskReference(build_curve(task["curve"], task), period=task["period"], duration=task["duration"])
    reference = ik_joint_reference(seeded, curve, dt=dt)
    return reference.times, reference.q


if __name__ == "__main__":
    main()
