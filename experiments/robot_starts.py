# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Run trained ESNs on the robot from start postures all over the joint space, from rest.

    uv run python experiments/robot_starts.py \\
        experiments/manual_demonstration_v2_consistent_robot_loop/robot_starts.toml --workers 30

``global_starts.py``'s runs, on the robot: each ESN of the ``[[esn]]`` tables (``key``,
``label``, and ``model``) drives the arm's reference with one tracker setting
(``[tracker]``, as in ``robot_esn.py``), from rest at every posture of a grid over both
joints' whole range, every ``step_deg`` (``[grid]``). Through the ESN's warm-up, the
tracker holds the arm at its start posture and the ESN is fed it, so that the reservoir
starts the task as it does on its own; then the ESN is fed the arm's measured joint
angles for ``duration`` seconds (``[evaluation]``). The ESNs must have been trained on
the same take, which every run is measured against as ``global_starts.py`` measures the
ESN's own runs:

- **the target**: whether the hand arrives within the goal radius and holds there for
  ``hold`` seconds, and where the run ends: ``target``, ``rest elsewhere``, or
  ``moving`` (the arm's joints stop at ±180 deg, so a run is never out of range;
  ``limit_s`` is how long a joint rests against a stop instead);
- **the take's path**: when the arm joins it for good, within ``path_threshold_deg``
  (``[analysis]``), and at what progress along it;
- **the reference and the arm**: the reference's first step, the ESN's first output,
  from the start posture, which is the ESN's own first step; and the largest joint
  torques and joint speed over the task.

The run directory receives, for each ESN:

- ``starts_<key>.csv``: one row per start posture, with the measures above;
- ``runs_<key>.npz``: every run's joint angles, reference, joint torques, distance from
  the take's path, and progress along it, every ESN period;

and ``take.npz``, as ``global_starts.py``'s. The runs are spread over ``--workers``
processes, each computing with one thread.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from global_starts import JOINT_LIMIT_DEG, PATH_SPACING_DEG, final_speed_dps, grid_postures, outcome, print_summary
from numpy.typing import NDArray
from robot_esn import posed, tracker_settings

from arm_esn_ctrl.autonomous import Setup, load_setup
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.metrics import distances_to_path, hold_metrics, join_index, path_progress, resample_by_length
from arm_esn_ctrl.storage import resolve_run_path, start_run
from arm_esn_ctrl.tracking import EsnSource, TrackerConfig, gain_posture, task_joint_angles, track

AT_LIMIT_DEG = 1e-9  # a joint this close to ±180 deg rests against a stop


@dataclass(frozen=True)
class Context:
    """What every run needs, loaded once per process."""

    esns: dict[str, ReachingEsn]
    setup: Setup
    path: NDArray[np.float64]  # the take's joint path, resampled finely
    simulator: dict[str, Any]
    tracker: TrackerConfig
    gains: tuple[NDArray[np.float64], NDArray[np.float64]]
    duration: float
    analysis: dict[str, Any]


_context: Context | None = None  # this process's, set by init_worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment configuration file (TOML)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="worker processes")
    args = parser.parse_args()
    config, run_dir = start_run(args.config)
    context = load_context(config)
    setup = context.setup
    take = next(iter(setup.demos.values()))
    dt = next(iter(context.esns.values())).config.dt
    np.savez_compressed(
        run_dir / "take.npz",
        times=dt * np.arange(len(take)),
        q=take,
        hand=next(iter(setup.demo_hands.values())),
        target=setup.target,
        radius=setup.radius,
    )
    print(f"{len(setup.starts)} start postures every {config['grid']['step_deg']:g} deg; kp {context.gains[0]},"
          f" kd {context.gains[1]}")  # fmt: skip

    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
    jobs = [(table["key"], i) for table in config["esn"] for i in range(len(setup.starts))]
    results: dict[str, list[tuple[dict[str, Any], dict[str, NDArray[Any]]]]] = {key: [] for key in context.esns}
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(config,)
    ) as pool:
        for k, ((key, _), result) in enumerate(zip(jobs, pool.map(run_job, jobs, chunksize=4), strict=True)):
            results[key].append(result)
            if (k + 1) % 216 == 0:
                print(f"  {k + 1} of {len(jobs)} runs", flush=True)

    for table in config["esn"]:
        key = table["key"]
        rows = [row for row, _ in results[key]]
        write_csv(run_dir / f"starts_{key}.csv", rows)
        arrays: dict[str, Any] = {name: np.array([a[name] for _, a in results[key]]) for name in results[key][0][1]}
        np.savez_compressed(
            run_dir / f"runs_{key}.npz",
            times=dt * np.arange(round(context.duration / dt) + 1),
            start_q_deg=np.array([[row["q1_deg"], row["q2_deg"]] for row in rows]),
            **arrays,
        )
        print(f"{table['label']} ({table['model']}) on the robot:")
        print_summary(rows)
        torque = np.array([max(row["peak_tau1_nm"], row["peak_tau2_nm"]) for row in rows])
        limit = np.array([row["limit_s"] for row in rows])
        print(
            f"  largest joint torque median {np.median(torque):.1f} N m (max {torque.max():.1f});"
            f" {np.count_nonzero(limit)} runs rest against a joint's stop, for up to {limit.max():.2f} s"
        )
    print(f"\nWrote the results to {run_dir}")


def load_context(config: dict[str, Any]) -> Context:
    """The ESNs, the take and the start postures, and the tracker."""
    grid = grid_postures(config["grid"]["step_deg"])
    evaluation = {"duration": config["evaluation"]["duration"], "hold": config["evaluation"]["hold"]}
    evaluation["extra_starts"] = {"grid": grid.tolist()}
    esns, trained_on = {}, None
    for table in config["esn"]:
        esn_path = resolve_run_path(table["model"])
        esns[table["key"]] = ReachingEsn.load(esn_path)
        with (esn_path.parent / "config.toml").open("rb") as f:
            demonstrations = tomllib.load(f)["demonstrations"]
        if trained_on not in (None, demonstrations):
            msg = f"the ESNs must be trained on the same take: {table['key']} was trained on {demonstrations}"
            raise ValueError(msg)
        trained_on = demonstrations
    assert trained_on is not None
    dt = next(iter(esns.values())).config.dt
    setup = load_setup({"demonstrations": trained_on, "esn": {"dt": dt}, "evaluation": evaluation})
    if len(setup.demos) != 1:
        msg = "the runs are compared with one take: the ESNs must be trained on one demonstration"
        raise ValueError(msg)
    with (resolve_run_path(trained_on["run"]) / "config.toml").open("rb") as f:
        simulator = tomllib.load(f)["simulator"]
    settings, table = tracker_settings(config["tracker"]), config["tracker"]
    if len(settings) != 1:
        msg = "[tracker] must give one setting"
        raise ValueError(msg)
    setting = settings[0]
    tracker = TrackerConfig(
        setting.law,
        setting.omega,
        table["acceleration_filter"],
        damping=setting.damping,
        reference_velocity=table.get("reference_velocity", True),
        control_period=table.get("control_period"),
    )
    take = next(iter(setup.demos.values()))
    return Context(
        esns,
        setup,
        resample_by_length(take, np.radians(PATH_SPACING_DEG)),
        simulator,
        tracker,
        setting.gains(tracker, setup.skeleton, gain_posture(setup.demos.values())),
        float(config["evaluation"]["duration"]),
        config["analysis"],
    )


def init_worker(config: dict[str, Any]) -> None:
    global _context
    _context = load_context(config)


def run_job(job: tuple[str, int]) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """One run in this worker."""
    assert _context is not None
    return run_start(_context, *job)


def run_start(context: Context, key: str, i: int) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """The run of ESN ``key`` from start posture ``i``: its row of ``starts_<key>.csv`` and its arrays."""
    esn, setup, duration = context.esns[key], context.setup, context.duration
    dt = esn.config.dt
    start = setup.starts[i].q
    log = track(
        posed(setup.skeleton, start),
        EsnSource(esn),
        context.tracker,
        context.gains,
        period=dt,
        warmup_steps=esn.config.warmup_steps,
        duration=duration,
        dt=context.simulator["dt"],
        enforce_limits=context.simulator.get("enforce_limits", True),
    )
    q = task_joint_angles(log, dt, duration)
    reference = task_joint_angles(log, dt, duration, channel="q_ref")
    torque = task_joint_angles(log, dt, duration, channel="tau")
    task = log.times >= 0.0
    speed = np.degrees(np.linalg.norm(log.channel("dq").reshape(len(log.times), -1)[task], axis=1))
    hand = endpoint_positions(setup.skeleton, q)
    distance = np.degrees(distances_to_path(q, context.path))
    progress = path_progress(q, context.path)
    joined = join_index(distance, context.analysis["path_threshold_deg"])
    hold = hold_metrics(setup.times, hand, setup.target, setup.radius, setup.hold)
    at_limit = np.any(np.degrees(np.abs(q[1:])) >= JOINT_LIMIT_DEG - AT_LIMIT_DEG, axis=1)
    peak_torque = np.abs(torque).max(axis=0)
    row = (
        {
            "start": i,
            "q1_deg": round(float(np.degrees(start[0])), 6),
            "q2_deg": round(float(np.degrees(start[1])), 6),
            "start_target_distance_m": float(np.linalg.norm(hand[0] - setup.target)),
            "start_path_distance_deg": float(distance[0]),
            "start_progress": float(progress[0]),
            "outcome": outcome(q, hand, setup.target, setup.radius, dt, context.analysis["rest_speed_dps"]),
        }
        | {name: hold[name] for name in ("arrived", "arrival_time_s", "success")}
        | {
            "final_q1_deg": float(np.degrees(q[-1, 0])),
            "final_q2_deg": float(np.degrees(q[-1, 1])),
            "final_distance_m": float(np.linalg.norm(hand[-1] - setup.target)),
            "final_speed_dps": final_speed_dps(q, dt),
            "max_path_distance_deg": float(distance.max()),
            "join_time_s": float("nan") if joined is None else float(setup.times[joined]),
            "join_progress": float("nan") if joined is None else float(progress[joined]),
            "first_step_deg": float(np.degrees(np.linalg.norm(reference[1] - reference[0]))),
            "peak_tau1_nm": float(peak_torque[0]),
            "peak_tau2_nm": float(peak_torque[1]),
            "peak_speed_dps": float(speed.max()),
            "limit_s": float(np.count_nonzero(at_limit) * dt),
        }
    )
    arrays = {
        "q": q.astype(np.float32),
        "q_ref": reference.astype(np.float32),
        "tau": torque.astype(np.float32),
        "path_distance_deg": distance.astype(np.float32),
        "progress": progress.astype(np.float32),
    }
    return row, arrays


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` to the CSV file ``path``, one column per key."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
