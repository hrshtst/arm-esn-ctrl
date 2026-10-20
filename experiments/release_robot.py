# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Hold the arm still on the robot, then let it go: at an ESN's unstable fixed points, and along the reach.

    uv run python experiments/release_robot.py experiments/manual_demonstration_v2_esn_screening/release_robot.toml

The configuration names a run of ``experiments/screen_esns.py`` and one of
``experiments/screen_against_robot.py`` (``[runs]``): the ESNs that pass on the
robot there are trained as the screen trained them, and each drives the robot's
reference with one tracker setting (``[tracker]``, as in ``robot_esn.py``):

- **Released** (``[release]``): for each of the ESN's unstable fixed points within
  ``path_deg`` of the take's joint path, the arm at rest at its posture, the ESN
  held there for ``hold`` s (its warm-up), then let go for ``duration`` s: when the
  hand arrives within the goal radius, from the release; how long the arm stays
  within ``stay_deg`` of the posture; whether the hand ends within the goal; and
  when the ESN on its own arrives from the same hold.
- **Blocks** (``[blocks]``): from the take's start, the tip held for each of
  ``holds`` s from each of ``onsets`` (task time, s), one block a run of
  ``duration`` s: ``robot_esn.py``'s metrics, and whether the run passes report
  011's rule under the block (it arrives, holds for the hold window, and ends within
  the goal); the arm's posture and its progress along the take's joint path at the
  middle of the block; the lead R there, how far the ESN's output leads that posture
  once its reservoir has settled there (see :mod:`arm_esn_ctrl.screening`); the
  distance from it to the ESN's nearest unstable fixed point; and the delay, how
  much later the hand arrives after the release than the take does from that
  progress.

The run directory receives ``release.csv`` and ``blocks.csv``, one row per run, and
a summary is printed: the robot's release arrival against the ESN's own; and, under
the blocks, the share failing and the delay, of the ESNs with an unstable fixed
point within ``path_deg`` of the path and of those without, and the delay's rank
correlations with the distance to the nearest unstable fixed point and with the
lead. The ESNs are spread over ``--workers`` processes, each computing with one
thread.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from robot_esn import FOLLOWING_WINDOW, RETURN_TOLERANCE, arm_metrics, posed, ringing_settings, tracker_settings
from scipy.stats import rankdata
from screen_esns import run_outcomes

from arm_esn_ctrl.autonomous import Setup, load_setup
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.disturbances import disturbance_spans, make_disturbance
from arm_esn_ctrl.dynamics import ClosedLoop
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.metrics import arrival_index, path_progress
from arm_esn_ctrl.screening import autonomous_runs, held_fixed_points
from arm_esn_ctrl.storage import resolve_run_path, start_run
from arm_esn_ctrl.tracking import EsnSource, TrackerConfig, gain_posture, task_joint_angles, track

SWEPT = ("leak_rate", "spectral_radius", "input_scaling", "seed")
METRICS = (
    "arrived",
    "arrival_time_s",
    "success",
    "hold_observed_s",
    "final_distance_m",
    "reference_lead",
    "following_gain",
)


@dataclass(frozen=True)
class Context:
    """What every ESN's runs need, loaded once per process."""

    config: dict[str, Any]
    esn: dict[str, Any]  # the screen's [esn] table
    setup: Setup
    take: NDArray[np.float64]
    take_progress: NDArray[np.float64]  # the take's progress along its own joint path, never falling back
    take_arrival_s: float
    simulator: dict[str, Any]
    tracker: TrackerConfig
    gains: tuple[NDArray[np.float64], NDArray[np.float64]]


_context: Context | None = None  # this process's, set by init_worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="configuration file (TOML)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="worker processes")
    args = parser.parse_args()
    config, run_dir = start_run(args.config)

    screen_dir = resolve_run_path(config["runs"]["screen"])
    unstable: dict[tuple[float, ...], list[dict[str, str]]] = {}
    for p in read_csv(screen_dir / "fixed_points.csv"):
        if int(p["n_unstable"]) > 0:
            unstable.setdefault(tuple(float(p[c]) for c in SWEPT), []).append(p)
    esns = [r for r in read_csv(resolve_run_path(config["runs"]["outcomes"]) / "esns.csv") if r["passes"] == "True"]
    jobs = [
        (
            {c: (int(float(r[c])) if c == "seed" else float(r[c])) for c in SWEPT},
            unstable.get(tuple(float(r[c]) for c in SWEPT), []),
        )
        for r in esns
    ]
    print(f"{len(jobs)} passing ESNs; {sum(len(fps) for _, fps in jobs)} unstable fixed points among them")

    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
    releases, blocks = [], []
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(config,)
    ) as pool:
        for k, (release_rows, block_rows) in enumerate(pool.map(esn_job, jobs)):
            releases += release_rows
            blocks += block_rows
            if (k + 1) % 50 == 0 or k + 1 == len(jobs):
                print(f"[{k + 1:4d}/{len(jobs)}] {len(releases)} releases, {len(blocks)} blocks", flush=True)

    for name, rows in (("release.csv", releases), ("blocks.csv", blocks)):
        write_csv(run_dir / name, rows)
    summarize(releases, blocks, config)
    print(f"\nWrote the results to {run_dir}")


def init_worker(config: dict[str, Any]) -> None:
    """Load this process's context: the take, as the screen loaded it, and the tracker."""
    global _context
    screen = read_toml(resolve_run_path(config["runs"]["screen"]) / "config.toml")
    setup = load_setup(
        {
            "demonstrations": screen["demonstrations"],
            "esn": {"dt": screen["esn"]["dt"]},
            "evaluation": config["evaluation"],
        }
    )
    take = next(iter(setup.demos.values()))
    with (resolve_run_path(screen["demonstrations"]["run"]) / "config.toml").open("rb") as f:
        simulator = tomllib.load(f)["simulator"]
    settings = tracker_settings(config["tracker"])
    if len(settings) != 1:
        msg = "[tracker] must give one setting"
        raise ValueError(msg)
    setting, table = settings[0], config["tracker"]
    tracker = TrackerConfig(
        setting.law,
        setting.omega,
        table["acceleration_filter"],
        damping=setting.damping,
        reference_velocity=table.get("reference_velocity", True),
        control_period=table.get("control_period"),
    )
    gains = setting.gains(tracker, setup.skeleton, gain_posture(setup.demos.values()))
    arrival = arrival_index(endpoint_positions(setup.skeleton, take), setup.target, setup.radius)
    assert arrival is not None
    _context = Context(
        config,
        screen["esn"],
        setup,
        take,
        np.maximum.accumulate(path_progress(take, take)),
        arrival * float(screen["esn"]["dt"]),
        simulator,
        tracker,
        gains,
    )


def esn_job(job: tuple[dict[str, Any], list[dict[str, str]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One ESN's runs in this worker."""
    assert _context is not None
    return esn_runs(_context, *job)


def esn_runs(
    context: Context, combination: dict[str, Any], unstable: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Train the ESN of ``combination`` and run it: its rows of release.csv and of blocks.csv."""
    config, setup = context.config, context.setup
    esn = ReachingEsn(EsnConfig(**(context.esn | combination)))
    esn.fit([context.take])
    loop = ClosedLoop.from_esn(esn)
    dt = esn.config.dt
    q_unstable = np.radians([[float(p["q1_deg"]), float(p["q2_deg"])] for p in unstable]).reshape(-1, 2)

    table = config["release"]
    hold, n_steps = round(table["hold"] / dt), round(table["duration"] / dt)
    releases = []
    for p, q0 in zip(unstable, q_unstable, strict=True):
        if float(p["path_distance_deg"]) > table["path_deg"]:
            continue
        q = robot_run(context, esn, q0, hold, table["duration"], None)
        robot = run_outcomes(setup, q[np.newaxis], dt)
        own = run_outcomes(setup, autonomous_runs(loop, q0[np.newaxis], hold, n_steps)[0], dt)
        moved = np.flatnonzero(np.degrees(np.abs(q - q0)).max(axis=1) > table["stay_deg"])
        releases.append(
            combination
            | {c: float(p[c]) for c in ("q1_deg", "q2_deg", "path_distance_deg", "progress", "growth_per_s")}
            | {
                "arrival_s": float(robot["arrival"][0]),
                "stays_s": float(moved[0] * dt) if len(moved) else np.inf,
                "ends_at_target": bool(robot["at_target"][0]),
                "own_arrival_s": float(own["arrival"][0]),
            }
        )

    table = config["blocks"]
    start = setup.starts[0].q
    near = bool(len(unstable)) and min(float(p["path_distance_deg"]) for p in unstable) <= config["release"]["path_deg"]
    blocks = []
    for block_hold in table["holds"]:
        for onset in table["onsets"]:
            disturbance = {
                "type": "block",
                "onset": float(onset),
                "release": float(onset) + float(block_hold),
                "stiffness": table["stiffness"],
                "damping": table["damping"],
            }
            log, q = robot_log(context, esn, start, esn.config.warmup_steps, table["duration"], disturbance)
            metrics = arm_metrics(
                log,
                0,
                setup,
                dt,
                [float(onset), float(onset) + float(block_hold) + 0.5],
                disturbance_spans(disturbance),
                config["evaluation"].get("return_tolerance", RETURN_TOLERANCE),
                ringing_settings(config["evaluation"]),
                config["evaluation"].get("following_window", FOLLOWING_WINDOW),
            )
            held = q[round((float(onset) + float(block_hold) / 2) / dt)]
            x, _ = held_fixed_points(loop, held[np.newaxis])
            progress = float(path_progress(held[np.newaxis], context.take)[0])
            taken = float(np.interp(progress, context.take_progress, np.arange(len(context.take)) * dt))
            remaining = context.take_arrival_s - taken  # the take's time from the held progress to its arrival
            arrival = float(metrics["arrival_time_s"]) if metrics["arrived"] else np.inf
            blocks.append(
                combination
                | {"near_path_unstable": near, "hold_s": float(block_hold), "onset_s": float(onset)}
                | {m: metrics[m] for m in METRICS}
                | {
                    "passes": bool(
                        metrics["arrived"]
                        and metrics["success"]
                        and metrics["hold_observed_s"] >= setup.hold
                        and metrics["final_distance_m"] <= setup.radius
                    ),
                    "held_q1_deg": float(np.degrees(held[0])),
                    "held_q2_deg": float(np.degrees(held[1])),
                    "held_progress": progress,
                    "lead_deg": float(np.degrees(np.linalg.norm(loop.posture(x[0]) - held))),
                    "nearest_unstable_deg": float(np.degrees(np.linalg.norm(q_unstable - held, axis=1)).min())
                    if len(q_unstable)
                    else np.inf,
                    "delay_s": arrival - (float(onset) + float(block_hold)) - remaining,
                }
            )
    return releases, blocks


def robot_run(
    context: Context,
    esn: ReachingEsn,
    start_q: NDArray[np.float64],
    warmup_steps: int,
    duration: float,
    disturbance: dict[str, Any] | None,
) -> NDArray[np.float64]:
    """The arm's joint angles over the task, every ESN period, from rest at ``start_q``."""
    return robot_log(context, esn, start_q, warmup_steps, duration, disturbance)[1]


def robot_log(
    context: Context,
    esn: ReachingEsn,
    start_q: NDArray[np.float64],
    warmup_steps: int,
    duration: float,
    disturbance: dict[str, Any] | None,
) -> tuple[Any, NDArray[np.float64]]:
    """The robot's log from rest at ``start_q``, the ESN held there ``warmup_steps``, and the task's joint angles."""
    setup, dt = context.setup, esn.config.dt
    start_hand = endpoint_positions(setup.skeleton, start_q[np.newaxis])[0]
    log = track(
        posed(setup.skeleton, start_q),
        EsnSource(esn),
        context.tracker,
        context.gains,
        period=dt,
        warmup_steps=warmup_steps,
        duration=duration,
        dt=context.simulator["dt"],
        enforce_limits=context.simulator.get("enforce_limits", True),
        external_force=make_disturbance(disturbance, start_hand, setup.target, dt=context.simulator["dt"]),
    )
    return log, task_joint_angles(log, dt, duration)


def summarize(releases: list[dict[str, Any]], blocks: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Print the release arrivals against the ESNs' own, and the blocks' failures and delays by group."""
    robot = np.array([r["arrival_s"] for r in releases])
    own = np.array([r["own_arrival_s"] for r in releases])
    both = np.isfinite(robot) & np.isfinite(own)
    late = config["release"]["late_s"]
    gap = robot[both] - own[both]
    print(f"\nReleased at {len(releases)} unstable fixed points; never arriving: robot {np.isinf(robot).sum()},")
    print(f"  own {np.isinf(own).sum()}; later than {late} s: robot {(robot > late).sum()}, own {(own > late).sum()}.")
    print(f"  Both arrive at {both.sum()}: the robot's arrival less the ESN's own, median {np.median(gap):+.2f} s,")
    print(f"  10th to 90th percentiles {np.percentile(gap, 10):+.2f} to {np.percentile(gap, 90):+.2f} s;")
    print(f"  rank correlation {rank_correlation(robot, own):+.2f}")

    column = {c: np.array([b[c] for b in blocks]) for c in ("leak_rate", "near_path_unstable", "hold_s", "passes")}
    delay, nearest, lead = (np.array([b[c] for b in blocks]) for c in ("delay_s", "nearest_unstable_deg", "lead_deg"))
    lr, near, hold, passes = column["leak_rate"], column["near_path_unstable"], column["hold_s"], column["passes"]
    print("\nBlocks: hold, group: runs, failing share; the delay of those that arrive, median (10th, 90th percentiles)")
    for h in np.unique(hold):
        for label, group in (("an unstable fixed point near the path", near), ("none near the path", ~near)):
            s = (hold == h) & group
            d = delay[s & np.isfinite(delay)]
            print(
                f"  {h:g} s, {label}: {s.sum()}, {(~passes[s]).mean():.3f};"
                f" {np.median(d):+.2f} s ({np.percentile(d, 10):+.2f}, {np.percentile(d, 90):+.2f})"
            )
    print("The delay's rank correlations (never arriving last) with the distance to the nearest unstable fixed point,")
    print("and with the lead at the held posture:")
    for h in np.unique(hold):
        print(f"  {h:g} s:")
        for v in [None, *np.unique(lr)]:
            s = (hold == h) & (True if v is None else lr == v)
            print(
                f"    {'all' if v is None else f'{v:g}'}:"
                f" {rank_correlation(delay[s], nearest[s]):+.2f}, {rank_correlation(delay[s], lead[s]):+.2f}"
            )


def rank_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Spearman's rank correlation over the entries where neither is NaN; an infinity ranks last."""
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 5:
        return np.nan
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])


def read_csv(path: Any) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_toml(path: Any) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def write_csv(path: Any, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
