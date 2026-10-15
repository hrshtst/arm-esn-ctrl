# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Run trained ESNs on their own from start postures all over the joint space, from rest (Stage 1).

    uv run python experiments/global_starts.py experiments/manual_demonstration_v2_esn_dynamics/global_starts.toml

Each ESN of the ``[[esn]]`` tables (``key``, ``label``, and ``model``, an
``esn.toml`` of a run, relative to the storage root) is run autonomously, as by
:meth:`arm_esn_ctrl.esn.ReachingEsn.generate`, from every posture of a grid over
both joints' whole range: from -180 deg to 180 deg in steps of ``step_deg``
(``[grid]``), each start posture held for the ESN's warm-up first. The ESNs must
have been trained on the same take, which every run is compared with:

- **the target**: whether the hand arrives within the goal radius and holds there
  (:func:`arm_esn_ctrl.metrics.hold_metrics`, over ``hold`` seconds), and where
  the run ends after ``duration`` seconds (``[evaluation]``). A run ends **at the
  target** when it ends within the goal radius and at rest (slower than
  ``rest_speed_dps``, over the last second), **at rest elsewhere** when it ends at
  rest outside it, **moving** otherwise, and **out of range** when a joint ever
  passes ±180 deg, a joint limit of the arm;
- **the take's path**, in joint space: the distance of the joint angles from the
  take's joint path, and how far along that path the nearest point lies (the
  progress, from 0 at its start to 1 at its end). A run **joins** the path when its
  distance falls below ``path_threshold_deg`` for good (``[analysis]``), at the
  progress it then has;
- **the take's reservoir states**, while the take is fed in (teacher forcing, as in
  training), every ``state_stride`` steps: the distance of the run's state from the
  nearest of them and that state's time in the take (the **phase**), and from the
  take's state at the same time (the time-matched distance), relative to the
  spread of the take's states (their RMS distance from their mean). The states are
  also projected onto the first ``n_components`` principal components of the
  take's;
- **the warm-up**: with each start posture held for ``hold`` seconds
  (``[settling]``), far longer than the warm-up, how far the state still moves
  after the warm-up, relative to the same spread. A small value means that the
  state the ESN starts from depends on the start posture alone, not on how long it
  was held.

The run directory receives, for each ESN:

- ``starts_<key>.csv``: one row per start posture, with the measures above;
- ``runs_<key>.npz``: every run's joint angles, distance from the take's path, and
  progress, at every step; its state measures and projections, every
  ``state_stride`` steps; the size of each step of the state while the start
  posture is held; the states at the start of the task and at the end of the run;
  and the take's projections;

and ``take.npz``: the take's joint angles and hand positions, and the target.
The figures are drawn by ``reports/015-what-the-esns-learned/make_figures.py``.
"""

from __future__ import annotations

import argparse
import csv
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from arm_esn_ctrl.autonomous import Setup, load_setup
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.metrics import distances_to_path, hold_metrics, join_index, path_progress, resample_by_length
from arm_esn_ctrl.states import principal_components, teacher_forced_states
from arm_esn_ctrl.storage import resolve_run_path, start_run

JOINT_LIMIT_DEG = 180.0  # the arm's joints turn from -180 to 180 deg
PATH_SPACING_DEG = 0.25  # the take's joint path is resampled this finely before measuring against it
FIRST_DIRECTION = 0.2  # a run's first direction is its mean velocity over this long from the start (s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    step = config["grid"]["step_deg"]
    grid = grid_postures(step)
    evaluation = {"duration": config["evaluation"]["duration"], "hold": config["evaluation"]["hold"]}
    evaluation["extra_starts"] = {"grid": grid.tolist()}
    analysis, settling = config["analysis"], config["settling"]
    trained_on = None
    for table in config["esn"]:
        esn_path = resolve_run_path(table["model"])
        esn = ReachingEsn.load(esn_path)
        with (esn_path.parent / "config.toml").open("rb") as f:
            demonstrations = tomllib.load(f)["demonstrations"]
        if trained_on not in (None, demonstrations):
            msg = f"the ESNs must be trained on the same take: {table['key']} was trained on {demonstrations}"
            raise ValueError(msg)
        trained_on = demonstrations
        setup = load_setup({"demonstrations": demonstrations, "esn": {"dt": esn.config.dt}, "evaluation": evaluation})
        if len(setup.demos) != 1:
            msg = "the runs are compared with one take: the ESNs must be trained on one demonstration"
            raise ValueError(msg)
        if not (run_dir / "take.npz").exists():
            take = next(iter(setup.demos.values()))
            np.savez_compressed(
                run_dir / "take.npz",
                times=esn.config.dt * np.arange(len(take)),
                q=take,
                hand=next(iter(setup.demo_hands.values())),
                target=setup.target,
                radius=setup.radius,
            )
        print(f"{table['label']} ({table['model']}): {len(setup.starts)} start postures every {step:g} deg")
        rows, arrays = run_esn(esn, setup, analysis, settling)
        with (run_dir / f"starts_{table['key']}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(run_dir / f"runs_{table['key']}.npz", **arrays)
        print_summary(rows)
    print(f"\nWrote the results to {run_dir}")


def grid_postures(step_deg: float) -> NDArray[np.float64]:
    """The start postures (deg): both joints from -180 deg to 180 deg in steps of ``step_deg``, the first fastest."""
    values = np.arange(-JOINT_LIMIT_DEG, JOINT_LIMIT_DEG - step_deg / 2, step_deg)
    q2, q1 = np.meshgrid(values, values, indexing="ij")
    return np.column_stack([q1.ravel(), q2.ravel()])


def closed_loop_run(
    esn: ReachingEsn, start_q: NDArray[np.float64], n_steps: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """An autonomous run from ``start_q``, as :meth:`ReachingEsn.generate` runs it, and its reservoir states.

    Returns the joint angles, from ``start_q`` at time 0, and the state at every
    time, which has seen the input up to it: ``n_steps + 1`` of each.
    """
    esn.reset()
    for _ in range(esn.config.warmup_steps):
        esn.step(start_q)
    output = esn.step(start_q)
    q, states = [start_q], [esn.state()]
    for _ in range(n_steps):
        q.append(output)
        output = esn.step(output)
        states.append(esn.state())
    return np.array(q), np.array(states)


def held_steps(esn: ReachingEsn, start_q: NDArray[np.float64], n_steps: int) -> NDArray[np.float64]:
    """The reservoir states while ``start_q`` is held for ``n_steps`` steps from the reset, one per step."""
    esn.reset()
    states = []
    for _ in range(n_steps):
        esn.step(start_q)
        states.append(esn.state())
    return np.array(states)


def nearest_states(
    states: NDArray[np.float64], reference: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """For each of ``states``, the distance to the nearest of the ``reference`` states and its index."""
    squared = np.sum(states**2, axis=1)[:, None] + np.sum(reference**2, axis=1)[None, :] - 2.0 * states @ reference.T
    nearest = np.argmin(squared, axis=1)
    rows = np.arange(len(states))
    return np.sqrt(np.maximum(squared[rows, nearest], 0.0)), nearest


def outcome(
    q: NDArray[np.float64],
    hand: NDArray[np.float64],
    target: NDArray[np.float64],
    radius: float,
    dt: float,
    rest: float,
) -> str:
    """Where a run ends: ``target``, ``rest elsewhere``, ``moving``, or ``out of range`` (see the module).

    ``q`` and ``hand`` are sampled every ``dt`` seconds; the run is at rest at its
    end when its joints move slower than ``rest`` (deg/s) over the last second.
    """
    if np.degrees(np.abs(q).max()) > JOINT_LIMIT_DEG:
        return "out of range"
    if final_speed_dps(q, dt) >= rest:
        return "moving"
    return "target" if np.linalg.norm(hand[-1] - target) <= radius else "rest elsewhere"


def final_speed_dps(q: NDArray[np.float64], dt: float) -> float:
    """How fast the joint angles ``q``, sampled every ``dt`` seconds, move over the last second, in deg/s."""
    n = round(1.0 / dt)
    return float(np.degrees(np.linalg.norm(q[-1] - q[-1 - n])))


def run_esn(
    esn: ReachingEsn, setup: Setup, analysis: dict[str, Any], settling: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run ``esn`` from every start posture of ``setup``; the rows of ``starts_<key>.csv`` and the arrays."""
    dt, warmup = esn.config.dt, esn.config.warmup_steps
    take = next(iter(setup.demos.values()))
    path = resample_by_length(take, np.radians(PATH_SPACING_DEG))
    taught = teacher_forced_states(esn, take)[warmup:]  # the take's states over the task, from t = 0
    components = principal_components(taught)
    spread = float(np.sqrt(np.mean(np.sum((taught - components.mean) ** 2, axis=1))))
    stride, n_components = analysis["state_stride"], analysis["n_components"]
    n_steps = len(setup.times) - 1
    hold_steps = round(settling["hold"] / dt)
    first = round(FIRST_DIRECTION / dt)
    print(f"  spread of the take's states {spread:.3f}; first components {np.round(components.explained[:3], 3)}")

    rows, q_runs, distance_runs, progress_runs = [], [], [], []
    nearest_runs, phase_runs, matched_runs, projections, held_runs, starts, finals = [], [], [], [], [], [], []
    for i, start in enumerate(setup.starts):
        held = held_steps(esn, start.q, hold_steps)
        held_runs.append(np.linalg.norm(np.diff(held, axis=0), axis=1) / spread)
        q, states = closed_loop_run(esn, start.q, n_steps)
        hand = endpoint_positions(setup.skeleton, q)
        distance = np.degrees(distances_to_path(q, path))
        progress = path_progress(q, path)
        sampled = states[::stride]
        nearest, phase = nearest_states(sampled, taught)
        matched = taught[np.minimum(np.arange(0, len(states), stride), len(taught) - 1)]
        joined = join_index(distance, analysis["path_threshold_deg"])
        hold = hold_metrics(setup.times, hand, setup.target, setup.radius, setup.hold)
        direction = np.degrees(q[first] - q[0]) / (first * dt)
        rows.append(
            {
                "start": i,
                "q1_deg": round(float(np.degrees(start.q[0])), 6),
                "q2_deg": round(float(np.degrees(start.q[1])), 6),
                "start_target_distance_m": float(np.linalg.norm(hand[0] - setup.target)),
                "start_path_distance_deg": float(distance[0]),
                "start_progress": float(progress[0]),
                "outcome": outcome(q, hand, setup.target, setup.radius, dt, analysis["rest_speed_dps"]),
            }
            | {key: hold[key] for key in ("arrived", "arrival_time_s", "success")}
            | {
                "final_q1_deg": float(np.degrees(q[-1, 0])),
                "final_q2_deg": float(np.degrees(q[-1, 1])),
                "final_distance_m": float(np.linalg.norm(hand[-1] - setup.target)),
                "final_speed_dps": final_speed_dps(q, dt),
                "max_path_distance_deg": float(distance.max()),
                "join_time_s": float("nan") if joined is None else float(setup.times[joined]),
                "join_progress": float("nan") if joined is None else float(progress[joined]),
                "first_dq1_dps": float(direction[0]),
                "first_dq2_dps": float(direction[1]),
                "first_step_deg": float(np.degrees(np.linalg.norm(q[1] - q[0]))),
                "start_state_distance": float(nearest[0] / spread),
                "start_phase_s": float(phase[0] * dt),
                "settling_change": float(np.linalg.norm(held[-1] - held[warmup]) / spread),
            }
        )
        q_runs.append(q.astype(np.float32))
        distance_runs.append(distance.astype(np.float32))
        progress_runs.append(progress.astype(np.float32))
        nearest_runs.append((nearest / spread).astype(np.float32))
        phase_runs.append((phase * dt).astype(np.float32))
        matched_runs.append((np.linalg.norm(sampled - matched, axis=1) / spread).astype(np.float32))
        projections.append(components.project(sampled, n_components).astype(np.float32))
        starts.append(states[0])
        finals.append(states[-1])
        if (i + 1) % 216 == 0:
            print(f"  {i + 1} of {len(setup.starts)} runs")
    arrays = {
        "times": setup.times,
        "state_times": setup.times[::stride],
        "held_times": dt * (np.arange(1, hold_steps) - warmup),  # when each step ends, on the task clock
        "start_q_deg": np.array([[row["q1_deg"], row["q2_deg"]] for row in rows]),
        "q": np.array(q_runs),
        "path_distance_deg": np.array(distance_runs),
        "progress": np.array(progress_runs),
        "state_distance": np.array(nearest_runs),
        "phase_s": np.array(phase_runs),
        "matched_distance": np.array(matched_runs),
        "projections": np.array(projections),
        "held_step": np.array(held_runs, dtype=np.float32),
        "start_states": np.array(starts),
        "final_states": np.array(finals),
        "take_projections": components.project(taught, n_components),
        "explained": components.explained,
        "spread": spread,
    }
    return rows, arrays


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print where the runs end, when they arrive, and how they join the take's path."""
    outcomes = [row["outcome"] for row in rows]
    print("  " + ", ".join(f"{name}: {outcomes.count(name)}" for name in dict.fromkeys(outcomes)))
    arrival = np.array([row["arrival_time_s"] for row in rows], dtype=float)
    arrived = arrival[np.isfinite(arrival)]
    if arrived.size:
        quantiles = np.round(np.quantile(arrived, [0.0, 0.1, 0.5, 0.9, 1.0]), 2).tolist()
        print(f"  {arrived.size} arrive; arrival time min, 10%, median, 90%, max: {quantiles} s")
    joined = np.array([row["join_progress"] for row in rows], dtype=float)
    print(
        f"  {np.isfinite(joined).sum()} join the take's path; median progress when they join"
        f" {np.nanmedian(joined) if np.isfinite(joined).any() else float('nan'):.2f}"
    )
    change = np.array([row["settling_change"] for row in rows])
    print(f"  state change after the warm-up, held: median {np.median(change):.3f}, max {change.max():.3f}")


if __name__ == "__main__":
    main()
