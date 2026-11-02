# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Screen trained ESNs by where they can rest and how they set off, on their own, without the robot.

    uv run python experiments/screen_esns.py experiments/manual_demonstration_v2_esn_screening/screen.toml

The configuration holds the ``[demonstrations]`` and ``[esn]`` tables of
``experiments/autonomous_esn.py`` and a ``[sweep]`` table that lists values for any
of the ``[esn]`` hyperparameters. Every combination of them is trained on the
demonstrations, as ``experiments/sweep_robot_esn.py`` trains it, and screened fed its
own output (see :mod:`arm_esn_ctrl.screening` and :mod:`arm_esn_ctrl.dynamics`), as
the ``[screen]`` table sets:

- **The fixed points:** the zeros of R(q) over a grid of every joint's range, from its
  lower limit to its upper one every ``grid_step_deg``, each refined by Newton's method on the ESN's update, and
  those that the runs from rest below come to rest at and the grid misses. For
  each: its posture, the hand's distance from the target, its distance from the
  take's joint path and how far along the path it lies, and its eigenvalues: the
  unstable modes, and the slowest mode's modulus, growth rate, time constant (if it
  decays), and period. Whether the held reservoir has one fixed point at each posture, as the
  zeros of R assume, is checked at every ``check_stride``-th posture of the grid in
  each joint, from random states as well as from the reset.
- **The lead:** |R(q)|, how far the ESN's output leads a posture held at q once its
  reservoir has settled there, as its reference leads an arm held still on the
  robot; its median and smallest over the grid's postures within ``band_deg`` of the
  take's joint path, up to ``max_progress`` along it.
- **From rest anywhere:** the ESN on its own for ``duration`` from rest at a
  coarser grid, every ``start_step_deg`` (:func:`arm_esn_ctrl.postures.start_grid`),
  and at ``ring_deg`` around the take's end posture (:func:`ring_directions`: 8
  directions for two joints, 24 for four), each start posture held for the ESN's
  warm-up first:
  how many arrive within the goal radius and end there, and the latest arrival.
- **Held at an unstable fixed point:** the ESN on its own from rest there, held
  for ``release_hold`` from the reset, then released: when it arrives, within
  ``release_duration``.
- **The rates along its run from the take's start** (see
  :func:`arm_esn_ctrl.dynamics.local_rates`), for ``rate_duration``, over windows of
  ``rate_window`` starting every ``rate_stride``: the largest rate at which a change
  of the state across the run grows, and the most a deviation of the output posture
  is scaled by the window's end, each's median and largest over the windows.

The ESNs are spread over ``--workers`` processes, each computing with one thread.
The run directory receives:

- ``screen.csv``: one row per ESN: its swept hyperparameters and its measures;
- ``fixed_points.csv``: one row per fixed point of each ESN, and whether the grid
  found it;
- ``maps.npz``: the swept values of each ESN, the grid (``grid_deg``, each joint's
  angles if they are the same for every joint, else ``grid_deg_q1``, ``grid_deg_q2``,
  ...), and each ESN's R over it and where the held reservoir's Newton's method
  converged.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from global_starts import PATH_SPACING_DEG, closed_loop_run
from numpy.typing import NDArray

from arm_esn_ctrl.autonomous import Setup, load_setup
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.dynamics import ClosedLoop, local_rates
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.metrics import arrival_index, distances_to_path, path_progress, resample_by_length
from arm_esn_ctrl.postures import joint_columns, joint_values, map_grid, start_grid
from arm_esn_ctrl.screening import (
    autonomous_runs,
    find_fixed_points,
    fixed_points_at_rest,
    held_fixed_points,
    posture_map,
    spectrum,
)
from arm_esn_ctrl.storage import start_run

RING_DIRECTIONS = 8  # for two joints, the start postures around the take's end posture, at each ring_deg
HAND_EVERY = 5  # the runs' hands are measured every this many steps


@dataclass(frozen=True)
class Context:
    """What every ESN's screening needs, loaded once per process."""

    config: dict[str, Any]
    setup: Setup
    take: NDArray[np.float64]  # the take's joint angles, every ESN period
    path: NDArray[np.float64]  # the take's joint path, resampled by length


_context: Context | None = None  # this process's, set by init_worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="screening configuration file (TOML)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="worker processes")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    context = load_context(config)
    sweep: dict[str, list[Any]] = config["sweep"]
    combinations = [dict(zip(sweep, values, strict=True)) for values in itertools.product(*sweep.values())]
    print(f"{len(combinations)} ESNs over {', '.join(f'{k} ({len(v)})' for k, v in sweep.items())}")

    # Many processes of one thread each, rather than threads that the processes would contend for.
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
    rows, fixed_rows, maps, converged = [], [], [], []
    started = time.perf_counter()
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(config,)
    ) as pool:
        for k, (row, fixed, residual, held_converged) in enumerate(pool.map(screen_job, combinations)):
            rows.append(row)
            fixed_rows += fixed
            maps.append(residual)
            converged.append(held_converged)
            values = ", ".join(f"{name}={row[name]:g}" for name in sweep)
            print(
                f"[{k + 1:4d}/{len(combinations)}] {values}: {row['fixed_points']} fixed points"
                f" ({row['unstable']} unstable, {row['attracting_off_target']} attracting off the target);"
                f" {row['starts_at_target']}/{row['starts']} starts end at the target;"
                f" {row['elapsed_s']:.0f} s ({time.perf_counter() - started:.0f} s in all)"
            )

    for name, table in (("screen.csv", rows), ("fixed_points.csv", fixed_rows)):
        with (run_dir / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    arrays: dict[str, Any] = {
        "swept": np.array(list(sweep)),
        "values": np.array([[row[name] for name in sweep] for row in rows], dtype=np.float64),
        **grid_arrays(map_grid(context.setup.skeleton, config["screen"]["grid_step_deg"])),
        "residual": np.array(maps),
        "converged": np.array(converged),
    }
    np.savez_compressed(run_dir / "maps.npz", **arrays)
    print(f"\nWrote the results to {run_dir}")


def load_context(config: dict[str, Any]) -> Context:
    """The context of every ESN's screening: the take, as the robot sweeps load it."""
    evaluation = {"duration": config["screen"]["duration"], "hold": 0.0, "start_offsets_deg": [[0.0]]}
    setup = load_setup(
        {"demonstrations": config["demonstrations"], "esn": {"dt": config["esn"]["dt"]}, "evaluation": evaluation}
    )
    if len(setup.demos) != 1:
        msg = "the ESNs are screened against one take: give one demonstration"
        raise ValueError(msg)
    take = next(iter(setup.demos.values()))
    return Context(config, setup, take, resample_by_length(take, np.radians(PATH_SPACING_DEG)))


def init_worker(config: dict[str, Any]) -> None:
    """Load this process's context."""
    global _context
    _context = load_context(config)


def screen_job(combination: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], Any, Any]:
    """Screen one ESN in this worker."""
    assert _context is not None
    return screen(_context, combination)


def screen(
    context: Context, combination: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], NDArray[np.float64], NDArray[np.bool_]]:
    """Train the ESN of ``combination`` and screen it: its row, its fixed points' rows, R, and the convergence map."""
    started = time.perf_counter()
    table, setup = context.config["screen"], context.setup
    esn = ReachingEsn(EsnConfig(**(context.config["esn"] | combination)))
    esn.fit([context.take])
    loop = ClosedLoop.from_esn(esn)
    dt = esn.config.dt
    end_q = context.take[-1]

    # From rest anywhere, and around the take's end posture.
    coarse = [np.radians(values) for values in start_grid(setup.skeleton, table["start_step_deg"])]
    starts = np.array(list(itertools.product(*coarse)))
    ring = np.array([end_q + np.radians(r) * d for r in table["ring_deg"] for d in ring_directions(len(end_q))])
    n_steps = round(table["duration"] / dt)
    grid_q, grid_x = autonomous_runs(loop, starts, esn.config.warmup_steps, n_steps)
    ring_q, ring_x = autonomous_runs(loop, ring, esn.config.warmup_steps, n_steps)
    own_q, own_x = autonomous_runs(loop, context.take[:1], esn.config.warmup_steps, n_steps)

    # The fixed points, from the zeros of R, and where those runs come to rest.
    pmap = posture_map(loop, map_grid(setup.skeleton, table["grid_step_deg"]))
    found = find_fixed_points(loop, pmap)
    at_rest = fixed_points_at_rest(
        loop, np.concatenate([grid_q, ring_q, own_q]), np.concatenate([grid_x, ring_x, own_x]), found.states
    )
    fixed = [describe(context, loop, x) | {"from_grid": True} for x in found.states]
    fixed += [describe(context, loop, x) | {"from_grid": False} for x in at_rest]
    attracting = [f for f in fixed if f["n_unstable"] == 0]
    unstable = [f for f in fixed if f["n_unstable"] > 0]
    at_target = [f for f in attracting if f["target_distance_m"] <= setup.radius]
    target = min(at_target, key=lambda f: f["end_posture_distance_deg"], default=None)

    # Whether the held reservoir has one fixed point at each posture: from random states as well.
    every = (slice(None, None, table["check_stride"]),) * len(pmap.grids_deg)
    q_check = pmap.postures().reshape(*pmap.converged.shape, -1)[every].reshape(-1, len(pmap.grids_deg))
    x_reset = pmap.held[every].reshape(len(q_check), -1)
    rng = np.random.default_rng(0)
    x_random, random_converged = held_fixed_points(loop, q_check, rng.uniform(-1.0, 1.0, x_reset.shape))
    held_multiple = int((np.abs(x_random - x_reset).max(axis=1) > 1e-8)[random_converged].sum())

    # The lead over a posture held near the take's path.
    q_grid = pmap.postures()
    lead = np.degrees(np.linalg.norm(pmap.residual.reshape(len(q_grid), -1), axis=1))
    near = (np.degrees(distances_to_path(q_grid, context.path)) <= table["band_deg"]) & (
        path_progress(q_grid, context.path) <= table["max_progress"]
    )
    least = int(np.flatnonzero(near)[np.argmin(lead[near])])
    grid_runs, ring_runs, own = (run_outcomes(setup, q, dt) for q in (grid_q, ring_q, own_q))

    # Held at each unstable fixed point, then released.
    release = np.nan
    if unstable:
        q_unstable = np.radians([joint_values(f, "q{j}_deg") for f in unstable])
        runs, _ = autonomous_runs(
            loop, q_unstable, round(table["release_hold"] / dt), round(table["release_duration"] / dt)
        )
        release = float(np.max(run_outcomes(setup, runs, dt)["arrival"]))

    # The rates along the run from the take's start.
    _, states = closed_loop_run(esn, context.take[0], round(table["rate_duration"] / dt))
    rates = local_rates(loop, states, round(table["rate_window"] / dt), round(table["rate_stride"] / dt), 1)
    across, posture = rates.across[:, 0], rates.posture.max(axis=1)

    nearest_unstable = min(unstable, key=lambda f: f["path_distance_deg"], default=None)
    row = combination | {
        "held_unconverged": int((~pmap.converged).sum()),
        "held_checked": len(q_check),
        "held_multiple": held_multiple,
        "candidates": found.candidates,
        "candidates_empty": found.empty,
        "fixed_points": len(fixed),
        "fixed_points_at_rest": len(at_rest),
        "attracting": len(attracting),
        "unstable": len(unstable),
        "target_found": target is not None,
        "target_end_posture_distance_deg": np.nan if target is None else target["end_posture_distance_deg"],
        "target_time_constant_s": np.nan if target is None else target["time_constant_s"],
        "target_period_s": np.nan if target is None else target["period_s"],
        "attracting_off_target": len(attracting) - len(at_target),
        "nearest_unstable_path_distance_deg": np.inf
        if nearest_unstable is None
        else nearest_unstable["path_distance_deg"],
        "nearest_unstable_progress": np.nan if nearest_unstable is None else nearest_unstable["progress"],
        "nearest_unstable_growth_per_s": np.nan if nearest_unstable is None else nearest_unstable["growth_per_s"],
        "release_arrival_s": release,
        "lead_median_deg": float(np.median(lead[near])),
        "lead_min_deg": float(lead[least]),
    }
    row |= joint_columns("lead_min_q{j}_deg", np.degrees(q_grid[least])) | {
        "starts": len(starts),
        "starts_arrived": int(np.isfinite(grid_runs["arrival"]).sum()),
        "starts_at_target": int(grid_runs["at_target"].sum()),
        "starts_latest_arrival_s": float(np.max(grid_runs["arrival"])),
        "ring_at_target": int(ring_runs["at_target"].sum()),
        "ring_latest_arrival_s": float(np.max(ring_runs["arrival"])),
        "own_arrival_s": float(own["arrival"][0]),
        "across_rate_median_per_s": float(np.median(across)),
        "across_rate_max_per_s": float(across.max()),
        "posture_scale_median": float(np.median(posture)),
        "posture_scale_max": float(posture.max()),
        "elapsed_s": time.perf_counter() - started,
    }
    fixed_rows = [combination | {"fixed_point": j} | f for j, f in enumerate(fixed)]
    return row, fixed_rows, pmap.residual, pmap.converged


def describe(context: Context, loop: ClosedLoop, x: NDArray[np.float64]) -> dict[str, Any]:
    """A fixed point's row: where it lies, against the take, and its eigenvalues."""
    setup = context.setup
    q = loop.posture(x)
    eigenvalues = spectrum(loop, x)
    modulus = np.abs(eigenvalues)
    slowest = eigenvalues[0]  # the slowest mode to decay, or the fastest to grow
    return joint_columns("q{j}_deg", np.degrees(q)) | {
        "target_distance_m": float(np.linalg.norm(endpoint_positions(setup.skeleton, q[np.newaxis])[0] - setup.target)),
        "end_posture_distance_deg": float(np.degrees(np.linalg.norm(q - context.take[-1]))),
        "path_distance_deg": float(np.degrees(distances_to_path(q[np.newaxis], context.path)[0])),
        "progress": float(path_progress(q[np.newaxis], context.path)[0]),
        "residual": float(np.abs(loop.advance(x) - x).max()),
        "n_unstable": int((modulus > 1.0).sum()),
        "max_modulus": float(modulus[0]),
        "growth_per_s": float(np.log(modulus[0]) / loop.dt),
        "time_constant_s": loop.time_constant(slowest),
        "period_s": np.nan if slowest.imag == 0 else float(2 * np.pi * loop.dt / abs(np.angle(slowest))),
    }


def run_outcomes(setup: Setup, q: NDArray[np.float64], dt: float) -> dict[str, NDArray[Any]]:
    """For each run of ``q`` (runs, steps, joints): when its hand arrives in the goal radius, and if it ends there."""
    arrival, at_target = np.full(len(q), np.inf), np.zeros(len(q), dtype=bool)
    for i, run in enumerate(q):
        hand = endpoint_positions(setup.skeleton, run[::HAND_EVERY])
        k = arrival_index(hand, setup.target, setup.radius)
        if k is not None:
            arrival[i] = k * HAND_EVERY * dt
        at_target[i] = np.linalg.norm(endpoint_positions(setup.skeleton, run[-1:])[0] - setup.target) <= setup.radius
    return {"arrival": arrival, "at_target": at_target}


def ring_directions(n_joints: int) -> NDArray[np.float64]:
    """The directions in joint space of the start postures around the take's end posture, unit vectors.

    For two joints, :data:`RING_DIRECTIONS` equally spaced in their plane; for more, along
    each joint and along every diagonal, the joints all moved by the same angle (24 for
    four joints). For two, these are the same eight.
    """
    if n_joints == 2:
        angles = 2 * np.pi * np.arange(RING_DIRECTIONS) / RING_DIRECTIONS
        return np.array([[np.cos(a), np.sin(a)] for a in angles])
    axes = [sign * row for row in np.eye(n_joints) for sign in (1.0, -1.0)]
    diagonals = [np.array(signs) / np.sqrt(n_joints) for signs in itertools.product((1.0, -1.0), repeat=n_joints)]
    return np.array(axes + diagonals)


def grid_arrays(grids: list[NDArray[np.float64]]) -> dict[str, NDArray[np.float64]]:
    """The grid of R for ``maps.npz``: one array if every joint has the same angles, else one per joint."""
    if all(np.array_equal(g, grids[0]) for g in grids):
        return {"grid_deg": grids[0]}
    return {f"grid_deg_q{j}": g for j, g in enumerate(grids, start=1)}


if __name__ == "__main__":
    main()
