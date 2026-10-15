# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Find where trained ESNs fed their own output can rest, and how fast those points and the take's path attract.

    uv run python experiments/fixed_points.py experiments/manual_demonstration_v2_esn_dynamics/fixed_points.toml

Each ESN of the ``[[esn]]`` tables (``key``, ``label``, and ``model``) is read as
the map G of its reservoir state that its autonomous runs follow
(:class:`arm_esn_ctrl.dynamics.ClosedLoop`). A run of ``global_starts.py`` with the
same ESNs (``run`` in ``[global_starts]``, relative to the storage root) gives,
for each start posture of its grid, the state at the start of the task and at the
end of the run. From them:

- **the fixed points**: Newton's method seeks a fixed point of G from each of
  those states. Fixed points closer than ``distinct`` (``[analysis]``, relative
  to the spread of the take's states) are one. The eigenvalues of the Jacobian at
  each tell whether it attracts the states near it (all inside the unit circle)
  and how fast: the slowest mode's time constant;
- **the warm-up**: with a start posture held, the reservoir follows F(., u)
  instead. From the state at the start of the task, F is iterated for
  ``held_duration`` seconds, and Newton's method then seeks its fixed point there.
  Its slowest eigenvalue tells how long the reservoir takes to settle at it, and
  the distance of the state after the warm-up from it how far the warm-up leaves
  the reservoir from settled.

Along the ESN's own run from the take's start, for the take's duration,
:func:`arm_esn_ctrl.dynamics.local_rates` measures, over windows of each of
``windows`` seconds starting every ``stride`` steps, how fast changes of the state
across the run decay, and how much a deviation of the output posture shrinks. Over
short windows, some changes can grow for a while before they decay, so the rates
depend on the window's length.

The run directory receives, for each ESN:

- ``fixed_points_<key>.csv``: one row per fixed point: its posture and its hand's
  distance from the target, the largest eigenvalue modulus, the number of
  eigenvalues outside the unit circle, the slowest mode's time constant and, if it
  oscillates, its period, and how many of the states from the starts and the ends
  of the runs lead to it;
- ``starts_<key>.csv``: one row per start posture: the fixed point that Newton's
  method reaches from the state at the start of its run and from that at its end
  (-1: none), and the held reservoir's fixed point: whether it was found, the
  largest eigenvalue modulus, the slowest time constant, the distance from it of
  the state after the warm-up, and the size of the last held step, relative to
  the spread;
- ``spectra_<key>.npz``: every fixed point's eigenvalues, by decreasing modulus,
  and, for the slowest modes at each, how the output posture moves (rad, complex);
- ``rates_<key>.npz``: the run's joint angles, and for each window length ``<w>``
  (s), for each window, its start time (``starts_s_<w>``), the run's progress along
  the take's joint path at its start (``progress_<w>``), and the rates
  (``exponents_<w>``, ``across_<w>``, and ``posture_<w>``).
"""

from __future__ import annotations

import argparse
import csv
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from global_starts import closed_loop_run
from numpy.typing import NDArray

from arm_esn_ctrl.autonomous import load_setup
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.dynamics import ClosedLoop, local_rates
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.metrics import path_progress
from arm_esn_ctrl.storage import resolve_run_path, start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    step1 = resolve_run_path(config["global_starts"]["run"])
    with (step1 / "config.toml").open("rb") as f:
        step1_models = {table["key"]: table["model"] for table in tomllib.load(f)["esn"]}
    analysis = config["analysis"]
    for table in config["esn"]:
        key = table["key"]
        if step1_models.get(key) != table["model"]:
            msg = f"the run {config['global_starts']['run']} did not run {key} with {table['model']}"
            raise ValueError(msg)
        esn_path = resolve_run_path(table["model"])
        esn = ReachingEsn.load(esn_path)
        with (esn_path.parent / "config.toml").open("rb") as f:
            demonstrations = tomllib.load(f)["demonstrations"]
        setup = load_setup(
            {
                "demonstrations": demonstrations,
                "esn": {"dt": esn.config.dt},
                "evaluation": {"duration": 1.0, "hold": 0.0, "start_offsets_deg": [[0.0, 0.0]]},
            }
        )
        loop = ClosedLoop.from_esn(esn)
        runs = dict(np.load(step1 / f"runs_{key}.npz"))
        spread = float(runs["spread"])
        print(f"{table['label']}: {len(runs['start_q_deg'])} start postures of {config['global_starts']['run']}")

        found: list[NDArray[np.float64]] = []
        reached: dict[str, list[int]] = {}
        for kind in ("start", "final"):
            reached[kind] = []
            for x in runs[f"{kind}_states"]:
                fixed, converged = loop.fixed_point(x)
                reached[kind].append(merge(found, fixed, converged, analysis["distinct"] * spread))
        rows: list[dict[str, Any]] = []
        spectra: dict[str, Any] = {}
        for j, x in enumerate(found):
            eigenvalues, vectors = np.linalg.eig(loop.jacobian(x))
            order = np.argsort(-np.abs(eigenvalues))
            eigenvalues, vectors = eigenvalues[order], vectors[:, order]
            q = loop.posture(x)
            slowest = eigenvalues[0]
            rows.append(
                {
                    "fixed_point": j,
                    "q1_deg": float(np.degrees(q[0])),
                    "q2_deg": float(np.degrees(q[1])),
                    "target_distance_m": float(
                        np.linalg.norm(endpoint_positions(setup.skeleton, q[np.newaxis])[0] - setup.target)
                    ),
                    "residual": float(np.abs(loop.advance(x) - x).max()),
                    "max_modulus": float(np.abs(slowest)),
                    "n_unstable": int(np.count_nonzero(np.abs(eigenvalues) > 1.0)),
                    "time_constant_s": loop.time_constant(complex(slowest)),
                    "period_s": period(complex(slowest), loop.dt),
                    "from_starts": reached["start"].count(j),
                    "from_ends": reached["final"].count(j),
                }
            )
            spectra[f"eigenvalues_{j}"] = eigenvalues
            # How the output posture moves in the slowest modes.
            spectra[f"modes_{j}"] = (
                loop.half_range[:, np.newaxis] * loop.readout.T @ vectors[:, : analysis["n_modes"]]
            ).T
            spectra[f"state_{j}"] = x
        write_csv(run_dir / f"fixed_points_{key}.csv", rows)
        np.savez_compressed(run_dir / f"spectra_{key}.npz", **spectra)
        print_fixed_points(rows, len(runs["start_q_deg"]))

        held = held_settling(loop, runs, round(analysis["held_duration"] / loop.dt), spread)
        write_csv(
            run_dir / f"starts_{key}.csv",
            [
                {"start": i, "q1_deg": float(q[0]), "q2_deg": float(q[1])}
                | {"start_fixed_point": reached["start"][i], "final_fixed_point": reached["final"][i]}
                | held[i]
                for i, q in enumerate(runs["start_q_deg"])
            ],
        )
        print_held(held, esn.config.warmup)

        take = next(iter(setup.demos.values()))
        q, states = closed_loop_run(esn, take[0], len(take) - 1)
        mismatch = float(np.abs(loop.posture(states[:-1]) - q[1:]).max())
        if mismatch > 1e-9:
            msg = f"the map G does not reproduce the ESN's run: its postures differ by up to {mismatch:.2g} rad"
            raise RuntimeError(msg)
        arrays: dict[str, Any] = {"times": loop.dt * np.arange(len(q)), "q": q, "windows_s": analysis["windows"]}
        print(f"  along its run (G reproduces it to {mismatch:.1g} rad), medians over the windows:")
        for seconds in analysis["windows"]:
            rates = local_rates(loop, states, round(seconds / loop.dt), analysis["stride"], analysis["n_rates"])
            arrays |= {
                f"starts_s_{seconds:g}": rates.starts * loop.dt,
                f"progress_{seconds:g}": path_progress(q[rates.starts], take),
                f"exponents_{seconds:g}": rates.exponents,
                f"across_{seconds:g}": rates.across,
                f"posture_{seconds:g}": rates.posture,
            }
            print(
                f"    {seconds:g} s: the largest rate across the run {np.median(rates.across[:, 0]):+.2f} /s"
                f" ({rates.across[:, 0].min():+.2f} to {rates.across[:, 0].max():+.2f}); a deviation of the posture"
                f" scaled by {np.median(rates.posture[:, 0]):.2f} at most"
            )
        np.savez_compressed(run_dir / f"rates_{key}.npz", **arrays)
    print(f"\nWrote the results to {run_dir}")


def merge(found: list[NDArray[np.float64]], x: NDArray[np.float64], converged: bool, tolerance: float) -> int:
    """The index in ``found`` of the fixed point ``x``, added unless another is within ``tolerance``; -1 if unfound."""
    if not converged:
        return -1
    for j, other in enumerate(found):
        if np.linalg.norm(other - x) <= tolerance:
            return j
    found.append(x)
    return len(found) - 1


def period(eigenvalue: complex, dt: float) -> float:
    """The period (s) of a mode of ``eigenvalue``, a step of ``dt`` seconds; NaN if it does not oscillate."""
    angle = abs(np.angle(eigenvalue))
    return float(2 * np.pi * dt / angle) if angle > 1e-12 else float("nan")


def held_settling(loop: ClosedLoop, runs: dict[str, Any], n_steps: int, spread: float) -> list[dict[str, Any]]:
    """For each start posture held, the reservoir's fixed point from the state after the warm-up (see the module)."""
    u = loop.normalize(np.radians(runs["start_q_deg"]))
    x = previous = runs["start_states"].copy()
    for _ in range(n_steps):  # every start posture at once
        previous, x = x, loop.held(x, u)
    last_step = np.linalg.norm(x - previous, axis=1) / spread
    result = []
    for i in range(len(x)):
        fixed, converged = loop.fixed_point(x[i], u[i])
        modulus = float(np.abs(np.linalg.eigvals(loop.held_jacobian(fixed, u[i]))).max()) if converged else float("nan")
        result.append(
            {
                "held_converged": converged,
                "held_max_modulus": modulus,
                "held_time_constant_s": loop.time_constant(modulus) if converged else float("nan"),
                "warmup_gap": float(np.linalg.norm(runs["start_states"][i] - fixed) / spread)
                if converged
                else float("nan"),
                "held_last_step": float(last_step[i]),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` to the CSV file ``path``, one column per key."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_fixed_points(rows: list[dict[str, Any]], n_starts: int) -> None:
    """Print each fixed point found, and from how many states it was found."""
    for row in rows:
        kind = "attracting" if row["n_unstable"] == 0 else f"{row['n_unstable']} unstable modes"
        print(
            f"  fixed point {row['fixed_point']} at ({row['q1_deg']:.2f}, {row['q2_deg']:.2f}) deg,"
            f" {1000 * row['target_distance_m']:.1f} mm from the target: {kind}, max |λ| {row['max_modulus']:.5f},"
            f" slowest {row['time_constant_s']:.2f} s; from {row['from_starts']} of {n_starts} starts and"
            f" {row['from_ends']} ends"
        )


def print_held(held: list[dict[str, Any]], warmup: float) -> None:
    """Print how the held reservoir settles, and how far the warm-up leaves it from settled."""
    converged = [row for row in held if row["held_converged"]]
    print(f"  held: a fixed point found for {len(converged)} of {len(held)} start postures", end="")
    if converged:
        tau = np.array([row["held_time_constant_s"] for row in converged])
        gap = np.array([row["warmup_gap"] for row in converged])
        print(
            f", slowest time constant median {np.median(tau):.2f} s (max {tau.max():.2f}),"
            f" left by the {warmup:g} s warm-up at median {np.median(gap):.3f} of the spread (max {gap.max():.3f})"
        )
    else:
        print()


if __name__ == "__main__":
    main()
