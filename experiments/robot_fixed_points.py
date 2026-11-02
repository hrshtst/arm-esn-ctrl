# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Read trained ESNs and the robot as one system: where it rests, how fast it attracts, and how it answers pulses.

    uv run python experiments/robot_fixed_points.py \\
        experiments/manual_demonstration_v2_consistent_robot_loop/robot_fixed_points.toml

The configuration names a run of ``push_pulses.py`` and one of ``fixed_points.py`` with
the same ESNs (``[runs]``). Each ESN of the pulses' run drives the arm with the pulses'
tracker, and the ESN and the arm are read as one map of the state z = (x, q, q̇), the
reservoir's state and the arm's joint angles and velocities, every ESN period
(:class:`arm_esn_ctrl.dynamics.RobotLoop`):

- **the fixed points**: each of the ESN's own fixed points that ``fixed_points.py``
  found, the arm at rest at its posture, is one of the system's; the eigenvalues of the
  system's Jacobian there tell whether it attracts and how fast, against those of the
  ESN on its own;
- **the rates along the robot's run** from the take's start, for the take's duration,
  as ``fixed_points.py`` measures them along the ESN's own run
  (:func:`arm_esn_ctrl.dynamics.local_rates`, ``[rates]``), the arm's joint angles
  weighed as the ESN's input and their velocities by how far they move them in a
  period;
- **the pulses, linearized**: each of the pulses' run's pulses on the ESN's arm,
  predicted by the system's Jacobians along the undisturbed run, from the pulse's
  onset for ``window`` seconds (``[linear]``), the pulse's force at each simulation
  step as the run had it (:func:`arm_esn_ctrl.dynamics.simulated_forces`): the
  deviation of the arm's joint angles and of the reference from the undisturbed run's,
  predicted and simulated.

The run directory receives, for each ESN:

- ``fixed_points_<key>.csv``: one row per fixed point: its posture, the map's residual
  there, and, on its own and with the arm, the largest eigenvalue modulus, the number
  of eigenvalues outside the unit circle, the slowest mode's growth rate and time
  constant and, if it oscillates, its period;
- ``spectra_<key>.npz``: every fixed point's eigenvalues, by decreasing modulus, on its
  own and with the arm;
- ``rates_<key>.npz``: as ``fixed_points.py``'s, along the robot's run;
- ``pulses_<key>.csv``: one row per pulse: the simulated deviation's RMS over the
  window, and the prediction's RMS error, of the joint angles and of the reference
  (deg), and the error relative to the deviation;
- ``pulses_<key>.npz``: every pulse's deviations over the window, simulated and
  predicted, every ESN period.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from fixed_points import period
from numpy.typing import NDArray
from push_pulses import Pulse, load_context
from robot_esn import posed

from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.disturbances import make_disturbance
from arm_esn_ctrl.dynamics import ClosedLoop, RobotLoop, StateRecorder, local_rates, simulated_forces
from arm_esn_ctrl.metrics import path_progress
from arm_esn_ctrl.storage import resolve_run_path, start_run
from arm_esn_ctrl.tracking import track


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment configuration file (TOML)")
    args = parser.parse_args()
    config, run_dir = start_run(args.config)
    pulses_dir = resolve_run_path(config["runs"]["pulses"])
    fixed_dir = resolve_run_path(config["runs"]["fixed_points"])
    pulses_config, fixed_config = read_toml(pulses_dir / "config.toml"), read_toml(fixed_dir / "config.toml")
    if {t["key"]: t["model"] for t in pulses_config["esn"]} != {t["key"]: t["model"] for t in fixed_config["esn"]}:
        msg = "the pulses' run and the fixed points' run must hold the same ESNs"
        raise ValueError(msg)
    context = load_context(pulses_config)
    setup = context.setup
    take = next(iter(setup.demos.values()))
    runs = dict(np.load(pulses_dir / "runs.npz"))
    print(f"The pulses' tracker: kp {context.gains[0]}, kd {context.gains[1]}")

    for table in pulses_config["esn"]:
        key, esn = table["key"], context.esns[table["key"]]
        loop = ClosedLoop.from_esn(esn)
        robot = RobotLoop.from_tracker(
            loop,
            setup.skeleton,
            context.tracker,
            context.gains,
            context.simulator["dt"],
            enforce_limits=context.simulator.get("enforce_limits", True),
        )
        print(f"{table['label']} and the arm: {robot.dimension} values in the state")

        rows, spectra = fixed_points(robot, dict(np.load(fixed_dir / f"spectra_{key}.npz")))
        write_csv(run_dir / f"fixed_points_{key}.csv", rows)
        np.savez_compressed(run_dir / f"spectra_{key}.npz", **spectra)
        for row in rows:
            print(
                f"  fixed point {row['fixed_point']} at ({row['q1_deg']:.2f}, {row['q2_deg']:.2f}) deg: growth"
                f" {row['growth_per_s']:+.3f} /s on its own, {row['system_growth_per_s']:+.3f} /s with the arm;"
                f" unstable modes {row['n_unstable']} and {row['system_n_unstable']}; residual {row['residual']:.1e}"
            )

        # The robot's run from the take's start, its states at every instant, for the pulses' duration.
        recorder = StateRecorder(esn)
        duration = float(pulses_config["evaluation"]["duration"])
        warmup = esn.config.warmup_steps
        log = track(
            posed(setup.skeleton, setup.starts[0].q),
            recorder,
            context.tracker,
            context.gains,
            period=loop.dt,
            warmup_steps=warmup,
            duration=duration,
            dt=context.simulator["dt"],
            enforce_limits=context.simulator.get("enforce_limits", True),
        )
        states = robot.run_states(log, recorder, warmup)
        nominal = int(np.flatnonzero((runs["arm"] == key) & ~np.isfinite(runs["onset"]))[0])
        q_run = states[:, loop.n_neurons : loop.n_neurons + 2]
        gap = float(np.abs(q_run - runs["q"][nominal, : len(q_run)]).max())
        mismatch = max(float(np.abs(robot.advance(z) - z_next).max()) for z, z_next in itertools.pairwise(states[1:]))
        if gap > 1e-6 or mismatch > 1e-9:
            msg = f"the run is not the pulses' undisturbed run ({gap:.1g} rad) or the map's ({mismatch:.1g})"
            raise RuntimeError(msg)
        print(f"  the robot's run is the pulses' undisturbed run to {gap:.1g} rad, and the map's to {mismatch:.1g}")

        rates = config["rates"]
        along = states[1 : len(take)]  # from the second instant, each state the image of the one before
        arrays: dict[str, Any] = {
            "times": loop.dt * np.arange(1, len(take)),
            "q": q_run[1 : len(take)],
            "windows_s": rates["windows"],
        }
        print("  along the robot's run, medians over the windows:")
        for seconds in rates["windows"]:
            r = local_rates(robot, along, round(seconds / loop.dt), rates["stride"], rates["n_rates"])
            arrays |= {
                f"starts_s_{seconds:g}": (r.starts + 1) * loop.dt,
                f"progress_{seconds:g}": path_progress(along[r.starts, loop.n_neurons : loop.n_neurons + 2], take),
                f"exponents_{seconds:g}": r.exponents,
                f"across_{seconds:g}": r.across,
                f"posture_{seconds:g}": r.posture,
            }
            print(
                f"    {seconds:g} s: the largest rate across the run {np.median(r.across[:, 0]):+.2f} /s"
                f" ({r.across[:, 0].min():+.2f} to {r.across[:, 0].max():+.2f}); a deviation of the arm's posture"
                f" scaled by {np.median(r.posture[:, 0]):.2f} at most"
            )
        np.savez_compressed(run_dir / f"rates_{key}.npz", **arrays)

        pulse_rows, pulse_arrays = linear_pulses(
            context, robot, states, runs, key, float(pulses_config["pulses"]["duration"]), config["linear"]["window"]
        )
        write_csv(run_dir / f"pulses_{key}.csv", pulse_rows)
        np.savez_compressed(run_dir / f"pulses_{key}.npz", **pulse_arrays)
        for force in sorted({row["force_n"] for row in pulse_rows}):
            error = np.array([row["q_relative_error"] for row in pulse_rows if row["force_n"] == force])
            print(
                f"  pulses of {force:g} N, linearized: the joint angles' relative error median {np.median(error):.3f},"
                f" 90th percentile {np.quantile(error, 0.9):.3f}, max {error.max():.3f}"
            )
    print(f"\nWrote the results to {run_dir}")


def fixed_points(robot: RobotLoop, spectra: dict[str, NDArray[Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The ESN's fixed points of ``fixed_points.py``'s ``spectra`` (their states), on its own and with the arm."""
    loop = robot.loop
    rows, arrays = [], {}
    for j in range(sum(name.startswith("state_") for name in spectra)):
        x = spectra[f"state_{j}"]
        z = robot.at_rest(x)
        q = loop.posture(x)
        row: dict[str, Any] = {
            "fixed_point": j,
            "q1_deg": float(np.degrees(q[0])),
            "q2_deg": float(np.degrees(q[1])),
            "residual": float(np.abs(robot.advance(z) - z).max()),
        }
        for prefix, jacobian in (("", loop.jacobian(x)), ("system_", robot.jacobian(z))):
            eigenvalues = np.linalg.eigvals(jacobian)
            eigenvalues = eigenvalues[np.argsort(-np.abs(eigenvalues))]
            row |= describe(eigenvalues, loop.dt, prefix)
            arrays[f"{prefix}eigenvalues_{j}"] = eigenvalues
        rows.append(row)
    return rows, arrays


def describe(eigenvalues: NDArray[Any], dt: float, prefix: str = "") -> dict[str, Any]:
    """The largest modulus, the unstable modes, and the slowest mode's growth, time constant, and period."""
    slowest = complex(eigenvalues[0])
    modulus = abs(slowest)
    return {
        f"{prefix}max_modulus": modulus,
        f"{prefix}n_unstable": int(np.count_nonzero(np.abs(eigenvalues) > 1.0)),
        f"{prefix}growth_per_s": float(np.log(modulus) / dt),
        f"{prefix}time_constant_s": float(-dt / np.log(modulus)) if modulus < 1.0 else float("inf"),
        f"{prefix}period_s": period(slowest, dt),
    }


def linear_pulses(
    context: Any,
    robot: RobotLoop,
    states: NDArray[np.float64],
    runs: dict[str, NDArray[Any]],
    key: str,
    duration: float,
    window: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Each pulse, ``duration`` s long, on the ESN ``key``'s arm, predicted along the undisturbed run's ``states``."""
    loop, setup = robot.loop, context.setup
    dt, h = loop.dt, context.simulator["dt"]
    n, steps = loop.n_neurons, round(window / dt)
    warmup_steps = context.esns[key].config.warmup_steps
    warmup = warmup_steps * dt  # as track() delays the force
    start_hand = endpoint_positions(setup.skeleton, setup.starts[0].q[np.newaxis])[0]
    nominal = int(np.flatnonzero((runs["arm"] == key) & ~np.isfinite(runs["onset"]))[0])
    pulsed = np.flatnonzero((runs["arm"] == key) & np.isfinite(runs["onset"]))
    readout = loop.posture_readout
    rows, simulated, predicted = [], [], []
    for onset in np.unique(runs["onset"][pulsed]):
        group = [int(i) for i in pulsed if runs["onset"][i] == onset]
        begin = round(onset / dt)
        forces = []
        for i in group:
            pulse = Pulse(float(onset), float(runs["angle_deg"][i]), float(runs["force_n"][i]), duration)
            push = make_disturbance(pulse.disturbance(), start_hand, setup.target, dt=h)
            assert push is not None
            forces.append(simulated_forces(push, warmup, h, (warmup_steps + begin + steps) * robot.substeps))
        deviation = np.zeros((len(group), robot.dimension))
        q_path, r_path = [np.zeros((len(group), 2))], [np.zeros((len(group), 2))]
        for k in range(begin, begin + steps):
            z, jacobian = states[k], robot.jacobian(states[k])
            first = (k + warmup_steps) * robot.substeps
            for g, force in enumerate(forces):
                push = force[first : first + robot.substeps]
                drive = np.zeros(robot.dimension)
                if np.any(push):
                    scale = 1e-6 / np.abs(push).max()
                    drive = (robot.advance(z, scale * push) - robot.advance(z, -scale * push)) / (2 * scale)
                deviation[g] = jacobian @ deviation[g] + drive
            q_path.append(deviation[:, n : n + 2].copy())
            r_path.append(deviation[:, :n] @ readout.T)
        q_pred, r_pred = np.stack(q_path, axis=1), np.stack(r_path, axis=1)
        span = slice(begin, begin + steps + 1)
        for g, i in enumerate(group):
            q_sim = (runs["q"][i, span] - runs["q"][nominal, span]).astype(np.float64)
            r_sim = (runs["q_ref"][i, span] - runs["q_ref"][nominal, span]).astype(np.float64)
            row: dict[str, Any] = {
                "onset": float(onset),
                "angle_deg": float(runs["angle_deg"][i]),
                "force_n": float(runs["force_n"][i]),
            }
            for name, sim, pred in (("q", q_sim, q_pred[g]), ("reference", r_sim, r_pred[g])):
                size, error = rms(sim), rms(pred - sim)
                row |= {
                    f"{name}_deviation_deg": float(np.degrees(size)),
                    f"{name}_error_deg": float(np.degrees(error)),
                    f"{name}_relative_error": float(error / size),
                }
            rows.append(row)
            simulated.append(np.stack([q_sim, r_sim]))
            predicted.append(np.stack([q_pred[g], r_pred[g]]))
    arrays = {
        "times": dt * np.arange(steps + 1),
        "onset": np.array([row["onset"] for row in rows]),
        "angle_deg": np.array([row["angle_deg"] for row in rows]),
        "force_n": np.array([row["force_n"] for row in rows]),
        "simulated": np.array(simulated),  # (pulse, [joint angles, reference], step, joint), rad
        "predicted": np.array(predicted),
    }
    return rows, arrays


def rms(deviation: NDArray[np.float64]) -> float:
    """The root mean square, over the steps, of a deviation's size (rad)."""
    return float(np.sqrt(np.mean(np.sum(deviation**2, axis=1))))


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` to the CSV file ``path``, one column per key."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
