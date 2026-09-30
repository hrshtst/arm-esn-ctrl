# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Train an ESN on one demonstration and run it autonomously (Stage 1).

    uv run python experiments/autonomous_esn.py configs/esn/autonomous_tvs_demo07.toml

The ESN is trained by teacher forcing on one demonstration. It then runs
autonomously, its output fed back as its next input, from the demonstration's
start posture and from slightly disturbed ones. Each run is compared with the
demonstrator's own reach from the same start posture, simulated with the
controller that made the demonstration. The run directory receives:

- ``esn_00.sklog.npz``, ...: the ESN's trajectories, one per start posture;
- ``demonstrator_00.sklog.npz``, ...: the demonstrator's reaches from the same postures;
- ``metrics.csv``: how far each ESN trajectory is from the demonstrator's;
- ``autonomous.png``: hand paths, joint angles, and hand speeds of both.

Replay a trajectory with ``uv run python third_party/skelarm/tools/player.py <file>``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from skelarm import StateLog, Task

from arm_esn_ctrl.demonstrations import (
    endpoint_positions,
    joint_trajectory_log,
    resample_joint_angles,
    simulate_reaches,
)
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.metrics import hand_speed, path_distance, reach_metrics
from arm_esn_ctrl.storage import start_run, storage_root

ESN_COLOR = "#2a78d6"
DEMONSTRATOR_COLOR = "#52514e"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"

# One start posture: joint angles of the ESN and the demonstrator, then their hand positions.
Run = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    esn_config = EsnConfig(**config["esn"])
    evaluation = config["evaluation"]
    demo_dir = storage_root() / config["demonstrations"]["run"]
    with (demo_dir / "config.toml").open("rb") as f:
        demo_config = tomllib.load(f)

    # Train on one demonstration.
    demo_log = StateLog.load(demo_dir / config["demonstrations"]["train"])
    _, q_demo = resample_joint_angles(demo_log, esn_config.dt)
    esn = ReachingEsn(esn_config)
    esn.fit(q_demo)
    one_step_error = rms_degrees(esn.one_step_predictions(q_demo) - q_demo[1:])
    print(f"Trained on {len(q_demo)} samples; one-step prediction error {one_step_error:.4f} deg RMS")

    # Run autonomously from each start posture, and let the demonstrator reach from the same postures.
    n_steps = round(evaluation["duration"] / esn_config.dt)
    times = esn_config.dt * np.arange(n_steps + 1)
    starts = [q_demo[0] + np.radians(offset) for offset in evaluation["start_offsets_deg"]]
    demonstrator_logs = simulate_reaches(demonstrator_config(demo_config, starts, evaluation["duration"]))
    skeleton = demo_log.build_skeleton()
    target = Task.from_dict(demo_config["task"]).require_target()

    runs: list[Run] = []
    rows = []
    for i, (start, offset, demonstrator_log) in enumerate(
        zip(starts, evaluation["start_offsets_deg"], demonstrator_logs, strict=True)
    ):
        q_esn = esn.generate(start, n_steps)
        _, q_ref = resample_joint_angles(demonstrator_log, esn_config.dt)
        hand_esn = endpoint_positions(skeleton, q_esn)
        hand_ref = endpoint_positions(skeleton, q_ref)
        joint_trajectory_log(skeleton, times, q_esn, demo_config["task"], producer="autonomous ESN").save(
            run_dir / f"esn_{i:02d}.sklog.npz"
        )
        demonstrator_log.save(run_dir / f"demonstrator_{i:02d}.sklog.npz")
        shape = reach_metrics(times, hand_esn, target)
        rows.append(
            {
                "start": i,
                "offset_q1_deg": offset[0],
                "offset_q2_deg": offset[1],
                "joint_rms_error_deg": rms_degrees(q_esn - q_ref),
                "path_distance_m": path_distance(hand_esn, hand_ref),
                "final_error_m": float(np.linalg.norm(hand_esn[-1] - target)),
                "peak_timing": shape["peak_timing"],
                "speed_profile_error": shape["speed_profile_error"],
            }
        )
        runs.append((q_esn, q_ref, hand_esn, hand_ref))

    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_metrics(rows)

    title = f"Autonomous ESN: {args.config.stem}"
    plot_runs(times, runs, target, title).savefig(run_dir / "autonomous.png", dpi=150)
    print(f"\nWrote the results to {run_dir}")
    print("Replay with:\n  uv run python third_party/skelarm/tools/player.py " + str(run_dir / "esn_00.sklog.npz"))


def demonstrator_config(demo_config: dict[str, Any], starts: list[np.ndarray], duration: float) -> dict[str, Any]:
    """The demonstration configuration, changed to reach from ``starts`` for ``duration`` seconds."""
    reference = copy.deepcopy(demo_config)
    reference["demonstrations"]["start_q"] = [np.degrees(start).tolist() for start in starts]
    reference["task"]["duration"] = duration
    return reference


def rms_degrees(error: np.ndarray) -> float:
    """Root mean square of an angle error given in radians, in degrees."""
    return float(np.degrees(np.sqrt(np.mean(error**2))))


def print_metrics(rows: list[dict[str, float]]) -> None:
    """Print the metrics as a table, one start posture per line."""
    print("start  offset (deg)    joint error  path distance  final error  peak    profile")
    print("                       (deg RMS)    (mm)           (mm)         timing  error")
    for r in rows:
        print(
            f"{int(r['start']):5d}  {r['offset_q1_deg']:+5.1f}, {r['offset_q2_deg']:+5.1f}"
            f"  {r['joint_rms_error_deg']:11.2f}"
            f"  {1000 * r['path_distance_m']:13.1f}  {1000 * r['final_error_m']:11.1f}"
            f"  {r['peak_timing']:6.2f}  {r['speed_profile_error']:7.2f}"
        )


def plot_runs(times: np.ndarray, runs: list[Run], target: np.ndarray, title: str) -> Figure:
    """Plot hand paths, joint angles, and hand speeds of the ESN against the demonstrator.

    Each run holds the joint angles of the ESN and the demonstrator, then their hand positions.
    """
    fig = Figure(figsize=(10, 8.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color="#0b0b0b")
    ax_hand, ax_speed, ax_q1, ax_q2 = fig.subplots(2, 2).flat
    for ax in (ax_hand, ax_speed, ax_q1, ax_q2):
        ax.set_facecolor(SURFACE_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.8)
        ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

    esn_line = {"color": ESN_COLOR, "linewidth": 1.5, "solid_capstyle": "round"}
    ref_line = {"color": DEMONSTRATOR_COLOR, "linewidth": 1.2, "linestyle": "--"}
    for i, (q_esn, q_ref, hand_esn, hand_ref) in enumerate(runs):
        esn_label = "ESN (autonomous)" if i == 0 else None
        ref_label = "demonstrator" if i == 0 else None
        ax_hand.plot(hand_ref[:, 0], hand_ref[:, 1], label=ref_label, **ref_line)
        ax_hand.plot(hand_esn[:, 0], hand_esn[:, 1], label=esn_label, **esn_line)
        ax_hand.plot(*hand_esn[0], marker="o", markersize=5, color=ESN_COLOR, markeredgecolor=SURFACE_COLOR)
        ax_speed.plot(times[: len(hand_ref)], hand_speed(times[: len(hand_ref)], hand_ref), **ref_line)
        ax_speed.plot(times, hand_speed(times, hand_esn), **esn_line)
        for ax, j in ((ax_q1, 0), (ax_q2, 1)):
            ax.plot(times[: len(q_ref)], np.degrees(q_ref[:, j]), **ref_line)
            ax.plot(times, np.degrees(q_esn[:, j]), **esn_line)

    ax_hand.plot(*target, marker="+", markersize=12, color="#0b0b0b", markeredgewidth=1.5)
    ax_hand.set(title="Hand paths", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax_hand.legend(frameon=False, labelcolor=TEXT_COLOR, loc="upper right")
    ax_speed.set(title="Hand speed", xlabel="time (s)", ylabel="speed (m/s)")
    ax_q1.set(title="Joint 1", xlabel="time (s)", ylabel="angle (deg)")
    ax_q2.set(title="Joint 2", xlabel="time (s)", ylabel="angle (deg)")
    return fig


if __name__ == "__main__":
    main()
