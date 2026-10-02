# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Train an ESN on demonstrations and run it autonomously (Stage 1).

    uv run python experiments/autonomous_esn.py configs/esn/autonomous_tvs_all.toml

The ESN is trained by teacher forcing on one or more demonstrations. It then
runs autonomously, its output fed back as its next input, from start postures
of two kinds:

- each training demonstration's start posture, plus the configured offsets
  (an offset of zero tests replication; others, slightly disturbed starts);
- further start postures that no demonstration starts from (``extra_start_q_deg``),
  which test whether the ESN generalizes outside the demonstrated trajectories.

Each run is compared with the demonstrator's own reach from the same start
posture, simulated with the controller that made the demonstrations. The run
directory receives:

- ``esn_00.sklog.npz``, ...: the ESN's trajectories, one per start posture;
- ``demonstrator_00.sklog.npz``, ...: the demonstrator's reaches from the same postures;
- ``metrics.csv``: for each start posture, the reach compared with the demonstrator's
  and the hold at the target (see :func:`arm_esn_ctrl.autonomous.run_metrics`);
- ``autonomous.png``: hand paths, joint angles, and hand speeds of both, over the
  training demonstrations.

Replay a trajectory with ``uv run python third_party/skelarm/tools/player.py <file>``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from arm_esn_ctrl.autonomous import Run, Start, load_setup, rms_degrees, run_autonomously, run_metrics
from arm_esn_ctrl.demonstrations import endpoint_positions, joint_trajectory_log
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.metrics import hand_speed
from arm_esn_ctrl.storage import start_run

ESN_COLOR = "#2a78d6"
TRAINING_COLOR = "#eb6834"
DEMONSTRATOR_COLOR = "#52514e"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    setup = load_setup(config)

    # Train on the demonstrations.
    esn = ReachingEsn(EsnConfig(**config["esn"]))
    esn.fit(list(setup.demos.values()))
    one_step_error = rms_degrees(np.vstack([esn.one_step_predictions(q) - q[1:] for q in setup.demos.values()]))
    n_samples = sum(len(q) for q in setup.demos.values())
    print(f"Trained on {len(setup.demos)} demonstrations ({n_samples} samples)")
    print(f"One-step prediction error {one_step_error:.4f} deg RMS")

    # Run autonomously from each start posture and compare with the demonstrator's reach from it.
    runs = run_autonomously(esn, setup)
    rows = []
    for i, (run, demonstrator_log) in enumerate(zip(runs, setup.demonstrator_logs, strict=True)):
        joint_trajectory_log(setup.skeleton, setup.times, run.q_esn, setup.task, producer="autonomous ESN").save(
            run_dir / f"esn_{i:02d}.sklog.npz"
        )
        demonstrator_log.save(run_dir / f"demonstrator_{i:02d}.sklog.npz")
        rows.append(
            {
                "start": i,
                "origin": run.start.origin,
                "start_q1_deg": float(np.degrees(run.start.q[0])),
                "start_q2_deg": float(np.degrees(run.start.q[1])),
            }
            | run_metrics(run, setup)
        )

    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_metrics(rows, setup.starts)

    title = f"Autonomous ESN: {args.config.stem}"
    training = [(q, endpoint_positions(setup.skeleton, q)) for q in setup.demos.values()]
    plot_runs(setup.times, runs, training, setup.target, title).savefig(run_dir / "autonomous.png", dpi=150)
    print(f"\nWrote the results to {run_dir}")
    print("Replay with:\n  uv run python third_party/skelarm/tools/player.py " + str(run_dir / "esn_00.sklog.npz"))


def print_metrics(rows: list[dict[str, Any]], starts: list[Start]) -> None:
    """Print the metrics as a table, one start posture per line, then a summary by kind of start."""
    print("                             ------------------ reach ------------------   ----------- hold -----------")
    print("start  origin                first step  path dist.  joint error  arrival   left  hold error  observed")
    print("                             (mm)        (mm)        (deg RMS)    delay (s) goal  (mm)        (s)")
    for r in rows:
        left = "-" if not r["arrived"] else ("yes" if r["left_goal"] else "no")
        print(
            f"{r['start']:5d}  {r['origin']:<20}  {1000 * r['first_step_m']:10.1f}"
            f"  {1000 * r['reach_path_distance_m']:10.1f}  {r['reach_joint_error_deg']:11.2f}"
            f"  {r['arrival_delay_s']:+9.2f}"
            f"  {left:>4}  {1000 * r['hold_error_m']:10.1f}  {r['hold_observed_s']:8.2f}"
        )
    for kind, demonstrated in (("demonstrated starts", True), ("other starts", False)):
        group = [r for r, s in zip(rows, starts, strict=True) if s.demonstrated == demonstrated]
        if group:
            first_step = 1000 * np.mean([r["first_step_m"] for r in group])
            distance = 1000 * np.mean([r["reach_path_distance_m"] for r in group])
            joint_error = np.mean([r["reach_joint_error_deg"] for r in group])
            successes = sum(r["success"] for r in group)
            hold_error = 1000 * np.nanmedian([r["hold_error_m"] for r in group])
            print(
                f"{len(group)} {kind}: mean first step {first_step:.1f} mm, mean reach path distance {distance:.1f} mm,"
                f" mean reach joint error {joint_error:.2f} deg; {successes} of {len(group)} arrive and stay,"
                f" median hold error {hold_error:.1f} mm"
            )


def plot_runs(
    times: np.ndarray,
    runs: list[Run],
    training: list[tuple[np.ndarray, np.ndarray]],
    target: np.ndarray,
    title: str,
) -> Figure:
    """Plot hand paths, joint angles, and hand speeds of the ESN against the demonstrator.

    ``training`` holds the joint angles and hand positions of each training
    demonstration, which are highlighted underneath the runs. Filled markers show
    where a training demonstration starts; hollow ones, start postures it does not.
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
    training_line = {"color": TRAINING_COLOR, "linewidth": 4.5, "solid_capstyle": "round", "zorder": 1}

    for i, (q_train, hand_train) in enumerate(training):
        t_train = times[1] * np.arange(len(q_train))
        label = "training demonstrations" if i == 0 else None
        ax_hand.plot(hand_train[:, 0], hand_train[:, 1], label=label, **training_line)
        ax_speed.plot(t_train, hand_speed(t_train, hand_train), **training_line)
        for ax, j in ((ax_q1, 0), (ax_q2, 1)):
            ax.plot(t_train, np.degrees(q_train[:, j]), **training_line)
    for i, run in enumerate(runs):
        esn_label = "ESN, run autonomously from each start" if i == 0 else None
        ref_label = "demonstrator, reaching from each start" if i == 0 else None
        ax_hand.plot(run.hand_ref[:, 0], run.hand_ref[:, 1], label=ref_label, **ref_line)
        ax_hand.plot(run.hand_esn[:, 0], run.hand_esn[:, 1], label=esn_label, **esn_line)
        face = ESN_COLOR if run.start.demonstrated else SURFACE_COLOR
        ax_hand.plot(*run.hand_esn[0], marker="o", markersize=6, color=ESN_COLOR, markerfacecolor=face)
        t_ref = times[: len(run.hand_ref)]
        ax_speed.plot(t_ref, hand_speed(t_ref, run.hand_ref), **ref_line)
        ax_speed.plot(times, hand_speed(times, run.hand_esn), **esn_line)
        for ax, j in ((ax_q1, 0), (ax_q2, 1)):
            ax.plot(times[: len(run.q_ref)], np.degrees(run.q_ref[:, j]), **ref_line)
            ax.plot(times, np.degrees(run.q_esn[:, j]), **esn_line)

    ax_hand.plot(*target, marker="+", markersize=12, color="#0b0b0b", markeredgewidth=1.5)
    ax_hand.set(title="Hand paths", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    handles, labels = ax_hand.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    # A jump in the ESN's first step would squash the reach speeds, so the axis stops above them.
    reach_speed = max(hand_speed(times[: len(run.hand_ref)], run.hand_ref).max() for run in runs)
    first_step_speed = max(np.linalg.norm(run.hand_esn[1] - run.hand_esn[0]) / times[1] for run in runs)
    ax_speed.set(title="Hand speed", xlabel="time (s)", ylabel="speed (m/s)", ylim=(0.0, 1.6 * reach_speed))
    if first_step_speed > 1.6 * reach_speed:
        ax_speed.annotate(
            f"ESN first steps reach {first_step_speed:.1f} m/s (off the scale)",
            (0.98, 0.96),
            xycoords="axes fraction",
            ha="right",
            va="top",
            color=TEXT_COLOR,
            fontsize=8,
        )
    ax_q1.set(title="Joint 1", xlabel="time (s)", ylabel="angle (deg)")
    ax_q2.set(title="Joint 2", xlabel="time (s)", ylabel="angle (deg)")
    return fig


if __name__ == "__main__":
    main()
