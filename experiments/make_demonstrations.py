# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Generate scripted reaching demonstrations and check how human-like they are.

    uv run python experiments/make_demonstrations.py experiments/demonstrations/reach_pds.toml

Runs the configured skelarm reaching controller once from every start posture
and writes into a new run directory:

- ``demo_00.sklog.npz``, ``demo_01.sklog.npz``, ...: the demonstrations. Replay
  one with ``uv run python third_party/skelarm/tools/player.py <file>``.
- ``metrics.csv``: the reach metrics of every demonstration (see
  :mod:`arm_esn_ctrl.metrics`).
- ``demonstrations.png``: hand paths, joint-space paths, and hand speed profiles.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from skelarm import StateLog, Task

from arm_esn_ctrl.demonstrations import endpoint_positions, simulate_reaches
from arm_esn_ctrl.metrics import hand_speed, minimum_jerk_profile, reach_metrics, speed_profile
from arm_esn_ctrl.storage import REPO_ROOT, start_run

# Colors: every demonstration is one series; the minimum-jerk model is a neutral reference.
DEMO_COLOR = "#2a78d6"
REFERENCE_COLOR = "#52514e"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="demonstration configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    target = Task.from_dict(config["task"]).require_target()
    logs = simulate_reaches(config)

    rows = []
    for i, log in enumerate(logs):
        log.save(run_dir / f"demo_{i:02d}.sklog.npz")
        hand = endpoint_positions(log.build_skeleton(), log.channel("q"))
        start_q1, start_q2 = config["demonstrations"]["start_q"][i]
        rows.append(
            {"demo": i, "start_q1_deg": start_q1, "start_q2_deg": start_q2} | reach_metrics(log.times, hand, target)
        )

    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_metrics(rows)

    plot_demonstrations(logs, target, title=args.config.stem).savefig(run_dir / "demonstrations.png", dpi=150)

    print(f"\nWrote {len(logs)} demonstrations to {run_dir}")
    player = (REPO_ROOT / "third_party/skelarm/tools/player.py").relative_to(REPO_ROOT)
    print(f"Replay one with:\n  uv run python {player} {run_dir / 'demo_00.sklog.npz'}")


def print_metrics(rows: list[dict[str, float]]) -> None:
    """Print the reach metrics as a table, one demonstration per line."""
    print("demo  movement  peak     peak    profile  peaks  path       overshoot  final")
    print("      time (s)  speed    timing  error           deviation             error (cm)")
    for r in rows:
        print(
            f"{int(r['demo']):4d}  {r['movement_time']:8.2f}  {r['peak_speed']:5.2f}    {r['peak_timing']:6.2f}"
            f"  {r['speed_profile_error']:7.2f}  {int(r['speed_peaks']):5d}  {r['path_deviation']:9.1%}"
            f"  {r['overshoot']:9.1%}  {100 * r['final_error']:10.2f}"
        )


def plot_demonstrations(logs: list[StateLog], target: np.ndarray, title: str) -> Figure:
    """Plot hand paths, joint-space paths, hand speeds, and speed profiles against minimum jerk."""
    fig = Figure(figsize=(10, 8.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"Demonstrations: {title}", color="#0b0b0b")
    ax_hand, ax_joint, ax_speed, ax_profile = fig.subplots(2, 2).flat
    for ax in (ax_hand, ax_joint, ax_speed, ax_profile):
        ax.set_facecolor(SURFACE_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.8)
        ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

    line = {"color": DEMO_COLOR, "linewidth": 1.5, "solid_capstyle": "round"}
    start_marker = {"color": DEMO_COLOR, "marker": "o", "markersize": 6, "markeredgecolor": SURFACE_COLOR}
    for i, log in enumerate(logs):
        t = log.times
        q = log.channel("q")
        hand = endpoint_positions(log.build_skeleton(), q)
        speed = hand_speed(t, hand)
        q_deg = np.degrees(q)

        ax_hand.plot(hand[:, 0], hand[:, 1], **line)
        ax_hand.plot(*hand[0], **start_marker)
        ax_hand.annotate(str(i), hand[0], xytext=(5, 5), textcoords="offset points", color=TEXT_COLOR, fontsize=8)
        ax_joint.plot(q_deg[:, 0], q_deg[:, 1], **line)
        ax_joint.plot(*q_deg[0], **start_marker)
        ax_joint.annotate(str(i), q_deg[0], xytext=(5, 5), textcoords="offset points", color=TEXT_COLOR, fontsize=8)
        ax_speed.plot(t, speed, **line)
        ax_profile.plot(*speed_profile(t, speed), label="demonstrations" if i == 0 else None, **line)

    ax_hand.plot(*target, marker="+", markersize=12, color="#0b0b0b", markeredgewidth=1.5)
    ax_hand.annotate(
        "target",
        target,
        xytext=(8, -14),
        textcoords="offset points",
        color=TEXT_COLOR,
        fontsize=8,
        bbox={"facecolor": SURFACE_COLOR, "edgecolor": "none", "pad": 1.0},
    )
    ax_hand.set(title="Hand paths", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax_joint.set(title="Joint-space paths", xlabel="joint 1 (deg)", ylabel="joint 2 (deg)", aspect="equal")
    ax_speed.set(title="Hand speed", xlabel="time (s)", ylabel="speed (m/s)")

    tau = np.linspace(0.0, 1.0, 200)
    reference = {"color": REFERENCE_COLOR, "linewidth": 1.5, "linestyle": "--"}
    ax_profile.plot(tau, minimum_jerk_profile(tau), label="minimum jerk", **reference)
    ax_profile.set(
        title="Speed profile against the minimum-jerk model",
        xlabel="normalized movement time",
        ylabel="speed / peak speed",
    )
    ax_profile.legend(frameon=False, labelcolor=TEXT_COLOR)
    return fig


if __name__ == "__main__":
    main()
