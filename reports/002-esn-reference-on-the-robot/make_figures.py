# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figure, tables, and animations of report 002.

    uv run python reports/002-esn-reference-on-the-robot/make_figures.py
    uv run python reports/002-esn-reference-on-the-robot/make_figures.py --animations

The summary compares the three arms (the ESN, the replayed demonstration, and the
demonstrator) across the five scenarios, at one representative gain of each
tracking law. It reads the ``metrics.csv`` of each run under ``results/``, writes
``results/summary/summary.png``, and prints the tables of the report.

With ``--animations``, it also exports animated GIFs of a few runs with skelarm's
player. Their logs are not kept in Git, so this reads them from the runs under
the storage root, where ``experiments/robot_esn.py`` wrote them.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
SCENARIOS = {  # label: run directory
    "nominal": "20261005-120454-nominal",
    "push": "20261005-120457-push",
    "block": "20261005-120459-block",
    "offset 3°": "20261005-120501-offset_3deg",
    "offset 10°": "20261005-120503-offset_10deg",
}
SETTINGS = [("computed_torque", 10.0), ("pd", 20.0)]  # the representative gains: law and omega (rad/s)
LAW_NAMES = {"computed_torque": "computed torque", "pd": "joint PD"}
ARMS = ("esn", "replay", "demonstrator")
ARM_LABELS = {"esn": "ESN", "replay": "replay", "demonstrator": "demonstrator"}
ARM_COLORS = {"esn": "#2a78d6", "replay": "#eb6834", "demonstrator": "#52514e"}  # as in experiments/robot_esn.py
ARM_MARKERS = {"esn": "o", "replay": "s", "demonstrator": "D"}
METRICS = [  # column, title, factor to the display unit, log scale
    ("reach_path_distance_m", "Path distance from the demonstrator's\nundisturbed reach (mm)", 1000.0, True),
    ("success", "Arrive and hold\n(% of start postures)", 100.0, False),
    ("peak_torque_nm", "Peak joint torque (N m)", 1.0, True),
    ("effort_n2m2s", "Integral of squared torque (N² m² s)", 1.0, True),
]
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
# The animations: (scenario, arm, start index) at computed torque, omega = 10 rad/s.
ANIMATIONS = [("block", "esn", 0), ("block", "replay", 0), ("block", "demonstrator", 0)]
ANIMATIONS += [("offset 10°", "esn", 0), ("offset 10°", "replay", 0)]
ANIMATION_FPS = 20.0  # a whole number of milliseconds per frame, so GIFs play in real time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--animations", action="store_true", help="also export GIFs from the logs in the storage root")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    means = {
        (scenario, law, omega, arm): mean_metrics(scenario, law, omega, arm)
        for scenario in SCENARIOS
        for law, omega in SETTINGS
        for arm in ARMS
    }
    plot_summary(means).savefig(SUMMARY / "summary.png", dpi=150)
    print_tables(means)
    print(f"Wrote {SUMMARY / 'summary.png'}")
    if args.animations:
        export_animations()


def mean_metrics(scenario: str, law: str, omega: float, arm: str) -> dict[str, float]:
    """The metrics of one arm in one scenario averaged over the start postures (the demonstrator has no gains)."""
    with (RESULTS / SCENARIOS[scenario] / "metrics.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["arm"] == arm]
    if arm != "demonstrator":
        rows = [r for r in rows if r["law"] == law and float(r["omega"]) == omega]
    means = {}
    for column, *_ in METRICS:
        values = [float(r[column] == "True") if column == "success" else float(r[column]) for r in rows]
        means[column] = float(np.mean(values))
    means["starts"] = len(rows)
    return means


def plot_summary(means: dict[tuple[str, str, float, str], dict[str, float]]) -> Figure:
    """One row per tracking law and one column per metric; each scenario shows the three arms side by side."""
    fig = Figure(figsize=(14, 7.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("The three arms across the scenarios: means over the start postures", color="#0b0b0b")
    axes = fig.subplots(len(SETTINGS), len(METRICS), squeeze=False)
    positions = np.arange(len(SCENARIOS))
    offsets = {"esn": -0.22, "replay": 0.0, "demonstrator": 0.22}
    for (law, omega), row in zip(SETTINGS, axes, strict=True):
        for (column, title, factor, log), ax in zip(METRICS, row, strict=True):
            ax.set_facecolor(SURFACE_COLOR)
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
            ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
            for spine in ax.spines.values():
                spine.set_color(GRID_COLOR)
            for arm in ARMS:
                values = [factor * means[(scenario, law, omega, arm)][column] for scenario in SCENARIOS]
                ax.plot(
                    positions + offsets[arm],
                    values,
                    linestyle="none",
                    marker=ARM_MARKERS[arm],
                    markersize=8,
                    color=ARM_COLORS[arm],
                    markeredgecolor=SURFACE_COLOR,
                    markeredgewidth=1.5,
                    label=ARM_LABELS[arm],
                )
            if log:
                ax.set_yscale("log")
            else:
                ax.set_ylim(-5, 105)
            ax.set_xticks(positions, list(SCENARIOS), rotation=30, ha="right")
            ax.set_title(title, color=TEXT_COLOR, fontsize=10)
        row[0].set_ylabel(f"{LAW_NAMES[law]}, ω = {omega:g} rad/s", color="#0b0b0b", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def print_tables(means: dict[tuple[str, str, float, str], dict[str, float]]) -> None:
    """Print one Markdown table per tracking law: each cell gives the ESN / replay / demonstrator."""
    for law, omega in SETTINGS:
        print(f"\n{LAW_NAMES[law]}, ω = {omega:g} rad/s (ESN / replay / demonstrator)\n")
        print("| Scenario | Starts | Path distance (mm) | Arrive and hold (%) | Peak torque (N m) | ∫τ² (N² m² s) |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for scenario in SCENARIOS:
            cells = []
            for column, _, factor, _ in METRICS:
                values = [factor * means[(scenario, law, omega, arm)][column] for arm in ARMS]
                digits = 0 if column == "success" or max(values) >= 100 else 1
                cells.append(" / ".join(f"{value:.{digits}f}" for value in values))
            starts = int(means[(scenario, law, omega, "esn")]["starts"])
            print(f"| {scenario} | {starts} | " + " | ".join(cells) + " |")


def export_animations() -> None:
    """Export GIFs of a few runs with skelarm's player, from the logs under the storage root."""
    from arm_esn_ctrl.storage import REPO_ROOT, storage_root

    player = REPO_ROOT / "third_party" / "skelarm" / "tools" / "player.py"
    for scenario, arm, start in ANIMATIONS:
        run = storage_root() / "results" / SCENARIOS[scenario]
        log = (
            run / f"{arm}_{start:02d}.sklog.npz"
            if arm == "demonstrator"
            else run / "computed_torque_w10" / f"{arm}_{start:02d}.sklog.npz"
        )
        gif = SUMMARY / f"{SCENARIOS[scenario].split('-', 2)[2]}_{arm}_{start:02d}.gif"
        subprocess.run(
            [sys.executable, str(player), str(log), "--export", str(gif), "--fps", f"{ANIMATION_FPS:g}"],
            check=True,
            env=os.environ | {"QT_QPA_PLATFORM": "offscreen"},  # render without a window
        )


if __name__ == "__main__":
    main()
