# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figure, tables, and animations of report 002.

    uv run python reports/002-esn-reference-on-the-robot/make_figures.py
    uv run python reports/002-esn-reference-on-the-robot/make_figures.py --animations

The summary compares the three arms (the ESN, the replayed demonstration, and the
demonstrator) across the five scenarios, at one representative gain of each
tracking law. The damping comparison follows the ESN and the replay as the
tracker's damping ratio falls, in the nominal, push, and block scenarios. Both
read the ``metrics.csv`` of the runs under ``results/``, write
``results/summary/summary.png`` and ``results/summary/damping.png``, and print the
tables of the report.

Two options read the run logs, which are not kept in Git, from the runs under
the storage root, where ``experiments/robot_esn.py`` wrote them:

- ``--traces`` compares the ESN's output with the arm's joint angles and the
  demonstrator's trajectory: it draws ``results/summary/traces.png`` (critically
  damped tracker) and ``results/summary/traces_underdamped.png`` (damping ratio
  0.1), and writes how far the output departs from the demonstration in every run
  to ``departures.csv``, ``offset_references.csv``, and ``damping_departures.csv``,
  and how the output rings with an underdamped arm to ``ringing.csv``, all in
  ``results/summary``, from which the tables of that comparison are printed
  (also without the option);
- ``--animations`` exports animated GIFs of a few runs with skelarm's player.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from skelarm import StateLog

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
# The damping comparison: one natural frequency, and damping ratios from critical down.
DAMPING_SCENARIOS = {  # label: run directory
    "nominal": "20261005-160737-nominal_damping",
    "push": "20261005-160739-push_damping",
    "block": "20261005-160741-block_damping",
}
DAMPINGS = [1.0, 0.5, 0.3, 0.1]
LAW_STYLES = {"computed_torque": "-", "pd": "--"}
# The animations: GIF name, run directory, tracker setting (None for the demonstrator), arm, and start index.
ANIMATIONS = [
    ("block_esn_00", SCENARIOS["block"], "computed_torque_w10", "esn", 0),
    ("block_replay_00", SCENARIOS["block"], "computed_torque_w10", "replay", 0),
    ("block_demonstrator_00", SCENARIOS["block"], None, "demonstrator", 0),
    ("offset_10deg_esn_00", SCENARIOS["offset 10°"], "computed_torque_w10", "esn", 0),
    ("offset_10deg_replay_00", SCENARIOS["offset 10°"], "computed_torque_w10", "replay", 0),
    ("block_damping_pd_z0.3_esn_00", DAMPING_SCENARIOS["block"], "pd_w10_z0.3", "esn", 0),
    ("block_damping_pd_z0.3_replay_00", DAMPING_SCENARIOS["block"], "pd_w10_z0.3", "replay", 0),
]
ANIMATION_FPS = 20.0  # a whole number of milliseconds per frame, so GIFs play in real time
# The comparison of the ESN's output with the arm and the demonstrator.
TRACE_SCENARIOS = ["nominal", "push", "block", "offset 10°"]
TRACE_SETTING = "computed_torque_w10"  # the tracker setting of the trace figure, from the first start posture
UNDERDAMPED_SETTING = "computed_torque_w10_z0.1"  # the same, with the damping ratio 0.1 (the damping runs)
DEPARTURE_WINDOWS = {"nominal": (0.0, 1.5), "push": (0.4, 1.4), "block": (0.3, 1.3)}  # the reach, and 1 s from onset
OFFSET_WINDOW = (0.0, 1.5)  # the reach from an offset start
DEPARTURES = SUMMARY / "departures.csv"
OFFSET_REFERENCES = SUMMARY / "offset_references.csv"
DAMPING_DEPARTURES = SUMMARY / "damping_departures.csv"
RINGING = SUMMARY / "ringing.csv"
RINGING_WINDOW = (1.5, 5.0)  # after the reach, while an underdamped arm rings around the target (s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--animations", action="store_true", help="also export GIFs from the logs in the storage root")
    parser.add_argument(
        "--traces",
        action="store_true",
        help="also compare the ESN's output with the arm, from the logs in the storage root",
    )
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
    stats = {
        (scenario, law, damping, arm): damping_stats(scenario, law, damping, arm)
        for scenario in DAMPING_SCENARIOS
        for law in LAW_NAMES
        for damping in DAMPINGS
        for arm in ("esn", "replay")
    }
    plot_damping(stats).savefig(SUMMARY / "damping.png", dpi=150)
    print_damping_tables(stats)
    print(f"Wrote {SUMMARY / 'summary.png'} and {SUMMARY / 'damping.png'}")
    if args.traces:
        title = "computed torque, ω = 10 rad/s"
        plot_traces(SCENARIOS, TRACE_SCENARIOS, TRACE_SETTING, title, 3.0).savefig(SUMMARY / "traces.png", dpi=150)
        underdamped = plot_traces(
            DAMPING_SCENARIOS, list(DAMPING_SCENARIOS), UNDERDAMPED_SETTING, f"{title}, ζ = 0.1", 5.0
        )
        underdamped.savefig(SUMMARY / "traces_underdamped.png", dpi=150)
        write_departures()
        print(f"Wrote the traces, {DEPARTURES.name}, {OFFSET_REFERENCES.name}, and {DAMPING_DEPARTURES.name}")
    print_departure_tables()
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


def damping_stats(scenario: str, law: str, damping: float, arm: str) -> dict[str, float]:
    """How the runs of one arm settle at one damping ratio, over the start postures.

    The share that arrives and holds, the share that has settled by the end of the
    run (it stays within the goal radius from some time on), the mean settling
    time of those, and the mean final distance to the target.
    """
    with (RESULTS / DAMPING_SCENARIOS[scenario] / "metrics.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["arm"] == arm and r["law"] == law and float(r["damping"]) == damping]
    settling = np.array([float(r["settling_time_s"]) for r in rows])
    settled = np.isfinite(settling)
    return {
        "hold": 100.0 * np.mean([r["success"] == "True" for r in rows]),
        "settled": 100.0 * float(np.mean(settled)),
        "settling_time": float(settling[settled].mean()) if settled.any() else float("nan"),
        "final_distance": 1000.0 * float(np.mean([float(r["final_distance_m"]) for r in rows])),
    }


def plot_damping(stats: dict[tuple[str, str, float, str], dict[str, float]]) -> Figure:
    """One column per scenario, one row per measure of settling, against the damping ratio (falling to the right)."""
    measures = [
        ("hold", "Arrive and hold (% of start postures)"),
        ("settled", "Settled by the end of the run\n(% of start postures)"),
        ("settling_time", "Settling time of the runs that settle (s)"),
    ]
    fig = Figure(figsize=(13, 9.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Lowering the tracker's damping ratio at ω = 10 rad/s: means over the 8 start postures", color="#0b0b0b"
    )
    axes = fig.subplots(len(measures), len(DAMPING_SCENARIOS), squeeze=False)
    for (measure, title), row in zip(measures, axes, strict=True):
        for scenario, ax in zip(DAMPING_SCENARIOS, row, strict=True):
            ax.set_facecolor(SURFACE_COLOR)
            ax.grid(color=GRID_COLOR, linewidth=0.8)
            ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
            for spine in ax.spines.values():
                spine.set_color(GRID_COLOR)
            for arm in ("esn", "replay"):
                for law in LAW_NAMES:
                    values = [stats[(scenario, law, damping, arm)][measure] for damping in DAMPINGS]
                    ax.plot(
                        DAMPINGS,
                        values,
                        linestyle=LAW_STYLES[law],
                        marker=ARM_MARKERS[arm],
                        markersize=6,
                        linewidth=1.8,
                        color=ARM_COLORS[arm],
                        label=f"{ARM_LABELS[arm]}, {LAW_NAMES[law]}",
                    )
            ax.set_xscale("log")
            ax.set_xticks(DAMPINGS, [f"{damping:g}" for damping in DAMPINGS])
            ax.minorticks_off()
            ax.set_xlim(1.25, 0.08)  # less damping to the right
            if measure != "settling_time":
                ax.set_ylim(-5, 105)
            ax.set_title(f"{scenario}: {title}", color=TEXT_COLOR, fontsize=10)
            ax.set_xlabel("damping ratio ζ", color=TEXT_COLOR, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def print_damping_tables(stats: dict[tuple[str, str, float, str], dict[str, float]]) -> None:
    """Print one Markdown table per tracking law: each cell gives the ESN / replay."""
    for law in LAW_NAMES:
        print(f"\n{LAW_NAMES[law]}, ω = 10 rad/s (ESN / replay)\n")
        columns = ["Hold, nominal (%)", "Hold, push (%)", "Hold, block (%)", "Settled, block (%)"]
        print("| ζ | " + " | ".join(columns) + " | Final distance, block (mm) |")
        print("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for damping in DAMPINGS:
            cells = [
                stats[("nominal", law, damping, "esn")]["hold"],
                stats[("nominal", law, damping, "replay")]["hold"],
                stats[("push", law, damping, "esn")]["hold"],
                stats[("push", law, damping, "replay")]["hold"],
                stats[("block", law, damping, "esn")]["hold"],
                stats[("block", law, damping, "replay")]["hold"],
                stats[("block", law, damping, "esn")]["settled"],
                stats[("block", law, damping, "replay")]["settled"],
            ]
            pairs = [f"{cells[i]:.0f} / {cells[i + 1]:.0f}" for i in range(0, len(cells), 2)]
            distance = [stats[("block", law, damping, arm)]["final_distance"] for arm in ("esn", "replay")]
            print(f"| {damping:g} | " + " | ".join(pairs) + f" | {distance[0]:.1f} / {distance[1]:.1f} |")


def joint_angles_at(log: StateLog, times: NDArray[np.float64], channel: str = "q") -> NDArray[np.float64]:
    """A log's joint angles (or reference ``q_ref``) interpolated at ``times``, in degrees."""
    values = log.channel(channel).reshape(len(log.times), -1)
    return np.degrees(np.column_stack([np.interp(times, log.times, values[:, j]) for j in range(values.shape[1])]))


def run_logs(scenario: str, setting: str, start: int, runs: dict[str, str] = SCENARIOS) -> dict[str, StateLog]:
    """The logs of one start posture of a scenario: the ESN's and the replay's, and the demonstrator's.

    ``runs`` maps the scenarios to their run directories. ``demonstration`` is the
    demonstrator's undisturbed reach from the same start posture: the nominal run's
    for the push and the block, the run's own otherwise.
    """
    from arm_esn_ctrl.storage import storage_root

    run = storage_root() / "results" / runs[scenario]
    logs = {
        "esn": StateLog.load(run / setting / f"esn_{start:02d}.sklog.npz"),
        "replay": StateLog.load(run / setting / f"replay_{start:02d}.sklog.npz"),
        "demonstrator": StateLog.load(run / f"demonstrator_{start:02d}.sklog.npz"),
    }
    undisturbed = run if scenario not in ("push", "block") else storage_root() / "results" / runs["nominal"]
    logs["demonstration"] = StateLog.load(undisturbed / f"demonstrator_{start:02d}.sklog.npz")
    return logs


def plot_traces(runs: dict[str, str], scenarios: list[str], setting: str, title: str, end: float) -> Figure:
    """The joint angles over time of the arms, the references, and the demonstrator, from the first start.

    One column per scenario, from 0 to ``end`` seconds; the bottom row shows how far
    the ESN's output and its arm are from the demonstration, and from each other.
    """
    fig = Figure(figsize=(4 * len(scenarios), 11), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        f"The ESN's output against the arm and the demonstrator: {title}, from the first start", color="#0b0b0b"
    )
    axes = fig.subplots(3, len(scenarios), sharex=True)
    for col, scenario in enumerate(scenarios):
        logs = run_logs(scenario, setting, 0, runs)
        esn = logs["esn"]
        times = esn.times[(esn.times > -1e-9) & (esn.times <= end)]
        demonstration = joint_angles_at(logs["demonstration"], times)
        output, arm = joint_angles_at(esn, times, "q_ref"), joint_angles_at(esn, times)
        lines = [
            (demonstration, "demonstrator, undisturbed, from this start", ARM_COLORS["demonstrator"], "-", 5.0, 0.3),
            (joint_angles_at(logs["replay"], times, "q_ref"), "replay: reference", ARM_COLORS["replay"], "--", 1.3, 1),
            (joint_angles_at(logs["replay"], times), "replay: arm", ARM_COLORS["replay"], "-", 1.3, 1.0),
            (output, "ESN: output (reference)", ARM_COLORS["esn"], "--", 1.6, 1.0),
            (arm, "ESN: arm", ARM_COLORS["esn"], "-", 1.6, 1.0),
        ]
        if scenario in ("push", "block"):
            disturbed = joint_angles_at(logs["demonstrator"], times)
            lines.insert(1, (disturbed, "demonstrator, disturbed", "#0b0b0b", "-.", 1.0, 1.0))
        for joint in range(2):
            ax = axes[joint, col]
            for values, label, color, style, width, alpha in lines:
                ax.plot(
                    times, values[:, joint], color=color, linestyle=style, linewidth=width, alpha=alpha, label=label
                )
            ax.set_title(f"{scenario}: joint {joint + 1} (deg)", color=TEXT_COLOR, fontsize=10)
        ax = axes[2, col]
        ax.plot(times, np.linalg.norm(output - demonstration, axis=1), color=ARM_COLORS["esn"], linewidth=1.6)
        ax.plot(
            times, np.linalg.norm(arm - demonstration, axis=1), color=ARM_COLORS["esn"], linestyle=":", linewidth=1.6
        )
        ax.plot(times, np.linalg.norm(output - arm, axis=1), color=ARM_COLORS["esn"], linestyle="-.", linewidth=1.0)
        ax.set_title(f"{scenario}: distances in joint space (deg)", color=TEXT_COLOR, fontsize=10)
        ax.set_xlabel("time (s)", color=TEXT_COLOR, fontsize=9)
        for ax in axes[:, col]:
            ax.set_facecolor(SURFACE_COLOR)
            ax.grid(color=GRID_COLOR, linewidth=0.8)
            ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
            for spine in ax.spines.values():
                spine.set_color(GRID_COLOR)
            if scenario in ("push", "block"):
                span = (0.4, 0.5) if scenario == "push" else (0.3, 0.8)
                ax.axvspan(*span, color="#f0efec", zorder=0)
    handles, labels = axes[0, scenarios.index("block")].get_legend_handles_labels()  # the block has every line
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    distance_labels = ["ESN output from the demonstration", "ESN arm from the demonstration", "ESN output from its arm"]
    axes[2, 0].legend(distance_labels, fontsize=8, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def write_departures() -> None:
    """Measure, in every run at the representative gains, how far the ESN's output departs from the demonstration.

    For the nominal, push, and block runs, the largest distance in joint space of
    the ESN's output and of its arm from the demonstrator's undisturbed reach,
    over the reach or 1 s from the disturbance's onset, and their ratio: 0 if the
    output replays the demonstration, 1 if it departs as far as the arm. For the
    offset runs, the RMS distance of the ESN's output over the reach from the
    demonstrator's reach from the actual start and from the replayed demonstration.
    """
    rows = []
    for scenario, (begin, end) in DEPARTURE_WINDOWS.items():
        for law, omega in SETTINGS:
            for start in range(8):
                logs = run_logs(scenario, f"{law}_w{omega:g}", start)
                rows.append(
                    {"scenario": scenario, "law": law, "omega": omega, "start": start} | departure(logs, begin, end)
                )
    write_csv(DEPARTURES, rows)
    rows = []
    for scenario in ("offset 3°", "offset 10°"):
        for law, omega in SETTINGS:
            for start in range(32):
                logs = run_logs(scenario, f"{law}_w{omega:g}", start)
                esn = logs["esn"]
                times = esn.times[(esn.times >= OFFSET_WINDOW[0] - 1e-9) & (esn.times <= OFFSET_WINDOW[1] + 1e-9)]
                reference = joint_angles_at(esn, times, "q_ref")
                from_start = reference - joint_angles_at(logs["demonstration"], times)
                from_replayed = reference - joint_angles_at(logs["replay"], times, "q_ref")
                rows.append(
                    {"scenario": scenario, "law": law, "omega": omega, "start": start}
                    | {"from_this_start_deg": float(np.sqrt(np.mean(np.sum(from_start**2, axis=1))))}
                    | {"from_replayed_deg": float(np.sqrt(np.mean(np.sum(from_replayed**2, axis=1))))}
                )
    write_csv(OFFSET_REFERENCES, rows)
    rows = []
    for scenario, (begin, end) in DEPARTURE_WINDOWS.items():
        for law in LAW_NAMES:
            for damping in DAMPINGS:
                for start in range(8):
                    setting = f"{law}_w10" + ("" if damping == 1.0 else f"_z{damping:g}")
                    logs = run_logs(scenario, setting, start, DAMPING_SCENARIOS)
                    rows.append(
                        {"scenario": scenario, "law": law, "omega": 10.0, "damping": damping, "start": start}
                        | departure(logs, begin, end)
                    )
    write_csv(DAMPING_DEPARTURES, rows)
    rows = []
    for scenario in ("push", "block"):
        for law in LAW_NAMES:
            for start in range(8):
                esn = run_logs(scenario, f"{law}_w10_z0.1", start, DAMPING_SCENARIOS)["esn"]
                begin, end = RINGING_WINDOW
                times = esn.times[(esn.times >= begin - 1e-9) & (esn.times <= end + 1e-9)]
                arm, output = joint_angles_at(esn, times), joint_angles_at(esn, times, "q_ref")
                for joint in range(arm.shape[1]):
                    a, b = arm[:, joint] - arm[:, joint].mean(), output[:, joint] - output[:, joint].mean()
                    if a.std() < 0.2:  # this joint has stopped ringing (deg)
                        continue
                    lag = (np.argmax(np.correlate(b, a, "full")) - (len(a) - 1)) * float(np.diff(times).mean())
                    rows.append(
                        {"scenario": scenario, "law": law, "omega": 10.0, "damping": 0.1, "start": start}
                        | {"joint": joint + 1, "lag_s": lag, "amplitude_ratio": float(b.std() / a.std())}
                    )
    write_csv(RINGING, rows)


def departure(logs: dict[str, StateLog], begin: float, end: float) -> dict[str, float]:
    """The largest distance of the ESN's output and of its arm from the demonstration from ``begin`` to ``end``."""
    esn = logs["esn"]
    times = esn.times[(esn.times >= begin - 1e-9) & (esn.times <= end + 1e-9)]
    demonstration = joint_angles_at(logs["demonstration"], times)
    output = float(np.linalg.norm(joint_angles_at(esn, times, "q_ref") - demonstration, axis=1).max())
    arm = float(np.linalg.norm(joint_angles_at(esn, times) - demonstration, axis=1).max())
    return {"output_departure_deg": output, "arm_departure_deg": arm, "ratio": output / arm}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def print_departure_tables() -> None:
    """Print the comparison of the ESN's output with the arm and the demonstration, if it has been measured."""
    if not DEPARTURES.exists() or not OFFSET_REFERENCES.exists():
        print("\n(No departures yet: run with --traces to measure them from the logs.)")
        return
    with DEPARTURES.open(newline="") as f:
        rows = list(csv.DictReader(f))
    print("\nThe ESN's output and arm against the demonstration: largest distance (deg), means over the 8 starts\n")
    print("| Scenario | Law | ESN output | ESN arm | Ratio (range) |")
    print("| --- | --- | ---: | ---: | ---: |")
    for scenario in DEPARTURE_WINDOWS:
        for law, omega in SETTINGS:
            group = [r for r in rows if r["scenario"] == scenario and r["law"] == law]
            ratio = [float(r["ratio"]) for r in group]
            output = np.mean([float(r["output_departure_deg"]) for r in group])
            arm = np.mean([float(r["arm_departure_deg"]) for r in group])
            setting = f"{LAW_NAMES[law]}, ω = {omega:g}"
            ratios = f"{np.mean(ratio):.2f} ({min(ratio):.2f} to {max(ratio):.2f})"
            print(f"| {scenario} | {setting} | {output:.2f} | {arm:.2f} | {ratios} |")
    if DAMPING_DEPARTURES.exists():
        with DAMPING_DEPARTURES.open(newline="") as f:
            damping_rows = list(csv.DictReader(f))
        print("\nThe ratio as the damping ratio falls, ω = 10 rad/s: means over the 8 starts (range)\n")
        print("| Law | ζ | nominal | push | block |")
        print("| --- | ---: | ---: | ---: | ---: |")
        for law in LAW_NAMES:
            for damping in DAMPINGS:
                cells = []
                for scenario in DEPARTURE_WINDOWS:
                    values = [
                        float(r["ratio"])
                        for r in damping_rows
                        if r["scenario"] == scenario and r["law"] == law and float(r["damping"]) == damping
                    ]
                    cells.append(f"{np.mean(values):.2f} ({min(values):.2f} to {max(values):.2f})")
                print(f"| {LAW_NAMES[law]} | {damping:g} | " + " | ".join(cells) + " |")
    if RINGING.exists():
        with RINGING.open(newline="") as f:
            ringing = list(csv.DictReader(f))
        print(f"\nThe ESN's output against its ringing arm, ζ = 0.1, {RINGING_WINDOW[0]:g} to {RINGING_WINDOW[1]:g} s:")
        for scenario in ("push", "block"):
            for law in LAW_NAMES:
                group = [r for r in ringing if r["scenario"] == scenario and r["law"] == law]
                lag = 1000 * np.array([float(r["lag_s"]) for r in group])
                amplitude = np.array([float(r["amplitude_ratio"]) for r in group])
                print(
                    f"  {scenario}, {LAW_NAMES[law]}: lag {np.median(lag):.0f} ms (median, {lag.min():.0f} to"
                    f" {lag.max():.0f}), amplitude ratio {np.median(amplitude):.2f} (median, {amplitude.min():.2f} to"
                    f" {amplitude.max():.2f}), {len(group)} ringing joints"
                )
    with OFFSET_REFERENCES.open(newline="") as f:
        rows = list(csv.DictReader(f))
    print("\nThe ESN's output from offset starts: RMS distance over the reach (deg), means over the 32 starts\n")
    print("| Scenario | Law | From the demonstrator's reach from this start | From the replayed demonstration |")
    print("| --- | --- | ---: | ---: |")
    for scenario in ("offset 3°", "offset 10°"):
        for law, omega in SETTINGS:
            group = [r for r in rows if r["scenario"] == scenario and r["law"] == law]
            this_start = np.mean([float(r["from_this_start_deg"]) for r in group])
            replayed = np.mean([float(r["from_replayed_deg"]) for r in group])
            print(f"| {scenario} | {LAW_NAMES[law]}, ω = {omega:g} | {this_start:.2f} | {replayed:.2f} |")


def export_animations() -> None:
    """Export GIFs of a few runs with skelarm's player, from the logs under the storage root."""
    from arm_esn_ctrl.storage import REPO_ROOT, storage_root

    player = REPO_ROOT / "third_party" / "skelarm" / "tools" / "player.py"
    for name, run_name, setting, arm, start in ANIMATIONS:
        run = storage_root() / "results" / run_name
        log = (
            run / f"{arm}_{start:02d}.sklog.npz" if setting is None else run / setting / f"{arm}_{start:02d}.sklog.npz"
        )
        gif = SUMMARY / f"{name}.gif"
        subprocess.run(
            [sys.executable, str(player), str(log), "--export", str(gif), "--fps", f"{ANIMATION_FPS:g}"],
            check=True,
            env=os.environ | {"QT_QPA_PLATFORM": "offscreen"},  # render without a window
        )


if __name__ == "__main__":
    main()
