# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 018.

    uv run python reports/018-trackers-under-one-protocol/make_figures.py
    uv run python reports/018-trackers-under-one-protocol/make_figures.py --logs

From the copies of the robot runs under ``results/`` (configurations, run records, and
metrics), it prints the tables of the report and draws into ``results/summary``:

- ``measures.png``: each arm's following, ringing, return, and farthest from the taught
  path against the tracker, in the mean over the block and the three scripted pushes,
  and over the ten pushes by hand (Sections 3.4 and 3.5);
- ``effort.png``: each arm's accumulated torque against the tracker, undisturbed and
  from the offset starts, and its excess over the same arm undisturbed under the
  block, the scripted pushes, and the pushes by hand (Section 3.6).

With ``--logs``, it reads the runs' logs, which are not kept in Git, from the runs
under the storage root, and draws ``series_<tracker>_esn1.png`` and
``series_<tracker>_esn2.png`` for each tracker: the tip force, and each ESN's joint
angles and torques over time against the replay's and the progress replay's, a
column per scenario, the ESN's lines over the replays' (Section 3.7).
"""

from __future__ import annotations

import argparse
import csv
import functools
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from skelarm import StateLog

from arm_esn_ctrl.demonstrations import resample_joint_angles
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.storage import resolve_run_path

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
# Report 017's copies of the search's step 4, which the search's tracker here repeats.
SEARCH_RESULTS = REPORT.parent / "017-search-under-one-protocol" / "results"
# The take, as report 005 copied it: what the ESNs were trained on, and what the replays replay.
TAKE_RUN = REPORT.parent / "005-tuned-on-the-robot" / "data" / "20261008-170034-reach_manual_v2"
DT = 0.01  # the ESNs' period (s)
DPI = 90
RADIUS = 0.02  # the goal radius (m)
HOLD = 2.0  # the hold window (s)

# The trackers, in the order of the figures and tables: the law, the damping ratio, and the reference velocity given.
Tracker = tuple[str, float, str]
TRACKERS: list[Tracker] = [
    (law, damping, velocity)
    for law, velocity in (("pd", "zero"), ("pd", "reference"), ("computed_torque", "reference"))
    for damping in (0.2, 0.4, 0.8)
]
GROUPS = {
    ("pd", "zero"): "joint PD,\nzero velocity",
    ("pd", "reference"): "joint PD,\nreference velocity",
    ("computed_torque", "reference"): "computed torque,\nreference velocity",
}
SEARCH_TRACKER: Tracker = ("pd", 0.4, "reference")
SCENARIOS = {
    "nominal": "undisturbed",
    "block": "the block",
    "pushes": "the pushes",
    "offsets": "the offset starts",
    "recorded000": "log 000",
    "recorded001": "log 001",
    "recorded002": "log 002",
}
RECORDED = ["recorded000", "recorded001", "recorded002"]
SEARCH_SCENARIOS = ["nominal", "block", "pushes", "offsets"]
# The earlier reports' copies of their robot runs of the same two ESNs, which joint PD here repeats: each report, the
# stem of its runs' configurations (with the scenario, the ESN's setting, and the reference velocity filled in), its
# scenarios, and the arms it ran.
EARLIER = [
    (
        "011-following-the-arm",
        "{scenario}_robot_{setting}_{velocity}_velocity_filtered",
        SEARCH_SCENARIOS,
        ("esn", "replay"),
    ),
    (
        "012-pushes-by-hand",
        "{scenario}_robot_{setting}_{velocity}_velocity_filtered",
        RECORDED,
        ("esn", "replay"),
    ),
    (
        "014-progress-replay",
        "{scenario}_w20_z0.4_robot_{setting}_{velocity}_velocity_filtered",
        [*SEARCH_SCENARIOS, *RECORDED],
        ("esn", "replay", "progress"),
    ),
]
# The chosen ESNs: their name and the tag of their runs.
CHOSEN = {"ESN #1": "esn1_lr0.03_sr1.2_is1.5_seed8", "ESN #2": "esn2_lr0.03_sr1.2_is1_seed2"}
# The two ESNs' settings, as the earlier reports name their runs.
SETTINGS = {"ESN #1": "lr0.03_sr1.2_is1.5_seed8", "ESN #2": "lr0.03_sr1.2_is1_seed2"}
# The arms of the tables and figures: name, the tag of the runs read, the arm in them, color, and marker. The
# replays are the same beside either ESN.
ARMS = [
    ("ESN #1", CHOSEN["ESN #1"], "esn", "#2a78d6", "o"),
    ("ESN #2", CHOSEN["ESN #2"], "esn", "#2a78d6", "s"),
    ("replay", CHOSEN["ESN #1"], "replay", "#eb6834", "^"),
    ("progress replay", CHOSEN["ESN #1"], "progress", "#1baf7a", "D"),
]
# The columns of the series over time: the scenario of the runs, the start drawn, and the title.
COLUMNS = [
    ("nominal", "demo_00_filtered", "undisturbed"),
    ("block", "demo_00_filtered", "under the block"),
    ("pushes", "demo_00_filtered", "under the pushes"),
    ("offsets", "demo_00_filtered +15,+15 deg", "from (+15°, +15°)"),
    ("offsets", "demo_00_filtered -15,-15 deg", "from (-15°, -15°)"),
    ("recorded000", "recorded 0", "log 000, by hand"),
    ("recorded001", "recorded 0", "log 001, by hand"),
    ("recorded002", "recorded 0", "log 002, by hand"),
]
START_CUT = 0.05  # s: from an offset start, the torque's axis leaves out the peak of the task's first 50 ms

TEXT_COLOR = "#52514e"
TITLE_COLOR = "#0b0b0b"
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAUGHT_COLOR = "#b9b7b0"
SERIES_COLORS = {"esn": "#2a78d6", "replay": "#eb6834", "progress": "#1baf7a"}
# The layers of the lines over time: the ESN's arm on top, its reference (dashed) under it, then the progress
# replay's arm and reference, the replay's at the bottom.
ARM_ZORDERS = {"esn": 3.0, "progress": 2.2, "replay": 2.0}
REFERENCE_ZORDERS = {"esn": 2.5, "progress": 2.1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also draw the runs over time from their logs")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {"measures.png": plot_measures(), "effort.png": plot_effort()}
    if args.logs:
        for tracker in TRACKERS:
            for name, tag in CHOSEN.items():
                figures[f"series_{tracker_tag(tracker)}_esn{name[-1]}.png"] = plot_series(name, tag, tracker)
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=DPI)
    print_tables()
    print(f"\nWrote {', '.join(figures)} to {SUMMARY}")


# ---------------------------------------------------------------------------- reading the runs


def value(text: str) -> Any:
    """A CSV field as a bool, a float, or the text itself."""
    if text in ("True", "False"):
        return text == "True"
    try:
        return float(text)
    except ValueError:
        return text


@functools.cache
def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [{key: value(text) for key, text in row.items()} for row in csv.DictReader(f)]


def run_name(stem: str, results: Path = RESULTS) -> str:
    """The name of the one run of a configuration."""
    matches = sorted(p.name for p in results.glob(f"*-{stem}") if p.is_dir())
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


def stem(scenario: str, tag: str, tracker: Tracker) -> str:
    """The configuration of the runs of one scenario, ESN, and reference velocity."""
    return f"{scenario}_robot_{tag}_{tracker[2]}_velocity"


def tracker_rows(scenario: str, tag: str, tracker: Tracker, arm: str) -> list[dict[str, Any]]:
    """One arm's runs with one tracker, from its start postures in order."""
    law, damping, _ = tracker
    rows = read_csv(RESULTS / run_name(stem(scenario, tag, tracker)) / "metrics.csv")
    return [r for r in rows if r["law"] == law and r["damping"] == damping and r["arm"] == arm]


def first_run(scenario: str, tag: str, tracker: Tracker, arm: str) -> dict[str, Any]:
    """One arm's run from its first start: the take's, or the recorded log's."""
    return tracker_rows(scenario, tag, tracker, arm)[0]


def moved_runs(tag: str, tracker: Tracker, arm: str) -> list[dict[str, Any]]:
    """One arm's runs from the 48 starts offset from the take's."""
    return [r for r in tracker_rows("offsets", tag, tracker, arm) if str(r["origin"]).endswith(" deg")]


@functools.cache
def windows(scenario: str) -> list[tuple[str, float]]:
    """Each disturbance of a scenario: its columns' suffix, and how long its window lasts after its end (s)."""
    with (RESULTS / run_name(stem(scenario, CHOSEN["ESN #1"], SEARCH_TRACKER)) / "config.toml").open("rb") as f:
        config = tomllib.load(f)
    spans = disturbance_spans(config.get("disturbance"))
    if not spans:
        return []
    suffixes = [""] if len(spans) == 1 else [f"_{k}" for k in range(1, len(spans) + 1)]
    ends = [onset for onset, _ in spans[1:]] + [float(config["evaluation"]["duration"])]
    return [(suffix, stop - end) for suffix, (_, end), stop in zip(suffixes, spans, ends, strict=True)]


def failure(run: dict[str, Any], scenario: str) -> str:
    """How a run fails by the rule's first point, or "": under any push, leaving the goal during the hold is excused."""
    if not np.isfinite(run["arrival_time_s"]):
        return "never arrives"
    if run["final_distance_m"] > RADIUS:
        return "ends away"
    if scenario == "pushes" or scenario in RECORDED:
        return ""
    if run["left_goal"]:
        return "leaves the goal during the hold"
    return "hold cut short" if not run["success"] or run["hold_observed_s"] < HOLD - 1e-9 else ""


def after_disturbances(scenarios: list[str], tag: str, tracker: Tracker, arm: str) -> dict[str, list[float]]:
    """After each disturbance of some scenarios: the following, ringing (mm), return (s), and farthest (mm).

    A hand not back within 5 mm of the taught path by the end of a disturbance's window
    counts as back then, as by the rule; ``not back`` counts those.
    """
    out: dict[str, list[float]] = {"following": [], "ringing": [], "return": [], "farthest": [], "not back": []}
    for scenario in scenarios:
        run = first_run(scenario, tag, tracker, arm)
        for suffix, window in windows(scenario):
            back = run[f"return_time_s{suffix}"]
            out["following"].append(run[f"following_gain{suffix}"])
            out["ringing"].append(1000 * run[f"ringing_m{suffix}"])
            out["return"].append(back if np.isfinite(back) else window)
            out["farthest"].append(1000 * run[f"off_path_m{suffix}"])
            out["not back"].append(float(not np.isfinite(back)))
    return out


def scripted(tag: str, tracker: Tracker, arm: str) -> dict[str, float]:
    """The rule's measures: the means over the block and the three scripted pushes."""
    return {k: float(np.mean(v)) for k, v in after_disturbances(["block", "pushes"], tag, tracker, arm).items()}


def by_hand(tag: str, tracker: Tracker, arm: str) -> dict[str, float]:
    """The same over the ten pushes by hand, the means, as the rule takes them; ``not back`` their count."""
    measures = after_disturbances(RECORDED, tag, tracker, arm)
    return {k: float(np.sum(v) if k == "not back" else np.mean(v)) for k, v in measures.items()}


def effort(scenario: str, tag: str, tracker: Tracker, arm: str, column: str = "task_torque_nms") -> float:
    """How hard the motors work over the task: from the offset starts the median over the 48, else the first run."""
    if scenario == "offsets":
        return float(np.median([r[column] for r in moved_runs(tag, tracker, arm)]))
    return float(first_run(scenario, tag, tracker, arm)[column])


def excess(scenarios: list[str], tag: str, tracker: Tracker, arm: str, column: str = "task_torque_nms") -> float:
    """How much more the motors work under some disturbances than undisturbed, in the mean over the scenarios."""
    base = effort("nominal", tag, tracker, arm, column)
    return float(np.mean([effort(s, tag, tracker, arm, column) - base for s in scenarios]))


def robot_log(scenario: str, tag: str, tracker: Tracker, arm: str, origin: str) -> StateLog:
    """One arm's log from a run under the storage root."""
    law, damping, _ = tracker
    run = next(r for r in tracker_rows(scenario, tag, tracker, arm) if r["origin"] == origin)
    path = resolve_run_path(f"results/{run_name(stem(scenario, tag, tracker))}")
    return StateLog.load(path / f"{law}_w20_z{damping:g}" / f"{arm}_{int(run['start']):02d}.sklog.npz")


def tip_force(log: StateLog) -> NDArray[np.float64]:
    """The size of the force at the tip over time: zero in an undisturbed run, whose log has no force."""
    try:
        return np.linalg.norm(log.channel("ext_force"), axis=1)
    except KeyError:
        return np.zeros(len(log.times))


@functools.cache
def taught() -> NDArray[np.float64]:
    """The take filtered at 2 Hz, what the ESNs were trained on, every 10 ms: its joint angles."""
    return resample_joint_angles(StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"), DT)[1]


def tracker_tag(tracker: Tracker) -> str:
    law, damping, velocity = tracker
    return f"{'pd' if law == 'pd' else 'ct'}_z{damping:g}_{velocity}"


def tracker_label(tracker: Tracker) -> str:
    law, damping, velocity = tracker
    if law == "computed_torque":
        return f"computed torque, ζ = {damping:g}"
    return f"joint PD, ζ = {damping:g}, {'a zero' if velocity == 'zero' else 'the reference'} velocity"


# ---------------------------------------------------------------------------- figures


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def positions() -> list[float]:
    """Each tracker's place along the x axis: the three damping ratios of a group together, a gap between groups."""
    return [k + 0.8 * (k // 3) for k in range(len(TRACKERS))]


def tracker_axis(ax: Axes, labels: bool) -> None:
    """The damping ratios under each point, and, if ``labels``, each group's law and velocity under them."""
    xs = positions()
    ax.set_xticks(xs, [f"{damping:g}" for _, damping, _ in TRACKERS])
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    if labels:
        for k, text in enumerate(GROUPS.values()):
            ax.annotate(
                text,
                xy=(xs[3 * k + 1], 0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -20),
                textcoords="offset points",
                ha="center",
                va="top",
                color=TEXT_COLOR,
                fontsize=8.5,
            )


def draw_arms(ax: Axes, values: dict[str, list[float]]) -> None:
    """Each arm's values against the tracker: a line through each group's three damping ratios."""
    xs = positions()
    for name, _, _, color, marker in ARMS:
        hollow = name == "ESN #2"
        for g in range(len(GROUPS)):
            part = slice(3 * g, 3 * g + 3)
            ax.plot(
                xs[part],
                values[name][part],
                color=color,
                linewidth=1.6,
                linestyle="--" if hollow else "-",
                marker=marker,
                markersize=6.5,
                markerfacecolor=SURFACE_COLOR if hollow else color,
                markeredgewidth=1.4,
                zorder=3.0 if name.startswith("ESN") else 2.0,
            )


def arm_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            color=color,
            linewidth=1.6,
            linestyle="--" if name == "ESN #2" else "-",
            marker=marker,
            markersize=6.5,
            markerfacecolor=SURFACE_COLOR if name == "ESN #2" else color,
            markeredgewidth=1.4,
            label=name,
        )
        for name, _, _, color, marker in ARMS
    ]


def plot_measures() -> Figure:
    """The rule's measures against the tracker: after the block and the scripted pushes, and the pushes by hand."""
    measures = [
        ("following", "following gain", 1.0),
        ("ringing", "ringing over the next 2 s (mm)", 0.0),
        ("return", "back within 5 mm of the path (s)", 0.0),
        ("farthest", "farthest from the path (mm)", 0.0),
    ]
    rows = [("after the block and\nthe three scripted pushes", scripted), ("after the ten pushes by hand", by_hand)]
    fig = Figure(figsize=(17, 9.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Each arm after a disturbance, against the tracker (the damping ratio ζ, the law, and the reference velocity"
        " given): the means over the disturbances",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(2, len(measures), sharex=True)
    for row, (title, measure_of) in enumerate(rows):
        values = {name: [measure_of(tag, t, arm) for t in TRACKERS] for name, tag, arm, _, _ in ARMS}
        for col, (key, label, _) in enumerate(measures):
            ax = axes[row, col]
            style(ax)
            draw_arms(ax, {name: [v[key] for v in vs] for name, vs in values.items()})
            ax.set_ylabel(f"{title}\n\n{label}" if col == 0 else label, color=TEXT_COLOR)
            if row == 0:
                ax.set_title(label.split(" (")[0], color=TITLE_COLOR, fontsize=10)
            tracker_axis(ax, labels=row == 1)
            low = min(0.0, ax.get_ylim()[0])
            ax.set_ylim(low, None)
    fig.legend(handles=arm_handles(), loc="outside lower center", ncol=len(ARMS), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_effort() -> Figure:
    """The accumulated torque against the tracker: undisturbed, from the offset starts, and its excess disturbed."""
    panels = [
        ("undisturbed (the progress replay mostly the replay)", lambda tag, t, arm: effort("nominal", tag, t, arm)),
        (
            "from the offset starts, the median over the 48\n(the progress replay mostly the replay)",
            lambda tag, t, arm: effort("offsets", tag, t, arm),
        ),
        ("under the block, more than undisturbed", lambda tag, t, arm: excess(["block"], tag, t, arm)),
        ("under the scripted pushes, more than undisturbed", lambda tag, t, arm: excess(["pushes"], tag, t, arm)),
        (
            "under the pushes by hand, more than undisturbed,\nthe mean over the three logs",
            lambda tag, t, arm: excess(RECORDED, tag, t, arm),
        ),
    ]
    fig = Figure(figsize=(16, 9.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "How hard the motors work over the task: each arm's accumulated torque ∫ Σ|τ| dt, against the tracker",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(2, 3).ravel()
    for ax, (title, of) in zip(axes, panels, strict=False):
        style(ax)
        draw_arms(ax, {name: [of(tag, t, arm) for t in TRACKERS] for name, tag, arm, _, _ in ARMS})
        ax.set_title(title, color=TITLE_COLOR, fontsize=9.5)
        ax.set_ylabel("accumulated torque (N m s)", color=TEXT_COLOR)
        ax.set_ylim(0.0, None)
        tracker_axis(ax, labels=True)
    axes[-1].axis("off")
    axes[-1].legend(handles=arm_handles(), loc="center", frameon=False, labelcolor=TEXT_COLOR, fontsize=11)
    return fig


def plot_series(name: str, tag: str, tracker: Tracker) -> Figure:
    """One ESN's joint angles and torques over time against both replays', with one tracker: a column per scenario."""
    fig = Figure(figsize=(24, 13.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        f"{name}'s arm (blue), the replay's (orange), and the progress replay's (aqua), with {tracker_label(tracker)};"
        " dashed: the references of the ESN and the progress replay; thick gray: the taught motion; from an offset"
        f" start, the torque's peak in the first {1000 * START_CUT:.0f} ms is cut",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(5, len(COLUMNS), sharex=True, height_ratios=[0.55, 1.0, 1.0, 1.0, 1.0])
    q_taught = np.degrees(taught())
    t_taught = DT * np.arange(len(q_taught))
    logs = [
        {arm: robot_log(scenario, tag, tracker, arm, origin) for arm in SERIES_COLORS}
        for scenario, origin, _ in COLUMNS
    ]
    angle_limits = []
    for joint in range(2):
        values = [np.degrees(log.channel("q")[:, joint]) for c in logs for log in c.values()]
        values.append(q_taught[:, joint])
        low, high = float(min(v.min() for v in values)), float(max(v.max() for v in values))
        angle_limits.append((low - 0.05 * (high - low), high + 0.05 * (high - low)))
    top = max(float(tip_force(log).max()) for c in logs for log in c.values())
    for col, ((scenario, _, title), arms) in enumerate(zip(COLUMNS, logs, strict=True)):
        with (RESULTS / run_name(stem(scenario, tag, tracker)) / "config.toml").open("rb") as f:
            spans = disturbance_spans(tomllib.load(f).get("disturbance"))
        for ax in axes[:, col]:
            style(ax)
            for onset, end in spans:
                ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
        axes[0, col].set_title(title, color=TITLE_COLOR, fontsize=10)
        for arm, color in SERIES_COLORS.items():
            axes[0, col].plot(
                arms[arm].times, tip_force(arms[arm]), color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm]
            )
        axes[0, col].set_ylim(-0.05 * top, 1.1 * top)
        for joint in range(2):
            ax = axes[1 + joint, col]
            ax.plot(t_taught, q_taught[:, joint], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            for arm, color in SERIES_COLORS.items():
                log = arms[arm]
                ax.plot(
                    log.times,
                    np.degrees(log.channel("q")[:, joint]),
                    color=color,
                    linewidth=1.2,
                    zorder=ARM_ZORDERS[arm],
                )
                if arm in REFERENCE_ZORDERS:
                    ax.plot(
                        log.times,
                        np.degrees(log.channel("q_ref")[:, joint]),
                        color=color,
                        linewidth=1.0,
                        linestyle="--",
                        zorder=REFERENCE_ZORDERS[arm],
                    )
            ax.set_ylim(*angle_limits[joint])
            ax = axes[3 + joint, col]
            shown = []
            for arm, color in SERIES_COLORS.items():
                log = arms[arm]
                ax.plot(log.times, log.channel("tau")[:, joint], color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm])
                kept = log.times >= (START_CUT if scenario == "offsets" else -np.inf)
                shown.append(log.channel("tau")[kept, joint])
            low, high = float(min(v.min() for v in shown)), float(max(v.max() for v in shown))
            ax.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
        axes[-1, col].set_xlim(-1.0, 24.0)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    axes[0, 0].set_ylabel("tip force (N)", color=TEXT_COLOR)
    for joint in range(2):
        axes[1 + joint, 0].set_ylabel(f"joint {joint + 1} angle (deg)", color=TEXT_COLOR)
        axes[3 + joint, 0].set_ylabel(f"joint {joint + 1} torque (N m)", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=SERIES_COLORS["esn"], linewidth=1.4, label=f"{name}'s arm"),
        Line2D([], [], color=SERIES_COLORS["replay"], linewidth=1.4, label="the replay's arm"),
        Line2D([], [], color=SERIES_COLORS["progress"], linewidth=1.4, label="the progress replay's arm"),
        Line2D([], [], color=TEXT_COLOR, linewidth=1.0, linestyle="--", label="its reference"),
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught motion"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- tables


def print_tables() -> None:
    print_checks()
    print_earlier()
    print_outcomes()
    print_undisturbed()
    print_offsets()
    print_disturbances()
    print_split()
    print_block()
    print_by_hand()
    print_effort()


def row_text(row: dict[str, Any]) -> dict[str, str]:
    return {key: str(v) for key, v in row.items()}


def print_checks() -> None:
    """The replays alike beside either ESN, and the search's tracker repeating the search's step 4."""
    for scenario in SCENARIOS:
        for tracker in TRACKERS:
            for arm in ("replay", "progress"):
                a = tracker_rows(scenario, CHOSEN["ESN #1"], tracker, arm)
                b = tracker_rows(scenario, CHOSEN["ESN #2"], tracker, arm)
                assert [row_text(r) for r in a] == [row_text(r) for r in b], (scenario, tracker, arm)
    print("The replays' runs beside ESN #1 and beside ESN #2: identical, in every scenario and with every tracker")
    count = 0
    for scenario in SEARCH_SCENARIOS:
        for tag in CHOSEN.values():
            old = read_csv(SEARCH_RESULTS / run_name(f"{scenario}_robot_{tag}", SEARCH_RESULTS) / "metrics.csv")
            new = [r for r in read_csv(RESULTS / run_name(stem(scenario, tag, SEARCH_TRACKER)) / "metrics.csv")]
            new = [r for r in new if r["law"] == "pd" and r["damping"] == 0.4]
            assert [row_text(r) for r in old] == [row_text(r) for r in new], (scenario, tag)
            count += len(new)
    print(f"Joint PD at 0.4, given the reference velocity: the {count} runs repeat report 017's step 4 to the digit")


def print_earlier() -> None:
    """Joint PD's runs against the earlier reports' runs of the same ESNs: every field both record."""
    for report, pattern, scenarios, arms in EARLIER:
        results = REPORT.parent / report / "results"
        count, largest = 0, 0.0
        for name, tag in CHOSEN.items():
            for scenario in scenarios:
                for velocity in ("reference", "zero"):
                    old_stem = pattern.format(scenario=scenario, setting=SETTINGS[name], velocity=velocity)
                    old = read_csv(results / run_name(old_stem, results) / "metrics.csv")
                    new = read_csv(RESULTS / run_name(f"{scenario}_robot_{tag}_{velocity}_velocity") / "metrics.csv")
                    for row in (r for r in old if r["arm"] in arms):
                        (match,) = [
                            r for r in new if all(r[key] == row[key] for key in ("law", "damping", "arm", "origin"))
                        ]
                        count += 1
                        largest = max(largest, largest_difference(row, match))
        print(f"Report {report[:3]}'s {count} runs ({', '.join(arms)}): repeated within a relative {largest:.1e}")


def largest_difference(a: dict[str, Any], b: dict[str, Any]) -> float:
    """The largest relative difference between two runs' rows in the fields both have; inf if another field differs."""
    largest = 0.0
    for key in a.keys() & b.keys():
        x, y = a[key], b[key]
        if isinstance(x, float) and isinstance(y, float):
            if np.isnan(x) and np.isnan(y):
                continue
            if np.isnan(x) != np.isnan(y):
                return float("inf")
            largest = max(largest, abs(x - y) / max(abs(x), abs(y), 1e-12))
        elif x != y:
            return float("inf")
    return largest


def print_outcomes() -> None:
    """Every run by the rule's first point; the arrivals over the trackers; who leaves the goal under a push."""
    print("\nRuns failing by the rule's first point:")
    failing = 0
    for name, tag, arm, _, _ in ARMS:
        for tracker in TRACKERS:
            for scenario in SCENARIOS:
                for run in tracker_rows(scenario, tag, tracker, arm):
                    if how := failure(run, scenario):
                        failing += 1
                        print(f"  {name}, {tracker_label(tracker)}, {scenario} {run['origin']}: {how}")
    runs = sum(len(tracker_rows(s, CHOSEN["ESN #1"], t, "esn")) for s in SCENARIOS for t in TRACKERS)
    print(f"  {failing} of the {4 * runs} runs of the four arms ({runs} each)")
    print("\nArrival (s), the earliest to the latest over the nine trackers; offsets: the latest over the 49 starts")
    print("| Arm | " + " | ".join(SCENARIOS.values()) + " |")
    print("| --- |" + " --- |" * len(SCENARIOS))
    for name, tag, arm, _, _ in ARMS:
        cells = []
        for scenario in SCENARIOS:
            if scenario == "offsets":
                times = [max(r["arrival_time_s"] for r in tracker_rows(scenario, tag, t, arm)) for t in TRACKERS]
            else:
                times = [first_run(scenario, tag, t, arm)["arrival_time_s"] for t in TRACKERS]
            cells.append(f"{min(times):.2f} to {max(times):.2f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\nLeaving the goal during the hold, excused under a push; and the latest arrivals under log 001:")
    for name, tag, arm, _, _ in ARMS:
        for tracker in TRACKERS:
            for scenario in SCENARIOS:
                for run in tracker_rows(scenario, tag, tracker, arm):
                    if run["left_goal"]:
                        print(
                            f"  {name}, {tracker_label(tracker)}, {scenario}: arrives {run['arrival_time_s']:.2f} s,"
                            f" ends {1000 * run['final_distance_m']:.1f} mm away"
                        )
    print("\nThe largest final distance (mm), over every run of each arm:")
    for name, tag, arm, _, _ in ARMS:
        worst = max(
            (r["final_distance_m"], s, t) for s in SCENARIOS for t in TRACKERS for r in tracker_rows(s, tag, t, arm)
        )
        print(f"  {name}: {1000 * worst[0]:.1f} mm ({worst[1]}, {tracker_label(worst[2])})")


def print_undisturbed() -> None:
    """Undisturbed, from the take's start: the path RMSE, the arrival, the peak torque, and the accumulated torque."""
    print("\nUndisturbed: path RMSE (mm), arrival (s), peak torque (N m), speed peaks; ESN #1 / ESN #2 / replay")
    print("| Tracker | Path RMSE (mm) | Arrival (s) | Peak torque (N m) | Speed peaks of the arm |")
    print("| --- | --- | --- | --- | --- |")
    for tracker in TRACKERS:
        runs = [first_run("nominal", tag, tracker, arm) for _, tag, arm, _, _ in ARMS[:3]]
        print(
            f"| {tracker_label(tracker)} | "
            + " | ".join(
                " / ".join(fmt.format(scale * r[column]) for r in runs)
                for column, scale, fmt in (
                    ("taught_path_rmse_m", 1000, "{:.1f}"),
                    ("arrival_time_s", 1, "{:.2f}"),
                    ("peak_torque_nm", 1, "{:.2f}"),
                    ("speed_peaks", 1, "{:.0f}"),
                )
            )
            + " |"
        )
    print(
        "  Undisturbed, the progress replay against the replay: arrival (s), path RMSE (mm), accumulated torque (N m s)"
    )
    for tracker in TRACKERS:
        a = first_run("nominal", CHOSEN["ESN #1"], tracker, "replay")
        b = first_run("nominal", CHOSEN["ESN #1"], tracker, "progress")
        if row_text(a | {"arm": ""}) == row_text(b | {"arm": ""}):
            print(f"    {tracker_label(tracker)}: the same to the digit")
        else:
            print(
                f"    {tracker_label(tracker)}: {b['arrival_time_s']:.2f} against {a['arrival_time_s']:.2f};"
                f" {1000 * b['taught_path_rmse_m']:.2f} against {1000 * a['taught_path_rmse_m']:.2f};"
                f" {b['task_torque_nms']:.2f} against {a['task_torque_nms']:.2f}"
            )
    reference = first_run("nominal", CHOSEN["ESN #1"], SEARCH_TRACKER, "esn")
    print(f"  the ESN's reference's peak speed {reference['peak_reference_speed_dps']:.1f} deg/s;", end=" ")
    replay = first_run("nominal", CHOSEN["ESN #1"], SEARCH_TRACKER, "replay")
    print(
        f"the replay's {replay['peak_reference_speed_dps']:.1f} deg/s;"
        f" the take's speed peaks {replay['taught_speed_peaks']:.0f}"
    )


def print_offsets() -> None:
    """From the 48 offset starts: the medians of the path RMSE and the first step, and the peak torque."""
    print("\nFrom the 48 offset starts: medians of the path RMSE (mm) and the first step (mm), peak torque median /")
    print("largest (N m); ESN #1 / ESN #2 / replay (the progress replay's alike)")
    print("| Tracker | Path RMSE (mm) | First step (mm) | Peak torque (N m) |")
    print("| --- | --- | --- | --- |")
    for tracker in TRACKERS:
        cells = []
        for column, scale in (("taught_path_rmse_m", 1000), ("first_step_m", 1000)):
            cells.append(
                " / ".join(
                    f"{np.median([scale * r[column] for r in moved_runs(tag, tracker, arm)]):.1f}"
                    for _, tag, arm, _, _ in ARMS[:3]
                )
            )
        cells.append(
            " / ".join(
                f"{np.median([r['peak_torque_nm'] for r in moved_runs(tag, tracker, arm)]):.0f},"
                f" {max(r['peak_torque_nm'] for r in moved_runs(tag, tracker, arm)):.0f}"
                for _, tag, arm, _, _ in ARMS[:3]
            )
        )
        print(f"| {tracker_label(tracker)} | " + " | ".join(cells) + " |")
    for tracker in TRACKERS:
        a = moved_runs(CHOSEN["ESN #1"], tracker, "replay")
        b = moved_runs(CHOSEN["ESN #1"], tracker, "progress")
        largest = max(abs(x["arrival_time_s"] - y["arrival_time_s"]) for x, y in zip(a, b, strict=True))
        differ = sum(abs(x["arrival_time_s"] - y["arrival_time_s"]) > 1e-9 for x, y in zip(a, b, strict=True))
        print(
            f"  {tracker_label(tracker)}: the progress replay's arrival differs from the replay's from {differ}"
            f" starts, by {largest:.2f} s at most"
        )


def print_disturbances() -> None:
    """After the block and the scripted pushes: the rule's means, each arm, each tracker."""
    print("\nAfter the block and the three scripted pushes, the means: ESN #1 / ESN #2 / replay / progress replay")
    print("| Tracker | Following | Ringing (mm) | Back within 5 mm (s) | Farthest (mm) |")
    print("| --- | --- | --- | --- | --- |")
    for tracker in TRACKERS:
        measures = [scripted(tag, tracker, arm) for _, tag, arm, _, _ in ARMS]
        print(f"| {tracker_label(tracker)} | " + measure_cells(measures) + " |")
    print("  not back before the next disturbance or the end, of the four:")
    for name, tag, arm, _, _ in ARMS:
        counts = [round(4 * scripted(tag, t, arm)["not back"]) for t in TRACKERS]
        print(f"    {name}: {counts}")


def print_split() -> None:
    """The following, ringing, and return after the block alone and in the mean over the three scripted pushes."""
    print(
        "\nAfter the block, and in the mean over the three scripted pushes: ESN #1 / ESN #2 / replay / progress replay"
    )
    print(
        "| Tracker | Following, block | Following, pushes | Ringing (mm), block | Ringing (mm), pushes |"
        " Back (s), block | Back (s), pushes |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for tracker in TRACKERS:
        block = [scripted_part("block", tag, tracker, arm) for _, tag, arm, _, _ in ARMS]
        pushes = [scripted_part("pushes", tag, tracker, arm) for _, tag, arm, _, _ in ARMS]
        cells = [
            " / ".join(fmt.format(unsigned_zero(m[key])) for m in part)
            for key, fmt in (("following", "{:.2f}"), ("ringing", "{:.1f}"), ("return", "{:.2f}"))
            for part in (block, pushes)
        ]
        print(f"| {tracker_label(tracker)} | " + " | ".join(cells) + " |")


def scripted_part(scenario: str, tag: str, tracker: Tracker, arm: str) -> dict[str, float]:
    """The means over one scenario's disturbances: the block's one, or the three scripted pushes."""
    return {k: float(np.mean(v)) for k, v in after_disturbances([scenario], tag, tracker, arm).items()}


def measure_cells(measures: list[dict[str, float]]) -> str:
    return " | ".join(
        " / ".join(fmt.format(unsigned_zero(m[key])) for m in measures)
        for key, fmt in (("following", "{:.2f}"), ("ringing", "{:.1f}"), ("return", "{:.2f}"), ("farthest", "{:.0f}"))
    )


def unsigned_zero(x: float) -> float:
    """``x``, but 0 for a value that would print as -0.00, such as the replay's following."""
    return 0.0 if abs(x) < 5e-3 else x


def print_block() -> None:
    """Under the block: the reference's lead at the release, the peak torque, and the block's peak force."""
    print("\nUnder the block: lead at the release (%), peak torque (N m), the block's peak force (N);")
    print("ESN #1 / ESN #2 / replay / progress replay")
    print("| Tracker | Lead at the release (%) | Peak torque (N m) | Peak force on the tip (N) |")
    print("| --- | --- | --- | --- |")
    for tracker in TRACKERS:
        runs = [first_run("block", tag, tracker, arm) for _, tag, arm, _, _ in ARMS]
        cells = [
            " / ".join(fmt.format(scale * r[column]) for r in runs)
            for column, scale, fmt in (
                ("reference_lead", 100, "{:.1f}"),
                ("peak_torque_nm", 1, "{:.1f}"),
                ("peak_external_force_n", 1, "{:.1f}"),
            )
        ]
        print(f"| {tracker_label(tracker)} | " + " | ".join(cells) + " |")


def print_by_hand() -> None:
    """After the ten pushes by hand: the means, each arm, each tracker; and how many pushes the hand is not back."""
    print("\nAfter the ten pushes by hand, the means: ESN #1 / ESN #2 / replay / progress replay")
    print("| Tracker | Following | Ringing (mm) | Back within 5 mm (s) | Farthest (mm) | Not back (of 10) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for tracker in TRACKERS:
        measures = [by_hand(tag, tracker, arm) for _, tag, arm, _, _ in ARMS]
        counts = " / ".join(f"{m['not back']:.0f}" for m in measures)
        print(f"| {tracker_label(tracker)} | " + measure_cells(measures) + f" | {counts} |")
    print("\nPer log, the ringing (mm): ESN #1 / ESN #2 / replay / progress replay, the means over its pushes")
    print("| Tracker | Log 000 | Log 001 | Log 002 |")
    print("| --- | --- | --- | --- |")
    for tracker in TRACKERS:
        cells = []
        for scenario in RECORDED:
            values = [
                np.mean(after_disturbances([scenario], tag, tracker, arm)["ringing"]) for _, tag, arm, _, _ in ARMS
            ]
            cells.append(" / ".join(f"{v:.1f}" for v in values))
        print(f"| {tracker_label(tracker)} | " + " | ".join(cells) + " |")


def print_effort() -> None:
    """The accumulated torque (N m s): undisturbed, from the offset starts, and its excess when disturbed."""
    print(
        "\nAccumulated torque (N m s): undisturbed, its excess under the block, the scripted pushes, and the pushes by"
    )
    print("hand (the mean over the logs), and from the offset starts (the median); ESN #1 / ESN #2 / replay /")
    print("progress replay")
    print("| Tracker | Undisturbed | Block, more | Pushes, more | By hand, more | Offset starts |")
    print("| --- | --- | --- | --- | --- | --- |")
    for tracker in TRACKERS:
        cells = [
            " / ".join(f"{effort('nominal', tag, tracker, arm):.2f}" for _, tag, arm, _, _ in ARMS),
            " / ".join(f"{excess(['block'], tag, tracker, arm):.2f}" for _, tag, arm, _, _ in ARMS),
            " / ".join(f"{excess(['pushes'], tag, tracker, arm):.2f}" for _, tag, arm, _, _ in ARMS),
            " / ".join(f"{excess(RECORDED, tag, tracker, arm):.2f}" for _, tag, arm, _, _ in ARMS),
            " / ".join(f"{effort('offsets', tag, tracker, arm):.2f}" for _, tag, arm, _, _ in ARMS),
        ]
        print(f"| {tracker_label(tracker)} | " + " | ".join(cells) + " |")
    print(
        "\nThe same with the squared torque (N^2 m^2 s) and the positive work (J): ESN #1 / ESN #2 / replay / progress"
    )
    for column in ("task_effort_n2m2s", "task_work_j"):
        print(f"  {column}:")
        for tracker in TRACKERS:
            cells = [
                " / ".join(f"{effort('nominal', tag, tracker, arm, column):.2f}" for _, tag, arm, _, _ in ARMS),
                " / ".join(f"{excess(['block'], tag, tracker, arm, column):.2f}" for _, tag, arm, _, _ in ARMS),
                " / ".join(f"{excess(['pushes'], tag, tracker, arm, column):.2f}" for _, tag, arm, _, _ in ARMS),
                " / ".join(f"{excess(RECORDED, tag, tracker, arm, column):.2f}" for _, tag, arm, _, _ in ARMS),
                " / ".join(f"{effort('offsets', tag, tracker, arm, column):.2f}" for _, tag, arm, _, _ in ARMS),
            ]
            print(f"  | {tracker_label(tracker)} | " + " | ".join(cells) + " |")
    print("\nPer log, the accumulated torque's excess (N m s): ESN #1 / ESN #2 / replay / progress replay")
    print("| Tracker | Log 000 | Log 001 | Log 002 |")
    print("| --- | --- | --- | --- |")
    for tracker in TRACKERS:
        cells = [" / ".join(f"{excess([s], tag, tracker, arm):.2f}" for _, tag, arm, _, _ in ARMS) for s in RECORDED]
        print(f"| {tracker_label(tracker)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
