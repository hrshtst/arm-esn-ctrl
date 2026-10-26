# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 017.

    uv run python reports/017-search-under-one-protocol/make_figures.py
    uv run python reports/017-search-under-one-protocol/make_figures.py --logs

From the copies of the runs under ``results/`` (configurations, run records, the
sweeps' ``seeds.csv``, the rule's ``esns.csv`` and ``settings.csv``, and the robot
runs' metrics), it prints the tables of the report and draws into
``results/summary``:

- ``grid.png``, ``grid_seeds10to19.png``: each setting's passing seeds, a panel per
  leak rate, the qualifying settings outlined (Sections 3.1 and 3.4);
- ``scenarios.png``: the share of the ESNs that fail in each scenario, and that pass,
  against the leak rate (Section 3.1);
- ``offsets.png``: from each offset start, the share of the ESNs passing the other
  three scenarios that fail from it (Section 3.2);
- ``following.png``: each setting's median ringing against its median following
  (Section 3.3);
- ``held_out.png``: each setting's passing seeds and median following, seeds 0 to 9
  against the held-out seeds 10 to 19 (Section 3.4).

With ``--logs``, it reads the robot runs' logs, which are not kept in Git, from the
runs under the storage root, and draws ``joints_esn1.png``, ``torques_esn1.png``,
``joints_esn2.png``, and ``torques_esn2.png``: each ESN's joint angles or torques
over time against the replay's and the progress replay's, a column per scenario, the
ESN's lines over the replays' (Section 3.6).
"""

from __future__ import annotations

import argparse
import csv
import functools
import math
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from skelarm import StateLog

from arm_esn_ctrl.demonstrations import resample_joint_angles
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.storage import resolve_run_path

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
# The take, as report 005 copied it: what the ESNs were trained on, and what the replays replay.
TAKE_RUN = REPORT.parent / "005-tuned-on-the-robot" / "data" / "20261008-170034-reach_manual_v2"
DT = 0.01  # the ESNs' period (s)
DPI = 90

LEAKS = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
RADII = [0.7, 0.9, 0.99, 1.2, 1.35, 1.5, 1.75]
SCALINGS = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
SEEDS_PER_SETTING = 10
OFFSETS = [-15, -10, -5, 0, 5, 10, 15]  # deg, in each joint
SCENARIOS = {"nominal": "undisturbed", "block": "the block", "pushes": "the pushes", "offsets": "the offset starts"}
DISTURBANCES = {"block": "the block", "push1": "push at 3 s", "push2": "push at 7 s", "push3": "push at 15 s"}
# The rule's runs: the search over seeds 0 to 9, the held-out seeds, and the size and ridge checks.
SELECTS = {
    "search": "select",
    "held_out": "select_seeds10to19",
    "size": "select_size",
    "ridge": "select_ridge",
}
# The chosen ESNs: their name, the tag of their runs, and their setting.
CHOSEN = {
    "ESN #1": ("esn1_lr0.03_sr1.2_is1.5_seed8", (0.03, 1.2, 1.5)),
    "ESN #2": ("esn2_lr0.03_sr1.2_is1_seed2", (0.03, 1.2, 1.0)),
}
# The robot runs' arms: name, the arm in the runs, and color. The replays are the same beside either ESN.
ARMS = [("ESN", "esn", "#2a78d6"), ("replay", "replay", "#eb6834"), ("progress replay", "progress", "#1baf7a")]
# The columns of the series over time: the scenario of the runs, the start drawn, and the title.
COLUMNS = [
    ("nominal", "demo_00_filtered", "undisturbed"),
    ("block", "demo_00_filtered", "under the block"),
    ("pushes", "demo_00_filtered", "under the pushes"),
    ("offsets", "demo_00_filtered +15,+15 deg", "from (+15°, +15°)"),
    ("offsets", "demo_00_filtered -15,-15 deg", "from (-15°, -15°)"),
]

TEXT_COLOR = "#52514e"
TITLE_COLOR = "#0b0b0b"
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAUGHT_COLOR = "#b9b7b0"
NONE_COLOR = "#f0efec"  # a heatmap cell of zero
# The sequential blue ramp, from its lightest step to its darkest.
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# The categorical slots, in order: the scenarios of scenarios.png.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
REPLAY_COLOR = "#eb6834"
# The layers of the lines over time: the ESN's arm on top, its reference (dashed) under it, then the progress
# replay's arm and reference, the replay's at the bottom.
ARM_ZORDERS = {"esn": 3.0, "progress": 2.2, "replay": 2.0}
REFERENCE_ZORDERS = {"esn": 2.5, "progress": 2.1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also draw the robot runs over time from their logs")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {
        "grid.png": plot_grid("search"),
        "grid_seeds10to19.png": plot_grid("held_out"),
        "scenarios.png": plot_scenarios(),
        "offsets.png": plot_offsets(),
        "following.png": plot_following(),
        "held_out.png": plot_held_out(),
    }
    if args.logs:
        for name, (tag, _) in CHOSEN.items():
            number = name[-1]
            for channel in ("joints", "torques"):
                figures[f"{channel}_esn{number}.png"] = plot_series(name, tag, channel)
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


def run_name(stem: str) -> str:
    """The name of the one run of a configuration."""
    matches = sorted(p.name for p in RESULTS.glob(f"*-{stem}") if p.is_dir())
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


def setting_of(row: dict[str, Any]) -> tuple[float, float, float]:
    return (row["leak_rate"], row["spectral_radius"], row["input_scaling"])


def esns(select: str) -> list[dict[str, Any]]:
    """The rule's rows of every ESN of one of its runs."""
    return read_csv(RESULTS / run_name(SELECTS[select]) / "esns.csv")


@functools.cache
def settings(select: str) -> dict[tuple[float, float, float], dict[str, Any]]:
    """The rule's rows of every setting of the grid, by leak rate, spectral radius, and input scaling."""
    return {setting_of(row): row for row in read_csv(RESULTS / run_name(SELECTS[select]) / "settings.csv")}


def failures(row: dict[str, Any]) -> list[tuple[str, tuple[int, int] | None, str]]:
    """How an ESN's runs fail: each run's scenario, its start offset (deg) for the offset starts, and how."""
    out = []
    for text in str(row["failures"]).split("; ") if row["failures"] else []:
        label, how = text.split(": ")
        if label in ("nominal", "block", "pushes"):
            out.append((label, None, how))
        elif label.endswith(" deg"):  # "demo_00_filtered +5,-10 deg"
            first, second = label.split()[-2].split(",")
            out.append(("offsets", (int(first), int(second)), how))
        else:  # the take's start, among the offset starts
            out.append(("offsets", (0, 0), how))
    return out


def failing_scenarios(row: dict[str, Any]) -> set[str]:
    return {scenario for scenario, _, _ in failures(row)}


def robot_rows(scenario: str, tag: str) -> list[dict[str, Any]]:
    return read_csv(RESULTS / run_name(f"{scenario}_robot_{tag}") / "metrics.csv")


def robot_run(scenario: str, tag: str, arm: str, origin: str = "demo_00_filtered") -> dict[str, Any]:
    return next(r for r in robot_rows(scenario, tag) if r["arm"] == arm and r["origin"] == origin)


def robot_log(scenario: str, tag: str, arm: str, origin: str) -> StateLog:
    """One arm's log from a robot run under the storage root."""
    start = int(robot_run(scenario, tag, arm, origin)["start"])
    run = resolve_run_path(f"results/{run_name(f'{scenario}_robot_{tag}')}")
    return StateLog.load(run / "pd_w20_z0.4" / f"{arm}_{start:02d}.sklog.npz")


def tip_force(log: StateLog) -> NDArray[np.float64]:
    """The size of the force at the tip over time: zero in an undisturbed run, whose log has no force."""
    try:
        return np.linalg.norm(log.channel("ext_force"), axis=1)
    except KeyError:
        return np.zeros(len(log.times))


@functools.cache
def robot_config(scenario: str) -> dict[str, Any]:
    with (RESULTS / run_name(f"{scenario}_robot_{CHOSEN['ESN #1'][0]}") / "config.toml").open("rb") as f:
        return tomllib.load(f)


@functools.cache
def taught() -> NDArray[np.float64]:
    """The take filtered at 2 Hz, what the ESNs were trained on, every 10 ms: its joint angles."""
    return resample_joint_angles(StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"), DT)[1]


# ---------------------------------------------------------------------------- figures


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def count_colormap(top: int) -> tuple[ListedColormap, BoundaryNorm]:
    """A discrete sequential map of the counts 0 to ``top``: zero near the surface, then the blue ramp."""
    ramp = LinearSegmentedColormap.from_list("blues", BLUES)
    colors = [NONE_COLOR] + [ramp(k / max(top - 1, 1)) for k in range(top)]
    return ListedColormap(colors), BoundaryNorm(np.arange(-0.5, top + 1.5), top + 1)


def ink_on(color: tuple[float, ...]) -> str:
    """The text color on a fill: dark on a light fill, white on a dark one."""
    r, g, b = color[:3]
    return TITLE_COLOR if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else "#ffffff"


def plot_grid(select: str) -> Figure:
    """Each setting's passing seeds: a panel per leak rate, spectral radius across and input scaling up."""
    rows = settings(select)
    cmap, norm = count_colormap(SEEDS_PER_SETTING)
    fig = Figure(figsize=(17, 11.5), facecolor=SURFACE_COLOR, layout="constrained")
    seeds = "0 to 9" if select == "search" else "10 to 19 (held out)"
    qualifying = sum(row["qualifies"] for row in rows.values())
    fig.suptitle(
        f"Seeds {seeds}: each setting's seeds that pass the rule's first point (of 10); outlined, the {qualifying}"
        " settings whose ten seeds all pass",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(3, 4, sharex=True, sharey=True)
    passing = Counter()
    for row in esns(select):
        passing[row["leak_rate"]] += row["passes"]
    for ax, leak in zip(axes.flat, LEAKS, strict=True):
        counts = np.array([[rows[(leak, r, s)]["passing_seeds"] for r in RADII] for s in SCALINGS])
        ax.imshow(counts, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        for i, scaling in enumerate(SCALINGS):
            for j, radius in enumerate(RADII):
                row = rows[(leak, radius, scaling)]
                count = int(row["passing_seeds"])
                text = str(count)
                for name, (_, setting) in CHOSEN.items():
                    if select == "search" and setting == (leak, radius, scaling):
                        text += f"\n{name[-2:]}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=ink_on(cmap(norm(count))))
                if row["qualifies"]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=TITLE_COLOR, linewidth=2))
        ax.set_title(f"leak rate {leak:g}: {passing[leak]} of 420 ESNs pass", color=TITLE_COLOR, fontsize=10)
        ax.set_xticks(range(len(RADII)), [f"{r:g}" for r in RADII])
        ax.set_yticks(range(len(SCALINGS)), [f"{s:g}" for s in SCALINGS])
        ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("spectral radius", color=TEXT_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("input scaling", color=TEXT_COLOR)
    return fig


def plot_scenarios() -> Figure:
    """The share of the ESNs at each leak rate that fail in each scenario, and that pass, for both seed sets."""
    fig = Figure(figsize=(14, 5.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The share of the 420 ESNs at each leak rate that fail in each scenario (an ESN can fail in several),"
        " and that pass in all four",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (select, title) in zip(
        axes, [("search", "seeds 0 to 9"), ("held_out", "seeds 10 to 19 (held out)")], strict=True
    ):
        style(ax)
        rows = esns(select)
        share: dict[str, list[float]] = {name: [] for name in [*SCENARIOS, "pass"]}
        for leak in LEAKS:
            at = [row for row in rows if row["leak_rate"] == leak]
            for scenario in SCENARIOS:
                share[scenario].append(100 * sum(scenario in failing_scenarios(row) for row in at) / len(at))
            share["pass"].append(100 * sum(row["passes"] for row in at) / len(at))
        for (scenario, words), color in zip(SCENARIOS.items(), SLOTS, strict=True):
            ax.plot(LEAKS, share[scenario], color=color, linewidth=2, marker="o", markersize=5, label=f"fail: {words}")
        ax.plot(
            LEAKS,
            share["pass"],
            color=TITLE_COLOR,
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=5,
            label="pass in all four",
        )
        ax.set_xscale("log")
        ax.set_xticks(LEAKS, [f"{leak:g}" for leak in LEAKS], minor=False, rotation=45)
        ax.minorticks_off()
        ax.set_xlim(0.008, 0.6)
        ax.set_ylim(0, 100)
        ax.set_title(title, color=TITLE_COLOR, fontsize=10)
        ax.set_xlabel("leak rate", color=TEXT_COLOR)
    axes[0].set_ylabel("share of the ESNs (%)", color=TEXT_COLOR)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def offset_failures(select: str) -> tuple[NDArray[np.float64], int]:
    """Of the ESNs that pass undisturbed, under the block, and under the pushes: the share (%) failing from each
    offset start (rows: joint 2's offset; columns: joint 1's), and how many such ESNs there are."""
    rows = [row for row in esns(select) if not failing_scenarios(row) & {"nominal", "block", "pushes"}]
    counts = np.zeros((len(OFFSETS), len(OFFSETS)))
    for row in rows:
        for scenario, offset, _ in failures(row):
            if scenario == "offsets" and offset is not None:
                counts[OFFSETS.index(offset[1]), OFFSETS.index(offset[0])] += 1
    return 100 * counts / len(rows), len(rows)


def plot_offsets() -> Figure:
    """From each offset start, the share of the ESNs passing the other three scenarios that fail from it."""
    fig = Figure(figsize=(13, 6.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "From each of the 49 starts: the share (%) of the ESNs passing undisturbed, under the block, and under"
        " the pushes that fail from it",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 2)
    shares = {select: offset_failures(select) for select in ("search", "held_out")}
    top = max(float(share.max()) for share, _ in shares.values())
    ramp = LinearSegmentedColormap.from_list("blues", ["#f3f7fd", *BLUES])
    image = None
    for ax, (select, title) in zip(
        axes, [("search", "seeds 0 to 9"), ("held_out", "seeds 10 to 19 (held out)")], strict=True
    ):
        share, count = shares[select]
        image = ax.imshow(share, cmap=ramp, vmin=0, vmax=top, origin="lower")
        for i in range(len(OFFSETS)):
            for j in range(len(OFFSETS)):
                ax.text(
                    j,
                    i,
                    f"{share[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=ink_on(ramp(share[i, j] / top)),
                )
        ax.add_patch(Rectangle((2.5, 2.5), 1, 1, fill=False, edgecolor=TITLE_COLOR, linewidth=1.5))
        ax.set_xticks(range(len(OFFSETS)), [f"{o:+d}" if o else "0" for o in OFFSETS])
        ax.set_yticks(range(len(OFFSETS)), [f"{o:+d}" if o else "0" for o in OFFSETS])
        ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{title}: {count} such ESNs", color=TITLE_COLOR, fontsize=10)
        ax.set_xlabel("joint 1's offset (deg)", color=TEXT_COLOR)
        ax.set_ylabel("joint 2's offset (deg)", color=TEXT_COLOR)
    assert image is not None
    bar = fig.colorbar(image, ax=axes, shrink=0.8, label="share failing from the start (%)")
    bar.ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    bar.ax.yaxis.label.set_color(TEXT_COLOR)
    return fig


def plot_following() -> Figure:
    """Each setting's median ringing against its median following, seeds 0 to 9, colored by its leak rate."""
    rows = settings("search")
    fig = Figure(figsize=(12, 7.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Seeds 0 to 9: each setting's ringing against its following after the block and the pushes, medians over"
        " its seeds;\nfilled, the settings whose ten seeds all pass; hollow, the rest",
        color=TITLE_COLOR,
    )
    ax = fig.subplots()
    style(ax)
    ramp = LinearSegmentedColormap.from_list("blues", BLUES[1:])
    guard = 10.70
    for k, leak in enumerate(LEAKS):
        color = ramp(k / (len(LEAKS) - 1))
        for qualifies in (False, True):
            points = [
                (row["following"], row["ringing_mm"])
                for (lr, _, _), row in rows.items()
                if lr == leak and bool(row["qualifies"]) == qualifies and np.isfinite(row["following"])
            ]
            if points:
                x, y = zip(*points, strict=True)
                ax.plot(
                    x,
                    y,
                    linestyle="none",
                    marker="o",
                    markersize=9 if qualifies else 5,
                    color=color,
                    markerfacecolor=color if qualifies else "none",
                    markeredgewidth=1.6 if qualifies else 1.2,
                    markeredgecolor=TITLE_COLOR if qualifies else color,
                    zorder=3 if qualifies else 2,
                    label=f"leak rate {leak:g}" if not qualifies else None,
                )
    ax.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=9,
        color=BLUES[4],
        markeredgecolor=TITLE_COLOR,
        markeredgewidth=1.6,
        label="ten seeds pass",
    )
    ax.plot([0], [guard], marker="D", markersize=8, color=REPLAY_COLOR, linestyle="none", label="the replay", zorder=4)
    ax.axhline(guard, color=REPLAY_COLOR, linewidth=1, linestyle="--", zorder=1)
    ax.annotate(
        "the guard: the replay's ringing, 10.70 mm",
        (0.02, guard),
        xytext=(0, 5),
        textcoords="offset points",
        fontsize=9,
        color=TEXT_COLOR,
    )
    for name, (_, setting) in CHOSEN.items():
        row = rows[setting]
        offset = (40, -55) if name == "ESN #1" else (-70, 50)
        ax.annotate(
            name,
            (row["following"], row["ringing_mm"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=TEXT_COLOR,
            arrowprops={"arrowstyle": "-", "color": TEXT_COLOR, "lw": 0.8},
        )
    ax.set_yscale("log")
    ax.set_yticks([3, 4, 6, 10, 20, 30], ["3", "4", "6", "10", "20", "30"])
    ax.minorticks_off()
    ax.set_xlim(-0.03, 1.0)
    ax.set_xlabel("following: median over the seeds of the mean gain under the block and the pushes", color=TEXT_COLOR)
    ax.set_ylabel("ringing (mm): median over the seeds of the mean after the four", color=TEXT_COLOR)
    ax.legend(loc="upper left", ncol=2, frameon=False, labelcolor=TEXT_COLOR, fontsize=9)
    return fig


def plot_held_out() -> Figure:
    """Each setting's passing seeds and median following: seeds 0 to 9 against the held-out seeds 10 to 19."""
    first, second = settings("search"), settings("held_out")
    fig = Figure(figsize=(14, 6.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("Each of the 504 settings, seeds 0 to 9 against the held-out seeds 10 to 19", color=TITLE_COLOR)
    left, right = fig.subplots(1, 2)
    counts = np.zeros((SEEDS_PER_SETTING + 1, SEEDS_PER_SETTING + 1), dtype=int)
    for key, row in first.items():
        counts[int(second[key]["passing_seeds"]), int(row["passing_seeds"])] += 1
    top = int(counts.max())
    ramp = LinearSegmentedColormap.from_list("blues", BLUES)
    shown = np.where(counts > 0, np.log(np.maximum(counts, 1)) / math.log(top), np.nan)
    left.imshow(shown, cmap=ramp, vmin=0, vmax=1, origin="lower")
    left.set_facecolor(NONE_COLOR)
    for i in range(SEEDS_PER_SETTING + 1):
        for j in range(SEEDS_PER_SETTING + 1):
            if counts[i, j]:
                left.text(
                    j, i, str(counts[i, j]), ha="center", va="center", fontsize=8, color=ink_on(ramp(shown[i, j]))
                )
    left.set_xticks(range(SEEDS_PER_SETTING + 1))
    left.set_yticks(range(SEEDS_PER_SETTING + 1))
    left.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR, length=0)
    for spine in left.spines.values():
        spine.set_visible(False)
    left.set_xlabel("passing seeds of 0 to 9", color=TEXT_COLOR)
    left.set_ylabel("passing seeds of 10 to 19", color=TEXT_COLOR)
    left.set_title("the number of settings with each pair of counts", color=TITLE_COLOR, fontsize=10)
    style(right)
    keys = [k for k in first if np.isfinite(first[k]["following"]) and np.isfinite(second[k]["following"])]
    for group, marker, label in (
        ("neither", "o", "qualifies with neither"),
        ("one", "o", "qualifies with one seed set"),
        ("both", "o", "qualifies with both"),
    ):
        chosen = [
            k
            for k in keys
            if {0: "neither", 1: "one", 2: "both"}[first[k]["qualifies"] + second[k]["qualifies"]] == group
        ]
        x = [first[k]["following"] for k in chosen]
        y = [second[k]["following"] for k in chosen]
        filled = group != "neither"
        right.plot(
            x,
            y,
            linestyle="none",
            marker=marker,
            markersize=7 if group == "both" else 5,
            color=BLUES[7] if group == "both" else BLUES[3],
            markerfacecolor=None if filled else "none",
            label=label,
            zorder=3 if filled else 2,
        )
    right.plot([0, 1], [0, 1], color=GRID_COLOR, linewidth=1.5, zorder=1)
    for name, (_, setting) in CHOSEN.items():
        offset = (40, -45) if name == "ESN #1" else (-70, 30)
        right.annotate(
            name,
            (first[setting]["following"], second[setting]["following"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color=TEXT_COLOR,
            arrowprops={"arrowstyle": "-", "color": TEXT_COLOR, "lw": 0.8},
        )
    right.set_xlim(0.3, 1.0)
    right.set_ylim(0.3, 1.0)
    right.set_aspect("equal")
    right.set_xlabel("median following, seeds 0 to 9", color=TEXT_COLOR)
    right.set_ylabel("median following, seeds 10 to 19", color=TEXT_COLOR)
    right.set_title("the median following", color=TITLE_COLOR, fontsize=10)
    right.legend(loc="upper left", frameon=False, labelcolor=TEXT_COLOR, fontsize=9)
    return fig


def plot_series(name: str, tag: str, channel: str) -> Figure:
    """One ESN's joint angles or torques over time against both replays': a column per scenario."""
    fig = Figure(figsize=(18, 9.4), facecolor=SURFACE_COLOR, layout="constrained")
    what = "Joint angles" if channel == "joints" else "Joint torques"
    fig.suptitle(
        f"{what} of {name}'s arm (blue), the replay's (orange), and the progress replay's (aqua), with the search's"
        " tracker"
        + (
            "; dashed: the references of the ESN and the progress replay; thick gray: the taught motion"
            if channel == "joints"
            else "; the peaks at an offset start are cut at ±6 N m"
        ),
        color=TITLE_COLOR,
    )
    axes = fig.subplots(3, len(COLUMNS), sharex="col", height_ratios=[0.55, 1.0, 1.0])
    q_taught = np.degrees(taught())
    t_taught = DT * np.arange(len(q_taught))
    logs = [{arm: robot_log(scenario, tag, arm, origin) for _, arm, _ in ARMS} for scenario, origin, _ in COLUMNS]
    limits = []
    for joint in range(2):
        key = "tau" if channel == "torques" else "q"
        values = [
            log.channel(key)[:, joint] * (1 if channel == "torques" else 180 / np.pi)
            for c in logs
            for log in c.values()
        ]
        if channel == "joints":
            values.append(q_taught[:, joint])
        low, high = float(min(v.min() for v in values)), float(max(v.max() for v in values))
        if channel == "torques":  # the start's peak at an offset start would flatten the rest
            low, high = max(low, -6.0), min(high, 6.0)
        limits.append((low - 0.05 * (high - low), high + 0.05 * (high - low)))
    top = max(float(tip_force(log).max()) for c in logs for log in c.values())
    for col, ((scenario, _, title), arms) in enumerate(zip(COLUMNS, logs, strict=True)):
        spans = disturbance_spans(robot_config(scenario).get("disturbance"))
        for row in range(3):
            ax = axes[row, col]
            style(ax)
            for onset, end in spans:
                ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
        axes[0, col].set_title(title, color=TITLE_COLOR, fontsize=10)
        for _, arm, color in ARMS:
            axes[0, col].plot(
                arms[arm].times, tip_force(arms[arm]), color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm]
            )
        axes[0, col].set_ylim(-0.05 * top, 1.1 * top)
        for joint in range(2):
            ax = axes[1 + joint, col]
            if channel == "joints":
                ax.plot(t_taught, q_taught[:, joint], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            for _, arm, color in ARMS:
                log = arms[arm]
                if channel == "joints":
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
                else:
                    ax.plot(
                        log.times, log.channel("tau")[:, joint], color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm]
                    )
            ax.set_ylim(*limits[joint])
            ax.set_xlim(-1.0, 20.0)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    axes[0, 0].set_ylabel("tip force (N)", color=TEXT_COLOR)
    unit = "deg" if channel == "joints" else "N m"
    for joint in range(2):
        axes[1 + joint, 0].set_ylabel(f"joint {joint + 1} ({unit})", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=color, linewidth=1.4, label=f"{name}'s arm" if arm == "esn" else arm_name)
        for arm_name, arm, color in ARMS
    ]
    if channel == "joints":
        handles += [
            Line2D([], [], color=TEXT_COLOR, linewidth=1.0, linestyle="--", label="its reference"),
            Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught motion"),
        ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- tables


def print_tables() -> None:
    print_grid()
    print_offsets()
    print_ranking()
    print_measures()
    print_held_out()
    print_checks()
    print_robot()


def print_grid() -> None:
    """Per leak rate: the ESNs that pass, how the others fail, the settings, and the following."""
    print("\nThe grid, per leak rate, seeds 0 to 9 (held-out seeds 10 to 19 in brackets):")
    print(
        "| Leak rate | ESNs passing (of 420) | Failing undisturbed / block / pushes / offsets | Settings: 10 / 9 seeds"
        " passing | Settings' median following |"
    )
    print("| ---: | ---: | --- | --- | --- |")
    for leak in LEAKS:
        cells = []
        for select in ("search", "held_out"):
            at = [row for row in esns(select) if row["leak_rate"] == leak]
            fail = [sum(s in failing_scenarios(row) for row in at) for s in SCENARIOS]
            rows = [row for key, row in settings(select).items() if key[0] == leak]
            ten = sum(row["passing_seeds"] == 10 for row in rows)
            nine = sum(row["passing_seeds"] == 9 for row in rows)
            following = [row["following"] for row in rows if np.isfinite(row["following"])]
            cells.append(
                (
                    sum(row["passes"] for row in at),
                    " / ".join(map(str, fail)),
                    f"{ten} / {nine}",
                    f"{min(following):.2f} to {max(following):.2f}",
                )
            )
        (a, b) = cells
        print(f"| {leak:g} | {a[0]} ({b[0]}) | {a[1]} ({b[1]}) | {a[2]} ({b[2]}) | {a[3]} |")
    for select in ("search", "held_out"):
        rows = esns(select)
        fails = Counter(how for row in rows for _, _, how in failures(row))
        runs = Counter(scenario for row in rows for scenario, _, _ in failures(row))
        print(
            f"{select}: {sum(r['passes'] for r in rows)} of {len(rows)} pass; failing runs by scenario {dict(runs)};"
            f" by how {dict(fails)}"
        )
    for name, axis, values in (("spectral radius", 1, RADII), ("input scaling", 2, SCALINGS)):
        rows = esns("search")
        shares = []
        for v in values:
            at = [row for row in rows if setting_of(row)[axis] == v]
            shares.append(f"{v:g}: {sum(r['passes'] for r in at)}/{len(at)}")
        print(f"Passing by {name}: {', '.join(shares)}")


def print_offsets() -> None:
    """What counting the offset starts for every ESN changes."""
    for select in ("search", "held_out"):
        rows = esns(select)
        three = [row for row in rows if not failing_scenarios(row) & {"nominal", "block", "pushes"}]
        by_setting: dict[tuple[float, float, float], list[bool]] = {}
        for row in rows:
            by_setting.setdefault(setting_of(row), []).append(
                not failing_scenarios(row) & {"nominal", "block", "pushes"}
            )
        without = sorted(k for k, passes in by_setting.items() if all(passes))
        qualifying = {k for k, row in settings(select).items() if row["qualifies"]}
        lost = [k for k in without if k not in qualifying]
        share, _ = offset_failures(select)
        failing_starts = Counter(
            len([f for f in failures(row) if f[0] == "offsets"]) for row in three if not row["passes"]
        )
        print(
            f"\n{select}: {len(three)} ESNs pass the three scenarios,"
            f" {sum(r['passes'] for r in rows)} also the offsets;"
            f" {len(without)} settings would qualify without the offsets, {len(qualifying)} with them; lost: {lost}"
        )
        print(
            f"  failing starts per ESN failing only there: median {np.median(list(failing_starts.elements())):.0f};"
            f" worst start {share.max():.0f}% at joint 2 {OFFSETS[int(np.argmax(share) // 7)]:+d}, joint 1"
            f" {OFFSETS[int(np.argmax(share) % 7)]:+d}; the demonstrated start {share[3, 3]:.1f}%"
        )
        print(
            "  share failing by joint 2's offset:",
            ", ".join(f"{o:+d}: {share[i].mean():.0f}%" for i, o in enumerate(OFFSETS)),
        )
        print(
            "  share failing by joint 1's offset:",
            ", ".join(f"{o:+d}: {share[:, j].mean():.0f}%" for j, o in enumerate(OFFSETS)),
        )


def ranked(select: str) -> list[tuple[tuple[float, float, float], dict[str, Any]]]:
    return sorted(((k, r) for k, r in settings(select).items() if r["rank"] != ""), key=lambda kr: kr[1]["rank"])


def print_ranking() -> None:
    """The qualifying settings, ranked, with their held-out outcome."""
    held = settings("held_out")
    print("\nThe qualifying settings, seeds 0 to 9, ranked by the median following:")
    print(
        "| Rank | Leak rate | Spectral radius | Input scaling | Following | Block / pushes at 3, 7, 15 s | Ringing |"
        " Return | Undisturbed path RMSE | Typical seed | Held out: passing seeds, following |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |")
    for key, row in ranked("search"):
        per = " / ".join(f"{row[f'following_{d}']:.2f}" for d in DISTURBANCES)
        h = held[key]
        print(
            f"| {int(row['rank'])} | {key[0]:g} | {key[1]:g} | {key[2]:g} | {row['following']:.2f} | {per} |"
            f" {row['ringing_mm']:.2f} mm | {row['return_s']:.2f} s | {row['path_rmse_mm']:.2f} mm |"
            f" {int(row['typical_seed'])} | {int(h['passing_seeds'])}, {h['following']:.2f} |"
        )


def print_measures() -> None:
    """The qualifying settings' best by the measures the earlier searches ranked by."""
    rows = ranked("search")
    print("\nThe qualifying settings' best by each measure (seeds 0 to 9):")
    for measure, largest in (("following", True), ("path_rmse_mm", False), ("ringing_mm", False), ("return_s", False)):
        best = sorted(rows, key=lambda kr: -kr[1][measure] if largest else kr[1][measure])[:2]
        print(f"  {measure}: " + "; ".join(f"{k} ({r[measure]:.2f}, rank {int(r['rank'])})" for k, r in best))


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def print_held_out() -> None:
    """Whether the qualifying settings qualify again with the held-out seeds, and keep their order."""
    first, second = settings("search"), settings("held_out")
    qa = {k for k, r in first.items() if r["qualifies"]}
    qb = {k for k, r in second.items() if r["qualifies"]}
    print(f"\nHeld out: {len(qb)} settings qualify, {len(qa & qb)} with both seed sets: {sorted(qa & qb)}")
    print(
        "| Rank (held out) | Leak rate | Spectral radius | Input scaling | Following | Ringing | Return |"
        " Path RMSE | Typical seed | Seeds 0 to 9: passing seeds, rank |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for key, row in ranked("held_out"):
        f = first[key]
        rank = int(f["rank"]) if f["rank"] != "" else "-"
        print(
            f"| {int(row['rank'])} | {key[0]:g} | {key[1]:g} | {key[2]:g} | {row['following']:.2f} |"
            f" {row['ringing_mm']:.2f} mm | {row['return_s']:.2f} s | {row['path_rmse_mm']:.2f} mm |"
            f" {int(row['typical_seed'])} | {int(f['passing_seeds'])}, {rank} |"
        )
    pooled = sorted(
        (k for k in first if first[k]["passing_seeds"] + second[k]["passing_seeds"] >= 19),
        key=lambda k: -(first[k]["passing_seeds"] + second[k]["passing_seeds"]),
    )
    print(
        "Settings passing 19 or 20 of the 20 seeds:",
        ", ".join(f"{k} {int(first[k]['passing_seeds'] + second[k]['passing_seeds'])}" for k in pooled),
    )
    for label, keys in (
        (
            "all settings",
            [k for k in first if np.isfinite(first[k]["following"]) and np.isfinite(second[k]["following"])],
        ),
        ("qualifying with either", sorted(qa | qb)),
        ("qualifying with both", sorted(qa & qb)),
    ):
        rho = spearman([first[k]["following"] for k in keys], [second[k]["following"] for k in keys])
        diffs = [abs(first[k]["following"] - second[k]["following"]) for k in keys]
        print(f"  following, {label} ({len(keys)}): rank correlation {rho:.2f}, largest change {max(diffs):.2f}")
    pa = [first[k]["passing_seeds"] for k in first]
    pb = [second[k]["passing_seeds"] for k in first]
    print(
        f"  passing seeds: correlation {np.corrcoef(pa, pb)[0, 1]:.2f}; held out pass"
        f" {sum(r['passes'] for r in esns('held_out'))} ESNs"
    )
    for name, (_, setting) in CHOSEN.items():
        failing = [r for r in esns("held_out") if setting_of(r) == setting and not r["passes"]]
        print(
            f"  {name}'s setting held out: {int(second[setting]['passing_seeds'])} of 10;",
            "; ".join(
                f"seed {int(r['seed'])}: {len(failures(r))} runs,"
                f" {sorted({f[1] for f in failures(r) if f[1] is not None})}"
                for r in failing
            ),
        )


def print_checks() -> None:
    """The size and ridge checks at ESN #1's setting."""
    for check, column, label in (("size", "n_neurons", "Neurons"), ("ridge", "ridge", "Ridge")):
        rows = read_csv(RESULTS / run_name(SELECTS[check]) / "settings.csv")
        print(f"\nThe {check} check at ESN #1's setting, seeds 0 to 9:")
        print(
            f"| {label} | Passing seeds | Following | Ringing | Return | Undisturbed path RMSE | How the others fail |"
        )
        print("| ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in rows:
            failing = [r for r in esns(check) if r[column] == row[column] and not r["passes"]]
            how = Counter()
            for r in failing:
                kinds = failing_scenarios(r)
                starts = sum(f[0] == "offsets" for f in failures(r))
                how[" + ".join(sorted(kinds)) + (f" ({starts} starts)" if kinds == {"offsets"} else "")] += 1
            print(
                f"| {row[column]:g} | {int(row['passing_seeds'])} | {row['following']:.2f} |"
                f" {row['ringing_mm']:.2f} mm |"
                f" {row['return_s']:.2f} s | {row['path_rmse_mm']:.2f} mm | {dict(how)} |"
            )


def print_robot() -> None:
    """The chosen ESNs on the robot against both replays, with the search's tracker."""
    tags = {name: tag for name, (tag, _) in CHOSEN.items()}
    arms = [
        ("ESN #1", tags["ESN #1"], "esn"),
        ("ESN #2", tags["ESN #2"], "esn"),
        ("replay", tags["ESN #1"], "replay"),
        ("progress replay", tags["ESN #1"], "progress"),
    ]
    for scenario in SCENARIOS:  # the replays do not depend on the ESN beside them
        for arm in ("replay", "progress"):
            a = [r for r in robot_rows(scenario, tags["ESN #1"]) if r["arm"] == arm]
            b = [r for r in robot_rows(scenario, tags["ESN #2"]) if r["arm"] == arm]
            assert [{k: str(v) for k, v in r.items()} for r in a] == [{k: str(v) for k, v in r.items()} for r in b]
    print("\nOn the robot, from the take's start: arrival, path RMSE; pass by the rule's first point")
    print("| Arm | Undisturbed | Under the block | Under the pushes |")
    print("| --- | --- | --- | --- |")
    for name, tag, arm in arms:
        cells = []
        for scenario in ("nominal", "block", "pushes"):
            r = robot_run(scenario, tag, arm)
            cells.append(
                f"{r['arrival_time_s']:.2f} s, {1000 * r['taught_path_rmse_m']:.2f} mm,"
                f" ends {1000 * r['final_distance_m']:.1f} mm away{', leaves the goal' if r['left_goal'] else ''}"
            )
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\nAfter each disturbance: ESN #1 / ESN #2 / replay / progress replay")
    for column, label, scale, digits in (
        ("following_gain", "following", 1, 2),
        ("ringing_m", "ringing (mm)", 1000, 1),
        ("return_time_s", "back within 5 mm (s)", 1, 2),
        ("off_path_m", "farthest (mm)", 1000, 0),
    ):
        cells = []
        for scenario, suffixes in (("block", [""]), ("pushes", ["_1", "_2", "_3"])):
            for suffix in suffixes:
                vals = [robot_run(scenario, tag, arm)[f"{column}{suffix}"] * scale for _, tag, arm in arms]
                cells.append(" / ".join("-" if not np.isfinite(v) else f"{v:.{digits}f}" for v in vals))
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("\nUnder the block: peak torque (N m), reference lead at the release (%)")
    for name, tag, arm in arms:
        r = robot_run("block", tag, arm)
        print(f"  {name}: {r['peak_torque_nm']:.1f} N m, lead {100 * r['reference_lead']:.1f}%")
    print("\nFrom the 49 offset starts: pass, path RMSE median, first step median, peak torque median / largest")
    radius = 0.02
    for name, tag, arm in arms:
        rows = [r for r in robot_rows("offsets", tag) if r["arm"] == arm]
        good = sum(
            bool(
                np.isfinite(r["arrival_time_s"])
                and r["final_distance_m"] <= radius
                and not r["left_goal"]
                and r["success"]
            )
            for r in rows
        )
        moved = [r for r in rows if str(r["origin"]).endswith(" deg")]
        print(
            f"  {name}: {good} of {len(rows)}; {1000 * np.median([r['taught_path_rmse_m'] for r in rows]):.1f} mm;"
            f" first step {1000 * np.median([r['first_step_m'] for r in moved]):.0f} mm;"
            f" {np.median([r['peak_torque_nm'] for r in moved]):.0f} /"
            f" {max(r['peak_torque_nm'] for r in moved):.0f} N m"
        )
    print("\nHow hard the motors work over the task: |tau| N m s, tau^2 N^2 m^2 s, positive work J (offsets: medians)")
    print("| Arm | Undisturbed | Under the block | Under the pushes | From the offset starts |")
    print("| --- | --- | --- | --- | --- |")
    for name, tag, arm in arms:
        cells = []
        for scenario in SCENARIOS:
            rows = [r for r in robot_rows(scenario, tag) if r["arm"] == arm]
            if scenario == "offsets":
                rows = [r for r in rows if str(r["origin"]).endswith(" deg")]
            values = [np.median([r[c] for r in rows]) for c in ("task_torque_nms", "task_effort_n2m2s", "task_work_j")]
            cells.append(f"{values[0]:.2f}, {values[1]:.2f}, {values[2]:.2f}")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
