# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 021.

    uv run python reports/021-search-on-a-four-link-arm/make_figures.py
    uv run python reports/021-search-on-a-four-link-arm/make_figures.py --logs

From the copies of the runs under ``results/`` (configurations, run records, the
sweeps' ``seeds.csv``, the rule's ``esns.csv`` and ``settings.csv``, and the robot
runs' metrics), it prints the tables of the report and draws into
``results/summary``:

- ``grid_first_rule.png``, ``grid_second_rule.png``: each setting's passing seeds
  under each rule, a panel per leak rate, the qualifying settings outlined (Sections
  3.1 and 3.5);
- ``scenarios.png``: the share of the ESNs that fail in each scenario, and that pass,
  against the leak rate (Section 3.1);
- ``by_scaling.png``: the passing ESNs by input scaling, on this arm and on the
  two-link arm (Section 3.3);
- ``offsets.png``: from each of the 35 offset starts, the share of the ESNs passing
  the other three scenarios that fail from it (Section 3.2);
- ``following.png``: each ESN's following against its input scaling and its leak rate
  (Section 3.4);
- ``gap.png``: the block gap of every ESN and of those passing the first rule (Section
  3.5).

With ``--logs``, it reads the robot runs' logs and the sweeps' ``runs.csv``, which are
not kept in Git, from the runs under the storage root. It prints the tables that need
them and draws ``gap_series.png``, the reference minus the arm in each joint around
the block (Section 3.6), and ``joints_esn1.png``, ``torques_esn1.png``,
``joints_esn2.png``, and ``torques_esn2.png``: each ESN's joint angles or torques over
time against the replay's and the progress replay's, a column per scenario, the ESN's
lines over the replays' (Section 3.7).
"""

from __future__ import annotations

import argparse
import csv
import functools
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
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.demonstrations import resample_joint_angles
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.storage import resolve_run_path
from arm_esn_ctrl.tracking import mass_matrix_at

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
TWO_LINK_RESULTS = REPORT.parent / "017-search-under-one-protocol" / "results"
# The take: what the ESNs were trained on, and what the replays replay.
TAKE_RUN = REPORT / "data" / "20261103-134628-reach_manual_four_dof"
DT = 0.01  # the ESNs' period (s)
DPI = 90

LEAKS = [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
RADII = [0.7, 0.9, 0.99, 1.2, 1.35, 1.5, 1.75]
SCALINGS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]  # the first two are the extension's
SEEDS_PER_SETTING = 10
OFFSETS = [-15, 0, 15]  # deg, in joints 3 and 4; joints 1 and 2 take 0 and +15
SCENARIOS = {"nominal": "undisturbed", "block": "the block", "pushes": "the pushes", "offsets": "the offset starts"}
# The rule's runs: over the grid, over the grid and the extension, the second rule, and its held-out seeds.
SELECTS = {
    "grid": "select",
    "extended": "select_extended",
    "second": "select_following",
    "held_out": "select_following_seeds10to19",
}
# The ESNs: their name, their runs' tag, and leak rate, spectral radius, input scaling, and seed.
CHOSEN = {
    "ESN #1": ("following_esn1_lr0.015_sr1.2_is0.05_seed0", (0.015, 1.2, 0.05, 0.0)),
    "ESN #2": ("following_esn2_lr0.02_sr1.35_is0.05_seed7", (0.02, 1.35, 0.05, 7.0)),
}
EARLIER = {
    "Earlier ESN #1": ("esn1_lr0.01_sr1.5_is0.02_seed7", (0.01, 1.5, 0.02, 7.0)),
    "Earlier ESN #2": ("esn2_lr0.02_sr1.5_is0.02_seed8", (0.02, 1.5, 0.02, 8.0)),
}
TWO_LINK = {
    "Two-link ESN #1": ("esn1_lr0.03_sr1.2_is1.5_seed8", (0.03, 1.2, 1.5, 8.0)),
    "Two-link ESN #2": ("esn2_lr0.03_sr1.2_is1_seed2", (0.03, 1.2, 1.0, 2.0)),
}
# The robot runs' arms: name, the arm in the runs, and color. The replays are the same beside either ESN.
ARMS = [("ESN", "esn", "#2a78d6"), ("replay", "replay", "#eb6834"), ("progress replay", "progress", "#1baf7a")]
TAKE_START = "demo_00_filtered"
NEAREST, FARTHEST = f"{TAKE_START} +15,+15,+15,+15 deg", f"{TAKE_START} +0,+0,-15,-15 deg"
# The columns of the series over time: the scenario of the runs, the start drawn, and the title.
COLUMNS = [
    ("nominal", TAKE_START, "undisturbed"),
    ("block", TAKE_START, "under the block"),
    ("pushes", TAKE_START, "under the pushes"),
    ("offsets", NEAREST, "from +15° in every joint"),
    ("offsets", FARTHEST, "from -15° in joints 3 and 4"),
]
LOWER = np.radians([-20.0, 0.0, 0.0, -20.0])  # the joint limits
UPPER = np.radians([30.0, 120.0, 150.0, 90.0])
BASE_LENGTH, LENGTHS = 0.05, np.array([0.1, 0.25, 0.25, 0.15])  # m
TRACKER = "pd_w20_z0.4"
ARM_CONFIG = REPORT.parent.parent / "experiments" / "demonstrations" / "reach_manual_four_dof.toml"
OMEGA, ZETA, HELD = 20.0, 0.4, 0.01  # the search's tracker: rad/s, its damping ratio, and the held torque's period (s)

TEXT_COLOR = "#52514e"
TITLE_COLOR = "#0b0b0b"
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAUGHT_COLOR = "#b9b7b0"
NONE_COLOR = "#f0efec"  # a heatmap cell of zero
FAILING_COLOR = "#c9c7c0"
# The sequential blue ramp, from its lightest step to its darkest.
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# The categorical slots, in order: the scenarios of scenarios.png.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ESN_COLORS = {"ESN #1": "#1c5cab", "ESN #2": "#6da7ec"}
REPLAY_COLOR, PROGRESS_COLOR, EARLIER_COLOR, TWO_LINK_COLOR = "#eb6834", "#1baf7a", "#52514e", "#7a4ca8"
# The layers of the lines over time: the ESN's arm on top, its reference (dashed) under it, then the progress
# replay's arm and reference, the replay's at the bottom.
ARM_ZORDERS = {"esn": 3.0, "progress": 2.2, "replay": 2.0}
REFERENCE_ZORDERS = {"esn": 2.5, "progress": 2.1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also read the runs' logs under the storage root")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {
        "grid_first_rule.png": plot_grid("extended"),
        "grid_second_rule.png": plot_grid("second"),
        "scenarios.png": plot_scenarios(),
        "by_scaling.png": plot_by_scaling(),
        "offsets.png": plot_offsets(),
        "following.png": plot_following(),
        "gap.png": plot_gap(),
    }
    if args.logs:
        figures["gap_series.png"] = plot_gap_series()
        for name, (tag, _) in CHOSEN.items():
            number = name[-1]
            for channel in ("joints", "torques"):
                figures[f"{channel}_esn{number}.png"] = plot_series(name, tag, channel)
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=DPI)
    print_tables(logs=args.logs)
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


def setting_of(row: dict[str, Any]) -> tuple[float, float, float]:
    return (row["leak_rate"], row["spectral_radius"], row["input_scaling"])


def esn_of(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (*setting_of(row), row["seed"])


def esns(select: str) -> list[dict[str, Any]]:
    """The rule's rows of every ESN of one of its runs."""
    return read_csv(RESULTS / run_name(SELECTS[select]) / "esns.csv")


def two_link_esns() -> list[dict[str, Any]]:
    """The rule's rows of report 017's ESNs, seeds 0 to 9."""
    return read_csv(TWO_LINK_RESULTS / run_name("select", TWO_LINK_RESULTS) / "esns.csv")


@functools.cache
def settings(select: str) -> dict[tuple[float, float, float], dict[str, Any]]:
    """The rule's rows of every setting of its grid, by leak rate, spectral radius, and input scaling."""
    return {setting_of(row): row for row in read_csv(RESULTS / run_name(SELECTS[select]) / "settings.csv")}


def failures(row: dict[str, Any]) -> list[tuple[str, tuple[int, ...] | None, str]]:
    """How an ESN's runs fail: each run's scenario, its start offset (deg) for the offset starts, and how."""
    out: list[tuple[str, tuple[int, ...] | None, str]] = []
    for text in str(row["failures"]).split("; ") if row["failures"] else []:
        label, how = text.split(": ")
        if label in ("nominal", "block", "pushes"):
            out.append((label, None, how))
        elif label.endswith(" deg"):  # "demo_00_filtered +0,+0,-15,+15 deg"
            out.append(("offsets", tuple(int(angle) for angle in label.split()[-2].split(",")), how))
        else:  # the take's start, among the offset starts
            out.append(("offsets", (0, 0, 0, 0), how))
    return out


def failing_scenarios(row: dict[str, Any]) -> set[str]:
    return {scenario for scenario, _, _ in failures(row)}


def robot_rows(scenario: str, tag: str, results: Path = RESULTS) -> list[dict[str, Any]]:
    return read_csv(results / run_name(f"{scenario}_robot_{tag}", results) / "metrics.csv")


def robot_run(scenario: str, tag: str, arm: str, origin: str = TAKE_START, results: Path = RESULTS) -> dict[str, Any]:
    return next(r for r in robot_rows(scenario, tag, results) if r["arm"] == arm and r["origin"] == origin)


def robot_log(scenario: str, tag: str, arm: str, origin: str = TAKE_START, results: Path = RESULTS) -> StateLog:
    """One arm's log from a robot run under the storage root."""
    start = int(robot_run(scenario, tag, arm, origin, results)["start"])
    run = resolve_run_path(f"results/{run_name(f'{scenario}_robot_{tag}', results)}")
    return StateLog.load(run / TRACKER / f"{arm}_{start:02d}.sklog.npz")


def tip_force(log: StateLog) -> NDArray[np.float64]:
    """The size of the force at the tip over time: zero in an undisturbed run, whose log has no force."""
    try:
        return np.linalg.norm(log.channel("ext_force"), axis=1)
    except KeyError:
        return np.zeros(len(log.times))


@functools.cache
def robot_config(scenario: str, tag: str = CHOSEN["ESN #1"][0], results: Path = RESULTS) -> dict[str, Any]:
    with (results / run_name(f"{scenario}_robot_{tag}", results) / "config.toml").open("rb") as f:
        return tomllib.load(f)


@functools.cache
def taught() -> NDArray[np.float64]:
    """The take filtered at 2 Hz, what the ESNs were trained on, every 10 ms: its joint angles."""
    return resample_joint_angles(StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"), DT)[1]


def hand(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """The hand's positions for rows of the four joint angles."""
    angles = np.cumsum(q, axis=-1)
    return np.stack(
        [BASE_LENGTH + np.sum(LENGTHS * np.cos(angles), axis=-1), np.sum(LENGTHS * np.sin(angles), axis=-1)], axis=-1
    )


def distance_to_taught_path(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Each point's distance from the taught hand path."""
    path = hand(taught())
    start, stop = path[:-1][None], path[1:][None]
    p = points[:, None]
    along = stop - start
    denominator = np.maximum(np.sum(along * along, axis=-1), 1e-12)
    share = np.clip(np.sum((p - start) * along, axis=-1) / denominator, 0, 1)
    return np.linalg.norm(p - (start + share[..., None] * along), axis=-1).min(axis=1)


def detour(log: StateLog) -> float:
    """How much farther from the taught hand path the hand goes than where it starts (m)."""
    times = np.arange(0.0, log.times[-1], DT)
    q = np.column_stack([np.interp(times, log.times, log.channel("q")[:, j]) for j in range(4)])
    distance = distance_to_taught_path(hand(q)[::2])
    return float(distance.max() - distance[0])


def joint_following(log: StateLog, replay: StateLog, onset: float, end: float) -> tuple[NDArray[np.float64], float]:
    """The following gain in joint space, from a disturbance's onset to 0.5 s past its end: per joint, and pooled."""
    t, q, reference = log.times, log.channel("q"), log.channel("q_ref")
    motion = np.column_stack([np.interp(t, replay.times, replay.channel("q_ref")[:, j]) for j in range(q.shape[1])])
    window = (t > onset - 1e-9) & (t < end + 0.5 - 1e-9)
    arm, ref = (q - motion)[window], (reference - motion)[window]
    arm, ref = arm - arm[0], ref - ref[0]
    return (ref * arm).sum(axis=0) / (arm * arm).sum(axis=0), float((ref * arm).sum() / (arm * arm).sum())


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
    """Each setting's passing seeds under one rule: a panel per leak rate, spectral radius across, input scaling up."""
    rows = settings(select)
    marked = EARLIER if select == "extended" else CHOSEN
    qualifying = sum(row["qualifies"] for row in rows.values())
    words = "the first rule" if select == "extended" else "the second rule (a block gap of 0.5 or less)"
    pair = "the earlier pair" if select == "extended" else "ESN #1 and ESN #2"
    fig = Figure(figsize=(17, 13.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        f"Seeds 0 to 9: each setting's seeds that pass {words} (of 10); outlined, the {qualifying} qualifying"
        f" setting{'s' if qualifying != 1 else ''}\n#1, #2: the settings of {pair}; below the white line, the"
        " extension's input scalings",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(3, 4, sharex=True, sharey=True)
    passing: Counter[float] = Counter()
    for row in esns(select):
        passing[row["leak_rate"]] += row["passes"]
    cmap, norm = count_colormap(SEEDS_PER_SETTING)
    for ax, leak in zip(axes.flat, LEAKS, strict=True):
        counts = np.array([[rows[(leak, r, s)]["passing_seeds"] for r in RADII] for s in SCALINGS])
        ax.imshow(counts, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        for i, scaling in enumerate(SCALINGS):
            for j, radius in enumerate(RADII):
                row = rows[(leak, radius, scaling)]
                count = int(row["passing_seeds"])
                text = str(count)
                for name, (_, esn) in marked.items():
                    if esn[:3] == (leak, radius, scaling):
                        text += f"\n{name[-2:]}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=ink_on(cmap(norm(count))))
                if row["qualifies"]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=TITLE_COLOR, linewidth=2))
        ax.axhline(1.5, color="#ffffff", linewidth=2)
        ax.set_xticks(range(len(RADII)), [f"{r:g}" for r in RADII])
        ax.set_yticks(range(len(SCALINGS)), [f"{s:g}" for s in SCALINGS])
        ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"leak rate {leak:g}: {passing[leak]} of 560 ESNs pass", color=TITLE_COLOR, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("spectral radius", color=TEXT_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("input scaling", color=TEXT_COLOR)
    return fig


def plot_scenarios() -> Figure:
    """The share of the ESNs at each leak rate that fail in each scenario, and that pass, under the first rule."""
    fig = Figure(figsize=(14, 5.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The share of the ESNs at each leak rate that fail in each scenario (an ESN can fail in several), and that"
        " pass in all four, under the first rule",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (select, title) in zip(
        axes,
        [("grid", "the grid: 420 ESNs per leak rate"), ("extended", "the grid and the extension: 560 per leak rate")],
        strict=True,
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


def passing_by_scaling() -> tuple[list[int], list[int]]:
    """The ESNs that pass the first rule per input scaling: over both grids here, and on the two-link arm."""
    four = [sum(row["passes"] for row in esns("extended") if row["input_scaling"] == s) for s in SCALINGS]
    two = [sum(row["passes"] for row in two_link_esns() if row["input_scaling"] == s) for s in SCALINGS[2:]]
    return four, two


def plot_by_scaling() -> Figure:
    """The ESNs passing the first rule against the input scaling, on both arms."""
    four, two = passing_by_scaling()
    fig = Figure(figsize=(12, 4.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("The ESNs that pass the first rule, of 840 per input scaling", color=TITLE_COLOR)
    left, right = fig.subplots(1, 2, width_ratios=[8, 6], sharey=False)
    colors = ["#104281"] * 2 + ["#2a78d6"] * 6
    for ax, counts, labels, bar_colors, title in (
        (left, four, SCALINGS, colors, "this arm: the extension (dark) and the grid"),
        (right, two, SCALINGS[2:], ["#52514e"] * 6, "the two-link arm (report 017's grid)"),
    ):
        style(ax)
        ax.grid(axis="x", visible=False)
        bars = ax.bar(range(len(labels)), counts, color=bar_colors, width=0.7)
        ax.bar_label(bars, color=TEXT_COLOR, padding=2)
        ax.set_xticks(range(len(labels)), [f"{s:g}" for s in labels])
        ax.set_ylim(0, 1.15 * max(counts))
        ax.set_xlabel("input scaling", color=TEXT_COLOR)
        ax.set_title(title, color=TITLE_COLOR, fontsize=10)
    left.set_ylabel("ESNs", color=TEXT_COLOR)
    return fig


def offset_failures() -> tuple[dict[tuple[int, ...], float], int]:
    """Of the grid's ESNs that pass undisturbed, under the block, and under the pushes: the share (%) failing from
    each start, and how many such ESNs there are."""
    rows = [row for row in esns("grid") if not failing_scenarios(row) & {"nominal", "block", "pushes"}]
    counts: Counter[tuple[int, ...]] = Counter()
    for row in rows:
        for start in {start for scenario, start, _ in failures(row) if scenario == "offsets" and start is not None}:
            counts[start] += 1
    return {start: 100 * count / len(rows) for start, count in counts.items()}, len(rows)


def plot_offsets() -> Figure:
    """From each offset start, the share of the ESNs passing the rest that fail: a panel per joint 1's and 2's."""
    share, count = offset_failures()
    fig = Figure(figsize=(13, 4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        f"Of the {count} ESNs of the grid passing undisturbed, under the block, and under the pushes: the share (%)"
        " failing from each offset start",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 4, sharey=True)
    ramp = LinearSegmentedColormap.from_list("blues", [NONE_COLOR, *BLUES])
    top = max(share.values())
    image = None
    for ax, (first, second) in zip(axes, [(0, 0), (0, 15), (15, 0), (15, 15)], strict=True):
        cells = np.array([[share.get((first, second, c, d), 0.0) for d in OFFSETS] for c in OFFSETS])
        image = ax.imshow(cells, cmap=ramp, vmin=0, vmax=top, origin="lower", aspect="equal")
        for i, third in enumerate(OFFSETS):
            for j, fourth in enumerate(OFFSETS):
                if (first, second, third, fourth) == (0, 0, 0, 0):
                    ax.text(j, i, "the\ntake", ha="center", va="center", fontsize=9, color=TEXT_COLOR)
                else:
                    color = ink_on(ramp(cells[i, j] / top))
                    ax.text(j, i, f"{cells[i, j]:.0f}", ha="center", va="center", color=color)
        ax.set_xticks(range(3), [f"{o:+d}" for o in OFFSETS])
        ax.set_yticks(range(3), [f"{o:+d}" for o in OFFSETS])
        ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"joint 1 {first:+d}°, joint 2 {second:+d}°", color=TITLE_COLOR, fontsize=10)
        ax.set_xlabel("joint 4's offset (deg)", color=TEXT_COLOR)
    axes[0].set_ylabel("joint 3's offset (deg)", color=TEXT_COLOR)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, label="failing ESNs (%)")
    return fig


def plot_following() -> Figure:
    """Every ESN's following against its input scaling and its leak rate, the passing ones and the earlier pair."""
    rows = esns("extended")
    following = np.array([row["following"] for row in rows])
    passes = np.array([row["passes"] for row in rows], dtype=bool)
    fig = Figure(figsize=(14, 6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Each ESN's following (the mean over the block and the three pushes), by the first rule: gray fails, blue"
        " passes; stars, the earlier pair; diamond, the two-link ESN #1",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 2, sharey=True)
    rng = np.random.default_rng(0)
    by_esn = {esn_of(row): row for row in rows}
    two_link = next(row for row in two_link_esns() if esn_of(row) == TWO_LINK["Two-link ESN #1"][1])
    for ax, axis, values, label in ((axes[0], 2, SCALINGS, "input scaling"), (axes[1], 0, LEAKS, "leak rate")):
        style(ax)
        index = {v: i for i, v in enumerate(values)}
        x = np.array([index[setting_of(row)[axis]] for row in rows]) + rng.uniform(-0.28, 0.28, len(rows))
        ax.scatter(x[~passes], following[~passes], s=5, color=FAILING_COLOR, alpha=0.5, zorder=1)
        ax.scatter(x[passes], following[passes], s=18, color="#2a78d6", alpha=0.8, zorder=2)
        ax.axhline(0.0, color=REPLAY_COLOR, linewidth=1.5, zorder=1)
        for _, esn in EARLIER.values():
            ax.plot(
                index[esn[axis]],
                by_esn[esn]["following"],
                "*",
                color="#eda100",
                markeredgecolor=TITLE_COLOR,
                markersize=15,
                zorder=4,
            )
        if axis == 0 and two_link["leak_rate"] in index:
            ax.plot(
                index[two_link["leak_rate"]],
                two_link["following"],
                "D",
                color=TWO_LINK_COLOR,
                markeredgecolor=TITLE_COLOR,
                markersize=9,
                zorder=4,
            )
        ax.set_xticks(range(len(values)), [f"{v:g}" for v in values])
        ax.set_ylim(-0.25, 1.25)
        ax.set_xlabel(label, color=TEXT_COLOR)
    axes[0].set_ylabel("following", color=TEXT_COLOR)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=FAILING_COLOR, label="fails"),
        Line2D([], [], marker="o", linestyle="", color="#2a78d6", label="passes"),
        Line2D([], [], color=REPLAY_COLOR, label="the replay"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def block_gaps() -> tuple[dict[tuple[float, float, float, float], float], float]:
    """Each ESN's block gap, by the second rule's run, and the replay's distance at the release (deg)."""
    replay = robot_run("block", CHOSEN["ESN #1"][0], "replay")["end_gap_deg"]
    return {esn_of(row): row["block_gap"] for row in esns("second")}, replay


def two_link_gap(tag: str) -> float:
    esn = robot_run("block", tag, "esn", results=TWO_LINK_RESULTS)["end_gap_deg"]
    return esn / robot_run("block", tag, "replay", results=TWO_LINK_RESULTS)["end_gap_deg"]


def plot_gap() -> Figure:
    """The block gap of every ESN, and of those passing the first rule, with the ESNs compared marked."""
    gaps, _ = block_gaps()
    rows = esns("extended")
    every = np.array([gaps[esn_of(row)] for row in rows])
    passing = np.array([gaps[esn_of(row)] for row in rows if row["passes"]])
    fig = Figure(figsize=(13, 7), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The block gap: the reference's distance from the arm's joint angles at the release, over the replay's; the"
        " last bar also holds the gaps above 1.2",
        color=TITLE_COLOR,
    )
    top, bottom = fig.subplots(2, 1, sharex=True, height_ratios=[1, 1.3])
    bins = np.arange(0, 1.2001, 0.05)
    for ax, data, color, title in (
        (top, every, FAILING_COLOR, f"all {len(every):,} ESNs of the grid and the extension"),
        (bottom, passing, "#2a78d6", f"the {len(passing)} ESNs that pass the first rule"),
    ):
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.hist(np.clip(data, 0, 1.2), bins=bins, color=color, edgecolor=SURFACE_COLOR, linewidth=1.2, zorder=2)
        ax.axvline(0.5, color=TITLE_COLOR, linestyle="--", linewidth=1.4, zorder=3)
        ax.set_title(title, color=TITLE_COLOR, fontsize=10, loc="left")
        ax.set_ylabel("ESNs", color=TEXT_COLOR)
    top.text(0.51, 0.86 * top.get_ylim()[1], "the second rule's limit, 0.5", color=TEXT_COLOR)
    height = 19.0
    bottom.set_ylim(0, 1.95 * height)
    marks = [
        ("ESN #2", gaps[CHOSEN["ESN #2"][1]], ESN_COLORS["ESN #1"], 1.72),
        ("ESN #1", gaps[CHOSEN["ESN #1"][1]], ESN_COLORS["ESN #1"], 1.42),
        ("two-link #2", two_link_gap(TWO_LINK["Two-link ESN #2"][0]), TWO_LINK_COLOR, 1.12),
        ("two-link #1", two_link_gap(TWO_LINK["Two-link ESN #1"][0]), TWO_LINK_COLOR, 0.86),
        ("earlier #2", gaps[EARLIER["Earlier ESN #2"][1]], EARLIER_COLOR, 1.15),
        ("earlier #1", gaps[EARLIER["Earlier ESN #1"][1]], EARLIER_COLOR, 1.15),
    ]
    for label, x, color, level in marks:
        bottom.plot([x, x], [0, height * level], color=color, linewidth=2, zorder=4)
        bottom.text(x + 0.005, height * level, label, va="center", color=TITLE_COLOR)
    bottom.set_xlabel("block gap", color=TEXT_COLOR)
    return fig


def plot_gap_series() -> Figure:
    """The reference minus the arm in each joint around the block."""
    series = [
        ("the replay", CHOSEN["ESN #1"][0], "replay", REPLAY_COLOR, "-"),
        ("earlier ESN #1", EARLIER["Earlier ESN #1"][0], "esn", EARLIER_COLOR, "--"),
        ("ESN #1", CHOSEN["ESN #1"][0], "esn", ESN_COLORS["ESN #1"], "-"),
        ("ESN #2", CHOSEN["ESN #2"][0], "esn", ESN_COLORS["ESN #2"], "-"),
    ]
    fig = Figure(figsize=(13, 7.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("The reference minus the arm in each joint, around the block (shaded)", color=TITLE_COLOR)
    axes = fig.subplots(2, 2, sharex=True)
    logs = [(label, robot_log("block", tag, arm), color, line) for label, tag, arm, color, line in series]
    for joint, ax in enumerate(axes.flat):
        style(ax)
        ax.axvspan(4.0, 5.0, color=DISTURBANCE_COLOR, zorder=0)
        ax.axhline(0.0, color=TEXT_COLOR, linewidth=0.8, zorder=1)
        for label, log, color, line in logs:
            window = (log.times >= 3.5) & (log.times <= 7.0)
            gap = np.degrees(log.channel("q_ref") - log.channel("q"))[window, joint]
            ax.plot(log.times[window], gap, color=color, linestyle=line, linewidth=2, label=label, zorder=3)
        ax.set_title(f"joint {joint + 1}", color=TITLE_COLOR, fontsize=10, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color=TEXT_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("reference minus arm (deg)", color=TEXT_COLOR)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_series(name: str, tag: str, channel: str) -> Figure:
    """One ESN's joint angles or torques over time against both replays': a column per scenario."""
    fig = Figure(figsize=(18, 12.5), facecolor=SURFACE_COLOR, layout="constrained")
    what = "Joint angles" if channel == "joints" else "Joint torques"
    fig.suptitle(
        f"{what} of {name}'s arm (blue), the replay's (orange), and the progress replay's (aqua), with the search's"
        " tracker"
        + (
            "; dashed: the references of the ESN and the progress replay; thick gray: the taught motion"
            if channel == "joints"
            else "; each column's own axis, leaving out the first 50 ms of a run"
        ),
        color=TITLE_COLOR,
    )
    joints = range(4)
    axes = fig.subplots(1 + len(joints), len(COLUMNS), sharex="col", height_ratios=[0.5, 1.0, 1.0, 1.0, 1.0])
    q_taught = np.degrees(taught())
    t_taught = DT * np.arange(len(q_taught))
    logs = [{arm: robot_log(scenario, tag, arm, origin) for _, arm, _ in ARMS} for scenario, origin, _ in COLUMNS]
    row_limits = []
    for joint in joints:
        values = [np.degrees(log.channel("q")[:, joint]) for column in logs for log in column.values()]
        values.append(q_taught[:, joint])
        low, high = float(min(v.min() for v in values)), float(max(v.max() for v in values))
        row_limits.append((low - 0.05 * (high - low), high + 0.05 * (high - low)))
    top = max(float(tip_force(log).max()) for column in logs for log in column.values())
    for col, ((scenario, _, title), arms) in enumerate(zip(COLUMNS, logs, strict=True)):
        spans = disturbance_spans(robot_config(scenario).get("disturbance"))
        for row in range(1 + len(joints)):
            style(axes[row, col])
            for onset, end in spans:
                axes[row, col].axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
        axes[0, col].set_title(title, color=TITLE_COLOR, fontsize=10)
        for _, arm, color in ARMS:
            axes[0, col].plot(
                arms[arm].times, tip_force(arms[arm]), color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm]
            )
        axes[0, col].set_ylim(-0.05 * top, 1.1 * top)
        for joint in joints:
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
            if channel == "joints":
                ax.set_ylim(*row_limits[joint])
            else:
                after = [log.channel("tau")[log.times >= 0.05, joint] for log in arms.values()]
                low, high = float(min(v.min() for v in after)), float(max(v.max() for v in after))
                ax.set_ylim(low - 0.05 * (high - low), high + 0.05 * (high - low))
            ax.set_xlim(-1.0, 23.9)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    axes[0, 0].set_ylabel("tip force (N)", color=TEXT_COLOR)
    unit = "deg" if channel == "joints" else "N m"
    for joint in joints:
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


def print_tables(*, logs: bool) -> None:
    print_gains()
    print_grid()
    print_scenarios()
    print_offsets()
    print_by_scaling()
    print_closest()
    print_first_pair()
    print_gap(logs=logs)
    print_second_rule()
    print_held_out()
    print_robot()
    if logs:
        print_compare()
        print_starts()
        print_checks()


def fmt(number: float, digits: int = 2) -> str:
    """A number with its digits, or a dash if it is not a number (a hand not back before the next disturbance)."""
    if not np.isfinite(number):
        return "\u2013"
    text = f"{number:.{digits}f}"
    return text.lstrip("-") if float(text) == 0 else text


def print_gains() -> None:
    """What each joint feels at the take's final posture, and the modes of the per-joint gains, against the bound."""
    inertia = mass_matrix_at(Skeleton.from_toml(ARM_CONFIG), taught()[-1])
    own = np.diag(inertia)
    free = 1.0 / np.diag(np.linalg.inv(inertia))
    modes = np.sort(np.linalg.eigvals(np.linalg.solve(inertia, np.diag(own))).real)
    bound = 2.0 / (2.0 * ZETA * OMEGA * HELD + OMEGA**2 * HELD**2 / 10.0)
    print("\nAt the take's final posture:")
    print(f"  each joint's own inertia M_ii {np.round(own, 4).tolist()} kg m^2")
    print(f"  with the other joints free, 1/(M^-1)_ii {np.round(free, 5).tolist()} kg m^2")
    print(f"  the ratio {np.round(own / free, 2).tolist()}")
    print(
        f"  the modes of the per-joint gains {np.round(modes, 2).tolist()}, against the held torque's bound {bound:.1f}"
    )


def print_grid() -> None:
    """Per leak rate: the ESNs that pass the first rule, how the others fail, and the settings' passing seeds."""
    for select, title, scalings, of in (
        ("grid", "The grid", SCALINGS[2:], 420),
        ("extended", "The grid and the extension", SCALINGS, 560),
    ):
        print(f"\n{title}, per leak rate, under the first rule (seeds 0 to 9):")
        print(
            f"| Leak rate | ESNs passing (of {of}) | Failing undisturbed / block / pushes / offsets |"
            " Best setting's seeds |"
        )
        print("| ---: | ---: | --- | ---: |")
        rows = esns(select)
        for leak in LEAKS:
            at = [row for row in rows if row["leak_rate"] == leak]
            fail = [sum(s in failing_scenarios(row) for row in at) for s in SCENARIOS]
            best = max(row["passing_seeds"] for key, row in settings(select).items() if key[0] == leak)
            print(f"| {leak:g} | {sum(row['passes'] for row in at)} | {' / '.join(map(str, fail))} | {best:.0f} |")
        fails = Counter(how for row in rows for _, _, how in failures(row))
        runs = Counter(scenario for row in rows for scenario, _, _ in failures(row))
        print(
            f"{sum(row['passes'] for row in rows)} of {len(rows)} pass; failing runs {dict(runs)}; by how {dict(fails)}"
        )
        seeds = Counter(int(row["passing_seeds"]) for row in settings(select).values())
        print(f"settings: {len(settings(select))}; seeds passing {sorted(seeds.items())}")
        per_scaling = [(s, sum(r["passes"] for r in rows if r["input_scaling"] == s)) for s in scalings]
        print(f"passing per input scaling: {per_scaling}")
        per_radius = [(r, sum(row["passes"] for row in rows if row["spectral_radius"] == r)) for r in RADII]
        print(f"passing per spectral radius: {per_radius}")


def print_scenarios() -> None:
    """The ESNs failing each scenario, and the failing runs from the offset starts."""
    for select, label, source in (("grid", "this arm", esns("grid")), ("two-link", "two-link", two_link_esns())):
        each = {s: sum(s in failing_scenarios(row) for row in source) for s in SCENARIOS}
        passing = {s: len(source) - each[s] for s in SCENARIOS}
        runs = [f for row in source for f in failures(row)]
        offsets = [f for f in runs if f[0] == "offsets"]
        how = Counter(f[2] for f in offsets)
        share = {k: round(100 * v / len(offsets), 1) for k, v in how.items()}
        print(f"\n{label} ({select}): ESNs failing each scenario {each}; passing each {passing}")
        print(
            f"  failing runs {len(runs)}, from offset starts {len(offsets)}"
            f" ({100 * len(offsets) / len(runs):.1f}%); how {share}"
        )


def print_offsets() -> None:
    """Over the grid: which offset starts fail the ESNs that pass the other three scenarios."""
    share, count = offset_failures()
    rows = [row for row in esns("grid") if not failing_scenarios(row) & {"nominal", "block", "pushes"}]
    only = [len([f for f in failures(row) if f[0] == "offsets"]) for row in rows if not row["passes"]]
    moved = [v for start, v in share.items() if start != (0, 0, 0, 0)]
    print(
        f"\nThe grid: {count} ESNs pass the other three, {sum(row['passes'] for row in rows)} also from the 36 starts;"
        f" an ESN failing only from them fails from a median of {np.median(only):.0f}; each start fails"
        f" {min(moved):.0f}% to {max(moved):.0f}%, median {np.median(moved):.0f}%; the take's start"
        f" {share.get((0, 0, 0, 0), 0.0):.0f}%"
    )
    for joint in (2, 3):
        means = {o: np.mean([v for s, v in share.items() if s[joint] == o and s != (0, 0, 0, 0)]) for o in OFFSETS}
        print(
            f"  mean share by joint {joint + 1}'s offset: " + ", ".join(f"{o:+d}: {m:.0f}%" for o, m in means.items())
        )


def print_by_scaling() -> None:
    four, two = passing_by_scaling()
    print(f"\nPassing the first rule by input scaling {SCALINGS}: here {four}; two-link {two}")


def print_closest() -> None:
    """The settings nearest to qualifying under the first rule."""
    print("\nThe settings of both grids with six or more passing seeds, under the first rule:")
    print(
        "| Leak rate | Spectral radius | Input scaling | Passing seeds | Following | Ringing | Return | Path RMSE |"
        " Rule |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    rows = settings("extended")
    for key, row in sorted(rows.items(), key=lambda kv: (-kv[1]["passing_seeds"], -kv[1]["following"])):
        if row["passing_seeds"] >= 6:
            chosen = "" if row["chosen"] == "" else f"earlier ESN #{row['chosen']:.0f}, seed {row['typical_seed']:.0f}"
            print(
                f"| {key[0]:g} | {key[1]:g} | {key[2]:g} | {row['passing_seeds']:.0f} | {row['following']:.2f} |"
                f" {row['ringing_mm']:.2f} mm | {row['return_s']:.2f} s | {row['path_rmse_mm']:.2f} mm | {chosen} |"
            )


def print_first_pair() -> None:
    """The earlier pair's following under each disturbance, against both replays and the two-link ESN #1."""
    print("\nThe following under the block and the pushes, and the reference's lead at the block's end:")
    print("| Arm | Block | Push 1 | Push 2 | Push 3 | Mean | Lead |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    tag_one, tag_two = EARLIER["Earlier ESN #1"][0], EARLIER["Earlier ESN #2"][0]
    for label, tag, arm in (
        ("Earlier ESN #1", tag_one, "esn"),
        ("Earlier ESN #2", tag_two, "esn"),
        ("Replay", tag_one, "replay"),
        ("Progress replay", tag_one, "progress"),
    ):
        block, pushes = robot_run("block", tag, arm), robot_run("pushes", tag, arm)
        gains = [block["following_gain"]] + [pushes[f"following_gain_{i}"] for i in (1, 2, 3)]
        cells = " | ".join(f"{abs(g):.2f}" if abs(g) < 0.005 else f"{g:.2f}" for g in gains)
        print(f"| {label} | {cells} | {np.mean(gains):.2f} | {100 * block['reference_lead']:.1f}% |")
    two = next(row for row in two_link_esns() if esn_of(row) == TWO_LINK["Two-link ESN #1"][1])
    gains = [two["following_block"], two["following_push1"], two["following_push2"], two["following_push3"]]
    cells = " | ".join(f"{g:.2f}" for g in gains)
    print(f"| Two-link ESN #1 | {cells} | {two['following']:.2f} | {two['block_lead_pct']:.1f}% |")
    passing = [row for row in esns("extended") if row["passes"]]
    failing = [row for row in esns("extended") if not row["passes"] and np.isfinite(row["following"])]
    for s in SCALINGS:
        at = [row["following"] for row in passing if row["input_scaling"] == s]
        print(f"  median following of the passing ESNs at input scaling {s:g}: {np.median(at):.2f} ({len(at)} ESNs)")
    print(
        f"  median following: passing {np.median([r['following'] for r in passing]):.2f},"
        f" failing {np.median([r['following'] for r in failing]):.2f}"
    )


def print_gap(*, logs: bool) -> None:
    """The block gap, and what the offset starts do to the ESNs with a small one."""
    gaps, replay = block_gaps()
    rows = esns("extended")
    every = [gaps[esn_of(row)] for row in rows]
    print(f"\nThe block gap: the replay's distance at the release is {replay:.2f} deg")
    print(
        f"  median: all {np.median(every):.2f}, passing the first rule"
        f" {np.median([gaps[esn_of(r)] for r in rows if r['passes']]):.2f};"
        f" above 1.2: {sum(g > 1.2 for g in every)} of all,"
        f" {sum(gaps[esn_of(r)] > 1.2 for r in rows if r['passes'])} of those passing"
    )
    for group in (EARLIER, CHOSEN):
        print("  " + ", ".join(f"{name} {gaps[esn]:.2f}" for name, (_, esn) in group.items()))
    print("  " + ", ".join(f"{name} {two_link_gap(tag):.2f}" for name, (tag, _) in TWO_LINK.items()))
    three = [row for row in rows if not failing_scenarios(row) & {"nominal", "block", "pushes"}]
    small = [row for row in three if gaps[esn_of(row)] <= 0.5]
    large = [row for row in three if gaps[esn_of(row)] > 0.5]
    print(
        f"  of {len(three)} ESNs passing the other three: {sum(r['passes'] for r in small)} of {len(small)}"
        f" with a gap of 0.5 or less pass the offsets, {sum(r['passes'] for r in large)} of {len(large)} with more"
    )
    runs = [f for row in small for f in failures(row) if f[0] == "offsets"]
    leaving = [f for f in runs if f[2] == "leaves the goal during the hold"]
    print(f"  their failing offset runs: {len(runs)}, of which {len(leaving)} leave the goal during the hold")
    if logs:
        print_leaving(small)


def print_leaving(small: list[dict[str, Any]]) -> None:
    """When the small-gap ESNs arrive from the offset starts, in the runs that leave the goal and in those that pass."""
    wanted = {esn_of(row) for row in small}
    leaving, passing = [], []
    for name in sorted(p.name for p in RESULTS.glob("*-sweep_lr*_offsets")) + sorted(
        p.name for p in RESULTS.glob("*-extend_lr*_offsets")
    ):
        run = resolve_run_path(f"results/{name}")
        with (RESULTS / name / "config.toml").open("rb") as f:
            leak = tomllib.load(f)["esn"]["leak_rate"]
        with (run / "runs.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                key = (float(leak), float(row["spectral_radius"]), float(row["input_scaling"]), float(row["seed"]))
                if key not in wanted or row["arrived"] != "True":
                    continue
                if float(row["final_distance_m"]) > 0.02:
                    continue
                if row["left_goal"] == "True":
                    leaving.append(float(row["arrival_time_s"]))
                elif row["failed"] != "True":
                    passing.append(float(row["arrival_time_s"]))
    print(
        f"  arrival, median: runs leaving the goal {np.median(leaving):.2f} s ({len(leaving)} runs),"
        f" runs passing {np.median(passing):.2f} s ({len(passing)} runs)"
    )


def print_second_rule() -> None:
    """The settings qualifying under the second rule, over seeds 0 to 9 and over the held-out seeds."""
    first, held = settings("second"), settings("held_out")
    rows = sorted((row for row in first.values() if row["qualifies"]), key=lambda row: row["rank"])
    print(f"\nThe second rule: {sum(r['passes'] for r in esns('second'))} of {len(esns('second'))} ESNs pass")
    print(
        "| Rank | Leak rate | Spectral radius | Input scaling | Seeds passing, 0-9 | Seeds passing, 10-19 | Gap, 0-9 |"
        " Gap, 10-19 | Following, 0-9 | Following, 10-19 | Ringing | Path RMSE | Typical seed |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        other = held[setting_of(row)]
        print(
            f"| {row['rank']:.0f} | {row['leak_rate']:g} | {row['spectral_radius']:g} | {row['input_scaling']:g} |"
            f" {row['passing_seeds']:.0f} | {other['passing_seeds']:.0f} | {row['block_gap']:.2f} |"
            f" {other['block_gap']:.2f} | {row['following']:.2f} | {other['following']:.2f} |"
            f" {row['ringing_mm']:.2f} mm | {row['path_rmse_mm']:.2f} mm | {row['typical_seed']:.0f} |"
        )
    largest_gap = max(abs(row["block_gap"] - held[setting_of(row)]["block_gap"]) for row in rows)
    largest_following = max(abs(row["following"] - held[setting_of(row)]["following"]) for row in rows)
    print(f"  largest change in a median: gap {largest_gap:.3f}, following {largest_following:.3f}")
    print(f"  within the guard: {sum(r['within_guard'] for r in rows)} of {len(rows)}")


def print_held_out() -> None:
    """How the held-out seeds' ESNs fail under the second rule."""
    rows = esns("held_out")
    failing = [row for row in rows if not row["passes"]]
    groups = Counter(tuple(sorted(failing_scenarios(row))) for row in failing)
    never = [f[1] for row in failing for f in failures(row) if f[0] == "offsets" and f[2] == "never arrives"]
    print(
        f"\nThe held-out seeds: {len(rows) - len(failing)} of {len(rows)} pass;"
        f" qualifying {[k for k, r in settings('held_out').items() if r['qualifies']]};"
        f" failing scenarios {dict(groups)}"
    )
    print(
        f"  never arriving offset runs {len(never)}, by joint 3's offset"
        f" {dict(Counter(s[2] for s in never if s is not None))}"
    )


def print_robot() -> None:
    """Each arm on the robot, from the runs' metrics."""
    tags = [
        ("ESN #1", CHOSEN["ESN #1"][0], "esn"),
        ("ESN #2", CHOSEN["ESN #2"][0], "esn"),
        ("Replay", CHOSEN["ESN #1"][0], "replay"),
        ("Progress replay", CHOSEN["ESN #1"][0], "progress"),
    ]
    print("\nFrom the take's start: arrival, path RMSE, and when the hand settles in the goal for good:")
    print("| Arm | Undisturbed | Under the block | Under the pushes |\n| --- | --- | --- | --- |")
    for label, tag, arm in tags:
        cells = []
        for scenario in ("nominal", "block", "pushes"):
            run = robot_run(scenario, tag, arm)
            cells.append(
                f"{run['arrival_time_s']:.2f} s, {1000 * run['taught_path_rmse_m']:.2f} mm,"
                f" {run['settling_time_s']:.2f} s"
            )
        print(f"| {label} | {' | '.join(cells)} |")

    print(
        "\nHow hard the motors work over the task: accumulated torque, squared torque, positive work, and peak torque:"
    )
    print("| Arm | Undisturbed | Under the block | Under the pushes | From the offset starts, medians |")
    print("| --- | --- | --- | --- | --- |")
    for label, tag, arm in tags:
        cells = []
        for scenario in ("nominal", "block", "pushes"):
            run = robot_run(scenario, tag, arm)
            cells.append(
                f"{run['task_torque_nms']:.2f}, {run['task_effort_n2m2s']:.2f}, {run['task_work_j']:.2f},"
                f" {run['peak_torque_nm']:.2f}"
            )
        moved = [r for r in robot_rows("offsets", tag) if r["arm"] == arm and str(r["origin"]).endswith(" deg")]
        cells.append(
            ", ".join(
                f"{np.median([r[key] for r in moved]):.2f}"
                for key in ("task_torque_nms", "task_effort_n2m2s", "task_work_j", "peak_torque_nm")
            )
        )
        print(f"| {label} | {' | '.join(cells)} |")

    print("\nUnder the block:")
    print(
        "| Arm | Following | Lead | Gap at the release (deg) | Ringing (mm) | Farthest (mm) | Back within 5 mm (s) |"
        " Peak torque (N m) | Peak force (N) | Squared torque, 4.0-5.5 s |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, tag, arm in tags:
        run = robot_run("block", tag, arm)
        print(
            f"| {label} | {abs(run['following_gain']):.2f} | {100 * run['reference_lead']:.1f}% |"
            f" {run['end_gap_deg']:.1f} |"
            f" {1000 * run['ringing_m']:.2f} | {1000 * run['off_path_m']:.1f} | {fmt(run['return_time_s'])} |"
            f" {run['peak_torque_nm']:.2f} | {run['peak_external_force_n']:.1f} | {run['effort_n2m2s']:.1f} |"
        )

    print("\nAfter each push, the pushes at 2.6, 6.7, and 14.9 s:")
    print("| Arm | Following | Ringing (mm) | Farthest (mm) | Back within 5 mm (s) |\n| --- | --- | --- | --- | --- |")
    for label, tag, arm in tags:
        run = robot_run("pushes", tag, arm)
        print(
            f"| {label} | "
            + " | ".join(
                " / ".join(fmt(scale * run[f"{key}_{i}"] + 0.0, digits) for i in (1, 2, 3))
                for key, scale, digits in (
                    ("following_gain", 1.0, 2),
                    ("ringing_m", 1000.0, 2),
                    ("off_path_m", 1000.0, 1),
                    ("return_time_s", 1.0, 2),
                )
            )
            + " |"
        )

    print("\nFrom the 35 offset starts (medians, and the extremes where named):")
    print(
        "| Arm | Arrive, end in the goal (of 36) | Also hold (of 36) | Arrival | Latest | Path RMSE | First step |"
        " Peak torque, median / largest | Accumulated torque, largest |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
    for label, tag, arm in tags:
        rows = [r for r in robot_rows("offsets", tag) if r["arm"] == arm]
        moved = [r for r in rows if str(r["origin"]).endswith(" deg")]
        reached = [r for r in rows if r["arrived"] and r["final_distance_m"] <= 0.02]
        print(
            f"| {label} | {len(reached)} | {sum(not r['left_goal'] and r['success'] for r in reached)} |"
            f" {np.median([r['arrival_time_s'] for r in moved]):.2f} s |"
            f" {max(r['arrival_time_s'] for r in moved):.2f} s |"
            f" {1000 * np.median([r['taught_path_rmse_m'] for r in moved]):.1f} mm |"
            f" {1000 * np.median([r['first_step_m'] for r in moved]):.1f} mm |"
            f" {np.median([r['peak_torque_nm'] for r in moved]):.1f} /"
            f" {max(r['peak_torque_nm'] for r in moved):.1f} N m |"
            f" {max(r['task_torque_nms'] for r in moved):.1f} N m s |"
        )
    left = [
        r["origin"].split()[1]
        for r in robot_rows("offsets", CHOSEN["ESN #1"][0])
        if r["arm"] == "esn" and r["left_goal"]
    ]
    print(f"  ESN #1 leaves the goal from: {left}")


def squared_after(log: StateLog, release: float) -> float:
    """The squared torque over the 1 s after a release (N^2 m^2 s)."""
    window = (log.times >= release) & (log.times < release + 1.0)
    return float((log.channel("tau")[window] ** 2).sum() * np.diff(log.times).mean())


def offset_rows(tag: str, arm: str, results: Path) -> list[dict[str, Any]]:
    """One arm's runs from the starts with an offset."""
    return [r for r in robot_rows("offsets", tag, results) if r["arm"] == arm and str(r["origin"]).endswith(" deg")]


def first_second_distance(tag: str, arm: str, results: Path) -> float:
    """From the offset starts: the median of the largest distance between the reference and the arm in the first s."""
    distances = []
    for row in offset_rows(tag, arm, results):
        log = robot_log("offsets", tag, arm, row["origin"], results)
        gap = (log.channel("q_ref") - log.channel("q"))[log.times < 1.0]
        distances.append(np.degrees(np.linalg.norm(gap, axis=1)).max())
    return float(np.median(distances))


def print_compared(name: str, tag: str, results: Path) -> None:
    """One ESN against the replay of its own arm."""

    def of(scenario: str) -> tuple[dict[str, Any], dict[str, Any], StateLog, StateLog]:
        runs = (robot_run(scenario, tag, arm, results=results) for arm in ("esn", "replay"))
        logs = (robot_log(scenario, tag, arm, results=results) for arm in ("esn", "replay"))
        return (*runs, *logs)  # type: ignore[return-value]

    block = robot_config("block", tag, results)["disturbance"]
    run, replay, log, replay_log = of("block")
    per_joint, pooled = joint_following(log, replay_log, block["onset"], block["release"])
    squared = squared_after(log, block["release"]) / squared_after(replay_log, block["release"])
    pushes_run, pushes_replay, pushes_log, pushes_replay_log = of("pushes")
    pushes_following = [
        joint_following(pushes_log, pushes_replay_log, push["onset"], push["onset"] + push["duration"])[1]
        for push in robot_config("pushes", tag, results)["disturbance"]
    ]
    esn_rows, replay_rows = offset_rows(tag, "esn", results), offset_rows(tag, "replay", results)
    peak = np.median([r["peak_torque_nm"] for r in esn_rows]) / np.median([r["peak_torque_nm"] for r in replay_rows])
    distance = first_second_distance(tag, "esn", results) / first_second_distance(tag, "replay", results)
    held = sum(r["arrived"] and r["success"] and not r["left_goal"] for r in esn_rows)
    print(
        f"{name}: gap {run['end_gap_deg'] / replay['end_gap_deg']:.2f} ({run['end_gap_deg']:.1f} deg, replay"
        f" {replay['end_gap_deg']:.1f}); joint following {np.round(per_joint, 2).tolist()}, all {pooled:.2f};"
        f" block peak {100 * run['peak_torque_nm'] / replay['peak_torque_nm']:.0f}% of the replay's;"
        f" squared torque in the 1 s after the release {100 * squared:.0f}%;"
        f" pushes joint following {np.round(pushes_following, 2).tolist()};"
        f" pushes peak {100 * pushes_run['peak_torque_nm'] / pushes_replay['peak_torque_nm']:.0f}%;"
        f" offsets peak {100 * peak:.0f}%; first-second distance {100 * distance:.0f}%;"
        f" hold from {held} of {len(esn_rows)}"
    )


def print_compare() -> None:
    """The earlier and the new ESNs on the robot, beside the two-link study's, each against its own arm's replay."""
    print("\nThe earlier pair, the new ESNs, and the two-link ESNs, each against the replay of its own arm:")
    for group, results in ((EARLIER, RESULTS), (CHOSEN, RESULTS), (TWO_LINK, TWO_LINK_RESULTS)):
        for name, (tag, _) in group.items():
            print_compared(name, tag, results)
    print_release()


def print_release() -> None:
    """The reference's distance from the arm at the release, joint by joint, and joint 3's and 4's positions there."""
    print(
        "\nAt the block's release, reference minus arm per joint (deg), and the arm's joints 3 and 4 during the block:"
    )
    for name, tag, arm in (
        ("ESN #1", CHOSEN["ESN #1"][0], "esn"),
        ("ESN #2", CHOSEN["ESN #2"][0], "esn"),
        ("Replay", CHOSEN["ESN #1"][0], "replay"),
        ("Progress replay", CHOSEN["ESN #1"][0], "progress"),
    ):
        log = robot_log("block", tag, arm)
        at = int(np.searchsorted(log.times, 5.0 - 1e-9))
        q = np.degrees(log.channel("q"))
        gap = np.degrees(log.channel("q_ref")[at] - log.channel("q")[at])
        during = (log.times >= 4.0) & (log.times < 5.0)
        print(
            f"  {name}: {np.round(np.abs(gap), 1).tolist()}; joint 3 at the release {q[at, 2]:.1f},"
            f" joint 4's largest {q[during, 3].max():.1f}"
        )


def print_starts() -> None:
    """From the offset starts: the references beyond the limits, the time at the stops, and the detours."""
    print("\nFrom the 36 starts, from the logs:")
    tags = [
        ("ESN #1", CHOSEN["ESN #1"][0], "esn"),
        ("ESN #2", CHOSEN["ESN #2"][0], "esn"),
        ("Replay", CHOSEN["ESN #1"][0], "replay"),
    ]
    for name, tag, arm in tags:
        beyond, stops, detours = [], [], []
        by_origin = {}
        for row in (r for r in robot_rows("offsets", tag) if r["arm"] == arm):
            log = robot_log("offsets", tag, arm, row["origin"])
            q, reference = log.channel("q"), log.channel("q_ref")
            task = log.times >= 0
            past = max(np.max(LOWER - reference[task], axis=0).max(), np.max(reference[task] - UPPER, axis=0).max())
            at_stop = ((q[task] <= LOWER + 1e-9) | (q[task] >= UPPER - 1e-9)).sum(axis=0) * np.diff(log.times).mean()
            beyond.append(float(np.degrees(max(past, 0.0))))
            stops.append(at_stop)
            by_origin[row["origin"]] = (beyond[-1], at_stop)
            if str(row["origin"]).endswith(" deg"):
                detours.append(1000.0 * detour(log))
        reached = np.array(beyond)
        held = np.array(stops)
        print(
            f"{name}: reference past a limit by more than 5 deg from {np.sum(reached > 5)} starts, more than 10 deg"
            f" from {np.sum(reached > 10)}, at most {reached.max():.1f} deg; joints 2 to 4 at a stop for more than"
            f" 0.1 s from {np.sum(held[:, 1:].max(axis=1) > 0.1)}, longest {held[:, 1:].max():.2f} s"
        )
        print(
            f"  detour: median {np.median(detours):.1f} mm, largest {np.max(detours):.1f} mm,"
            f" over 50 mm from {np.sum(np.array(detours) > 50)} of 35"
        )
        for origin in (NEAREST, FARTHEST):
            past, at_stop = by_origin[origin]
            print(
                f"  {origin.split()[1]}: past a limit {past:.1f} deg; at a stop (joints 1 to 4)"
                f" {np.round(at_stop, 2).tolist()} s"
            )


def print_checks() -> None:
    """The new ESNs' robot runs against their sweeps' runs, of the same ESN."""
    print("\nThe new ESNs' robot runs against the same ESNs' runs in the extension's sweeps:")
    for name, (tag, esn) in CHOSEN.items():
        largest, count = 0.0, 0
        for scenario in SCENARIOS:
            robot = [r for r in robot_rows(scenario, tag) if r["arm"] == "esn"]
            sweep = resolve_run_path(f"results/{run_name(f'extend_lr{esn[0]:g}_{scenario}')}")
            with (sweep / "runs.csv").open(newline="") as f:
                rows = [
                    {key: value(text) for key, text in row.items()}
                    for row in csv.DictReader(f)
                    if (float(row["spectral_radius"]), float(row["input_scaling"]), float(row["seed"])) == esn[1:]
                ]
            for one, other in zip(robot, rows, strict=True):
                for key in set(one) & set(other):
                    a, b = one[key], other[key]
                    if isinstance(a, float) and isinstance(b, float) and np.isfinite(a) and np.isfinite(b):
                        if max(abs(a), abs(b)) > 1e-3:
                            largest = max(largest, abs(a - b) / max(abs(a), abs(b)))
                        count += 1
        print(f"  {name}: {count} values, the largest relative difference {largest:.1e}")


if __name__ == "__main__":
    main()
