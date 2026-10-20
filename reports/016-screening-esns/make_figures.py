# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 016.

    uv run python reports/016-screening-esns/make_figures.py

It reads the study's runs under the storage root, whose arrays are not kept in Git,
copies their configurations, run records, and CSV files into ``results/``, prints the
tables of the report, and draws into ``results/summary``:

- the screen (Section 3.1): ``posture_map.png`` (R over the joint space for ESN #1 and
  #2, with their fixed points), ``fixed_points.png`` (every fixed point of the 1,050
  ESNs, by leak rate), and ``screen.png`` (the screen's measures by leak rate, of the
  grid and of the held-out ESNs);
- against the robot (Sections 3.2 and 3.4): ``rules.png`` (the share failing among
  the ESNs each rule flags and among the others, by leak rate), ``scores.png`` (the
  areas under the ROC curve and the rank correlations within each leak rate), and
  ``following.png`` (the following against the largest scaling of a posture's
  deviation);
- held still on the robot (Section 3.3): ``release.png`` (the robot's arrival after a
  release against the ESN's own, and where the releases were), ``blocks.png`` (the
  blocks' failures and delays), and ``shown_joints.png``, ``shown_torques.png``, and
  ``shown_paths.png`` (ESN #1 and #2 against the replay: undisturbed, under two of
  the blocks, and released at ESN #2's unstable fixed point beside the path).
"""

from __future__ import annotations

import csv
import functools
import shutil
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles
from arm_esn_ctrl.storage import resolve_run_path
from arm_esn_ctrl.tracking import task_joint_angles

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
STUDY = "results/manual_demonstration_v2_esn_screening"
FOLLOWING = "results/manual_demonstration_v2_robot_following_search"
# The study's runs, and what of each is copied into results/: its configuration and run record, and its CSV files.
RUNS: dict[str, tuple[str, bool]] = {
    "screen_first": (f"{STUDY}/20261020-134112-screen", False),
    "screen_20s": (f"{STUDY}/20261020-151050-screen", False),
    "check_5deg": (f"{STUDY}/20261020-160829-screen_5deg_seed0", True),
    "screen": (f"{STUDY}/20261020-165217-screen", True),
    "against": (f"{STUDY}/20261020-194204-against_robot", True),
    "release": (f"{STUDY}/20261020-182057-release_robot", True),
    "screen_heldout": (f"{STUDY}/20261020-200400-screen_seeds10to19", True),
    "against_heldout": (f"{STUDY}/20261020-210939-against_robot_seeds10to19", True),
    "block5_esn1": (f"{STUDY}/20261020-212045-block5to8_robot_lr0.03_sr1.2_is1.5_seed8", True),
    "block5_esn2": (f"{STUDY}/20261020-212045-block5to8_robot_lr0.03_sr1.2_is1_seed2", True),
    "block8_esn1": (f"{STUDY}/20261020-212045-block8to11_robot_lr0.03_sr1.2_is1.5_seed8", True),
    "block8_esn2": (f"{STUDY}/20261020-212045-block8to11_robot_lr0.03_sr1.2_is1_seed2", True),
    "released_esn1": (f"{STUDY}/20261020-212252-released_robot_lr0.03_sr1.2_is1.5_seed8", True),
    "released_esn2": (f"{STUDY}/20261020-212252-released_robot_lr0.03_sr1.2_is1_seed2", True),
    # Report 011's undisturbed runs, copied in its own results.
    "nominal_esn1": (
        f"{FOLLOWING}/20261014-155211-nominal_robot_lr0.03_sr1.2_is1.5_seed8_reference_velocity_filtered",
        False,
    ),
    "nominal_esn2": (
        f"{FOLLOWING}/20261014-155211-nominal_robot_lr0.03_sr1.2_is1_seed2_reference_velocity_filtered",
        False,
    ),
}
# The take, as report 005 copied it: what the ESNs were trained on, and what the replay replays.
TAKE_RUN = REPORT.parent / "005-tuned-on-the-robot" / "data" / "20261008-170034-reach_manual_v2"
DT = 0.01  # the ESNs' period (s)
DPI = 90
LEAK_RATES = (0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1)
SETTING = "pd_w20_z0.4"  # joint PD at omega = 20 rad/s and the damping ratio 0.4

# The named ESNs: leak rate, spectral radius, input scaling, seed.
NAMED = {
    "ESN #1": (0.03, 1.2, 1.5, 8),
    "ESN #2": (0.03, 1.2, 1.0, 2),
    "ESN #3": (0.03, 1.35, 1.5, 6),
    "report 007's": (0.02, 1.35, 2.0, 7),
}
# The arms shown on the robot: key, label, color; and the scenarios: key, title.
ARMS = [("esn1", "ESN #1", "#2a78d6"), ("esn2", "ESN #2", "#5b2a86"), ("replay", "replay", "#eb6834")]
ARM_ZORDERS = {"esn1": 3.0, "esn2": 3.0, "replay": 2.0}
SCENARIOS = [
    ("nominal", "undisturbed"),
    ("block5", "tip held from 5 s to 8 s"),
    ("block8", "tip held from 8 s to 11 s"),
    ("released", "released at ESN #2's unstable fixed point\nbeside the path, after 4 s there"),
]
# The two sets of ESNs: seeds 0 to 9, the grid of report 011, and seeds 10 to 19, held out.
SETS = [
    ("against", "screen", "seeds 0 to 9", "#24527a"),
    ("against_heldout", "screen_heldout", "seeds 10 to 19, held out", "#1b9e77"),
]

TEXT_COLOR = "#52514e"
TITLE_COLOR = "#0b0b0b"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAUGHT_COLOR = "#b9b7b0"
BASE_COLOR = "#d9d8d3"
UNSTABLE_COLOR = "#c0392b"
ATTRACTING_COLOR = "#2a78d6"
NEAR_COLOR = "#c0392b"
NONE_COLOR = "#24527a"


def main() -> None:
    copy_results()
    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {
        "posture_map.png": plot_posture_map(),
        "fixed_points.png": plot_fixed_points(),
        "screen.png": plot_screen(),
        "rules.png": plot_rules(),
        "scores.png": plot_scores(),
        "following.png": plot_following(),
        "release.png": plot_release(),
        "blocks.png": plot_blocks(),
        "shown_joints.png": plot_shown("q"),
        "shown_torques.png": plot_shown("tau"),
        "shown_paths.png": plot_shown_paths(),
    }
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=DPI)
    print_screen()
    print_check()
    print_rules()
    print_claims()
    print_named()
    print_release()
    print_shown()
    print(f"\nWrote {', '.join(figures)} to {SUMMARY}")


# ---------------------------------------------------------------------------- the runs


def run_dir(name: str) -> Path:
    return resolve_run_path(RUNS[name][0])


def copy_results() -> None:
    """Copy each run's configuration and run record, and for most also its CSV files, into ``results/<run>``."""
    for name, (reference, with_csv) in RUNS.items():
        if name.startswith("nominal"):
            continue
        source, target = run_dir(name), RESULTS / Path(reference).name
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.glob("*.toml")) + (sorted(source.glob("*.csv")) if with_csv else []):
            shutil.copy2(path, target / path.name)


def value(text: str) -> Any:
    """A CSV field as a bool, a float, or the text itself."""
    if text in ("True", "False"):
        return text == "True"
    try:
        return float(text)
    except ValueError:
        return text


@functools.cache
def read_csv(name: str, file: str) -> list[dict[str, Any]]:
    with (run_dir(name) / file).open(newline="") as f:
        return [{key: value(text) for key, text in row.items()} for row in csv.DictReader(f)]


def key_of(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (row["leak_rate"], row["spectral_radius"], row["input_scaling"], int(row["seed"]))


@functools.cache
def screen_by_key(name: str) -> dict[tuple[float, float, float, int], dict[str, Any]]:
    return {key_of(row): row for row in read_csv(name, "screen.csv")}


@functools.cache
def maps() -> dict[str, NDArray[Any]]:
    with np.load(run_dir("screen") / "maps.npz") as data:
        return {name: data[name] for name in data.files}


def column(rows: list[dict[str, Any]], name: str) -> NDArray[np.float64]:
    return np.array([float(row[name]) for row in rows])


def leak_rate_groups(rows: list[dict[str, Any]]) -> list[NDArray[np.bool_]]:
    lr = column(rows, "leak_rate")
    return [lr == v for v in LEAK_RATES]


# ---------------------------------------------------------------------------- the arm and the take


@functools.cache
def skeleton() -> Skeleton:
    return Skeleton.from_toml(TAKE_RUN / "config.toml")


@functools.cache
def taught() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The take filtered at 2 Hz, what the ESNs were trained on, every 10 ms: its joint angles and its hand path."""
    q = resample_joint_angles(StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"), DT)[1]
    return q, endpoint_positions(skeleton(), q)


@functools.cache
def goal() -> tuple[NDArray[np.float64], float]:
    with (TAKE_RUN / "config.toml").open("rb") as f:
        target = tomllib.load(f)["task"]["target"]
    return np.asarray(target["pos"], dtype=np.float64), float(target["tolerance"])


# ---------------------------------------------------------------------------- drawing


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def new_figure(width: float, height: float, title: str) -> Figure:
    fig = Figure(figsize=(width, height), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color=TITLE_COLOR)
    return fig


def draw_take_in_joint_space(ax: Axes) -> None:
    """The take's joint path (thick gray), its start (circle), and its end (star), in degrees."""
    q = np.degrees(taught()[0])
    ax.plot(q[:, 0], q[:, 1], color=TAUGHT_COLOR, linewidth=4.0, zorder=4, solid_capstyle="round")
    ax.plot(*q[0], "o", color=TITLE_COLOR, markersize=6, markerfacecolor="white", zorder=6)
    ax.plot(*q[-1], "*", color=TITLE_COLOR, markersize=12, markerfacecolor="white", zorder=6)


def joint_axes(ax: Axes) -> None:
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    ax.set_aspect("equal")
    ax.set_xlabel("joint 1 (deg)", color=TEXT_COLOR)
    ax.set_ylabel("joint 2 (deg)", color=TEXT_COLOR)


def near_take_axes(ax: Axes) -> None:
    """The joint space around the take, where every fixed point off the target lies."""
    ax.set_xlim(-40, 160)
    ax.set_ylim(-100, 160)
    ax.set_xticks(np.arange(-40, 161, 40))
    ax.set_yticks(np.arange(-80, 161, 40))
    ax.set_aspect("equal")
    ax.set_xlabel("joint 1 (deg)", color=TEXT_COLOR)
    ax.set_ylabel("joint 2 (deg)", color=TEXT_COLOR)


def take_handles() -> list[Line2D]:
    return [
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the take's path"),
        Line2D([], [], linestyle="none", marker="o", color=TITLE_COLOR, markerfacecolor="white", label="its start"),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="*",
            markersize=12,
            color=TITLE_COLOR,
            markerfacecolor="white",
            label="its end, at the target",
        ),
    ]


def leak_rate_axis(ax: Axes) -> None:
    ax.set_xticks(range(len(LEAK_RATES)), [f"{v:g}" for v in LEAK_RATES])
    ax.set_xlim(-0.6, len(LEAK_RATES) - 0.4)
    ax.set_xlabel("leak rate", color=TEXT_COLOR)


def set_handles() -> list[Line2D]:
    return [Line2D([], [], color=color, marker="o", linewidth=1.2, label=label) for *_, label, color in SETS]


# ---------------------------------------------------------------------------- the screen


def plot_posture_map() -> Figure:
    """R over the joint space for ESN #1 and #2, the zero curves of its components, and their fixed points."""
    data = maps()
    grid = data["grid_deg"]
    step = float(grid[1] - grid[0])
    extent = (grid[0] - step / 2, grid[-1] + step / 2, grid[0] - step / 2, grid[-1] + step / 2)
    fig = new_figure(
        13.5,
        6.9,
        "R(q): how far the ESN's output leads a posture held at q once its reservoir has settled there (every 10°),"
        "\nits zero curves, and its fixed points: where both components are zero",
    )
    axes = fig.subplots(1, 2)
    norm = LogNorm(0.1, 180.0)
    image = None
    for ax, name in zip(axes, ("ESN #1", "ESN #2"), strict=True):
        style(ax)
        (index,) = np.flatnonzero(np.all(data["values"] == np.array(NAMED[name], dtype=np.float64), axis=1))
        r = np.degrees(data["residual"][index])  # (q1, q2, joint)
        image = ax.imshow(
            np.linalg.norm(r, axis=-1).T, origin="lower", extent=extent, cmap="Greys", norm=norm, zorder=1
        )
        for joint, color in ((0, "#2a78d6"), (1, "#d81b60")):
            ax.contour(grid, grid, r[:, :, joint].T, levels=[0.0], colors=[color], linewidths=1.4, zorder=3)
        draw_take_in_joint_space(ax)
        for row in fixed_rows_of("screen", NAMED[name]):
            stable = row["n_unstable"] == 0
            ax.plot(
                row["q1_deg"],
                row["q2_deg"],
                "*" if stable else "X",
                color=ATTRACTING_COLOR if stable else UNSTABLE_COLOR,
                markersize=15 if stable else 11,
                markeredgecolor="white",
                zorder=7,
            )
        joint_axes(ax)
        ax.set_title(name, color=TITLE_COLOR)
    assert image is not None
    fig.colorbar(image, ax=axes, label="|R| (deg)", shrink=0.8)
    handles = [
        Line2D([], [], color="#2a78d6", linewidth=1.4, label="R's joint-1 component zero"),
        Line2D([], [], color="#d81b60", linewidth=1.4, label="R's joint-2 component zero"),
        Line2D(
            [], [], linestyle="none", marker="*", markersize=13, color=ATTRACTING_COLOR, label="attracting fixed point"
        ),
        Line2D([], [], linestyle="none", marker="X", markersize=10, color=UNSTABLE_COLOR, label="unstable fixed point"),
        *take_handles(),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def fixed_rows_of(name: str, key: tuple[float, float, float, int]) -> list[dict[str, Any]]:
    return [row for row in read_csv(name, "fixed_points.csv") if key_of(row) == key]


def plot_fixed_points() -> Figure:
    """Every fixed point of the 1,050 ESNs of seeds 0 to 9 off the target, by leak rate."""
    rows = read_csv("screen", "fixed_points.csv")
    radius = goal()[1]
    fig = new_figure(
        14.0,
        7.6,
        "The fixed points of the 1,050 ESNs (seeds 0 to 9) other than the target's, by leak rate (150 ESNs each)",
    )
    axes = fig.subplots(2, 4).ravel()
    for ax, lr in zip(axes, LEAK_RATES, strict=False):
        style(ax)
        draw_take_in_joint_space(ax)
        mine = [row for row in rows if row["leak_rate"] == lr and row["target_distance_m"] > radius]
        unstable = np.array([[r["q1_deg"], r["q2_deg"]] for r in mine if r["n_unstable"] > 0]).reshape(-1, 2)
        attracting = np.array([[r["q1_deg"], r["q2_deg"]] for r in mine if r["n_unstable"] == 0]).reshape(-1, 2)
        ax.plot(unstable[:, 0], unstable[:, 1], "x", color=UNSTABLE_COLOR, markersize=4, alpha=0.6, zorder=5)
        ax.plot(attracting[:, 0], attracting[:, 1], "o", color=ATTRACTING_COLOR, markersize=4, alpha=0.7, zorder=5)
        near_take_axes(ax)
        ax.set_title(
            f"leak rate {lr:g}: {len(unstable)} unstable, {len(attracting)} attracting", color=TITLE_COLOR, fontsize=10
        )
    axes[-1].axis("off")
    handles = [
        Line2D([], [], linestyle="none", marker="x", color=UNSTABLE_COLOR, label="unstable fixed point"),
        Line2D(
            [], [], linestyle="none", marker="o", color=ATTRACTING_COLOR, label="attracting fixed point off the target"
        ),
        *take_handles(),
    ]
    axes[-1].legend(handles=handles, loc="center", frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_screen() -> Figure:
    """The screen's measures by leak rate, of the grid and of the held-out ESNs."""
    fig = new_figure(13.5, 7.4, "The screen's measures by leak rate (150 ESNs each)")
    axes = fig.subplots(2, 2).ravel()
    panels = [
        (
            "share with an attracting fixed point off the target",
            lambda r: column(r, "attracting_off_target") > 0,
            "share",
        ),
        (
            "share with an unstable fixed point within 30° of the path",
            lambda r: column(r, "nearest_unstable_path_distance_deg") <= 30.0,
            "share",
        ),
        ("median lead R near the path (deg)", lambda r: column(r, "lead_median_deg"), "spread"),
        ("largest scaling of a posture's deviation along the run", lambda r: column(r, "posture_scale_max"), "spread"),
    ]
    for ax, (title, measure, kind) in zip(axes, panels, strict=True):
        style(ax)
        for k, (_, screen, label, color) in enumerate(SETS):
            rows = read_csv(screen, "screen.csv")
            values = measure(rows)
            x = np.arange(len(LEAK_RATES)) + (k - 0.5) * 0.18
            if kind == "share":
                ax.plot(x, [values[g].mean() for g in leak_rate_groups(rows)], "o-", color=color, label=label)
            else:
                q = np.array([np.percentile(values[g], [25, 50, 75]) for g in leak_rate_groups(rows)])
                ax.errorbar(
                    x,
                    q[:, 1],
                    yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                    fmt="o",
                    color=color,
                    capsize=3,
                    label=label,
                )
        ax.set_title(title, color=TITLE_COLOR, fontsize=10)
        leak_rate_axis(ax)
        if kind == "share":
            ax.set_ylim(0, 1.02)
    axes[3].set_yscale("log")
    axes[3].set_yticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    fig.legend(
        handles=set_handles(),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        labelcolor=TEXT_COLOR,
        title="median and quartiles, where not a share",
    )
    return fig


# ---------------------------------------------------------------------------- against the robot


def rule_row(name: str, rule: str, group: str) -> dict[str, Any]:
    (row,) = [r for r in read_csv(name, "rules.csv") if r["rule"] == rule and str(r["leak_rate"]) == group]
    return row


def group_name(lr: float) -> str:
    return str(float(f"{lr:g}"))


def plot_rules() -> Figure:
    """The share failing among the ESNs each rule flags and among the others, by leak rate."""
    fig = new_figure(
        14.0,
        5.4,
        "The share of ESNs failing on the robot (report 011's rule) among those each rule flags (filled) and the"
        " others (open),\nthe marker's area by how many ESNs it is of (of 150); gray: among all",
    )
    axes = fig.subplots(1, 3, sharey=True)
    titles = {
        "a": "(a) an attracting fixed point off the target",
        "b": "(b) a median lead under 10° near the path",
        "c": "(c) an unstable fixed point within 30° of the path,\nor arriving late after a release there",
    }
    for ax, (rule, title) in zip(axes, titles.items(), strict=True):
        style(ax)
        base = [rule_row("against", rule, group_name(lr))["failing"] / 150 for lr in LEAK_RATES]
        ax.bar(range(len(LEAK_RATES)), base, color=BASE_COLOR, width=0.7, zorder=1)
        for k, (against, _, _, color) in enumerate(SETS):
            x = np.arange(len(LEAK_RATES)) + (k - 0.5) * 0.2
            rows = [rule_row(against, rule, group_name(lr)) for lr in LEAK_RATES]
            flagged = np.array([r["flagged"] for r in rows])
            others = np.array([r["esns"] for r in rows]) - flagged
            ax.scatter(x, [r["failing_share_flagged"] for r in rows], s=marker_area(flagged), color=color, zorder=3)
            ax.scatter(
                x,
                [r["failing_share_unflagged"] for r in rows],
                s=marker_area(others),
                facecolor="white",
                edgecolor=color,
                linewidth=1.4,
                zorder=3,
            )
        ax.set_title(title, color=TITLE_COLOR, fontsize=10)
        leak_rate_axis(ax)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("share failing", color=TEXT_COLOR)
    handles = [
        *set_handles(),
        Line2D([], [], linestyle="none", marker="o", color=TEXT_COLOR, label="flagged"),
        Line2D([], [], linestyle="none", marker="o", color=TEXT_COLOR, markerfacecolor="white", label="not flagged"),
        Patch(color=BASE_COLOR, label="all, seeds 0 to 9"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def marker_area(n: NDArray[Any]) -> NDArray[np.float64]:
    """A marker's area (pt^2) for a share of ``n`` ESNs of 150: 150 at 150, none at none."""
    return 150.0 * np.asarray(n, dtype=np.float64) / 150.0 + 4.0 * (np.asarray(n) > 0)


def score(name: str, measure: str, against: str, group: str) -> float:
    (row,) = [
        r
        for r in read_csv(name, "scores.csv")
        if r["measure"] == measure and r["against"] == against and str(r["leak_rate"]) == group
    ]
    return float(row["value"])


def plot_scores() -> Figure:
    """The areas under the ROC curve and the rank correlations within each leak rate."""
    fig = new_figure(13.5, 7.4, "Within each leak rate: what the screen tells of the robot's outcomes")
    axes = fig.subplots(2, 2).ravel()
    panels = [
        (
            "(b) a smaller lead, for ending away under the pushes:\narea under the ROC curve",
            "smaller lead",
            "ends_away",
            0.5,
        ),
        (
            "an unstable fixed point nearer the path, for failing:\narea under the ROC curve (ESNs with one)",
            "nearer unstable fixed point",
            "failing",
            0.5,
        ),
        (
            "(d) the largest scaling of a posture's deviation, with the following:\nrank correlation",
            "posture_scale_max",
            "following",
            0.0,
        ),
        (
            "(c) the arrival after a release, with the arrival under the block:\nrank correlation (ESNs with one)",
            "release arrival",
            "arrival_block_s",
            0.0,
        ),
    ]
    for ax, (title, measure, against, chance) in zip(axes, panels, strict=True):
        style(ax)
        ax.axhline(chance, color=TEXT_COLOR, linewidth=1.0, linestyle=":")
        for k, (name, _, label, color) in enumerate(SETS):
            x = np.arange(len(LEAK_RATES)) + (k - 0.5) * 0.18
            ax.plot(
                x, [score(name, measure, against, group_name(lr)) for lr in LEAK_RATES], "o-", color=color, label=label
            )
        ax.set_title(title, color=TITLE_COLOR, fontsize=10)
        leak_rate_axis(ax)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(-0.2, 1)
    axes[3].set_ylim(-0.5, 0.5)
    fig.legend(handles=set_handles(), loc="outside lower center", ncol=2, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_following() -> Figure:
    """The following against the largest scaling of a posture's deviation along the ESN's own run, by leak rate."""
    fig = new_figure(
        14.0,
        7.6,
        "Report 011's following (the mean following gain over the block and the three pushes) against the largest"
        " scaling of a posture's deviation\nover 4 s windows along the ESN's own run from the take's start",
    )
    axes = fig.subplots(2, 4, sharex=True, sharey=True).ravel()
    for ax, lr in zip(axes, LEAK_RATES, strict=False):
        style(ax)
        texts = []
        for name, _, _, color in SETS:
            rows = [r for r in read_csv(name, "esns.csv") if r["leak_rate"] == lr]
            ax.plot(
                column(rows, "posture_scale_max"),
                column(rows, "following"),
                "o",
                color=color,
                markersize=3.5,
                alpha=0.6,
            )
            texts.append(f"{score(name, 'posture_scale_max', 'following', group_name(lr)):+.2f}")
        ax.set_xscale("log")
        ax.set_title(f"leak rate {lr:g}: rank correlation {' / '.join(texts)}", color=TITLE_COLOR, fontsize=9)
    axes[3].tick_params(labelbottom=True)
    for ax in (*axes[3:7],):
        ax.set_xlabel("largest scaling of a posture's deviation", color=TEXT_COLOR)
    for ax in axes[[0, 4]]:
        ax.set_ylabel("following", color=TEXT_COLOR)
    axes[-1].axis("off")
    axes[-1].legend(handles=set_handles(), loc="center", frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- held still on the robot


def plot_release() -> Figure:
    """The robot's arrival after a release against the ESN's own, and where the releases were."""
    rows = read_csv("release", "release.csv")
    robot, own = column(rows, "arrival_s"), column(rows, "own_arrival_s")
    never = 66.0  # where a release that never arrives within 60 s is drawn
    fig = new_figure(
        13.5,
        6.4,
        "Released on the robot at each unstable fixed point within 30° of the path of the 620 passing ESNs (810),"
        "\nafter 4 s at rest there: when the hand arrives, from the release",
    )
    left, right = fig.subplots(1, 2, width_ratios=[1.0, 1.1])
    style(left)
    left.plot([0, 60], [0, 60], color=TEXT_COLOR, linewidth=1.0, linestyle=":")
    left.plot(
        np.where(np.isfinite(own), own, never),
        np.where(np.isfinite(robot), robot, never),
        "o",
        color=ATTRACTING_COLOR,
        markersize=3.5,
        alpha=0.5,
    )
    neither = int((~np.isfinite(robot) & ~np.isfinite(own)).sum())
    left.annotate(
        f"{neither}: neither arrives",
        (never, never),
        xytext=(-8, -16),
        textcoords="offset points",
        ha="right",
        color=TEXT_COLOR,
        fontsize=9,
    )
    left.axhline(60.0, color=GRID_COLOR)
    left.axvline(60.0, color=GRID_COLOR)
    left.set_xlim(0, 70)
    left.set_ylim(0, 70)
    left.set_xticks([0, 10, 20, 30, 40, 50, 60, never], ["0", "10", "20", "30", "40", "50", "60", "never"])
    left.set_yticks([0, 10, 20, 30, 40, 50, 60, never], ["0", "10", "20", "30", "40", "50", "60", "never"])
    left.set_xlabel("the ESN on its own (s)", color=TEXT_COLOR)
    left.set_ylabel("on the robot (s)", color=TEXT_COLOR)
    left.set_aspect("equal")
    style(right)
    draw_take_in_joint_space(right)
    q = np.column_stack([column(rows, "q1_deg"), column(rows, "q2_deg")])
    arrives = np.isfinite(robot)
    norm = Normalize(0.0, 30.0)
    points = right.scatter(q[arrives, 0], q[arrives, 1], c=robot[arrives], cmap="viridis", norm=norm, s=14, zorder=5)
    right.plot(
        q[~arrives, 0], q[~arrives, 1], "X", color=UNSTABLE_COLOR, markersize=7, markeredgecolor="white", zorder=6
    )
    near_take_axes(right)
    fig.colorbar(points, ax=right, label="arrival on the robot (s)", shrink=0.8, extend="max")
    right.legend(
        handles=[
            Line2D([], [], linestyle="none", marker="X", color=UNSTABLE_COLOR, label="never arrives within 60 s"),
            *take_handles(),
        ],
        loc="lower left",
        frameon=False,
        labelcolor=TEXT_COLOR,
        fontsize=9,
    )
    return fig


def plot_blocks() -> Figure:
    """The blocks along the reach: the share failing, and the delay, by onset, hold, and group."""
    rows = read_csv("release", "blocks.csv")
    onset, hold, near = (
        column(rows, "onset_s"),
        column(rows, "hold_s"),
        np.array([r["near_path_unstable"] for r in rows]),
    )
    passes, delay = np.array([r["passes"] for r in rows]), column(rows, "delay_s")
    onsets = np.unique(onset)
    fig = new_figure(
        13.5,
        5.4,
        "The 620 passing ESNs on the robot, the tip held for 1 s or 3 s from each onset, one block a run (8,680 runs)",
    )
    left, right = fig.subplots(1, 2)
    for ax in (left, right):
        style(ax)
    for h, dash in ((1.0, "-"), (3.0, "--")):
        for group, color in ((near, NEAR_COLOR), (~near, NONE_COLOR)):
            s = (hold == h) & group
            left.plot(onsets, [(~passes[s & (onset == o)]).mean() for o in onsets], dash, marker="o", color=color)
            q = np.array([np.percentile(delay[s & (onset == o) & np.isfinite(delay)], [25, 50, 75]) for o in onsets])
            right.plot(onsets + (0.08 if h == 3.0 else -0.08), q[:, 1], dash, marker="o", color=color)
            right.vlines(onsets + (0.08 if h == 3.0 else -0.08), q[:, 0], q[:, 2], color=color, linewidth=1.0)
    left.set_title("failing report 011's rule under a block", color=TITLE_COLOR, fontsize=10)
    left.set_ylabel("share failing", color=TEXT_COLOR)
    right.set_title(
        "the delay: the arrival after the release, less the take's\nfrom the progress held (median and quartiles)",
        color=TITLE_COLOR,
        fontsize=10,
    )
    right.set_ylabel("delay (s)", color=TEXT_COLOR)
    right.axhline(0.0, color=TEXT_COLOR, linewidth=1.0, linestyle=":")
    for ax in (left, right):
        ax.set_xlabel("onset of the block (s)", color=TEXT_COLOR)
        ax.set_xticks(onsets)
    handles = [
        Line2D([], [], color=NEAR_COLOR, marker="o", label="with an unstable fixed point within 30° of the path (420)"),
        Line2D([], [], color=NONE_COLOR, marker="o", label="without (200)"),
        Line2D([], [], color=TEXT_COLOR, linestyle="-", label="held 1 s"),
        Line2D([], [], color=TEXT_COLOR, linestyle="--", label="held 3 s"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def shown_log(scenario: str, arm: str) -> tuple[StateLog, float]:
    """A shown run's log, and its duration (s): the replay's from ESN #1's run."""
    esn = "esn2" if arm == "esn2" else "esn1"
    run = run_dir(f"{scenario}_{esn}")
    with (run / "config.toml").open("rb") as f:
        duration = float(tomllib.load(f)["evaluation"]["duration"])
    name = "replay" if arm == "replay" else "esn"
    return StateLog.load(run / SETTING / f"{name}_00.sklog.npz"), duration


def shown_channel(log: StateLog, duration: float, channel: str) -> NDArray[np.float64]:
    if channel not in log.channel_names:
        return np.zeros((round(duration / DT) + 1, 2))
    return task_joint_angles(log, DT, duration, channel)


def shown_handles(*, references: bool) -> list[Line2D]:
    handles = [Line2D([], [], color=color, linewidth=1.4, label=label) for _, label, color in ARMS]
    if references:
        handles += [
            Line2D([], [], color=TEXT_COLOR, linewidth=1.0, linestyle="--", label="the ESN's reference"),
            Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught motion"),
        ]
    return handles


def disturbance_span(scenario: str) -> tuple[float, float] | None:
    with (run_dir(f"{scenario}_esn1") / "config.toml").open("rb") as f:
        table = tomllib.load(f).get("disturbance")
    return None if table is None else (float(table["onset"]), float(table["release"]))


def plot_shown(channel: str) -> Figure:
    """The joint angles or torques of ESN #1, ESN #2, and the replay in the four shown scenarios."""
    what = "Joint angles" if channel == "q" else "Joint torques"
    fig = new_figure(
        15.0,
        9.6,
        f"{what} on the robot, joint PD at ω 20, ζ 0.4, given the reference velocity: ESN #1 (blue), ESN #2 (purple),"
        " and the replay (orange)" + ("; dashed: the ESNs' references" if channel == "q" else ""),
    )
    axes = fig.subplots(3, len(SCENARIOS), sharex=True, sharey="row", height_ratios=[0.6, 1.0, 1.0])
    q_taught = np.degrees(taught()[0])
    t_taught = DT * np.arange(len(q_taught))
    for col, (scenario, title) in enumerate(SCENARIOS):
        span = disturbance_span(scenario)
        for row in range(3):
            style(axes[row, col])
            if span is not None:
                axes[row, col].axvspan(*span, color=DISTURBANCE_COLOR, zorder=0)
        axes[0, col].set_title(title, color=TITLE_COLOR, fontsize=10)
        for key, _, color in ARMS:
            log, duration = shown_log(scenario, key)
            times = DT * np.arange(round(duration / DT) + 1)
            axes[0, col].plot(
                times,
                np.linalg.norm(shown_channel(log, duration, "ext_force"), axis=1),
                color=color,
                linewidth=1.0,
                zorder=ARM_ZORDERS[key],
            )
            values = shown_channel(log, duration, channel)
            reference = shown_channel(log, duration, "q_ref") if channel == "q" and key != "replay" else None
            for joint in range(2):
                ax = axes[1 + joint, col]
                scale = np.degrees if channel == "q" else (lambda v: v)
                ax.plot(times, scale(values[:, joint]), color=color, linewidth=1.2, zorder=ARM_ZORDERS[key])
                if reference is not None:
                    ax.plot(
                        times, np.degrees(reference[:, joint]), color=color, linewidth=1.0, linestyle="--", zorder=2.5
                    )
        if channel == "q":
            for joint in range(2):
                axes[1 + joint, col].plot(t_taught, q_taught[:, joint], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
        axes[-1, col].set_xlim(0, 30)
    axes[0, 0].set_ylabel("tip force (N)", color=TEXT_COLOR)
    unit = "deg" if channel == "q" else "N m"
    for joint in range(2):
        axes[1 + joint, 0].set_ylabel(f"joint {joint + 1} ({unit})", color=TEXT_COLOR)
    handles = shown_handles(references=channel == "q")
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_shown_paths() -> Figure:
    """The hand paths of ESN #1, ESN #2, and the replay in the four shown scenarios."""
    fig = new_figure(12.0, 12.5, "The hand paths of the same runs, from 0 s to the run's end")
    axes = fig.subplots(len(SCENARIOS), 1, sharex=True, sharey=True)
    target, radius = goal()
    hand_taught = taught()[1]
    for ax, (scenario, title) in zip(axes, SCENARIOS, strict=True):
        style(ax)
        ax.plot(hand_taught[:, 0], hand_taught[:, 1], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
        ax.add_patch(Circle(tuple(target), radius, facecolor="none", edgecolor=TITLE_COLOR, linewidth=1.0, zorder=6))
        for key, _, color in ARMS:
            log, duration = shown_log(scenario, key)
            hand = endpoint_positions(skeleton(), shown_channel(log, duration, "q"))
            ax.plot(hand[:, 0], hand[:, 1], color=color, linewidth=1.2, zorder=ARM_ZORDERS[key])
            ax.plot(*hand[0], "o", color=color, markersize=5, markerfacecolor="white", zorder=7)
        ax.set_title(title.replace("\n", " "), color=TITLE_COLOR, fontsize=10)
        ax.set_aspect("equal")
        ax.set_ylabel("y (m)", color=TEXT_COLOR)
    axes[-1].set_xlabel("x (m)", color=TEXT_COLOR)
    handles = [
        *shown_handles(references=False),
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the taught hand path"),
        Line2D(
            [], [], linestyle="none", marker="o", color=TEXT_COLOR, markerfacecolor="white", label="the hand at 0 s"
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=9,
            color=TITLE_COLOR,
            markerfacecolor="none",
            label="the goal radius",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- the tables


def print_screen() -> None:
    """Step 1's measures by leak rate, of seeds 0 to 9 and of seeds 10 to 19."""
    radius = goal()[1]
    for name, label in (("screen", "seeds 0 to 9"), ("screen_heldout", "seeds 10 to 19")):
        rows = read_csv(name, "screen.csv")
        fixed: dict[tuple[float, float, float, int], list[dict[str, Any]]] = {}
        for p in read_csv(name, "fixed_points.csv"):
            fixed.setdefault(key_of(p), []).append(p)
        print(f"\nThe screen, {label}: per leak rate (150 ESNs each)")
        print("| | " + " | ".join(f"{v:g}" for v in LEAK_RATES) + " |")
        print("| --- |" + " ---: |" * len(LEAK_RATES))
        groups = leak_rate_groups(rows)

        def line(title: str, values: NDArray[Any], how: str, groups: list[NDArray[np.bool_]] = groups) -> None:
            digits = 1 if how == "deg" else 2
            cells = [
                f"{values[g].sum():.0f}" if how == "count" else f"{np.median(values[g]):.{digits}f}" for g in groups
            ]
            print(f"| {title} | " + " | ".join(cells) + " |")

        in_goal = np.array(
            [
                any(p["target_distance_m"] <= radius and p["n_unstable"] == 0 for p in fixed.get(key_of(r), []))
                for r in rows
            ]
        )
        line("an attracting fixed point within the goal", in_goal, "count")
        line("an attracting fixed point off the target", column(rows, "attracting_off_target") > 0, "count")
        line("an unstable fixed point", column(rows, "unstable") > 0, "count")
        line("... within 30° of the path", column(rows, "nearest_unstable_path_distance_deg") <= 30.0, "count")
        released = (column(rows, "unstable") > 0) & ~(column(rows, "release_arrival_s") <= 9.58)
        line("... released, arriving after 9.58 s or never", released, "count")
        line("all 144 starts from rest end at the target", column(rows, "starts_at_target") == 144, "count")
        line("median lead near the path (deg)", column(rows, "lead_median_deg"), "deg")
        tc = column(rows, "target_time_constant_s")
        cells = [f"{np.median(tc[g & np.isfinite(tc)]):.2f}" for g in groups]
        print("| the target's slowest time constant (s), median | " + " | ".join(cells) + " |")
        line("largest scaling of a posture's deviation, median", column(rows, "posture_scale_max"), "x")
        held = column(rows, "held_unconverged") > 0
        multiple = column(rows, "held_multiple") > 0
        rho = column(rows, "spectral_radius")
        print(
            f"held reservoir not settling at some posture: {held.sum():.0f} ESNs ({held[rho == 1.75].sum():.0f} at"
            f" spectral radius 1.75); another held fixed point: {multiple.sum():.0f}"
            f" ({multiple[rho == 1.75].sum():.0f})"
        )
        fps = read_csv(name, "fixed_points.csv")
        print(
            f"fixed points: {len(fps)}, {sum(not p['from_grid'] for p in fps)} only where runs come to rest;"
            f" time per ESN, median {np.median(column(rows, 'elapsed_s')):.0f} s"
        )


def print_check() -> None:
    """The 5° check against the 10° grid, both with the runs from rest 20 s long."""
    fine: dict[tuple[float, float, float, int], list[NDArray[np.float64]]] = {}
    coarse: dict[tuple[float, float, float, int], list[NDArray[np.float64]]] = {}
    for name, store in (("check_5deg", fine), ("screen_20s", coarse)):
        for p in read_csv(name, "fixed_points.csv"):
            if int(p["seed"]) == 0:
                store.setdefault(key_of(p), []).append(np.array([p["q1_deg"], p["q2_deg"]]))
    same = lambda q, qs: any(np.abs(q - o).max() < 1e-4 for o in qs)  # noqa: E731
    missed = sum(not same(q, coarse.get(k, [])) for k in fine for q in fine[k])
    extra = sum(not same(q, fine.get(k, [])) for k in coarse for q in coarse[k])
    print(
        f"\nThe 5° check: {sum(map(len, fine.values()))} fixed points every 5°, {sum(map(len, coarse.values()))} every"
        f" 10°; missed at 10°: {missed}; found only at 10°: {extra}"
    )


def print_rules() -> None:
    """The rules' counts over all the ESNs, and the areas under the ROC curve for failing."""
    for name, _, label, _ in SETS:
        print(f"\nThe rules, {label}")
        print("| Rule | Flagged | Its outcome caught | Failing caught | Passing rejected | Qualifying rejected |")
        print("| --- | ---: | --- | ---: | ---: | ---: |")
        for rule in ("a", "a_far", "b", "c"):
            r = rule_row(name, rule, "all")
            print(
                f"| ({rule}) | {r['flagged']:.0f} | {r['outcome']}, {r['outcome_caught']:.0f} of"
                f" {r['outcome_esns']:.0f} | {r['failing_caught']:.0f} of {r['failing']:.0f} |"
                f" {r['passing_rejected']:.0f} of {r['passing']:.0f} |"
                f" {r['qualifying_rejected']:.0f} of {r['qualifying']:.0f} |"
            )
        print("| Rule | " + " | ".join(f"{v:g}" for v in LEAK_RATES) + " |")
        for rule in ("a", "b", "c"):
            cells = []
            for lr in LEAK_RATES:
                r = rule_row(name, rule, group_name(lr))
                flagged = "none" if np.isnan(r["failing_share_flagged"]) else f"{r['failing_share_flagged']:.2f}"
                others = "none" if np.isnan(r["failing_share_unflagged"]) else f"{r['failing_share_unflagged']:.2f}"
                cells.append(f"{flagged} / {others}")
            print(f"| ({rule}) | " + " | ".join(cells) + " |")
        aucs = [
            ("the leak rate", "leak rate"),
            ("the lead", "smaller lead"),
            ("(a)", "flag_a"),
            ("(b)", "flag_b"),
            ("(c)", "flag_c"),
        ]
        print("AUC for failing over all: " + ", ".join(f"{t} {score(name, m, 'failing', 'all'):.3f}" for t, m in aucs))
        print(
            f"rank correlation of the leak rate with the following: {score(name, 'leak rate', 'following', 'all'):+.2f}"
        )
        for measure, against in (
            ("smaller lead", "ends_away"),
            ("nearer unstable fixed point", "failing"),
            ("posture_scale_max", "following"),
            ("posture_scale_median", "following"),
            ("across_rate_median_per_s", "following"),
            ("release arrival", "arrival_block_s"),
            ("release arrival", "arrival_nominal_s"),
            ("across_rate_max_per_s", "return_s"),
        ):
            cells = [f"{score(name, measure, against, group_name(lr)):+.2f}" for lr in LEAK_RATES]
            print(
                f"{measure} vs {against}: all {score(name, measure, against, 'all'):+.2f}; within " + ", ".join(cells)
            )
        print(
            "posture_scale_max vs following, pooled within:"
            f" {score(name, 'posture_scale_max', 'following', 'within each leak rate'):+.2f}"
        )


def print_claims() -> None:
    """Step 4's claims, of the held-out ESNs."""
    name = "against_heldout"
    excess = {}
    for lr in LEAK_RATES:
        r = rule_row(name, "a", group_name(lr))
        excess[lr] = r["failing_share_flagged"] - r["failing_share_unflagged"]
    first = all(excess[lr] > 0 for lr in (0.015, 0.02, 0.07, 0.1))
    second = all(excess[lr] < excess[other] for lr in (0.04, 0.05) for other in (0.015, 0.02, 0.07, 0.1))
    print("\nStep 4's claims, of seeds 10 to 19")
    print(
        f"1. (a)'s excess by leak rate: {', '.join(f'{lr:g} {e:+.2f}' for lr, e in excess.items())};"
        f" larger at the four: {first}; smaller at 0.04 and 0.05 than at each of them: {second}"
    )
    second_claim = [score(name, "posture_scale_max", "following", group_name(lr)) for lr in LEAK_RATES[:5]]
    print(f"2. {', '.join(f'{v:+.2f}' for v in second_claim)}: holds {all(v >= 0.5 for v in second_claim)}")
    third = [score(name, "nearer unstable fixed point", "failing", group_name(lr)) for lr in LEAK_RATES]
    print(f"3. {', '.join(f'{v:.2f}' for v in third)}: holds {all(v > 0.5 for v in third)}")
    lr_auc = score(name, "leak rate", "failing", "all")
    others = {m: score(name, m, "failing", "all") for m in ("smaller lead", "flag_a", "flag_b", "flag_c")}
    print(
        f"4. the leak rate {lr_auc:.3f}; {', '.join(f'{m} {v:.3f}' for m, v in others.items())}:"
        f" holds {all(v <= lr_auc for v in others.values())}"
    )


def print_named() -> None:
    """The named ESNs: the screen's measures and the robot's outcomes."""
    screen = screen_by_key("screen")
    esns = {key_of(r): r for r in read_csv("against", "esns.csv")}
    print("\nThe named ESNs")
    print("| | ESN #1 | ESN #2 | ESN #3 | report 007's |")
    print("| --- | --- | --- | --- | --- |")
    rows = [
        (
            "fixed points (attracting, unstable)",
            lambda s, e: f"{s['fixed_points']:.0f} ({s['attracting']:.0f}, {s['unstable']:.0f})",
        ),
        (
            "the target's slowest mode",
            lambda s, e: f"{s['target_time_constant_s']:.2f} s, period {s['target_period_s']:.1f} s",
        ),
        (
            "the nearest unstable fixed point to the path",
            lambda s, e: (
                "none"
                if not np.isfinite(s["nearest_unstable_path_distance_deg"])
                else f"{s['nearest_unstable_path_distance_deg']:.1f}°, {s['nearest_unstable_progress']:.2f} of the way"
            ),
        ),
        (
            "released there after 4 s, arriving",
            lambda s, e: (
                ""
                if not np.isfinite(s["release_arrival_s"]) and s["unstable"] == 0
                else f"{s['release_arrival_s']:.2f} s"
            ),
        ),
        ("median lead near the path", lambda s, e: f"{s['lead_median_deg']:.1f}°"),
        ("largest scaling of a posture's deviation", lambda s, e: f"{s['posture_scale_max']:.2f}"),
        ("flagged by", lambda s, e: ", ".join(f"({c[5:]})" for c in ("flag_a", "flag_b", "flag_c") if e[c]) or "none"),
        ("on the robot", lambda s, e: f"{'passes' if e['passes'] else 'fails'}, following {e['following']:.2f}"),
    ]
    for title, cell in rows:
        print(f"| {title} | " + " | ".join(cell(screen[k], esns[k]) for k in NAMED.values()) + " |")


def print_release() -> None:
    """Step 3's releases and blocks."""
    rows = read_csv("release", "release.csv")
    robot, own = column(rows, "arrival_s"), column(rows, "own_arrival_s")
    both = np.isfinite(robot) & np.isfinite(own)
    gap = robot[both] - own[both]
    print(
        f"\nReleases: {len(rows)}; both arrive {both.sum()}, neither {(~np.isfinite(robot) & ~np.isfinite(own)).sum()};"
        f" robot less own: median {np.median(gap):+.2f} s, 10th to 90th {np.percentile(gap, 10):+.2f} to"
        f" {np.percentile(gap, 90):+.2f} s; later than 9.58 s: robot {(robot > 9.58).sum()}, own {(own > 9.58).sum()}"
    )
    never = [r for r in rows if not np.isfinite(r["arrival_s"])]
    keys = {key_of(r) for r in never}
    stays = column(never, "stays_s")
    print(
        f"never on the robot: {len(never)} in {len(keys)} ESNs, by leak rate "
        f"{', '.join(f'{lr:g} {sum(k[0] == lr for k in keys)}' for lr in LEAK_RATES)}; leaving within 5 s"
        f" {(stays < 5).sum()}, staying 60 s {np.isinf(stays).sum()}; growth median"
        f" {np.median(column(never, 'growth_per_s')):.2f} /s (the others"
        f" {np.median(column([r for r in rows if np.isfinite(r['arrival_s'])], 'growth_per_s')):.2f})"
    )
    blocks = read_csv("release", "blocks.csv")
    near, hold = np.array([b["near_path_unstable"] for b in blocks]), column(blocks, "hold_s")
    passes, onset = np.array([b["passes"] for b in blocks]), column(blocks, "onset_s")
    for h in (1.0, 3.0):
        print(
            f"blocks held {h:g} s: failing with / without an unstable fixed point near the path"
            f" {(~passes[(hold == h) & near]).mean():.3f} / {(~passes[(hold == h) & ~near]).mean():.3f};"
            " by onset "
            + ", ".join(f"{o:g} {(~passes[(hold == h) & (onset == o)]).mean():.3f}" for o in np.unique(onset))
        )


def print_shown() -> None:
    """The shown runs' arrival, settling, and peak torque."""
    print("\nThe shown runs: arrival, settling, peak torque over the effort window")
    for scenario, title in SCENARIOS:
        cells = []
        for key, label, _ in ARMS:
            esn = "esn2" if key == "esn2" else "esn1"
            (row,) = [
                r
                for r in read_csv(f"{scenario}_{esn}", "metrics.csv")
                if r["arm"] == ("replay" if key == "replay" else "esn") and r["damping"] == 0.4
            ]
            cells.append(
                f"{label} {row['arrival_time_s']:.2f} s, settles {row['settling_time_s']:.2f} s,"
                f" {row['peak_torque_nm']:.1f} N m"
            )
        print(f"{title.splitlines()[0]}: " + "; ".join(cells))


if __name__ == "__main__":
    main()
