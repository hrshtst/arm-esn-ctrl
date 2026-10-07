# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 004.

    uv run python reports/004-manual-demonstration/make_figures.py
    uv run python reports/004-manual-demonstration/make_figures.py --logs --animations

From the take in ``data/`` and the copies of the runs under ``results/``
(configurations, run records, and per-run metrics), it draws into
``results/summary``:

- ``take.png``: the take taught by hand, as recorded and filtered (Section 3.1);
- ``settings003.png``: report 003's three settings over the grid of start offsets,
  trained on either take (Section 3.2);
- ``sweeps.png``: what the sweeps trade: failures, swings, and jumps (Section 3.4);
- ``region.png``: past the edge, where no run fails, swings, or jumps (Section 3.4);
- ``robot.png``: candidate F and the replay on the robot from the 169 start offsets
  (Section 3.6);

and prints the tables of the report. With ``--logs``, it reads the run logs, which
are not kept in Git, from the runs under the storage root, and also draws
``candidates.png`` (the hand paths of the candidates, Section 3.5), ``dwell.png``
(how candidate F settles at the target, Section 3.6), and ``pushes.png`` (candidate F
pushed and held, Section 3.7), and writes ``candidates.csv``, ``dwell.csv``, and
``pushes.csv``, from which their tables are printed (also without the option). It
also draws the joint angles of the examples, scenario by scenario, with the trackers
critically damped (``example_<name>.png``, Section 3.8) and underdamped
(``underdamped_<name>.png``, Section 3.9). With ``--animations``, it animates each
example, the arms side by side, each rendered by skelarm's player: with the trackers
critically damped (``example_<name>.gif``, Section 3.8) and with joint PD underdamped
(``underdamped_<name>.gif``, Section 3.9).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.demonstrations import endpoint_positions, load_joint_angles
from arm_esn_ctrl.metrics import distances_to_path, path_progress
from arm_esn_ctrl.storage import resolve_run_path

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
TAKE_RUN = REPORT / "data" / "20261006-152823-reach_manual_single"
SKELETON = Skeleton.from_toml(TAKE_RUN / "config.toml")
TARGET = np.array([0.0, 1.2])
RADIUS = 0.02  # the goal: within this distance of the target (m)
DT = 0.01  # the ESNs' period (s)

# Report 003's settings (Section 3.2): key, name, and configuration stem.
SETTINGS = {
    "first": ("first settings", "grid_single_demo_settings"),
    "eight": ("eight-demonstration settings", "grid_multi_demo_settings"),
    "tuned": ("tuned settings", "grid_tuned_settings"),
}
TAKES = {"raw": "as recorded", "filtered": "filtered"}
# The candidates of the sweeps (Section 3.5): name and settings (leak rate, spectral radius, input scaling, ridge).
CANDIDATES = {
    "A": (0.5, 1.2, 0.03, 1e-4),
    "B": (0.2, 0.6, 1.0, 1.0),
    "C": (0.7, 1.05, 0.1, 1e-3),
    "D": (0.5, 1.05, 0.1, 1e-3),
    "E": (0.7, 1.05, 0.2, 3e-3),
    "F": (0.7, 1.05, 0.3, 1e-2),
}
SHOWN_CANDIDATES = ["A", "B", "C", "E", "F"]
MAX_DETOUR = 0.05  # m: a run that strays from the taught path by more than this beyond its start swings
MAX_FIRST_STEP = 0.03  # m: a first step longer than this is a jump
# The robot (Sections 3.6 and 3.7): tracker settings of candidate F's runs: label, configuration suffix,
# and the directory of the setting in the run.
TRACKERS = {
    "computed torque, ω = 10": ("", "computed_torque_w10"),
    "joint PD, ω = 10": ("", "pd_w10"),
    "joint PD, ω = 20": ("_pd_gains", "pd_w20"),
    "joint PD, ω = 40": ("_pd_gains", "pd_w40"),
}
UNDERDAMPED = {
    "computed torque, ζ = 0.5": ("_damping", "computed_torque_w10_z0.5"),
    "computed torque, ζ = 0.3": ("_damping", "computed_torque_w10_z0.3"),
    "computed torque, ζ = 0.1": ("_damping", "computed_torque_w10_z0.1"),
    "joint PD ω = 20, ζ = 0.1": ("_damping", "pd_w20_z0.1"),
}
SCENARIOS = ["nominal", "offsets", "push_across", "push_forward", "push_backward", "block"]
PUSHES = {"forward (10 N)": "push_forward", "backward (10 N)": "push_backward", "across (5 N)": "push_across"}
PUSH_TIME = 4.32  # s: when the pushes start
DWELL_AFTER = 1.0  # s: the dwell is measured from this long after the arrival to the end of the run
CANDIDATES_CSV = SUMMARY / "candidates.csv"
DWELL_CSV = SUMMARY / "dwell.csv"
PUSHES_CSV = SUMMARY / "pushes.csv"

ESN_COLOR = "#2a78d6"
REPLAY_COLOR = "#eb6834"
TAUGHT_COLOR = "#eb6834"
RECORDED_COLOR = "#a3a29d"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
OUTCOME_COLORS = ["#2a78d6", "#eb6834", "#52514e"]  # arrive and hold, leave the goal, never arrive
OUTCOMES = ["arrive and hold", "leave the goal", "never arrive"]
# The sweeps: one color per ridge, light to dark as the ridge grows (sequential).
RIDGE_COLORS = ["#9ec5f0", "#5a9be3", "#2a78d6", "#174e94"]
# What a combination of the sweep past the edge does: no run fails, swings, or jumps; some swing or jump; some fail.
REGION_COLORS = ["#2a78d6", "#9ec5f0", "#d9d8d3"]
REGION_LABELS = ["none fails, swings, or jumps", "none fails; some swing or jump", "some fail"]
PUSH_COLORS = {
    "undisturbed": "#52514e",
    "forward (10 N)": "#2a78d6",
    "backward (10 N)": "#4a3aa7",
    "across (5 N)": "#eb6834",
}
DWELL_COLORS = ["#2a78d6", "#4a3aa7", "#eb6834", "#52514e"]
UNDISTURBED_COLOR = "#9ec5f0"  # candidate F's undisturbed arm, under the disturbed one
DISTURBANCE_COLOR = "#f0efec"
# The examples, scenario by scenario (Sections 3.8 and 3.9): name, scenario, and start offset (deg) from the
# demonstrated start, as in report 003.
EXAMPLES = [
    ("nominal", "nominal", (0.0, 0.0)),
    ("offset ahead", "offsets", (10.0, -10.0)),  # roughly along the taught motion
    ("offset across", "offsets", (-10.0, -10.0)),  # mostly across it
    ("push across", "push_across", (0.0, 0.0)),
    ("push forward", "push_forward", (0.0, 0.0)),
    ("push backward", "push_backward", (0.0, 0.0)),
    ("block", "block", (0.0, 0.0)),
]
# When each disturbance acts (task time, s), and what it is.
SPANS = {"push_across": (4.32, 4.42), "push_forward": (4.32, 4.42), "push_backward": (4.32, 4.42), "block": (4.1, 4.6)}
DISTURBANCES = {"push_across": "pushed (5 N)", "push_forward": "pushed (10 N)", "push_backward": "pushed (10 N)"}
# The tracker settings of the examples' figures: label, configuration suffix, and setting directory.
EXAMPLE_SETTINGS = [
    ("computed torque, ω = 10 rad/s", "", "computed_torque_w10"),
    ("joint PD, ω = 20 rad/s", "_pd_gains", "pd_w20"),
    ("joint PD, ω = 40 rad/s", "_pd_gains", "pd_w40"),
]
UNDERDAMPED_SETTINGS = [
    ("computed torque, ω = 10 rad/s, ζ = 0.1", "_damping", "computed_torque_w10_z0.1"),
    ("joint PD, ω = 20 rad/s, ζ = 0.1", "_damping", "pd_w20_z0.1"),
]
DAMPINGS = [1.0, 0.5, 0.3, 0.1]
EXAMPLE_END = 16.0  # the joint angles are drawn until this task time (s): the whole run
# The animations of the examples: the panels (label, arm, configuration suffix, setting directory, label color),
# the task times they cover (s), and their frame rate; a whole number of milliseconds per frame, so that GIFs play
# in real time.
ANIMATION_PANELS = [
    ("F, joint PD 20 rad/s", "esn", "_pd_gains", "pd_w20", ESN_COLOR),
    ("replay, joint PD 20 rad/s", "replay", "_pd_gains", "pd_w20", REPLAY_COLOR),
    ("F, computed torque 10 rad/s", "esn", "", "computed_torque_w10", "#4a3aa7"),
]
# The animations of the underdamped examples: candidate F and the replay, both with joint PD underdamped.
UNDERDAMPED_ANIMATION_PANELS = [
    ("F, joint PD 20 rad/s", "esn", "_damping", "pd_w20_z0.1", ESN_COLOR),
    ("replay, joint PD 20 rad/s", "replay", "_damping", "pd_w20_z0.1", REPLAY_COLOR),
]
UNDERDAMPED_NOTE = "tracker underdamped: damping ratio 0.1"
ANIMATION_SPAN = (-0.2, 16.0)
ANIMATION_FPS = 10.0
ANIMATION_HOLD_MS = 1500  # the last frame stays this long before the GIF loops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also analyze the run logs in the storage root")
    parser.add_argument("--animations", action="store_true", help="also export GIFs from the logs in the storage root")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {
        "take.png": plot_take(),
        "settings003.png": plot_settings003(),
        "sweeps.png": plot_sweeps(),
        "region.png": plot_region(),
        "robot.png": plot_robot(),
    }
    if args.logs:
        figures["candidates.png"] = plot_candidates()
        figures["dwell.png"] = plot_dwell()
        figures["pushes.png"] = plot_pushes()
        for name, scenario, offset in EXAMPLES:
            figures[f"example_{slug(name)}.png"] = plot_example(name, scenario, offset, EXAMPLE_SETTINGS)
            figures[f"underdamped_{slug(name)}.png"] = plot_example(name, scenario, offset, UNDERDAMPED_SETTINGS)
        write_candidates()
        write_dwell()
        write_pushes()
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=150)
    print_tables()
    if args.animations:
        for name, scenario, offset in EXAMPLES:
            export_example_animation(name, scenario, offset, ANIMATION_PANELS, "example")
            export_example_animation(
                name, scenario, offset, UNDERDAMPED_ANIMATION_PANELS, "underdamped", UNDERDAMPED_NOTE
            )
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


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [{key: value(text) for key, text in row.items()} for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_name(stem: str) -> str:
    """The name of the one run of a configuration, by its stem."""
    matches = sorted(p.name for p in RESULTS.glob(f"*-{stem}") if p.is_dir())
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


def rows_of(stem: str, name: str = "metrics.csv") -> list[dict[str, Any]]:
    return read_csv(RESULTS / run_name(stem) / name)


def offset_of(origin: str) -> tuple[float, float]:
    """The start offset (deg) in a start's origin, such as "demo_00 +2.5,-5 deg"; (0, 0) for "demo_00"."""
    match = re.search(r"([+-][\d.]+),([+-][\d.]+) deg", origin)
    return (0.0, 0.0) if match is None else (float(match[1]), float(match[2]))


def demonstrated(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The run from the demonstrated start: the start without an offset in its origin."""
    return next(r for r in rows if " deg" not in r["origin"])


def setting_of(row: dict[str, Any]) -> str:
    """The directory of a robot run's tracker setting, as robot_esn.py names it."""
    name = f"{row['law']}_w{row['omega']:g}"
    return name if row["damping"] == 1.0 else f"{name}_z{row['damping']:g}"


def log_of(stem: str, *parts: str) -> StateLog:
    """A log of a run, read from the storage root."""
    return StateLog.load(resolve_run_path(f"results/{run_name(stem)}").joinpath(*parts))


def hand_of(log: StateLog) -> NDArray[np.float64]:
    return endpoint_positions(SKELETON, log.channel("q").reshape(len(log.times), -1))


def taught() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The take as recorded at the ESNs' period: joint angles and hand positions."""
    q = load_joint_angles(TAKE_RUN / "demo_00.sklog.npz", DT)[1]
    return q, endpoint_positions(SKELETON, q)


def detours(hand: NDArray[np.float64], taught_hand: NDArray[np.float64], every: int = 5) -> float:
    """How much farther from the taught hand path a hand strays than where it started (m)."""
    distances = distances_to_path(hand[::every], taught_hand[::every])
    return float(distances.max() - distances[0])


# ---------------------------------------------------------------------------- figures without the logs


def sci(x: float) -> str:
    """A ridge as people write it: 1e-4, 3e-3, or 1."""
    if x >= 0.1:
        return f"{x:g}"
    exponent = int(np.floor(np.log10(x)))
    return f"{x / 10**exponent:g}e{exponent}"


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def draw_map(ax: Axes, offsets: list[tuple[float, float]], values: list[float], norm: Normalize, cmap: Any) -> Any:
    """Draw values over the grid of start offsets (deg), the demonstrated start marked with a cross."""
    xs, ys = sorted({o[0] for o in offsets}), sorted({o[1] for o in offsets})
    cell = dict(zip(offsets, values, strict=True))
    data = np.array([[cell.get((x, y), np.nan) for x in xs] for y in ys])
    step_x, step_y = xs[1] - xs[0], ys[1] - ys[0]
    extent = (xs[0] - step_x / 2, xs[-1] + step_x / 2, ys[0] - step_y / 2, ys[-1] + step_y / 2)
    image = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, norm=norm)
    ax.plot(0.0, 0.0, marker="+", markersize=10, color="#0b0b0b", markeredgewidth=1.5)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR, labelsize=8)
    return image


def ring(ax: Axes) -> None:
    """The goal around the target, and the target."""
    angle = np.linspace(0.0, 2 * np.pi, 100)
    ax.plot(TARGET[0] + RADIUS * np.cos(angle), TARGET[1] + RADIUS * np.sin(angle), color="#0b0b0b", linewidth=0.8)
    ax.plot(*TARGET, marker="+", markersize=10, color="#0b0b0b", markeredgewidth=1.5)


def plot_take() -> Figure:
    """The take as recorded and filtered: hand path, hand speed, and distance to the target."""
    fig = Figure(figsize=(14, 4.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("The take taught by hand, as recorded and filtered", color="#0b0b0b")
    ax_path, ax_speed, ax_target = fig.subplots(1, 3, width_ratios=[1, 1.4, 1.4])
    for ax in (ax_path, ax_speed, ax_target):
        style(ax)
    for name, color, width, label in (
        ("demo_00", RECORDED_COLOR, 2.2, "as recorded"),
        ("demo_00_filtered", ESN_COLOR, 1.2, "filtered (first-order low-pass, 8 Hz)"),
    ):
        log = StateLog.load(TAKE_RUN / f"{name}.sklog.npz")
        t, hand = log.times, hand_of(log)
        speed = np.linalg.norm(np.gradient(hand, t, axis=0), axis=1)
        ax_path.plot(*hand.T, color=color, linewidth=width, label=label)
        ax_speed.plot(t, speed, color=color, linewidth=width * 0.6)
        ax_target.plot(t, 1000 * np.linalg.norm(hand - TARGET, axis=1), color=color, linewidth=width)
    ring(ax_path)
    ax_path.set(title="Hand path", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax_path.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=8, loc="lower left")
    ax_speed.set(title="Hand speed: the cursor's whole pixels", xlabel="time (s)", ylabel="speed (m/s)", ylim=(0, 1.0))
    ax_target.axhline(1000 * RADIUS, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax_target.set(title="Hand to target (dotted: the goal)", xlabel="time (s)", ylabel="distance (mm)", ylim=(0, None))
    for ax in (ax_speed, ax_target):
        ax.axvspan(0.0, 4.0, color="#f0efec", zorder=0)
    ax_target.text(2.0, 420, "pause", ha="center", color=TEXT_COLOR, fontsize=9)
    return fig


def outcome(row: dict[str, Any]) -> int:
    """0: arrive and hold; 1: arrive, then leave the goal; 2: never arrive."""
    return 0 if row["success"] else (1 if row["arrived"] else 2)


def plot_settings003() -> Figure:
    """The outcome of report 003's settings over the grid of start offsets, trained on either take."""
    fig = Figure(figsize=(11, 7.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("Report 003's settings on the take: outcome over the 169 start offsets", color="#0b0b0b")
    axes = fig.subplots(2, 3)
    cmap = ListedColormap(OUTCOME_COLORS)
    norm = Normalize(-0.5, 2.5)
    for row_axes, (take, how) in zip(axes, TAKES.items(), strict=True):
        for ax, (name, stem) in zip(row_axes, SETTINGS.values(), strict=True):
            rows = rows_of(f"{stem}_{take}")
            draw_map(ax, [offset_of(r["origin"]) for r in rows], [outcome(r) for r in rows], norm, cmap)
            holds = sum(r["success"] for r in rows)
            ax.set_title(f"{name}, take {how}\n{holds} of {len(rows)} arrive and hold", color=TEXT_COLOR, fontsize=9)
            ax.set_xlabel("joint 1 offset (deg)", color=TEXT_COLOR, fontsize=8)
            ax.set_ylabel("joint 2 offset (deg)", color=TEXT_COLOR, fontsize=8)
    fig.legend(
        handles=[Patch(color=c, label=label) for c, label in zip(OUTCOME_COLORS, OUTCOMES, strict=True)],
        loc="outside lower center",
        ncol=3,
        frameon=False,
        labelcolor=TEXT_COLOR,
    )
    return fig


def sweep_runs(prefix: str) -> list[tuple[str, list[dict[str, Any]]]]:
    """The ridge (from the run name) and the rows of each sweep run whose configuration stem starts with prefix."""
    found = []
    for path in sorted(RESULTS.glob(f"*-{prefix}*")):
        ridge = re.search(r"_ridge_([0-9e.-]+)_", path.name)
        found.append((ridge[1] if ridge else "", read_csv(path / "sweep.csv")))
    return sorted(found, key=lambda item: float(item[0]) if item[0] else 0.0)  # by ridge, smallest first


def not_mild_share(prefix: str) -> dict[tuple[float, ...], float]:
    """Per combination of a sweep with runs.csv, the share of runs that swing or jump (%)."""
    shares = {}
    for path in sorted(RESULTS.glob(f"*-{prefix}*")):
        rows = read_csv(path / "runs.csv")
        names = [
            k for k in rows[0] if k not in ("start", "origin", "success", "first_step_m", "detour_m", "hold_error_m")
        ]
        ridge = re.search(r"_ridge_([0-9e.-]+)_", path.name)
        groups: dict[tuple[float, ...], list[dict[str, Any]]] = {}
        for r in rows:
            key = (*(float(r[n]) for n in names), *((float(ridge[1]),) if ridge else ()))
            groups.setdefault(key, []).append(r)
        for key, group in groups.items():
            bad = [r for r in group if r["detour_m"] > MAX_DETOUR or r["first_step_m"] > MAX_FIRST_STEP]
            shares[key] = 100.0 * len(bad) / len(group)
    return shares


def plot_sweeps() -> Figure:
    """The trade-offs of the sweeps: failed runs against jumps or not-mild runs, per combination; and the noise."""
    fig = Figure(figsize=(15, 4.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("The sweeps: what each combination trades (one point per combination)", color="#0b0b0b")
    ax_main, ax_around, ax_edge, ax_noise = fig.subplots(1, 4)
    for ax in (ax_main, ax_around, ax_edge, ax_noise):
        style(ax)

    for take, marker in (("raw", "o"), ("filtered", "^")):
        for i, ridge in enumerate(("1e-6", "1")):
            ((_, rows),) = [r for r in sweep_runs(f"sweep_main_ridge_{ridge}_{take}")]
            ax_main.scatter(
                [100 * r["failures"] / r["runs"] for r in rows],
                [100 * r["jumps"] / r["runs"] for r in rows],
                s=14,
                marker=marker,
                color=RIDGE_COLORS[2 * i + 1],
                alpha=0.7,
                label=f"ridge {ridge}, take {TAKES[take]}",
            )
    ax_main.set(
        title="Main three parameters (169 starts)", xlabel="runs that fail (%)", ylabel="first steps over 30 mm (%)"
    )
    ax_main.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=7)

    for ax, prefix, title in (
        (ax_around, "sweep_around_a_ridge_", "Around candidate A (49 starts)"),
        (ax_edge, "sweep_past_edge_ridge_", "Past the edge (49 starts)"),
    ):
        shares = not_mild_share(prefix)
        for i, (ridge, rows) in enumerate(sweep_runs(prefix)):
            names = [k for k in ("leak_rate", "spectral_radius", "input_scaling") if k in rows[0]]
            xs = [100 * r["failures"] / r["runs"] for r in rows]
            ys = [shares[(*(float(r[n]) for n in names), float(ridge))] for r in rows]
            ax.scatter(xs, ys, s=14, color=RIDGE_COLORS[i], alpha=0.8, label=f"ridge {ridge}")
        ax.set(title=title, xlabel="runs that fail (%)", ylabel="runs that swing or jump (%)")
        ax.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=7)

    rows = [r for r in rows_of("sweep_noise_raw_lr0.7_sr1.05_is0.1", "sweep.csv") if r["ridge"] == 1e-2]
    rows.sort(key=lambda r: r["teacher_noise_deg"])
    noise = [r["teacher_noise_deg"] for r in rows]
    for key, label, style_name in (("failures", "fail", "-"), ("swings", "swing", "--"), ("jumps", "jump", ":")):
        ax_noise.plot(
            noise,
            [100 * r[key] / r["runs"] for r in rows],
            color=ESN_COLOR,
            linestyle=style_name,
            marker="o",
            label=f"runs that {label}",
        )
    ax_noise.set(
        title="Teacher noise, candidate C's reservoir\n(ridge 1e-2, 49 starts)",
        xlabel="noise on the input (deg)",
        ylabel="runs (%)",
        ylim=(-3, 103),
    )
    ax_noise.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=7)
    return fig


def plot_region() -> Figure:
    """Past the edge: which combinations no run fails, swings, or jumps from, per ridge and leak rate."""
    sweeps = sweep_runs("sweep_past_edge_ridge_")
    leaks = sorted({r["leak_rate"] for _, rows in sweeps for r in rows})
    fig = Figure(figsize=(12, 7.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("Past the edge: where no run fails, swings, or jumps (49 starts)", color="#0b0b0b")
    axes = fig.subplots(len(leaks), len(sweeps), sharex=True, sharey=True)
    cmap = ListedColormap(REGION_COLORS)
    for row_axes, leak in zip(axes, leaks, strict=True):
        for ax, (ridge, rows) in zip(row_axes, sweeps, strict=True):
            scalings = sorted({r["input_scaling"] for r in rows})
            radii = sorted({r["spectral_radius"] for r in rows})
            grid = np.full((len(radii), len(scalings)), np.nan)
            for r in rows:
                if r["leak_rate"] != leak:
                    continue
                kind = 2 if r["failures"] else (0 if r["swings"] == 0 and r["jumps"] == 0 else 1)
                grid[radii.index(r["spectral_radius"]), scalings.index(r["input_scaling"])] = kind
            ax.imshow(grid, origin="lower", cmap=cmap, norm=Normalize(-0.5, 2.5), aspect="auto")
            ax.set_xticks(range(len(scalings)), [f"{s:g}" for s in scalings])
            ax.set_yticks(range(len(radii)), [f"{s:g}" for s in radii])
            ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR, labelsize=8)
            ax.set_title(f"ridge {ridge}, leak rate {leak:g}", color=TEXT_COLOR, fontsize=9)
            if ax in axes[-1]:
                ax.set_xlabel("input scaling", color=TEXT_COLOR, fontsize=8)
            if ax is row_axes[0]:
                ax.set_ylabel("spectral radius", color=TEXT_COLOR, fontsize=8)
    fig.legend(
        handles=[Patch(color=c, label=label) for c, label in zip(REGION_COLORS, REGION_LABELS, strict=True)],
        loc="outside lower center",
        ncol=3,
        frameon=False,
        labelcolor=TEXT_COLOR,
    )
    return fig


def offsets_rows(label: str) -> list[dict[str, Any]]:
    """Candidate F's robot runs from the 169 start offsets at one tracker setting, both arms."""
    suffix, setting = TRACKERS[label]
    return [r for r in rows_of(f"offsets_candidate_f{suffix}_raw") if setting_of(r) == setting]


def plot_robot() -> Figure:
    """Candidate F and the replay on the robot from the 169 start offsets, per tracker setting."""
    fig = Figure(figsize=(11, 4.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("Candidate F and the replay on the robot, from the 169 start offsets", color="#0b0b0b")
    ax_hold, ax_torque = fig.subplots(1, 2)
    for ax in (ax_hold, ax_torque):
        style(ax)
    labels = list(TRACKERS)
    x = np.arange(len(labels))
    for k, (arm, color, name) in enumerate((("esn", ESN_COLOR, "candidate F"), ("replay", REPLAY_COLOR, "replay"))):
        holds, torques = [], []
        for label in labels:
            rows = [r for r in offsets_rows(label) if r["arm"] == arm]
            holds.append(100.0 * np.mean([r["success"] for r in rows]))
            torques.append(float(np.mean([r["peak_torque_nm"] for r in rows])))
        ax_hold.bar(x + (k - 0.5) * 0.38, holds, width=0.36, color=color, label=name)
        ax_torque.bar(x + (k - 0.5) * 0.38, torques, width=0.36, color=color, label=name)
    for ax in (ax_hold, ax_torque):
        ax.set_xticks(x, [label.replace(", ", "\n") for label in labels], fontsize=8)
    ax_hold.set(title="Runs that arrive and hold", ylabel="% of 169 starts", ylim=(0, 105))
    ax_torque.set(title="Peak joint torque, mean over the starts", ylabel="N m (log scale)", yscale="log")
    handles, names = ax_hold.get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=2, frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- the logs


def candidate_runs(name: str) -> tuple[list[dict[str, Any]], list[NDArray[np.float64]]]:
    """A candidate's grid run: its metrics rows and the hand paths of its runs, in start order."""
    stem = f"grid_candidate_{name.lower()}_raw"
    rows = rows_of(stem)
    hands = [hand_of(log_of(stem, f"esn_{int(r['start']):02d}.sklog.npz")) for r in rows]
    return rows, hands


def plot_candidates() -> Figure:
    """The hand paths of the candidates from the 169 start offsets, and how far each strays beyond its start."""
    _, taught_hand = taught()
    fig = Figure(figsize=(17, 7.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("The candidates from the 169 start offsets: hand paths, and detours beyond the start", color="#0b0b0b")
    axes = fig.subplots(2, len(SHOWN_CANDIDATES))
    norm = Normalize(0.0, 300.0)
    for (ax_path, ax_map), name in zip(axes.T, SHOWN_CANDIDATES, strict=True):
        style(ax_path)
        rows, hands = candidate_runs(name)
        for hand in hands:
            ax_path.plot(*hand.T, color=ESN_COLOR, linewidth=0.5, alpha=0.4)
        ax_path.plot(*taught_hand.T, color=TAUGHT_COLOR, linewidth=2.5)
        ring(ax_path)
        leak, radius, scaling, ridge = CANDIDATES[name]
        ax_path.set_title(
            f"{name}: leak rate {leak:g}, spectral radius {radius:g},\ninput scaling {scaling:g}, ridge {sci(ridge)}",
            color=TEXT_COLOR,
            fontsize=8,
        )
        ax_path.set(xlabel="x (m)", ylabel="y (m)", aspect="equal", xlim=(-0.4, 1.0), ylim=(-0.3, 1.35))
        values = [1000 * detours(hand, taught_hand) for hand in hands]
        draw_map(ax_map, [offset_of(r["origin"]) for r in rows], values, norm, "Blues")
        ax_map.set_title(f"detours over 50 mm: {sum(v > 50 for v in values)}", color=TEXT_COLOR, fontsize=9)
        ax_map.set_xlabel("joint 1 offset (deg)", color=TEXT_COLOR, fontsize=8)
        ax_map.set_ylabel("joint 2 offset (deg)", color=TEXT_COLOR, fontsize=8)
    label = "detour beyond the start (mm, 300 and over alike)"
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=list(axes[1]), shrink=0.8)
    bar.set_label(label, fontsize=8)
    fig.legend(
        handles=[
            Line2D([], [], color=ESN_COLOR, label="ESN's runs"),
            Line2D([], [], color=TAUGHT_COLOR, linewidth=2.5, label="taught path"),
        ],
        loc="outside upper right",
        frameon=False,
        labelcolor=TEXT_COLOR,
    )
    return fig


def write_candidates() -> None:
    """Per candidate: failures, detours, jumps, arrival, and how its runs gather onto a common route."""
    _, taught_hand = taught()
    out = []
    for name, (leak, radius, scaling, ridge) in CANDIDATES.items():
        rows, hands = candidate_runs(name)
        det = np.array([detours(hand, taught_hand) for hand in hands])
        first = np.array([r["first_step_m"] for r in rows])
        arrival = np.array([r["arrival_time_s"] for r in rows])
        start = demonstrated(rows)
        route = rows_of(f"route_candidate_{name.lower()}_raw", "convergence.csv")
        out.append(
            {"candidate": name, "leak_rate": leak, "spectral_radius": radius, "input_scaling": scaling, "ridge": ridge}
            | {
                "runs": len(rows),
                "failures": sum(not r["success"] for r in rows),
                "detours_over_50mm": int(np.sum(det > MAX_DETOUR)),
                "largest_detour_m": float(det.max()),
                "first_steps_over_30mm": int(np.sum(first > MAX_FIRST_STEP)),
                "largest_first_step_m": float(first.max()),
                "median_arrival_s": float(np.nanmedian(arrival)),
                "demonstrated_arrival_s": float(start["arrival_time_s"]),
                "median_join_time_s": float(np.nanmedian([r["join_time_s"] for r in route])),
                "median_target_distance_at_join_m": float(
                    np.nanmedian([r["target_distance_at_join_m"] for r in route])
                ),
            }
        )
    write_csv(CANDIDATES_CSV, out)


def dwell_stats(log: StateLog, arrival: float) -> tuple[float, float, float]:
    """From DWELL_AFTER after arrival to the end: the hand's largest distance to the target (m), its RMS speed
    (m/s), and the RMS speed of the reference (deg/s)."""
    t = log.times
    window = t >= arrival + DWELL_AFTER
    hand = hand_of(log)[window]
    dt = np.diff(t[window])
    speed = np.linalg.norm(np.diff(hand, axis=0), axis=1) / dt
    ref = log.channel("q_ref").reshape(len(t), -1)[window]
    ref_speed = np.degrees(np.linalg.norm(np.diff(ref, axis=0), axis=1) / dt)
    return (
        float(np.linalg.norm(hand - TARGET, axis=1).max()),
        float(np.sqrt(np.mean(speed**2))),
        float(np.sqrt(np.mean(ref_speed**2))),
    )


def write_dwell() -> None:
    """Per scenario and tracker setting of candidate F: how still its arm stays at the target after arriving."""
    out = []
    for label, (suffix, setting) in {**TRACKERS, **UNDERDAMPED}.items():
        for scenario in SCENARIOS:
            stem = f"{scenario}_candidate_f{suffix}_raw"
            rows = [r for r in rows_of(stem) if r["arm"] == "esn" and setting_of(r) == setting]
            if not rows:  # the underdamped runs of the offsets start from two offsets only, all here
                continue
            stats, never = [], 0
            for r in rows:
                if not np.isfinite(r["arrival_time_s"]):
                    never += 1
                    continue
                log = log_of(stem, setting, f"esn_{int(r['start']):02d}.sklog.npz")
                if log.times[-1] > r["arrival_time_s"] + DWELL_AFTER + DT:
                    stats.append(dwell_stats(log, r["arrival_time_s"]))
            worst = np.array(stats).max(axis=0) if stats else np.full(3, np.nan)
            out.append(
                {"tracker": label, "scenario": scenario, "runs": len(rows), "never_arrive": never}
                | {"holds": sum(r["success"] for r in rows)}
                | {"worst_distance_m": worst[0], "worst_speed_mps": worst[1], "worst_reference_speed_dps": worst[2]}
            )
    write_csv(DWELL_CSV, out)


def plot_dwell() -> Figure:
    """Candidate F from the demonstrated start, undisturbed: the hand's distance to the target under each tracker."""
    fig = Figure(figsize=(12, 4.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("Candidate F settling at the target, undisturbed, from the demonstrated start", color="#0b0b0b")
    ax_damped, ax_under = fig.subplots(1, 2, sharey=True)
    for ax, group, title in (
        (ax_damped, TRACKERS, "Critically damped trackers"),
        (ax_under, UNDERDAMPED, "Underdamped trackers"),
    ):
        style(ax)
        for (label, (suffix, setting)), color in zip(group.items(), DWELL_COLORS, strict=True):
            log = log_of(f"nominal_candidate_f{suffix}_raw", setting, "esn_00.sklog.npz")
            distance = 1000 * np.linalg.norm(hand_of(log) - TARGET, axis=1)
            ax.plot(log.times, distance, color=color, linewidth=1.3, label=label)
        ax.axhline(1000 * RADIUS, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
        ax.set(title=title, xlabel="time (s)", ylim=(0, 120), xlim=(4, 16))
        ax.legend(
            frameon=False, labelcolor=TEXT_COLOR, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2
        )
    ax_damped.set_ylabel("hand to target (mm; dotted: the goal)")
    return fig


def push_logs(label: str) -> dict[str, StateLog]:
    """Candidate F's logs from the demonstrated start at one tracker setting: undisturbed and pushed."""
    suffix, setting = TRACKERS[label]
    logs = {"undisturbed": log_of(f"nominal_candidate_f{suffix}_raw", setting, "esn_00.sklog.npz")}
    for push, stem in PUSHES.items():
        logs[push] = log_of(f"{stem}_candidate_f{suffix}_raw", setting, "esn_00.sklog.npz")
    return logs


def plot_pushes() -> Figure:
    """Candidate F pushed: distance to the target, progress along the taught path, and hand paths."""
    taught_q, taught_hand = taught()
    shown = ["computed torque, ω = 10", "joint PD, ω = 20", "joint PD, ω = 40"]
    fig = Figure(figsize=(15, 11), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"Candidate F pushed at {PUSH_TIME:g} s for 0.1 s, from the demonstrated start", color="#0b0b0b")
    axes = fig.subplots(len(shown), 3, width_ratios=[1.4, 1.4, 1])
    for row_axes, label in zip(axes, shown, strict=True):
        ax_target, ax_progress, ax_path = row_axes
        for ax in row_axes:
            style(ax)
        for push, log in push_logs(label).items():
            color = PUSH_COLORS[push]
            hand = hand_of(log)
            q = log.channel("q").reshape(len(log.times), -1)
            ax_target.plot(
                log.times, 1000 * np.linalg.norm(hand - TARGET, axis=1), color=color, linewidth=1.2, label=push
            )
            ax_progress.plot(log.times, path_progress(q, taught_q), color=color, linewidth=1.2)
            ax_path.plot(*hand.T, color=color, linewidth=1.0)
        for ax in (ax_target, ax_progress):
            ax.axvspan(PUSH_TIME, PUSH_TIME + 0.1, color="#f0efec", zorder=0)
            ax.set_xlim(0, 16)
        ax_path.plot(*taught_hand.T, color=RECORDED_COLOR, linewidth=3, zorder=0)
        ring(ax_path)
        ax_target.set(title=f"{label}: hand to target", ylabel="mm", ylim=(0, None))
        ax_progress.set(title="progress along the taught path", ylim=(-0.05, 1.05))
        ax_path.set(title="hand paths (gray: taught)", aspect="equal", xlim=(-0.4, 0.9), ylim=(0.3, 1.3))
    for ax in axes[-1][:2]:
        ax.set_xlabel("time (s)")
    axes[0][0].legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=8)
    return fig


def write_pushes() -> None:
    """Per tracker setting and push: candidate F's arrival against its undisturbed run, how far it strays, its hold,
    and the peak torques of F and of the replay in the second after the push."""
    _, taught_hand = taught()
    out = []
    for label, (suffix, setting) in {**TRACKERS, **UNDERDAMPED}.items():
        nominal = next(
            r for r in rows_of(f"nominal_candidate_f{suffix}_raw") if r["arm"] == "esn" and setting_of(r) == setting
        )
        for push, stem in PUSHES.items():
            rows = rows_of(f"{stem}_candidate_f{suffix}_raw")
            esn = next(r for r in rows if r["arm"] == "esn" and setting_of(r) == setting)
            replay = next(r for r in rows if r["arm"] == "replay" and setting_of(r) == setting)
            log = log_of(f"{stem}_candidate_f{suffix}_raw", setting, "esn_00.sklog.npz")
            after = log.times >= PUSH_TIME
            hand = hand_of(log)[after]
            out.append(
                {"tracker": label, "push": push, "undisturbed_arrival_s": nominal["arrival_time_s"]}
                | {
                    "arrival_s": esn["arrival_time_s"],
                    "arrival_shift_s": esn["arrival_time_s"] - nominal["arrival_time_s"],
                    "holds": esn["success"],
                    "settling_time_s": esn["settling_time_s"],
                    "largest_distance_from_taught_path_m": float(distances_to_path(hand[::5], taught_hand[::5]).max()),
                    "farthest_from_target_m": float(np.linalg.norm(hand - TARGET, axis=1).max()),
                    "final_distance_m": esn["final_distance_m"],
                    "peak_torque_nm": esn["peak_torque_nm"],
                    "replay_holds": replay["success"],
                    "replay_peak_torque_nm": replay["peak_torque_nm"],
                }
            )
    write_csv(PUSHES_CSV, out)


# ---------------------------------------------------------------------------- the examples


def slug(name: str) -> str:
    return name.replace(" ", "_")


def example_log_path(scenario: str, suffix: str, setting: str, arm: str, offset: tuple[float, float]) -> Path:
    """The log of one arm in a robot run of candidate F, from the start at ``offset`` (deg)."""
    stem = f"{scenario}_candidate_f{suffix}_raw"
    start = next(int(r["start"]) for r in rows_of(stem) if r["arm"] == arm and offset_of(r["origin"]) == offset)
    return resolve_run_path(f"results/{run_name(stem)}") / setting / f"{arm}_{start:02d}.sklog.npz"


def joint_angles_at(log: StateLog, times: NDArray[np.float64], channel: str = "q") -> NDArray[np.float64]:
    """A log's joint angles (deg), or its reference with channel "q_ref", at the given times."""
    q = np.degrees(log.channel(channel).reshape(len(log.times), -1))
    return np.column_stack([np.interp(times, log.times, q[:, j]) for j in range(q.shape[1])])


def example_title(name: str, offset: tuple[float, float]) -> str:
    """Where an example starts, for titles."""
    if offset == (0.0, 0.0):
        return f"{name}, from the demonstrated start"
    return f"{name}: start offset by {offset[0]:+g}° in joint 1 and {offset[1]:+g}° in joint 2"


def plot_example(name: str, scenario: str, offset: tuple[float, float], settings: list[tuple[str, str, str]]) -> Figure:
    """The joint angles of one example over time, one column per tracker setting, one row per joint.

    Each column draws candidate F's arm (solid) and its output, the reference it gives
    the tracker (dashed); the replay's arm; and the take, which is the replay's
    reference (thick gray). Under a disturbance, it also draws F's undisturbed arm from
    the same start, and shades the disturbance.
    """
    take_log = StateLog.load(TAKE_RUN / "demo_00.sklog.npz")
    times = np.arange(0.0, EXAMPLE_END + 1e-9, 0.01)
    take = joint_angles_at(take_log, np.clip(times, 0.0, float(take_log.times[-1])))
    fig = Figure(figsize=(5.4 * len(settings) + 1.0, 7.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"Joint angles over time: {example_title(name, offset)}", color="#0b0b0b")
    axes = fig.subplots(2, len(settings), sharex=True, squeeze=False)
    for column, (label, suffix, setting) in enumerate(settings):
        esn = StateLog.load(example_log_path(scenario, suffix, setting, "esn", offset))
        replay = StateLog.load(example_log_path(scenario, suffix, setting, "replay", offset))
        lines = [(take, "the take (the replay's reference)", RECORDED_COLOR, "-", 4.0)]
        if scenario in SPANS:
            undisturbed = StateLog.load(example_log_path("nominal", suffix, setting, "esn", offset))
            lines.append(
                (joint_angles_at(undisturbed, times), "candidate F, undisturbed: arm", UNDISTURBED_COLOR, "-", 1.4)
            )
        lines += [
            (joint_angles_at(replay, times), "replay: arm", REPLAY_COLOR, "-", 1.2),
            (joint_angles_at(esn, times, "q_ref"), "candidate F: output (reference)", ESN_COLOR, "--", 1.6),
            (joint_angles_at(esn, times), "candidate F: arm", ESN_COLOR, "-", 1.6),
        ]
        for joint in range(2):
            ax = axes[joint, column]
            style(ax)
            if scenario in SPANS:
                ax.axvspan(*SPANS[scenario], color=DISTURBANCE_COLOR, zorder=0)
            for values, line_label, color, line, width in lines:
                ax.plot(times, values[:, joint], color=color, linestyle=line, linewidth=width, label=line_label)
            ax.set_title(f"{label}: joint {joint + 1}", color=TEXT_COLOR, fontsize=9)
            if column == 0:
                ax.set_ylabel(f"joint {joint + 1} (deg)", color=TEXT_COLOR)
        axes[1, column].set_xlabel("time (s)", color=TEXT_COLOR)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def export_example_animation(
    name: str,
    scenario: str,
    offset: tuple[float, float],
    panels: list[tuple[str, str, str, str, str]],
    prefix: str,
    note: str | None = None,
) -> None:
    """Animate an example into ``<prefix>_<name>.gif``: the arms of ``panels`` side by side, rendered by skelarm.

    Each panel is a label, an arm ("esn" or "replay"), a configuration suffix, a
    tracker setting, and the label's color. The player exports each arm's run as a
    GIF (``--export``); its frames, which the GIF merges where the arm rests, are
    spread back over time, cropped to where the arms move, labeled, and tiled into
    one GIF with the task time and the disturbance. With a ``note``, the header has
    two lines: the example, then the note with the time. A header line too wide for
    the panels is set in a smaller font.
    """
    import tempfile

    from PIL import Image, ImageDraw, ImageFont, ImageSequence

    from arm_esn_ctrl.storage import REPO_ROOT

    player = REPO_ROOT / "third_party" / "skelarm" / "tools" / "player.py"
    frames, starts = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (_, arm, suffix, setting, _) in enumerate(panels):
            path = example_log_path(scenario, suffix, setting, arm, offset)
            exported = Path(tmp) / f"{i}.gif"
            subprocess.run(
                [sys.executable, str(player), str(path), "--export", str(exported), "--fps", f"{ANIMATION_FPS:g}"],
                check=True,
                capture_output=True,
                env=os.environ | {"QT_QPA_PLATFORM": "offscreen"},  # render without a window
            )
            with Image.open(exported) as gif:
                spread = []
                for frame in ImageSequence.Iterator(gif):
                    repeats = max(round(frame.info.get("duration", 1000.0 / ANIMATION_FPS) * ANIMATION_FPS / 1000.0), 1)
                    spread += [np.asarray(frame.convert("RGB"))] * repeats
            frames.append(np.array(spread))
            starts.append(float(StateLog.load(path).times[0]))
    ink = np.any([(f < 235).any(axis=3).any(axis=0) for f in frames], axis=0)
    rows, cols = np.nonzero(ink)
    margin = 24
    top, bottom = int(max(rows.min() - margin, 0)), int(rows.max() + margin)
    left, right = int(max(cols.min() - margin, 0)), int(cols.max() + margin)
    width, height = right - left, bottom - top
    label_height, gap = 30, 8
    canvas_width = len(frames) * width + (len(frames) - 1) * gap
    small = ImageFont.load_default(size=15)

    def header_lines(t: float) -> list[str]:
        status = ""
        if scenario in SPANS and SPANS[scenario][0] <= t < SPANS[scenario][1]:
            status = f"   {DISTURBANCES.get(scenario, 'blocked')}"
        clock = f"t = {t:+.1f} s{status}"
        title = example_title(name, offset)
        return [f"{title}   {clock}"] if note is None else [title, f"{note}   {clock}"]

    # One font for every frame: the largest, up to 20, in which every header line fits. The widest lines are
    # those with a disturbance's status, or with the last time.
    lines = [line for t in (*SPANS.get(scenario, ()), ANIMATION_SPAN[1]) for line in header_lines(t)]
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    size = 20
    while size > 12 and max(measure.textlength(line, font=ImageFont.load_default(size=size)) for line in lines) > (
        canvas_width - 20
    ):
        size -= 1
    font = ImageFont.load_default(size=size)
    header = 44 if note is None else 44 + size + 6
    images = []
    for t in np.arange(ANIMATION_SPAN[0], ANIMATION_SPAN[1] + 1e-9, 1.0 / ANIMATION_FPS):
        canvas = Image.new("RGB", (canvas_width, header + label_height + height), "white")
        draw = ImageDraw.Draw(canvas)
        for row, line in enumerate(header_lines(t)):
            draw.text((10, 10 + row * (size + 6)), line, fill="#0b0b0b", font=font)
        for i, (label, _, _, _, color) in enumerate(panels):
            k = int(np.clip(round((t - starts[i]) * ANIMATION_FPS), 0, len(frames[i]) - 1))
            x = i * (width + gap)
            canvas.paste(Image.fromarray(frames[i][k][top:bottom, left:right]), (x, header + label_height))
            draw.text((x + 8, header + 4), label, fill=color, font=small)
        images.append(canvas)
    # One palette for every frame, so that the GIF stores only what changes from frame to frame.
    # It is learned from all the frames, so that brief colors, such as a push's arrow, keep theirs.
    sample = Image.new("RGB", (images[0].width, len(images) * images[0].height))
    for i, image in enumerate(images):
        sample.paste(image, (0, i * images[0].height))
    palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    images = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    durations = [round(1000.0 / ANIMATION_FPS)] * (len(images) - 1) + [ANIMATION_HOLD_MS]
    gif = SUMMARY / f"{prefix}_{slug(name)}.gif"
    images[0].save(gif, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {gif.name}: {len(images)} frames")


# ---------------------------------------------------------------------------- tables


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.array([float(r[key]) for r in rows])
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def print_tables() -> None:
    """Print the tables of the report."""
    print("== 3.1 The take: as recorded and filtered")
    for r in read_csv(TAKE_RUN / "metrics.csv"):
        print(
            f"  {r['demo']}: moves at {r['onset_time_s']:.2f} s, arrives at {r['arrival_time_s']:.2f} s of"
            f" {r['length_s']:.2f} s; peak speed {r['peak_speed']:.2f} m/s; {int(r['speed_peaks'])} speed peaks;"
            f" joint jitter {r['jitter_deg']:.3f} deg; final error {1000 * r['final_error']:.1f} mm;"
            f" {int(r['samples'])} samples at {r['achieved_rate_hz']:.1f} Hz, longest interval"
            f" {1000 * r['longest_interval_s']:.1f} ms"
        )

    print("\n== 3.2 Report 003's settings on the take (169 starts)")
    print(
        "settings, take: arrive and hold; first step mean (mm); demonstrated start: arrival (s), final distance (mm),"
    )
    print("joint error (deg), jitter (deg) and speed peaks against the take's")
    for name, stem in SETTINGS.values():
        for take, how in TAKES.items():
            rows = rows_of(f"{stem}_{take}")
            d = demonstrated(rows)
            offset = [r for r in rows if r is not d]
            print(
                f"  {name}, {how}: {sum(r['success'] for r in rows)}/{len(rows)};"
                f" {1000 * mean(offset, 'first_step_m'):.1f};"
                f" {d['arrival_time_s']:.2f}, {1000 * d['final_distance_m']:.0f}, {d['taught_joint_error_deg']:.2f},"
                f" {d['jitter_deg']:.4f} vs {d['taught_jitter_deg']:.4f},"
                f" {int(d['speed_peaks'])} vs {int(d['taught_speed_peaks'])}"
            )

    print("\n== 3.3 Report 003's ESNs on the robot, undisturbed from the demonstrated start, computed torque ω = 10")
    print("ESN, take: ESN arm holds; final distance (mm); peak torque (N m) | replay: holds; peak torque (N m); effort")
    for key, suffix in (("tuned", "tuned"), ("eight", "multi_demo_settings")):
        for take, how in TAKES.items():
            rows = [r for r in rows_of(f"nominal_{suffix}_{take}") if setting_of(r) == "computed_torque_w10"]
            e = next(r for r in rows if r["arm"] == "esn")
            p = next(r for r in rows if r["arm"] == "replay")
            print(
                f"  {SETTINGS[key][0]}, {how}: {e['success']}; {1000 * e['final_distance_m']:.0f};"
                f" {e['peak_torque_nm']:.1f}"
                f" | {p['success']}; {p['peak_torque_nm']:.1f}; {p['effort_n2m2s']:.1f}"
            )

    print("\n== 3.4 Sweeps: combinations; robust (no run fails); robust without swings and jumps")
    for prefix in (
        "sweep_main_ridge_",
        "sweep_ridge_warmup_",
        "sweep_around_a_ridge_",
        "sweep_noise_",
        "sweep_past_edge_ridge_",
    ):
        rows = [r for _, rs in sweep_runs(prefix) for r in rs]
        robust = [r for r in rows if r["failures"] == 0]
        clean = [r for r in robust if r["jumps"] == 0 and r.get("swings", 0) == 0]
        has_swings = "swings" in rows[0]
        print(
            f"  {prefix}*: {len(rows)}; {len(robust)}; {len(clean) if has_swings else '-'}"
            f" (jump-free robust: {sum(r['jumps'] == 0 for r in robust)})"
        )

    if CANDIDATES_CSV.exists():
        print(
            "\n== 3.5 Candidates from the 169 starts: failures; detours over 50 mm (largest, mm);"
            " first steps over 30 mm"
        )
        print(
            "(largest, mm); median arrival (s; demonstrated start);"
            " route: median join time (s), distance to target (mm)"
        )
        for r in read_csv(CANDIDATES_CSV):
            print(
                f"  {r['candidate']}: {int(r['failures'])};"
                f" {int(r['detours_over_50mm'])} ({1000 * r['largest_detour_m']:.0f});"
                f" {int(r['first_steps_over_30mm'])} ({1000 * r['largest_first_step_m']:.0f});"
                f" {r['median_arrival_s']:.1f} ({r['demonstrated_arrival_s']:.1f}); {r['median_join_time_s']:.1f},"
                f" {1000 * r['median_target_distance_at_join_m']:.0f}"
            )

    print("\n== 3.6 Candidate F and the replay from the 169 start offsets: arrive and hold; peak torque mean (N m);")
    print("peak reference speed mean (deg/s); first step mean (mm)")
    for label in TRACKERS:
        cells = []
        for arm in ("esn", "replay"):
            rows = [r for r in offsets_rows(label) if r["arm"] == arm]
            cells.append(
                f"{sum(r['success'] for r in rows)}/{len(rows)}, {mean(rows, 'peak_torque_nm'):.0f},"
                f" {mean(rows, 'peak_reference_speed_dps'):.0f}, {1000 * mean(rows, 'first_step_m'):.1f}"
            )
        print(f"  {label}: F {cells[0]} | replay {cells[1]}")
    if DWELL_CSV.exists():
        print(
            f"\n== 3.6 Dwell of candidate F, from {DWELL_AFTER:g} s after arrival:"
            " worst over the scenarios of the hand's"
        )
        print(
            "distance to the target (mm), its RMS speed (mm/s), and the reference's RMS speed (deg/s); runs that hold"
        )
        by_tracker: dict[str, list[dict[str, Any]]] = {}
        for r in read_csv(DWELL_CSV):
            by_tracker.setdefault(r["tracker"], []).append(r)
        for label, rows in by_tracker.items():
            never = sum(int(r["never_arrive"]) for r in rows)
            print(
                f"  {label}: {1000 * np.nanmax([r['worst_distance_m'] for r in rows]):.0f},"
                f" {1000 * np.nanmax([r['worst_speed_mps'] for r in rows]):.1f},"
                f" {np.nanmax([r['worst_reference_speed_dps'] for r in rows]):.2f};"
                f" {sum(int(r['holds']) for r in rows)}/{sum(int(r['runs']) for r in rows)} hold"
                + (f"; {never} never arrive" if never else "")
            )
    if PUSHES_CSV.exists():
        print(
            "\n== 3.7 Candidate F pushed: arrival (s) and its shift; holds; largest distance from the taught path (mm);"
        )
        print("farthest from the target (mm); peak torque F | replay (N m)")
        for r in read_csv(PUSHES_CSV):
            arrival = f"{r['arrival_s']:.2f} ({r['arrival_shift_s']:+.2f})" if np.isfinite(r["arrival_s"]) else "never"
            print(
                f"  {r['tracker']}, {r['push']}: {arrival}; {'holds' if r['holds'] else 'fails'};"
                f" {1000 * r['largest_distance_from_taught_path_m']:.0f}; {1000 * r['farthest_from_target_m']:.0f};"
                f" {r['peak_torque_nm']:.1f} | {r['replay_peak_torque_nm']:.1f}"
            )

    print("\n== 3.9 Underdamped trackers, the 7 example runs of each arm (nominal, the two offsets, three pushes,")
    print("block): arrive and hold; settle within the run; worst final distance (mm), for F | replay")
    for law, name in (("computed_torque", "computed torque, ω = 10"), ("pd", "joint PD, ω = 20")):
        for damping in DAMPINGS:
            cells = []
            for arm in ("esn", "replay"):
                rows = [
                    r
                    for scenario in SCENARIOS
                    for r in rows_of(f"{scenario}_candidate_f_damping_raw")
                    if r["arm"] == arm and r["law"] == law and r["damping"] == damping
                ]
                settled = sum(np.isfinite(r["settling_time_s"]) for r in rows)
                worst = 1000 * max(r["final_distance_m"] for r in rows)
                cells.append(f"{sum(r['success'] for r in rows)}/{len(rows)}; {settled}/{len(rows)}; {worst:.1f}")
            print(f"  {name}, ζ = {damping:g}: F {cells[0]} | replay {cells[1]}")


if __name__ == "__main__":
    main()
