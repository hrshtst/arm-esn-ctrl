# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 020.

    uv run python reports/020-the-esn-and-the-robot-as-one-system/make_figures.py

It reads report 020's runs and report 019's under the storage root, whose arrays are not
kept in Git, copies report 020's configurations, run records, and CSV files into
``results/``, prints the tables of the report, scores the claims, runs its checks, and
draws into ``results/summary``:

- from rest anywhere on the robot (Section 3.1): ``arrival.png`` (the arrival, its
  difference from the ESN's own, and the peak torque over the start postures),
  ``portrait.png`` (the arm's runs from every 30° in joint space), ``hands.png`` (their
  hand paths), and ``series_esn1.png`` and ``series_esn2.png`` (the joint angles and
  torques over time from six starts, against the ESN's own runs);
- ESN #1 and ESN #2 with the arm (Section 3.2): ``spectra.png`` (the eigenvalues at the
  fixed points), ``rates.png`` (the rates along the robot's run against the ESN's own),
  ``pulses_esn1.png`` and ``pulses_esn2.png`` (four pulses' deviations over time,
  simulated and linearized), and ``pulse_errors.png`` (every pulse's error);
- every fixed point of the screens with the arm (Sections 3.3 and 3.4): ``growth.png``
  (the growth rate with the arm against on its own), ``releases.png`` (the stay after a
  release against the growth rate), and ``rule_a.png`` (rule (a) on its own and with
  the arm, within each leak rate).

Report 015's script, pointed at report 020's runs and at report 019's, gives the start
postures, the take, and its drawing helpers.
"""

from __future__ import annotations

import csv
import functools
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from numpy.typing import NDArray
from scipy.stats import rankdata

from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
STUDY = "results/manual_demonstration_v2_consistent_robot_loop"
EARLIER = "results/manual_demonstration_v2_consistent_dynamics"
# Report 020's runs, under the storage root.
RUNS = {
    "starts": f"{STUDY}/20261102-122820-robot_starts",
    "fixed_points": f"{STUDY}/20261102-112847-robot_fixed_points",
    "screens": f"{STUDY}/20261102-132045-screen_robot_loop",
    # Added after the claims' runs: the releases nudged off their fixed points, and the rates weighed otherwise.
    "nudged": f"{STUDY}/20261102-132553-nudge_releases",
    "weighed": f"{STUDY}/20261102-133837-robot_rates_weighed",
}
# Report 019's runs that report 020 compares with: the ESNs on their own from rest anywhere, and their fixed points.
EARLIER_RUNS = {
    "global_starts": "20261027-180441-global_starts",
    "fixed_points": "20261027-183137-fixed_points",
}
DT = 0.01  # the ESNs' period (s)
DPI = 90
ESNS = [("esn1", "ESN #1"), ("esn2", "ESN #2")]
ESN_COLOR = "#2a78d6"
OWN_COLOR = "#52514e"
LIGHT_COLOR = "#1baf7a"  # the pulses of 3 N
SERIES_SECONDS = 20.0  # the runs over time are drawn this long: every one rests by then
SHOWN_PULSES = [(4.0, 0.0), (4.0, 180.0), (7.0, 90.0), (12.0, 270.0)]  # report 019's: onset (s), angle (deg)
FIXED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#5b2a86", "#c0392b"]
SET_LABELS = ["seeds 0 to 9", "seeds 10 to 19, held out"]
FAST_FROM = 0.15

TEXT_COLOR = "#52514e"
TITLE_COLOR = "#0b0b0b"
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"
TAUGHT_COLOR = "#b9b7b0"
MISSING_COLOR = "#d9d8d3"


def main() -> None:
    copy_results()
    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {
        "arrival.png": plot_arrival(),
        "portrait.png": plot_portrait(),
        "hands.png": plot_hands(),
        "series_esn1.png": plot_series("esn1", "ESN #1"),
        "series_esn2.png": plot_series("esn2", "ESN #2"),
        "spectra.png": plot_spectra(),
        "rates.png": plot_rates(),
        "pulses_esn1.png": plot_pulses("esn1", "ESN #1"),
        "pulses_esn2.png": plot_pulses("esn2", "ESN #2"),
        "pulse_errors.png": plot_pulse_errors(),
        "growth.png": plot_growth(),
        "releases.png": plot_releases(),
        "rule_a.png": plot_rule_a(),
        "nudged.png": plot_nudged(),
    }
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=DPI)
    print_starts()
    print_fixed_points()
    print_rates()
    print_pulses()
    print_changes()
    print_rule_a()
    print_releases()
    print_nudged()
    print_stayed()
    print_weighings()
    print_claims()
    print_checks()
    print(f"\nWrote {', '.join(figures)} to {SUMMARY}")


# ---------------------------------------------------------------------------- the runs


def run_dir(name: str) -> Path:
    return resolve_run_path(RUNS[name])


def copy_results() -> None:
    """Copy each run's configuration, run record, and CSV files into ``results/<run>``."""
    for name in RUNS:
        target = RESULTS / Path(RUNS[name]).name
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(run_dir(name).glob("*.toml")) + sorted(run_dir(name).glob("*.csv")):
            shutil.copy2(path, target / path.name)


def load_script(report: str, name: str) -> ModuleType:
    """An earlier report's make_figures.py, as a module named ``name``."""
    path = REPO_ROOT / "reports" / report / "make_figures.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@functools.cache
def own() -> Any:
    """Report 015's script, pointed at report 019's runs of the ESNs on their own."""
    module: Any = load_script("015-what-the-esns-learned", "report_015_own")
    module.STUDY = EARLIER
    module.RESULTS, module.SUMMARY = RESULTS, SUMMARY
    module.RUNS.update(EARLIER_RUNS)
    return module


@functools.cache
def robot() -> Any:
    """Report 015's script, pointed at report 020's runs on the robot: from rest anywhere, and the rates."""
    module: Any = load_script("015-what-the-esns-learned", "report_015_robot")
    module.STUDY = STUDY
    module.RESULTS, module.SUMMARY = RESULTS, SUMMARY
    module.RUNS.update({"global_starts": Path(RUNS["starts"]).name, "fixed_points": Path(RUNS["fixed_points"]).name})
    return module


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
        return [{k: value(v) for k, v in row.items()} for row in csv.DictReader(f)]


@functools.cache
def step2(name: str) -> list[dict[str, Any]]:
    return read_csv(run_dir("fixed_points") / name)


@functools.cache
def step3(name: str) -> list[dict[str, Any]]:
    return read_csv(run_dir("screens") / name)


@functools.cache
def npz(name: str, file: str) -> dict[str, NDArray[Any]]:
    with np.load(run_dir(name) / file) as data:
        return {k: data[k] for k in data.files}


def column(rows: list[dict[str, Any]], name: str) -> NDArray[Any]:
    return np.array([row[name] for row in rows])


def rank_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Spearman's rank correlation over the entries where neither is NaN; an infinity ranks last."""
    ok = ~np.isnan(x) & ~np.isnan(y)
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])


def arrivals(key: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The arrival from each start, on the robot and on its own (NaN: never)."""
    return column(robot().starts(key), "arrival_time_s"), column(own().starts(key), "arrival_time_s")


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


def joint_map(ax: Axes, values: NDArray[np.float64], cmap: Any, norm: Any) -> None:
    """``values``, one per start posture, over the joint space, with the take's path and the offset starts."""
    r = robot()
    r.style(ax)
    ax.imshow(r.grid_of(values), origin="lower", extent=r.grid_extent(), cmap=cmap, norm=norm, zorder=1)
    r.draw_take_in_joint_space(ax)
    ax.add_patch(Rectangle((15.0, 15.0), 30.0, 30.0, fill=False, linestyle="--", color=TITLE_COLOR, zorder=5))
    r.joint_axes(ax)


# ---------------------------------------------------------------------------- step 1


def plot_arrival() -> Figure:
    """Over the start postures: the robot's arrival, its difference from the ESN's own, and the peak torque."""
    fig = new_figure(
        13.5,
        19.0,
        "On the robot from rest at each start posture: when the hand arrives (top; gray: never, or not at rest at the"
        " target at the end),\nhow much later than the ESN on its own from the same start (middle), and the largest"
        " joint torque (bottom)",
    )
    axes = fig.subplots(3, len(ESNS))
    r = robot()
    times = {key: arrivals(key) for key, _ in ESNS}
    finite = np.concatenate([t[np.isfinite(t)] for t, _ in times.values()])
    arrival_norm = Normalize(float(finite.min()), float(finite.max()))
    differences = np.concatenate([(a - b)[np.isfinite(a - b)] for a, b in times.values()])
    reach = max(float(np.quantile(np.abs(differences), 0.99)), 0.1)
    difference_norm = TwoSlopeNorm(0.0, -reach, reach)
    torques = {key: np.maximum(column(r.starts(key), "peak_tau1_nm"), column(r.starts(key), "peak_tau2_nm"))
               for key, _ in ESNS}  # fmt: skip
    every_torque = np.concatenate(list(torques.values()))
    torque_norm = LogNorm(float(every_torque.min()), float(every_torque.max()))
    blues = colormaps["Blues"].with_extremes(bad=MISSING_COLOR)
    diverging = colormaps["RdBu_r"].with_extremes(bad=MISSING_COLOR)
    for col, (key, label) in enumerate(ESNS):
        rows = r.starts(key)
        at_target = column(rows, "outcome") == "target"
        robot_arrival, own_arrival = times[key]
        joint_map(axes[0, col], np.where(at_target, robot_arrival, np.nan), blues, arrival_norm)
        counts = {name: int(np.count_nonzero(column(rows, "outcome") == name)) for name in r.OUTCOMES}
        axes[0, col].set_title(
            f"{label}: " + ", ".join(f"{name} {count}" for name, count in counts.items() if count),
            color=TITLE_COLOR,
            fontsize=10,
        )
        joint_map(axes[1, col], robot_arrival - own_arrival, diverging, difference_norm)
        joint_map(axes[2, col], torques[key], colormaps["Purples"], torque_norm)
    fig.colorbar(ScalarMappable(norm=arrival_norm, cmap="Blues"), ax=list(axes[0]), label="arrival (s)", shrink=0.8)
    fig.colorbar(ScalarMappable(norm=difference_norm, cmap="RdBu_r"), ax=list(axes[1]), shrink=0.8, extend="both",
                 label="arrival on the robot, later than on its own (s)")  # fmt: skip
    fig.colorbar(ScalarMappable(norm=torque_norm, cmap="Purples"), ax=list(axes[2]), shrink=0.8,
                 label="largest joint torque (N m)")  # fmt: skip
    handles = [*r.take_handles(), Line2D([], [], color=TITLE_COLOR, linestyle="--", label="the 49 offset starts")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_portrait() -> Figure:
    """The arm's runs from every 30 deg in joint space: the reference's first step (dashed), the arm colored by time."""
    fig = new_figure(
        13.5,
        7.2,
        "On the robot from rest: the arm's runs from every 30° (dots) in joint space, the reference's first step"
        " (dashed), and the arm's path, colored by time",
    )
    axes = fig.subplots(1, len(ESNS))
    r = robot()
    norm = Normalize(0.0, r.COLOR_TIME)
    for ax, (key, label) in zip(axes, ESNS, strict=True):
        r.style(ax)
        data = r.runs(key)
        for i in r.every(30.0):
            q = np.degrees(data["q"][i].astype(np.float64))
            reference = np.degrees(data["q_ref"][i, :2].astype(np.float64))
            ax.plot(reference[:, 0], reference[:, 1], color=TEXT_COLOR, linestyle="--", linewidth=0.6, zorder=2)
            ax.plot(*q[0], "o", color=TEXT_COLOR, markersize=2.5, zorder=2)
            r.time_lines(ax, q, data["times"], norm, 1.0)
        r.draw_take_in_joint_space(ax)
        r.joint_axes(ax)
        ax.set_title(label, color=TITLE_COLOR, fontsize=10)
    fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=list(axes), label="time (s)", shrink=0.8, extend="max")
    fig.legend(handles=r.take_handles(), loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_hands() -> Figure:
    """The hand's paths of the arm's runs from every 30 deg, the reference's first step dashed, colored by time."""
    fig = new_figure(
        13.5,
        7.0,
        "On the robot from rest: the hand's paths from every 30°, the reference's first step dashed, then colored by"
        " time",
    )
    axes = fig.subplots(1, len(ESNS), sharex=True, sharey=True)
    r = robot()
    norm = Normalize(0.0, r.COLOR_TIME)
    goal, radius = r.target()
    reach = float(np.linalg.norm(r.arm_points(np.zeros(2))[-1]))
    for ax, (key, label) in zip(axes, ESNS, strict=True):
        r.style(ax)
        data = r.runs(key)
        for i in r.every(30.0):
            hand = endpoint_positions(r.skeleton(), data["q"][i].astype(np.float64))
            pointed = endpoint_positions(r.skeleton(), data["q_ref"][i, :2].astype(np.float64))
            ax.plot(pointed[:, 0], pointed[:, 1], color=TEXT_COLOR, linestyle="--", linewidth=0.5, zorder=2)
            r.time_lines(ax, hand, data["times"], norm, 0.8)
        ax.plot(*r.taught()[1].T, color=TAUGHT_COLOR, linewidth=4.0, zorder=4)
        ax.add_patch(Circle(tuple(goal), radius, fill=False, color=TITLE_COLOR, linewidth=1.2, zorder=5))
        ax.add_patch(Circle((0.0, 0.0), reach, fill=False, color=GRID_COLOR, linewidth=1.0, zorder=1))
        ax.plot(0.0, 0.0, "o", color=TITLE_COLOR, markersize=5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)", color=TEXT_COLOR)
        ax.set_title(label, color=TITLE_COLOR, fontsize=10)
    axes[0].set_ylabel("y (m)", color=TEXT_COLOR)
    fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=list(axes), label="time (s)", shrink=0.8, extend="max")
    handles = [
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the take's hand path"),
        Line2D([], [], color=TITLE_COLOR, linewidth=1.2, label="the goal radius"),
        Line2D([], [], color=GRID_COLOR, linewidth=1.0, label="the arm's reach"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_series(key: str, label: str) -> Figure:
    """One ESN's runs from six starts over time: the arm against the ESN's own run, and the joint torques."""
    r, o = robot(), own()
    starts = r.SHOWN_STARTS
    fig = new_figure(
        16.0,
        11.0,
        f"{label} on the robot from rest at six starts: the joint angles (the arm solid, its reference dotted, the ESN"
        f" on its own dashed, the take thick gray) and the joint torques, over the first {SERIES_SECONDS:g} s",
    )
    axes = fig.subplots(4, len(starts), sharex=True, sharey="row")
    data, alone = r.runs(key), o.runs(key)
    times = data["times"]
    shown = times <= SERIES_SECONDS + 1e-9
    q_taught = np.degrees(r.taught()[0])
    t_taught = DT * np.arange(len(q_taught))
    for col, start in enumerate(starts):
        i = r.start_index(start)
        if o.start_index(start) != i:
            msg = "the start postures on the robot and on its own differ"
            raise RuntimeError(msg)
        for joint in range(2):
            ax = axes[joint, col]
            style(ax)
            ax.plot(t_taught, q_taught[:, joint], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
            ax.plot(alone["times"][shown], np.degrees(alone["q"][i, shown, joint]), color=OWN_COLOR, linestyle="--",
                    linewidth=1.1, zorder=2)  # fmt: skip
            ax.plot(times[shown], np.degrees(data["q_ref"][i, shown, joint]), color=ESN_COLOR, linestyle=":",
                    linewidth=1.0, zorder=2.5)  # fmt: skip
            ax.plot(times[shown], np.degrees(data["q"][i, shown, joint]), color=ESN_COLOR, linewidth=1.4, zorder=3)
            ax = axes[2 + joint, col]
            style(ax)
            ax.plot(times[shown], data["tau"][i, shown, joint], color=ESN_COLOR, linewidth=1.2, zorder=3)
            ax.set_yscale("symlog", linthresh=1.0)
        axes[0, col].set_title(f"from ({start[0]:g}°, {start[1]:g}°)", color=TITLE_COLOR, fontsize=10)
        axes[3, col].set_xlabel("time (s)", color=TEXT_COLOR)
    for joint in range(2):
        axes[joint, 0].set_ylabel(f"joint {joint + 1} (deg)", color=TEXT_COLOR)
        axes[2 + joint, 0].set_ylabel(f"joint {joint + 1} torque (N m)", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=ESN_COLOR, linewidth=1.4, label="the arm, on the robot"),
        Line2D([], [], color=ESN_COLOR, linestyle=":", linewidth=1.0, label="its reference, the ESN's output"),
        Line2D([], [], color=OWN_COLOR, linestyle="--", linewidth=1.1, label="the ESN on its own (report 019)"),
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="the take"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- step 2


def plot_spectra() -> Figure:
    """The eigenvalues at each ESN's fixed points, on its own and with the arm."""
    fig = new_figure(
        13.5,
        12.5,
        "The eigenvalues at each fixed point, on its own (circles) and with the arm (dots): all (top), and near 1"
        " (bottom)",
    )
    axes = fig.subplots(2, len(ESNS))
    circle = np.exp(1j * np.linspace(0.0, 2 * np.pi, 721))
    for col, (key, label) in enumerate(ESNS):
        rows = step2(f"fixed_points_{key}.csv")
        spectra = npz("fixed_points", f"spectra_{key}.npz")
        for row_index, (ax, zoom) in enumerate(zip(axes[:, col], (False, True), strict=True)):
            style(ax)
            ax.plot(np.real(circle), np.imag(circle), color=TAUGHT_COLOR, linewidth=1.2, zorder=1)
            for j, row in enumerate(rows):
                color = FIXED_COLORS[j % len(FIXED_COLORS)]
                alone, together = spectra[f"eigenvalues_{j}"], spectra[f"system_eigenvalues_{j}"]
                ax.plot(np.real(alone), np.imag(alone), "o", markerfacecolor="none", markeredgecolor=color,
                        markersize=6, zorder=2)  # fmt: skip
                ax.plot(np.real(together), np.imag(together), ".", color=color, markersize=4, zorder=3)
                if row_index == 0:
                    kind = "attracting" if row["n_unstable"] == 0 else f"{int(row['n_unstable'])} unstable modes"
                    ax.plot([], [], "o", color=color,
                            label=f"({row['q1_deg']:.1f}°, {row['q2_deg']:.1f}°), {kind}")  # fmt: skip
            ax.set_aspect("equal")
            if zoom:
                ax.set_xlim(0.95, 1.01)
                ax.set_ylim(-0.03, 0.03)
            ax.set_xlabel("real part", color=TEXT_COLOR)
        axes[0, col].set_title(label, color=TITLE_COLOR, fontsize=10)
        axes[0, col].legend(fontsize=8, frameon=False, labelcolor=TEXT_COLOR, loc="lower left")
    axes[0, 0].set_ylabel("imaginary part", color=TEXT_COLOR)
    axes[1, 0].set_ylabel("imaginary part", color=TEXT_COLOR)
    return fig


def plot_rates() -> Figure:
    """Along the run from the take's start, on the robot and on its own: the largest rate across it, and posture."""
    fig = new_figure(
        13.5,
        9.2,
        "Along the run from the take's start, over windows of 2 s and 4 s starting at each time: the largest rate at"
        " which a change across the run grows (top; negative: it decays),\nand the most a deviation of the posture is"
        " scaled by the window's end (bottom), on the robot (solid) and on its own (dashed); dotted: the take arrives",
    )
    axes = fig.subplots(2, len(ESNS), sharex=True, sharey="row")
    r, o = robot(), own()
    windows = [2.0, 4.0]
    colors = [colormaps["viridis"](0.25), colormaps["viridis"](0.7)]
    for col, (key, label) in enumerate(ESNS):
        for module, linestyle in ((o, "--"), (r, "-")):
            data = module.rates(key)
            for window, color in zip(windows, colors, strict=True):
                starts_s = data[f"starts_s_{window:g}"]
                axes[0, col].plot(starts_s, data[f"across_{window:g}"][:, 0], color=color, linestyle=linestyle,
                                  linewidth=1.4)  # fmt: skip
                axes[1, col].plot(starts_s, data[f"posture_{window:g}"][:, 0], color=color, linestyle=linestyle,
                                  linewidth=1.4)  # fmt: skip
        for row in range(2):
            style(axes[row, col])
            axes[row, col].axvline(r.taught_arrival(), color=TEXT_COLOR, linestyle=":", linewidth=1.2)
        axes[0, col].axhline(0.0, color=TEXT_COLOR, linewidth=0.8)
        axes[1, col].axhline(1.0, color=TEXT_COLOR, linewidth=0.8)
        axes[1, col].set_yscale("log")
        axes[0, col].set_title(label, color=TITLE_COLOR, fontsize=10)
        axes[1, col].set_xlabel("start of the window (s)", color=TEXT_COLOR)
    axes[0, 0].set_ylabel("largest rate across the run (1/s)", color=TEXT_COLOR)
    axes[1, 0].set_ylabel("posture deviation, scaled by", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=c, linewidth=1.4, label=f"{w:g} s windows") for w, c in zip(windows, colors, strict=True)
    ]
    handles += [
        Line2D([], [], color=TEXT_COLOR, linewidth=1.4, label="on the robot"),
        Line2D([], [], color=TEXT_COLOR, linestyle="--", linewidth=1.4, label="on its own (report 019)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def pulse_index(data: dict[str, NDArray[Any]], onset: float, angle: float, force: float) -> int:
    match = (data["onset"] == onset) & (data["angle_deg"] == angle) & (data["force_n"] == force)
    return int(np.flatnonzero(match)[0])


def plot_pulses(key: str, label: str) -> Figure:
    """Four pulses on one ESN's arm: the deviations from the undisturbed run, simulated and linearized, over time."""
    data = npz("fixed_points", f"pulses_{key}.npz")
    fig = new_figure(
        15.0,
        11.0,
        f"{label}'s arm pushed by four pulses of 3 N and 6 N for 0.2 s: the deviation of the arm's joint angles and"
        " of its reference from the undisturbed run,\nsimulated (solid) and predicted by the system's Jacobians along"
        " the undisturbed run (dashed); the pulse shaded",
    )
    axes = fig.subplots(4, len(SHOWN_PULSES), sharex=True, sharey="row")
    times = data["times"]
    for col, (onset, angle) in enumerate(SHOWN_PULSES):
        for force, color in ((6.0, ESN_COLOR), (3.0, LIGHT_COLOR)):
            i = pulse_index(data, onset, angle, force)
            for kind in range(2):  # the arm's joint angles, then the reference
                for joint in range(2):
                    ax = axes[2 * kind + joint, col]
                    ax.plot(times, np.degrees(data["simulated"][i, kind, :, joint]), color=color, linewidth=1.4)
                    ax.plot(times, np.degrees(data["predicted"][i, kind, :, joint]), color=color, linestyle="--",
                            linewidth=1.4)  # fmt: skip
        for row in range(4):
            style(axes[row, col])
            axes[row, col].axvspan(0.0, 0.2, color="#f0efec", zorder=0)
            axes[row, col].axhline(0.0, color=TEXT_COLOR, linewidth=0.8)
        words = {0.0: "forward", 90.0: "across, to the left", 180.0: "backward", 270.0: "across, to the right"}
        axes[0, col].set_title(f"at {onset:g} s, {words[angle]}", color=TITLE_COLOR, fontsize=10)
        axes[3, col].set_xlabel("time from the onset (s)", color=TEXT_COLOR)
    for joint in range(2):
        axes[joint, 0].set_ylabel(f"arm, joint {joint + 1} (deg)", color=TEXT_COLOR)
        axes[2 + joint, 0].set_ylabel(f"reference, joint {joint + 1} (deg)", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=ESN_COLOR, linewidth=1.4, label="6 N, simulated"),
        Line2D([], [], color=ESN_COLOR, linestyle="--", linewidth=1.4, label="6 N, linearized"),
        Line2D([], [], color=LIGHT_COLOR, linewidth=1.4, label="3 N, simulated"),
        Line2D([], [], color=LIGHT_COLOR, linestyle="--", linewidth=1.4, label="3 N, linearized"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_pulse_errors() -> Figure:
    """Every pulse's relative error of the linearized joint angles, over its onset and direction."""
    fig = new_figure(
        13.5,
        11.0,
        "The pulses linearized: the predicted joint angles' RMS error over 2 s from the onset,\nrelative to the"
        " simulated deviation's RMS, for each onset (rows) and direction (columns)",
    )
    axes = fig.subplots(len(ESNS), 2)
    errors = {key: step2(f"pulses_{key}.csv") for key, _ in ESNS}
    top = max(float(np.max(column(rows, "q_relative_error"))) for rows in errors.values())
    norm = Normalize(0.0, top)
    for row_index, (key, label) in enumerate(ESNS):
        rows = errors[key]
        onsets = np.unique(column(rows, "onset"))
        angles = np.unique(column(rows, "angle_deg"))
        for col, force in enumerate((3.0, 6.0)):
            ax = axes[row_index, col]
            grid = np.full((len(onsets), len(angles)), np.nan)
            for row in rows:
                if row["force_n"] == force:
                    grid[np.searchsorted(onsets, row["onset"]), np.searchsorted(angles, row["angle_deg"])] = row[
                        "q_relative_error"
                    ]
            ax.imshow(grid, origin="lower", aspect="auto", cmap="Blues", norm=norm)
            for (a, b), v in np.ndenumerate(grid):
                ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 * top else TITLE_COLOR)  # fmt: skip
            ax.set_xticks(range(len(angles)), [f"{a:g}°" for a in angles])
            ax.set_yticks(range(len(onsets)), [f"{o:g} s" for o in onsets])
            ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
            ax.set_title(f"{label}, {force:g} N", color=TITLE_COLOR, fontsize=10)
            ax.set_xlabel("direction, from toward the target, counterclockwise", color=TEXT_COLOR)
        axes[row_index, 0].set_ylabel("onset", color=TEXT_COLOR)
    fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=list(axes.ravel()), shrink=0.6,
                 label="relative RMS error")  # fmt: skip
    return fig


# ---------------------------------------------------------------------------- step 3


def leak_rate_colors() -> tuple[Any, Normalize]:
    """The leak rates' colormap, by the logarithm of the leak rate."""
    return colormaps["viridis"], LogNorm(0.01, 0.5)


def plot_growth() -> Figure:
    """The growth rate with the arm against on its own, at every fixed point of both screens; and the ratio."""
    fig = new_figure(
        13.5,
        6.8,
        "Every fixed point of both screens, the arm at rest at it: the slowest mode's growth rate with the arm against"
        " on its own (left; negative: it attracts),\nand their ratio within each leak rate, median and 10% to 90%"
        " (right)",
    )
    left, right = fig.subplots(1, 2)
    cmap, norm = leak_rate_colors()
    for n in range(2):
        rows = step3(f"fixed_points_{n}.csv")
        left.scatter(column(rows, "growth_per_s"), column(rows, "system_growth_per_s"), s=3, alpha=0.4,
                     c=column(rows, "leak_rate"), cmap=cmap, norm=norm, linewidths=0, zorder=3)  # fmt: skip
    style(left)
    low, high = left.get_xlim()
    left.plot([low, high], [low, high], color=TEXT_COLOR, linewidth=0.8, zorder=2)
    left.axhline(0.0, color=TEXT_COLOR, linewidth=0.6)
    left.axvline(0.0, color=TEXT_COLOR, linewidth=0.6)
    left.set_xlabel("growth rate on its own (1/s)", color=TEXT_COLOR)
    left.set_ylabel("growth rate with the arm (1/s)", color=TEXT_COLOR)
    changes = step3("changes.csv")
    for n, marker in ((0, "o"), (1, "s")):
        rows = [c for c in changes if c["screen"] == n]
        lr = column(rows, "leak_rate")
        position = np.arange(len(lr)) + (n - 0.5) * 0.15
        right.errorbar(position, column(rows, "growth_ratio_median"),
                       yerr=[column(rows, "growth_ratio_median") - column(rows, "growth_ratio_p10"),
                             column(rows, "growth_ratio_p90") - column(rows, "growth_ratio_median")],
                       fmt=marker, color=ESN_COLOR if n == 0 else LIGHT_COLOR, capsize=2,
                       label=SET_LABELS[n])  # fmt: skip
        right.set_xticks(range(len(lr)), [f"{v:g}" for v in lr], rotation=45)
    style(right)
    right.axhline(1.0, color=TEXT_COLOR, linewidth=0.8)
    right.set_xlabel("leak rate", color=TEXT_COLOR)
    right.set_ylabel("|growth with the arm| / |growth on its own|", color=TEXT_COLOR)
    right.legend(fontsize=8, frameon=False, labelcolor=TEXT_COLOR)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=left, label="leak rate", shrink=0.8)
    return fig


def plot_releases() -> Figure:
    """The stay after each release against the growth rate on its own and with the arm."""
    rows = step3("release.csv")
    stays = column(rows, "stays_s")
    top = 70.0  # an arm that stays the whole release is drawn here
    fig = new_figure(
        13.5,
        6.8,
        "Released at an unstable fixed point near the path (994 releases of the passing ESNs): how long the arm stays"
        " within 1° of it,\nagainst the slowest mode's growth rate on its own (left) and with the arm (right); drawn at"
        f" {top:g} s: stays the whole 60 s",
    )
    axes = fig.subplots(1, 2, sharey=True)
    cmap, norm = leak_rate_colors()
    for ax, (name, words) in zip(
        axes, (("growth_per_s", "on its own"), ("system_growth_per_s", "with the arm")), strict=True
    ):
        style(ax)
        ax.scatter(column(rows, name), np.where(np.isinf(stays), top, stays), s=10, c=column(rows, "leak_rate"),
                   cmap=cmap, norm=norm, linewidths=0, alpha=0.8, zorder=3)  # fmt: skip
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(60.0, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
        ax.set_xlabel(f"growth rate {words} (1/s)", color=TEXT_COLOR)
    axes[0].set_ylabel("stays (s)", color=TEXT_COLOR)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(axes), label="leak rate", shrink=0.8)
    return fig


def plot_rule_a() -> Figure:
    """Rule (a) on its own and with the arm: the area under the ROC curve for failing, within each leak rate."""
    scores = step3("scores.csv")
    fig = new_figure(
        13.5,
        6.2,
        "Rule (a), an attracting fixed point off the target, taken on the ESN's own (circles) and with the arm (dots):"
        " the area under the ROC curve for failing on the robot, within each leak rate",
    )
    axes = fig.subplots(1, 2, sharey=True)
    for n, ax in enumerate(axes):
        style(ax)
        for flag, marker, color, words in (
            ("flag_a", "o", OWN_COLOR, "on its own"),
            ("flag_a_arm", ".", ESN_COLOR, "with the arm"),
        ):
            rows = [s for s in scores if s["screen"] == n and s["measure"] == flag and s["leak_rate"] != "all"]
            overall = next(s for s in scores if s["screen"] == n and s["measure"] == flag and s["leak_rate"] == "all")
            ax.plot(range(len(rows)), column(rows, "value"), marker, color=color, markersize=8 if marker == "." else 6,
                    markerfacecolor=color if marker == "." else "none",
                    label=f"{words}: {overall['value']:.3f} over all")  # fmt: skip
            ax.set_xticks(range(len(rows)), [f"{s['leak_rate']:g}" for s in rows], rotation=45)
        ax.axhline(0.5, color=TEXT_COLOR, linewidth=0.8)
        ax.set_xlabel("leak rate", color=TEXT_COLOR)
        ax.set_title(SET_LABELS[n], color=TITLE_COLOR, fontsize=10)
        ax.legend(fontsize=8, frameon=False, labelcolor=TEXT_COLOR)
    axes[0].set_ylabel("area under the ROC curve for failing", color=TEXT_COLOR)
    return fig


def plot_nudged() -> Figure:
    """The releases nudged off their fixed points: the stay against the growth rate, and the robot against its own."""
    rows = read_csv(run_dir("nudged") / "release.csv")
    never, top = 62.0, 70.0  # never arriving, staying the whole release, drawn here
    fig = new_figure(
        16.0,
        6.2,
        "Report 019's releases run again from 1e-9° off the fixed point (added after the claims): how long the arm"
        " stays against the growth rate on its own, as run (circles) and nudged (dots), left;\nthe robot's arrival"
        f" against the ESN's own, as run (middle) and nudged (right); never arriving drawn at {never:g} s, staying"
        f" the whole 60 s at {top:g} s",
    )
    axes = fig.subplots(1, 3)
    cmap, norm = leak_rate_colors()
    lr, growth = column(rows, "leak_rate"), column(rows, "growth_per_s")
    stays, nudged = column(rows, "stays_s"), column(rows, "nudged_stays_s")
    ax = axes[0]
    style(ax)
    ax.scatter(growth, np.where(np.isinf(stays), top, stays), s=14, facecolors="none", edgecolors=OWN_COLOR,
               linewidths=0.6, zorder=2)  # fmt: skip
    ax.scatter(growth, np.where(np.isinf(nudged), top, nudged), s=6, c=lr, cmap=cmap, norm=norm, linewidths=0,
               zorder=3)  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("growth rate on its own (1/s)", color=TEXT_COLOR)
    ax.set_ylabel("stays (s)", color=TEXT_COLOR)
    for ax, (robot_column, own_column, words) in zip(
        axes[1:],
        (("arrival_s", "own_arrival_s", "as run (report 019)"), ("nudged_arrival_s", "nudged_own_arrival_s", "nudged")),
        strict=True,
    ):
        style(ax)
        x = np.nan_to_num(column(rows, own_column), posinf=never)
        y = np.nan_to_num(column(rows, robot_column), posinf=never)
        ax.scatter(x, y, s=8, c=lr, cmap=cmap, norm=norm, linewidths=0, alpha=0.8, zorder=3)
        ax.plot([0, never], [0, never], color=TEXT_COLOR, linewidth=0.8, zorder=2)
        ax.set_xlabel("arrival on its own (s)", color=TEXT_COLOR)
        ax.set_ylabel("arrival on the robot (s)", color=TEXT_COLOR)
        ax.set_title(words, color=TITLE_COLOR, fontsize=10)
        ax.set_aspect("equal")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(axes), label="leak rate", shrink=0.8)
    return fig


# ---------------------------------------------------------------------------- tables


def quantiles(values: NDArray[np.float64], digits: int = 2) -> str:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "n/a"
    return f"{np.median(finite):.{digits}f} ({finite.min():.{digits}f} to {finite.max():.{digits}f})"


def print_starts() -> None:
    print("\nStep 1, from rest anywhere on the robot: median (range) over the start postures, against on its own")
    r, o = robot(), own()
    for key, label in ESNS:
        rows, alone = r.starts(key), o.starts(key)
        outcomes = {name: int(np.count_nonzero(column(rows, "outcome") == name)) for name in r.OUTCOMES}
        print(f"\n{label}: {len(rows)} starts; " + ", ".join(f"{k} {v}" for k, v in outcomes.items() if v))
        for name, words, digits in [
            ("arrival_time_s", "arrival (s)", 2),
            ("final_distance_m", "final distance from the target (m)", 4),
            ("join_time_s", "joins the path at (s)", 2),
            ("join_progress", "... at a progress of", 2),
            ("first_step_deg", "the reference's first step (deg)", 1),
            ("max_path_distance_deg", "largest distance from the take's path (deg)", 1),
        ]:
            print(
                f"  {words}: robot {quantiles(column(rows, name), digits)},"
                f" own {quantiles(column(alone, name), digits)}"
            )
        joined = column(rows, "join_time_s")
        print(
            f"  join the path: robot {np.isfinite(joined).sum()}, own {np.isfinite(column(alone, 'join_time_s')).sum()}"
        )
        torque = np.maximum(column(rows, "peak_tau1_nm"), column(rows, "peak_tau2_nm"))
        print(f"  largest joint torque (N m): {quantiles(torque, 1)};"
              f" joint 1 {quantiles(column(rows, 'peak_tau1_nm'), 1)},"
              f" joint 2 {quantiles(column(rows, 'peak_tau2_nm'), 1)}")  # fmt: skip
        print(f"  largest joint speed (deg/s): {quantiles(column(rows, 'peak_speed_dps'), 0)}")
        limit = column(rows, "limit_s")
        print(f"  runs resting against a joint's stop: {np.count_nonzero(limit)}, for up to {limit.max():.2f} s")
        a, b = arrivals(key)
        both = np.isfinite(a) & np.isfinite(b)
        diff = a[both] - b[both]
        print(
            f"  both arrive: {both.sum()}; robot later by a median {np.median(diff):+.2f} s"
            f" (10% {np.quantile(diff, 0.1):+.2f},"
            f" 90% {np.quantile(diff, 0.9):+.2f}); within 0.25 s: {np.mean(np.abs(diff) <= 0.25):.0%};"
            f" rank correlation {rank_correlation(a[both], b[both]):+.3f}"
        )
        near = column(rows, "start_path_distance_deg") <= 15.0
        print(
            f"  from the {near.sum()} starts within 15 deg of the path: robot later by a median"
            f" {np.median((a - b)[near & both]):+.2f} s;"
            f" within 0.25 s {np.mean(np.abs((a - b)[near & both]) <= 0.25):.0%}"
        )
        for start in r.SHOWN_STARTS:
            i = r.start_index(start)
            print(
                f"  from ({start[0]:g}, {start[1]:g}): arrival robot {rows[i]['arrival_time_s']:.2f} s, own"
                f" {alone[i]['arrival_time_s']:.2f} s; torque {torque[i]:.1f} N m;"
                f" first step {rows[i]['first_step_deg']:.1f} deg"
            )


def print_fixed_points() -> None:
    print("\nStep 2, the fixed points on its own and with the arm:")
    for key, label in ESNS:
        for row in step2(f"fixed_points_{key}.csv"):
            print(
                f"  {label} ({row['q1_deg']:.2f}, {row['q2_deg']:.2f}) deg: unstable {int(row['n_unstable'])} and"
                f" {int(row['system_n_unstable'])}; growth {row['growth_per_s']:+.3f}"
                f" and {row['system_growth_per_s']:+.3f} /s; time constant {row['time_constant_s']:.2f}"
                f" and {row['system_time_constant_s']:.2f} s; period"
                f" {row['period_s']:.2f} and {row['system_period_s']:.2f} s; residual {row['residual']:.1e}"
            )
        spectra = npz("fixed_points", f"spectra_{key}.npz")
        arm_modes = []
        for j in range(len(step2(f"fixed_points_{key}.csv"))):
            alone, together = spectra[f"eigenvalues_{j}"], spectra[f"system_eigenvalues_{j}"]
            far = [lam for lam in together if np.min(np.abs(alone - lam)) > 1e-3]
            arm_modes.append(sorted(far, key=lambda lam: -abs(lam))[:4])
        print(f"  {label}'s eigenvalues with the arm more than 1e-3 from any on its own: {arm_modes}")


def print_rates() -> None:
    print("\nStep 2, the rates along the run: median over the windows, robot and own")
    r, o = robot(), own()
    for key, label in ESNS:
        for window in (0.5, 1.0, 2.0, 4.0):
            a, b = r.rates(key), o.rates(key)
            across_r, across_o = a[f"across_{window:g}"][:, 0], b[f"across_{window:g}"][:, 0]
            posture_r, posture_o = a[f"posture_{window:g}"][:, 0], b[f"posture_{window:g}"][:, 0]
            print(
                f"  {label}, {window:g} s: across {np.median(across_r):+.3f} and {np.median(across_o):+.3f} /s"
                f" (difference {np.median(across_r) - np.median(across_o):+.3f}); ranges {across_r.min():+.2f} to"
                f" {across_r.max():+.2f} and {across_o.min():+.2f} to {across_o.max():+.2f}; posture scaled by"
                f" {np.median(posture_r):.2f} and {np.median(posture_o):.2f}"
            )


def print_pulses() -> None:
    print("\nStep 2, the pulses linearized: relative RMS error, median (10% to 90%, max)")
    for key, label in ESNS:
        rows = step2(f"pulses_{key}.csv")
        for force in (3.0, 6.0):
            chosen = [row for row in rows if row["force_n"] == force]
            for name in ("q", "reference"):
                e = column(chosen, f"{name}_relative_error")
                size = column(chosen, f"{name}_deviation_deg")
                print(
                    f"  {label}, {force:g} N, {name}: {np.median(e):.3f} ({np.quantile(e, 0.1):.3f} to"
                    f" {np.quantile(e, 0.9):.3f}, max {e.max():.3f}); deviation RMS {quantiles(size, 2)} deg"
                )
        three = {(row["onset"], row["angle_deg"]): row["q_relative_error"] for row in rows if row["force_n"] == 3.0}
        ratio = [
            row["q_relative_error"] / three[(row["onset"], row["angle_deg"])] for row in rows if row["force_n"] == 6.0
        ]
        print(f"  {label}: the error at 6 N over that at 3 N, median (range) {quantiles(np.array(ratio))}")


def print_changes() -> None:
    print("\nStep 3, the arm's changes within each leak rate (screen: fixed points, unstable, changed, growth ratio)")
    for c in step3("changes.csv"):
        print(
            f"  {int(c['screen'])}, {c['leak_rate']:g}: {int(c['fixed_points'])}, {int(c['unstable'])} unstable;"
            f" {int(c['unstable_to_attracting'])} to attracting, {int(c['attracting_to_unstable'])} to unstable;"
            f" ratio {c['growth_ratio_median']:.3f} ({c['growth_ratio_p10']:.3f} to {c['growth_ratio_p90']:.3f})"
        )


def print_rule_a() -> None:
    print("\nStep 3, rule (a): what changes with the arm (after the claims: within the stops on its own, and beyond)")
    for n in range(2):
        fixed, esns = step3(f"fixed_points_{n}.csv"), step3(f"esns_{n}.csv")
        radius = 0.02
        by_esn: dict[tuple[float, ...], list[dict[str, Any]]] = {}
        for p in fixed:
            by_esn.setdefault((p["leak_rate"], p["spectral_radius"], p["input_scaling"], p["seed"]), []).append(p)
        failing = ~column(esns, "passes").astype(bool)
        flags: dict[str, list[bool]] = {"within the stops, on its own": [], "beyond the stops, on its own": []}
        for e in esns:
            off = [p for p in by_esn.get((e["leak_rate"], e["spectral_radius"], e["input_scaling"], e["seed"]), [])
                   if p["target_distance_m"] > radius and p["n_unstable"] == 0]  # fmt: skip
            flags["within the stops, on its own"].append(any(p["within_stops"] for p in off))
            flags["beyond the stops, on its own"].append(any(not p["within_stops"] for p in off))
        for name, flag in [("rule (a), on its own", column(esns, "flag_a")), ("rule (a), with the arm",
                           column(esns, "flag_a_arm"))] + [(k, np.array(v)) for k, v in flags.items()]:  # fmt: skip
            flag = flag.astype(bool)
            print(f"  {SET_LABELS[n]}, {name}: flags {flag.sum()}, failing among them {failing[flag].mean():.2f},"
                  f" among the others {failing[~flag].mean():.2f};"
                  f" area under the ROC curve {auc(flag, failing):.3f}")  # fmt: skip


def auc(flag: NDArray[np.bool_], positive: NDArray[np.bool_]) -> float:
    """The area under the ROC curve of a flag: the chance a positive is flagged over a negative, ties half."""
    pos, neg = flag[positive].astype(float), flag[~positive].astype(float)
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def print_releases() -> None:
    print("\nStep 3, the releases: scores")
    for s in step3("scores.csv"):
        if s["against"] in ("stay", "stays put"):
            print(f"  {s['measure']} against {s['against']}, {s['leak_rate']}: {s['statistic']} {s['value']:.3f}"
                  f" over {int(s['n'])} (flagged {s['flagged']})")  # fmt: skip
    rows = step3("release.csv")
    stays = column(rows, "stays_s")
    put = np.isinf(stays)
    lr = column(rows, "leak_rate")
    for name in ("growth_per_s", "system_growth_per_s"):
        g = column(rows, name)
        print(f"  {name}: where the arm stays put {quantiles(g[put], 3)}; elsewhere {quantiles(g[~put], 3)}")
    print(f"  stays put: {put.sum()}, at leak rates {sorted(set(lr[put].tolist()))}")
    attracting = column(rows, "system_n_unstable") == 0
    print(f"  attracting with the arm: {attracting.sum()}, of which the arm stays put at {(attracting & put).sum()}")


def print_nudged() -> None:
    print("\nAdded after the claims: the releases nudged by 1e-9 deg, by leak rate")
    rows = read_csv(run_dir("nudged") / "release.csv")
    system = {(r["leak_rate"], r["spectral_radius"], r["input_scaling"], r["seed"], r["q1_deg"], r["q2_deg"]):
              r["system_growth_per_s"] for r in step3("release.csv")}  # fmt: skip
    lr = column(rows, "leak_rate")
    for name, group in (("all", lr > 0), ("up to 0.1", lr <= 0.1), (f"from {FAST_FROM:g}", lr >= FAST_FROM)):
        sub = [r for r, g in zip(rows, group, strict=True) if g]
        stays, nudged = column(sub, "stays_s"), column(sub, "nudged_stays_s")
        print(f"  {name}: {len(sub)} releases; stay the whole release as run {np.isinf(stays).sum()}, nudged"
              f" {np.isinf(nudged).sum()}")  # fmt: skip
        for words, (a, b) in (("as run", ("arrival_s", "own_arrival_s")),
                              ("nudged", ("nudged_arrival_s", "nudged_own_arrival_s"))):  # fmt: skip
            robot_arrival, own_arrival = column(sub, a), column(sub, b)
            both = np.isfinite(robot_arrival) & np.isfinite(own_arrival)
            diff = robot_arrival[both] - own_arrival[both]
            print(
                f"    {words}: both arrive {both.sum()}, only the robot"
                f" {(np.isfinite(robot_arrival) & ~np.isfinite(own_arrival)).sum()}, only on its own"
                f" {(~np.isfinite(robot_arrival) & np.isfinite(own_arrival)).sum()}, neither"
                f" {(~np.isfinite(robot_arrival) & ~np.isfinite(own_arrival)).sum()}; the robot a median"
                f" {np.median(diff):+.2f} s later (90% {np.quantile(diff, 0.9):+.2f}); within 0.25 s"
                f" {np.mean(np.abs(diff) <= 0.25):.0%};"
                f" rank correlation {rank_correlation(robot_arrival[both], own_arrival[both]):+.3f}"
            )
        growth = column(sub, "growth_per_s")
        arm = np.array([system[(r["leak_rate"], r["spectral_radius"], r["input_scaling"], r["seed"], r["q1_deg"],
                                r["q2_deg"])] for r in sub])  # fmt: skip
        ok = np.isfinite(arm)
        print(f"    the nudged stay's rank correlation with the growth rate on its own"
              f" {rank_correlation(growth[ok], nudged[ok]):+.3f},"
              f" with the arm {rank_correlation(arm[ok], nudged[ok]):+.3f}"
              f" (over {ok.sum()})")  # fmt: skip


def print_stayed() -> None:
    print("\nAdded after the claims: the releases where the arm stayed the whole 60 s as run, nudged")
    rows = [r for r in read_csv(run_dir("nudged") / "release.csv") if np.isinf(r["stays_s"])]
    nudged, growth = column(rows, "nudged_stays_s"), column(rows, "growth_per_s")
    left = np.isfinite(nudged)
    print(
        f"  {len(rows)}: {left.sum()} leave, after {quantiles(nudged[left], 2)} s; their departure's rank correlation"
        f" with the growth rate on its own {rank_correlation(growth[left], nudged[left]):+.3f}; those that stay grow"
        f" at {quantiles(growth[~left], 3)} /s, at leak rates {sorted(set(column(rows, 'leak_rate')[~left].tolist()))}"
    )
    for name in ("arrival_s", "nudged_arrival_s", "own_arrival_s", "nudged_own_arrival_s"):
        print(f"  {name}: {np.isfinite(column(rows, name)).sum()} arrive")


def print_weighings() -> None:
    print("\nAdded after the claims: the rates along the robot's run, the arm's values weighed otherwise")
    o = own()
    for w in read_csv(run_dir("weighed") / "weighings.csv"):
        alone = float(np.median(o.rates(w["esn"])[f"across_{w['window_s']:g}"][:, 0]))
        print(
            f"  {w['esn']}, joint angles x{w['angles']:g}, velocities x{w['velocities']:g}, {w['window_s']:g} s:"
            f" median {w['median_per_s']:+.3f} /s ({w['min_per_s']:+.2f} to {w['max_per_s']:+.2f}); on its own"
            f" {alone:+.3f}"
        )


def score(screen: int, measure: str, against: str, group: str) -> float:
    for s in step3("scores.csv"):
        if s["screen"] == screen and s["measure"] == measure and s["against"] == against and s["leak_rate"] == group:
            return float(s["value"])
    msg = f"no score {measure} against {against} over {group}"
    raise KeyError(msg)


def print_claims() -> None:
    print("\nThe claims:")
    r, o = robot(), own()
    for key, label in ESNS:
        rows = r.starts(key)
        at_target = int(np.count_nonzero(column(rows, "outcome") == "target"))
        joined = int(np.isfinite(column(rows, "join_time_s")).sum())
        a, b = arrivals(key)
        both = np.isfinite(a) & np.isfinite(b)
        rho = rank_correlation(a[both], b[both])
        print(f"  step 1, {label}: 1. at the target from {at_target} of {len(rows)}; 2. join {joined};"
              f" 3. rank correlation {rho:+.3f} over {both.sum()}")  # fmt: skip
    for key, label in ESNS:
        fixed = step2(f"fixed_points_{key}.csv")
        same = all(row["n_unstable"] == row["system_n_unstable"] for row in fixed)
        residual = max(row["residual"] for row in fixed)
        print(f"  step 2, {label}: 1. residual at most {residual:.1e}, the same unstable modes: {same}")
        for window in (2.0, 4.0):
            difference = np.median(r.rates(key)[f"across_{window:g}"][:, 0]) - np.median(
                o.rates(key)[f"across_{window:g}"][:, 0]
            )
            print(f"    2. {window:g} s windows: the median rate across the run differs by {difference:+.3f} /s")
    fast = f">={FAST_FROM:g}"
    print(
        f"  step 3: 1. at leak rates from {FAST_FROM:g}, the stay's rank correlation with the growth rate with the arm"
        f" {score(0, 'growth with the arm', 'stay', fast):+.3f}, on its own {score(0, 'own growth', 'stay', fast):+.3f}"
    )
    print(
        f"    2. the area under the ROC curve for staying put, with the arm"
        f" {score(0, 'growth with the arm', 'stays put', 'all'):.3f}, on its own"
        f" {score(0, 'own growth', 'stays put', 'all'):.3f}"
    )
    for n in range(2):
        print(
            f"    3. {SET_LABELS[n]}: rule (a) with the arm {score(n, 'flag_a_arm', 'failing', 'all'):.3f}, on its own"
            f" {score(n, 'flag_a', 'failing', 'all'):.3f}"
        )


def print_checks() -> None:
    print("\nChecks:")
    r, o = robot(), own()
    for key, label in ESNS:
        first = np.abs(column(r.starts(key), "first_step_deg") - column(o.starts(key), "first_step_deg")).max()
        print(f"  {label}: the reference's first step is the ESN's own to {first:.1e} deg")
        spectra, earlier = npz("fixed_points", f"spectra_{key}.npz"), o.spectra(key)
        gaps = [
            float(
                np.abs(
                    np.sort_complex(spectra[f"eigenvalues_{j}"]) - np.sort_complex(earlier[f"eigenvalues_{j}"])
                ).max()
            )
            for j in range(len(step2(f"fixed_points_{key}.csv")))
        ]
        print(f"  {label}: the eigenvalues on its own are report 019's to {max(gaps):.1e}")
    for n in range(2):
        rows = step3(f"fixed_points_{n}.csv")
        ways = {way: sum(row["found"] == way for row in rows) for way in ("reset", "cells", "random", "")}
        within = [row for row in rows if row["found"] and row["within_stops"]]
        print(
            f"  screen {n}: {len(rows)} fixed points; found from the reset {ways['reset']}, from the cells"
            f" {ways['cells']}, from random states {ways['random']}, not found {ways['']}; beyond the stops"
            f" {sum(not row['within_stops'] for row in rows)}; the system's residual at most"
            f" {max(row['residual'] for row in within):.1e} at the {len(within)} others"
        )
    weighed = read_csv(run_dir("weighed") / "weighings.csv")
    for key, _ in ESNS:
        base = next(w for w in weighed if w["esn"] == key and w["angles"] == 1.0 and w["velocities"] == 1.0
                    and w["window_s"] == 4.0)  # fmt: skip
        step2_median = float(np.median(robot().rates(key)["across_4"][:, 0]))
        print(f"  {key}: the weighed run's 4 s median at the planned weighing is step 2's to"
              f" {abs(base['median_per_s'] - step2_median):.1e} /s")  # fmt: skip
    nudged = read_csv(run_dir("nudged") / "release.csv")
    print(f"  the nudged releases: {len(nudged)}, as many as the release run's")


if __name__ == "__main__":
    main()
