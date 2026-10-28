# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 019.

    uv run python reports/019-dynamics-under-one-protocol/make_figures.py

It reads part 3's runs under the storage root, whose arrays are not kept in Git, copies
their configurations, run records, and CSV files into ``results/``, prints the tables of
the report, runs its checks against reports 015 and 016, and draws into
``results/summary``:

- report 015's analyses of ESN #1 and ESN #2 (Sections 3.1 to 3.4), drawn by report
  015's own code pointed at these runs: ``arrival.png`` (the arrival over the start
  postures), ``series.png`` (the joint angles from six starts), ``warmup.png``,
  ``pulse_maps.png`` (the measures of the pulses of 6 N), ``fixed_points.png``, and
  ``rates.png``;
- ``pulses_esn1.png`` and ``pulses_esn2.png``: each ESN's joint angles and torques over
  time against both replays', undisturbed and under four pulses (Section 3.3);
- the screen against the robot (Sections 3.5 and 3.6): ``screen.png`` (the screen's
  measures by leak rate), ``rules.png`` (the share failing among the ESNs each rule
  flags and among the others), and ``scores.png`` (within each leak rate);
- held still on the robot (Section 3.7): ``release.png`` (the robot's arrival after a
  release against the ESN's own), ``blocks.png`` (the blocks' failures and delays),
  and ``shown_esn1.png`` and ``shown_esn2.png``: each ESN's joint angles and torques
  over time against both replays', undisturbed, under two of the blocks, and released
  at ESN #2's unstable fixed point beside the path.
"""

from __future__ import annotations

import csv
import functools
import importlib.util
import shutil
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from scipy.stats import rankdata
from skelarm import StateLog

from arm_esn_ctrl.demonstrations import resample_joint_angles
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
STUDY = "results/manual_demonstration_v2_consistent_dynamics"
SEARCH = "results/manual_demonstration_v2_consistent_search"
# Part 3's runs, under the storage root.
RUNS = {
    "global_starts": f"{STUDY}/20261027-180441-global_starts",
    "warmup_esn1": f"{STUDY}/20261027-180441-warmup_esn1_lr0.03_sr1.2_is1.5_seed8",
    "warmup_esn2": f"{STUDY}/20261027-180441-warmup_esn2_lr0.03_sr1.2_is1_seed2",
    "pulses": f"{STUDY}/20261027-180441-push_pulses",
    "fixed_points": f"{STUDY}/20261027-183137-fixed_points",
    "screen": f"{STUDY}/20261027-183607-screen",
    "screen_heldout": f"{STUDY}/20261028-064044-screen_seeds10to19",
    "against": f"{STUDY}/20261028-044632-against_robot",
    "against_heldout": f"{STUDY}/20261028-115552-against_robot_seeds10to19",
    "release": f"{STUDY}/20261028-044731-release_robot",
    "block5_esn1": f"{STUDY}/20261028-155031-block5to8_robot_esn1_lr0.03_sr1.2_is1.5_seed8",
    "block5_esn2": f"{STUDY}/20261028-155031-block5to8_robot_esn2_lr0.03_sr1.2_is1_seed2",
    "block8_esn1": f"{STUDY}/20261028-155031-block8to11_robot_esn1_lr0.03_sr1.2_is1.5_seed8",
    "block8_esn2": f"{STUDY}/20261028-155031-block8to11_robot_esn2_lr0.03_sr1.2_is1_seed2",
    "released_esn1": f"{STUDY}/20261028-155031-released_robot_esn1_lr0.03_sr1.2_is1.5_seed8",
    "released_esn2": f"{STUDY}/20261028-155031-released_robot_esn2_lr0.03_sr1.2_is1_seed2",
    # The search's step 4, undisturbed, which report 017 keeps.
    "nominal_esn1": f"{SEARCH}/20261026-132722-nominal_robot_esn1_lr0.03_sr1.2_is1.5_seed8",
    "nominal_esn2": f"{SEARCH}/20261026-132722-nominal_robot_esn2_lr0.03_sr1.2_is1_seed2",
}
COPIED_ELSEWHERE = ("nominal_esn1", "nominal_esn2")
# The take, as report 005 copied it: what the ESNs were trained on, and what the replays replay.
TAKE_RUN = REPORT.parent / "005-tuned-on-the-robot" / "data" / "20261008-170034-reach_manual_v2"
DT = 0.01  # the ESNs' period (s)
DPI = 90
SETTING = "pd_w20_z0.4"  # the search's tracker: joint PD at omega = 20 rad/s and the damping ratio 0.4
LEAK_RATES = (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5)
PER_LEAK_RATE = 420  # ESNs of each set at each leak rate: 7 spectral radii, 6 input scalings, 10 seeds
LATE_S = 9.58  # the take's arrival (s)
# Report 016's grid, within the search's: its leak rates, spectral radii, and input scalings.
REPORT_016_GRID = ((0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1), (0.99, 1.2, 1.35, 1.5, 1.75), (1.0, 1.5, 2.0))

# The named ESNs: leak rate, spectral radius, input scaling, seed.
NAMED = {
    "ESN #1": (0.03, 1.2, 1.5, 8),
    "ESN #2": (0.03, 1.2, 1.0, 2),
    "ESN #3": (0.03, 1.35, 1.5, 6),
    "report 007's": (0.02, 1.35, 2.0, 7),
}
# The two sets of ESNs: the step 6 run, the screen, the label, and the color.
SETS = [
    ("against", "screen", "seeds 0 to 9", "#24527a"),
    ("against_heldout", "screen_heldout", "seeds 10 to 19, held out", "#1b9e77"),
]
# The arms over time, the ESN's lines over the replays': color, and the layers of the arm and of its reference.
SERIES_COLORS = {"esn": "#2a78d6", "replay": "#eb6834", "progress": "#1baf7a"}
ARM_ZORDERS = {"esn": 3.0, "progress": 2.2, "replay": 2.0}
REFERENCE_ZORDERS = {"esn": 2.5, "progress": 2.1}
# The pulses drawn over time, as report 015 drew them: none, and (onset, angle, force).
SHOWN_PULSES: list[tuple[float, float, float] | None] = [
    None,
    (4.0, 0.0, 6.0),
    (4.0, 180.0, 6.0),
    (7.0, 90.0, 6.0),
    (12.0, 270.0, 6.0),
]
# The shown runs on the robot: the runs' name, and the column's title.
SHOWN = [
    ("nominal", "undisturbed (the search's step 4)"),
    ("block5", "the tip held from 5 s to 8 s"),
    ("block8", "the tip held from 8 s to 11 s"),
    ("released", "released at ESN #2's unstable fixed point\nbeside the path, after 4 s there"),
]

TEXT_COLOR = "#52514e"
TITLE_COLOR = "#0b0b0b"
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAUGHT_COLOR = "#b9b7b0"
BASE_COLOR = "#d9d8d3"
NEAR_COLOR = "#c0392b"
NONE_COLOR = "#24527a"

Key = tuple[float, float, float, int]  # leak rate, spectral radius, input scaling, seed


def main() -> None:
    copy_results()
    SUMMARY.mkdir(parents=True, exist_ok=True)
    report_015 = earlier_015()
    figures = {
        "arrival.png": report_015.plot_arrival(),
        "series.png": report_015.plot_series(),
        "warmup.png": report_015.plot_warmup(),
        "pulse_maps.png": report_015.plot_pulse_maps(6.0),
        "fixed_points.png": report_015.plot_fixed_points(),
        "rates.png": report_015.plot_rates(),
        "pulses_esn1.png": plot_over_time("ESN #1", pulse_columns("esn1")),
        "pulses_esn2.png": plot_over_time("ESN #2", pulse_columns("esn2")),
        "screen.png": plot_screen(),
        "rules.png": plot_rules(),
        "scores.png": plot_scores(),
        "release.png": plot_release(),
        "blocks.png": plot_blocks(),
        "shown_esn1.png": plot_over_time("ESN #1", shown_columns("esn1")),
        "shown_esn2.png": plot_over_time("ESN #2", shown_columns("esn2")),
    }
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=DPI)
    report_015.print_global_starts()
    report_015.print_warmup()
    report_015.print_pulses()
    report_015.print_fixed_points()
    print_screen()
    print_named()
    print_rules()
    print_aucs()
    print_claims()
    print_release()
    print_shown()
    print_checks()
    print(f"\nWrote {', '.join(figures)} to {SUMMARY}")


# ---------------------------------------------------------------------------- the runs


def run_dir(name: str) -> Path:
    return resolve_run_path(RUNS[name])


def run_name(name: str) -> str:
    return Path(RUNS[name]).name


def copy_results() -> None:
    """Copy each run's configuration, run record, and CSV files into ``results/<run>``, but report 017's own."""
    for name in RUNS:
        if name in COPIED_ELSEWHERE:
            continue
        target = RESULTS / run_name(name)
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(run_dir(name).glob("*.toml")) + sorted(run_dir(name).glob("*.csv")):
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
def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [{key: value(text) for key, text in row.items()} for row in csv.DictReader(f)]


def rows_of(name: str, file: str) -> list[dict[str, Any]]:
    return read_csv(RESULTS / run_name(name) / file)


def config_of(name: str) -> dict[str, Any]:
    with (run_dir(name) / "config.toml").open("rb") as f:
        return tomllib.load(f)


def key_of(row: dict[str, Any]) -> Key:
    return (row["leak_rate"], row["spectral_radius"], row["input_scaling"], int(row["seed"]))


def column(rows: list[dict[str, Any]], name: str) -> NDArray[np.float64]:
    return np.array([r[name] for r in rows], dtype=np.float64)


def by_key(rows: list[dict[str, Any]]) -> dict[Key, dict[str, Any]]:
    return {key_of(r): r for r in rows}


def leak_rate_groups(rows: list[dict[str, Any]]) -> list[NDArray[np.bool_]]:
    lr = column(rows, "leak_rate")
    return [lr == v for v in LEAK_RATES]


def in_report_016(key: Key) -> bool:
    """Whether an ESN's setting is in report 016's grid (and report 011's)."""
    return all(v in values for v, values in zip(key[:3], REPORT_016_GRID, strict=True))


def rule_row(name: str, rule: str, group: str) -> dict[str, Any]:
    (row,) = [r for r in rows_of(name, "rules.csv") if r["rule"] == rule and str(r["leak_rate"]) == group]
    return row


def group_name(lr: float) -> str:
    return str(float(lr))


def score(name: str, measure: str, against: str, group: str) -> float:
    (row,) = [
        r
        for r in rows_of(name, "scores.csv")
        if r["measure"] == measure and r["against"] == against and str(r["leak_rate"]) == group
    ]
    return float(row["value"])


def load_script(report: str) -> ModuleType:
    """An earlier report's make_figures.py, as a module."""
    path = REPO_ROOT / "reports" / report / "make_figures.py"
    spec = importlib.util.spec_from_file_location(f"report_{report[:3]}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@functools.cache
def earlier_015() -> ModuleType:
    """Report 015's script, pointed at part 3's runs and drawing into this report's summary."""
    module: Any = load_script("015-what-the-esns-learned")
    module.STUDY = STUDY
    module.RESULTS, module.SUMMARY = RESULTS, SUMMARY
    for name in module.RUNS:
        module.RUNS[name] = run_name(name)
    return module


@functools.cache
def earlier_runs(report: str) -> dict[str, Path]:
    """An earlier report's runs, by its script's names, under the storage root."""
    module = load_script(report)
    if report.startswith("015"):
        return {name: resolve_run_path(f"{module.STUDY}/{run}") for name, run in module.RUNS.items()}
    return {name: resolve_run_path(run) for name, (run, _) in module.RUNS.items()}


@functools.cache
def taught() -> NDArray[np.float64]:
    """The take filtered at 2 Hz, what the ESNs were trained on, every 10 ms: its joint angles."""
    return resample_joint_angles(StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"), DT)[1]


def rank_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Spearman's rank correlation over the entries where neither is NaN; an infinity ranks last."""
    ok = ~np.isnan(x) & ~np.isnan(y)
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])


def auc(score_: NDArray[np.float64], positive: NDArray[np.bool_]) -> float:
    """The chance that a positive's score exceeds a negative's, ties counted half."""
    pos, neg = score_[positive], score_[~positive]
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------- over time


@dataclass
class Series:
    """One arm's run over time: its joint angles, its reference, its torques, and the force at its tip."""

    times: NDArray[np.float64]
    q: NDArray[np.float64]
    q_ref: NDArray[np.float64]
    tau: NDArray[np.float64]
    force: NDArray[np.float64]


@dataclass
class Column:
    """A column of a figure over time: its title, its disturbances' spans, and each arm's series."""

    title: str
    spans: list[tuple[float, float]]
    arms: dict[str, Series]


@functools.cache
def pulse_runs() -> dict[str, NDArray[Any]]:
    with np.load(run_dir("pulses") / "runs.npz") as data:
        return {name: data[name] for name in data.files}


def pulse_columns(esn: str) -> list[Column]:
    """One ESN's runs under the pulses, and both replays', undisturbed and under report 015's four shown pulses."""
    data = pulse_runs()
    length = config_of("pulses")["pulses"]["duration"]
    columns = []
    for pulse in SHOWN_PULSES:
        arms = {}
        for arm, key in (("esn", esn), ("replay", "replay"), ("progress", "progress")):
            match = data["arm"] == key
            if pulse is None:
                match &= ~np.isfinite(data["onset"])
            else:
                onset, angle, force = pulse
                match &= (data["onset"] == onset) & (data["angle_deg"] == angle) & (data["force_n"] == force)
            (i,) = np.flatnonzero(match)
            arms[arm] = Series(
                data["times"],
                data["q"][i].astype(np.float64),
                data["q_ref"][i].astype(np.float64),
                data["tau"][i].astype(np.float64),
                np.linalg.norm(data["ext_force"][i], axis=1),
            )
        spans = [] if pulse is None else [(pulse[0], pulse[0] + length)]
        columns.append(Column(earlier_015().pulse_words(pulse), spans, arms))
    return columns


def shown_columns(esn: str) -> list[Column]:
    """One ESN's shown runs, and both replays' beside it."""
    columns = []
    for scenario, title in SHOWN:
        run = run_dir(f"{scenario}_{esn}")
        arms = {}
        for arm in SERIES_COLORS:
            log = StateLog.load(run / SETTING / f"{arm}_00.sklog.npz")
            try:
                force = np.linalg.norm(log.channel("ext_force"), axis=1)
            except KeyError:  # an undisturbed run's log has no force
                force = np.zeros(len(log.times))
            arms[arm] = Series(log.times, log.channel("q"), log.channel("q_ref"), log.channel("tau"), force)
        columns.append(Column(title, disturbance_spans(config_of(f"{scenario}_{esn}").get("disturbance")), arms))
    return columns


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


def padded(values: list[NDArray[np.float64]]) -> tuple[float, float]:
    """An axis's limits around ``values``, with a margin of 5% on each side."""
    low, high = float(min(v.min() for v in values)), float(max(v.max() for v in values))
    return low - 0.05 * (high - low), high + 0.05 * (high - low)


def plot_over_time(name: str, columns: list[Column]) -> Figure:
    """One ESN's joint angles and torques over time against both replays', with the search's tracker: a column each.

    The joint angles share their axes across the columns; the torques' fit each column.
    """
    fig = new_figure(
        4.6 * len(columns) + 1.0,
        13.5,
        f"{name}'s arm (blue), the replay's (orange), and the progress replay's (aqua), with joint PD at ω 20 rad/s,"
        " ζ 0.4, given the reference velocity;\ndashed: the references of the ESN and the progress replay; thick gray:"
        " the taught motion",
    )
    axes = fig.subplots(5, len(columns), height_ratios=[0.55, 1.0, 1.0, 1.0, 1.0])
    q_taught = np.degrees(taught())
    t_taught = DT * np.arange(len(q_taught))
    top = max(float(s.force.max()) for c in columns for s in c.arms.values())
    angle_limits = []
    for joint in range(2):
        values = [np.degrees(s.q[:, joint]) for c in columns for s in c.arms.values()] + [q_taught[:, joint]]
        angle_limits.append(padded(values))
    for col, column_ in enumerate(columns):
        start = min(float(s.times[0]) for s in column_.arms.values())
        end = max(float(s.times[-1]) for s in column_.arms.values())
        for ax in axes[:, col]:
            style(ax)
            for onset, stop in column_.spans:
                ax.axvspan(onset, stop, color=DISTURBANCE_COLOR, zorder=0)
            ax.set_xlim(start, end)
        axes[0, col].set_title(column_.title, color=TITLE_COLOR, fontsize=10)
        for arm, color in SERIES_COLORS.items():
            s = column_.arms[arm]
            axes[0, col].plot(s.times, s.force, color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm])
            for joint in range(2):
                ax = axes[1 + joint, col]
                ax.plot(s.times, np.degrees(s.q[:, joint]), color=color, linewidth=1.2, zorder=ARM_ZORDERS[arm])
                if arm in REFERENCE_ZORDERS:
                    ax.plot(
                        s.times,
                        np.degrees(s.q_ref[:, joint]),
                        color=color,
                        linewidth=1.0,
                        linestyle="--",
                        zorder=REFERENCE_ZORDERS[arm],
                    )
                axes[3 + joint, col].plot(s.times, s.tau[:, joint], color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm])
        for joint in range(2):
            axes[1 + joint, col].plot(t_taught, q_taught[:, joint], color=TAUGHT_COLOR, linewidth=4.0, zorder=1)
        axes[0, col].set_ylim(-0.05 * top, 1.1 * top)
        for joint in range(2):
            axes[1 + joint, col].set_ylim(*angle_limits[joint])
            axes[3 + joint, col].set_ylim(*padded([s.tau[:, joint] for s in column_.arms.values()]))
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


# ---------------------------------------------------------------------------- the screen against the robot


def leak_rate_axis(ax: Axes, *, narrow: bool = False) -> None:
    """The leak rates along the x axis, tilted on a ``narrow`` axis."""
    ax.set_xticks(range(len(LEAK_RATES)), [f"{v:g}" for v in LEAK_RATES], rotation=45 if narrow else 0)
    ax.set_xlim(-0.6, len(LEAK_RATES) - 0.4)
    ax.set_xlabel("leak rate", color=TEXT_COLOR)


def set_handles() -> list[Line2D]:
    return [Line2D([], [], color=color, marker="o", linewidth=1.2, label=label) for _, _, label, color in SETS]


def screen_measures(name: str) -> dict[str, NDArray[np.float64]]:
    """Per ESN of a screen: what screen.png and the screen's table show."""
    rows = rows_of(name, "screen.csv")
    radius = 0.02
    within = {
        key_of(p)
        for p in rows_of(name, "fixed_points.csv")
        if p["n_unstable"] == 0 and p["target_distance_m"] <= radius
    }
    return {
        "attracting within the goal": np.array([key_of(r) in within for r in rows], dtype=np.float64),
        "attracting off the target": (column(rows, "attracting_off_target") > 0).astype(np.float64),
        "unstable": (column(rows, "unstable") > 0).astype(np.float64),
        "unstable within 30° of the path": (column(rows, "nearest_unstable_path_distance_deg") <= 30.0).astype(
            np.float64
        ),
        "released, arriving after 9.58 s or never": (
            (column(rows, "unstable") > 0) & ~(column(rows, "release_arrival_s") <= LATE_S)
        ).astype(np.float64),
        "all 144 starts from rest end at the target": (column(rows, "starts_at_target") == 144).astype(np.float64),
        "lead": column(rows, "lead_median_deg"),
        "time constant": column(rows, "target_time_constant_s"),
        "posture scaling": column(rows, "posture_scale_max"),
    }


def plot_screen() -> Figure:
    """The screen's measures by leak rate, of both sets."""
    fig = new_figure(15.0, 7.6, f"The screen's measures by leak rate ({PER_LEAK_RATE} ESNs each)")
    axes = fig.subplots(2, 3)
    panels = [
        ("attracting off the target", "share with an attracting fixed point\noff the target", "share"),
        ("unstable within 30° of the path", "share with an unstable fixed point\nwithin 30° of the path", "share"),
        (
            "all 144 starts from rest end at the target",
            "share whose 144 runs from rest\nall end at the target",
            "share",
        ),
        ("lead", "median lead near the path (deg)", "median"),
        ("time constant", "the target's slowest time constant (s),\nmedian", "median"),
        ("posture scaling", "largest scaling of a posture's\ndeviation, median", "median"),
    ]
    for ax, (measure, label, how) in zip(axes.flat, panels, strict=True):
        style(ax)
        for k, (_, screen, _, color) in enumerate(SETS):
            values = screen_measures(screen)[measure]
            groups = leak_rate_groups(rows_of(screen, "screen.csv"))
            y = [
                values[g].mean() if how == "share" else float(np.median(values[g][np.isfinite(values[g])]))
                for g in groups
            ]
            ax.plot(np.arange(len(LEAK_RATES)) + (k - 0.5) * 0.15, y, "o-", color=color, markersize=4)
        ax.set_ylabel(label, color=TEXT_COLOR)
        leak_rate_axis(ax, narrow=True)
    fig.legend(handles=set_handles(), loc="outside lower center", ncol=2, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_rules() -> Figure:
    """The share failing among the ESNs each rule flags and among the others, by leak rate, with the share among all."""
    fig = new_figure(
        15.0,
        10.0,
        "The share of ESNs failing the search's rule on the robot among those each rule flags (filled) and the"
        f" others (open), by leak rate ({PER_LEAK_RATE} ESNs each); gray: among all, seeds 0 to 9",
    )
    axes = fig.subplots(3, 1, sharex=True)
    titles = {
        "a": "(a) an attracting fixed point off the target",
        "b": "(b) a median lead near the path under 10°",
        "c": "(c) an unstable fixed point within 30° of the path, or a late release",
    }
    for ax, rule in zip(axes, ("a", "b", "c"), strict=True):
        style(ax)
        base = [rule_row("against", rule, group_name(lr))["failing"] / PER_LEAK_RATE for lr in LEAK_RATES]
        ax.bar(range(len(LEAK_RATES)), base, color=BASE_COLOR, width=0.7, zorder=1)
        for k, (against, _, _, color) in enumerate(SETS):
            x = np.arange(len(LEAK_RATES)) + (k - 0.5) * 0.2
            rows = [rule_row(against, rule, group_name(lr)) for lr in LEAK_RATES]
            ax.plot(x, [r["failing_share_flagged"] for r in rows], "o", color=color, markersize=7, zorder=3)
            ax.plot(
                x,
                [r["failing_share_unflagged"] for r in rows],
                "o",
                color=color,
                markerfacecolor="white",
                markersize=7,
                zorder=3,
            )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("share failing", color=TEXT_COLOR)
        ax.set_title(titles[rule], color=TITLE_COLOR, fontsize=10)
    leak_rate_axis(axes[-1])
    handles = [
        *set_handles(),
        Line2D([], [], color=TEXT_COLOR, marker="o", linestyle="none", label="flagged"),
        Line2D([], [], color=TEXT_COLOR, marker="o", markerfacecolor="white", linestyle="none", label="not flagged"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_scores() -> Figure:
    """Within each leak rate: the areas under the ROC curve and the rank correlations of report 016's claims."""
    fig = new_figure(15.0, 7.6, "Within each leak rate, of both sets")
    axes = fig.subplots(2, 2, sharex=True)
    panels = [
        ("flag_a", "failing", "(a): area under the ROC curve\nfor failing", 0.5),
        ("nearer unstable fixed point", "failing", "an unstable fixed point nearer the path:\nAUC for failing", 0.5),
        ("smaller lead", "ends_away", "a smaller lead: AUC for\nending away under the pushes", 0.5),
        (
            "posture_scale_max",
            "following",
            "largest scaling of a posture's deviation:\nrank correlation with following",
            0.0,
        ),
    ]
    for ax, (measure, against, label, chance) in zip(axes.flat, panels, strict=True):
        style(ax)
        ax.axhline(chance, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
        for k, (name, _, _, color) in enumerate(SETS):
            x = np.arange(len(LEAK_RATES)) + (k - 0.5) * 0.15
            ax.plot(x, [score(name, measure, against, group_name(lr)) for lr in LEAK_RATES], "o-", color=color)
        ax.set_ylabel(label, color=TEXT_COLOR)
    for ax in axes[-1]:
        leak_rate_axis(ax)
    fig.legend(handles=set_handles(), loc="outside lower center", ncol=2, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_release() -> Figure:
    """The robot's arrival after a release against the ESN's own after the same hold."""
    rows = rows_of("release", "release.csv")
    robot, own = column(rows, "arrival_s"), column(rows, "own_arrival_s")
    fig = new_figure(
        13.0,
        6.2,
        f"Released on the robot at each unstable fixed point within 30° of the path of the passing ESNs ({len(rows)}),"
        " after 4 s there",
    )
    left, right = fig.subplots(1, 2)
    for ax in (left, right):
        style(ax)
    cap = 62.0  # never arriving within 60 s is drawn here
    shown_robot, shown_own = np.where(np.isfinite(robot), robot, cap), np.where(np.isfinite(own), own, cap)
    left.plot([0, 60], [0, 60], color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    left.scatter(shown_own, shown_robot, s=10, color=NONE_COLOR, alpha=0.5, zorder=3)
    left.set_xlabel("the ESN on its own: arrival from the release (s); 62: never", color=TEXT_COLOR)
    left.set_ylabel("the robot: arrival from the release (s); 62: never", color=TEXT_COLOR)
    left.set_xlim(0, 64)
    left.set_ylim(0, 64)
    left.set_aspect("equal")
    both = np.isfinite(robot) & np.isfinite(own)
    right.hist(robot[both] - own[both], bins=np.arange(-1.0, 3.01, 0.05), color=NONE_COLOR)
    right.set_xlabel("the robot's arrival less the ESN's own (s), where both arrive", color=TEXT_COLOR)
    right.set_ylabel("releases", color=TEXT_COLOR)
    return fig


def plot_blocks() -> Figure:
    """The passing ESNs under the blocks: the share failing, and the delay, by onset."""
    blocks = rows_of("release", "blocks.csv")
    near, hold = column(blocks, "near_path_unstable") > 0, column(blocks, "hold_s")
    passes, onset, delay = column(blocks, "passes") > 0, column(blocks, "onset_s"), column(blocks, "delay_s")
    n_esns = len({key_of(b) for b in blocks})
    fig = new_figure(
        13.0,
        6.2,
        f"The {n_esns:,} passing ESNs on the robot, the tip held for 1 s or 3 s from each onset, one block a run"
        f" ({len(blocks):,} runs)",
    )
    left, right = fig.subplots(1, 2)
    onsets = np.unique(onset)
    for ax in (left, right):
        style(ax)
        ax.set_xlabel("block onset (s)", color=TEXT_COLOR)
    for group, color in ((near, NEAR_COLOR), (~near, NONE_COLOR)):
        for h, style_ in ((1.0, "-"), (3.0, "--")):
            s = group & (hold == h)
            left.plot(onsets, [(~passes[s & (onset == o)]).mean() for o in onsets], "o" + style_, color=color)
            right.plot(
                onsets,
                [np.median(delay[s & (onset == o) & np.isfinite(delay)]) for o in onsets],
                "o" + style_,
                color=color,
            )
    left.set_ylabel("share failing the rule under the block", color=TEXT_COLOR)
    right.set_ylabel("delay of those arriving, median (s)", color=TEXT_COLOR)
    right.axhline(0.0, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    handles = [
        Line2D([], [], color=NEAR_COLOR, marker="o", label="with an unstable fixed point within 30° of the path"),
        Line2D([], [], color=NONE_COLOR, marker="o", label="without"),
        Line2D([], [], color=TEXT_COLOR, linestyle="-", label="held 1 s"),
        Line2D([], [], color=TEXT_COLOR, linestyle="--", label="held 3 s"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- the tables


def print_screen() -> None:
    """Step 5's measures by leak rate, of both sets."""
    for _, name, label, _ in SETS:
        rows = rows_of(name, "screen.csv")
        measures = screen_measures(name)
        groups = leak_rate_groups(rows)
        print(f"\nThe screen, {label}: per leak rate ({PER_LEAK_RATE} ESNs each)")
        print("| | " + " | ".join(f"{v:g}" for v in LEAK_RATES) + " |")
        print("| --- |" + " ---: |" * len(LEAK_RATES))
        for measure in (
            "attracting within the goal",
            "attracting off the target",
            "unstable",
            "unstable within 30° of the path",
            "released, arriving after 9.58 s or never",
            "all 144 starts from rest end at the target",
        ):
            print(f"| {measure} | " + " | ".join(f"{measures[measure][g].sum():.0f}" for g in groups) + " |")
        for measure, digits in (("lead", 1), ("time constant", 2), ("posture scaling", 2)):
            values = measures[measure]
            cells = [f"{np.median(values[g & np.isfinite(values)]):.{digits}f}" for g in groups]
            print(f"| {measure}, median | " + " | ".join(cells) + " |")
        held = column(rows, "held_unconverged") > 0
        rho = column(rows, "spectral_radius")
        fps = rows_of(name, "fixed_points.csv")
        print(
            f"held reservoir not settling at some posture: {held.sum():.0f} ESNs ({held[rho == 1.75].sum():.0f} at"
            f" 1.75, {held[rho == 1.5].sum():.0f} at 1.5); fixed points {len(fps)},"
            f" {sum(not p['from_grid'] for p in fps)} only where runs come to rest; time per ESN, median"
            f" {np.median(column(rows, 'elapsed_s')):.0f} s"
        )


def print_named() -> None:
    """The named ESNs: the screen's measures and the robot's outcomes."""
    screen = by_key(rows_of("screen", "screen.csv"))
    esns = by_key(rows_of("against", "esns.csv"))
    print("\nThe named ESNs")
    print("| | " + " | ".join(NAMED) + " |")
    print("| --- |" + " --- |" * len(NAMED))
    lines = [
        (
            "fixed points (attracting, unstable)",
            lambda s, e: f"{s['fixed_points']:.0f} ({s['attracting']:.0f}, {s['unstable']:.0f})",
        ),
        (
            "the target's slowest mode",
            lambda s, e: f"{s['target_time_constant_s']:.2f} s, period {s['target_period_s']:.1f} s",
        ),
        (
            "the unstable fixed point nearest the path",
            lambda s, e: (
                "none"
                if not np.isfinite(s["nearest_unstable_path_distance_deg"])
                else f"{s['nearest_unstable_path_distance_deg']:.1f}°, {s['nearest_unstable_progress']:.2f} of the way"
            ),
        ),
        (
            "released after 4 s, the latest arrival",
            lambda s, e: "" if s["unstable"] == 0 else f"{s['release_arrival_s']:.2f} s",
        ),
        ("median lead near the path", lambda s, e: f"{s['lead_median_deg']:.1f}°"),
        ("largest scaling of a posture's deviation", lambda s, e: f"{s['posture_scale_max']:.2f}"),
        ("flagged by", lambda s, e: ", ".join(f"({c[5:]})" for c in ("flag_a", "flag_b", "flag_c") if e[c]) or "none"),
        (
            "on the robot",
            lambda s, e: (
                f"{'passes' if e['passes'] else 'fails'}, following {e['following']:.2f}"
                + (f", fails from {e['offset_failures']:.0f} offset starts" if e["offset_failures"] else "")
            ),
        ),
    ]
    for title, cell in lines:
        print(f"| {title} | " + " | ".join(cell(screen[k], esns[k]) for k in NAMED.values()) + " |")


def print_rules() -> None:
    """The rules' counts over all the ESNs, and the shares failing by leak rate."""
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
        print("| Share failing, flagged / not | " + " | ".join(f"{v:g}" for v in LEAK_RATES) + " |")
        print("| --- |" + " --- |" * len(LEAK_RATES))
        for rule in ("a", "b", "c"):
            cells = []
            for lr in LEAK_RATES:
                r = rule_row(name, rule, group_name(lr))
                flagged = "none" if np.isnan(r["failing_share_flagged"]) else f"{r['failing_share_flagged']:.2f}"
                others = "none" if np.isnan(r["failing_share_unflagged"]) else f"{r['failing_share_unflagged']:.2f}"
                cells.append(f"{flagged} / {others}")
            print(f"| ({rule}) | " + " | ".join(cells) + " |")
        shares = [rule_row(name, "a", group_name(lr))["failing"] / PER_LEAK_RATE for lr in LEAK_RATES]
        print("| all | " + " | ".join(f"{s:.2f}" for s in shares) + " |")
        esns = rows_of(name, "esns.csv")
        failing = [e for e in esns if not e["passes"]]
        only_offsets = [e for e in failing if e["offset_failures"] > 0 and not e["ends_away"]]
        print(
            f"failing {len(failing)}: ending away under the pushes {sum(e['ends_away'] for e in failing)}, never"
            f" arriving {sum(e['never_arrives'] for e in failing)}, failing from an offset start"
            f" {sum(e['offset_failures'] > 0 for e in failing)} (and not ending away {len(only_offsets)})"
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
            ("posture_scale_max", "return_s"),
        ):
            cells = [f"{score(name, measure, against, group_name(lr)):+.2f}" for lr in LEAK_RATES]
            print(
                f"{measure} vs {against}: all {score(name, measure, against, 'all'):+.2f}; within " + ", ".join(cells)
            )
        print(
            "posture_scale_max vs following, pooled within:"
            f" {score(name, 'posture_scale_max', 'following', 'within each leak rate'):+.2f};"
            f" the leak rate vs following {score(name, 'leak rate', 'following', 'all'):+.2f}"
        )


@functools.cache
def failing_without_offsets(name: str) -> dict[Key, bool]:
    """Whether each ESN fails the rule undisturbed, under the block, or under the pushes: report 016's outcome."""
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    from screen_against_robot import outcomes, robot_runs

    robot, _ = robot_runs([run for run in config_of(name)["runs"]["robot"] if not run.endswith("_offsets")])
    return {k: not outcomes(v, 0.02)["passes"] for k, v in robot.items()}


def aucs(name: str, members: NDArray[np.bool_], failing: NDArray[np.bool_]) -> dict[str, float]:
    """The areas under the ROC curve for failing of the leak rate, the lead, and the rules' flags."""
    esns = rows_of(name, "esns.csv")
    measures = {
        "the leak rate": column(esns, "leak_rate"),
        "the lead": -column(esns, "lead_median_deg"),
        "(a)": column(esns, "flag_a"),
        "(b)": column(esns, "flag_b"),
        "(c)": column(esns, "flag_c"),
    }
    return {label: auc(m[members], failing[members]) for label, m in measures.items()}


def print_aucs() -> None:
    """The areas under the ROC curve for failing, over report 016's grid and the search's, with the offsets or not."""
    print("\nArea under the ROC curve for failing: | the leak rate | the lead | (a) | (b) | (c) | failing")
    for name, _, label, _ in SETS:
        esns = rows_of(name, "esns.csv")
        old = np.array([in_report_016(key_of(e)) for e in esns])
        failing = np.array([not e["passes"] for e in esns])
        without = np.array([failing_without_offsets(name)[key_of(e)] for e in esns])
        for grid, members in (("report 016's grid", old), ("the search's grid", np.ones(len(esns), dtype=bool))):
            for outcome, fails in (("without the offset starts", without), ("with them", failing)):
                values = aucs(name, members, fails)
                print(
                    f"| {label}, {grid}, {outcome} ({members.sum()}) | "
                    + " | ".join(f"{v:.3f}" for v in values.values())
                    + f" | {fails[members].sum()} |"
                )
        offsets_only = failing & ~without
        flag_a = column(esns, "flag_a") > 0
        print(
            f"  failing only from an offset start: {offsets_only.sum()},"
            f" flagged by (a) {flag_a[offsets_only].mean():.2f};"
            f" of the passing, {flag_a[~failing].mean():.2f}"
        )


def print_claims() -> None:
    """Step 6's and step 7's claims, as the experiment's README fixed them."""
    print("\nStep 6's claims")
    for name, _, label, _ in SETS:
        excess = [rule_row(name, "a", group_name(lr)) for lr in LEAK_RATES]
        first = [r["failing_share_flagged"] - r["failing_share_unflagged"] for r in excess]
        second = [score(name, "posture_scale_max", "following", group_name(lr)) for lr in LEAK_RATES[1:6]]
        third = [score(name, "nearer unstable fixed point", "failing", group_name(lr)) for lr in LEAK_RATES]
        lr_auc = score(name, "leak rate", "failing", "all")
        others = {m: score(name, m, "failing", "all") for m in ("smaller lead", "flag_a", "flag_b", "flag_c")}
        print(f"{label}:")
        print(f"  1. (a)'s excess: {', '.join(f'{e:+.2f}' for e in first)}; holds {all(e > 0 for e in first)}")
        print(f"  2. {', '.join(f'{v:+.2f}' for v in second)}; holds {all(v >= 0.5 for v in second)}")
        print(f"  3. {', '.join(f'{v:.2f}' for v in third)}; holds {all(v > 0.5 for v in third)}")
        print(
            f"  4. the leak rate {lr_auc:.3f}; {', '.join(f'{m} {v:.3f}' for m, v in others.items())};"
            f" holds {all(v <= lr_auc for v in others.values())}"
        )
    releases = rows_of("release", "release.csv")
    robot, own = column(releases, "arrival_s"), column(releases, "own_arrival_s")
    both = np.isfinite(robot) & np.isfinite(own)
    blocks = rows_of("release", "blocks.csv")
    near, hold = column(blocks, "near_path_unstable") > 0, column(blocks, "hold_s")
    passes, lr = column(blocks, "passes") > 0, column(blocks, "leak_rate")
    delay, nearest, lead = column(blocks, "delay_s"), column(blocks, "nearest_unstable_deg"), column(blocks, "lead_deg")
    print("Step 7's claims")
    print(f"  1. rank correlation {rank_correlation(robot[both], own[both]):+.3f} over {both.sum()} releases")
    for h in (1.0, 3.0):
        s = hold == h
        print(
            f"  2. held {h:g} s: failing with / without {(~passes[s & near]).mean():.3f} /"
            f" {(~passes[s & ~near]).mean():.3f}"
        )
        cells = [f"{v:g} {rank_correlation(delay[s & (lr == v)], nearest[s & (lr == v)]):+.2f}" for v in LEAK_RATES]
        leads = [f"{v:g} {rank_correlation(delay[s & (lr == v)], lead[s & (lr == v)]):+.2f}" for v in LEAK_RATES]
        print(
            f"  3. held {h:g} s: the delay against the distance to the nearest unstable fixed point, all"
            f" {rank_correlation(delay[s], nearest[s]):+.2f}; within {', '.join(cells)}"
        )
        print(f"     against the lead, all {rank_correlation(delay[s], lead[s]):+.2f}; within {', '.join(leads)}")


def print_release() -> None:
    """Step 7's releases and blocks."""
    rows = rows_of("release", "release.csv")
    robot, own = column(rows, "arrival_s"), column(rows, "own_arrival_s")
    both = np.isfinite(robot) & np.isfinite(own)
    gap = robot[both] - own[both]
    print(
        f"\nReleases: {len(rows)} in {len({key_of(r) for r in rows})} ESNs; both arrive {both.sum()}, neither"
        f" {(~np.isfinite(robot) & ~np.isfinite(own)).sum()},"
        f" only the robot never {(~np.isfinite(robot) & np.isfinite(own)).sum()},"
        f" only the ESN never {(np.isfinite(robot) & ~np.isfinite(own)).sum()};"
        f" robot less own: median {np.median(gap):+.2f} s,"
        f" 10th to 90th {np.percentile(gap, 10):+.2f} to {np.percentile(gap, 90):+.2f} s, within 0.25 s"
        f" {(np.abs(gap) <= 0.25).sum()};"
        f" later than {LATE_S} s: robot {(robot > LATE_S).sum()}, own {(own > LATE_S).sum()}"
    )
    never = [r for r in rows if not np.isfinite(r["arrival_s"])]
    keys = {key_of(r) for r in never}
    stays = column(never, "stays_s")
    against = by_key(rows_of("against", "esns.csv"))
    qualifying = sum(against[k]["qualifying"] for k in keys)
    print(
        f"never on the robot: {len(never)} in {len(keys)} ESNs ({qualifying} qualifying), by leak rate "
        f"{', '.join(f'{lr:g} {sum(k[0] == lr for k in keys)}' for lr in LEAK_RATES)}; leaving within 5 s"
        f" {(stays < 5).sum()}, staying 60 s {np.isinf(stays).sum()}; progress median"
        f" {np.median(column(never, 'progress')):.2f}; growth median {np.median(column(never, 'growth_per_s')):.2f} /s"
        f" (the others {np.median(column([r for r in rows if np.isfinite(r['arrival_s'])], 'growth_per_s')):.2f});"
        f" with an attracting fixed point off the target {sum(against[k]['attracting_off_target'] > 0 for k in keys)}"
    )
    lr = column(rows, "leak_rate")
    for label, members in (("up to 0.1", lr <= 0.1), ("0.15 to 0.5", lr > 0.1)):
        s = both & members
        gap = robot[s] - own[s]
        print(
            f"  leak rates {label}: {members.sum()} releases, both arrive {s.sum()}: rank correlation"
            f" {rank_correlation(robot[s], own[s]):+.3f}, within 0.25 s {(np.abs(gap) <= 0.25).mean():.2f}, median"
            f" {np.median(gap):+.2f} s, 90th percentile {np.percentile(gap, 90):+.2f} s; only one of them arrives"
            f" {(np.isfinite(robot) != np.isfinite(own))[members].sum()}; the robot stays within 1° for 60 s"
            f" {np.isinf(column(rows, 'stays_s'))[members].sum()}"
        )
    print(f"  never on the robot, qualifying: {sorted(k for k in keys if against[k]['qualifying'])}")
    old_never = {
        key_of(r)
        for r in read_csv(earlier_runs("016-screening-esns")["release"] / "release.csv")
        if not np.isfinite(r["arrival_s"])
    }
    print(
        f"  report 016's {len(old_never)} ESNs never arriving after a release:"
        f" {sum(against[k]['passes'] for k in old_never)} pass here,"
        f" {sum(against[k]['offset_failures'] > 0 for k in old_never)} fail from an offset start"
    )
    passing = [e for e in rows_of("against", "esns.csv") if e["passes"]]
    near = [e for e in passing if e["nearest_unstable_path_distance_deg"] <= 30.0]
    print(f"passing {len(passing)}, with an unstable fixed point within 30° of the path {len(near)}")
    blocks = rows_of("release", "blocks.csv")
    near_b, hold = column(blocks, "near_path_unstable") > 0, column(blocks, "hold_s")
    passes, onset, delay = column(blocks, "passes") > 0, column(blocks, "onset_s"), column(blocks, "delay_s")
    arrived = column(blocks, "arrived") > 0
    print(
        f"blocks {len(blocks)}, failing {(~passes).sum()}, of which arriving {(~passes & arrived).sum()};"
        " by onset and hold:"
    )
    for h in (1.0, 3.0):
        print(
            f"  held {h:g} s: with / without {(~passes[(hold == h) & near_b]).mean():.3f} /"
            f" {(~passes[(hold == h) & ~near_b]).mean():.3f}; by onset "
            + ", ".join(f"{o:g} {(~passes[(hold == h) & (onset == o)]).mean():.3f}" for o in np.unique(onset))
        )
        lr = column(blocks, "leak_rate")
        print(
            "  within each leak rate, with / without (ESNs): "
            + "; ".join(
                f"{v:g} {(~passes[(hold == h) & (lr == v) & near_b]).mean():.2f} /"
                f" {(~passes[(hold == h) & (lr == v) & ~near_b]).mean():.2f}"
                f" ({((hold == h) & (lr == v) & near_b).sum() // 7} / {((hold == h) & (lr == v) & ~near_b).sum() // 7})"
                for v in LEAK_RATES
            )
        )
        d = delay[(hold == h) & np.isfinite(delay)]
        print(
            f"  delay held {h:g} s: median {np.median(d):+.2f} s"
            f" (10th {np.percentile(d, 10):+.2f}, 90th {np.percentile(d, 90):+.2f})"
        )


def print_shown() -> None:
    """The shown runs: arrival, whether they hold, and the peak torque over the run or the block."""
    print("\nThe shown runs: arrival (peak torque over the effort window)")
    for esn in ("esn1", "esn2"):
        for scenario, title in SHOWN:
            cells = []
            for arm in SERIES_COLORS:
                (row,) = [r for r in read_csv(run_dir(f"{scenario}_{esn}") / "metrics.csv") if r["arm"] == arm]
                held = "" if row["success"] else ", leaves the goal"
                cells.append(f"{arm} {row['arrival_time_s']:.2f} s{held} ({row['peak_torque_nm']:.1f} N m)")
            print(f"  {esn}, {title.splitlines()[0]}: " + "; ".join(cells))


# ---------------------------------------------------------------------------- the checks


def differing_columns(new: list[dict[str, Any]], old: list[dict[str, Any]]) -> dict[str, float]:
    """The columns both tables have whose values differ, with the largest difference (or how many rows differ)."""
    if len(new) != len(old):
        msg = f"{len(new)} rows against {len(old)}"
        raise ValueError(msg)
    differing = {}
    for name in [c for c in new[0] if c in old[0]]:
        pairs = [(a[name], b[name]) for a, b in zip(new, old, strict=True) if str(a[name]) != str(b[name])]
        if not pairs:
            continue
        numeric = all(isinstance(a, float) and isinstance(b, float) for a, b in pairs)
        differing[name] = max(abs(a - b) for a, b in pairs) if numeric else float(len(pairs))
    return differing


def print_checks() -> None:
    print("\nChecks against report 015's runs (columns that differ, the largest difference):")
    old = earlier_runs("015-what-the-esns-learned")
    files = [
        ("global_starts", "starts_esn1.csv"),
        ("global_starts", "starts_esn2.csv"),
        ("warmup_esn1", "warmup.csv"),
        ("warmup_esn2", "warmup.csv"),
        ("pulses", "metrics.csv"),
        ("fixed_points", "fixed_points_esn1.csv"),
        ("fixed_points", "fixed_points_esn2.csv"),
        ("fixed_points", "starts_esn1.csv"),
        ("fixed_points", "starts_esn2.csv"),
    ]
    for name, file in files:
        differing = differing_columns(rows_of(name, file), read_csv(old[name] / file))
        print(f"  {name}/{file}: " + (", ".join(f"{c} {d:.3g}" for c, d in differing.items()) or "identical"))

    print("Checks against report 016's runs, over its ESNs:")
    old = earlier_runs("016-screening-esns")
    counts = ("fixed_points", "attracting", "unstable", "attracting_off_target", "starts_at_target", "ring_at_target")
    measures = ("lead_median_deg", "posture_scale_max", "across_rate_median_per_s", "target_time_constant_s")
    for name, earlier in (("screen", "screen"), ("screen_heldout", "screen_heldout")):
        new = by_key(rows_of(name, "screen.csv"))
        earlier_rows = read_csv(old[earlier] / "screen.csv")
        changed = [r for r in earlier_rows if any(new[key_of(r)][c] != r[c] for c in counts)]
        largest = {
            m: max(
                abs(new[key_of(r)][m] - r[m]) / abs(r[m])
                for r in earlier_rows
                if np.isfinite(r[m]) and r[m] != 0 and r not in changed
            )
            for m in measures
        }
        print(
            f"  {name}: {len(earlier_rows)} ESNs, {len(changed)} with another count of fixed points or of starts at"
            f" the target ({', '.join(f'{k[1]:g}' for k in sorted(key_of(r) for r in changed))} spectral radius);"
            " the others' measures, largest relative difference "
            + ", ".join(f"{m} {d:.1e}" for m, d in largest.items())
        )
    for name, earlier in (("against", "against"), ("against_heldout", "against_heldout")):
        esns = rows_of(name, "esns.csv")
        old_esns = by_key(read_csv(old[earlier] / "esns.csv"))
        members = [e for e in esns if in_report_016(key_of(e))]
        same = sum(failing_without_offsets(name)[key_of(e)] == (not old_esns[key_of(e)]["passes"]) for e in members)
        print(f"  {name}: report 016's outcome, without the offset starts, the same for {same} of {len(members)} ESNs")
    releases = rows_of("release", "release.csv")
    pairs = [(r, same_release(r, releases)) for r in read_csv(old["release"] / "release.csv")]
    pairs = [(o, n) for o, n in pairs if n is not None]
    differing = [
        (o, n)
        for o, n in pairs
        if any(str(o[c]) != str(n[c]) for c in ("arrival_s", "own_arrival_s", "ends_at_target"))
    ]
    print(
        f"  releases: {len(pairs)} of report 016's are released here too, {len(pairs) - len(differing)} with the same"
        " arrivals; "
        + "; ".join(
            f"{key_of(o)}: robot {n['arrival_s']:.2f} s (016: {o['arrival_s']:.2f}), own {n['own_arrival_s']:.2f} s"
            f" ({o['own_arrival_s']:.2f})"
            for o, n in differing
        )
    )
    blocks = {block_key(b): b for b in rows_of("release", "blocks.csv")}
    old_blocks = [b for b in read_csv(old["release"] / "blocks.csv") if block_key(b) in blocks]
    same = sum(all(str(blocks[block_key(b)][c]) == str(b[c]) for c in ("passes", "arrival_time_s")) for b in old_blocks)
    delay = max(abs(blocks[block_key(b)]["delay_s"] - b["delay_s"]) for b in old_blocks if np.isfinite(b["delay_s"]))
    apart = {
        c: [b for b in old_blocks if abs(blocks[block_key(b)][c] - b[c]) > 1e-6 * max(1.0, abs(b[c]))]
        for c in ("lead_deg", "nearest_unstable_deg")
    }
    print(
        f"  blocks: {len(old_blocks)} of report 016's runs are run here too, {same} with the same outcome and arrival;"
        f" the delay within {delay:.1e} s; "
        + "; ".join(
            f"{c} differs by more than rounding in {len(b)} runs of the ESNs {sorted({key_of(r) for r in b})}"
            for c, b in apart.items()
        )
    )
    for scenario in ("block5", "block8", "released"):
        for esn in ("esn1", "esn2"):
            new_rows = read_csv(run_dir(f"{scenario}_{esn}") / "metrics.csv")
            old_rows = read_csv(old[f"{scenario}_{esn}"] / "metrics.csv")
            for arm in ("esn", "replay"):
                (a,) = [r for r in new_rows if r["arm"] == arm]
                (b,) = [r for r in old_rows if r["arm"] == arm]
                differing = differing_columns([a], [b])
                print(
                    f"  shown {scenario} {esn} {arm}: "
                    + (", ".join(f"{c} {d:.3g}" for c, d in differing.items()) or "identical")
                )


def same_release(row: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The release among ``rows`` of the same ESN at the same posture, to 10^-4 deg, if any."""
    for r in rows:
        if key_of(r) == key_of(row) and max(abs(r["q1_deg"] - row["q1_deg"]), abs(r["q2_deg"] - row["q2_deg"])) < 1e-4:
            return r
    return None


def block_key(row: dict[str, Any]) -> tuple[Key, float, float]:
    return (key_of(row), row["hold_s"], row["onset_s"])


if __name__ == "__main__":
    main()
