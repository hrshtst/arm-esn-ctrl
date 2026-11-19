# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 028.

    uv run python reports/028-a-reach-and-a-periodic-motion-in-one-esn/make_figures.py

It reads the copies of the runs under ``results/`` (their configurations, run records, and CSV files), the copies of
report 023's and report 027's robot runs under their own ``results/``, and the searches' runs, the robot runs' logs,
and the takes the ESNs were trained on under the storage root, which are not kept in Git. It prints the tables of the
report and draws into ``results/summary``:

- ``starts.png``: the target, the ellipse, the ten takes' hand paths, and the four sets of start postures;
- ``search.png``: the fine search's settings by their reaches that hold and their ellipse runs that cycle, and the
  same of the 100 ESNs scored again with the untaught starts;
- ``reach_paths.png``: the hand's paths with the context reach from the 45 starts, and around the target;
- ``reach_distance.png``: the hand's distance to the target over time in those runs;
- ``ellipse_paths.png``: the hand's paths with the context ellipse from the eight starts on the curve and from the 36
  untaught starts;
- ``ellipse_off_curve.png``: the hand's distance from the ellipse over time in those runs;
- ``to_reach.png``, ``to_ellipse.png``, ``back_and_forth.png``: the hand's paths and, over time, its distance to the
  target and from the ellipse when the context changes, in claim 7's, claim 8's, and claim 9's runs;
- ``<scenario>_joints.png`` and ``<scenario>_torques.png``: the joint angles or torques of the ESNs' arms over time
  against the replay's, in each scenario of ``SCENARIOS``: with the context reach from take 00's start, from the
  first take's start from which each ESN's reach does not hold, and from report 023's unseen start; with the context
  ellipse from start 0 on the curve and from report 027's untaught start; from ellipse to reach from starts 0 and 4;
  from reach to ellipse from takes 00 and 03; and back and forth from start 0.
"""

from __future__ import annotations

import csv
import functools
import tomllib
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.autonomous import Cycle, Setup, load_setups, task_cycle, task_evaluations, task_names
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.metrics import cycle_metrics, directness, distances_to_path, hold_metrics
from arm_esn_ctrl.storage import resolve_run_path
from arm_esn_ctrl.tracking import task_joint_angles

REPORT = Path(__file__).resolve().parent
REPO_ROOT = REPORT.parents[1]
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
REPORTS = REPO_ROOT / "reports"
ARM_CONFIG = REPO_ROOT / "experiments" / "demonstrations" / "reach_manual_v2_no_limits.toml"  # stage 1's robot
EXPERIMENT = "manual_demonstration_v2_reach_and_ellipse"
TRACKER = "pd_w20_z0.4"  # the one tracker's directory in a robot run
DT = 0.01  # the ESNs' period (s)
HOLD = 2.0  # the hold window after arrival (s)
DPI = 90

with ARM_CONFIG.open("rb") as _f:
    _ARM = tomllib.load(_f)
SKELETON = Skeleton.from_toml(ARM_CONFIG)
TARGET = np.array(_ARM["task"]["target"]["pos"])  # the reach's target (m)
TOLERANCE = float(_ARM["task"]["target"]["tolerance"])  # the goal's radius (m)

ESNS = {"ESN #1": "1", "ESN #2": "2"}  # the two-task ESNs, by the number in their runs' names
# The robot runs of the claims, by their configurations' names without the ESN's number: each task with its context
# alone (claim 6), from the ellipse to the reach (claim 7), from the reach to the ellipse (claim 8), and back and forth
# (claim 9).
EACH_TASK, TO_REACH, TO_ELLIPSE, BACK_AND_FORTH = (
    "robot_esn",
    "to_reach_robot_esn",
    "to_ellipse_robot_esn",
    "back_and_forth_robot_esn",
)
# The ESNs that do one task, ESN #1 and ESN #2 of these reports, whose robot runs are compared with claim 6's.
ONE_TASK = {"reach": "023-one-target-from-many-starts", "ellipse": "027-a-periodic-motion-by-hand"}
SEARCHES = {
    "coarse_search": "The coarse search",
    "fine_search": "The fine search",
    "untaught_search": "The scoring with the untaught starts",
}
NAMES = ("leak_rate", "spectral_radius", "input_scaling", "ridge")
BEST = 5  # the best settings of the coarse and the fine search printed
STILL_TURNS = 0.5  # a hand that turns less than this over a run's last five taught periods does not go around
ON_CURVE = 8  # the starts on the curve, an eighth of a loop apart; start 0 is the ellipse take's
UNSEEN_SHOWN = "unseen 1"  # the unseen start drawn over time with the reach, report 023's
# The scenarios drawn over time: the robot run, the task it starts with, the time drawn (s), and the figure's title.
# Their starts are series_starts'.
SCENARIOS: dict[str, tuple[str, str, tuple[float, float], str]] = {
    "reach": (EACH_TASK, "reach", (-1.0, 24.0), "With the context reach"),
    "ellipse": (EACH_TASK, "ellipse", (-1.0, 20.0), "With the context ellipse, the first 20 s of 60 s"),
    "to_reach": (TO_REACH, "ellipse", (-1.0, 40.0), "From ellipse to reach (claim 7)"),
    "to_ellipse": (TO_ELLIPSE, "reach", (-1.0, 60.0), "From reach to ellipse (claim 8)"),
    "back_and_forth": (BACK_AND_FORTH, "ellipse", (-1.0, 100.0), "Back and forth (claim 9)"),
}
# The stretches of the workspace drawn, x and y (m): all of it, the curve, the paths from the starts on the curve,
# and around the target.
WHOLE_VIEW = ((-0.95, 0.95), (-0.95, 0.95))
CURVE_VIEW = ((-0.34, 0.14), (0.40, 0.70))
STARTS_VIEW = ((-0.36, 0.44), (0.36, 0.86))
TARGET_VIEW = ((-0.67, -0.53), (0.33, 0.47))

TEXT_COLOR = "#52514e"
TITLE_COLOR = "#0b0b0b"
GRID_COLOR = "#e1e0d9"
SURFACE_COLOR = "#fcfcfb"
TAUGHT_COLOR = "#b9b7b0"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ESN_COLOR, REPLAY_COLOR, GOAL_COLOR = "#2a78d6", "#eb6834", "#7b3fa0"
ESN_COLORS = {"ESN #1": "#1c5cab", "ESN #2": "#6da7ec"}
FAIL_COLOR = "#eda100"  # a reach that does not hold
TURNED_COLOR = "#1baf7a"  # claim 8's takes that rest with joint 1 a whole turn from the others
# The layers of the lines over time: the ESN's arm on top, its reference (dashed) under it, the replay's below.
ARM_ZORDERS = {"esn": 3.0, "replay": 2.0}
REFERENCE_ZORDER = 2.5


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Callable[[], Figure]] = {
        "starts.png": plot_starts,
        "search.png": plot_search,
        "reach_paths.png": plot_reach_paths,
        "reach_distance.png": plot_reach_distance,
        "ellipse_paths.png": plot_ellipse_paths,
        "ellipse_off_curve.png": plot_ellipse_off_curve,
        "to_reach.png": functools.partial(plot_change, TO_REACH),
        "to_ellipse.png": functools.partial(plot_change, TO_ELLIPSE),
        "back_and_forth.png": functools.partial(plot_change, BACK_AND_FORTH),
    }
    for scenario in SCENARIOS:
        for channel in ("joints", "torques"):
            figures[f"{scenario}_{channel}.png"] = functools.partial(plot_series, scenario, channel)
    for file, plot in figures.items():
        plot().savefig(SUMMARY / file, dpi=DPI)
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
    """The name of the one run of a configuration under ``results``."""
    matches = sorted(p.name for p in results.glob(f"*-{stem}") if p.is_dir())
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


@functools.cache
def config(directory: Path) -> dict[str, Any]:
    """The configuration of a run, by its directory."""
    with (directory / "config.toml").open("rb") as f:
        return tomllib.load(f)


def config_of(stem: str) -> dict[str, Any]:
    """The configuration of a run of this report, by its configuration's name."""
    return config(RESULTS / run_name(stem))


def rows(run: str, arm: str, task: str) -> list[dict[str, Any]]:
    """One arm's runs of a robot run that start with ``task``'s context, in the order of their starts."""
    table = read_csv(RESULTS / run_name(run) / "metrics.csv")
    return [row for row in table if row["arm"] == arm and row["task"] == task]


def one_task_rows(task: str, base: str, n: str, arm: str) -> list[dict[str, Any]]:
    """One arm's runs of a robot run of ESN #``n`` of the report that does ``task`` alone, from that report's copy."""
    results = REPORTS / ONE_TASK[task] / "results"
    table = read_csv(results / run_name(f"{base}{n}", results) / "metrics.csv")
    return [row for row in table if row["arm"] == arm]


def start_of(run: str, task: str, origin: str) -> int:
    return int(next(row["start"] for row in rows(run, "esn", task) if row["origin"] == origin))


@functools.lru_cache(maxsize=8)
def log(run: str, arm: str, task: str, start: int) -> StateLog:
    """One arm's log from a robot run under the storage root, by the task the run starts with; a few are kept."""
    path = resolve_run_path(f"results/{EXPERIMENT}/{run_name(run)}")
    return StateLog.load(path / task / TRACKER / f"{arm}_{start:02d}.sklog.npz")


def duration(run: str, task: str) -> float:
    """How long a robot run that starts with ``task``'s context lasts, after the warm-up (s)."""
    evaluation = config_of(run)["evaluation"]
    return float(evaluation.get("tasks", {}).get(task, {}).get("duration", evaluation.get("duration")))


def first_task(run: str) -> str:
    """The task a robot run whose context changes starts with."""
    return str(config_of(run)["evaluation"]["start_tasks"][0])


def changes(run: str) -> list[tuple[float, str]]:
    """When the context of a robot run changes, on the task clock (s), and to which task."""
    return [(float(table["time"]), str(table["task"])) for table in config_of(run).get("switch", [])]


def esn_run(n: str) -> str:
    """The directory of the run that trained ESN #``n``, as its robot runs name it."""
    return Path(config_of(f"{EACH_TASK}{n}")["esn"]["model"]).parent.name


def scores(run: str) -> list[dict[str, Any]]:
    return read_csv(RESULTS / run_name(run) / "scores.csv")


@functools.cache
def search_runs(run: str) -> list[dict[str, Any]]:
    """A search's runs, one row a run: its ``runs.csv`` under the storage root, which is not copied under
    ``results/``. A search that does not sweep the seed does not list it; its configuration's is added."""
    table = read_csv(resolve_run_path(f"results/{EXPERIMENT}/{run_name(run)}") / "runs.csv")
    seed = config_of(run)["esn"]["seed"]
    return [row if "seed" in row else row | {"seed": seed} for row in table]


def setting_of(row: dict[str, Any]) -> tuple[float, ...]:
    return tuple(row[name] for name in NAMES)


def esn_of(row: dict[str, Any]) -> tuple[tuple[float, ...], int]:
    """A search's ESN: its setting and its seed."""
    return setting_of(row), int(row["seed"])


def chosen(n: str) -> tuple[tuple[float, ...], int]:
    """ESN #``n``'s setting and seed, from its training's configuration."""
    esn = config(RESULTS / esn_run(n))["esn"]
    return setting_of(esn), int(esn["seed"])


@functools.cache
def setups() -> dict[str, Setup]:
    """Each task's setup, by its context: its takes as the ESNs were trained on them, read from the storage root,
    and the start postures of claim 6's runs in their order."""
    trained = config(RESULTS / esn_run("1"))
    evaluation = config_of(f"{EACH_TASK}1")["evaluation"]
    loaded = load_setups({"demonstrations": trained["demonstrations"], "esn": trained["esn"], "evaluation": evaluation})
    return dict(zip(task_names(trained["demonstrations"]), loaded, strict=True))


@functools.cache
def cycle() -> Cycle:
    """What the runs with the context ellipse are measured against: the curve, its center, and the taught period."""
    names = tuple(setups())
    evaluation = task_evaluations(config_of(f"{EACH_TASK}1")["evaluation"], names)[names.index("ellipse")]
    measured = task_cycle(setups()["ellipse"], evaluation)
    assert measured is not None
    return measured


def posture(task: str, start: int) -> NDArray[np.float64]:
    """A start posture of a task's runs (deg): the same in claim 6's runs and in those whose context changes."""
    return np.degrees(setups()[task].starts[start].q)


@functools.cache
def turned() -> tuple[str, ...]:
    """The reach takes that rest at the target with joint 1 a whole turn below the others: below -180 deg."""
    return tuple(name for name, q in setups()["reach"].demos.items() if q[-1, 0] < -np.pi)


# ---------------------------------------------------------------------------- measures


def hand(q: NDArray[np.float64]) -> NDArray[np.float64]:
    return endpoint_positions(SKELETON, q)


def clock(run: str, task: str) -> NDArray[np.float64]:
    """A robot run's times on the task clock, every ESN period from 0 to its end, as its measures take them (s)."""
    return np.arange(round(duration(run, task) / DT) + 1, dtype=np.float64) * np.float64(DT)


@functools.cache
def joints(run: str, arm: str, task: str, start: int) -> NDArray[np.float64]:
    """A run's joint angles at ``clock``'s times (rad)."""
    return task_joint_angles(log(run, arm, task, start), DT, duration(run, task))


@functools.cache
def places(run: str, arm: str, task: str, start: int) -> NDArray[np.float64]:
    """A run's hand at ``clock``'s times (m)."""
    return hand(joints(run, arm, task, start))


def to_target(run: str, arm: str, task: str, start: int) -> NDArray[np.float64]:
    """A run's distance to the target at ``clock``'s times (m)."""
    return np.linalg.norm(places(run, arm, task, start) - TARGET, axis=1)


@functools.cache
def off_curve(run: str, arm: str, task: str, start: int) -> NDArray[np.float64]:
    """A run's distance from the ellipse at ``clock``'s times (m)."""
    return distances_to_path(places(run, arm, task, start), cycle().curve)


def holds(row: dict[str, Any]) -> bool:
    """A reach that arrives within the goal and stays for the 2 s after."""
    return bool(row["success"]) and row["hold_observed_s"] >= HOLD - 1e-9


def holds_after(row: dict[str, Any]) -> bool:
    """After the last change of context, to the reach: the hand arrives and stays for the 2 s after."""
    return bool(row["switch_success"]) and row["switch_hold_observed_s"] >= HOLD - 1e-9


def on_curve(row: dict[str, Any]) -> bool:
    """A run with the context ellipse from a start on the curve, not an untaught one."""
    return not str(row["origin"]).startswith("untaught")


def index_at(times: NDArray[np.float64], time: float) -> int:
    return int(np.searchsorted(times, time - 1e-9))


def excursion(run: str, start: int) -> dict[str, float]:
    """How the hand of ESN's reach that does not hold leaves the goal and comes back, from its distance to the target
    every 10 ms: when it first enters (s), when it first leaves (s), how far it goes out (m) and when (s), when it
    enters for the last time and stays (s), how many times it leaves, and its distance at the end (m)."""
    times = clock(run, "reach")
    distance = to_target(run, "esn", "reach", start)
    inside = distance <= TOLERANCE
    first = int(np.argmax(inside))
    leaves = first + int(np.argmax(~inside[first:]))
    back = int(np.flatnonzero(~inside)[-1]) + 1
    assert inside[first] and back < len(times), f"{run} from start {start}: not in the goal at the end"
    farthest = leaves + int(np.argmax(distance[leaves:back]))
    return {
        "enters": float(times[first]),
        "leaves": float(times[leaves]),
        "farthest": float(distance[farthest]),
        "farthest_at": float(times[farthest]),
        "back": float(times[back]),
        "exits": float(np.count_nonzero(inside[first:-1] & ~inside[first + 1 :])),
        "final": float(distance[-1]),
    }


def in_goal_since(distance: NDArray[np.float64], times: NDArray[np.float64], until: int) -> float:
    """When the hand entered the goal for the last time before sample ``until`` and stayed in up to it (s); NaN if it
    is outside then."""
    outside = np.flatnonzero(distance[: until + 1] > TOLERANCE)
    if outside.size == 0:
        return float(times[0])
    return float("nan") if outside[-1] == until else float(times[outside[-1] + 1])


def peaks(state: StateLog, begin: float, end: float) -> tuple[float, float, float]:
    """Over a stretch of a log: the reference's fastest joint speed (deg/s), measured as the robot runs measure it
    (its largest step in a joint over the shortest step in time), the largest joint torque (N m), and the arm's own
    fastest joint speed (deg/s)."""
    window = (state.times > begin - 1e-9) & (state.times < end + 1e-9)
    q_ref, times = state.channel("q_ref")[window], state.times[window]
    speed = np.degrees(np.abs(np.diff(q_ref, axis=0)).max() / np.diff(times).min())
    torque = np.abs(state.channel("tau")[window]).max()
    return float(speed), float(torque), float(np.degrees(np.abs(state.channel("dq")[window]).max()))


def angle_about_center(point: NDArray[np.float64]) -> float:
    """A hand's angle about the ellipse's center, from the x axis, 0 to 360 (deg)."""
    offset = point - cycle().center
    return float(np.degrees(np.arctan2(offset[1], offset[0])) % 360.0)


def column(runs: list[dict[str, Any]], key: str) -> list[float]:
    return [float(run[key]) for run in runs]


def spread(values: list[float], fmt: str, scale: float = 1.0) -> str:
    """Values as their median and range, ``median (smallest to largest)``; NaN left out and counted."""
    finite = [scale * v for v in values if not np.isnan(v)]
    if not finite:
        return "none"
    text = f"{np.median(finite):{fmt}} ({min(finite):{fmt}} to {max(finite):{fmt}})"
    return text if len(finite) == len(values) else f"{text}, {len(finite)} of {len(values)}"


def passing(runs: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]) -> dict[Any, dict[str, list[int]]]:
    """A search's runs grouped by ``key``: each task's runs that pass and its runs, as ``[passed, total]``."""
    grouped: dict[Any, dict[str, list[int]]] = {}
    for run in runs:
        tally = grouped.setdefault(key(run), {"reach": [0, 0], "ellipse": [0, 0]})[str(run["task"])]
        tally[0] += not run["failed"]
        tally[1] += 1
    return grouped


def start_set(run: dict[str, Any]) -> str:
    """The set a run's start is of: the takes', on the curve, untaught, or unseen."""
    origin = str(run["origin"])
    if origin.startswith(("untaught", "unseen")):
        return origin.split()[0]
    return "takes'" if run["task"] == "reach" else "on the curve"


# ---------------------------------------------------------------------------- figures


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def workspace(ax: Axes, view: tuple[tuple[float, float], ...], title: str = "") -> None:
    """The workspace's axes over ``view``: the ring the hand reaches, the target and its 20 mm, the ellipse, and
    equal scales."""
    style(ax)
    upper, forearm = (link.prop.length for link in SKELETON.links[1:])
    for radius in (abs(upper - forearm), upper + forearm):
        ax.add_patch(Circle((0.0, 0.0), radius, fill=False, edgecolor=GRID_COLOR, linewidth=1.2, zorder=1))
    ax.add_patch(Circle(tuple(TARGET), TOLERANCE, fill=False, edgecolor=GOAL_COLOR, linewidth=1.6, zorder=5))
    ax.plot(*TARGET, marker="+", color=GOAL_COLOR, markersize=9, zorder=5)
    curve = cycle().curve
    ax.plot(curve[:, 0], curve[:, 1], color=GOAL_COLOR, linewidth=1.4, zorder=2)
    ax.plot(0.0, 0.0, marker="s", color=TEXT_COLOR, markersize=4, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(*view[0])
    ax.set_ylim(*view[1])
    if title:
        ax.set_title(title, color=TITLE_COLOR, fontsize=10)


def draw_path(
    ax: Axes, points: NDArray[np.float64], color: str, alpha: float = 0.8, width: float = 1.0, zorder: float = 3.0
) -> None:
    """A hand path, its start as a dot."""
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=width, alpha=alpha, zorder=zorder)
    ax.plot(*points[0], marker="o", color=color, markersize=3.5, zorder=zorder + 0.5)


def xy_labels(axes: Any) -> None:
    for ax in axes[-1]:
        ax.set_xlabel("x (m)", color=TEXT_COLOR)
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)", color=TEXT_COLOR)


def mark_changes(ax: Axes, run: str, named: bool) -> None:
    """Each change of a run's context as a gray line, and with ``named`` the new task's name at its top."""
    for time, task in changes(run):
        ax.axvline(time, color=TEXT_COLOR, linewidth=1.0, zorder=1.8)  # above the grid
        if named:
            ax.annotate(
                f" to {task}",
                (time, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -3),
                textcoords="offset points",
                va="top",
                color=TITLE_COLOR,
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.1", "facecolor": SURFACE_COLOR, "edgecolor": "none", "alpha": 0.85},
                zorder=6,
            )


def distance_axes(ax: Axes, title: str, end: float, low: float) -> None:
    """The axes of a distance over time (mm): logarithmic from ``low``, with the 20 mm line."""
    style(ax)
    ax.axhline(1e3 * TOLERANCE, color=GOAL_COLOR, linewidth=1.2, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(low, 2500.0)
    ax.set_xlim(0.0, end)
    ax.set_title(title, color=TITLE_COLOR, fontsize=10)


def plot_starts() -> Figure:
    fig = Figure(figsize=(14, 7.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The target and its 20 mm, the ellipse (purple), the ten takes' hand paths (gray), and the four sets of start"
        " postures by their hands;\nright: around the ellipse, the starts on it numbered by the curve's angle, 45°"
        " apart, 0 the ellipse take's start",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 2)
    reach, ellipse = setups()["reach"], setups()["ellipse"]
    groups = {
        "takes'": [start.q for start in reach.starts if start.demonstrated],
        "unseen": [start.q for start in reach.starts if start.group == "unseen"],
        "on the curve": [start.q for start in ellipse.starts if start.group in ("demonstrated", "on_curve")],
        "untaught": [start.q for start in ellipse.starts if start.group == "untaught"],
    }
    points = {key: hand(np.array(q)) for key, q in groups.items()}
    markers: dict[str, dict[str, Any]] = {
        "takes'": {"marker": "o", "color": TITLE_COLOR, "markersize": 7},
        "on the curve": {"marker": "D", "color": TITLE_COLOR, "markersize": 5},
        "untaught": {"marker": "s", "color": REPLAY_COLOR, "markersize": 5},
        "unseen": {"marker": "^", "color": ESN_COLOR, "markerfacecolor": "none", "markersize": 6},
    }
    for ax, view in zip(axes, (WHOLE_VIEW, CURVE_VIEW), strict=True):
        workspace(ax, view)
        for path in [*reach.demo_hands.values(), *ellipse.demo_hands.values()]:
            ax.plot(path[:, 0], path[:, 1], color=TAUGHT_COLOR, linewidth=3.0, zorder=1.5)
        for key, marker in markers.items():
            ax.plot(points[key][:, 0], points[key][:, 1], linestyle="", zorder=4, **marker)
        ax.set_xlabel("x (m)", color=TEXT_COLOR)
    for index, point in enumerate(points["takes'"]):
        axes[0].annotate(
            f"{index:02d}", point, textcoords="offset points", xytext=(7, 5), color=TITLE_COLOR, fontsize=9
        )
    center = cycle().center
    for index, point in enumerate(points["on the curve"]):
        x, y = 14 * (center - point) / np.linalg.norm(center - point)  # inward, toward the center (points)
        axes[1].annotate(str(index), point, textcoords="offset points", xytext=(x, y), ha="center", va="center")
    axes[0].set_ylabel("y (m)", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=TAUGHT_COLOR, linewidth=3.0, label="a take"),
        Line2D([], [], linestyle="", **markers["takes'"], label="a reach take's start"),
        Line2D([], [], linestyle="", **markers["on the curve"], label="start on the curve"),
        Line2D([], [], linestyle="", **markers["untaught"], label="untaught start"),
        Line2D([], [], linestyle="", **markers["unseen"], label="unseen start"),
        Line2D([], [], linestyle="", marker="+", color=GOAL_COLOR, markersize=9, label="the target"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_search() -> Figure:
    """The fine search's settings, and the 100 ESNs scored again, by their reaches that hold and ellipse runs that
    cycle."""
    fig = Figure(figsize=(15, 7.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The search by each task's runs that pass; a marker's area grows with the points it stands for."
        "\nLeft: the fine search's 81 settings, five reservoirs each, ringed: the 20 scored again."
        " Right: those 20 settings' 100 ESNs, scored again with the untaught starts",
        color=TITLE_COLOR,
    )
    left, right = fig.subplots(1, 2)
    fine = scores("fine_search")
    fine_totals = next(iter(passing(search_runs("fine_search"), setting_of).values()))
    rescored = {setting_of(row) for row in scores("untaught_search")}
    ridges = sorted({row["ridge"] for row in fine})
    shapes = dict(zip(ridges, ("o", "s", "^"), strict=True))
    colors = dict(zip(ridges, (BLUES[2], BLUES[5], BLUES[9]), strict=True))
    style(left)
    for ridge in ridges:
        mine = [row for row in fine if row["ridge"] == ridge]
        spots = Counter((int(row["passing_runs_reach"]), int(row["passing_runs_ellipse"])) for row in mine)
        xs, ys = zip(*spots, strict=True)
        left.scatter(
            xs,
            ys,
            s=[30 * spots[p] for p in spots],
            marker=shapes[ridge],
            color=colors[ridge],
            alpha=0.8,
            edgecolors=TITLE_COLOR,
            linewidths=0.4,
            zorder=3,
            label=f"ridge {ridge:g}",
        )
    ringed = [row for row in fine if setting_of(row) in rescored]
    left.scatter(
        [row["passing_runs_reach"] for row in ringed],
        [row["passing_runs_ellipse"] for row in ringed],
        s=170,
        facecolors="none",
        edgecolors=TITLE_COLOR,
        linewidths=1.0,
        zorder=4,
        label="scored again",
    )
    left.set_xlabel(f"reaches that hold, of {fine_totals['reach'][1]}", color=TEXT_COLOR)
    left.set_ylabel(f"ellipse runs that cycle, of {fine_totals['ellipse'][1]}", color=TEXT_COLOR)
    left.set_title("the fine search's settings", color=TITLE_COLOR, fontsize=10)
    left.legend(frameon=False, labelcolor=TEXT_COLOR, loc="upper left")
    left.set_xlim(-1, fine_totals["reach"][1] + 1)
    left.set_ylim(-1, fine_totals["ellipse"][1] + 1)

    style(right)
    esns = passing(search_runs("untaught_search"), esn_of)
    reach_total, ellipse_total = (next(iter(esns.values()))[task][1] for task in ("reach", "ellipse"))
    picks = {label: chosen(n) for label, n in ESNS.items()}
    kin = {label: [esn for esn in esns if esn[0] == pick[0] and esn != pick] for label, pick in picks.items()}
    rest = [esn for esn in esns if all(esn[0] != pick[0] for pick in picks.values())]
    spots = Counter((esns[esn]["reach"][0], esns[esn]["ellipse"][0]) for esn in rest)
    xs, ys = zip(*spots, strict=True)
    right.scatter(xs, ys, s=[30 * spots[p] for p in spots], color=TAUGHT_COLOR, alpha=0.9, zorder=2)
    settings = len({esn[0] for esn in rest})
    for label, others in kin.items():
        right.scatter(
            [esns[esn]["reach"][0] for esn in others],
            [esns[esn]["ellipse"][0] for esn in others],
            s=60,
            facecolors="none",
            edgecolors=ESN_COLORS[label],
            linewidths=1.8,
            zorder=3,
        )
        pick = picks[label]
        point = (esns[pick]["reach"][0], esns[pick]["ellipse"][0])
        right.scatter(*point, s=140, marker="*", color=ESN_COLORS[label], edgecolors=TITLE_COLOR, zorder=4)
        right.annotate(
            f"two-task {label}:\n{setting_text(pick[0])}, seed {pick[1]}",
            point,
            xytext=(2, 37) if label == "ESN #1" else (2, 31),  # in the empty middle, away from the points (data)
            va="center",
            color=TITLE_COLOR,
            fontsize=10,
            arrowprops={"arrowstyle": "-", "color": ESN_COLORS[label], "linewidth": 1.0},
        )
    right.axvline(reach_total, color=TEXT_COLOR, linewidth=0.8, linestyle="--", zorder=1.8)
    right.axhline(ellipse_total, color=TEXT_COLOR, linewidth=0.8, linestyle="--", zorder=1.8)
    right.set_xlabel(f"reaches that hold, of {reach_total} (the takes' 9 and the untaught 36)", color=TEXT_COLOR)
    right.set_ylabel(f"ellipse runs that cycle, of {ellipse_total} (on the curve 8, untaught 36)", color=TEXT_COLOR)
    right.set_title("the ESNs scored again; dashed: all pass", color=TITLE_COLOR, fontsize=10)
    handles = [
        Line2D([], [], linestyle="", marker="o", color=TAUGHT_COLOR, label=f"an ESN of the other {settings} settings"),
        *(
            Line2D(
                [],
                [],
                linestyle="",
                marker="o",
                markerfacecolor="none",
                markeredgecolor=ESN_COLORS[label],
                markeredgewidth=1.8,
                label=f"another seed of {label}'s setting",
            )
            for label in ESNS
        ),
        *(
            Line2D([], [], linestyle="", marker="*", markersize=12, color=ESN_COLORS[label], label=f"two-task {label}")
            for label in ESNS
        ),
    ]
    right.legend(handles=handles, frameon=False, labelcolor=TEXT_COLOR, loc="lower left")
    right.set_xlim(-1, reach_total + 1)
    right.set_ylim(-1, ellipse_total + 1)
    return fig


def arms() -> list[tuple[str, str, str, str]]:
    """The arms of claim 6 drawn side by side: the name, the robot run, the arm in it, and its color."""
    esns = [(label, f"{EACH_TASK}{n}", "esn", ESN_COLORS[label]) for label, n in ESNS.items()]
    return [*esns, ("the replay", f"{EACH_TASK}1", "replay", REPLAY_COLOR)]


def plot_reach_paths() -> Figure:
    """The hand's paths with the context reach from the 45 starts: the workspace above, around the target below."""
    fig = Figure(figsize=(15, 10.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The hand's paths with the context reach, from the nine takes' starts and the 36 unseen ones (dots), over 24 s;"
        "\nyellow and thick: the reaches that do not hold; below: around the target, its 20 mm in purple",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(2, 3)
    for col, (label, run, arm, color) in enumerate(arms()):
        for row, view in enumerate((WHOLE_VIEW, TARGET_VIEW)):
            ax = axes[row, col]
            workspace(ax, view, f"{label}" if row == 0 else f"{label}, around the target")
            for run_row in rows(run, arm, "reach"):
                points = places(run, arm, "reach", int(run_row["start"]))
                if holds(run_row):
                    draw_path(ax, points, color, 0.6, 0.9)
                else:
                    draw_path(ax, points, FAIL_COLOR, 1.0, 1.8, zorder=4.0)
    xy_labels(axes)
    handles = [Line2D([], [], color=color, label=f"{label}, holds") for label, _, _, color in arms()]
    handles += [Line2D([], [], color=FAIL_COLOR, linewidth=1.8, label="does not hold")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_reach_distance() -> Figure:
    fig = Figure(figsize=(15, 5.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The hand's distance to the target with the context reach, from the 45 starts; yellow and thick: the reaches"
        " that do not hold;\npurple: the goal's radius, 20 mm",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(1, 3, sharey=True)
    for ax, (label, run, arm, color) in zip(axes, arms(), strict=True):
        distance_axes(ax, label, duration(run, "reach"), 1.0)
        times = clock(run, "reach")
        for run_row in rows(run, arm, "reach"):
            distance = 1e3 * to_target(run, arm, "reach", int(run_row["start"]))
            if holds(run_row):
                ax.plot(times, distance, color=color, linewidth=0.8, alpha=0.6, zorder=3)
            else:
                ax.plot(times, distance, color=FAIL_COLOR, linewidth=1.6, zorder=4)
        ax.set_xlabel("time (s)", color=TEXT_COLOR)
    axes[0].set_ylabel("distance to the target (mm)", color=TEXT_COLOR)
    return fig


def plot_ellipse_paths() -> Figure:
    fig = Figure(figsize=(15, 9.2), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The hand's paths with the context ellipse over 60 s, from the eight starts on the curve (above) and from the"
        " 36 untaught starts (below, dots);\npurple: the ellipse",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(2, 3, height_ratios=[0.62, 1.0])
    for col, (label, run, arm, color) in enumerate(arms()):
        for row, (view, heading) in enumerate(
            ((STARTS_VIEW, "from the starts on the curve"), (WHOLE_VIEW, "from the untaught starts"))
        ):
            ax = axes[row, col]
            workspace(ax, view, f"{label}, {heading}")
            for run_row in rows(run, arm, "ellipse"):
                if on_curve(run_row) == (row == 0):
                    draw_path(ax, places(run, arm, "ellipse", int(run_row["start"])), color, 0.5, 0.7)
    xy_labels(axes)
    return fig


def plot_ellipse_off_curve() -> Figure:
    fig = Figure(figsize=(15, 9.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The hand's distance from the ellipse with the context ellipse, the ESN's arm (blue) over the replay's"
        " (orange);\npurple: 20 mm, the curve RMS a run that cycles stays within; the axis is cut at 0.1 mm, below"
        " which the distance dips where the hand crosses the curve",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(2, len(ESNS), sharey=True, sharex=True)
    for col, (label, n) in enumerate(ESNS.items()):
        run = f"{EACH_TASK}{n}"
        times = clock(run, "ellipse")
        for row, heading in enumerate(("from the eight starts on the curve", "from the 36 untaught starts")):
            ax = axes[row, col]
            distance_axes(ax, f"{label}, {heading}", duration(run, "ellipse"), 0.1)
            for arm, color in (("esn", ESN_COLORS[label]), ("replay", REPLAY_COLOR)):
                for run_row in rows(run, arm, "ellipse"):
                    if on_curve(run_row) == (row == 0):
                        distance = 1e3 * off_curve(run, arm, "ellipse", int(run_row["start"]))
                        ax.plot(times, distance, color=color, linewidth=0.7, alpha=0.7, zorder=ARM_ZORDERS[arm])
            axes[row, 0].set_ylabel("from the ellipse (mm)", color=TEXT_COLOR)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)", color=TEXT_COLOR)
    return fig


def plot_change(base: str) -> Figure:
    """The runs whose context changes: for each ESN, the hand's paths, faint before the first change and strong after
    it, and over time its distance to the target and from the ellipse, the changes marked."""
    run1 = f"{base}{ESNS['ESN #1']}"
    task = first_task(run1)
    moments = changes(run1)
    end = duration(run1, task)
    when = ", ".join(f"to {new} at {time:g} s" for time, new in moments)
    starts = "the nine reach takes' starts" if task == "reach" else "the eight starts on the curve"
    fig = Figure(figsize=(17, 10.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        f"The context changes from {task} {when}, from {starts} (dots), over {end:g} s."
        "\nLeft: the hand's paths, faint before the first change and strong after it; diamonds: the hand at each"
        " change."
        "\nMiddle and right: the hand's distance to the target and from the ellipse; gray lines: the changes; purple:"
        " 20 mm",
        color=TITLE_COLOR,
    )
    axes = fig.subplots(len(ESNS), 3, width_ratios=[1.0, 1.15, 1.15])
    for row, (label, n) in enumerate(ESNS.items()):
        run = f"{base}{n}"
        paths, near, far = axes[row]
        workspace(paths, WHOLE_VIEW, f"{label}: the hand's paths")
        distance_axes(near, f"{label}: to the target", end, 1.0)
        distance_axes(far, f"{label}: from the ellipse", end, 0.1)
        for ax in (near, far):
            mark_changes(ax, run, named=True)
        times = clock(run, task)
        marks = [index_at(times, time) for time, _ in moments]
        for run_row in rows(run, "esn", task):
            start = int(run_row["start"])
            color = TURNED_COLOR if run_row["origin"] in turned() else ESN_COLORS[label]
            points = places(run, "esn", task, start)
            paths.plot(*points[: marks[0] + 1].T, color=color, linewidth=0.8, alpha=0.3, zorder=3)
            paths.plot(*points[marks[0] :].T, color=color, linewidth=1.1, alpha=0.85, zorder=3.2)
            paths.plot(*points[0], marker="o", color=color, markersize=3.5, zorder=3.5)
            for k in marks:
                paths.plot(*points[k], marker="D", color=color, markeredgecolor=TITLE_COLOR, markersize=5, zorder=6)
            near.plot(times, 1e3 * to_target(run, "esn", task, start), color=color, linewidth=0.8, alpha=0.8, zorder=3)
            far.plot(times, 1e3 * off_curve(run, "esn", task, start), color=color, linewidth=0.8, alpha=0.8, zorder=3)
        paths.set_ylabel("y (m)", color=TEXT_COLOR)
        near.set_ylabel("distance (mm)", color=TEXT_COLOR)
    axes[-1, 0].set_xlabel("x (m)", color=TEXT_COLOR)
    for ax in axes[-1, 1:]:
        ax.set_xlabel("time (s)", color=TEXT_COLOR)
    handles = [Line2D([], [], color=color, linewidth=1.4, label=label) for label, color in ESN_COLORS.items()]
    if task == "reach":
        names = " and ".join(name.split("_")[1] for name in turned())
        handles.append(
            Line2D([], [], color=TURNED_COLOR, linewidth=1.4, label=f"takes {names}, joint 1 a whole turn below")
        )
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def shown_untaught() -> str:
    """The untaught start drawn over time with the ellipse: the replay's largest torque from it is the median of the
    36, the lower of the two middle ones, as report 027 chooses it."""
    replay = [row for row in rows(f"{EACH_TASK}1", "replay", "ellipse") if not on_curve(row)]
    order = np.argsort(column(replay, "peak_torque_nm"))
    return str(replay[order[(len(replay) - 1) // 2]]["origin"])


def first_failing_take(label: str) -> str:
    """The first reach take's start from which an ESN's reach does not hold."""
    runs = rows(f"{EACH_TASK}{ESNS[label]}", "esn", "reach")
    return str(next(row["origin"] for row in runs if str(row["origin"]).startswith("demo") and not holds(row)))


def series_starts(scenario: str) -> list[tuple[str, tuple[str, ...], str]]:
    """The starts drawn over time in a scenario: the origin, the ESNs drawn from it, and what it is called."""
    both = tuple(ESNS)
    if scenario == "reach":
        starts = [("demo_00_filtered", both, "take 00's start")]
        for label in ESNS:
            origin = first_failing_take(label)
            starts.append((origin, (label,), f"take {origin.split('_')[1]}'s start, does not hold"))
        return [*starts, (UNSEEN_SHOWN, both, f"unseen start {UNSEEN_SHOWN.split()[1]}")]
    if scenario == "ellipse":
        untaught = shown_untaught()
        return [
            ("demo_00_filtered", both, "start 0 on the curve"),
            (untaught, both, f"untaught start {untaught.split()[1]}"),
        ]
    if scenario == "to_ellipse":
        return [("demo_00_filtered", both, "take 00's start"), ("demo_03_filtered", both, "take 03's start")]
    if scenario == "to_reach":
        return [("demo_00_filtered", both, "start 0 on the curve (0°)"), ("on_curve 3", both, "start 4 on it (180°)")]
    return [("demo_00_filtered", both, "start 0 on the curve")]


def series_columns(scenario: str) -> list[tuple[str, str, int, str]]:
    """The columns of a scenario over time: the ESN, its run, the start, and the title."""
    base, task, _, _ = SCENARIOS[scenario]
    columns = []
    for origin, labels, what in series_starts(scenario):
        for label in labels:
            run = f"{base}{ESNS[label]}"
            start = start_of(run, task, origin)
            q1, q2 = posture(task, start)
            columns.append((label, run, start, f"{label}, {what}\n({q1:.0f}°, {q2:.0f}°)"))
    return columns


def plot_series(scenario: str, channel: str) -> Figure:
    """The joint angles or torques of the ESNs' arms over time against the replay's: a column per ESN and start."""
    columns = series_columns(scenario)
    _, task, (begin, end), heading = SCENARIOS[scenario]
    fig = Figure(figsize=(19, 7.8), facecolor=SURFACE_COLOR, layout="constrained")
    what = "joint angles" if channel == "joints" else "joint torques"
    notes = ["dashed: the ESN's reference"] if channel == "joints" else []
    if changes(columns[0][1]):
        notes.append("gray lines: the context changes; the replay keeps to its take")
    text = "; ".join(notes)
    if channel == "torques":
        text += (";\n" if notes else "") + (
            "each column's axis is cut just above the largest torque of the ESN's arm after the first 50 ms"
        )
    fig.suptitle(f"{heading}: {what} of the ESNs' arms (blue) and the replay's (orange);\n{text}", color=TITLE_COLOR)
    axes = fig.subplots(2, len(columns), sharex=True)
    for col, (label, run, start, title) in enumerate(columns):
        axes[0, col].set_title(title, color=TITLE_COLOR, fontsize=10)
        logs = {arm: log(run, arm, task, start) for arm in ("esn", "replay")}
        for joint in range(2):
            ax = axes[joint, col]
            style(ax)
            mark_changes(ax, run, named=joint == 0)
            for arm, state in logs.items():
                color = ESN_COLORS[label] if arm == "esn" else REPLAY_COLOR
                shown = (state.times >= begin - 1e-9) & (state.times <= end + 1e-9)
                times = state.times[shown]
                if channel == "joints":
                    q = np.degrees(state.channel("q")[shown, joint])
                    ax.plot(times, q, color=color, linewidth=1.2, zorder=ARM_ZORDERS[arm])
                    if arm == "esn":
                        reference = np.degrees(state.channel("q_ref")[shown, joint])
                        ax.plot(times, reference, color=color, linewidth=1.0, linestyle="--", zorder=REFERENCE_ZORDER)
                else:
                    ax.plot(
                        times, state.channel("tau")[shown, joint], color=color, linewidth=1.0, zorder=ARM_ZORDERS[arm]
                    )
            if channel == "torques":
                esn = logs["esn"]
                later = (esn.times >= max(begin, 0.05)) & (esn.times <= end + 1e-9)
                top = max(1.15 * float(np.abs(esn.channel("tau")[later]).max()), 0.5)
                ax.set_ylim(-top, top)
            ax.set_xlim(begin, end)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    unit = "deg" if channel == "joints" else "N m"
    for joint in range(2):
        axes[joint, 0].set_ylabel(f"joint {joint + 1} ({unit})", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=color, linewidth=1.4, label=f"{label}'s arm") for label, color in ESN_COLORS.items()
    ]
    if channel == "joints":
        handles += [Line2D([], [], color=TEXT_COLOR, linewidth=1.0, linestyle="--", label="the ESN's reference")]
    handles += [Line2D([], [], color=REPLAY_COLOR, linewidth=1.4, label="the replay's arm")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- tables


def print_table(header: list[str], body: list[list[str]]) -> None:
    """Rows of text in columns as wide as their widest cell, the header first."""
    widths = [max(len(line[k]) for line in [header, *body]) for k in range(len(header))]
    for line in [header, *body]:
        print("  " + "  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)).rstrip())


def setting_text(setting: tuple[float, ...]) -> str:
    return "lr {:g}, sr {:g}, is {:g}, ridge {:g}".format(*setting)


def print_tables() -> None:
    print_search()
    print_esns()
    print_reach()
    print_ellipse()
    print_to_reach()
    print_to_ellipse()
    print_back_and_forth()
    print_checks()


def print_search() -> None:
    print("\n== The search")
    fine_rank = {setting_of(row): rank for rank, row in enumerate(scores("fine_search"), start=1)}
    for run, title in SEARCHES.items():
        ranked = scores(run)
        runs = search_runs(run)
        totals = passing(runs, setting_of)[setting_of(ranked[0])]
        reach_total, ellipse_total = totals["reach"][1], totals["ellipse"][1]
        print(
            f"\n{title} ({run_name(run)}): {len(ranked)} settings, {int(ranked[0]['runs'])} runs each,"
            f" {reach_total} with the context reach and {ellipse_total} with ellipse"
        )
        counts = Counter(int(row["passing_runs"]) for row in ranked)
        print("  settings by their runs that pass (runs: settings): " + ", ".join(f"{k}: {counts[k]}" for k in counts))
        scoring = run == "untaught_search"
        print_settings(ranked if scoring else ranked[:BEST], reach_total, ellipse_total, fine_rank if scoring else None)
        if run == "fine_search":
            print_fine(ranked, runs)
        print_search_runs(runs, config_of(run)["evaluation"]["tasks"]["ellipse"], by_start=scoring)
        print_search_esns(runs)
    print_chosen_seeds()


def print_settings(
    ranked: list[dict[str, Any]], reach_total: int, ellipse_total: int, fine_rank: dict[tuple[float, ...], int] | None
) -> None:
    """The settings in their order, with their runs that pass, each task's, their medians, and their ranks."""
    print("  the best settings:" if fine_rank is None else "  every setting:")
    header = ["rank", "setting", f"pass, of {int(ranked[0]['runs'])}", f"reach, of {reach_total}"]
    header += [f"ellipse, of {ellipse_total}", "median detour ratio", "median curve RMS (mm)", "ranks (detour, RMS)"]
    if fine_rank is not None:
        header += ["typical seed", "fine search's rank"]
    body = []
    for rank, row in enumerate(ranked, start=1):
        line = [str(rank), setting_text(setting_of(row)), str(int(row["passing_runs"]))]
        line += [str(int(row["passing_runs_reach"])), str(int(row["passing_runs_ellipse"]))]
        line += [f"{row['median_detour_ratio']:.3f}", f"{1e3 * row['median_curve_rms_m']:.2f}"]
        line += [f"{int(row['detour_ratio_rank'])}, {int(row['curve_rms_rank'])}"]
        if fine_rank is not None:
            line += [str(int(row["typical_seed"])), str(fine_rank[setting_of(row)])]
        body.append(line)
    print_table(header, body)


def print_fine(ranked: list[dict[str, Any]], runs: list[dict[str, Any]]) -> None:
    """The fine search's 20 best by each parameter's value, the settings around the 20th, and the seeds of the
    best."""
    count = len(scores("untaught_search"))
    best = ranked[:count]
    print(f"  the {count} best, scored again, by each parameter's value:")
    for key in NAMES:
        counts = Counter(row[key] for row in best)
        print(f"    {key}: " + ", ".join(f"{v:g}: {counts[v]}" for v in sorted({row[key] for row in ranked})))
    last = int(best[-1]["passing_runs"])
    around = [
        (rank, row)
        for rank, row in enumerate(ranked, start=1)
        if int(row["passing_runs"]) == last or rank in (count - 1, count + 1)
    ]
    print(f"  the settings around the {count}th, which passes {last}:")
    for rank, row in around:
        print(
            f"    {rank:2d}. {setting_text(setting_of(row))}: {int(row['passing_runs'])}, ranks"
            f" {int(row['detour_ratio_rank'])} and {int(row['curve_rms_rank'])}"
        )
    esns = passing(runs, esn_of)
    print(f"  each seed of the {BEST} best: reaches that hold and ellipse runs that cycle")
    for rank, row in enumerate(ranked[:BEST], start=1):
        seeds = sorted(esn for esn in esns if esn[0] == setting_of(row))
        each = ", ".join(
            f"seed {seed}: {esns[(setting, seed)]['reach'][0]} and {esns[(setting, seed)]['ellipse'][0]}"
            for setting, seed in seeds
        )
        totals = esns[seeds[0]]
        print(f"    {rank}. (of {totals['reach'][1]} and {totals['ellipse'][1]}) {each}")
    least = 6  # of the eight starts on the curve
    steady = [
        rank
        for rank, row in enumerate(ranked, start=1)
        if all(tally["ellipse"][0] >= least for esn, tally in esns.items() if esn[0] == setting_of(row))
    ]
    print(f"  the settings whose every seed cycles from at least {least} starts, by their rank: {steady}")


def reach_failure(run: dict[str, Any]) -> str:
    """How a reach that does not hold fails."""
    if not run["arrived"]:
        return "never arrives"
    return "arrives and leaves within the 2 s" if run["left_goal"] else "arrives too late for its 2 s to be seen"


def why_not(others: list[dict[str, Any]], evaluation: dict[str, Any]) -> str:
    """Why runs with the context ellipse do not cycle: under ``STILL_TURNS`` turns over the last five taught periods
    (and of these, backwards), too few turns, the pace kept but too far from the curve, or too many turns; and the
    count of the runs in none of these, which shows that they add up."""
    least, most = evaluation["cycle_turns"]
    tolerance = evaluation["curve_tolerance"]
    still = [run for run in others if run["turns"] < STILL_TURNS]
    backwards = sum(run["turns"] <= -STILL_TURNS for run in still)
    slow = [run for run in others if STILL_TURNS <= run["turns"] < least]
    off = [run for run in others if least <= run["turns"] <= most and run["curve_rms_m"] > tolerance]
    fast = [run for run in others if run["turns"] > most]
    rest = len(others) - len(still) - len(slow) - len(off) - len(fast)
    return (
        f"{len(still)} turn under {STILL_TURNS:g} turns ({backwards} of them {STILL_TURNS:g} turns or more backwards),"
        f" {len(slow)} go around too slowly, {len(off)} keep the pace but are over {1e3 * tolerance:g} mm in curve"
        f" RMS, {len(fast)} go around too fast, {rest} none of these"
    )


def print_search_runs(runs: list[dict[str, Any]], evaluation: dict[str, Any], by_start: bool) -> None:
    """A search's runs: the reaches that hold and the ellipse runs that cycle, by start set with ``by_start``, by each
    parameter's value, and why the others do not."""
    reach = [run for run in runs if run["task"] == "reach"]
    ellipse = [run for run in runs if run["task"] == "ellipse"]
    agree = all(holds(run) == (not run["failed"]) for run in reach)
    agree = agree and all(bool(run["cycles"]) == (not run["failed"]) for run in ellipse)
    print(f"  over all its runs (a run passes if it does not fail, which agrees with holds and cycles: {agree}):")
    for task, own, verb in (("reach", reach, "hold"), ("ellipse", ellipse, "cycle")):
        passed = [run for run in own if not run["failed"]]
        text = f"    {task}: {len(passed)} of {len(own)} {verb}"
        if by_start:
            groups = sorted({start_set(run) for run in own}, key=lambda name: name == "untaught")
            text += (
                " ("
                + ", ".join(
                    f"{sum(not run['failed'] for run in own if start_set(run) == group)} of"
                    f" {sum(start_set(run) == group for run in own)} from the {group} starts"
                    for group in groups
                )
                + ")"
            )
        print(text)
        for key in NAMES:
            counts = ", ".join(
                f"{v:g}: {sum(not run['failed'] for run in own if run[key] == v)}/{sum(run[key] == v for run in own)}"
                for v in sorted({run[key] for run in own})
            )
            print(f"      by {key}: {counts}")
        failed = [run for run in own if run["failed"]]
        if task == "reach":
            kinds = Counter(reach_failure(run) for run in failed)
            print(f"      of the {len(failed)} that do not hold: " + ", ".join(f"{n} {k}" for k, n in kinds.items()))
        else:
            print(f"      of the {len(failed)} that do not cycle: {why_not(failed, evaluation)}")


def print_search_esns(runs: list[dict[str, Any]]) -> None:
    """A search's ESNs, a setting and a seed each: how many pass every reach, every ellipse run, both, and the ESNs
    with the fewest runs that fail."""
    esns = passing(runs, esn_of)
    full = {
        task: {esn for esn, tally in esns.items() if tally[task][0] == tally[task][1]} for task in ("reach", "ellipse")
    }
    both = full["reach"] & full["ellipse"]
    failures = {esn: sum(t[1] - t[0] for t in tally.values()) for esn, tally in esns.items()}
    ridges = Counter(esn[0][3] for esn in both)
    print(
        f"  its {len(esns)} ESNs: {len(full['reach'])} hold every reach, {len(full['ellipse'])} cycle from every start,"
        f" {len(both)} do both (by ridge: {dict(sorted(ridges.items()))}); {sum(v == 1 for v in failures.values())}"
        " fail one run"
    )
    seen = passing([run | {"failed": not run["success"]} for run in runs if run["task"] == "reach"], esn_of)
    every = {esn for esn, tally in seen.items() if tally["reach"][0] == tally["reach"][1]}
    print(
        f"    if a reach that arrives too late for its 2 s to be seen counted as held (success alone): {len(every)}"
        f" hold every reach, {len(every & full['ellipse'])} do both"
    )
    fewest = min(failures.values())
    for esn in sorted(esn for esn, v in failures.items() if v == fewest):
        tally = esns[esn]
        print(
            f"    the fewest that fail, {fewest}: {setting_text(esn[0])}, seed {esn[1]}: reach"
            f" {tally['reach'][0]} of {tally['reach'][1]}, ellipse {tally['ellipse'][0]} of {tally['ellipse'][1]}"
        )


def print_chosen_seeds() -> None:
    """The seeds of the two best settings of the scoring, by start set, and the chosen ESNs' runs that fail there."""
    runs = search_runs("untaught_search")
    sweep = {esn_of(row): row for row in read_csv(RESULTS / run_name("untaught_search") / "sweep.csv")}
    ranked = scores("untaught_search")
    print("\nThe seeds of the scoring's two best settings: their runs that pass, the reach from the takes' 9 and the")
    print("untaught 36 starts, the ellipse from the 8 on the curve and the untaught 36; T: the typical seed")
    header = ["setting", "seed", f"pass, of {int(ranked[0]['runs'] // ranked[0]['seeds'])}", "reach: takes'"]
    header += ["untaught", "ellipse: on the curve", "untaught", "median curve RMS (mm)"]
    body = []
    for row in ranked[: len(ESNS)]:
        setting = setting_of(row)
        for seed in sorted(esn[1] for esn in sweep if esn[0] == setting):
            own = [run for run in runs if esn_of(run) == (setting, seed)]
            counts = [
                sum(not run["failed"] for run in own if run["task"] == task and start_set(run) == group)
                for task, group in (("reach", "takes'"), ("reach", "untaught"), ("ellipse", "on the curve"))
            ]
            counts.append(
                sum(not run["failed"] for run in own if run["task"] == "ellipse" and start_set(run) == "untaught")
            )
            mark = " T" if seed == int(row["typical_seed"]) else ""
            body.append(
                [setting_text(setting), f"{seed}{mark}", str(sum(not run["failed"] for run in own))]
                + [str(c) for c in counts]
                + [f"{1e3 * sweep[(setting, seed)]['median_curve_rms_m']:.3f}"]
            )
    print_table(header, body)
    print("The chosen ESNs' runs that fail in the scoring:")
    for label, n in ESNS.items():
        for run in runs:
            if esn_of(run) == chosen(n) and run["failed"]:
                if run["task"] == "reach":
                    text = (
                        f"{reach_failure(run)}, arrives at {run['arrival_time_s']:.2f} s, ends"
                        f" {1e3 * run['final_distance_m']:.1f} mm from the target"
                    )
                else:
                    text = f"turns {run['turns']:.2f}, curve RMS {1e3 * run['curve_rms_m']:.1f} mm"
                print(f"  two-task {label}, {run['task']} from {run['origin']}: {text}")


def print_esns() -> None:
    print("\n== The ESNs")
    for label, n in ESNS.items():
        run = esn_run(n)
        trained = config(RESULTS / run)
        esn = trained["esn"]
        with (RESULTS / run / "esn.toml").open("rb") as f:
            saved = tomllib.load(f)
        swept = ", ".join(f"{key} {esn[key]:g}" for key in (*NAMES, "seed"))
        fixed = ", ".join(f"{key} {esn[key]}" for key in esn if key not in (*NAMES, "seed"))
        extra = ", ".join(f"{key} {saved['esn'][key]}" for key in saved["esn"] if key not in esn)
        print(f"two-task {label}, run {run}:")
        print(f"  swept: {swept}")
        print(f"  fixed: {fixed}; and in esn.toml: {extra}")
        print(f"  the contexts in the input's order: {saved['contexts']}")
        for table in trained["demonstrations"]:
            print(f"  trained with the context {table['context']} on {table['run']}: {', '.join(table['train'])}")
    for task, report in ONE_TASK.items():
        results = REPORTS / report / "results"
        for label, n in ESNS.items():
            model = config(results / run_name(f"{EACH_TASK}{n}", results))["esn"]["model"]
            esn = config(results / Path(model).parent.name)["esn"]
            print(
                f"report {report[:3]}'s {label}, which does the {task} alone: "
                + ", ".join(f"{key} {esn[key]:g}" for key in (*NAMES, "seed"))
            )
    measured = cycle()
    first, last = measured.loop
    print(
        f"the ellipse take: taught period {measured.period:.4f} s, five taught periods {5 * measured.period:.2f} s;"
        f" the replay's loop from {DT * first:.2f} s to {DT * last:.2f} s"
    )
    rests = {name.split("_")[1]: np.degrees(q[-1, 0]) for name, q in setups()["reach"].demos.items()}
    print("the reach takes' joint 1 at their rest (deg): " + ", ".join(f"{k} {v:.0f}" for k, v in rests.items()))
    q1 = np.degrees(setups()["ellipse"].demos["demo_00_filtered"][:, 0])
    print(f"the ellipse take's joint 1: {q1.min():.0f} to {q1.max():.0f} deg")
    tracker = config_of(f"{EACH_TASK}1")["tracker"]
    print(
        f"the tracker: {tracker['laws']}, omega {tracker['omegas']} rad/s, damping ratio {tracker['dampings']},"
        f" reference velocity {tracker['reference_velocity']}, control period {tracker['control_period']} s"
    )


def print_reach() -> None:
    print(
        "\n== Claim 6, the reach: with the context reach for 24 s from the nine takes' starts and the 36 unseen ones;"
        "\nbeside report 023's ESNs, which do the reach alone, from the same starts; the replay is the same in both"
    )
    groups = [
        ("two-task ESN #1", rows(f"{EACH_TASK}1", "esn", "reach")),
        ("two-task ESN #2", rows(f"{EACH_TASK}2", "esn", "reach")),
        ("the replay", rows(f"{EACH_TASK}1", "replay", "reach")),
        *((f"023's {label}", one_task_rows("reach", EACH_TASK, n, "esn")) for label, n in ESNS.items()),
    ]
    header = ["", "starts", "hold", "end within 20 mm", "arrival time (s)", "detour ratio"]
    header += ["fastest reference joint speed (deg/s)", "largest torque (N m)"]
    body = []
    for label, runs in groups:
        for which, prefix in (("takes', 9", "demo"), ("unseen, 36", "unseen"), ("all, 45", "")):
            part = [row for row in runs if str(row["origin"]).startswith(prefix)]
            body.append(
                [
                    label,
                    which,
                    str(sum(map(holds, part))),
                    str(sum(row["final_distance_m"] <= TOLERANCE for row in part)),
                    spread(column(part, "arrival_time_s"), ".2f"),
                    spread(column(part, "detour_ratio"), ".3f"),
                    spread(column(part, "peak_reference_speed_dps"), ".0f"),
                    spread(column(part, "peak_torque_nm"), ".1f"),
                ]
            )
    print_table(header, body)
    for label, runs in groups:
        print(f"  {label}: every reach ends within {1e3 * max(column(runs, 'final_distance_m')):.1f} mm of the target")
    print("\nThe reaches that do not hold, from the hand's distance to the target every 10 ms (times in s, from 0):")
    header = ["", "start", "posture (deg)", "first in", "metrics' arrival", "first out", "farthest (mm)", "at"]
    header += ["in to stay", "times out", "at the end (mm)"]
    body = []
    farthest = []
    for label, n in ESNS.items():
        run = f"{EACH_TASK}{n}"
        for row in rows(run, "esn", "reach"):
            if holds(row):
                continue
            start = int(row["start"])
            out = excursion(run, start)
            farthest.append(out["farthest"])
            q1, q2 = posture("reach", start)
            body.append(
                [
                    f"two-task {label}",
                    str(row["origin"]),
                    f"({q1:.0f}, {q2:.0f})",
                    f"{out['enters']:.2f}",
                    f"{row['arrival_time_s']:.2f}",
                    f"{out['leaves']:.2f}",
                    f"{1e3 * out['farthest']:.1f}",
                    f"{out['farthest_at']:.2f}",
                    f"{out['back']:.2f}",
                    f"{out['exits']:.0f}",
                    f"{1e3 * out['final']:.1f}",
                ]
            )
    print_table(header, body)
    print(f"  {len(body)} do not hold; they go out to {1e3 * min(farthest):.1f} to {1e3 * max(farthest):.1f} mm")


def print_ellipse() -> None:
    print(
        "\n== Claim 6, the ellipse: with the context ellipse for 60 s from the eight starts on the curve and the 36"
        " untaught ones;\nbeside report 027's ESNs, which do the ellipse alone: its runs from the starts on the curve"
        " last 120 s, those from the untaught 60 s;\nthe cycle measures are over each run's last five taught periods,"
        " the joining time from the run's start"
    )
    groups = []
    for label, n in ESNS.items():
        own = rows(f"{EACH_TASK}{n}", "esn", "ellipse")
        groups.append((f"two-task {label}", [r for r in own if on_curve(r)], [r for r in own if not on_curve(r)]))
    replay = rows(f"{EACH_TASK}1", "replay", "ellipse")
    groups.append(("the replay", [r for r in replay if on_curve(r)], [r for r in replay if not on_curve(r)]))
    for label, n in ESNS.items():
        curve = one_task_rows("ellipse", EACH_TASK, n, "esn")
        groups.append((f"027's {label}", curve, one_task_rows("ellipse", "untaught_robot_esn", n, "esn")))
    header = ["", "starts", "cycle", "curve RMS (mm)", "period ratio", "joining time (s)"]
    header += ["fastest reference joint speed (deg/s)", "largest torque (N m)"]
    body = []
    for label, curve, untaught in groups:
        for which, part in ((f"on the curve, {len(curve)}", curve), (f"untaught, {len(untaught)}", untaught)):
            body.append(
                [
                    label,
                    which,
                    str(sum(bool(row["cycles"]) for row in part)),
                    spread(column(part, "curve_rms_m"), ".2f", 1e3),
                    spread(column(part, "period_ratio"), ".4f"),
                    spread(column(part, "joining_time_s"), ".2f"),
                    spread(column(part, "peak_reference_speed_dps"), ".0f"),
                    spread(column(part, "peak_torque_nm"), ".1f"),
                ]
            )
    print_table(header, body)


def print_to_reach() -> None:
    (time, _), *_ = changes(f"{TO_REACH}1")
    print(
        f"\n== Claim 7: the context changes from ellipse to reach at {time:g} s, from the eight starts on the curve,"
        f" in runs of {duration(f'{TO_REACH}1', 'ellipse'):g} s;\nthe hand at the change: its angle about the"
        " ellipse's center from the x axis, its distance to the target; after it: the arrival, whether it holds, the"
        " detour ratio\n(stage 1's, from the hand at the change), the distance at the end, and over the 2 s after the"
        " change the reference's fastest joint speed and the largest torque"
    )
    header = ["", "start", "angle (deg)", "to the target (m)", "arrival after (s)", "holds", "detour ratio"]
    header += ["at the end (mm)", "speed (deg/s)", "torque (N m)"]
    body = []
    summary = []
    for label, n in ESNS.items():
        run = f"{TO_REACH}{n}"
        times = clock(run, "ellipse")
        k = index_at(times, time)
        runs = rows(run, "esn", "ellipse")
        detours = []
        for row in runs:
            start = int(row["start"])
            points = places(run, "esn", "ellipse", start)
            detour = directness(points[k:], TARGET, TOLERANCE)[0]
            detours.append(detour)
            body.append(
                [
                    f"two-task {label}",
                    f"{start} ({360 * start // ON_CURVE}°)",
                    f"{angle_about_center(points[k]):.0f}",
                    f"{np.linalg.norm(points[k] - TARGET):.3f}",
                    f"{row['switch_arrival_time_s']:.2f}",
                    str(holds_after(row)),
                    f"{detour:.3f}",
                    f"{1e3 * row['switch_final_distance_m']:.1f}",
                    f"{row['switch_peak_reference_speed_dps_1']:.0f}",
                    f"{row['switch_peak_torque_nm_1']:.1f}",
                ]
            )
        summary.append(
            f"  two-task {label}: {sum(map(holds_after, runs))} of {len(runs)} arrive and hold; arrival"
            f" {spread(column(runs, 'switch_arrival_time_s'), '.2f')} s after the change, detour ratio"
            f" {spread(detours, '.3f')}, speed {spread(column(runs, 'switch_peak_reference_speed_dps_1'), '.0f')}"
            f" deg/s, torque {spread(column(runs, 'switch_peak_torque_nm_1'), '.1f')} N m"
        )
    print_table(header, body)
    print("\n".join(summary))


def print_to_ellipse() -> None:
    (time, _), *_ = changes(f"{TO_ELLIPSE}1")
    end = duration(f"{TO_ELLIPSE}1", "reach")
    names = " and ".join(name.split("_")[1] for name in turned())
    print(
        f"\n== Claim 8: the context changes from reach to ellipse at {time:g} s, from the nine reach takes' starts, in"
        f" runs of {end:g} s;\nbefore the change: joint 1 at the change, when the hand last entered the goal and stayed"
        f" in up to the change, and whether that is {HOLD:g} s or more;\nafter it: whether the hand cycles over the"
        " run's last five taught periods, its turns and curve RMS there, the joining time from the change, over the"
        " 2 s after\nthe change the reference's fastest joint speed, the largest torque, and the arm's own fastest"
        f" joint speed, that speed over the rest of the run, and joint 1 at the end;\ntakes {names} rest with joint 1"
        " a whole turn below the others"
    )
    header = ["", "take", "joint 1 (deg)", "in the goal since (s)", "held", "cycles", "turns", "curve RMS (mm)"]
    header += ["joining (s)", "speed (deg/s)", "torque (N m)", "arm's speed (deg/s)", "to the end", "joint 1 at end"]
    body = []
    measured: dict[str, list[dict[str, float]]] = {}
    for label, n in ESNS.items():
        run = f"{TO_ELLIPSE}{n}"
        times = clock(run, "reach")
        k = index_at(times, time)
        for row in rows(run, "esn", "reach"):
            start = int(row["start"])
            since = in_goal_since(to_target(run, "esn", "reach", start), times, k)
            state = log(run, "esn", "reach", start)
            speed, torque, arm_speed = peaks(state, time, time + HOLD)
            later = peaks(state, time, end)[2]
            q1 = np.degrees(joints(run, "esn", "reach", start)[:, 0])
            one = {
                "turned": float(row["origin"] in turned()),
                "cycles": float(bool(row["switch_cycles"])),
                "joining": row["switch_joining_time_s"],
                "speed": row["switch_peak_reference_speed_dps_1"],
                "torque": row["switch_peak_torque_nm_1"],
                "arm": arm_speed,
                "later": later,
                "log_speed": speed,
                "log_torque": torque,
            }
            measured.setdefault(label, []).append(one)
            body.append(
                [
                    f"two-task {label}",
                    str(row["origin"]).split("_")[1],
                    f"{q1[k]:.0f}",
                    f"{since:.2f}",
                    str(time - since >= HOLD - 1e-9),
                    str(bool(row["switch_cycles"])),
                    f"{row['switch_turns']:.2f}",
                    f"{1e3 * row['switch_curve_rms_m']:.2f}",
                    f"{row['switch_joining_time_s']:.2f}",
                    f"{one['speed']:.0f}",
                    f"{one['torque']:.1f}",
                    f"{arm_speed:.0f}",
                    f"{later:.0f}",
                    f"{q1[-1]:.0f}",
                ]
            )
    print_table(header, body)
    for label, ones in measured.items():
        for heading, part in (
            ("all nine", ones),
            (f"takes {names}", [e for e in ones if e["turned"]]),
            ("the other seven", [e for e in ones if not e["turned"]]),
        ):
            print(
                f"  two-task {label}, {heading}: {sum(e['cycles'] for e in part):.0f} of {len(part)} cycle; joining"
                f" {spread([e['joining'] for e in part], '.2f')} s, speed {spread([e['speed'] for e in part], '.0f')}"
                f" deg/s, torque {spread([e['torque'] for e in part], '.1f')} N m, arm's speed"
                f" {spread([e['arm'] for e in part], '.0f')} deg/s in the 2 s,"
                f" {spread([e['later'] for e in part], '.0f')} deg/s to the end"
            )
        agree = all(np.isclose(e["log_speed"], e["speed"]) and np.isclose(e["log_torque"], e["torque"]) for e in ones)
        print(f"    the peaks after the change from the logs equal the metrics': {agree}")


def print_back_and_forth() -> None:
    run1 = f"{BACK_AND_FORTH}1"
    moments = changes(run1)
    end = duration(run1, "ellipse")
    measured = cycle()
    when = ", ".join(f"to {task} at {time:g} s" for time, task in moments)
    print(
        f"\n== Claim 9: back and forth, the context changing from ellipse {when}, from the eight starts on the curve,"
        f" in runs of {end:g} s;\nafter the first change: the arrival and whether the hand holds before the second"
        " (from its log); after the second: the joining time;\nover the five taught periods before the third"
        f" ({moments[2][0] - measured.periods * measured.period:.2f} s to {moments[2][0]:g} s): the turns, the curve"
        " RMS, and whether the hand cycles, from its log and as the metrics count it;\nafter the third: the arrival,"
        " whether it holds, and the distance at the end"
    )
    header = ["", "start", "arrival 1 (s)", "holds 1", "joining 2 (s)", "turns", "curve RMS (mm)", "cycles"]
    header += ["metrics' cycles", "arrival 3 (s)", "holds 3", "at the end (mm)"]
    body = []
    peak_body = []
    summary = []
    for label, n in ESNS.items():
        run = f"{BACK_AND_FORTH}{n}"
        times = clock(run, "ellipse")
        runs = rows(run, "esn", "ellipse")
        firsts = []
        befores = []
        for row in runs:
            start = int(row["start"])
            points = places(run, "esn", "ellipse", start)
            stretch = (times >= moments[0][0] - 1e-9) & (times <= moments[1][0] + 1e-9)
            first = hold_metrics(times[stretch] - moments[0][0], points[stretch], TARGET, TOLERANCE, HOLD)
            held = bool(first["success"]) and float(first["hold_observed_s"]) >= HOLD - 1e-9
            firsts.append(held)
            part = times <= moments[2][0] + 1e-9
            before = cycle_metrics(
                times[part],
                points[part],
                measured.curve,
                measured.center,
                measured.period,
                measured.tolerance,
                measured.periods,
                measured.turns,
            )
            befores.append(before)
            body.append(
                [
                    f"two-task {label}",
                    str(start),
                    f"{row['switch_arrival_time_s_1']:.2f}",
                    str(held),
                    f"{row['switch_joining_time_s_2']:.2f}",
                    f"{before['turns']:.2f}",
                    f"{1e3 * float(before['curve_rms_m']):.2f}",
                    str(before["cycles"]),
                    str(row["cycles_before_switch_3"]),
                    f"{row['switch_arrival_time_s_3']:.2f}",
                    str(holds_after(row)),
                    f"{1e3 * row['switch_final_distance_m']:.1f}",
                ]
            )
            peak_body.append(
                [
                    f"two-task {label}",
                    str(start),
                    *(
                        f"{row[f'switch_peak_reference_speed_dps_{k}']:.0f}, {row[f'switch_peak_torque_nm_{k}']:.1f}"
                        for k in range(1, len(moments) + 1)
                    ),
                ]
            )
        summary.append(
            f"  two-task {label}: arrive after the first change in"
            f" {spread(column(runs, 'switch_arrival_time_s_1'), '.2f')} s, {sum(firsts)} of {len(runs)} hold; join"
            f" after the second in {spread(column(runs, 'switch_joining_time_s_2'), '.2f')} s; before the third"
            f" {sum(bool(row['cycles_before_switch_3']) for row in runs)} of {len(runs)} cycle, curve RMS"
            f" {spread([float(b['curve_rms_m']) for b in befores], '.2f', 1e3)} mm; after it"
            f" {sum(map(holds_after, runs))} of {len(runs)} arrive and hold, arrival"
            f" {spread(column(runs, 'switch_arrival_time_s_3'), '.2f')} s"
        )
        for k in range(1, len(moments) + 1):
            summary.append(
                f"    after change {k}: speed {spread(column(runs, f'switch_peak_reference_speed_dps_{k}'), '.0f')}"
                f" deg/s, torque {spread(column(runs, f'switch_peak_torque_nm_{k}'), '.1f')} N m"
            )
    print_table(header, body)
    print("\nThe reference's fastest joint speed (deg/s) and the largest torque (N m) over the 2 s after each change:")
    print_table(
        ["", "start", *(f"after change {k} ({time:g} s)" for k, (time, _) in enumerate(moments, start=1))], peak_body
    )
    print("\n".join(summary))


def print_checks() -> None:
    print("\n== Checks")
    commits: dict[tuple[str, bool], list[str]] = {}
    for path in sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name != "summary"):
        with (path / "run.toml").open("rb") as f:
            record = tomllib.load(f)
        commits.setdefault((record["commit"][:7], record["uncommitted_changes"]), []).append(path.name)
    for (commit, dirty), runs in commits.items():
        print(
            f"commit {commit}{' WITH UNCOMMITTED CHANGES' if dirty else ', no uncommitted changes'}: {', '.join(runs)}"
        )
    bases = {
        EACH_TASK: ("reach", "ellipse"),
        TO_REACH: ("ellipse",),
        TO_ELLIPSE: ("reach",),
        BACK_AND_FORTH: ("ellipse",),
    }
    counts = {}
    for base, tasks in bases.items():
        for n in ESNS.values():
            run = f"{base}{n}"
            assert config_of(run)["esn"]["model"].endswith(f"{esn_run(n)}/esn.toml")
            counts[run] = {task: [len(rows(run, arm, task)) for arm in ("esn", "replay")] for task in tasks}
    print(f"each robot run is its ESN's; its runs of the ESN and of the replay with each first context: {counts}")
    for base, tasks in bases.items():
        same = all(same_rows(rows(f"{base}1", "replay", task), rows(f"{base}2", "replay", task)) for task in tasks)
        print(f"{base}: the replay beside ESN #1 equals the one beside ESN #2: {same}")
    print(
        "claim 6's replay with the context reach equals report 023's replay:"
        f" {same_rows(rows(f'{EACH_TASK}1', 'replay', 'reach'), one_task_rows('reach', EACH_TASK, '1', 'replay'))}"
    )
    print_same_start(BACK_AND_FORTH, "ellipse", TO_REACH, "ellipse", changes(f"{BACK_AND_FORTH}1")[1][0])
    print_same_start(TO_REACH, "ellipse", EACH_TASK, "ellipse", changes(f"{TO_REACH}1")[0][0])
    print_same_start(TO_ELLIPSE, "reach", EACH_TASK, "reach", changes(f"{TO_ELLIPSE}1")[0][0])
    print_contexts()


def same_rows(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> bool:
    """Whether two lists of runs agree in every column they share, numbers alike within rounding, NaN alike."""
    if len(first) != len(second):
        return False
    for a, b in zip(first, second, strict=True):
        for key in set(a) & set(b) - {"task"}:
            x, y = a[key], b[key]
            if isinstance(x, float) and isinstance(y, float):
                if not np.isclose(x, y, equal_nan=True):
                    return False
            elif x != y:
                return False
    return True


def print_same_start(run: str, task: str, other: str, other_task: str, until: float) -> None:
    """Whether a run's ESN arm moves as another run's from the same starts up to ``until``, by their logs' joint
    angles: the largest difference (deg) over each ESN's runs."""
    gaps = []
    for n in ESNS.values():
        for row in rows(f"{run}{n}", "esn", task):
            start = int(row["start"])
            a, b = log(f"{run}{n}", "esn", task, start), log(f"{other}{n}", "esn", other_task, start)
            assert a.extra["tracking"]["start"] == b.extra["tracking"]["start"] == row["origin"]
            shown = a.times <= until + 1e-9
            assert np.array_equal(a.times[shown], b.times[: np.count_nonzero(shown)])
            gaps.append(float(np.abs(a.channel("q")[shown] - b.channel("q")[: np.count_nonzero(shown)]).max()))
    print(
        f"{run}'s ESN runs up to {until:g} s against {other}'s from the same starts: the joint angles differ by at"
        f" most {np.degrees(max(gaps)):.3g} deg over {len(gaps)} runs"
    )


def print_contexts() -> None:
    """Whether the ESN logs' context is the first task's until each change and the new one's from it."""
    names = tuple(setups())
    right = []
    for base in (EACH_TASK, TO_REACH, TO_ELLIPSE, BACK_AND_FORTH):
        for n in ESNS.values():
            run = f"{base}{n}"
            for task in ("reach", "ellipse") if base == EACH_TASK else (first_task(run),):
                for row in rows(run, "esn", task):
                    state = log(run, "esn", task, int(row["start"]))
                    named = np.argmax(state.channel("context"), axis=1)
                    expected = np.full(len(named), names.index(task))
                    for time, new in changes(run):
                        expected[state.times >= time - 1e-9] = names.index(new)
                    right.append(bool(np.array_equal(named, expected)))
    print(
        f"the context in every ESN log is the first task's until each change and the new one's from it, the contexts"
        f" in the order {names}: {sum(right)} of {len(right)} logs"
    )


if __name__ == "__main__":
    main()
