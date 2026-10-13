# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 007.

    uv run python reports/007-ringing-with-joint-pd/make_figures.py
    uv run python reports/007-ringing-with-joint-pd/make_figures.py --logs --animations

From report 005's copy of the take and the copies of the runs under ``results/``
(configurations, run records, metrics, and sweep tables), it draws into
``results/summary``:

- ``grid.png``: the grid with joint PD, each setting's median ringing over its ten
  seeds, and its seeds that fail (Section 3.2);

and prints the tables of the report. The validation compares the arms of COMPARED:
the chosen ESN and the replay with joint PD, and the replay with computed torque;
the chosen ESN with computed torque is a note. With ``--logs``, it reads the runs'
logs, which are not kept in Git, from the runs under the storage root, and also
draws, with ``experiments/robot_esn.py``'s own functions, the joint angles and the
joint torques of the chosen ESN's arm and the replay with joint PD in the
undisturbed run, the block, the pushes, and the offset starts of OFFSETS
(``<scenario>_joints.png`` and ``<scenario>_torques.png``, Section 3.6), and how far
each compared arm goes from the taught path under the block and the pushes
(``disturbances.png``, Section 3.5). With ``--animations``, it animates the
undisturbed run, the block, the pushes, and the offset starts of OFFSETS
(``<name>.gif``), the arms of PANELS side by side, each rendered by skelarm's
player. With ``--videos``, it exports every arm (the chosen ESN's and the replay's)
with each tracker in those five runs as an MP4 for slides, with the player's side
panel, into VIDEOS under the storage root, since they are not kept in Git.

    uv run python reports/007-ringing-with-joint-pd/make_figures.py --videos

The design of the pushes is reproduced by ``design_pushes.py`` (Section 3.1).
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import re
import subprocess
import sys
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.autonomous import Setup, load_setup
from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.metrics import distances_to_path
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path, storage_root
from arm_esn_ctrl.tracking import TrackerConfig, error_dynamics, gain_scale, tracking_gains

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
# The take, as report 005 copied it: the runs here read it from the storage root, as there.
TAKE_RUN = REPORT.parent / "005-tuned-on-the-robot" / "data" / "20261008-170034-reach_manual_v2"
DT = 0.01  # the ESNs' period (s)
GOAL = 0.02  # the goal's radius (m): a run ending farther from the target ends away from it
# The tracker of every robot run: its settings' directories in a run, and their names.
SETTINGS = {"computed_torque_w20_z0.1": "computed torque", "pd_w20_z0.1": "joint PD"}
# The ESN validated on the robot: its configuration tag, and its name.
CHOSEN = "lr0.02_sr1.35_is2_seed7"
CHOSEN_NAME = "chosen ESN (leak rate 0.02, seed 7)"
# The grid: its leak rates (one sweep each per scenario) and scenarios.
LEAKS = (0.02, 0.03, 0.04, 0.05, 0.07, 0.1)
SCENARIOS = ("nominal", "block", "pushes")
# After each disturbance (the block's release, then the three pushes), a hand not back on the path before the next
# one or the end of the 24 s run counts as back then (s after its end).
RETURN_CAPS = (24.0 - 5.5, 7.0 - 3.5, 15.0 - 7.5, 24.0 - 15.5)
# Report 006's chosen ESN: leak rate, spectral radius, input scaling, and seed.
REFERENCE = (0.03, 1.35, 1.5, 8)
# The shortlist tried in the apps: its letter, and its configuration tag; B was chosen.
SHORTLIST = {"A": "lr0.02_sr0.99_is1.5_seed8", "B": "lr0.02_sr1.35_is2_seed7", "C": "lr0.02_sr1.35_is1.5_seed6"}
# The offset starts (deg) shown: the largest offset, and the one from which report 005's ESN failed.
OFFSETS = [(15.0, 15.0), (0.0, -10.0)]
BACK_WITHIN = 5.0  # mm: an arm is back on the taught path once it stays within this of it

ESN_COLOR = "#2a78d6"
REPLAY_COLOR = "#eb6834"
REPLAY_CT_COLOR = "#9a5b13"  # the replay with computed torque
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"

# The arms of the chosen ESN's runs, by name.
ARMS = {"esn": CHOSEN_NAME, "replay": "replay"}
PD = "pd_w20_z0.1"  # the tracker of the comparisons; the figures of the chosen ESN's runs show it alone
# The arms compared: the arm, its tracker setting, its name, and its color. The chosen ESN with computed torque is
# a note (NOTED).
COMPARED = [
    ("esn", PD, "chosen ESN, joint PD", ESN_COLOR),
    ("replay", PD, "replay, joint PD", REPLAY_COLOR),
    ("replay", "computed_torque_w20_z0.1", "replay, computed torque", REPLAY_CT_COLOR),
]
NOTED = ("esn", "computed_torque_w20_z0.1", "chosen ESN, computed torque")

# The panels of every animation: the label (the arm's reference, then its tracker), the tracker setting, the arm,
# and the label's color.
PANELS = [
    (f"{CHOSEN_NAME}\ntracker: joint PD", PD, "esn", ESN_COLOR),
    ("replay of the take by time\ntracker: joint PD", PD, "replay", REPLAY_COLOR),
    ("replay of the take by time\ntracker: computed torque", "computed_torque_w20_z0.1", "replay", REPLAY_CT_COLOR),
]
# The animations: name, scenario, the start offset (deg) from the demonstrated start, and the title.
ANIMATIONS = [
    ("nominal", "nominal", (0.0, 0.0), "undisturbed, from the demonstrated start"),
    ("block", "block", (0.0, 0.0), "the tip held from 4.5 s to 5.5 s"),
    ("pushes", "pushes", (0.0, 0.0), "three pushes, at 3 s, 7 s, and 15 s"),
    (
        "offset",
        "offsets",
        OFFSETS[0],
        "start offset by +15 deg in both joints (turned counterclockwise, the elbow more bent)",
    ),
    ("offset_straight", "offsets", OFFSETS[1], "start offset by -10 deg in joint 2 (the elbow straighter)"),
]
ANIMATION_SPAN = (-0.2, 19.0)
ANIMATION_FPS = 10.0
ANIMATION_HOLD_MS = 1500  # the last frame stays this long before the GIF loops

# The videos for slides: each animation's run, with every arm and tracker, named by these.
VIDEOS = storage_root() / "reports" / REPORT.name / "videos"
VIDEO_FPS = 30.0
VIDEO_ARMS = {"esn": "chosen_esn", "replay": "replay"}
VIDEO_TRACKERS = {"pd_w20_z0.1": "joint_pd", "computed_torque_w20_z0.1": "computed_torque"}
VIDEOS_README = """# Videos of report 007's runs, for slides

These MP4s show the runs behind the animations of report 007, *Less ringing with
joint PD* (`reports/007-ringing-with-joint-pd/` in the arm-esn-ctrl repository), one
arm per video. They are not kept in Git. The take's
videos are report 005's, in `reports/005-tuned-on-the-robot/videos/` here.

All are H.264 at {fps:g} fps, 1104 x 800, rendered by skelarm's player with its side
panel: the time, the joint angles, the tip's position, and the tip's speed. The
purple dot is the target. The videos have no labels: their names say what they
show.

`<scenario>_<arm>_<tracker>.mp4`, {count} in all:

| Name | Scenario |
| --- | --- |
{scenarios}

- **Arms:** `chosen_esn`, driven by the chosen ESN (leak rate 0.02, spectral radius
  1.35, input scaling 2, seed 7) from the arm's measured posture; `replay`,
  tracking the take replayed by time.
- **Trackers:** `joint_pd` and `computed_torque`, at omega = 20 rad/s with damping
  ratio 0.1, given a zero reference velocity.
- **Span:** the whole run, from the warm-up at -1 s to the end at 24 s. The side
  panel also shows the external force while a disturbance acts, and the red arrow
  is the force at the tip.

[`videos.csv`](videos.csv) lists each video's scenario, arm, tracker, and the log it
shows, relative to the storage root.

## Reproducing them

From the repository, at commit `{commit}`:

    uv run python reports/007-ringing-with-joint-pd/make_figures.py --videos

It reads the logs under the storage root and overwrites the files here, this
README included, in about a minute. The runs are deterministic, so the videos show
the same motion.

Exported on {day} at commit `{commit}`{dirty}.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also analyze the run logs in the storage root")
    parser.add_argument("--animations", action="store_true", help="also export GIFs from the logs in the storage root")
    parser.add_argument("--videos", action="store_true", help="also export MP4s for slides into the storage root")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {"grid.png": plot_grid()}
    if args.logs:
        for scenario in ("nominal", "block", "pushes"):
            figures[f"{scenario}_joints.png"], figures[f"{scenario}_torques.png"] = plot_runs(scenario)
        for name, scenario, offset, _ in ANIMATIONS:
            if scenario == "offsets":
                figures[f"{name}_joints.png"], figures[f"{name}_torques.png"] = plot_runs(scenario, offset)
        figures["disturbances.png"] = plot_disturbances()
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=150)
    print_tables()
    if args.logs:
        print_block()
    if args.animations:
        for name, scenario, offset, title in ANIMATIONS:
            export_animation(name, scenario, offset, title)
    if args.videos:
        export_videos()
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


def stem_of(scenario: str) -> str:
    """The configuration stem of a validation run of the chosen ESN."""
    return f"{scenario}_robot_{CHOSEN}_filtered"


def offset_of(origin: str) -> tuple[float, float]:
    """The start offset (deg) in a start's origin, such as "demo_00_filtered +5,-10 deg"; (0, 0) without one."""
    match = re.search(r"([+-][\d.]+),([+-][\d.]+) deg", origin)
    return (0.0, 0.0) if match is None else (float(match[1]), float(match[2]))


def setting_of(row: dict[str, Any]) -> str:
    """The directory of a robot run's tracker setting, as robot_esn.py names it."""
    name = f"{row['law']}_w{row['omega']:g}"
    return name if row["damping"] == 1.0 else f"{name}_z{row['damping']:g}"


def run_dir(stem: str) -> Path:
    """A run's directory under the storage root, with its logs."""
    return resolve_run_path(f"results/{run_name(stem)}")


def start_of(stem: str, arm: str, offset: tuple[float, float]) -> int:
    """The index of the start offset by ``offset`` (deg) from the demonstrated start, among a run's starts."""
    return next(int(r["start"]) for r in rows_of(stem) if r["arm"] == arm and offset_of(r["origin"]) == offset)


def robot_setup(stem: str) -> tuple[Setup, dict[str, Any]]:
    """What a robot run was compared with (as robot_esn.py loads it), and its configuration."""
    with (RESULTS / run_name(stem) / "config.toml").open("rb") as f:
        config = tomllib.load(f)
    esn_path = resolve_run_path(config["esn"]["model"])
    esn = ReachingEsn.load(esn_path)
    with (esn_path.parent / "config.toml").open("rb") as f:
        demonstrations = tomllib.load(f)["demonstrations"]
    setup = load_setup(
        {"demonstrations": demonstrations, "esn": {"dt": esn.config.dt}, "evaluation": config["evaluation"]}
    )
    return setup, config


def taught_hand() -> NDArray[np.float64]:
    """The hand path of the take filtered at 2 Hz, what the ESNs were trained on, every 10 ms."""
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    q = resample_joint_angles(StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"), DT)[1]
    return endpoint_positions(skeleton, q)


# ---------------------------------------------------------------------------- figures without the logs


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def grid_run(leak: float, scenario: str) -> Path:
    """The grid's sweep run of one leak rate and scenario."""
    return RESULTS / run_name(f"sweep_robot_ringing_lr{leak:g}_{scenario}_filtered")


def failure_of(run: dict[str, Any]) -> str:
    """How a run of the grid fails, or "" if it does not."""
    if not np.isfinite(run["arrival_time_s"]):
        return "never arrives"
    if run["final_distance_m"] > GOAL:
        return "ends away"
    if run["left_goal"]:
        return "leaves the goal during the hold"
    return "hold cut short" if run["failed"] else ""


@functools.cache
def grid_seeds() -> dict[tuple[float, float, float], list[dict[str, Any]]]:
    """Each seed of every setting of the grid, by (leak rate, spectral radius, input scaling), with joint PD.

    Per seed: ``failing``, the scenarios where its run fails, so that it does not
    qualify; ``pushes_failure``, how its run under the pushes fails (see
    :func:`failure_of`); ``rmse``, the undisturbed path RMSE (mm); ``ringings``,
    ``offs``, and ``returns``, after each disturbance (the block's release, then the
    three pushes), the ringing (mm), the farthest from the path (mm), and the return
    onto it (s, a hand not back counting as back at RETURN_CAPS); ``ringing``, the
    mean ringing; ``latest``, the latest return; and ``lead``, the block's reference
    lead at the release (% of the path).
    """
    out: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for leak in LEAKS:
        runs = {
            scenario: {
                (r["spectral_radius"], r["input_scaling"], r["seed"]): r
                for r in read_csv(grid_run(leak, scenario) / "runs.csv")
            }
            for scenario in SCENARIOS
        }
        for radius, scaling, seed in sorted(runs["nominal"]):
            n, b, p = (runs[scenario][radius, scaling, seed] for scenario in SCENARIOS)
            failing = [scenario for scenario, r in zip(SCENARIOS, (n, b, p), strict=True) if failure_of(r)]
            ringings = [1000 * b["ringing_m"]] + [1000 * p[f"ringing_m_{k}"] for k in (1, 2, 3)]
            returns = [b["return_time_s"]] + [p[f"return_time_s_{k}"] for k in (1, 2, 3)]
            returns = [r if np.isfinite(r) else cap for r, cap in zip(returns, RETURN_CAPS, strict=True)]
            out.setdefault((leak, radius, scaling), []).append(
                {
                    "seed": int(seed),
                    "failing": failing,
                    "pushes_failure": failure_of(p),
                    "pushes_arrive_and_stay": bool(np.isfinite(p["arrival_time_s"]) and p["final_distance_m"] <= GOAL),
                    "rmse": 1000 * n["taught_path_rmse_m"],
                    "ringings": ringings,
                    "ringing": float(np.mean(ringings)),
                    "offs": [1000 * b["off_path_m"]] + [1000 * p[f"off_path_m_{k}"] for k in (1, 2, 3)],
                    "returns": returns,
                    "latest": max(returns),
                    "lead": 100 * b["reference_lead"],
                }
            )
    return out


def qualifies(seeds: list[dict[str, Any]], *, hold_under_pushes: bool = True) -> bool:
    """A setting whose seeds all arrive, hold, and end in the goal in the three scenarios (the grid's rule).

    Without ``hold_under_pushes``, it is enough under the pushes to arrive and end in
    the goal, as a comparison.
    """
    if hold_under_pushes:
        return not any(s["failing"] for s in seeds)
    return all(set(s["failing"]) <= {"pushes"} and s["pushes_arrive_and_stay"] for s in seeds)


def median(seeds: list[dict[str, Any]], key: str, i: int | None = None) -> float:
    """The median over seeds of a measure, or of its ``i``-th entry."""
    return float(np.median([s[key] if i is None else s[key][i] for s in seeds]))


def ranked(*, hold_under_pushes: bool = True) -> list[tuple[float, float, float]]:
    """The qualifying settings, the smallest median ringing first."""
    settings = grid_seeds()
    ok = [k for k, seeds in settings.items() if qualifies(seeds, hold_under_pushes=hold_under_pushes)]
    return sorted(ok, key=lambda k: median(settings[k], "ringing"))


def typical_seed(seeds: list[dict[str, Any]]) -> int:
    """The seed nearest the medians of the RMSE, the ringing, and the latest return, each by its interquartile range."""
    x = np.array([[s["rmse"], s["ringing"], s["latest"]] for s in seeds])
    iqr = np.subtract(*np.percentile(x, [75, 25], axis=0))
    iqr[iqr == 0] = 1.0
    return int(seeds[int(np.argmin(np.sum(np.abs(x - np.median(x, axis=0)) / iqr, axis=1)))]["seed"])


def plot_grid() -> Figure:
    """Heatmaps of each setting's median ringing over its seeds, one panel per leak rate, with its failing seeds.

    A cell's number is the median over the ten seeds of the mean ringing after the
    four disturbances (mm), with joint PD; below it, how many seeds fail, if any. The
    settings that qualify are framed.
    """
    settings = grid_seeds()
    radii = sorted({k[1] for k in settings})
    inputs = sorted({k[2] for k in settings})
    values = [median(seeds, "ringing") for seeds in settings.values()]
    norm = LogNorm(vmin=min(values), vmax=max(values))
    fig = Figure(figsize=(2.6 * len(LEAKS) + 1.2, 3.4), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The grid, joint PD: median over ten seeds of the ringing after the block and the pushes (mm);"
        " below it, the seeds that fail; framed: the settings that qualify",
        color="#0b0b0b",
    )
    axes = fig.subplots(1, len(LEAKS), sharey=True, squeeze=False)[0]
    for ax, leak in zip(axes, LEAKS, strict=True):
        grid = np.full((len(inputs), len(radii)), np.nan)
        for (lr, radius, scaling), seeds in settings.items():
            if lr != leak:
                continue
            y, x = inputs.index(scaling), radii.index(radius)
            v = median(seeds, "ringing")
            grid[y, x] = v
            failing = sum(bool(s["failing"]) for s in seeds)
            text = f"{v:.1f}" + (f"\n{failing} fail" if failing else "")
            ax.text(x, y, text, ha="center", va="center", fontsize=7, color="white" if norm(v) > 0.6 else "#0b0b0b")
            if not failing:
                ax.add_patch(Rectangle((x - 0.45, y - 0.45), 0.9, 0.9, fill=False, edgecolor="#0b0b0b", linewidth=2))
        ax.imshow(grid, cmap="Blues", norm=norm, origin="lower", aspect="auto")
        ax.set_xticks(range(len(radii)), [f"{v:g}" for v in radii], fontsize=7)
        ax.set_yticks(range(len(inputs)), [f"{v:g}" for v in inputs], fontsize=7)
        ax.set_title(f"leak rate {leak:g}", color="#0b0b0b", fontsize=9)
        ax.set_xlabel("spectral radius", color=TEXT_COLOR, fontsize=8)
    axes[0].set_ylabel("input scaling", color=TEXT_COLOR)
    fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=axes.tolist(), shrink=0.9, label="mm")
    return fig


# ---------------------------------------------------------------------------- figures from the logs


def runner(name: str) -> Any:
    """An experiment runner of ``experiments/``, such as robot_esn, loaded by its path to use its functions."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "experiments" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered first, as an import does: its dataclasses look it up
    spec.loader.exec_module(module)
    return module


def plot_runs(scenario: str, offset: tuple[float, float] = (0.0, 0.0)) -> tuple[Figure, Figure]:
    """The joint angles and the joint torques of the chosen ESN's run of a scenario, by robot_esn.py's functions.

    The run is the one from the start offset by ``offset`` (deg) from the
    demonstrated start, with joint PD; the figures show the replay with it too.
    """
    robot_esn = runner("robot_esn")
    stem = stem_of(scenario)
    setup, config = robot_setup(stem)
    settings = [setting for setting in robot_esn.tracker_settings(config["tracker"]) if setting.name == PD]
    spans = disturbance_spans(config.get("disturbance"))
    start = start_of(stem, "esn", offset)
    title = f"The chosen ESN on the robot, joint PD: {scenario}"
    directory = run_dir(stem)
    return (
        robot_esn.plot_joints(directory, setup, settings, spans, title, start),
        robot_esn.plot_torques(directory, setup, settings, spans, title, start),
    )


def path_distances(scenario: str, setting: str, arm: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The times (every 10 ms) and the hand's distance (mm) from the taught path in a run of a scenario."""
    log = StateLog.load(run_dir(stem_of(scenario)) / setting / f"{arm}_00.sklog.npz")
    task = log.times >= 0
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    hand = endpoint_positions(skeleton, log.channel("q").reshape(len(log.times), -1)[task])
    every = round(DT / float(np.median(np.diff(log.times))))
    return log.times[task][::every], 1000 * distances_to_path(hand[::every], taught_hand())


def plot_disturbances() -> Figure:
    """Each compared arm's distance from the taught path under the block and under the pushes, over time."""
    fig = Figure(figsize=(13, 7), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The block and the pushes: how far each arm goes from the taught path, and how it comes back",
        color="#0b0b0b",
    )
    axes = fig.subplots(2, 1, sharex=True)
    for ax, scenario, title in zip(
        axes, ("block", "pushes"), ("the tip held from 4.5 s to 5.5 s", "pushes at 3 s, 7 s, and 15 s"), strict=True
    ):
        style(ax)
        with (RESULTS / run_name(stem_of(scenario)) / "config.toml").open("rb") as f:
            spans = disturbance_spans(tomllib.load(f)["disturbance"])
        for onset, end in spans:
            ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
        for arm, setting, label, color in COMPARED:
            times, distance = path_distances(scenario, setting, arm)
            ax.plot(times, distance, color=color, linewidth=1.3, label=label)
        ax.axhline(BACK_WITHIN, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
        ax.set(title=title, ylabel="from the taught path (mm)", ylim=(0, None), xlim=(0, 20))
    axes[-1].set_xlabel("time (s)", color=TEXT_COLOR)
    axes[0].legend(frameon=False, labelcolor=TEXT_COLOR)
    return fig


def print_block() -> None:
    """The block, from the logs: the force at the tip, how far the reference runs ahead, and the hand's return.

    The peak force of metrics.csv is the grip's: the damper stops the moving hand.
    How hard the arm then presses is the mean force over the block's last 0.1 s,
    when its reference (the ESN's output, or the take) is the farthest ahead of the
    held hand. After the release: the farthest the hand goes from the taught path,
    and when it is back within BACK_WITHIN of it for good (counted from the release,
    up to 12 s).
    """
    print("\n== 3.4 The block, from the logs: arm and tracker: force at the tip (N) at the grip (peak, first 0.1 s)")
    print("and at the release (mean, last 0.1 s); the reference's hand ahead of the arm's at the release (mm); after")
    print("the release, the farthest from the taught path (mm), and back within 5 mm (s)")
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    stem = stem_of("block")
    with (RESULTS / run_name(stem) / "config.toml").open("rb") as f:
        block = tomllib.load(f)["disturbance"]
    onset, release = block["onset"], block["release"]
    for arm, setting, label, *_ in [*COMPARED, NOTED]:
        log = StateLog.load(run_dir(stem) / setting / f"{arm}_00.sklog.npz")
        t = log.times
        force = np.linalg.norm(log.channel("ext_force").reshape(len(t), -1), axis=1)
        grip = force[(t >= onset) & (t < onset + 0.1)].max()
        pressing = force[(t >= release - 0.1) & (t < release)].mean()
        held = int(np.searchsorted(t, release)) - 1  # the last sample held
        hand, reference = (
            endpoint_positions(skeleton, log.channel(channel).reshape(len(t), -1)[held : held + 1])
            for channel in ("q", "q_ref")
        )
        ahead = 1000 * float(np.linalg.norm(reference - hand))
        times, distance = path_distances("block", setting, arm)
        after = (times >= release) & (times < 12.0)
        back = times[np.flatnonzero(after & (distance > BACK_WITHIN))[-1] + 1] - release
        print(f"  {label}: {grip:.1f}, {pressing:.1f}; {ahead:.0f}; {distance[after].max():.0f}, {back:.1f}")


# ---------------------------------------------------------------------------- animations


def player_export(log: Path, out: Path, fps: float, *, panel: bool = False) -> None:
    """Render a run's log into ``out`` (a GIF or an MP4) with skelarm's player, without a window.

    With ``panel``, each frame has the player's side panel: the time, the joint
    angles, the tip, and the external force.
    """
    player = REPO_ROOT / "third_party" / "skelarm" / "tools" / "player.py"
    command = [sys.executable, str(player), str(log), "--export", str(out), "--fps", f"{fps:g}"]
    subprocess.run(
        command + (["--panel"] if panel else []),
        check=True,
        capture_output=True,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen"},  # render without a window
    )


def video_name(scenario: str, offset: tuple[float, float]) -> str:
    """The name of a scenario's videos: the scenario, or an offset start by its offset (deg), such as offset_+15_+15."""
    return scenario if offset == (0.0, 0.0) else f"offset_{offset[0]:+g}_{offset[1]:+g}"


def export_videos() -> None:
    """Export each animation's run, with every arm and tracker, as MP4s for slides into VIDEOS.

    Each video is one log, rendered by skelarm's player at VIDEO_FPS with its side
    panel: each arm's whole run, the warm-up included. A run's video
    is named ``<scenario>_<arm>_<tracker>.mp4`` (see :func:`video_name`), such as
    ``offset_+15_+15_replay_joint_pd.mp4``; ``videos.csv`` lists each video's
    scenario, arm, tracker, and log, and ``README.md`` (VIDEOS_README) explains
    them and how to reproduce them, with the commit they were exported at.
    """
    VIDEOS.mkdir(parents=True, exist_ok=True)
    # Each video, its scenario, arm, and tracker, and its log.
    videos = []
    for _, scenario, offset, title in ANIMATIONS:
        name = video_name(scenario, offset)
        stem = stem_of(scenario)
        for arm, arm_name in VIDEO_ARMS.items():
            start = start_of(stem, arm, offset)
            for setting, tracker in VIDEO_TRACKERS.items():
                log = run_dir(stem) / setting / f"{arm}_{start:02d}.sklog.npz"
                videos.append((VIDEOS / f"{name}_{arm_name}_{tracker}.mp4", title, ARMS[arm], SETTINGS[setting], log))
    rows = []
    for video, scenario, arm, tracker, log in videos:
        player_export(log, video, VIDEO_FPS, panel=True)
        rows.append(
            {
                "video": video.name,
                "scenario": scenario,
                "arm": arm,
                "tracker": tracker,
                "log": str(log.relative_to(storage_root())),
            }
        )
    write_csv(VIDEOS / "videos.csv", rows)
    git = [["git", "rev-parse", "--short", "HEAD"], ["git", "status", "--porcelain"]]
    commit, changes = (subprocess.run(c, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout for c in git)
    scenarios = "\n".join(
        f"| `{video_name(scenario, offset)}` | {title} |" for _, scenario, offset, title in ANIMATIONS
    )
    readme = VIDEOS_README.format(
        count=len(rows),
        scenarios=scenarios,
        fps=VIDEO_FPS,
        commit=commit.strip(),
        dirty=" (with uncommitted changes)" if changes.strip() else "",
        day=date.today().isoformat(),
    )
    (VIDEOS / "README.md").write_text(readme)
    print(f"wrote {len(rows)} videos and their README to {VIDEOS}")


def export_animation(
    name: str,
    scenario: str,
    offset: tuple[float, float],
    title: str,
) -> None:
    """Animate one start of a scenario into ``<name>.gif``: the arms of PANELS side by side, rendered by skelarm.

    Each panel is a label, a tracker setting, an arm ("esn" or "replay"), and the
    label's color. The player exports each arm's run as a GIF
    (``--export``); its frames, which the GIF merges where the arm rests, are spread
    back over time, cropped to where the arms move, labeled, and tiled into one GIF
    with the task time and the disturbance acting.
    """
    import tempfile

    from PIL import Image, ImageDraw, ImageFont, ImageSequence

    stem = stem_of(scenario)
    with (RESULTS / run_name(stem) / "config.toml").open("rb") as f:
        disturbance = tomllib.load(f).get("disturbance")
    tables = [] if disturbance is None else [disturbance] if isinstance(disturbance, dict) else disturbance
    spans = disturbance_spans(disturbance)
    frames, starts = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (_, setting, arm, _) in enumerate(PANELS):
            path = run_dir(stem) / setting / f"{arm}_{start_of(stem, arm, offset):02d}.sklog.npz"
            exported = Path(tmp) / f"{i}.gif"
            player_export(path, exported, ANIMATION_FPS)
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
    top, bottom = max(int(rows.min()) - margin, 0), int(rows.max()) + margin
    left, right = max(int(cols.min()) - margin, 0), int(cols.max()) + margin
    width, height = right - left, bottom - top
    header, label_height, gap = 44, 48, 8
    canvas_width = len(frames) * width + (len(frames) - 1) * gap

    def header_line(t: float) -> str:
        status = ""
        for table, (begin, end) in zip(tables, spans, strict=True):
            if begin <= t < end:
                status = f"   pushed ({table['force']:g} N)" if table["type"] == "push" else "   blocked"
        return f"{title}   t = {t:+.1f} s{status}"

    # One font for every frame: the largest, up to 20, in which the header fits, with a status or the last time.
    lines = [header_line(t) for t in (*[begin for begin, _ in spans], ANIMATION_SPAN[1])]
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    size = 20
    while size > 12 and max(measure.textlength(line, font=ImageFont.load_default(size=size)) for line in lines) > (
        canvas_width - 20
    ):
        size -= 1
    font, small = ImageFont.load_default(size=size), ImageFont.load_default(size=15)
    images = []
    for t in np.arange(ANIMATION_SPAN[0], ANIMATION_SPAN[1] + 1e-9, 1.0 / ANIMATION_FPS):
        canvas = Image.new("RGB", (canvas_width, header + label_height + height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), header_line(t), fill="#0b0b0b", font=font)
        for i, (label, _, _, color) in enumerate(PANELS):
            k = int(np.clip(round((t - starts[i]) * ANIMATION_FPS), 0, len(frames[i]) - 1))
            x = i * (width + gap)
            canvas.paste(Image.fromarray(frames[i][k][top:bottom, left:right]), (x, header + label_height))
            draw.text((x + 8, header + 4), label, fill=color, font=small)
        images.append(canvas)
    # One palette for every frame, learned from all of them, so that the GIF stores only what changes and brief
    # colors, such as a push's arrow, keep theirs.
    sample = Image.new("RGB", (images[0].width, len(images) * images[0].height))
    for i, image in enumerate(images):
        sample.paste(image, (0, i * images[0].height))
    palette = sample.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    images = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]
    durations = [round(1000.0 / ANIMATION_FPS)] * (len(images) - 1) + [ANIMATION_HOLD_MS]
    out = SUMMARY / f"{name}.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {out.name}: {len(images)} frames")


# ---------------------------------------------------------------------------- tables


def print_tables() -> None:
    print_grid()
    print_shortlist()
    print_validation()


def print_grid() -> None:
    settings = grid_seeds()
    print("== 3.2 The grid, joint PD: settings that qualify (of 15 per leak rate), by the rule and, as a comparison,")
    print("with arriving and ending in the goal enough under the pushes; then the failing runs of 150 per scenario")
    for leak in LEAKS:
        group = {k: v for k, v in settings.items() if k[0] == leak}
        strict = sum(qualifies(v) for v in group.values())
        relaxed = sum(qualifies(v, hold_under_pushes=False) for v in group.values())
        fails = {scenario: sum(scenario in s["failing"] for v in group.values() for s in v) for scenario in SCENARIOS}
        print(f"  leak rate {leak:g}: {strict} ({relaxed}); {fails}")

    print("\n== 3.2 How the runs under the pushes fail, per leak rate")
    for leak in LEAKS:
        kinds: dict[str, int] = {}
        for k, seeds in settings.items():
            for s in seeds:
                if k[0] == leak and s["pushes_failure"]:
                    kinds[s["pushes_failure"]] = kinds.get(s["pushes_failure"], 0) + 1
        print(f"  leak rate {leak:g}: {kinds}")

    def show(keys: list[tuple[float, float, float]]) -> None:
        for rank, k in enumerate(keys, start=1):
            seeds = settings[k]
            mark = "  <- report 006's setting" if k == REFERENCE[:3] else ""
            each = [
                " / ".join(f"{median(seeds, measure, i):.{digits}f}" for i in range(4))
                for measure, digits in (("ringings", 1), ("offs", 0), ("returns", 1))
            ]
            print(
                f"  {rank:2d}. leak rate {k[0]:g}, spectral radius {k[1]:g}, input scaling {k[2]:g}:"
                f" {median(seeds, 'ringing'):5.1f} | {median(seeds, 'rmse'):.2f} | {' | '.join(each)} |"
                f" {median(seeds, 'lead'):.1f}{mark}"
            )

    print(
        "\n== 3.2 The qualifying settings, ranked: medians over the seeds of the ringing (mm) | undisturbed path RMSE"
    )
    print(
        "(mm) | after the block's release and pushes 1, 2, 3: ringing (mm) | farthest from the path (mm) | back on it"
    )
    print("(s) | the block's reference lead (%)")
    show(ranked())
    print("\n== 3.2 As a comparison, with arriving and ending in the goal enough under the pushes")
    show(ranked(hold_under_pushes=False))
    failing = [(s["seed"], s["failing"], s["pushes_failure"]) for s in settings[REFERENCE[:3]] if s["failing"]]
    print(f"\n  report 006's setting, its failing seeds: {failing}")


def print_shortlist() -> None:
    settings = grid_seeds()
    print("\n== 3.3 The shortlist, the top three by the rule with their typical seeds, and report 006's chosen ESN:")
    print("after the block's release and pushes 1, 2, 3, the ringing (mm) and its mean | undisturbed path RMSE (mm) |")
    print("the returns (s) | the block's lead (%) | on its own, the starts that arrive and hold")
    tops = ranked()[:3]
    for letter, tag in [*SHORTLIST.items(), ("006", None)]:
        if tag is None:
            leak, radius, scaling, seed = REFERENCE
            which = "report 006's chosen ESN"
            alone = ""
        else:
            leak, radius, scaling, seed = tag_values(tag)
            seeds = settings[leak, radius, scaling]
            which = f"rank {tops.index((leak, radius, scaling)) + 1}, typical seed {typical_seed(seeds)}"
            rows = rows_of(f"grid_{tag}_filtered")
            alone = f" | {sum(r['success'] for r in rows)}/{len(rows)}"
        s = next(x for x in settings[leak, radius, scaling] if x["seed"] == seed)
        print(
            f"  {letter}: leak rate {leak:g}, spectral radius {radius:g}, input scaling {scaling:g}, seed {seed}"
            f" ({which}): {' / '.join(f'{r:.1f}' for r in s['ringings'])}, mean {s['ringing']:.1f} |"
            f" {s['rmse']:.2f} | {' / '.join(f'{r:.1f}' for r in s['returns'])} | {s['lead']:.1f}{alone}"
        )


def tag_values(tag: str) -> tuple[float, float, float, int]:
    """The leak rate, spectral radius, input scaling, and seed in a tag such as lr0.02_sr1.35_is2_seed7."""
    lr, sr, scaling, seed = re.fullmatch(r"lr([\d.]+)_sr([\d.]+)_is([\d.]+)_seed(\d+)", tag).groups()  # type: ignore[union-attr]
    return float(lr), float(sr), float(scaling), int(seed)


def print_validation() -> None:
    print("\n== 3.4 The chosen ESN's parameters (its esn.toml)")
    with (RESULTS / run_name(f"grid_{CHOSEN}_filtered") / "esn.toml").open("rb") as f:
        record = tomllib.load(f)
    print("  " + ", ".join(f"{key} {value}" for key, value in record["esn"].items()))

    def rounded(x: NDArray[np.float64], decimals: int) -> list[float]:
        return np.round(x, decimals).tolist()

    center, half = (np.degrees(record["normalization"][key]) for key in ("center_rad", "half_range_rad"))
    print(f"  normalization: center {rounded(center, 2)} deg, half range {rounded(half, 2)} deg")

    print("\n== 3.4 The trackers' gains, recomputed from omega and the damping ratio as robot_esn.py does, at the")
    print("posture where the take ends, and the natural frequency (rad/s) and damping ratio they give at its start")
    with (RESULTS / run_name(stem_of("nominal")) / "config.toml").open("rb") as f:
        tracker = tomllib.load(f)["tracker"]
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    q = resample_joint_angles(StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"), DT)[1]
    end, start = q[-1], q[0]
    print(f"  end posture {rounded(np.degrees(end), 2)} deg, start {rounded(np.degrees(start), 2)} deg")
    for law in tracker["laws"]:
        for omega in tracker["omegas"]:
            for damping in tracker["dampings"]:
                config = TrackerConfig(law, omega, tracker["acceleration_filter"], damping=damping)
                kp, kd = tracking_gains(config, skeleton, end)
                frequency, ratio = error_dynamics(law, (kp, kd), skeleton, start)
                print(
                    f"  {law}, omega {omega:g}, damping ratio {damping:g}: scale (M_ii) at the end"
                    f" {rounded(gain_scale(law, skeleton, end), 4)}, kp {rounded(kp, 4)}, kd {rounded(kd, 4)};"
                    f" at the start: scale {rounded(gain_scale(law, skeleton, start), 4)},"
                    f" omega {rounded(frequency, 2)}, damping ratio {rounded(ratio, 3)}"
                )
    filtered, velocity = tracker["acceleration_filter"], tracker["reference_velocity"]
    print(f"  acceleration filter {filtered:g} s, reference velocity: {velocity}")

    def runs_of(scenario: str) -> list[tuple[str, dict[str, Any]]]:
        """The compared arms' runs of a scenario from the demonstrated start, then the noted one, by name."""
        rows = [r for r in rows_of(stem_of(scenario)) if r["start"] == 0]
        return [
            (label, next(r for r in rows if r["arm"] == arm and setting_of(r) == setting))
            for arm, setting, label, *_ in [*COMPARED, NOTED]
        ]

    print("\n== 3.4 Validation, from the demonstrated start: arm and tracker: arrival (s), holds, path RMSE (mm)")
    for label, r in runs_of("nominal"):
        print(f"  nominal, {label}: {r['arrival_time_s']:.2f}, {r['success']}, {1000 * r['taught_path_rmse_m']:.2f}")
    print("block: also the peak force (N); at the release, the reference's lead (% of the path) and its gap from the")
    print("arm (deg); after it, the farthest from the path (mm), back within 5 mm (s), and the ringing (mm)")
    for label, r in runs_of("block"):
        print(
            f"  block, {label}: {r['arrival_time_s']:.2f}, {r['success']}, {1000 * r['taught_path_rmse_m']:.2f},"
            f" {r['peak_external_force_n']:.1f} N; {100 * r['reference_lead']:.1f}%, {r['end_gap_deg']:.1f} deg;"
            f" {1000 * r['off_path_m']:.0f} mm, {r['return_time_s']:.1f} s, {1000 * r['ringing_m']:.1f} mm"
        )
    print("pushes: also the distance from the target at the end (mm); each push: at its end, the lead (% of the path)")
    print("and the gap (deg); after it, the farthest from the path (mm), back within 5 mm (s), and the ringing (mm)")
    for label, r in runs_of("pushes"):
        each = " | ".join(
            f"{100 * r[f'reference_lead_{k}']:+.1f}%, {r[f'end_gap_deg_{k}']:.1f} deg;"
            f" {1000 * r[f'off_path_m_{k}']:.0f} mm, {r[f'return_time_s_{k}']:.2f} s,"
            f" {1000 * r[f'ringing_m_{k}']:.1f} mm"
            for k in (1, 2, 3)
        )
        print(
            f"  pushes, {label}: {r['arrival_time_s']:.2f}, {r['success']}, {1000 * r['taught_path_rmse_m']:.1f},"
            f" {1000 * r['final_distance_m']:.1f} | {each}"
        )

    print(
        "\n== 3.4 Offsets: arm and tracker: starts that hold, median path RMSE (mm), median peak torque (N m), failed"
    )
    print("starts; then the offset starts shown: arrival (s), holds, path RMSE (mm), peak torque (N m)")
    offsets = rows_of(stem_of("offsets"))
    for arm, setting, label, *_ in [*COMPARED, NOTED]:
        group = [r for r in offsets if r["arm"] == arm and setting_of(r) == setting]
        fails = [r["origin"].split()[1] for r in group if not r["success"]]
        print(
            f"  {label}: {len(group) - len(fails)}/{len(group)},"
            f" {1000 * np.median([r['taught_path_rmse_m'] for r in group]):.1f},"
            f" {np.median([r['peak_torque_nm'] for r in group]):.2f}, {fails}"
        )
        for r in group:
            if offset_of(r["origin"]) in OFFSETS:
                print(
                    f"    offset {offset_of(r['origin'])}: {r['arrival_time_s']:.2f}, {r['success']},"
                    f" {1000 * r['taught_path_rmse_m']:.1f}, {r['peak_torque_nm']:.1f}"
                )


if __name__ == "__main__":
    main()
