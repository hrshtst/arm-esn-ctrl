# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 005.

    uv run python reports/005-tuned-on-the-robot/make_figures.py
    uv run python reports/005-tuned-on-the-robot/make_figures.py --logs --animations

From the take in ``data/`` and the copies of the runs under ``results/``
(configurations, run records, metrics, and sweep tables), it draws into
``results/summary``:

- ``take.png``: the take, as recorded and filtered at 8 Hz and 2 Hz, as the ESN
  sees it, every 10 ms, with faint postures of the arm along the 2 Hz take (Section
  3.1);
- ``stage1.png``: the first sweep on the robot, the main parameters (Section 3.3);
- ``stage3.png``: the finer sweep around the best of stage 1 (Section 3.5);
- ``seeds.png``: the five best combinations, each with ten random reservoirs
  (Section 3.6);

and prints the tables of the report. With ``--logs``, it reads the runs' logs,
which are not kept in Git, from the runs under the storage root, and also draws,
with ``experiments/robot_esn.py``'s own functions, the joint angles and the joint
torques of the chosen ESN's arm and the replay in the undisturbed run, the block,
the pushes, and the offset start of OFFSET (``<scenario>_joints.png`` and
``<scenario>_torques.png``, Section 3.9), and how far the pushes carry each arm off
the taught path (``flexibility.png`` and ``flexibility.csv``, Section 3.8). With
``--animations``, it animates the undisturbed run, the block, the pushes, the
offset start of OFFSET, and one from which the chosen ESN fails (``<name>.gif``),
the arms side by side, each rendered by skelarm's player. With ``--videos``, it
exports the take, as recorded and filtered at 2 Hz, and every arm (the chosen ESN's
and the replay's) with each tracker in those five runs as an MP4 for slides, with
the player's side panel, into VIDEOS under the storage root, since they are not
kept in Git.

    uv run python reports/005-tuned-on-the-robot/make_figures.py --videos
"""

from __future__ import annotations

import argparse
import csv
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
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.autonomous import Setup, load_setup
from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles, smooth_take
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.drawing import draw_postures, spread_along_path
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.metrics import distances_to_path
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path, storage_root

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
TAKE_RUN = REPORT / "data" / "20261008-170034-reach_manual_v2"
DT = 0.01  # the ESNs' period (s)
# The tracker of every robot run: its settings' directories in a run, and their names.
SETTINGS = {"computed_torque_w20_z0.1": "computed torque", "pd_w20_z0.1": "joint PD"}
# The ESN validated on the robot: its configuration tag, and its name.
CHOSEN = "fine_best"
CHOSEN_NAME = "chosen ESN (leak rate 0.05, seed 0)"
OFFSET = (15.0, 15.0)  # the offset start (deg) shown: the largest from which the chosen ESN arrives and holds
PUSH_TIMES = [(4.0, 4.2, 8.0), (8.0, 8.2, 12.0), (13.0, 13.2, 17.0)]  # each push: onset, end, and window end (s)
BACK_WITHIN = 5.0  # mm: an arm is back on the taught path once it stays within this of it
FLEXIBILITY_CSV = SUMMARY / "flexibility.csv"

ESN_COLOR = "#2a78d6"
CT_COLOR = "#4a3aa7"  # the chosen ESN with computed torque, in the animations
REPLAY_COLOR = "#eb6834"
RECORDED_COLOR = "#a3a29d"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAKE_COLORS = {"as recorded": "#a3a29d", "filtered at 8 Hz": "#eb6834", "filtered at 2 Hz": "#2a78d6"}

# The arms compared, both in the chosen ESN's runs: their name and color.
ARMS = {"esn": (CHOSEN_NAME, ESN_COLOR), "replay": ("replay", REPLAY_COLOR)}

# The panels of every animation: the label (the arm's reference, then its tracker), the tracker setting, the arm,
# and the label's color.
PANELS = [
    (f"{CHOSEN_NAME}\ntracker: joint PD", "pd_w20_z0.1", "esn", ESN_COLOR),
    ("replay of the take by time\ntracker: joint PD", "pd_w20_z0.1", "replay", REPLAY_COLOR),
    (f"{CHOSEN_NAME}\ntracker: computed torque", "computed_torque_w20_z0.1", "esn", CT_COLOR),
]
# The animations: name, scenario, the start offset (deg) from the demonstrated start, and the title.
ANIMATIONS = [
    ("nominal", "nominal", (0.0, 0.0), "undisturbed, from the demonstrated start"),
    ("block", "block", (0.0, 0.0), "the tip held from 4.5 s to 5.5 s"),
    ("pushes", "pushes", (0.0, 0.0), "three pushes, at 4 s, 8 s, and 13 s"),
    (
        "offset",
        "offsets",
        OFFSET,
        "start offset by +15 deg in both joints (turned counterclockwise, the elbow more bent)",
    ),
    ("offset_failure", "offsets", (0.0, -10.0), "start offset by -10 deg in joint 2 (the elbow straighter)"),
]
ANIMATION_SPAN = (-0.2, 17.0)
ANIMATION_FPS = 10.0
ANIMATION_HOLD_MS = 1500  # the last frame stays this long before the GIF loops

# The videos for slides: the take, and each animation's run, with every arm and tracker, named by these.
VIDEOS = storage_root() / "reports" / REPORT.name / "videos"
VIDEO_FPS = 30.0
# The take's videos: each log of the take, its video's name, and what it shows.
TAKE_VIDEOS = {
    "demo_00": ("take_as_recorded", "the take, as recorded"),
    "demo_00_filtered": ("take_filtered_2hz", "the take, filtered at 2 Hz"),
}
VIDEO_ARMS = {"esn": "chosen_esn", "replay": "replay"}
VIDEO_TRACKERS = {"pd_w20_z0.1": "joint_pd", "computed_torque_w20_z0.1": "computed_torque"}
VIDEOS_README = """# Videos of report 005's take and runs, for slides

These MP4s show the take of report 005, *A remade arm, and an ESN tuned on the
robot* (`reports/005-tuned-on-the-robot/` in the arm-esn-ctrl repository), and the
runs behind its animations, one arm per video. They are not kept in Git.

All are H.264 at {fps:g} fps, 1104 x 800, rendered by skelarm's player with its side
panel: the time, the joint angles, the tip's position, and, where the log holds the
joint velocities, the tip's speed. The purple dot is the target. The videos have no
labels: their names say what they show.

## The take

The take taught by hand, from 0 s to its end at 18.5 s:

| Name | Take |
| --- | --- |
{takes}

The arm takes the logged postures; nothing is simulated. The filtered take is what
the ESNs were trained on and what the replay replays. The take as recorded moves in
the cursor's steps of a few millimeters, and its panel shows no tip speed, since its
log holds no joint velocities.

## The runs

`<scenario>_<arm>_<tracker>.mp4`, {count} in all:

| Name | Scenario |
| --- | --- |
{scenarios}

- **Arms:** `chosen_esn`, driven by the chosen ESN (leak rate 0.05, spectral radius
  0.99, input scaling 2, seed 0) from the arm's measured posture; `replay`, tracking
  the take replayed by time.
- **Trackers:** `joint_pd` and `computed_torque`, at omega = 20 rad/s with damping
  ratio 0.1, given a zero reference velocity.
- **Span:** the whole run, from the warm-up at -1 s to the end at 22 s. The side
  panel also shows the external force while a disturbance acts, and the red arrow
  is the force at the tip.

[`videos.csv`](videos.csv) lists each video's scenario, arm, tracker (none for the
take), and the log it shows, relative to the storage root.

## Reproducing them

From the repository, at commit `{commit}`:

    uv run python reports/005-tuned-on-the-robot/make_figures.py --videos

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
    figures = {
        "take.png": plot_take(),
        "stage1.png": plot_stage("sweep_robot_main_lr", "Stage 1: the main parameters"),
        "stage3.png": plot_stage("sweep_robot_fine_lr", "Stage 3: a finer grid, to input scaling 3"),
        "seeds.png": plot_seeds(),
    }
    if args.logs:
        for scenario in ("nominal", "block", "pushes"):
            figures[f"{scenario}_joints.png"], figures[f"{scenario}_torques.png"] = plot_runs(scenario)
        figures["offset_joints.png"], figures["offset_torques.png"] = plot_runs("offsets", OFFSET)
        figures["flexibility.png"] = plot_flexibility()
        write_flexibility()
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


def plot_take() -> Figure:
    """The take as recorded and filtered at 8 Hz and 2 Hz, every 10 ms: hand path, joint angles, and hand speed."""
    with (TAKE_RUN / "config.toml").open("rb") as f:
        config = tomllib.load(f)
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    recorded = StateLog.load(TAKE_RUN / "demo_00.sklog.npz")
    takes = {
        "as recorded": recorded,
        "filtered at 8 Hz": smooth_take(recorded, skeleton, config["task"], {"kind": "lowpass", "cutoff_hz": 8.0}),
        "filtered at 2 Hz": StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"),
    }
    fig = Figure(figsize=(14, 8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("The take of the remade arm, every 10 ms, as the ESN sees it", color="#0b0b0b")
    grid = fig.add_gridspec(2, 2)
    ax_path, ax_speed = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_q1, ax_q2 = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    for ax in (ax_path, ax_speed, ax_q1, ax_q2):
        style(ax)
    target = np.asarray(config["task"]["target"]["pos"], dtype=np.float64)
    for label, log in takes.items():
        times, q = resample_joint_angles(log, DT)
        hand = endpoint_positions(skeleton, q)
        color, width = TAKE_COLORS[label], 2.2 if label == "as recorded" else 1.2
        speed = np.linalg.norm(np.gradient(hand, DT, axis=0), axis=1)
        ax_path.plot(*hand.T, color=color, linewidth=width, label=label)
        if label == "filtered at 2 Hz":
            draw_postures(ax_path, skeleton, q[spread_along_path(hand)])
        ax_speed.plot(times, speed, color=color, linewidth=0.9 if label == "as recorded" else 1.2, label=label)
        ax_q1.plot(times, np.degrees(q[:, 0]), color=color, linewidth=width)
        ax_q2.plot(times, np.degrees(q[:, 1]), color=color, linewidth=width)
    ax_path.plot(*target, marker="+", markersize=14, color="#0b0b0b", markeredgewidth=1.5)
    ax_path.set(
        title="Hand path, with the arm along the take (cross: the target)",
        xlabel="x (m)",
        ylabel="y (m)",
        aspect="equal",
    )
    ax_path.legend(frameon=False, labelcolor=TEXT_COLOR)
    ax_speed.set(title="Hand speed", xlabel="time (s)", ylabel="m/s", xlim=(0, 11), ylim=(0, 0.6))
    ax_q1.set(title="Joint 1", xlabel="time (s)", ylabel="deg", xlim=(0, 11))
    ax_q2.set(title="Joint 2", xlabel="time (s)", ylabel="deg", xlim=(0, 11))
    return fig


def stage_rows(prefix: str) -> list[dict[str, Any]]:
    """The combinations of the sweep runs whose stems start with ``prefix``, each with its leak rate."""
    rows = []
    for path in sorted(RESULTS.glob(f"*-{prefix}*")):
        leak = float(re.search(r"_lr([\d.]+)_", path.name)[1])  # type: ignore[index]
        rows += [r | {"leak_rate": leak} for r in read_csv(path / "sweep.csv")]
    return rows


def plot_stage(prefix: str, title: str) -> Figure:
    """Heatmaps of a sweep's worst path RMSE, input scaling against spectral radius, one panel per leak rate.

    A cell's number is the worst path RMSE (mm) over the two trackers; a cell with a
    failed run is marked with an x instead.
    """
    rows = stage_rows(prefix)
    leaks = sorted({r["leak_rate"] for r in rows})
    inputs = sorted({r["input_scaling"] for r in rows})
    radii = sorted({r["spectral_radius"] for r in rows})
    fig = Figure(figsize=(2.6 * len(leaks) + 1.2, 3.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"{title}: worst path RMSE from the taught path (mm); x: a run fails", color="#0b0b0b")
    axes = fig.subplots(1, len(leaks), sharey=True, squeeze=False)[0]
    held = [1000 * r["worst_path_rmse_m"] for r in rows if r["failures"] == 0]
    norm = LogNorm(vmin=min(held), vmax=max(held))
    for ax, leak in zip(axes, leaks, strict=True):
        grid = np.full((len(radii), len(inputs)), np.nan)
        for r in rows:
            if r["leak_rate"] == leak:
                grid[radii.index(r["spectral_radius"]), inputs.index(r["input_scaling"])] = (
                    1000 * r["worst_path_rmse_m"] if r["failures"] == 0 else np.nan
                )
        ax.imshow(grid, cmap="Blues", norm=norm, origin="lower", aspect="auto")
        for r in rows:
            if r["leak_rate"] != leak:
                continue
            y, x = radii.index(r["spectral_radius"]), inputs.index(r["input_scaling"])
            if r["failures"]:
                ax.text(x, y, "x", ha="center", va="center", fontsize=8, color=TEXT_COLOR)
            else:
                v = 1000 * r["worst_path_rmse_m"]
                ax.text(
                    x,
                    y,
                    f"{v:.3g}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if norm(v) > 0.6 else "#0b0b0b",
                )
        ax.set_xticks(range(len(inputs)), [f"{v:g}" for v in inputs], fontsize=7)
        ax.set_yticks(range(len(radii)), [f"{v:g}" for v in radii], fontsize=7)
        ax.set_title(f"leak rate {leak:g}", color="#0b0b0b", fontsize=9)
        ax.set_xlabel("input scaling", color=TEXT_COLOR, fontsize=8)
    axes[0].set_ylabel("spectral radius", color=TEXT_COLOR)
    fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=axes.tolist(), shrink=0.9, label="mm")
    return fig


def seed_rows() -> dict[str, list[dict[str, Any]]]:
    """The seed check's runs, by their combination ("lr0.03_sr0.99_is2"), each sorted by seed."""
    groups = {}
    for path in sorted(RESULTS.glob("*-sweep_robot_seeds_filtered_*")):
        tag = path.name.split("filtered_")[1]
        groups[tag] = sorted(read_csv(path / "sweep.csv"), key=lambda r: r["seed"])
    return dict(sorted(groups.items(), key=lambda item: np.mean([r["worst_path_rmse_m"] for r in item[1]])))


def plot_seeds() -> Figure:
    """The worst path RMSE of the five best combinations with ten random reservoirs each, seeds joined by lines."""
    groups = seed_rows()
    fig = Figure(figsize=(9, 4.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("Stage 4: the five best combinations, each with ten random reservoirs (seeds 0 to 9)", color="#0b0b0b")
    ax = fig.subplots()
    style(ax)
    tags = list(groups)
    values = np.array([[1000 * r["worst_path_rmse_m"] for r in groups[tag]] for tag in tags])
    for seed in range(values.shape[1]):
        ax.plot(range(len(tags)), values[:, seed], color=GRID_COLOR, linewidth=0.8, zorder=1)
    for k in range(len(tags)):
        ax.scatter(np.full(values.shape[1], k), values[k], color=ESN_COLOR, s=18, zorder=2)
        mean, sd = values[k].mean(), values[k].std(ddof=1)
        ax.errorbar(k + 0.18, mean, yerr=sd, fmt="o", color="#0b0b0b", markersize=5, capsize=4, zorder=3)
    labels = [re.sub(r"lr([\d.]+)_sr([\d.]+)_is([\d.]+)", r"leak rate \1\ninput scaling \3", tag) for tag in tags]
    ax.set_xticks(range(len(tags)), labels, fontsize=8)
    ax.set_ylabel("worst path RMSE (mm)", color=TEXT_COLOR)
    ax.set_title(
        "dots: seeds (gray lines join each seed); black: mean and SD; spectral radius 0.99",
        color=TEXT_COLOR,
        fontsize=9,
    )
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
    demonstrated start.
    """
    robot_esn = runner("robot_esn")
    stem = stem_of(scenario)
    setup, config = robot_setup(stem)
    settings = robot_esn.tracker_settings(config["tracker"])
    spans = disturbance_spans(config.get("disturbance"))
    start = start_of(stem, "esn", offset)
    title = f"The chosen ESN on the robot: {scenario}"
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


def plot_flexibility() -> Figure:
    """The hand's distance from the taught path under the pushes, for the chosen ESN and the replay, per tracker."""
    fig = Figure(figsize=(13, 7), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("Three pushes: how far each arm is carried off the taught path, and back", color="#0b0b0b")
    axes = fig.subplots(len(SETTINGS), 1, sharex=True)
    for ax, (setting, name) in zip(axes, SETTINGS.items(), strict=True):
        style(ax)
        for onset, end, _ in PUSH_TIMES:
            ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
        for arm, (label, color) in ARMS.items():
            times, distance = path_distances("pushes", setting, arm)
            ax.plot(times, distance, color=color, linewidth=1.3, label=label)
        ax.axhline(BACK_WITHIN, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
        ax.set(title=name, ylabel="from the taught path (mm)", ylim=(0, None), xlim=(0, 18))
    axes[-1].set_xlabel("time (s)", color=TEXT_COLOR)
    axes[0].legend(frameon=False, labelcolor=TEXT_COLOR)
    return fig


def write_flexibility() -> None:
    """Per tracker, arm, and push: the farthest from the taught path, and when the hand settles back within 5 mm.

    The window of a push runs from its onset to the next push, or 4 s; the hand
    settles when it stays within BACK_WITHIN of the path to the window's end (NaN if
    it does not), counted from the push's end.
    """
    out = []
    for setting, name in SETTINGS.items():
        for arm, (label, _) in ARMS.items():
            times, distance = path_distances("pushes", setting, arm)
            row: dict[str, Any] = {"tracker": name, "arm": label}
            for k, (onset, end, stop) in enumerate(PUSH_TIMES, start=1):
                window = (times >= onset) & (times < stop)
                outside = np.flatnonzero(window & (distance > BACK_WITHIN))
                settled = len(outside) and times[outside[-1]] < stop - 0.05
                row[f"push{k}_farthest_mm"] = float(distance[window].max())
                row[f"push{k}_settles_s"] = float(times[outside[-1] + 1] - end) if settled else float("nan")
            out.append(row)
    write_csv(FLEXIBILITY_CSV, out)


def print_block() -> None:
    """The block, from the logs: the force at the tip, how far the reference runs ahead, and the hand's return.

    The peak force of metrics.csv is the grip's: the damper stops the moving hand.
    How hard the arm then presses is the mean force over the block's last 0.1 s,
    when its reference (the ESN's output, or the take) is the farthest ahead of the
    held hand. After the release: the farthest the hand goes from the taught path,
    and when it is back within BACK_WITHIN of it for good (counted from the release,
    up to 12 s).
    """
    print(
        "\n== 3.7 The block, from the logs: arm, tracker: force at the tip (N) at the grip (peak, first 0.1 s) and at"
    )
    print("the release (mean, last 0.1 s); the reference's hand ahead of the arm's at the release (mm); after the")
    print("release, the farthest from the taught path (mm), and back within 5 mm (s)")
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    stem = stem_of("block")
    with (RESULTS / run_name(stem) / "config.toml").open("rb") as f:
        block = tomllib.load(f)["disturbance"]
    onset, release = block["onset"], block["release"]
    for arm, (label, _) in ARMS.items():
        for setting, name in SETTINGS.items():
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
            print(
                f"  {label}, {name}: {grip:.1f}, {pressing:.1f}; {ahead:.0f}; {distance[after].max():.0f}, {back:.1f}"
            )


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
    """Export the take and each animation's run, with every arm and tracker, as MP4s for slides into VIDEOS.

    Each video is one log, rendered by skelarm's player at VIDEO_FPS with its side
    panel: the take, as recorded and filtered, from the take's run under the storage
    root (TAKE_VIDEOS), and each arm's whole run, the warm-up included. A run's video
    is named ``<scenario>_<arm>_<tracker>.mp4`` (see :func:`video_name`), such as
    ``offset_+15_+15_replay_joint_pd.mp4``; ``videos.csv`` lists each video's
    scenario, arm, tracker, and log, and ``README.md`` (VIDEOS_README) explains
    them and how to reproduce them, with the commit they were exported at.
    """
    VIDEOS.mkdir(parents=True, exist_ok=True)
    take_run = resolve_run_path(f"results/{TAKE_RUN.name}")
    # Each video, its scenario, arm, and tracker (none for the take), and its log.
    videos = [
        (VIDEOS / f"{name}.mp4", title, "", "", take_run / f"{log}.sklog.npz")
        for log, (name, title) in TAKE_VIDEOS.items()
    ]
    for _, scenario, offset, title in ANIMATIONS:
        name = video_name(scenario, offset)
        stem = stem_of(scenario)
        for arm, arm_name in VIDEO_ARMS.items():
            start = start_of(stem, arm, offset)
            for setting, tracker in VIDEO_TRACKERS.items():
                log = run_dir(stem) / setting / f"{arm}_{start:02d}.sklog.npz"
                videos.append(
                    (VIDEOS / f"{name}_{arm_name}_{tracker}.mp4", title, ARMS[arm][0], SETTINGS[setting], log)
                )
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
    takes = "\n".join(f"| `{name}` | {title} |" for name, title in TAKE_VIDEOS.values())
    scenarios = "\n".join(
        f"| `{video_name(scenario, offset)}` | {title} |" for _, scenario, offset, title in ANIMATIONS
    )
    readme = VIDEOS_README.format(
        takes=takes,
        count=len(rows) - len(TAKE_VIDEOS),
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
    take = read_csv(TAKE_RUN / "metrics.csv")
    print("== 3.1 The take: moves (s), arrives (s), length (s), joint jitter (deg), final error (mm)")
    for r in take:
        print(
            f"  {r['demo']}: {r['onset_time_s']:.2f}, {r['arrival_time_s']:.2f}, {r['length_s']:.2f},"
            f" {r['jitter_deg']:.4f}, {1000 * r['final_error']:.1f}"
        )

    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    with (TAKE_RUN / "config.toml").open("rb") as f:
        task = tomllib.load(f)["task"]
    target = np.asarray(task["target"]["pos"], dtype=np.float64)
    recorded = StateLog.load(TAKE_RUN / "demo_00.sklog.npz")
    times, q = resample_joint_angles(recorded, DT)
    hand = endpoint_positions(skeleton, q)
    distance = np.linalg.norm(hand - target, axis=1)
    progress = 1 - distance / distance[0]
    shares = ", ".join(f"{100 * progress[np.argmin(abs(times - t))]:.1f}% at {t:g} s" for t in (4.5, 5.5))
    shares += f" (the block); 50% at {times[np.argmax(progress >= 0.5)]:.2f} s"
    raw_hand = endpoint_positions(skeleton, recorded.channel("q").reshape(len(recorded.times), -1))
    steps = np.linalg.norm(np.diff(raw_hand, axis=0), axis=1)
    reach = (recorded.times[1:] - recorded.times[0] >= 0.64) & (recorded.times[1:] - recorded.times[0] <= 9.64)
    print(f"  the hand: {shares}; joint 2 up to {np.degrees(q[:, 1].max()):.1f} deg; during the reach,")
    still, step = 100 * np.mean(steps[reach] == 0), 1000 * np.median(steps[reach][steps[reach] > 0])
    print(f"  {still:.0f}% of the samples do not move, median step {step:.1f} mm")
    takes = {"as recorded": recorded} | {
        f"filtered at {cutoff:g} Hz": smooth_take(recorded, skeleton, task, {"kind": "lowpass", "cutoff_hz": cutoff})
        for cutoff in (8.0, 2.0)
    }
    for label, log in takes.items():
        t, qq = resample_joint_angles(log, DT)
        h = endpoint_positions(skeleton, qq)
        acceleration = np.gradient(np.gradient(h, DT, axis=0), DT, axis=0)
        moving = (t >= 0.5) & (t <= 9.8)
        apart = 1000 * np.linalg.norm(h - hand, axis=1).max()
        rms = np.sqrt(np.mean(np.sum(acceleration[moving] ** 2, axis=1)))
        print(f"  {label}: RMS hand acceleration (10 ms) {rms:.2f} m/s^2; at most {apart:.1f} mm from the take")

    print("\n== 3.2 The first ESN (grid_test_filtered) on the robot, 20 s runs: arm, tracker: arrival (s), holds,")
    print("observed hold (s), taught joint error (deg); its run predates the path RMSE, which stage 0 measures")
    for r in rows_of("nominal_test_filtered"):
        print(
            f"  {r['arm']}, {SETTINGS[setting_of(r)]}: {r['arrival_time_s']:.2f}, {r['success']},"
            f" {r['hold_observed_s']:.2f}, {r['taught_joint_error_deg']:.2f}"
        )
    alone = rows_of("grid_test_filtered")
    print(f"  on its own: {sum(r['success'] for r in alone)} of {len(alone)} starts arrive and hold")
    base = rows_of("sweep_robot_baseline_filtered", "sweep.csv")[0]
    per_setting = ", ".join(
        f"{name} {1000 * base[f'{setting}_path_rmse_m']:.2f} mm (arrival {base[f'{setting}_arrival_time_s']:.2f} s)"
        for setting, name in SETTINGS.items()
    )
    print(f"  stage 0, the same ESN in 22 s runs: path RMSE {per_setting}; {int(base['failures'])} failed")

    def print_stage(title: str, prefix: str) -> None:
        rows = stage_rows(prefix)
        ok = [r for r in rows if r["failures"] == 0]
        print(f"\n== {title}: {len(ok)} of {len(rows)} combinations arrive and hold;")
        print("the best per leak rate, and the top 5")
        for leak in sorted({r["leak_rate"] for r in rows}):
            group = [r for r in ok if r["leak_rate"] == leak]
            if group:
                b = min(group, key=lambda r: r["worst_path_rmse_m"])
                print(
                    f"  leak rate {leak:g}: {len(group)} hold; best spectral radius {b['spectral_radius']:g},"
                    f" input scaling {b['input_scaling']:g}: {1000 * b['worst_path_rmse_m']:.2f} mm,"
                    f" latest arrival {b['latest_arrival_s']:.2f} s"
                )
        weak = [r for r in ok if r["input_scaling"] <= 0.3]
        if weak:
            worst = [1000 * r["worst_path_rmse_m"] for r in weak]
            print(f"  with input scaling 0.3 or less: {len(weak)} hold, at {min(worst):.1f} to {max(worst):.1f} mm")
        runs = [r for path in sorted(RESULTS.glob(f"*-{prefix}*")) for r in read_csv(path / "runs.csv")]
        failed = [r for r in runs if r["failed"]]
        never = sum(not r["arrived"] for r in failed)
        print(f"  {len(failed)} failed runs, {never} never arrive")
        gaps = [
            abs(r[f"{setting}_path_rmse_m"] - r[f"{setting}_reference_path_rmse_m"]) / r[f"{setting}_path_rmse_m"]
            for r in ok
            for setting in SETTINGS
        ]
        print(f"  the ESN's output path RMSE is within {100 * max(gaps):.0f}% of the arm's in every one that holds")
        for r in sorted(ok, key=lambda r: r["worst_path_rmse_m"])[:6]:
            print(
                f"  top: leak {r['leak_rate']:g}, radius {r['spectral_radius']:g}, input {r['input_scaling']:g}:"
                f" {1000 * r['worst_path_rmse_m']:.2f} mm"
            )

    print_stage("3.3 Stage 1", "sweep_robot_main_lr")

    print("\n== 3.4 Stage 2: worst path RMSE (mm) over ridge (columns) and warm-up (rows); * = a failed run")
    for path in sorted(RESULTS.glob("*-sweep_robot_ridge_warmup_filtered_*")):
        rows = read_csv(path / "sweep.csv")
        ridges, warmups = sorted({r["ridge"] for r in rows}), sorted({r["warmup"] for r in rows})
        print(f"  {path.name.split('filtered_')[1]}: ridge " + " ".join(f"{x:>7g}" for x in ridges))
        for w in warmups:
            cells = [next(r for r in rows if r["ridge"] == x and r["warmup"] == w) for x in ridges]
            print(
                f"    warm-up {w:<4g} "
                + " ".join(f"{1000 * c['worst_path_rmse_m']:6.2f}" + ("*" if c["failures"] else " ") for c in cells)
            )

    print_stage("3.5 Stage 3", "sweep_robot_fine_lr")

    for name, x_key, y_key in (("sweep_robot_slow_leak_filtered", "input_scaling", "leak_rate"),):
        rows = rows_of(name, "sweep.csv")
        print("\n== 3.5 Slower leak rates: worst path RMSE (mm) [latest arrival, s]")
        for lr in sorted({r[y_key] for r in rows}):
            for sr in sorted({r["spectral_radius"] for r in rows}):
                cells = sorted(
                    (r for r in rows if r[y_key] == lr and r["spectral_radius"] == sr), key=lambda r: r[x_key]
                )
                print(
                    f"  leak rate {lr:g}, spectral radius {sr:g}: "
                    + "  ".join(
                        f"input {c[x_key]:g} {1000 * c['worst_path_rmse_m']:.2f} [{c['latest_arrival_s']:.2f}]"
                        for c in cells
                    )
                )
    rows = rows_of("sweep_robot_size_filtered", "sweep.csv")
    print("\n== 3.5 Size and sparsity: worst path RMSE (mm), failed runs")
    for sp in sorted({r["sparsity"] for r in rows}):
        cells = sorted((r for r in rows if r["sparsity"] == sp), key=lambda r: r["n_neurons"])
        print(
            f"  sparsity {sp:g}: "
            + "  ".join(
                f"{int(c['n_neurons'])} {1000 * c['worst_path_rmse_m']:.2f} ({int(c['failures'])})" for c in cells
            )
        )

    print("\n== 3.6 Seed check: worst path RMSE (mm) mean +- SD, median, min, max; per seed 0 to 9; all hold")
    for tag, rows in seed_rows().items():
        w = np.array([1000 * r["worst_path_rmse_m"] for r in rows])
        holds = sum(r["failures"] == 0 for r in rows)
        print(
            f"  {tag}: {holds}/10 hold; {w.mean():.2f} +- {w.std(ddof=1):.2f}, {np.median(w):.2f}, {w.min():.2f},"
            f" {w.max():.2f} | " + " ".join(f"{x:.2f}" for x in w)
        )

    groups = seed_rows()
    for seed in range(10):
        ranks = []
        for rows in groups.values():
            order = np.argsort([-r["worst_path_rmse_m"] for r in rows])  # the worst first
            ranks.append(int(np.flatnonzero(order == seed)[0]) + 1)
        print(f"  seed {seed}: rank from the worst in each combination: {ranks}")
    a, b = groups["lr0.03_sr0.99_is2"], groups["lr0.04_sr0.99_is2"]
    better = sum(x["worst_path_rmse_m"] < y["worst_path_rmse_m"] for x, y in zip(a, b, strict=True))
    print(f"  leak rate 0.03 beats 0.04 (input scaling 2) for {better} of 10 seeds")

    print("\n== 3.7 Validation: arm, scenario, tracker: arrival (s), holds, path RMSE (mm); block: peak force (N);")
    print("offsets: starts that hold, median path RMSE (mm), median peak torque (N m), failed starts")
    for arm, (label, _) in ARMS.items():
        for scenario in ("nominal", "block"):
            for r in rows_of(stem_of(scenario)):
                if r["arm"] != arm:
                    continue
                force = f", {r['peak_external_force_n']:.2f} N" if scenario == "block" else ""
                print(
                    f"  {label}, {scenario}, {SETTINGS[setting_of(r)]}: {r['arrival_time_s']:.2f}, {r['success']},"
                    f" {1000 * r['taught_path_rmse_m']:.2f}{force}"
                )
        offsets = [r for r in rows_of(stem_of("offsets")) if r["arm"] == arm]
        for setting, name in SETTINGS.items():
            group = [r for r in offsets if setting_of(r) == setting]
            fails = [r["origin"].split()[1] for r in group if not r["success"]]
            never = sum(not r["arrived"] for r in group if not r["success"])
            print(
                f"  {label}, offsets, {name}: {len(group) - len(fails)}/{len(group)},"
                f" {1000 * np.median([r['taught_path_rmse_m'] for r in group]):.1f},"
                f" {np.median([r['peak_torque_nm'] for r in group]):.2f}, {fails} ({never} never arrive)"
            )
        for r in rows_of(stem_of("offsets")):
            if r["arm"] == arm and offset_of(r["origin"]) == OFFSET:
                print(
                    f"  {label}, offset {OFFSET}, {SETTINGS[setting_of(r)]}: {r['arrival_time_s']:.2f}, {r['success']},"
                    f" {1000 * r['taught_path_rmse_m']:.1f}, peak torque {r['peak_torque_nm']:.1f}"
                )

    print("\n== 3.8 Pushes: arm, tracker: arrival (s), holds, path RMSE (mm), reference lead at the end of each push")
    for r in rows_of(stem_of("pushes")):
        leads = " ".join(f"{r[f'reference_lead_{k}']:+.3f}" for k in (1, 2, 3))
        print(
            f"  {ARMS[r['arm']][0]}, {SETTINGS[setting_of(r)]}: {r['arrival_time_s']:.2f}, {r['success']},"
            f" {1000 * r['taught_path_rmse_m']:.1f}, {leads}"
        )
    if FLEXIBILITY_CSV.exists():
        print("\n== 3.8 Each push: the farthest from the taught path (mm), and when the hand settles within 5 mm (s)")
        for r in read_csv(FLEXIBILITY_CSV):
            pushes = []
            for k in (1, 2, 3):
                settle = r[f"push{k}_settles_s"]
                pushes.append(
                    f"{r[f'push{k}_farthest_mm']:.0f} mm, "
                    + (f"{settle:.1f} s" if np.isfinite(settle) else "not within the window")
                )
            print(f"  {r['arm']}, {r['tracker']}: " + " | ".join(pushes))


if __name__ == "__main__":
    main()
