# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 005.

    uv run python reports/005-tuned-on-the-robot/make_figures.py
    uv run python reports/005-tuned-on-the-robot/make_figures.py --logs --animations

From the take in ``data/`` and the copies of the runs under ``results/``
(configurations, run records, metrics, and sweep tables), it draws into
``results/summary``:

- ``take.png``: the take, as recorded and filtered at 8 Hz and 2 Hz, as the ESN
  sees it, every 10 ms (Section 3.1);
- ``stage1.png``: the first sweep on the robot, the main parameters (Section 3.3);
- ``stage3.png``: the finer sweep around the best of stage 1 (Section 3.5);
- ``seeds.png``: the five best combinations, each with ten random reservoirs
  (Section 3.6);

and prints the tables of the report. With ``--logs``, it reads the runs' logs,
which are not kept in Git, from the runs under the storage root, and also draws,
with ``experiments/robot_esn.py``'s own functions, the joint angles and the joint
torques of the chosen ESN's arm and the replay in the undisturbed run, the block,
and the pushes (``<scenario>_joints.png`` and ``<scenario>_torques.png``, Section
3.9), and how far the pushes carry each arm off the taught path
(``flexibility.png`` and ``flexibility.csv``, Section 3.8). With
``--animations``, it animates the undisturbed run, the block, the pushes, and an
offset start (``<scenario>.gif``), the arms side by side, each rendered by
skelarm's player.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import tomllib
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
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.metrics import distances_to_path
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
TAKE_RUN = REPORT / "data" / "20261008-170034-reach_manual_v2"
DT = 0.01  # the ESNs' period (s)
# The tracker of every robot run: its settings' directories in a run, and their names.
SETTINGS = {"computed_torque_w20_z0.1": "computed torque", "pd_w20_z0.1": "joint PD"}
# The two ESNs validated on the robot: their configuration tag, and their name.
ESNS = {
    "fine_best": "chosen ESN (leak rate 0.05, seed 0)",
    "final": "ESN of the seed check (leak rate 0.03, seed 4)",
}
CHOSEN = "fine_best"
SCENARIOS = ["nominal", "block", "offsets", "pushes"]
PUSH_TIMES = [(4.0, 4.2, 8.0), (8.0, 8.2, 12.0), (13.0, 13.2, 17.0)]  # each push: onset, end, and window end (s)
BACK_WITHIN = 5.0  # mm: an arm is back on the taught path once it stays within this of it
FLEXIBILITY_CSV = SUMMARY / "flexibility.csv"

ESN_COLOR = "#2a78d6"
FINAL_COLOR = "#4a3aa7"
REPLAY_COLOR = "#eb6834"
RECORDED_COLOR = "#a3a29d"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
TAKE_COLORS = {"as recorded": "#a3a29d", "filtered at 8 Hz": "#eb6834", "filtered at 2 Hz": "#2a78d6"}

# The animations: name, scenario, the start offset (deg) from the demonstrated start, the title, and the panels:
# label, ESN tag, tracker setting, arm, and the label's color.
ANIMATIONS = [
    (
        "nominal",
        "nominal",
        (0.0, 0.0),
        "undisturbed, from the demonstrated start",
        [
            ("chosen ESN, joint PD", CHOSEN, "pd_w20_z0.1", "esn", ESN_COLOR),
            ("replay, joint PD", CHOSEN, "pd_w20_z0.1", "replay", REPLAY_COLOR),
            ("chosen ESN, computed torque", CHOSEN, "computed_torque_w20_z0.1", "esn", FINAL_COLOR),
        ],
    ),
    (
        "block",
        "block",
        (0.0, 0.0),
        "the tip held from 1.15 s to 1.65 s",
        [
            ("chosen ESN, joint PD", CHOSEN, "pd_w20_z0.1", "esn", ESN_COLOR),
            ("replay, joint PD", CHOSEN, "pd_w20_z0.1", "replay", REPLAY_COLOR),
            ("chosen ESN, computed torque", CHOSEN, "computed_torque_w20_z0.1", "esn", FINAL_COLOR),
        ],
    ),
    (
        "pushes",
        "pushes",
        (0.0, 0.0),
        "three pushes, at 4 s, 8 s, and 13 s",
        [
            ("chosen ESN, joint PD", CHOSEN, "pd_w20_z0.1", "esn", ESN_COLOR),
            ("seed-check ESN, joint PD", "final", "pd_w20_z0.1", "esn", FINAL_COLOR),
            ("replay, joint PD", CHOSEN, "pd_w20_z0.1", "replay", REPLAY_COLOR),
        ],
    ),
    (
        "offset",
        "offsets",
        (0.0, -10.0),
        "start offset by -10 deg in joint 2 (the elbow straighter)",
        [
            ("chosen ESN, joint PD", CHOSEN, "pd_w20_z0.1", "esn", ESN_COLOR),
            ("seed-check ESN, joint PD", "final", "pd_w20_z0.1", "esn", FINAL_COLOR),
            ("replay, joint PD", CHOSEN, "pd_w20_z0.1", "replay", REPLAY_COLOR),
        ],
    ),
]
ANIMATION_SPAN = (-0.2, 17.0)
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
        "stage1.png": plot_stage("sweep_robot_main_lr", "Stage 1: the main parameters"),
        "stage3.png": plot_stage("sweep_robot_fine_lr", "Stage 3: a finer grid, to input scaling 3"),
        "seeds.png": plot_seeds(),
    }
    if args.logs:
        for scenario in ("nominal", "block", "pushes"):
            figures[f"{scenario}_joints.png"], figures[f"{scenario}_torques.png"] = plot_runs(scenario)
        figures["flexibility.png"] = plot_flexibility()
        write_flexibility()
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=150)
    print_tables()
    if args.animations:
        for name, scenario, offset, title, panels in ANIMATIONS:
            export_animation(name, scenario, offset, title, panels)
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


def stem_of(scenario: str, esn: str) -> str:
    """The configuration stem of a validation run of one of the two ESNs."""
    return f"{scenario}_robot_{esn}_filtered"


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
        ax_speed.plot(times, speed, color=color, linewidth=0.9 if label == "as recorded" else 1.2, label=label)
        ax_q1.plot(times, np.degrees(q[:, 0]), color=color, linewidth=width)
        ax_q2.plot(times, np.degrees(q[:, 1]), color=color, linewidth=width)
    ax_path.plot(*target, marker="+", markersize=14, color="#0b0b0b", markeredgewidth=1.5)
    ax_path.set(title="Hand path (cross: the target)", xlabel="x (m)", ylabel="y (m)", aspect="equal")
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


def plot_runs(scenario: str) -> tuple[Figure, Figure]:
    """The joint angles and the joint torques of the chosen ESN's run of a scenario, by robot_esn.py's functions."""
    robot_esn = runner("robot_esn")
    stem = stem_of(scenario, CHOSEN)
    setup, config = robot_setup(stem)
    settings = robot_esn.tracker_settings(config["tracker"])
    spans = disturbance_spans(config.get("disturbance"))
    title = f"The chosen ESN on the robot: {scenario}"
    directory = run_dir(stem)
    return (
        robot_esn.plot_joints(directory, setup, settings, spans, title),
        robot_esn.plot_torques(directory, setup, settings, spans, title),
    )


def push_distances(esn: str, setting: str, arm: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The times (every 10 ms) and the hand's distance (mm) from the taught path in a pushes run."""
    log = StateLog.load(run_dir(stem_of("pushes", esn)) / setting / f"{arm}_00.sklog.npz")
    task = log.times >= 0
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    hand = endpoint_positions(skeleton, log.channel("q").reshape(len(log.times), -1)[task])
    every = round(DT / float(np.median(np.diff(log.times))))
    return log.times[task][::every], 1000 * distances_to_path(hand[::every], taught_hand())


def arms_of_pushes() -> list[tuple[str, str, str, str]]:
    """The arms compared under the pushes: label, ESN tag, arm, and color (the replay is the same in both runs)."""
    return [
        (ESNS[CHOSEN], CHOSEN, "esn", ESN_COLOR),
        (ESNS["final"], "final", "esn", FINAL_COLOR),
        ("replay", CHOSEN, "replay", REPLAY_COLOR),
    ]


def plot_flexibility() -> Figure:
    """The hand's distance from the taught path under the pushes, for the two ESNs and the replay, per tracker."""
    fig = Figure(figsize=(13, 7), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle("Three pushes: how far each arm is carried off the taught path, and back", color="#0b0b0b")
    axes = fig.subplots(len(SETTINGS), 1, sharex=True)
    for ax, (setting, name) in zip(axes, SETTINGS.items(), strict=True):
        style(ax)
        for onset, end, _ in PUSH_TIMES:
            ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
        for label, esn, arm, color in arms_of_pushes():
            times, distance = push_distances(esn, setting, arm)
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
        for label, esn, arm, _ in arms_of_pushes():
            times, distance = push_distances(esn, setting, arm)
            row: dict[str, Any] = {"tracker": name, "arm": label}
            for k, (onset, end, stop) in enumerate(PUSH_TIMES, start=1):
                window = (times >= onset) & (times < stop)
                outside = np.flatnonzero(window & (distance > BACK_WITHIN))
                settled = len(outside) and times[outside[-1]] < stop - 0.05
                row[f"push{k}_farthest_mm"] = float(distance[window].max())
                row[f"push{k}_settles_s"] = float(times[outside[-1] + 1] - end) if settled else float("nan")
            out.append(row)
    write_csv(FLEXIBILITY_CSV, out)


# ---------------------------------------------------------------------------- animations


def export_animation(
    name: str,
    scenario: str,
    offset: tuple[float, float],
    title: str,
    panels: list[tuple[str, str, str, str, str]],
) -> None:
    """Animate one start of a scenario into ``<name>.gif``: the arms of ``panels`` side by side, rendered by skelarm.

    Each panel is a label, an ESN tag, a tracker setting, an arm ("esn" or
    "replay"), and the label's color. The player exports each arm's run as a GIF
    (``--export``); its frames, which the GIF merges where the arm rests, are spread
    back over time, cropped to where the arms move, labeled, and tiled into one GIF
    with the task time and the disturbance acting.
    """
    import tempfile

    from PIL import Image, ImageDraw, ImageFont, ImageSequence

    player = REPO_ROOT / "third_party" / "skelarm" / "tools" / "player.py"
    with (RESULTS / run_name(stem_of(scenario, CHOSEN)) / "config.toml").open("rb") as f:
        disturbance = tomllib.load(f).get("disturbance")
    tables = [] if disturbance is None else [disturbance] if isinstance(disturbance, dict) else disturbance
    spans = disturbance_spans(disturbance)
    frames, starts = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (_, esn, setting, arm, _) in enumerate(panels):
            stem = stem_of(scenario, esn)
            start = next(int(r["start"]) for r in rows_of(stem) if r["arm"] == arm and offset_of(r["origin"]) == offset)
            path = run_dir(stem) / setting / f"{arm}_{start:02d}.sklog.npz"
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
    top, bottom = max(int(rows.min()) - margin, 0), int(rows.max()) + margin
    left, right = max(int(cols.min()) - margin, 0), int(cols.max()) + margin
    width, height = right - left, bottom - top
    header, label_height, gap = 44, 30, 8
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
        for i, (label, _, _, _, color) in enumerate(panels):
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
    shares = ", ".join(
        f"{100 * share:g}% at {times[np.argmax(distance <= distance[0] * (1 - share))]:.2f} s" for share in (0.034, 0.5)
    )
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

    print("\n== 3.7 Validation: ESN, scenario, tracker: arrival (s), holds, path RMSE (mm); block: holding force (N);")
    print("offsets: starts that hold, median path RMSE (mm), median peak torque (N m), failed starts")
    for esn, label in ESNS.items():
        for scenario in ("nominal", "block"):
            for r in rows_of(stem_of(scenario, esn)):
                if r["arm"] != "esn":
                    continue
                force = f", {r['peak_external_force_n']:.2f} N" if scenario == "block" else ""
                print(
                    f"  {label}, {scenario}, {SETTINGS[setting_of(r)]}: {r['arrival_time_s']:.2f}, {r['success']},"
                    f" {1000 * r['taught_path_rmse_m']:.2f}{force}"
                )
        offsets = [r for r in rows_of(stem_of("offsets", esn)) if r["arm"] == "esn"]
        for setting, name in SETTINGS.items():
            group = [r for r in offsets if setting_of(r) == setting]
            fails = [r["origin"].split()[1] for r in group if not r["success"]]
            never = sum(not r["arrived"] for r in group if not r["success"])
            print(
                f"  {label}, offsets, {name}: {len(group) - len(fails)}/{len(group)},"
                f" {1000 * np.median([r['taught_path_rmse_m'] for r in group]):.1f},"
                f" {np.median([r['peak_torque_nm'] for r in group]):.2f}, {fails} ({never} never arrive)"
            )
    replays = [r for r in rows_of(stem_of("offsets", CHOSEN)) if r["arm"] == "replay"]
    for setting, name in SETTINGS.items():
        group = [r for r in replays if setting_of(r) == setting]
        print(
            f"  replay, offsets, {name}: {sum(r['success'] for r in group)}/{len(group)},"
            f" {1000 * np.median([r['taught_path_rmse_m'] for r in group]):.1f},"
            f" {np.median([r['peak_torque_nm'] for r in group]):.2f}"
        )
    for scenario in ("nominal", "block"):
        for r in rows_of(stem_of(scenario, CHOSEN)):
            if r["arm"] == "replay":
                force = f", {r['peak_external_force_n']:.2f} N" if scenario == "block" else ""
                print(
                    f"  replay, {scenario}, {SETTINGS[setting_of(r)]}: {r['arrival_time_s']:.2f}, {r['success']},"
                    f" {1000 * r['taught_path_rmse_m']:.2f}{force}"
                )

    print("\n== 3.8 Pushes: arm, tracker: arrival (s), holds, path RMSE (mm), reference lead at the end of each push")
    for esn, label in ESNS.items():
        for r in rows_of(stem_of("pushes", esn)):
            if r["arm"] == "replay" and esn != CHOSEN:
                continue
            name = label if r["arm"] == "esn" else "replay"
            leads = " ".join(f"{r[f'reference_lead_{k}']:+.3f}" for k in (1, 2, 3))
            print(
                f"  {name}, {SETTINGS[setting_of(r)]}: {r['arrival_time_s']:.2f}, {r['success']},"
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
