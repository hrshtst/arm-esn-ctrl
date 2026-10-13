# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 006.

    uv run python reports/006-wider-search-on-the-robot/make_figures.py
    uv run python reports/006-wider-search-on-the-robot/make_figures.py --logs --animations

From report 005's copy of the take and the copies of the runs under ``results/``
(configurations, run records, metrics, and sweep tables), it draws into
``results/summary``:

- ``stage1.png``: the wide grid, each setting's median over its ten seeds
  (Section 3.1);
- ``star.png``: Step B, the star around candidate #2, every seed undisturbed,
  under the pushes, and under the block (Section 3.4);

and prints the tables of the report. With ``--logs``, it reads the runs' logs,
which are not kept in Git, from the runs under the storage root, and also draws,
with ``experiments/robot_esn.py``'s own functions, the joint angles and the joint
torques of the chosen ESN's arm and the replay in the undisturbed run, the block,
the pushes, and the offset starts of OFFSETS (``<scenario>_joints.png`` and
``<scenario>_torques.png``, Section 3.8), and how far the pushes carry each arm off
the taught path (``flexibility.png`` and ``flexibility.csv``, Section 3.7). With
``--animations``, it animates the undisturbed run, the block, the pushes, and the
offset starts of OFFSETS (``<name>.gif``), the arms side by side, each rendered by
skelarm's player. With ``--videos``, it exports every arm (the chosen ESN's and the
replay's) with each tracker in those five runs as an MP4 for slides, with the
player's side panel, into VIDEOS under the storage root, since they are not kept in
Git; report 005 exported the take's.

    uv run python reports/006-wider-search-on-the-robot/make_figures.py --videos
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
REPORT_005 = REPORT.parent / "005-tuned-on-the-robot" / "results"
DT = 0.01  # the ESNs' period (s)
GOAL = 0.02  # the goal's radius (m): a run ending farther from the target ends away from it
# The tracker of every robot run: its settings' directories in a run, and their names.
SETTINGS = {"computed_torque_w20_z0.1": "computed torque", "pd_w20_z0.1": "joint PD"}
# The ESN validated on the robot: its configuration tag, and its name.
CHOSEN = "lr0.03_sr1.35_is1.5_seed8"
CHOSEN_NAME = "chosen ESN (input scaling 1.5, seed 8)"
# Candidate #2's setting, the center of Step B's star and the reference of the small grid.
CENTER = {"leak_rate": 0.03, "spectral_radius": 1.35, "input_scaling": 2.0}
# The candidates of stage 1 tried in the apps: number, configuration tag, and why each.
CANDIDATES = {
    1: ("lr0.03_sr1.5_is1.5_seed2", "stage 1's best setting"),
    2: ("lr0.03_sr1.35_is2_seed5", "the setting steadiest over its seeds"),
    3: ("lr0.03_sr1.5_is1_seed7", "the weakest input among the best settings"),
    4: ("lr0.05_sr1.5_is1.5_seed3", "report 005's leak rate with a spectral radius over 1"),
    5: ("lr0.07_sr1.5_is1.5_seed3", "a faster leak rate"),
    6: ("lr0.1_sr1.2_is1.5_seed4", "a faster leak rate still"),
    7: ("lr0.03_sr0.99_is1.5_seed2", "#1's reservoir with spectral radius 0.99"),
}
KEPT = (1, 2, 6, 7)  # the candidates of Step A
# Step B's star: each line's tag and its swept key.
LINES = {"sr": "spectral_radius", "lr": "leak_rate", "is": "input_scaling"}
# The offset starts (deg) shown: the largest offset, and the one from which report 005's ESN failed.
OFFSETS = [(15.0, 15.0), (0.0, -10.0)]
PUSH_TIMES = [(4.0, 4.2, 8.0), (8.0, 8.2, 12.0), (13.0, 13.2, 17.0)]  # each push: onset, end, and window end (s)
# A hand not back on the path by the next push, or by the end of the run, counts as back then (s after the push).
RETURN_CAPS = {1: 8.0 - 4.2, 3: 22.0 - 13.2}
BACK_WITHIN = 5.0  # mm: an arm is back on the taught path once it stays within this of it
FLEXIBILITY_CSV = SUMMARY / "flexibility.csv"

ESN_COLOR = "#2a78d6"
CT_COLOR = "#4a3aa7"  # the chosen ESN with computed torque, in the animations
REPLAY_COLOR = "#eb6834"
LOST_COLOR = "#c03c86"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"

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
        OFFSETS[0],
        "start offset by +15 deg in both joints (turned counterclockwise, the elbow more bent)",
    ),
    ("offset_straight", "offsets", OFFSETS[1], "start offset by -10 deg in joint 2 (the elbow straighter)"),
]
ANIMATION_SPAN = (-0.2, 17.0)
ANIMATION_FPS = 10.0
ANIMATION_HOLD_MS = 1500  # the last frame stays this long before the GIF loops

# The videos for slides: each animation's run, with every arm and tracker, named by these.
VIDEOS = storage_root() / "reports" / REPORT.name / "videos"
VIDEO_FPS = 30.0
VIDEO_ARMS = {"esn": "chosen_esn", "replay": "replay"}
VIDEO_TRACKERS = {"pd_w20_z0.1": "joint_pd", "computed_torque_w20_z0.1": "computed_torque"}
VIDEOS_README = """# Videos of report 006's runs, for slides

These MP4s show the runs behind the animations of report 006, *A wider search on
the robot, under disturbances* (`reports/006-wider-search-on-the-robot/` in the
arm-esn-ctrl repository), one arm per video. They are not kept in Git. The take's
videos are report 005's, in `reports/005-tuned-on-the-robot/videos/` here.

All are H.264 at {fps:g} fps, 1104 x 800, rendered by skelarm's player with its side
panel: the time, the joint angles, the tip's position, and the tip's speed. The
purple dot is the target. The videos have no labels: their names say what they
show.

`<scenario>_<arm>_<tracker>.mp4`, {count} in all:

| Name | Scenario |
| --- | --- |
{scenarios}

- **Arms:** `chosen_esn`, driven by the chosen ESN (leak rate 0.03, spectral radius
  1.35, input scaling 1.5, seed 8) from the arm's measured posture; `replay`,
  tracking the take replayed by time.
- **Trackers:** `joint_pd` and `computed_torque`, at omega = 20 rad/s with damping
  ratio 0.1, given a zero reference velocity.
- **Span:** the whole run, from the warm-up at -1 s to the end at 22 s. The side
  panel also shows the external force while a disturbance acts, and the red arrow
  is the force at the tip.

[`videos.csv`](videos.csv) lists each video's scenario, arm, tracker, and the log it
shows, relative to the storage root.

## Reproducing them

From the repository, at commit `{commit}`:

    uv run python reports/006-wider-search-on-the-robot/make_figures.py --videos

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
    figures = {"stage1.png": plot_stage1(), "star.png": plot_star()}
    if args.logs:
        for scenario in ("nominal", "block", "pushes"):
            figures[f"{scenario}_joints.png"], figures[f"{scenario}_torques.png"] = plot_runs(scenario)
        for name, scenario, offset, _ in ANIMATIONS:
            if scenario == "offsets":
                figures[f"{name}_joints.png"], figures[f"{name}_torques.png"] = plot_runs(scenario, offset)
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


def leak_of(path: Path) -> float:
    """The leak rate of a stage 1 sweep run, by its name, such as ...-sweep_robot_wide_lr0.03_filtered."""
    return float(re.search(r"_lr([\d.]+)_filtered", path.name)[1])  # type: ignore[index]


def stage1(name: str) -> list[dict[str, Any]]:
    """The rows of a table of stage 1's sweeps (sweep.csv: each seed; seeds.csv: each setting), with the leak rate."""
    return [
        r | {"leak_rate": leak_of(path)}
        for path in sorted(RESULTS.glob("*-sweep_robot_wide_lr*_filtered"))
        for r in read_csv(path / name)
    ]


def stays(setting: dict[str, Any]) -> bool:
    """A setting of seeds.csv whose seeds all arrive and hold, and are still at the target at the end of the run."""
    return setting["failing_seeds"] == 0 and setting["worst_final_distance_m"] <= GOAL


def plot_stage1() -> Figure:
    """Heatmaps of each setting's median worst path RMSE over its ten seeds, one panel per leak rate.

    A cell's number is the median over the seeds of the worst path RMSE (mm) of the
    two trackers; a cell with a seed that fails, or ends away from the target, is
    marked with an x instead.
    """
    rows = stage1("seeds.csv")
    leaks = sorted({r["leak_rate"] for r in rows})
    inputs = sorted({r["input_scaling"] for r in rows})
    radii = sorted({r["spectral_radius"] for r in rows})
    fig = Figure(figsize=(2.4 * len(leaks) + 1.2, 3.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Stage 1: median over ten seeds of the worst path RMSE from the taught path (mm);"
        " x: a seed fails or ends away from the target",
        color="#0b0b0b",
    )
    axes = fig.subplots(1, len(leaks), sharey=True, squeeze=False)[0]
    held = [1000 * r["median_worst_path_rmse_m"] for r in rows if stays(r)]
    norm = LogNorm(vmin=min(held), vmax=max(held))
    for ax, leak in zip(axes, leaks, strict=True):
        grid = np.full((len(radii), len(inputs)), np.nan)
        for r in rows:
            if r["leak_rate"] != leak:
                continue
            y, x = radii.index(r["spectral_radius"]), inputs.index(r["input_scaling"])
            v = 1000 * r["median_worst_path_rmse_m"]
            if stays(r):
                grid[y, x] = v
                ax.text(
                    x,
                    y,
                    f"{v:.3g}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if norm(v) > 0.6 else "#0b0b0b",
                )
            else:
                ax.text(x, y, "x", ha="center", va="center", fontsize=8, color=TEXT_COLOR)
        ax.imshow(grid, cmap="Blues", norm=norm, origin="lower", aspect="auto")
        ax.set_xticks(range(len(inputs)), [f"{v:g}" for v in inputs], fontsize=7)
        ax.set_yticks(range(len(radii)), [f"{v:g}" for v in radii], fontsize=7)
        ax.set_title(f"leak rate {leak:g}", color="#0b0b0b", fontsize=9)
        ax.set_xlabel("input scaling", color=TEXT_COLOR, fontsize=8)
    axes[0].set_ylabel("spectral radius", color=TEXT_COLOR)
    fig.colorbar(ScalarMappable(norm=norm, cmap="Blues"), ax=axes.tolist(), shrink=0.9, label="mm")
    return fig


def setting_measures(runs: dict[str, Path], values: dict[str, float]) -> list[dict[str, Any]]:
    """Each seed of one setting of a sweep run undisturbed, under the pushes, and under the block.

    ``runs`` gives the sweep run of each scenario ("nominal", "pushes", "block"),
    and ``values`` the setting's swept values other than the seed. Per seed:

    - ``lost``: a pushed run never arrives or ends away from the target;
    - ``failing``: a run fails undisturbed or under the block;
    - ``rmse``: the worst path RMSE undisturbed (mm);
    - ``back1``, ``back3``: under the pushes, when the hand is back on the path after
      the first push and after the push at the target (s, the slower tracker; a hand
      not back by the next push or the end of the run counts as back then);
    - ``lead``: under the block, the reference's lead over the arm at the release (%
      of the path, the larger of the two trackers);
    - ``off``, ``back``: after the release, the farthest from the path (mm), and when
      the hand is back on it (s, the slower tracker).
    """

    def rows(scenario: str, name: str) -> list[dict[str, Any]]:
        return [r for r in read_csv(runs[scenario] / name) if all(r[k] == v for k, v in values.items())]

    nominal, block = rows("nominal", "sweep.csv"), rows("block", "sweep.csv")
    pushed, blocked = rows("pushes", "runs.csv"), rows("block", "runs.csv")
    out = []
    for seed in sorted({r["seed"] for r in nominal}):
        p = [r for r in pushed if r["seed"] == seed]
        b = [r for r in blocked if r["seed"] == seed]
        n = next(r for r in nominal if r["seed"] == seed)
        k = next(r for r in block if r["seed"] == seed)
        back = {
            push: max(r[f"return_time_s_{push}"] if np.isfinite(r[f"return_time_s_{push}"]) else cap for r in p)
            for push, cap in RETURN_CAPS.items()
        }
        out.append(
            {
                "seed": int(seed),
                "lost": any(r["final_distance_m"] > GOAL or not np.isfinite(r["arrival_time_s"]) for r in p),
                "failing": n["failures"] + k["failures"] > 0,
                "rmse": 1000 * n["worst_path_rmse_m"],
                "back1": back[1],
                "back3": back[3],
                "lead": 100 * max(r["reference_lead"] for r in b),
                "off": 1000 * k["worst_off_path_m"],
                "back": k["latest_return_s"],
            }
        )
    return out


def star_runs(line: str) -> dict[str, Path]:
    """The sweep runs of one line of Step B's star, by scenario."""
    return {s: RESULTS / run_name(f"sweep_robot_star_{line}_{s}_filtered") for s in ("nominal", "pushes", "block")}


def star_values(line: str) -> list[float]:
    return sorted({r[LINES[line]] for r in read_csv(star_runs(line)["nominal"] / "sweep.csv")})


def star_setting(line: str, value: float) -> list[dict[str, Any]]:
    """The seeds of one setting of Step B's star."""
    return setting_measures(star_runs(line), {LINES[line]: value})


def plot_star() -> Figure:
    """Step B: every seed of every setting of the star, undisturbed, under the pushes, and under the block.

    One column per line of the star, one row per measure; each seed is a dot,
    seed 8 (the chosen ESN's reservoir) dark, and the median a line; a seed lost
    under the pushes is a cross at the top of the return panel; the dotted line
    marks candidate #2's value.
    """
    measures = [
        ("rmse", "undisturbed:\nworst path RMSE (mm)"),
        ("back3", "pushes: back on the path after\nthe push at the target (s)"),
        ("lead", "block: reference lead\nat the release (% of the path)"),
        ("off", "block: farthest from\nthe path after the release (mm)"),
    ]
    fig = Figure(figsize=(13, 11), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Step B, the star around candidate #2 (dotted): each seed a dot (seed 8 dark), the median a line;"
        " x: a seed lost under the pushes",
        color="#0b0b0b",
    )
    axes = fig.subplots(len(measures), len(LINES))
    for col, (line, key) in enumerate(LINES.items()):
        values = star_values(line)
        for row, (name, label) in enumerate(measures):
            ax = axes[row, col]
            style(ax)
            medians = []
            for x, value in enumerate(values):
                seeds = star_setting(line, value)
                v = np.array([s[name] for s in seeds])
                lost = np.array([s["lost"] for s in seeds]) if name == "back3" else np.zeros(len(seeds), bool)
                jitter = (np.arange(len(seeds)) - (len(seeds) - 1) / 2) * 0.03
                ax.scatter((x + jitter)[~lost], v[~lost], s=12, color="#a3a29d", zorder=2)
                eight = np.array([s["seed"] == 8 for s in seeds]) & ~lost
                ax.scatter((x + jitter)[eight], v[eight], s=16, color="#0b0b0b", zorder=3)
                if lost.any():
                    ax.scatter(
                        (x + jitter)[lost],
                        np.full(lost.sum(), RETURN_CAPS[3]),
                        s=30,
                        marker="x",
                        color=LOST_COLOR,
                        zorder=3,
                    )
                medians.append(np.median(v[~lost]))
            ax.plot(range(len(values)), medians, color=ESN_COLOR, linewidth=1.6, zorder=1)
            ax.axvline(values.index(CENTER[key]), color=ESN_COLOR, linewidth=0.8, linestyle=":")
            ax.set_xticks(range(len(values)), [f"{v:g}" for v in values])
            if row == 0:
                ax.set_title(key.replace("_", " "), color="#0b0b0b")
            if col == 0:
                ax.set_ylabel(label, color=TEXT_COLOR, fontsize=9)
            if row == len(measures) - 1:
                ax.set_xlabel(key.replace("_", " "), color=TEXT_COLOR)
    return fig


def grid_runs(radius: float) -> dict[str, Path]:
    """The sweep runs of the small grid at one spectral radius, by scenario."""
    return {
        s: RESULTS / run_name(f"sweep_robot_small_grid_sr{radius:g}_{s}_filtered")
        for s in ("nominal", "pushes", "block")
    }


def grid_settings() -> dict[tuple[float, float, float], list[dict[str, Any]]]:
    """The seeds of every setting of the small grid, by (spectral radius, leak rate, input scaling)."""
    out = {}
    for radius in (1.2, 1.35):
        runs = grid_runs(radius)
        rows = read_csv(runs["nominal"] / "sweep.csv")
        for leak in sorted({r["leak_rate"] for r in rows}):
            for scaling in sorted({r["input_scaling"] for r in rows}):
                out[radius, leak, scaling] = setting_measures(runs, {"leak_rate": leak, "input_scaling": scaling})
    return out


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
        "\n== 3.6 The block, from the logs: arm, tracker: force at the tip (N) at the grip (peak, first 0.1 s) and at"
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
    print_stage1()
    print_candidates()
    print_step_a()
    print_star()
    print_small_grid()
    print_validation()


def print_stage1() -> None:
    settings, seeds = stage1("seeds.csv"), stage1("sweep.csv")

    def seeds_of(leak: float, radius: float, scaling: float) -> list[dict[str, Any]]:
        group = [
            r
            for r in seeds
            if r["leak_rate"] == leak and r["spectral_radius"] == radius and r["input_scaling"] == scaling
        ]
        return sorted(group, key=lambda r: r["seed"])

    def setting_of_stage1(leak: float, radius: float, scaling: float) -> dict[str, Any]:
        return next(
            r
            for r in settings
            if r["leak_rate"] == leak and r["spectral_radius"] == radius and r["input_scaling"] == scaling
        )

    print("== 3.1 Stage 1: report 005's seed check (stage 4) repeated: largest difference of the worst path RMSE (m)")
    for leak in (0.03, 0.05):
        before = read_csv(
            REPORT_005
            / next(p.name for p in REPORT_005.glob(f"*-sweep_robot_seeds_filtered_lr{leak:g}_sr0.99_is2"))
            / "sweep.csv"
        )
        now = seeds_of(leak, 0.99, 2.0)
        diff = max(
            abs(a["worst_path_rmse_m"] - b["worst_path_rmse_m"])
            for a, b in zip(sorted(before, key=lambda r: r["seed"]), now, strict=True)
        )
        print(f"  leak rate {leak:g}: {diff:.2g}")

    print(
        "\n== 3.1 Settings whose ten seeds all arrive, hold, and stay at the target to the end (of 28 per leak rate);"
    )
    print("the best of each leak rate: median, mean +- SD, min to max of the worst path RMSE (mm), median arrival (s)")
    for leak in sorted({r["leak_rate"] for r in settings}):
        group = [r for r in settings if r["leak_rate"] == leak]
        hold = [r for r in group if r["failing_seeds"] == 0]
        ok = [r for r in group if stays(r)]
        line = f"  leak rate {leak:g}: {len(ok)} ({len(hold)} hold; {len(hold) - len(ok)} of those drift away)"
        if ok:
            top = min(ok, key=lambda r: r["median_worst_path_rmse_m"])
            line += (
                f"; best spectral radius {top['spectral_radius']:g}, input scaling {top['input_scaling']:g}:"
                f" {1000 * top['median_worst_path_rmse_m']:.2f}, {1000 * top['mean_worst_path_rmse_m']:.2f} +-"
                f" {1000 * top['sd_worst_path_rmse_m']:.2f}, {1000 * top['min_worst_path_rmse_m']:.2f} to"
                f" {1000 * top['max_worst_path_rmse_m']:.2f}, {top['median_latest_arrival_s']:.2f}"
            )
        print(line)
    print("  the best eight overall:")
    for r in sorted((r for r in settings if stays(r)), key=lambda r: r["median_worst_path_rmse_m"])[:8]:
        print(
            f"    leak rate {r['leak_rate']:g}, spectral radius {r['spectral_radius']:g}, input scaling"
            f" {r['input_scaling']:g}: {1000 * r['median_worst_path_rmse_m']:.2f},"
            f" SD {1000 * r['sd_worst_path_rmse_m']:.2f}"
        )

    print("\n== 3.1 Q1: at each leak rate and input scaling, the best spectral radius over 1 (all seeds stay) against")
    print("0.99: medians (mm), and the seeds it beats 0.99 for; it helps if its median is lower and it wins 8 of 10")
    for leak in sorted({r["leak_rate"] for r in settings}):
        for scaling in sorted({r["input_scaling"] for r in settings}):
            over = [
                r
                for r in settings
                if r["leak_rate"] == leak and r["input_scaling"] == scaling and r["spectral_radius"] > 1 and stays(r)
            ]
            if not over:
                continue
            best = min(over, key=lambda r: r["median_worst_path_rmse_m"])
            base = setting_of_stage1(leak, 0.99, scaling)
            a = seeds_of(leak, best["spectral_radius"], scaling)
            b = seeds_of(leak, 0.99, scaling)
            wins = sum(x["worst_path_rmse_m"] < y["worst_path_rmse_m"] for x, y in zip(a, b, strict=True))
            helps = best["median_worst_path_rmse_m"] < base["median_worst_path_rmse_m"] and wins >= 8
            print(
                f"  leak rate {leak:g}, input scaling {scaling:g}: spectral radius {best['spectral_radius']:g}"
                f" {1000 * best['median_worst_path_rmse_m']:.2f} against {1000 * base['median_worst_path_rmse_m']:.2f}"
                f"{'' if stays(base) else ' (0.99 fails)'}, {wins}/10{'  <- helps' if helps else ''}"
            )

    print("\n== 3.1 Q2: input scaling 2 against 1.5 at leak rate 0.03, seed by seed: medians (mm), the seeds 2 beats")
    print("1.5 for, and the median gain (mm); 2 was needed if it wins 8 of 10 with a median gain over 0.2 mm")
    for radius in (1.5, 1.35, 1.2, 1.1):
        a, b = seeds_of(0.03, radius, 2.0), seeds_of(0.03, radius, 1.5)
        gain = 1000 * np.array([y["worst_path_rmse_m"] - x["worst_path_rmse_m"] for x, y in zip(a, b, strict=True)])
        print(
            f"  spectral radius {radius:g}:"
            f" {1000 * setting_of_stage1(0.03, radius, 2.0)['median_worst_path_rmse_m']:.2f}"
            f" against {1000 * setting_of_stage1(0.03, radius, 1.5)['median_worst_path_rmse_m']:.2f},"
            f" {int(np.sum(gain > 0))}/10, {np.median(gain):+.2f}"
        )


def tag_values(tag: str) -> tuple[float, float, float, int]:
    """The leak rate, spectral radius, input scaling, and seed in a tag such as lr0.03_sr1.35_is2_seed5."""
    lr, sr, scaling, seed = re.fullmatch(r"lr([\d.]+)_sr([\d.]+)_is([\d.]+)_seed(\d+)", tag).groups()  # type: ignore[union-attr]
    return float(lr), float(sr), float(scaling), int(seed)


def print_candidates() -> None:
    print("\n== 3.2 The candidates tried in the apps: setting and seed; this seed's worst path RMSE on the robot and")
    print("its setting's median (mm); on its own, the starts that arrive and hold, of 49")
    seeds, settings = stage1("sweep.csv"), stage1("seeds.csv")
    for number, (tag, why) in CANDIDATES.items():
        leak, radius, scaling, seed = tag_values(tag)

        def same(r: dict[str, Any]) -> bool:
            return r["leak_rate"] == leak and r["spectral_radius"] == radius and r["input_scaling"] == scaling  # noqa: B023

        mine = next(r for r in seeds if same(r) and r["seed"] == seed)
        median = next(r for r in settings if same(r))["median_worst_path_rmse_m"]
        alone = rows_of(f"grid_{tag}_filtered")
        print(
            f"  #{number} {tag} ({why}): {1000 * mine['worst_path_rmse_m']:.2f}, {1000 * median:.2f};"
            f" {sum(r['success'] for r in alone)}/{len(alone)}"
        )


def print_step_a() -> None:
    print("\n== 3.3 Step A, the kept candidates in report 005's scenarios, CT / PD: undisturbed path RMSE (mm); block:")
    print("reference lead at the release (% of the path); pushes: distance from the target at the end (mm); offsets:")
    print("starts that arrive and hold, of 49")
    for number in KEPT:
        tag = CANDIDATES[number][0]

        def per_setting(scenario: str, column: str, factor: float, tag: str = tag) -> str:
            rows = [r for r in rows_of(f"{scenario}_robot_{tag}_filtered") if r["arm"] == "esn" and r["start"] == 0]
            by = {setting_of(r): r[column] for r in rows}
            return " / ".join(f"{factor * by[s]:.1f}" for s in SETTINGS)

        offsets = [r for r in rows_of(f"offsets_robot_{tag}_filtered") if r["arm"] == "esn"]
        holds = " / ".join(f"{sum(r['success'] for r in offsets if setting_of(r) == s)}" for s in SETTINGS)
        print(
            f"  #{number} {tag}: {per_setting('nominal', 'taught_path_rmse_m', 1000)};"
            f" {per_setting('block', 'reference_lead', 100)};"
            f" {per_setting('pushes', 'final_distance_m', 1000)}; {holds}"
        )


def print_star() -> None:
    print("\n== 3.4 Step B, the star around #2: each setting over its ten seeds: lost and failing seeds | median")
    print("undisturbed worst path RMSE (mm) | back on the path after the push at the target: median / slowest (s) |")
    print("after the first push: median (s), seeds not back by the next push | block: median lead (%), median farthest")
    print("from the path (mm), median back (s)")
    for line, key in LINES.items():
        for value in star_values(line):
            seeds = star_setting(line, value)
            kept = [s for s in seeds if not s["lost"]]
            back3 = np.array([s["back3"] for s in kept])
            slowest = "not back" if back3.max() >= RETURN_CAPS[3] else f"{back3.max():.1f}"
            back1 = np.array([s["back1"] for s in seeds])
            mark = "  <- #2" if value == CENTER[key] else ""
            print(
                f"  {key} {value:g}: lost {[s['seed'] for s in seeds if s['lost']] or '-'},"
                f" failing {[s['seed'] for s in seeds if s['failing']] or '-'} |"
                f" {np.median([s['rmse'] for s in seeds]):.2f} | {np.median(back3):.1f} / {slowest} |"
                f" {np.median(back1):.1f}, {int(np.sum(back1 >= RETURN_CAPS[1]))} |"
                f" {np.median([s['lead'] for s in seeds]):.1f}, {np.median([s['off'] for s in seeds]):.0f},"
                f" {np.median([s['back'] for s in seeds]):.1f}{mark}"
            )


def print_small_grid() -> None:
    settings = grid_settings()
    reference = settings[CENTER["spectral_radius"], CENTER["leak_rate"], CENTER["input_scaling"]]
    ref_back = np.array([s["back3"] for s in reference])
    ref_rmse = np.median([s["rmse"] for s in reference])
    ref_off = np.median([s["off"] for s in reference])
    print(
        "\n== 3.5 The small grid, by the rule: each setting (spectral radius, leak rate, input scaling): lost, failing"
    )
    print("seeds | median worst path RMSE (mm) | after the push at the target: median / slowest (s), seeds back sooner")
    print("than with #2's setting, median gain (s) | block: median farthest from the path (mm) | verdict")
    verdicts = {}
    for setting, seeds in settings.items():
        gain = ref_back - np.array([s["back3"] for s in seeds])
        rmse, off = np.median([s["rmse"] for s in seeds]), np.median([s["off"] for s in seeds])
        conditions = {
            "1": not any(s["lost"] or s["failing"] for s in seeds),
            "2": int(np.sum(gain > 0)) >= 8 and np.median(gain) > 0.3,
            "3": rmse <= ref_rmse + 0.1 and off <= ref_off + 10.0,
        }
        if seeds is reference:
            verdict = "#2's setting"
        elif all(conditions.values()):
            verdict = "replaces #2's"
            verdicts[setting] = float(np.median(gain))
        else:
            verdict = "fails " + ", ".join(k for k, ok in conditions.items() if not ok)
        back = np.array([s["back3"] for s in seeds])
        print(
            f"  {setting}: lost {sum(s['lost'] for s in seeds)}, failing {sum(s['failing'] for s in seeds)} |"
            f" {rmse:.2f} | {np.median(back):.1f} / {back.max():.1f}, {int(np.sum(gain > 0))}/10,"
            f" {np.median(gain):+.2f} | {off:.0f} | {verdict}"
        )
    if verdicts:
        best = max(verdicts, key=lambda s: verdicts[s])
        seeds = settings[best]
        x = np.array([[s["rmse"], s["back3"], s["back1"], s["lead"], s["off"]] for s in seeds])
        iqr = np.subtract(*np.percentile(x, [75, 25], axis=0))
        score = np.sum(np.abs(x - np.median(x, axis=0)) / iqr, axis=1)
        score[[s["lost"] for s in seeds]] = np.inf
        order = np.argsort(score)
        print(
            f"  the winner {best}; its typical seeds, nearest its medians first:"
            f" {[seeds[i]['seed'] for i in order[:3]]}"
        )


def print_validation() -> None:
    print("\n== 3.6 The chosen ESN's parameters (its esn.toml)")
    with (RESULTS / run_name(f"grid_{CHOSEN}_filtered") / "esn.toml").open("rb") as f:
        record = tomllib.load(f)
    print("  " + ", ".join(f"{key} {value}" for key, value in record["esn"].items()))

    def rounded(x: NDArray[np.float64], decimals: int) -> list[float]:
        return np.round(x, decimals).tolist()

    center, half = (np.degrees(record["normalization"][key]) for key in ("center_rad", "half_range_rad"))
    print(f"  normalization: center {rounded(center, 2)} deg, half range {rounded(half, 2)} deg")

    print("\n== 3.6 The trackers' gains, recomputed from omega and the damping ratio as robot_esn.py does, at the")
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

    print("\n== 3.6 Validation: arm, scenario, tracker: arrival (s), holds, path RMSE (mm); block: peak force (N),")
    print("reference lead at the release (% of the path), after it the farthest from the path (mm) and back within")
    print("5 mm (s); offsets: starts that hold, median path RMSE (mm), median peak torque (N m), failed starts")
    for arm, (label, _) in ARMS.items():
        for scenario in ("nominal", "block"):
            for r in rows_of(stem_of(scenario)):
                if r["arm"] != arm:
                    continue
                force = (
                    f", {r['peak_external_force_n']:.2f} N, {100 * r['reference_lead']:.1f}%,"
                    f" {1000 * r['off_path_m']:.0f} mm, {r['return_time_s']:.1f} s"
                    if scenario == "block"
                    else ""
                )
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
            if r["arm"] == arm and offset_of(r["origin"]) in OFFSETS:
                print(
                    f"  {label}, offset {offset_of(r['origin'])}, {SETTINGS[setting_of(r)]}: {r['arrival_time_s']:.2f},"
                    f" {r['success']}, {1000 * r['taught_path_rmse_m']:.1f}, peak torque {r['peak_torque_nm']:.1f}"
                )

    print(
        "\n== 3.7 Pushes: arm, tracker: arrival (s), holds, path RMSE (mm), distance from the target at the end (mm);"
    )
    print("each push: the reference's lead at its end (fraction of the path), the farthest from the path (mm), and")
    print("back within 5 mm (s, before the next push or the end)")
    for r in rows_of(stem_of("pushes")):
        each = " | ".join(
            f"{r[f'reference_lead_{k}']:+.3f}, {1000 * r[f'off_path_m_{k}']:.0f}, {r[f'return_time_s_{k}']:.2f}"
            for k in (1, 2, 3)
        )
        print(
            f"  {ARMS[r['arm']][0]}, {SETTINGS[setting_of(r)]}: {r['arrival_time_s']:.2f}, {r['success']},"
            f" {1000 * r['taught_path_rmse_m']:.1f}, {1000 * r['final_distance_m']:.1f} | {each}"
        )
    if FLEXIBILITY_CSV.exists():
        print(
            "\n== 3.7 Each push, from the logs: the farthest from the taught path (mm), and when the hand settles"
            " within 5 mm (s)"
        )
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
