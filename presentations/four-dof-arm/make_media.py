# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Videos and plots of the four-link arm's results, for slides.

    uv run python presentations/four-dof-arm/make_media.py

From the take and from step 5's runs of the four-link consistent search (ESN #1 and
ESN #2 against the replay and the progress replay, with the search's tracker;
``experiments/manual_demonstration_four_dof_consistent_search/README.md``), under the
storage root, it writes into ``presentations/four-dof-arm/`` there a folder per scene:

- ``00_take``: the take, as recorded and as filtered, and its joint angles and tip path;
- ``01_nominal`` to ``05_offset_+0_+0_-15_-15``: one run of each arm, undisturbed, under
  the block, under the pushes, and from two offset starts: a video of each arm
  (``esn1.mp4``, ``esn2.mp4``, ``replay.mp4``, ``progress_replay.mp4``), each ESN beside
  both replays (``compare_esn1.mp4``, ``compare_esn2.mp4``), and each ESN's joint angles,
  torques, and tip path against both replays' (``joints_esn1.png``, ``torques_esn1.png``,
  ``tip_xy_esn1.png``, and the same for ESN #2);
- ``06_offsets_all``: the tip paths of each arm from all 36 starts (``tip_xy_esn1.png``,
  ``tip_xy_esn2.png``).

The folder's ``README.md`` explains them, with ``demo/README.md`` (how to run the arms
live in the simulator) as a section, and ``media.csv`` lists every file with the logs it
shows. The videos are rendered by skelarm's player (``--export``); the replays do not
depend on the ESN, and are taken from ESN #1's runs, checked to equal ESN #2's.
``--readme-only`` rewrites the README alone, leaving the media as they are.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from skelarm import Skeleton, StateLog, compute_forward_kinematics

from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path, storage_root

OUT = storage_root() / "presentations" / "four-dof-arm"
DEMO = Path(__file__).parent / "demo" / "README.md"  # the live demonstration, a section of the READMEs
SEARCH = "results/manual_demonstration_four_dof_consistent_search"
TAKE_RUN = "results/demonstrations/20261103-134628-reach_manual_four_dof"
TAKES = {  # the take: its file, relative to the storage root, and what it is
    "take_as_recorded": ("data/reach_manual_four_dof/reach_manual_four_dof.sklog.npz", "the take as recorded by hand"),
    "take_filtered": (f"{TAKE_RUN}/demo_00_filtered.sklog.npz", "the take filtered at 2 Hz: what the ESNs learn"),
}
TRACKER = "pd_w20_z0.4"  # the search's tracker's folder in a run of robot_esn.py
TARGET, RADIUS = np.array([0.0, 0.4]), 0.02  # the goal (m)

# The chosen ESNs of step 5: their key, label, configuration name, and setting.
ESNS = {
    "esn1": (
        "ESN #1",
        "following_esn1_lr0.015_sr1.2_is0.05_seed0",
        "leak rate 0.015, spectral radius 1.2, input scaling 0.05, seed 0",
    ),
    "esn2": (
        "ESN #2",
        "following_esn2_lr0.02_sr1.35_is0.05_seed7",
        "leak rate 0.02, spectral radius 1.35, input scaling 0.05, seed 7",
    ),
}
# The arms of a run: their log's prefix, label, and color (report 018's).
REPLAYS = {"replay": "Replay (by time)", "progress": "Progress replay"}
COLORS = {"esn": "#2a78d6", "replay": "#eb6834", "progress": "#1baf7a"}
ZORDERS = {"esn": 3.0, "progress": 2.2, "replay": 2.0}
VIDEO_NAMES = {"replay": "replay", "progress": "progress_replay"}
# The scenes: folder, scenario, offset start (None: the take's start), and title.
SCENES = [
    ("01_nominal", "nominal", None, "Undisturbed"),
    ("02_block", "block", None, "Blocked: the tip held from 4.0 to 5.0 s"),
    ("03_pushes", "pushes", None, "Pushed: 3 N at 2.6 s, 5 N at 6.7 s, 4 N at 14.9 s"),
    ("04_offset_+15_+15_+15_+15", "offsets", "+15,+15,+15,+15", "From the start offset by +15° in every joint"),
    ("05_offset_+0_+0_-15_-15", "offsets", "+0,+0,-15,-15", "From the start offset by -15° in joints 3 and 4"),
]
ALL_STARTS = "06_offsets_all"

FPS = 30.0
TEXT_COLOR, TITLE_COLOR, GRID_COLOR = "#52514e", "#0b0b0b", "#e1e0d9"
DISTURBANCE_COLOR, TAUGHT_COLOR = "#f0efec", "#b9b7b0"
FIGSIZE, DPI = (12.0, 6.75), 150  # 1800 x 1012 px: 16:9
ALL_STARTS_SIZE = (15.0, 4.6)  # three panels of the plane side by side: 2250 x 690 px
START_CUT = 0.05  # s: from an offset start, the torque's axis leaves out the task's first 50 ms
EVERY = 5  # the plots draw every 5th sample of a run's 2 ms log: every 10 ms


# ---------------------------------------------------------------------------- the runs


def lowercase(text: str) -> str:
    """``text`` with its first letter lowercase: a title inside a sentence."""
    return text[:1].lower() + text[1:]


def run_dir(scenario: str, esn: str) -> Path:
    """Step 5's run of an ESN in a scenario, under the storage root."""
    name = f"{scenario}_robot_{ESNS[esn][1]}"
    (path,) = sorted(resolve_run_path(SEARCH).glob(f"*-{name}"))
    return path


def start_of(scenario: str, esn: str, offset: str | None) -> int:
    """The index of a run's start: 0, the take's, or the one offset by ``offset`` (such as "+15,+15,+15,+15")."""
    if offset is None:
        return 0
    with (run_dir(scenario, esn) / "metrics.csv").open(newline="") as f:
        (start,) = {int(r["start"]) for r in csv.DictReader(f) if r["origin"].endswith(f" {offset} deg")}
    return start


def log_path(scenario: str, esn: str, arm: str, start: int) -> Path:
    return run_dir(scenario, esn) / TRACKER / f"{arm}_{start:02d}.sklog.npz"


def config_of(scenario: str) -> dict[str, Any]:
    with (run_dir(scenario, "esn1") / "config.toml").open("rb") as f:
        return tomllib.load(f)


def disturbances(scenario: str) -> list[tuple[dict[str, Any], tuple[float, float]]]:
    """A scenario's disturbances, each its table and its span (task time, s)."""
    table = config_of(scenario).get("disturbance")
    tables = [] if table is None else [table] if isinstance(table, dict) else table
    return list(zip(tables, disturbance_spans(table), strict=True))


def check_replays_equal(scenarios: dict[str, int] | None = None) -> None:
    """The replays' runs of ESN #1 and ESN #2 are the same in the scenarios, given with their number of starts."""
    for scenario, n_starts in (scenarios or {"nominal": 1, "block": 1, "pushes": 1, "offsets": 36}).items():
        for arm in REPLAYS:
            for start in range(n_starts):
                a, b = (StateLog.load(log_path(scenario, esn, arm, start)) for esn in ESNS)
                if not np.array_equal(a.channel("q"), b.channel("q")):
                    msg = f"the {arm} runs of {scenario}, start {start}, differ between the ESNs' runs"
                    raise ValueError(msg)


class Run:
    """A run's log, every 10 ms: its times, joint angles and reference (deg), torques, and tip path."""

    def __init__(self, path: Path) -> None:
        log = StateLog.load(path)
        n = len(log.times)
        names = log.channel_names
        self.path = path
        self.t = log.times[::EVERY]
        self.q = np.degrees(log.channel("q").reshape(n, -1))[::EVERY]
        self.q_ref = np.degrees(log.channel("q_ref").reshape(n, -1))[::EVERY] if "q_ref" in names else None
        self.tau = log.channel("tau").reshape(n, -1)[::EVERY] if "tau" in names else None
        self.skeleton = log.build_skeleton()
        self.tip = endpoint_positions(self.skeleton, np.radians(self.q))


def posture(skeleton: Skeleton, q_deg: NDArray[np.float64]) -> NDArray[np.float64]:
    """The arm's joints and tip at joint angles ``q_deg``, from the base: (x, y) per point."""
    for link, angle in zip(skeleton.links[1:], np.radians(q_deg), strict=True):
        link.q = float(angle)
    compute_forward_kinematics(skeleton)
    return np.array([[0.0, 0.0], *[[link.xe, link.ye] for link in skeleton.links]])


# ---------------------------------------------------------------------------- the plots


def style(ax: Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(visible=True, color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)


def figure(title: str, size: tuple[float, float] = FIGSIZE) -> Figure:
    fig = Figure(figsize=size, dpi=DPI, facecolor="white", layout="constrained")
    fig.suptitle(title, color=TITLE_COLOR, fontsize=15)
    return fig


def arm_handles(esn: str, *, references: bool) -> list[Line2D]:
    handles = [Line2D([], [], color=COLORS["esn"], linewidth=2.0, label=ESNS[esn][0])]
    handles += [Line2D([], [], color=COLORS[arm], linewidth=2.0, label=label) for arm, label in REPLAYS.items()]
    if references:
        handles += [
            Line2D([], [], color=TEXT_COLOR, linewidth=1.4, linestyle="--", label="reference (ESN, progress replay)"),
            Line2D([], [], color=TAUGHT_COLOR, linewidth=5.0, label="the take"),
        ]
    return handles


def series_plot(
    runs: dict[str, Run], esn: str, title: str, channel: str, spans: list[tuple[float, float]], taught: Run, cut: float
) -> Figure:
    """Each joint's angle or torque over time: the ESN's over both replays', the disturbances shaded."""
    fig = figure(title)
    axes = fig.subplots(2, 2, sharex=True).ravel()
    for joint, ax in enumerate(axes):
        style(ax)
        for onset, end in spans:
            ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
        if channel == "joints":
            ax.plot(taught.t, taught.q[:, joint], color=TAUGHT_COLOR, linewidth=5.0, zorder=1)
        lows, highs = [], []
        for arm, run in runs.items():
            values = run.q if channel == "joints" else run.tau
            assert values is not None
            ax.plot(run.t, values[:, joint], color=COLORS[arm], linewidth=1.6, zorder=ZORDERS[arm])
            if channel == "joints" and arm != "replay" and run.q_ref is not None:
                ax.plot(
                    run.t, run.q_ref[:, joint], color=COLORS[arm], linewidth=1.2, linestyle="--", zorder=ZORDERS[arm]
                )
            shown = values[run.t >= cut, joint]
            lows.append(shown.min())
            highs.append(shown.max())
        if channel == "torques" and cut > 0:  # from an offset start, the first instants' torque would crush the rest
            pad = 0.08 * (max(highs) - min(lows))
            ax.set_ylim(min(lows) - pad, max(highs) + pad)
        ax.set_xlim(runs["esn"].t[0], runs["esn"].t[-1])
        ax.set_ylabel(f"joint {joint + 1} ({'deg' if channel == 'joints' else 'N m'})", color=TEXT_COLOR, fontsize=12)
        if joint >= 2:
            ax.set_xlabel("time (s)", color=TEXT_COLOR, fontsize=12)
    handles = arm_handles(esn, references=channel == "joints")
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=12)
    return fig


def draw_goal(ax: Axes) -> None:
    ax.add_patch(Circle(tuple(TARGET), RADIUS, facecolor="none", edgecolor="#8a2be2", linewidth=1.5, zorder=4))
    ax.plot(*TARGET, marker="o", markersize=4, color="#8a2be2", zorder=4)


def tip_plot(runs: dict[str, Run], esn: str, title: str, taught: Run) -> Figure:
    """The tip paths in the plane, with the take's, the arm's start posture, and the goal."""
    fig = figure(title)
    ax = fig.subplots()
    style(ax)
    first = runs["esn"]
    ax.plot(*posture(first.skeleton, first.q[0]).T, color="#9a9890", linewidth=3.0, marker="o", markersize=5, zorder=1)
    ax.plot(*taught.tip.T, color=TAUGHT_COLOR, linewidth=6.0, zorder=1.5)
    for arm, run in runs.items():
        ax.plot(*run.tip.T, color=COLORS[arm], linewidth=1.8, zorder=ZORDERS[arm])
    ax.plot(*first.tip[0], marker="o", markersize=7, color=TITLE_COLOR, zorder=5)
    draw_goal(ax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel("y (m)", color=TEXT_COLOR, fontsize=12)
    handles = [
        *arm_handles(esn, references=False),
        Line2D([], [], color=TAUGHT_COLOR, linewidth=5.0, label="the take"),
        Line2D([], [], color="#9a9890", linewidth=3.0, marker="o", label="start posture"),
        Line2D([], [], color="#8a2be2", marker="o", linestyle="none", markerfacecolor="none", label="goal (20 mm)"),
    ]
    fig.legend(handles=handles, loc="outside right center", frameon=False, fontsize=12)
    return fig


def passes(row: dict[str, str]) -> bool:
    """Whether a run passes by the search's rule: it arrives, holds for 2 s, and ends within the goal."""
    return (
        np.isfinite(float(row["arrival_time_s"]))
        and float(row["final_distance_m"]) <= RADIUS
        and row["left_goal"] == "False"
        and row["success"] == "True"
        and float(row["hold_observed_s"]) >= 2.0 - 1e-9
    )


def all_starts_plot(esn: str, taught: Run, scenario: str = "offsets") -> tuple[Figure, list[Path]]:
    """Each arm's tip paths from a scenario's starts, a panel per arm, each titled with how many reach and hold."""
    label, _, setting = ESNS[esn]
    with (run_dir(scenario, esn) / "metrics.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    n_starts = len({r["start"] for r in rows})
    fig = figure(f"From all {n_starts} starts: {label} ({setting}) and both replays", ALL_STARTS_SIZE)
    axes = fig.subplots(1, 3, sharex=True, sharey=True)
    logs = []
    for ax, (arm, name) in zip(axes, [("esn", label), *REPLAYS.items()], strict=True):
        style(ax)
        ax.plot(*taught.tip.T, color=TAUGHT_COLOR, linewidth=5.0, zorder=1)
        starts = sorted(int(r["start"]) for r in rows if r["arm"] == arm)
        for start in starts:
            path = log_path(scenario, esn, arm, start)
            logs.append(path)
            run = Run(path)
            ax.plot(*run.tip.T, color=COLORS[arm], linewidth=1.0, alpha=0.7, zorder=2)
            ax.plot(*run.tip[0], marker="o", markersize=3.5, color=TITLE_COLOR, zorder=3)
        succeeded = sum(passes(r) for r in rows if r["arm"] == arm)
        ax.set_title(f"{name}: {succeeded} of {len(starts)} reach and hold", color=TITLE_COLOR, fontsize=13)
        draw_goal(ax)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)", color=TEXT_COLOR, fontsize=12)
    axes[0].set_ylabel("y (m)", color=TEXT_COLOR, fontsize=12)
    return fig, logs


def take_plots(recorded: Run, filtered: Run) -> tuple[Figure, Figure]:
    joints = figure("The take: joint angles (gray: as recorded; black: filtered at 2 Hz, what the ESNs learn)")
    for joint, ax in enumerate(joints.subplots(2, 2, sharex=True).ravel()):
        style(ax)
        ax.plot(recorded.t, recorded.q[:, joint], color=TAUGHT_COLOR, linewidth=3.0)
        ax.plot(filtered.t, filtered.q[:, joint], color=TITLE_COLOR, linewidth=1.4)
        ax.set_ylabel(f"joint {joint + 1} (deg)", color=TEXT_COLOR, fontsize=12)
        if joint >= 2:
            ax.set_xlabel("time (s)", color=TEXT_COLOR, fontsize=12)
    tip = figure("The take: the tip's path, from the start posture to the goal")
    ax = tip.subplots()
    style(ax)
    ax.plot(*posture(filtered.skeleton, filtered.q[0]).T, color="#9a9890", linewidth=3.0, marker="o", markersize=5)
    ax.plot(*posture(filtered.skeleton, filtered.q[-1]).T, color="#cfcdc6", linewidth=3.0, marker="o", markersize=5)
    ax.plot(*recorded.tip.T, color=TAUGHT_COLOR, linewidth=4.0)
    ax.plot(*filtered.tip.T, color=TITLE_COLOR, linewidth=1.4)
    draw_goal(ax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel("y (m)", color=TEXT_COLOR, fontsize=12)
    handles = [
        Line2D([], [], color=TAUGHT_COLOR, linewidth=4.0, label="as recorded"),
        Line2D([], [], color=TITLE_COLOR, linewidth=1.4, label="filtered at 2 Hz"),
        Line2D([], [], color="#9a9890", linewidth=3.0, marker="o", label="start posture"),
        Line2D([], [], color="#cfcdc6", linewidth=3.0, marker="o", label="end posture"),
        Line2D([], [], color="#8a2be2", marker="o", linestyle="none", markerfacecolor="none", label="goal (20 mm)"),
    ]
    tip.legend(handles=handles, loc="outside right center", frameon=False, fontsize=12)
    return joints, tip


# ---------------------------------------------------------------------------- the videos


def player_export(log: Path, out: Path, *, panel: bool) -> None:
    """Render a log into an MP4 with skelarm's player, without a window, with or without its side panel."""
    player = REPO_ROOT / "third_party" / "skelarm" / "tools" / "player.py"
    command = [sys.executable, str(player), str(log), "--export", str(out), "--fps", f"{FPS:g}"]
    subprocess.run(
        command + (["--panel"] if panel else []),
        check=True,
        capture_output=True,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen"},
    )


def status(t: float, scenario: str) -> str:
    for table, (onset, end) in disturbances(scenario):
        if onset <= t < end:
            return f"pushed, {table['force']:g} N" if table["type"] == "push" else "blocked"
    return ""


def compare_video(out: Path, title: str, scenario: str, panels: list[tuple[str, str, Path]], t0: float) -> None:
    """Tile the arms' canvas-only videos side by side, cropped to where they move, under a title and the time."""
    readers: list[Any] = [imageio.get_reader(path) for _, _, path in panels]
    counts = [r.count_frames() for r in readers]
    if len(set(counts)) != 1:
        msg = f"the videos of {out.name} have {counts} frames"
        raise ValueError(msg)
    ink = None
    for reader in readers:
        for frame in reader:
            mask = (frame < 235).any(axis=2)
            ink = mask if ink is None else ink | mask
    assert ink is not None
    rows, cols = np.nonzero(ink)
    margin = 30
    top, bottom = max(int(rows.min()) - margin, 0), min(int(rows.max()) + margin, ink.shape[0])
    left, right = max(int(cols.min()) - margin, 0), min(int(cols.max()) + margin, ink.shape[1])
    width, height = right - left, bottom - top
    header, label_height, gap = 56, 40, 12
    canvas_width = len(panels) * width + (len(panels) - 1) * gap
    canvas_size = (-(-canvas_width // 16) * 16, -(-(header + label_height + height) // 16) * 16)
    font, small = ImageFont.load_default(size=26), ImageFont.load_default(size=22)
    readers = [imageio.get_reader(path) for _, _, path in panels]
    writer: Any = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    with writer:
        for k, frames in enumerate(zip(*readers, strict=True)):
            t = t0 + k / FPS
            canvas = Image.new("RGB", canvas_size, "white")
            draw = ImageDraw.Draw(canvas)
            now = status(t, scenario)
            draw.text((14, 14), f"{title}     t = {t:+.1f} s", fill=TITLE_COLOR, font=font)
            if now:
                draw.text((canvas_width - 14, 14), now, fill="#c0392b", font=font, anchor="ra")
            for i, ((label, color, _), frame) in enumerate(zip(panels, frames, strict=True)):
                x = i * (width + gap)
                canvas.paste(Image.fromarray(frame[top:bottom, left:right]), (x, header + label_height))
                draw.text((x + 10, header + 6), label, fill=color, font=small)
            writer.append_data(np.asarray(canvas))


# ---------------------------------------------------------------------------- the whole


README = """\
# The four-link arm's results, for slides

Videos and plots of the four-link consistent search's chosen ESNs on the robot, against
the two replays: step 5 of `experiments/manual_demonstration_four_dof_consistent_search/`
in the arm-esn-ctrl repository (its README.md gives the protocol). They are not kept in
Git; `presentations/four-dof-arm/make_media.py` there makes them.

## What they show

- **The arm:** skelarm's four-link arm (links of 0.1, 0.25, 0.25, and 0.15 m on a
  0.05 m base, 3 kg in all) on a horizontal plane, from the start posture
  (-20, 10, 60, 50) deg to the goal at (0, 0.4) m, 20 mm in radius (the purple circle).
- **The take:** one reach taught by hand, 21.4 s (`00_take`); the ESNs learn it filtered
  at 2 Hz, and the replays replay it.
- **The arms**, each from the arm's measured joint angles every 10 ms:
{arms}
  - **Replay (by time)**, orange: the take replayed by time, whatever the arm does.
  - **Progress replay**, green: the take replayed by time, but never more than 2.5 deg
    along its joint path ahead of the arm's progress, so that it waits for the arm.
  Both ESNs have 400 neurons, sparsity 0.05, and a random bias, trained by teacher
  forcing with ridge 1e-2 and a 1 s warm-up; the rule of step 5, which asks the
  reference to follow the arm, chose them.
- **The tracker** of every arm: joint PD at omega = 20 rad/s and a damping ratio of 0.4,
  its gains the arm's whole mass matrix at the take's end, given the reference's
  velocity, its torque computed every 10 ms and held between.
- **Time:** every run holds its start for the 1 s warm-up (t < 0); the task runs from
  t = 0 to 23.9 s.

## The folders

| Folder | Scene |
| --- | --- |
| `00_take` | the take: `take_as_recorded.mp4`, `take_filtered.mp4`, `joints.png`, `tip_xy.png` |
{scenes}
| `06_offsets_all` | the tip paths of each arm from all 36 offset starts (every combination of -15, 0, and 15 deg in each joint within the joint limits): `tip_xy_esn1.png`, `tip_xy_esn2.png` |

In each of `01` to `05`:

| File | What it shows |
| --- | --- |
| `esn1.mp4`, `esn2.mp4`, `replay.mp4`, `progress_replay.mp4` | one arm, with the player's side panel (the time, the joint angles, the tip, and the external force); the red arrow is the force at the tip |
| `compare_esn1.mp4`, `compare_esn2.mp4` | ESN #1 (or #2), the replay, and the progress replay side by side, with the time and the disturbance acting |
| `joints_esn1.png`, `joints_esn2.png` | each joint's angle over time: the ESN over both replays, the take thick gray, the ESN's and the progress replay's references dashed, the disturbances shaded |
| `torques_esn1.png`, `torques_esn2.png` | each joint's torque over time, likewise{torque_note} |
| `tip_xy_esn1.png`, `tip_xy_esn2.png` | the tip paths in the plane, with the take's, the start posture, and the goal |

The replays' runs do not depend on the ESN: ESN #1's and ESN #2's runs give the same
replays, and their videos are made once. All videos are H.264 at {fps:g} fps, rendered by
skelarm's player; the plots are 1800 x 1012 px (16:9), but for `06_offsets_all`'s, 2250 x 690 px. [`media.csv`](media.csv) lists
every file with the logs it shows, relative to the storage root.

## Live in the simulator

From the arm-esn-ctrl repository, as its `presentations/four-dof-arm/demo/README.md` says:

{demo}

## Reproducing them

From the repository, at commit `{commit}`{dirty}:

    uv run python presentations/four-dof-arm/make_media.py

It reads the runs under the storage root and overwrites the files here, this README
included; with `--readme-only`, it rewrites this README alone. This README was written
on {day}.
"""  # noqa: E501


def live_demo() -> str:
    """``demo/README.md`` as a section of a README: without its title, its headings one level down."""
    lines = DEMO.read_text().splitlines()[1:]
    return "\n".join(f"#{line}" if line.startswith("#") else line for line in lines).strip()


def commit_state() -> tuple[str, str]:
    """The repository's commit, and `` (with uncommitted changes)`` if it has any, else nothing."""
    git = [["git", "rev-parse", "--short", "HEAD"], ["git", "status", "--porcelain"]]
    commit, changes = (subprocess.run(c, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout for c in git)
    return commit.strip(), " (with uncommitted changes)" if changes.strip() else ""


def write_readme() -> None:
    """The folder's README.md, naming the commit of the code and the day."""
    commit, dirty = commit_state()
    arms = "\n".join(f"  - **{label}**, blue: {setting}." for label, _, setting in ESNS.values())
    scenes = "\n".join(f"| `{name}` | {title} |" for name, _, _, title in SCENES)
    (OUT / "README.md").write_text(
        README.format(
            arms=arms,
            scenes=scenes,
            torque_note=(
                " (from an offset start, the axis leaves out the task's first 50 ms, whose peak would crush the rest)"
            ),
            fps=FPS,
            demo=live_demo(),
            commit=commit,
            dirty=dirty,
            day=date.today().isoformat(),
        )
    )


def export_scenes(
    out: Path,
    scenes: list[tuple[str, str, str | None, str]],
    taught: Run,
    add: Callable[[str, str, str, list[Path]], None],
    exports: list[tuple[Path, Path, bool]],
) -> None:
    """Each scene's videos and plots, in a folder of ``out``; ``exports`` (log, video, panel) are rendered too."""
    compares = []
    with tempfile.TemporaryDirectory() as tmp:
        for folder_name, scenario, offset, title in scenes:
            folder = out / folder_name
            folder.mkdir(exist_ok=True)
            start = start_of(scenario, "esn1", offset)
            spans = [span for _, span in disturbances(scenario)]
            canvas = {}
            for arm in REPLAYS:
                log = log_path(scenario, "esn1", arm, start)
                exports.append((log, folder / f"{VIDEO_NAMES[arm]}.mp4", True))
                add(folder_name, f"{VIDEO_NAMES[arm]}.mp4", f"{REPLAYS[arm]}: {lowercase(title)}", [log])
                canvas[arm] = Path(tmp) / f"{folder_name}_{arm}.mp4"
                exports.append((log, canvas[arm], False))
            for esn, (label, _, setting) in ESNS.items():
                assert start_of(scenario, esn, offset) == start
                log = log_path(scenario, esn, "esn", start)
                exports.append((log, folder / f"{esn}.mp4", True))
                add(folder_name, f"{esn}.mp4", f"{label} ({setting}): {lowercase(title)}", [log])
                canvas[esn] = Path(tmp) / f"{folder_name}_{esn}.mp4"
                exports.append((log, canvas[esn], False))
                panels = [(label, COLORS["esn"], canvas[esn])] + [(REPLAYS[a], COLORS[a], canvas[a]) for a in REPLAYS]
                logs = [log_path(scenario, esn, arm, start) for arm in ("esn", *REPLAYS)]
                add(
                    folder_name,
                    f"compare_{esn}.mp4",
                    f"{label}, the replay, and the progress replay: {lowercase(title)}",
                    logs,
                )
                runs = {arm: Run(path) for arm, path in zip(("esn", *REPLAYS), logs, strict=True)}
                compares.append((folder / f"compare_{esn}.mp4", title, scenario, panels, float(runs["esn"].t[0])))
                cut = START_CUT if offset is not None else -np.inf
                against = f"{label} ({setting}) against the replay and the progress replay"
                for channel, what in (("joints", "Joint angles"), ("torques", "Joint torques")):
                    fig = series_plot(runs, esn, f"{what}, {lowercase(title)}\n{against}", channel, spans, taught, cut)
                    fig.savefig(folder / f"{channel}_{esn}.png")
                    add(
                        folder_name, f"{channel}_{esn}.png", f"{what.lower()} over time, {label} and both replays", logs
                    )
                tip_plot(runs, esn, f"Tip paths, {lowercase(title)}\n{against}", taught).savefig(
                    folder / f"tip_xy_{esn}.png"
                )
                add(folder_name, f"tip_xy_{esn}.png", f"the tip paths, {label} and both replays", logs)
            print(f"plotted {folder_name}")
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda e: player_export(e[0], e[1], panel=e[2]), exports))
        print(f"exported {len(exports)} videos")
        for video, title, scenario, panels, t0 in compares:
            compare_video(video, title, scenario, panels, t0)
            print(f"composed {video.parent.name}/{video.name}")


def export_all() -> None:
    check_replays_equal()
    OUT.mkdir(parents=True, exist_ok=True)
    media: list[dict[str, str]] = []
    root = storage_root()

    def add(folder: str, name: str, shows: str, logs: list[Path]) -> None:
        media.append(
            {"folder": folder, "file": name, "shows": shows, "logs": "; ".join(str(p.relative_to(root)) for p in logs)}
        )

    exports: list[tuple[Path, Path, bool]] = []  # (log, video, panel)
    taught = Run(root / TAKES["take_filtered"][0])

    # The take.
    folder = OUT / "00_take"
    folder.mkdir(exist_ok=True)
    for name, (path, shows) in TAKES.items():
        exports.append((root / path, folder / f"{name}.mp4", True))
        add(folder.name, f"{name}.mp4", shows, [root / path])
    recorded = Run(root / TAKES["take_as_recorded"][0])
    for name, fig in zip(("joints", "tip_xy"), take_plots(recorded, taught), strict=True):
        fig.savefig(folder / f"{name}.png")
        add(
            folder.name,
            f"{name}.png",
            f"the take's {'joint angles' if name == 'joints' else 'tip path'}",
            [recorded.path, taught.path],
        )

    # The scenes.
    export_scenes(OUT, SCENES, taught, add, exports)

    folder = OUT / ALL_STARTS
    folder.mkdir(exist_ok=True)
    for esn in ESNS:
        fig, logs = all_starts_plot(esn, taught)
        fig.savefig(folder / f"tip_xy_{esn}.png")
        add(folder.name, f"tip_xy_{esn}.png", f"tip paths from all 36 starts, {ESNS[esn][0]} and both replays", logs)

    with (OUT / "media.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(media[0]))
        writer.writeheader()
        writer.writerows(media)
    write_readme()
    print(f"wrote {len(media)} files and their README to {OUT}")


def readme_only(description: str | None) -> bool:
    """Whether the command line of the script ``description`` describes asks for the README alone."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--readme-only", action="store_true", help="rewrite README.md alone, leaving the media")
    return bool(parser.parse_args().readme_only)


if __name__ == "__main__":
    if readme_only(__doc__):
        write_readme()
        print(f"wrote the README to {OUT}")
    else:
        export_all()
