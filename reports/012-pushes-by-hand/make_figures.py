# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 012.

    uv run python reports/012-pushes-by-hand/make_figures.py
    uv run python reports/012-pushes-by-hand/make_figures.py --logs --animations

From the copies of the recorded logs under ``data/`` and of the runs under
``results/`` (configurations, run records, and metrics), it draws into
``results/summary``:

- ``forces.png``: each log's tip force over time, with its pushes (Section 2);
- ``pushes.png``: after each push, the following, ringing, return, and farthest from
  the taught path of ESN #1, ESN #2, and the replay against the damping ratio, the
  median over the ten pushes with their range (Section 3.4);

and prints the tables of the report. With ``--logs``, it reads the runs' logs, which
are not kept in Git, from the runs under the storage root, and compares each recording
with its rerun by the same ESN and tracker (Section 2). With ``--animations``, it
animates log 001 with ESN #1, ESN #2, and the replay side by side, each rendered by
skelarm's player, at the damping ratio 0.4 given the reference velocity and given a
zero reference velocity (``recorded001_reference.gif``, ``recorded001_zero.gif``).
With ``--videos``, it exports every arm of ARMS with every tracker under every log,
and each recording as made in the app, as MP4s for slides, with the player's side
panel, into VIDEOS under the storage root, since they are not kept in Git.

    uv run python reports/012-pushes-by-hand/make_figures.py --videos
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import subprocess
import sys
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from skelarm import RecordedForce, StateLog, Task

from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path, storage_root

REPORT = Path(__file__).resolve().parent
DATA = REPORT / "data"
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"

LOGS = ("000", "001", "002")
THRESHOLD = 0.5  # N: the force acts above this
MIN_GAP = 0.2  # s: spans closer than this are one push
DURATION = 24.0  # s: the runs' task, after the warm-up
DAMPINGS = (0.2, 0.4, 0.8)
VELOCITIES = {"zero": "zero reference velocity", "reference": "reference velocity"}
RECORDING = ("lr0.03_sr1.2_is1_seed2", "reference", 0.4)  # the arm and tracker the pushes were given to: ESN #2's
# The arms compared: name, the tag of the runs, the arm in them, color, and marker. The replay is the same in every
# ESN's runs; it is read from ESN #1's.
ARMS = [
    ("ESN #1", "lr0.03_sr1.2_is1.5_seed8", "esn", "#2a78d6", "o"),
    ("ESN #2", "lr0.03_sr1.2_is1_seed2", "esn", "#5b2a86", "s"),
    ("replay", "lr0.03_sr1.2_is1.5_seed8", "replay", "#eb6834", "D"),
]
# The measures after each push: the column in metrics.csv, its label, and its scale.
MEASURES = [
    ("following_gain", "following gain", 1.0),
    ("ringing_m", "ringing over the next 2 s (mm)", 1000.0),
    ("return_time_s", "back within 5 mm\nof the path (s)", 1.0),
    ("off_path_m", "farthest from\nthe path (mm)", 1000.0),
]

TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
FORCE_COLOR = "#52514e"
LINESTYLES = {"zero": "-", "reference": "--"}

# The animations, of log 001 at the damping ratio 0.4: per panel, the label, the arm's runs' tag, the arm, and the
# label's color.
ANIMATED_LOG = "001"
ANIMATED_DAMPING = 0.4
PANELS = [
    ("ESN #1 (input scaling 1.5)", ARMS[0][1], "esn", ARMS[0][3]),
    ("ESN #2 (input scaling 1)", ARMS[1][1], "esn", ARMS[1][3]),
    ("replay of the take by time", ARMS[2][1], "replay", ARMS[2][3]),
]
ANIMATION_SPAN = (-0.2, DURATION)
ANIMATION_FPS = 10.0
ANIMATION_HOLD_MS = 1500  # the last frame stays this long before the GIF loops

# The videos for slides: every log, with every arm of ARMS and every tracker, and each recording, named by these.
VIDEOS = storage_root() / "reports" / REPORT.name / "videos"
VIDEO_FPS = 30.0
VIDEO_ARMS = {"ESN #1": "esn1", "ESN #2": "esn2", "replay": "replay"}
# Per log, what it shows: the headings of Sections 3.1 to 3.3, and when each push begins (Section 2).
VIDEO_LOGS = {
    "000": "log 000: four short pushes, from 2.4, 6.1, 8.6, and 14.0 s",
    "001": "log 001: three long pushes, one strong (14 N at its peak), from 2.3, 7.2, and 14.4 s",
    "002": "log 002: three short pushes, one forward, from 2.6, 6.0, and 13.6 s",
}
VIDEOS_README = """# Videos of report 012's runs, for slides

These MP4s show the runs of report 012, *Pushes by hand*
(`reports/012-pushes-by-hand/` in the arm-esn-ctrl repository), one arm per video:
the three recorded logs, each with every arm the report compares and every tracker,
and the recordings themselves. They are not kept in Git. The take's videos are
report 005's, in `reports/005-tuned-on-the-robot/videos/` here.

All are H.264 at {fps:g} fps, 1104 x 800, rendered by skelarm's player with its side
panel: the time, the joint angles, the tip's position, the tip's speed, and the
external force. The red arrow is the force at the tip; the purple dot is the
target. The videos have no labels: their names say what they show.

`<log>_<arm>_<tracker>.mp4`, {count} in all, and `<log>_as_recorded.mp4`, {recordings}:

| Name | Pushes |
| --- | --- |
{logs}

- **The pushes:** in the robot app, ESN #2's arm, with the tracker
  `pd_z0.4_reference_velocity`, was pushed by dragging its tip with the mouse. The
  runs replay each log's tip force by time, so that every arm and tracker meets the
  same force at the same times; replayed, the force no longer reacts to the arm.
- **The recordings:** `<log>_as_recorded.mp4` is the app's own log, ESN #2's arm as
  it was pushed, until the recording ends, at 19.4 to 19.7 s. The app's joint PD
  was then about 0.4% stiffer in joint 1 than the runs'; ESN #2's run with the same
  tracker retraces each recording within 0.45 deg in the joints.
- **Arms:** `esn1`, driven by ESN #1 (leak rate 0.03, spectral radius 1.2, input
  scaling 1.5, seed 8), and `esn2`, by ESN #2 (leak rate 0.03, spectral radius 1.2,
  input scaling 1, seed 2), each from the arm's measured joint angles every 10 ms;
  `replay`, tracking the take replayed by time. Both ESNs are report 011's: 400
  neurons, sparsity 0.05, and a random bias, trained by teacher forcing with ridge
  1e-2 and a 1 s warm-up, without noise, on the take filtered at 2 Hz
  (`results/demonstrations/20261008-170034-reach_manual_v2` here).
- **Trackers:** joint PD at omega = 20 rad/s, `pd_z<damping ratio>_<velocity>`: the
  damping ratios 0.2, 0.4, and 0.8, each given a zero reference velocity
  (`zero_velocity`) or the reference's (`reference_velocity`), with the torque
  computed every 10 ms and held between. The report's animations show log 001 with
  `pd_z0.4_reference_velocity` and `pd_z0.4_zero_velocity`.
- **Span:** the whole run, from the warm-up at -1 s to the end at 24 s, from the
  demonstrated start.

[`videos.csv`](videos.csv) lists each video's scenario, arm, tracker, and the log it
shows, relative to the storage root.

## Reproducing them

From the repository, at commit `{commit}`:

    uv run python reports/012-pushes-by-hand/make_figures.py --videos

It reads the logs under the storage root and overwrites the files here, this
README included, in about four minutes. The runs are deterministic, so the videos
show the same motion.

Exported on {day} at commit `{commit}`{dirty}.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also compare the recordings with their reruns")
    parser.add_argument("--animations", action="store_true", help="also export GIFs from the logs in the storage root")
    parser.add_argument("--videos", action="store_true", help="also export MP4s for slides into the storage root")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {"forces.png": plot_forces(), "pushes.png": plot_pushes()}
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=150)
    print_tables()
    if args.logs:
        print_reruns()
    if args.animations:
        for velocity in VELOCITIES:
            export_animation(velocity)
    if args.videos:
        export_videos()
    print(f"\nWrote {', '.join(figures)} to {SUMMARY}")


# ---------------------------------------------------------------------------- reading the logs and the runs


@functools.cache
def recording(log: str) -> StateLog:
    return StateLog.load(DATA / f"recorded_forces_{log}.sklog.npz")


@functools.cache
def pushes(log: str) -> list[tuple[float, float]]:
    """When each push of a log acts, as the runs found them."""
    return RecordedForce.from_log(recording(log)).spans(THRESHOLD, MIN_GAP)


def force_of(log: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A log's times and tip force."""
    record = recording(log)
    return np.asarray(record.times), record.channel("ext_force")


def direction(angle: float) -> str:
    """A force's direction in words, by its angle (deg) from the reach's direction, counterclockwise.

    The reach runs leftward above the base, so that counterclockwise from it is
    down, toward the base, and clockwise is up, away from it.
    """
    across = "down" if angle > 0 else "up"
    size = abs(angle)
    if size <= 30:
        return "forward"
    if size >= 150:
        return "backward"
    if 60 <= size <= 120:
        return across
    return f"{'forward' if size < 90 else 'backward'} and {across}"


def describe_pushes(log: str) -> list[dict[str, Any]]:
    """Each push of a log: its span, peak and mean force, impulse, direction, and where the recorded tip was."""
    record = recording(log)
    times, force = force_of(log)
    skeleton = record.build_skeleton()
    hand = endpoint_positions(skeleton, record.channel("q"))
    target = Task.from_dict(record.extra["playback"]["task"]).require_target()
    reach = (target - hand[0]) / np.linalg.norm(target - hand[0])
    rows = []
    for onset, end in pushes(log):
        acting = (times > onset - 1e-9) & (times < end - 1e-9)
        mean = force[acting].mean(axis=0)
        angle = float(np.degrees(np.arctan2(reach[0] * mean[1] - reach[1] * mean[0], reach @ mean)))
        first = int(np.flatnonzero(acting)[0])
        travelled = 1.0 - np.linalg.norm(hand[first] - target) / np.linalg.norm(hand[0] - target)
        rows.append(
            {
                "onset": onset,
                "end": end,
                "peak": float(np.linalg.norm(force[acting], axis=1).max()),
                "mean": float(np.linalg.norm(mean)),
                "impulse": float(np.linalg.norm(force[acting], axis=1).sum() * np.median(np.diff(times))),
                "angle": angle,
                "direction": direction(angle),
                "travelled": float(travelled),
            }
        )
    return rows


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


def run_name(stem: str) -> str:
    """The name of the one run of a configuration, by its stem."""
    matches = sorted(p.name for p in RESULTS.glob(f"*-{stem}") if p.is_dir())
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


def stem_of(log: str, tag: str, velocity: str) -> str:
    """A run's configuration stem, by its log, its ESN's tag, and its reference velocity."""
    return f"recorded{log}_robot_{tag}_{velocity}_velocity_filtered"


def run_of(log: str, tag: str, arm: str, velocity: str, damping: float) -> dict[str, Any]:
    """One arm's run with one tracker."""
    rows = read_csv(RESULTS / run_name(stem_of(log, tag, velocity)) / "metrics.csv")
    return next(r for r in rows if r["damping"] == damping and r["arm"] == arm)


def return_cap(log: str, k: int) -> float:
    """How long after push ``k`` (from 1) the return is watched (s): until the next push, or the run's end."""
    spans = pushes(log)
    nxt = spans[k][0] if k < len(spans) else DURATION
    return nxt - spans[k - 1][1]


def measure_of(run: dict[str, Any], key: str, log: str, k: int) -> float:
    """A measure after push ``k`` of a run, scaled; a hand not back on the path counts as back at the window's end."""
    x = float(run[f"{key}_{k}"])
    if key == "return_time_s" and not np.isfinite(x):
        return return_cap(log, k)
    return x * dict((m, s) for m, _, s in MEASURES)[key]


def every_push(name_tag_arm: tuple[str, str, str], velocity: str, damping: float, key: str) -> NDArray[np.float64]:
    """A measure after each of the ten pushes, of one arm with one tracker."""
    _, tag, arm = name_tag_arm
    values = [
        measure_of(run_of(log, tag, arm, velocity, damping), key, log, k)
        for log in LOGS
        for k in range(1, len(pushes(log)) + 1)
    ]
    return np.array(values)


def not_back(name_tag_arm: tuple[str, str, str], velocity: str, damping: float) -> int:
    """How many of the ten pushes an arm is not back on the path after, before the next push or the run's end."""
    _, tag, arm = name_tag_arm
    return sum(
        not np.isfinite(run_of(log, tag, arm, velocity, damping)[f"return_time_s_{k}"])
        for log in LOGS
        for k in range(1, len(pushes(log)) + 1)
    )


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


# ---------------------------------------------------------------------------- figures


def plot_forces() -> Figure:
    """Each log's tip force over time, its pushes shaded and labeled with their mean force and direction."""
    fig = Figure(figsize=(14, 7.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The recorded tip force (its magnitude), pushed by hand while ESN #2 ran with joint PD at 0.4, given the"
        " reference velocity;\nshaded: the pushes, above 0.5 N and joined across lulls under 0.2 s, each with its mean"
        " force and direction",
        color="#0b0b0b",
    )
    axes = fig.subplots(len(LOGS), 1, sharex=True, sharey=True)
    top = 1.15 * max(float(np.linalg.norm(force_of(log)[1], axis=1).max()) for log in LOGS)
    for ax, log in zip(axes, LOGS, strict=True):
        style(ax)
        times, force = force_of(log)
        for k, push in enumerate(describe_pushes(log), start=1):
            ax.axvspan(push["onset"], push["end"], color=DISTURBANCE_COLOR, zorder=0)
            ax.annotate(
                f"push {k}: {push['mean']:.1f} N,\n{push['direction']}",
                xy=(push["end"], push["peak"]),
                xytext=(6, -2),
                textcoords="offset points",
                va="top",
                color=TEXT_COLOR,
                fontsize=9,
            )
        ax.plot(times, np.linalg.norm(force, axis=1), color=FORCE_COLOR, linewidth=1.0, zorder=2)
        ax.set_ylabel(f"log {log}\ntip force (N)", color=TEXT_COLOR)
        ax.set_xlim(-1.0, 20.0)
        ax.set_ylim(0, top)
    axes[-1].set_xlabel("time (s)", color=TEXT_COLOR)
    return fig


def plot_pushes() -> Figure:
    """After each push, each measure of the three arms against the damping ratio: the median over the ten pushes.

    The bars span the ten pushes, from the least to the most; a hand not back on
    the path counts as back at the next push or the run's end.
    """
    fig = Figure(figsize=(15, 7.0), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "After each push by hand, joint PD, the torque every 10 ms: ESN #1 (blue), ESN #2 (purple), and the replay"
        " (orange);\nthe median over the ten pushes of the three logs, the bars from the least to the most",
        color="#0b0b0b",
    )
    axes = fig.subplots(len(VELOCITIES), len(MEASURES), sharex=True, sharey="col")
    shifts = {name: shift for (name, *_), shift in zip(ARMS, (-0.02, 0.0, 0.02), strict=True)}
    for row, (velocity, velocity_name) in enumerate(VELOCITIES.items()):
        for col, (key, label, _) in enumerate(MEASURES):
            ax = axes[row, col]
            style(ax)
            for name, tag, arm, color, marker in ARMS:
                values = [every_push((name, tag, arm), velocity, d, key) for d in DAMPINGS]
                x = np.array(DAMPINGS) + shifts[name]
                medians = [float(np.median(v)) for v in values]
                low = [m - float(v.min()) for m, v in zip(medians, values, strict=True)]
                high = [float(v.max()) - m for m, v in zip(medians, values, strict=True)]
                ax.errorbar(
                    x,
                    medians,
                    yerr=[low, high],
                    color=color,
                    marker=marker,
                    markersize=4,
                    linewidth=1.4,
                    linestyle=LINESTYLES[velocity],
                    elinewidth=1.0,
                    capsize=2,
                    zorder=2 if arm == "replay" else 3,
                )
            ax.set_xticks(DAMPINGS)
            ax.set_ylim(-0.05 if key == "following_gain" else 0, 1.0 if key == "following_gain" else None)
            if row == 0:
                ax.set_title(label.replace("\n", " "), color="#0b0b0b", fontsize=10)
            else:
                ax.set_xlabel("damping ratio", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel(velocity_name, color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=color, marker=marker, markersize=4, linewidth=1.4, label=name)
        for name, _, _, color, marker in ARMS
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- the animations and the videos


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


def export_videos() -> None:
    """Export every log, with every arm of ARMS and every tracker, and each recording, as MP4s for slides into VIDEOS.

    Each video is one log, rendered by skelarm's player at VIDEO_FPS with its side
    panel: each arm's whole run, the warm-up included. A run's video is named
    ``recorded<log>_<arm>_<tracker>.mp4`` (VIDEO_ARMS), such as
    ``recorded001_esn1_pd_z0.4_zero_velocity.mp4``, and a recording's
    ``recorded<log>_as_recorded.mp4``; ``videos.csv`` lists each video's scenario, arm,
    tracker, and the log it shows, and ``README.md`` (VIDEOS_README) explains them
    and how to reproduce them, with the commit they were exported at.
    """
    VIDEOS.mkdir(parents=True, exist_ok=True)
    rows, recordings = [], []
    for log, title in VIDEO_LOGS.items():
        for arm_name, tag, arm, _, _ in ARMS:
            for damping in DAMPINGS:
                for velocity, given in VELOCITIES.items():
                    run = resolve_run_path(f"results/{run_name(stem_of(log, tag, velocity))}")
                    path = run / f"pd_w20_z{damping:g}" / f"{arm}_00.sklog.npz"
                    video = VIDEOS / f"recorded{log}_{VIDEO_ARMS[arm_name]}_pd_z{damping:g}_{velocity}_velocity.mp4"
                    rows.append((video, title, arm_name, f"joint PD, damping ratio {damping:g}, {given}", path))
        recorded = storage_root() / "data" / "recorded_forces" / f"recorded_forces_{log}.sklog.npz"
        tracker = "joint PD, damping ratio 0.4, reference velocity, in the app"
        recordings.append((VIDEOS / f"recorded{log}_as_recorded.mp4", title, "ESN #2", tracker, recorded))
    for video, *_, path in rows + recordings:
        player_export(path, video, VIDEO_FPS, panel=True)
    with (VIDEOS / "videos.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "scenario", "arm", "tracker", "log"])
        for video, title, arm_name, tracker, path in rows + recordings:
            writer.writerow([video.name, title, arm_name, tracker, path.relative_to(storage_root())])
    git = [["git", "rev-parse", "--short", "HEAD"], ["git", "status", "--porcelain"]]
    commit, changes = (subprocess.run(c, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout for c in git)
    readme = VIDEOS_README.format(
        count=len(rows),
        recordings=len(recordings),
        logs="\n".join(f"| `recorded{log}` | {title} |" for log, title in VIDEO_LOGS.items()),
        fps=VIDEO_FPS,
        commit=commit.strip(),
        dirty=" (with uncommitted changes)" if changes.strip() else "",
        day=date.today().isoformat(),
    )
    (VIDEOS / "README.md").write_text(readme)
    print(f"wrote {len(rows) + len(recordings)} videos and their README to {VIDEOS}")


def export_animation(velocity: str) -> None:
    """Animate log 001 into ``recorded001_<velocity>.gif``: the arms of PANELS side by side, by skelarm.

    The player exports each arm's run as a GIF, with the recorded force as a red
    arrow at the tip; its frames, which the GIF merges where the arm rests, are spread
    back over time, cropped to where the arms move, labeled, and tiled into one GIF
    with the task time and the force while a push acts.
    """
    import tempfile

    from PIL import Image, ImageDraw, ImageFont, ImageSequence

    given = "a zero reference velocity" if velocity == "zero" else "the reference velocity"
    title = f"log {ANIMATED_LOG}; joint PD at {ANIMATED_DAMPING:g}, given {given}, the torque every 10 ms"
    recorded = RecordedForce.from_log(recording(ANIMATED_LOG))
    spans = pushes(ANIMATED_LOG)
    frames, starts = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (_, tag, arm, _) in enumerate(PANELS):
            path = resolve_run_path(f"results/{run_name(stem_of(ANIMATED_LOG, tag, velocity))}")
            path = path / f"pd_w20_z{ANIMATED_DAMPING:g}" / f"{arm}_00.sklog.npz"
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
    header, label_height, gap = 44, 32, 8
    canvas_width = len(frames) * width + (len(frames) - 1) * gap

    def header_line(t: float) -> str:
        status = ""
        for k, (begin, end) in enumerate(spans, start=1):
            if begin <= t < end:
                status = f"   push {k}: {np.linalg.norm(recorded(t)):.1f} N"
        return f"{title}   t = {t:+.1f} s{status}"

    # One font for every frame: the largest, up to 20, in which the header fits, with a push or the last time.
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
    out = SUMMARY / f"recorded{ANIMATED_LOG}_{velocity}.gif"
    images[0].save(out, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(f"wrote {out.name}: {len(images)} frames")


# ---------------------------------------------------------------------------- tables


TRACKERS = [(velocity, damping) for velocity in VELOCITIES for damping in DAMPINGS]


def tracker_name(velocity: str, damping: float) -> str:
    return f"ζ {damping:g}, {VELOCITIES[velocity]}"


def print_tables() -> None:
    print_logs()
    for log in LOGS:
        print_log(log)
    print_summary()
    print_completion()


def print_logs() -> None:
    print("\n== The pushes")
    print("| Log | Push | From, to | Length | Peak force | Mean force | Impulse | Direction | Where the tip was |")
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |")
    for log in LOGS:
        for k, p in enumerate(describe_pushes(log), start=1):
            print(
                f"| {log} | {k} | {p['onset']:.2f} to {p['end']:.2f} s | {p['end'] - p['onset']:.2f} s |"
                f" {p['peak']:.1f} N | {p['mean']:.1f} N | {p['impulse']:.1f} N s |"
                f" {p['direction']} ({p['angle']:+.0f}°) | {100 * p['travelled']:.0f}% of the way |"
            )


def print_log(log: str) -> None:
    n = len(pushes(log))
    header = " | ".join(f"Push {k}" for k in range(1, n + 1))
    for key, label, scale in MEASURES:
        arms = ARMS[:2] if key == "following_gain" else ARMS
        print(f"\n== Log {log}: {label.replace(chr(10), ' ')}, {' / '.join(a[0] for a in arms)}")
        print(f"| Tracker | {header} |")
        print(f"| --- |{' --- |' * n}")
        for velocity, damping in TRACKERS:
            cells = []
            for k in range(1, n + 1):
                values = [run_of(log, tag, arm, velocity, damping)[f"{key}_{k}"] for _, tag, arm, _, _ in arms]
                cells.append(" / ".join(fmt(v, scale, key) for v in values))
            print(f"| {tracker_name(velocity, damping)} | {' | '.join(cells)} |")


def fmt(x: float, scale: float, key: str) -> str:
    if not np.isfinite(x):
        return "\N{EN DASH}"
    digits = {"following_gain": 2, "ringing_m": 1, "return_time_s": 2, "off_path_m": 0}[key]
    return f"{x * scale:.{digits}f}"


def print_summary() -> None:
    print("\n== Over the ten pushes: the median (least to most)")
    print("| Tracker | Following, ESN #1 / ESN #2 | Ringing (mm) | Back within 5 mm (s); not back | Farthest (mm) |")
    print("| --- | --- | --- | --- | --- |")
    for velocity, damping in TRACKERS:
        cells = []
        for key, _, _ in MEASURES:
            arms = ARMS[:2] if key == "following_gain" else ARMS
            parts = []
            for name, tag, arm, _, _ in arms:
                v = every_push((name, tag, arm), velocity, damping, key)
                digits = {"following_gain": 2, "ringing_m": 1, "return_time_s": 1, "off_path_m": 0}[key]
                part = f"{np.median(v):.{digits}f} ({v.min():.{digits}f} to {v.max():.{digits}f})"
                if key == "return_time_s":
                    part += f"; {not_back((name, tag, arm), velocity, damping)}"
                parts.append(part)
            cells.append(" / ".join(parts))
        print(f"| {tracker_name(velocity, damping)} | {' | '.join(cells)} |")


def print_completion() -> None:
    print("\n== Completion: arrival (s), * if the hand leaves the goal in its hold; final distance (mm); peak torque")
    for log in LOGS:
        print(f"-- log {log}")
        for velocity, damping in TRACKERS:
            cells = []
            for _, tag, arm, _, _ in ARMS:
                r = run_of(log, tag, arm, velocity, damping)
                mark = "*" if r["left_goal"] else ""
                final = 1000 * r["final_distance_m"]
                cells.append(f"{r['arrival_time_s']:.2f}{mark}, {final:.1f} mm, {r['peak_torque_nm']:.1f} N m")
            print(f"| {tracker_name(velocity, damping)} | {' | '.join(cells)} |")


def print_reruns() -> None:
    """Each recording against its rerun by the arm and tracker it was recorded with (under the storage root)."""
    tag, velocity, damping = RECORDING
    print("\n== Each recording against its rerun by ESN #2, joint PD at 0.4, given the reference velocity")
    for log in LOGS:
        record = recording(log)
        path = resolve_run_path(f"results/{run_name(stem_of(log, tag, velocity))}")
        again = StateLog.load(path / f"pd_w20_z{damping:g}" / "esn_00.sklog.npz")
        n = min(len(again.times), len(record.times))
        if not np.allclose(again.times[:n], record.times[:n]):
            msg = f"log {log}: the rerun's clock is not the recording's"
            raise ValueError(msg)
        same = np.array_equal(again.channel("ext_force")[: n - 1], record.channel("ext_force")[: n - 1])
        dq = float(np.degrees(np.abs(again.channel("q")[:n] - record.channel("q")[:n])).max())
        skeleton = record.build_skeleton()
        tips = endpoint_positions(skeleton, again.channel("q")[:n]) - endpoint_positions(
            skeleton, record.channel("q")[:n]
        )
        print(
            f"log {log}: the same force {same}; the joints differ by {dq:.2f} deg at most,"
            f" the tip by {1000 * float(np.linalg.norm(tips, axis=1).max()):.1f} mm"
        )


if __name__ == "__main__":
    with (RESULTS / run_name(stem_of(LOGS[0], ARMS[0][1], "zero")) / "config.toml").open("rb") as f:
        if tomllib.load(f)["evaluation"]["duration"] != DURATION:
            sys.exit("the runs' duration is not DURATION")
    main()
