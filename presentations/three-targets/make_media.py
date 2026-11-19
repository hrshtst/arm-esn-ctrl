# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Videos and figures of report 025, three targets by context, for slides.

    uv run python presentations/three-targets/make_media.py [--workers 8] [--only FOLDER ...]

From the takes and the robot runs of report 025
(``reports/025-three-targets-by-context/README.md``, Sections 2 and 3.2 to 3.6), under
the storage root, it writes into ``presentations/three-targets/`` there a folder per
section:

- ``00_takes``: the takes and the start postures, and each of the 27 takes at five and
  ten times its speed, alone and one after another;
- ``3.2_undisturbed``: the hand paths of ESN #1 and ESN #2, and every undisturbed run of
  the two ESNs and of the replay with each context, from the takes' starts and from the
  36 unseen starts, alone and one after another (Sections 3.2 and 3.3);
- ``3.4_change_at_rest``: the report's figures of the changes of context, and every run of
  the two ESNs whose context changes at 16 s, alone and nine after another;
- ``3.5_change_on_the_way``: the same with the change at 4 s;
- ``3.6_demonstrations``: the two runs pushed by hand in the robot app while the target
  changes, at one, five, and ten times their speed.

The figures are the report's, drawn by its ``make_figures.py`` at a higher resolution. A
video is a log in skelarm's player (``--export --panel``), with the three targets drawn
and the one the context names emphasised, and the speed and what it shows written in the
side panel. Each run's or take's video is kept under ``parts/``; the videos one after
another join them without encoding them again. The folder's ``README.md`` explains them,
and ``media.csv`` lists every file with the logs it shows. The replays do not depend on
the ESN: they are taken from ESN #1's runs. ``--workers`` sets the number of processes
that render; ``--only`` renders the named folders alone, while the README and
``media.csv`` are always written in full. It overwrites its own files and deletes
nothing.
"""

from __future__ import annotations

import argparse
import csv
import functools
import importlib.util
import multiprocessing
import os
import subprocess
import sys
import tempfile
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont
from skelarm import StateLog

from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path, storage_root

OUT = storage_root() / "presentations" / "three-targets"
PLAYER = REPO_ROOT / "third_party" / "skelarm" / "tools" / "player.py"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Report 025's figure script, whose helpers find the runs and whose functions draw the figures.
_SPEC = importlib.util.spec_from_file_location(
    "make_figures_025", REPO_ROOT / "reports" / "025-three-targets-by-context" / "make_figures.py"
)
assert _SPEC is not None and _SPEC.loader is not None
report: Any = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)

with report.ARM_CONFIG.open("rb") as _f:
    _ARM = tomllib.load(_f)
TASK = _ARM["task"]  # the three targets, as the player draws them
LABELS: tuple[str, ...] = report.LABELS

FOLDERS = ("00_takes", "3.2_undisturbed", "3.4_change_at_rest", "3.5_change_on_the_way", "3.6_demonstrations")
TIMING_FOLDERS = {"at rest": FOLDERS[2], "on the way": FOLDERS[3]}
# The arms of the undisturbed videos: their label, the number of the robot run they are taken from, and their arm.
ARMS = {"esn1": ("ESN #1", "1", "esn"), "esn2": ("ESN #2", "2", "esn"), "replay": ("replay", "1", "replay")}
FPS = 30.0
SPEEDS = (5, 10)  # the speeds of the takes' and the robot runs' videos
DEMONSTRATION_SPEEDS = (1, 5, 10)
DPI = 200  # the figures', where the report draws them at 90
# imageio's quality of the videos with their text: CRF 10 of libx264. Its top, 10, is lossless in the profile
# High 4:4:4 Predictive, which PowerPoint does not play.
QUALITY = 8
# The text in the side panel, which starts at x = 800 px; its readouts end above y = 250 px.
TEXT_COLOR = "#0b0b0b"
TEXT_X, SPEED_Y, LINES_Y, LINE_STEP = 816, 270, 400, 42  # px
SPEED_SIZE, LINE_SIZE = 110, 30  # px: "take 001's start" fits in the panel's 304 px


class Medium(NamedTuple):
    """A file of the output: its path in the output folder, what it shows, the logs it shows, and how it is made."""

    path: str
    shows: str
    logs: list[Path]
    make: functools.partial[None]  # called with ``out``, the file to write


# ---------------------------------------------------------------------------- the runs


def take_log(label: str, index: int) -> Path:
    """A take to a target filtered at 2 Hz, what the ESNs learned, under the storage root."""
    run = report.DEMO_RUNS[label].name
    return resolve_run_path(f"results/demonstrations/{run}/demo_{index:02d}_filtered.sklog.npz")


def take_name(label: str, index: int) -> str:
    """A take's own name, as recorded: ``take 001``."""
    return Path(report.take_rows(label, True)[index]["take"]).name.split(".")[0].replace("_", " ")


def robot_log(stem: str, arm: str, task: str, start: int) -> Path:
    """One arm's log with one target's context from a robot run, under the storage root."""
    return report.stored(stem) / task / report.TRACKER / f"{arm}_{start:02d}.sklog.npz"


def start_names(task: str, origin: str) -> tuple[str, str]:
    """A run's start as the end of a file name and as a line of text: ``take_start_00`` and "take 001's start" for
    the start of the take demo_00 to ``task``, ``unseen_07`` and "unseen 7" for an unseen start."""
    if origin.startswith("demo_"):
        index = int(origin.split("_")[1])
        return f"take_start_{index:02d}", f"{take_name(task, index)}'s start"
    return f"unseen_{int(origin.split()[1]):02d}", origin


def demonstration_log(name: str) -> Path:
    """The run of an ESN pushed by hand in the robot app, under the storage root (the report keeps a copy)."""
    return storage_root() / "data" / "recorded_forces" / report.DEMONSTRATIONS[name].name


# ---------------------------------------------------------------------------- the files


def report_figure(name: str, *args: str, out: Path) -> None:
    """One of report 025's figures, drawn by its function ``name`` and saved at ``DPI``."""
    getattr(report, name)(*args).savefig(out, dpi=DPI)


def with_targets(log: Path, target: str, *, out: Path) -> None:
    """Save a copy of a log for the player that draws the three targets and emphasises the one named at each step.

    The copy's ``extra.playback.task`` is the three targets, and its channel ``active_target`` is the target its
    ``context`` names at each step, the largest of its columns, which are the targets in their order. A log without
    a context, a take's or a replay's, names ``target`` throughout. The other channels are copied as they are."""
    state = StateLog.load(log)
    extra = state.extra | {"playback": {"task": TASK}}
    copy = StateLog(state.build_skeleton(), producer=state.producer, channel_meta=state.channel_meta, extra=extra)
    channels = {name: state.channel(name) for name in state.channel_names}
    if "context" in channels:
        active = np.argmax(channels["context"], axis=1)
    else:
        active = np.full(len(state), LABELS.index(target))
    for k, time in enumerate(state.times):
        copy.record(float(time), **{name: values[k] for name, values in channels.items()}, active_target=active[k])
    copy.save(out)


def player_export(log: Path, speed: float, *, out: Path) -> None:
    """Render a log into an MP4 with skelarm's player and its side panel, without a window, at ``speed`` times."""
    command = [sys.executable, str(PLAYER), str(log), "--export", str(out), "--fps", f"{FPS:g}", "--panel"]
    subprocess.run(
        [*command, "--speed", f"{speed:g}"],
        check=True,
        capture_output=True,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen"},
    )


def clip(log: Path, target: str, speed: int, lines: tuple[str, ...], *, out: Path) -> None:
    """A log in skelarm's player at ``speed`` times, the three targets drawn and the named one emphasised, with the
    speed big and ``lines`` below it written in the side panel."""
    bold = findfont(FontProperties(family="DejaVu Sans", weight="bold"))
    big, small = ImageFont.truetype(bold, SPEED_SIZE), ImageFont.truetype(bold, LINE_SIZE)
    with tempfile.TemporaryDirectory() as tmp:
        shown, raw = Path(tmp) / "shown.sklog.npz", Path(tmp) / "raw.mp4"
        with_targets(log, target, out=shown)
        player_export(shown, speed, out=raw)
        reader: Any = imageio.get_reader(raw)
        writer: Any = imageio.get_writer(out, fps=FPS, codec="libx264", quality=QUALITY, macro_block_size=16)
        with writer:
            for frame in reader:
                image = Image.fromarray(frame)
                draw = ImageDraw.Draw(image)
                draw.text((TEXT_X, SPEED_Y), f"\N{MULTIPLICATION SIGN}{speed}", fill=TEXT_COLOR, font=big)
                for k, line in enumerate(lines):
                    draw.text((TEXT_X, LINES_Y + k * LINE_STEP), line, fill=TEXT_COLOR, font=small)
                writer.append_data(np.asarray(image))
        reader.close()


def join(parts: list[Path], *, out: Path) -> None:
    """Clips one after another, joined by ffmpeg's concat demuxer without encoding them again: they share their
    encoder's settings, frame size, and rate."""
    with tempfile.TemporaryDirectory() as tmp:
        listing = Path(tmp) / "parts.txt"
        listing.write_text("".join(f"file '{part}'\n" for part in parts))
        command = [FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing)]
        subprocess.run([*command, "-c", "copy", "-movflags", "+faststart", str(out)], check=True, capture_output=True)


def joined(path: str, shows: str, parts: list[Medium]) -> Medium:
    logs = [log for part in parts for log in part.logs]
    return Medium(path, shows, logs, functools.partial(join, [OUT / part.path for part in parts]))


# ---------------------------------------------------------------------------- the media


def takes_media() -> list[Medium]:
    folder = FOLDERS[0]
    takes = [(label, index) for label in LABELS for index in range(report.N_TAKES)]
    items = [
        Medium(
            f"{folder}/starts.png",
            "the 27 takes' hand paths in their target's color, their starts, and the untaught and the unseen starts:"
            " the report's Section 2 figure",
            [take_log(*take) for take in takes],
            functools.partial(report_figure, "plot_starts"),
        )
    ]
    for speed in SPEEDS:
        parts = []
        for label, index in takes:
            name, log = take_name(label, index), take_log(label, index)
            parts.append(
                Medium(
                    f"{folder}/parts/x{speed}/{name.replace(' ', '_')}_to_{label}.mp4",
                    f"{name} (demo_{index:02d} to {label}), filtered at 2 Hz, at {speed} times its speed",
                    [log],
                    functools.partial(clip, log, label, speed, (name, f"to {label}")),
                )
            )
        shows = f"the 27 takes one after another, A's nine, B's, then C's, at {speed} times their speed"
        items += [*parts, joined(f"{folder}/all_takes_x{speed}.mp4", shows, parts)]
    return items


def undisturbed_media() -> list[Medium]:
    folder = FOLDERS[1]
    items = []
    for name, n in report.ESNS.items():
        stem = report.robot(n)
        logs = [
            robot_log(stem, arm, task, int(row["start"]))
            for task in LABELS
            for arm in ("esn", "replay")
            for row in report.rows(stem, arm, task)
        ]
        items.append(
            Medium(
                f"{folder}/paths_esn{n}.png",
                f"the hand's paths of {name} over the replay's, undisturbed, with each context, from the takes' starts"
                " and from the 36 unseen starts: the report's figure of Sections 3.2 and 3.3",
                logs,
                functools.partial(report_figure, "plot_paths", name),
            )
        )
    for key, (label, n, arm) in ARMS.items():
        for task in LABELS:
            for which, kind, heading in (
                ("demo", "takes_starts", f"the nine starts of the takes to {task}"),
                ("unseen", "unseen_starts", "the 36 unseen starts"),
            ):
                for speed in SPEEDS:
                    runs = report.group(report.robot(n), arm, task, which)
                    parts = [undisturbed_part(key, task, row, speed) for row in runs]
                    shows = (
                        f"{label}'s runs with the context {task} from {heading}, one after another, at {speed} times"
                    )
                    items += [*parts, joined(f"{folder}/{key}_context_{task}_{kind}_x{speed}.mp4", shows, parts)]
    return items


def undisturbed_part(key: str, task: str, row: dict[str, Any], speed: int) -> Medium:
    """The video of an undisturbed run of an arm of ``ARMS`` with the context ``task``."""
    label, n, arm = ARMS[key]
    end, start = start_names(task, row["origin"])
    log = robot_log(report.robot(n), arm, task, int(row["start"]))
    what = label
    if arm == "replay":
        what = f"the replay of {take_name(task, int(row['replayed'].split('_')[1]))}"
    return Medium(
        f"{FOLDERS[1]}/parts/x{speed}/{key}_context_{task}_{end}.mp4",
        f"{what} with the context {task}, undisturbed, from {start}, at {speed} times its speed",
        [log],
        functools.partial(clip, log, task, speed, (label, f"context {task}", start)),
    )


def change_logs(n: str, timing: str, arm: str, origin: str = "") -> list[Path]:
    """The logs of one arm in the runs of ESN #``n`` whose context changes: every run that changes, or with
    ``origin`` those from that start of the first target."""
    logs = []
    for first, new in report.CHANGES:
        stem = report.switch(n, timing, new)
        runs = [row for row in report.rows(stem, arm, first) if origin in ("", row["origin"])]
        logs += [robot_log(stem, arm, first, int(row["start"])) for row in runs]
    return logs


def change_figures(folder: str) -> list[Medium]:
    """The report's figures of the changes of context, at rest and on the way."""
    items = []
    for name, n in report.ESNS.items():
        items.append(
            Medium(
                f"{folder}/switch_paths_esn{n}.png",
                f"the hand's paths of {name} when the context changes, at rest (above) and on the way (below)",
                [log for timing in report.TIMINGS for log in change_logs(n, timing, "esn")],
                functools.partial(report_figure, "plot_switch_paths", name),
            )
        )
        logs = [
            log
            for timing in report.TIMINGS
            for arm in ("esn", "replay")
            for log in change_logs(n, timing, arm, report.TAKE_SHOWN)
        ]
        items += [
            Medium(
                f"{folder}/switch_{channel}_esn{n}.png",
                f"the joint {what} of {name}'s arm and the replay's over time when the context changes, one run of"
                " each change from the first target's take demo_00's start, at rest (above) and on the way (below)",
                logs,
                functools.partial(report_figure, "plot_switch_series", name, channel),
            )
            for channel, what in (("joints", "angles"), ("torques", "torques"))
        ]
    items.append(
        Medium(
            f"{folder}/switch_distance.png",
            "the hand's distance to the new target over time from the change, in every run that changes, of each ESN,"
            " at rest (above) and on the way (below)",
            [log for n in report.ESNS.values() for timing in report.TIMINGS for log in change_logs(n, timing, "esn")],
            functools.partial(report_figure, "plot_switch_distance"),
        )
    )
    return items


def change_media(timing: str) -> list[Medium]:
    """The ESNs' runs whose context changes at rest or on the way, and the report's figures of the changes."""
    folder = TIMING_FOLDERS[timing]
    items = change_figures(folder)
    for name, n in report.ESNS.items():
        for first, new in report.CHANGES:
            stem = report.switch(n, timing, new)
            change = f"{first} \N{RIGHTWARDS ARROW} {new} at {report.change_time(stem):g} s"
            for speed in SPEEDS:
                parts = []
                for row in report.rows(stem, "esn", first):
                    end, start = start_names(first, row["origin"])
                    log = robot_log(stem, "esn", first, int(row["start"]))
                    parts.append(
                        Medium(
                            f"{folder}/parts/x{speed}/esn{n}_{first}_to_{new}_{end}.mp4",
                            f"{name} from {start}, its context changed {change}, at {speed} times its speed",
                            [log],
                            functools.partial(clip, log, first, speed, (name, change, start)),
                        )
                    )
                shows = f"{name}'s nine runs whose context changes {change}, one after another, at {speed} times"
                items += [*parts, joined(f"{folder}/esn{n}_{first}_to_{new}_x{speed}.mp4", shows, parts)]
    return items


def demonstrations_media() -> list[Medium]:
    folder = FOLDERS[4]
    items = []
    for name, n in report.ESNS.items():
        log = demonstration_log(name)
        items.append(
            Medium(
                f"{folder}/demonstration_esn{n}.png",
                f"{name}'s run pushed by hand in the robot app while the target changes: the report's Section 3.6"
                " figure",
                [log],
                functools.partial(report_figure, "plot_demonstration", name),
            )
        )
        items += [
            Medium(
                f"{folder}/esn{n}_x{speed}.mp4",
                f"{name}'s run pushed by hand in the robot app while the target changes, "
                + ("at its own speed" if speed == 1 else f"at {speed} times its speed"),
                [log],
                functools.partial(clip, log, LABELS[0], speed, (name, "pushed by hand")),
            )
            for speed in DEMONSTRATION_SPEEDS
        ]
    return items


# ---------------------------------------------------------------------------- the README


README = """\
# Three targets by context, for slides

Videos and figures of report 025 of the arm-esn-ctrl repository
(`reports/025-three-targets-by-context/README.md`), the second stage of the multi-task
study: three targets, reached by one ESN, which is told by a context which one. The
folders follow the report's Sections 2 and 3.2 to 3.6. They are not kept in Git;
`presentations/three-targets/make_media.py` there makes them.

## What they show

- **The arm:** a two-link arm on a horizontal plane, its links {lengths} m long and of
  {masses} kg, with no joint limits, simulated every {sim_ms:g} ms.
- **The targets**, each {tol_mm:g} mm in radius and drawn purple, teal, and orange as in
  the recorder: {targets}.
- **The takes:** 27 reaches taught by hand, nine to each target, from starts the
  maintainer chose. They are named as recorded, `take 001` to `take 027`; the ESNs learn
  them filtered at 2 Hz, and the replays play them filtered. The videos show them
  filtered.
- **The start postures:** each target's nine takes' starts, and 36 unseen starts, from
  which nothing was searched or scored, `unseen 0` to `unseen 35` in the order of the
  configurations.
- **The arms**, each giving the reference of the same tracker:
{arms}
  - **Replay:** the take to the target the context names that starts nearest the run's
    start, played by time, whatever the arm does.

  Both ESNs have {neurons} tanh neurons, sparsity {sparsity:g}, a bias per neuron drawn
  uniformly from [−1, 1], a {warmup:g} s warm-up on the held start posture, and a {period_ms:g} ms
  period, and are trained by teacher forcing without noise. Their input is the arm's two
  joint angles every {period_ms:g} ms and the context, one value per target: 1 for the target
  to reach and 0 for the other two, fed in divided by the input scaling, so that its
  weights have the scale 1.
- **The tracker** of every arm: joint PD at ω = {omega:g} rad/s and a damping ratio of {damping:g},
  given the reference's velocity, its torque computed every {control_ms:g} ms and held between.
- **Time:** every robot run holds its start posture for the {warmup:g} s warm-up, from t = −{warmup:g} s;
  the task begins at t = 0. The undisturbed runs end at t = {short:g} s. The context changes
  at {rest:g} s in `3.4_change_at_rest`, when every hand rests at its first target, and the
  runs end at {rest_end:g} s; at {way:g} s in `3.5_change_on_the_way`, while the hand is on its way,
  and they end at {way_end:g} s. The two demonstrations, in which the maintainer changed the
  target and pushed the hand in the robot app, last about 60 s.

## The targets in the videos

Every video draws the three targets, and emphasises the one the context names: a solid
ring with a filled center, the other two dashed. The player is given a copy of each log,
written to a temporary folder and never beside the original, whose task is the three
targets of `experiments/demonstrations/reach_manual_v2_targets.toml` and whose channel
`active_target` is, at each step, the target the log's `context` channel names; skelarm's
player emphasises that target frame by frame. So the emphasis moves when the context
changes. The takes and the replays have no context: they emphasise their own target
throughout. The panel's "Active target" counts from 1: 1 is A, 2 is B, and 3 is C.

Each video is played at a speed, written big in the side panel (`×5`), with below it what
it shows: the take and its target, or the arm, the context or its change, and the start.

## The folders

| Folder | What it holds |
| --- | --- |
| `00_takes` | The takes (Section 2): `starts.png`, `all_takes_x5.mp4`, `all_takes_x10.mp4`, and `parts/x5` and `parts/x10`, each take alone. |
| `3.2_undisturbed` | ESN #1, ESN #2, and the replay, undisturbed, with each context (Sections 3.2 and 3.3): `paths_esn1.png`, `paths_esn2.png`, the runs one after another, and in `parts/x5` and `parts/x10` each run alone. |
| `3.4_change_at_rest` | The context changed at {rest:g} s (Section 3.4): the report's seven figures of the changes, the nine runs of each change one after another, and in `parts/x5` and `parts/x10` each run alone. |
| `3.5_change_on_the_way` | The same with the change at {way:g} s (Section 3.5). The seven figures are those of `3.4_change_at_rest`, so that each folder is complete. |
| `3.6_demonstrations` | The two runs pushed by hand while the target changes (Section 3.6): `demonstration_esn1.png`, `demonstration_esn2.png`, and each run whole at one, five, and ten times its speed. |

## The figures

| File | What it shows |
| --- | --- |
| `00_takes/starts.png` | the 27 takes' hand paths in their target's color, their starts, the 36 untaught starts (the search's scoring), and the 36 unseen starts |
| `3.2_undisturbed/paths_esn1.png`, `paths_esn2.png` | the hand's paths of ESN #1 (or #2) in blue over the replay's in orange, with each context: from the takes' starts above, from the unseen starts below |
| `3.4_…` and `3.5_…`: `switch_paths_esn1.png`, `switch_paths_esn2.png` | the hand's paths when the context changes, in the color of the first target, a diamond where the hand is at the change: at rest above, on the way below |
| `3.4_…` and `3.5_…`: `switch_distance.png` | the hand's distance to the new target from the change, in every run that changes, in the color of the new target |
| `3.4_…` and `3.5_…`: `switch_joints_esn1.png`, `switch_torques_esn1.png`, and ESN #2's | the joint angles or torques over time of one run of each change, from the first target's take demo_00's start, the ESN's arm over the replay's: at rest above, on the way below |
| `3.6_demonstrations/demonstration_esn1.png`, `demonstration_esn2.png` | the run pushed by hand: the hand's path in the color of the named target, and over time the joint angles, the torques, the tip force, the distance to the named target, and the context |

They are the report's, drawn by its `make_figures.py`, at {dpi} dpi where the report's
are at 90.

## The videos

| File | What it shows |
| --- | --- |
| `00_takes/parts/x5/take_001_to_A.mp4`, … | a take, named as recorded, and its target |
| `00_takes/all_takes_x5.mp4`, `all_takes_x10.mp4` | the 27 takes one after another: A's nine (demo_00 to demo_08), then B's, then C's |
| `3.2_undisturbed/parts/x5/esn1_context_A_take_start_00.mp4`, … | a run of `esn1`, `esn2`, or `replay` with the context `A`, `B`, or `C`, from the start of the take demo_00 to demo_08 to that target (`take_start_00` to `take_start_08`) or from an unseen start (`unseen_00` to `unseen_35`). The text names the take whose start it is, as recorded. |
| `3.2_undisturbed/esn1_context_A_takes_starts_x5.mp4`, `esn1_context_A_unseen_starts_x5.mp4`, … | the nine runs from the takes' starts, in the takes' order, or the 36 from the unseen starts, in their order, one after another |
| `3.4_…` and `3.5_…`: `parts/x5/esn1_A_to_B_take_start_00.mp4`, … | a run of ESN #1 (or #2) from the start of the take demo_00 to demo_08 to the first target, with its context, which changes to the new target |
| `3.4_…` and `3.5_…`: `esn1_A_to_B_x5.mp4`, … | the nine runs of a change one after another, in the takes' order |
| `3.6_demonstrations/esn1_x1.mp4`, `esn1_x5.mp4`, `esn1_x10.mp4`, and ESN #2's | the run pushed by hand whole; the force at the hand is drawn as an arrow while it acts |

The suffix `x5`, `x10`, or `x1` is the speed: five, ten, or one times as fast as the run.

## Notes

- **The replays** do not depend on the ESN. They are taken from ESN #1's runs; the
  report checks that ESN #2's are the same.
- **No replay when the context changes.** A replay keeps to its take, whatever the
  context does, so `3.4_change_at_rest` and `3.5_change_on_the_way` have the ESNs' runs
  alone; their figures draw the replay as the report does.
- **The two demonstrations** are the logs the robot app saved, kept in the storage root's
  `data/recorded_forces` (and copied in the report's `data/recorded_forces`). A push
  lasts 0.12 to 0.50 s. The player shows the log every 1/30 s of the video, so at five
  times a push shows in one to three frames, and at ten times some pushes show in none:
  the videos at `x1` show every push.
- **The videos** are H.264 (High profile, yuv420p) at {fps:g} fps, 1104 x 800 px. Each
  part is the player's export with the text drawn on it, encoded again by libx264 at a
  CRF of {crf}. The videos one after another join their parts with ffmpeg's concat
  demuxer, copying the streams without encoding them again.
- [`media.csv`](media.csv) lists every file with the logs it shows, relative to the
  storage root.

## Reproducing them

From the repository, at commit `{commit}`{dirty}:

    uv run python presentations/three-targets/make_media.py --workers 8

It reads the takes and the runs under the storage root and overwrites the files here, this
README included; `--only` renders the named folders alone. This README was written on
{day}.
"""  # noqa: E501, RUF001


def commit_state() -> tuple[str, str]:
    """The repository's commit, and `` (with uncommitted changes)`` if it has any, else nothing."""
    git = [["git", "rev-parse", "--short", "HEAD"], ["git", "status", "--porcelain"]]
    commit, changes = (subprocess.run(c, cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout for c in git)
    return commit.strip(), " (with uncommitted changes)" if changes.strip() else ""


def esn_of(n: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The ``[esn]`` table of ESN #``n``, as saved with it, and the ``[[demonstrations]]`` it was trained on."""
    model = resolve_run_path(report.config_of(report.robot(n))["esn"]["model"])
    with model.open("rb") as f:
        esn = tomllib.load(f)["esn"]
    with (model.parent / "config.toml").open("rb") as f:
        return esn, tomllib.load(f)["demonstrations"]


def write_readme() -> None:
    """The output folder's README.md, its numbers read from the runs' configurations."""
    esns = {name: esn_of(n) for name, n in report.ESNS.items()}
    fixed = ["n_neurons", "sparsity", "bias", "bias_scaling", "warmup", "dt", "teacher_noise_deg"]
    fixed += ["context_scaling", "context_times_input_scaling"]
    first = next(iter(esns.values()))[0]
    if any(esn[key] != first[key] for esn, _ in esns.values() for key in fixed):
        msg = f"the ESNs differ in what the README gives as fixed: {fixed}"
        raise ValueError(msg)
    if not (
        first["bias"]
        and first["bias_scaling"] == 1.0
        and first["teacher_noise_deg"] == 0.0
        and first["context_scaling"] == 1.0
        and not first["context_times_input_scaling"]
    ):
        msg = "the README's bias, training, or context does not match the ESNs'"
        raise ValueError(msg)
    nine = [f"demo_{index:02d}_filtered.sklog.npz" for index in range(report.N_TAKES)]
    every_take = [(label, f"results/demonstrations/{report.DEMO_RUNS[label].name}", nine) for label in LABELS]
    for _, demonstrations in esns.values():
        if [(d["context"], d["run"], d["train"]) for d in demonstrations] != every_take:
            msg = "an ESN is not trained on the 27 takes, each with its target's context"
            raise ValueError(msg)
    arms = "\n".join(
        f"  - **{name}:** leak rate {esn['leak_rate']:g}, spectral radius {esn['spectral_radius']:g}, input scaling"
        f" {esn['input_scaling']:g}, ridge {esn['ridge']:g},\n    seed {esn['seed']}; trained on the 27 takes, each"
        " with its target's context."
        for name, (esn, _) in esns.items()
    )
    targets = [f"{label} at ({x:g}, {y:g}) m" for label, (x, y) in report.TARGETS.items()]
    tracker = report.config_of(report.robot("1"))["tracker"]
    links = report.SKELETON.links[1:]
    rest, way = (report.switch("1", timing, "A") for timing in report.TIMINGS)
    commit, dirty = commit_state()
    (OUT / "README.md").write_text(
        README.format(
            lengths=" and ".join(f"{link.prop.length:g}" for link in links),
            masses=" and ".join(f"{link.prop.m:g}" for link in links),
            sim_ms=1e3 * _ARM["simulator"]["dt"],
            targets=(", ".join(targets[:-1]) + f", and {targets[-1]}").replace("-", "\N{MINUS SIGN}"),
            tol_mm=1e3 * report.TOLERANCE,
            arms=arms,
            neurons=first["n_neurons"],
            sparsity=first["sparsity"],
            warmup=first["warmup"],
            period_ms=1e3 * first["dt"],
            omega=tracker["omegas"][0],
            damping=tracker["dampings"][0],
            control_ms=1e3 * tracker["control_period"],
            short=report.run_end(report.robot("1")),
            rest=report.change_time(rest),
            rest_end=report.run_end(rest),
            way=report.change_time(way),
            way_end=report.run_end(way),
            dpi=DPI,
            fps=FPS,
            crf=int((1 - QUALITY / 10) * 51),  # as imageio-ffmpeg sets it
            commit=commit,
            dirty=dirty,
            day=date.today().isoformat(),
        )
    )


def write_media_csv(items: list[Medium]) -> None:
    root = storage_root()
    with (OUT / "media.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "file", "shows", "logs"])
        for item in items:
            folder, _, name = item.path.rpartition("/")
            writer.writerow([folder, name, item.shows, "; ".join(str(p.relative_to(root)) for p in item.logs)])


# ---------------------------------------------------------------------------- the whole


def render(items: list[Medium], workers: int) -> None:
    """Make the files of ``items`` in ``workers`` processes: the videos first, as they take longest, and the videos
    one after another last, once their parts are made."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # for the processes started from here on
    for item in items:
        (OUT / item.path).parent.mkdir(parents=True, exist_ok=True)
    first = sorted(
        (item for item in items if item.make.func is not join),
        key=lambda item: (not item.path.endswith(".mp4"), not item.path.startswith(FOLDERS[4])),
    )
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for batch in (first, [item for item in items if item.make.func is join]):
            futures = {pool.submit(item.make, out=OUT / item.path): item for item in batch}
            for done, future in enumerate(as_completed(futures), start=1):
                if future.exception() is not None:
                    print(f"failed: {futures[future].path}", flush=True)
                    pool.shutdown(cancel_futures=True)
                    future.result()
                if done % 50 == 0 or done == len(futures):
                    print(f"made {done} of {len(futures)} files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8, help="processes that render the files (default: 8)")
    parser.add_argument(
        "--only", nargs="+", choices=FOLDERS, metavar="FOLDER", help=f"render these folders alone, of {FOLDERS}"
    )
    args = parser.parse_args()
    items = takes_media() + undisturbed_media()
    items += [item for timing in report.TIMINGS for item in change_media(timing)] + demonstrations_media()
    chosen = [item for item in items if args.only is None or item.path.split("/")[0] in args.only]
    print(f"{len(chosen)} of {len(items)} files to make into {OUT}", flush=True)
    render(chosen, args.workers)
    write_media_csv(items)
    write_readme()
    print(f"wrote media.csv and README.md to {OUT}")


if __name__ == "__main__":
    main()
