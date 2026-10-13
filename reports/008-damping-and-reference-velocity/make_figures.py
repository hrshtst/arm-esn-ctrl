# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 008.

    uv run python reports/008-damping-and-reference-velocity/make_figures.py
    uv run python reports/008-damping-and-reference-velocity/make_figures.py --logs --animations

From the copies of the runs under ``results/`` (configurations, run records, and
metrics), it draws into ``results/summary``:

- ``disturbed.png``: after the block's release and each push, the ringing and the
  return onto the taught path of the chosen ESN and the replay against the damping
  ratio, given a zero reference velocity or the reference's (Section 3.2);
- ``undisturbed.png``: undisturbed, the path RMSE and the arrival, and from the
  offset starts, the median and largest peak torque, against the damping ratio
  (Sections 3.1 and 3.3);
- ``control_period.png``: after each disturbance, the ringing and the return of both
  arms against the damping ratio, with the tracker's torque computed every 2 ms or
  every 10 ms, given a zero reference velocity (Section 3.4);

and prints the tables of the report. With ``--logs``, it reads the runs' logs, which
are not kept in Git, from the runs under the storage root, and also draws each arm's
distance from the taught path under the block and the pushes, one panel per tracker
(``block_distances.png``, ``pushes_distances.png``, Section 3.2), and copies two of
the runs' own figures (FIGURES). With ``--animations``, it animates the block and the
pushes with the arms of PANELS side by side, each rendered by skelarm's player
(``block.gif``, ``pushes.gif``).
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullFormatter
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.metrics import distances_to_path
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
# The take, as report 005 copied it: the runs here read it from the storage root, as there.
TAKE_RUN = REPORT.parent / "005-tuned-on-the-robot" / "data" / "20261008-170034-reach_manual_v2"
DT = 0.01  # the ESNs' period (s)
DAMPINGS = (0.1, 0.4, 0.8)
# The trackers' reference velocity: the configurations' tag, and its name.
VELOCITIES = {"zero": "zero reference velocity", "reference": "reference velocity"}
# How often the tracker's torque is computed: every simulation step (2 ms), or every 10 ms and held between, the
# configurations' tag for each, and its name.
CONTROLS = {"": "every 2 ms", "_control10ms": "every 10 ms"}
ARMS = {"esn": "chosen ESN", "replay": "replay"}
# The disturbances, each with its run, the suffix of its columns in metrics.csv, its name, and, for a hand not back
# on the path before the next disturbance or the end of the 24 s run, when it counts as back (s after its end).
DISTURBANCES = [
    ("block", "", "the block's release", 24.0 - 5.5),
    ("pushes", "_1", "the push forward at 3 s", 7.0 - 3.5),
    ("pushes", "_2", "the push across at 7 s", 15.0 - 7.5),
    ("pushes", "_3", "the push across at 15 s", 24.0 - 15.5),
]
BACK_WITHIN = 5.0  # mm: an arm is back on the taught path once it stays within this of it

ESN_COLOR = "#2a78d6"
REPLAY_COLOR = "#eb6834"
REPLAY_LIGHT_COLOR = "#f2a57f"  # the replay at the lightest damping, in the animation
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
COLORS = {"esn": ESN_COLOR, "replay": REPLAY_COLOR}
LINESTYLES = {"zero": "-", "reference": "--"}

# The animation of the pushes, given a zero reference velocity: per panel, the label (the arm's reference, then its
# tracker), the tracker setting, the arm, and the label's color.
PANELS = [
    ("chosen ESN\ntracker: joint PD, damping ratio 0.8", "pd_w20_z0.8", "esn", ESN_COLOR),
    ("replay of the take by time\ntracker: joint PD, damping ratio 0.8", "pd_w20_z0.8", "replay", REPLAY_COLOR),
    ("replay of the take by time\ntracker: joint PD, damping ratio 0.1", "pd_w20_z0.1", "replay", REPLAY_LIGHT_COLOR),
]
# The runs' own figures copied into the summary, by the name there: the scenario, the reference velocity, and the
# figure in the run.
FIGURES = {
    "pushes_joints.png": ("pushes", "zero", "joints.png"),
    "offset_torques.png": ("offsets", "reference", "torques_offset_+15_+15.png"),
}
ANIMATION_SPAN = (-0.2, 19.0)
ANIMATION_FPS = 10.0
ANIMATION_HOLD_MS = 1500  # the last frame stays this long before the GIF loops


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also analyze the run logs in the storage root")
    parser.add_argument("--animations", action="store_true", help="also export a GIF from the logs in the storage root")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {
        "disturbed.png": plot_disturbed(),
        "undisturbed.png": plot_undisturbed(),
        "control_period.png": plot_control_period(),
    }
    if args.logs:
        figures |= {f"{scenario}_distances.png": plot_distances(scenario) for scenario in ("block", "pushes")}
        for name, (scenario, velocity, figure) in FIGURES.items():
            shutil.copyfile(
                resolve_run_path(f"results/{run_name(stem_of(scenario, velocity))}") / figure, SUMMARY / name
            )
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=150)
    print_tables()
    if args.animations:
        export_animation("block", stem_of("block", "zero"), "the block, from 4.5 s to 5.5 s")
        export_animation("pushes", stem_of("pushes", "zero"), "three pushes, at 3 s, 7 s, and 15 s")
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


def run_name(stem: str) -> str:
    """The name of the one run of a configuration, by its stem."""
    matches = sorted(p.name for p in RESULTS.glob(f"*-{stem}") if p.is_dir())
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


def stem_of(scenario: str, velocity: str, control: str = "") -> str:
    """A configuration's stem, by its scenario, its reference velocity, and its control period (CONTROLS)."""
    return f"{scenario}_robot_pd_{velocity}_velocity{control}_filtered"


def rows_of(scenario: str, velocity: str, control: str = "") -> list[dict[str, Any]]:
    """The metrics of a scenario's run with the trackers given ``velocity``."""
    return read_csv(RESULTS / run_name(stem_of(scenario, velocity, control)) / "metrics.csv")


def run_of(scenario: str, velocity: str, damping: float, arm: str, start: int = 0, control: str = "") -> dict[str, Any]:
    """One arm's run from one start, with one tracker."""
    return next(
        r
        for r in rows_of(scenario, velocity, control)
        if r["damping"] == damping and r["arm"] == arm and r["start"] == start
    )


def returned(run: dict[str, Any], suffix: str, cap: float) -> float:
    """When the hand is back on the path after a disturbance (s), or ``cap`` if it is not back by the window's end."""
    back = run[f"return_time_s{suffix}"]
    return float(back) if np.isfinite(back) else cap


@functools.cache
def taught_hand() -> NDArray[np.float64]:
    """The hand path of the take filtered at 2 Hz, what the ESN was trained on, every 10 ms."""
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    q = resample_joint_angles(StateLog.load(TAKE_RUN / "demo_00_filtered.sklog.npz"), DT)[1]
    return endpoint_positions(skeleton, q)


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


# ---------------------------------------------------------------------------- figures without the logs


def plot_disturbed() -> Figure:
    """After each disturbance, the ringing and the return of both arms against the damping ratio.

    Solid lines: the trackers given a zero reference velocity; dashed: given the
    reference's velocity. A hand not back on the path before the next disturbance
    is drawn at the window's end, marked with a cross.
    """
    fig = Figure(figsize=(14, 6.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "After each disturbance, joint PD: the chosen ESN (blue) and the replay (orange), against the damping ratio;"
        " solid: zero reference velocity, dashed: reference velocity",
        color="#0b0b0b",
    )
    axes = fig.subplots(2, len(DISTURBANCES), sharex=True)
    for col, (scenario, suffix, name, cap) in enumerate(DISTURBANCES):
        for row, measure in enumerate(("ringing", "return")):
            ax = axes[row, col]
            style(ax)
            for velocity, linestyle in LINESTYLES.items():
                for arm, color in COLORS.items():
                    runs = [run_of(scenario, velocity, damping, arm) for damping in DAMPINGS]
                    if measure == "ringing":
                        y = [1000 * r[f"ringing_m{suffix}"] for r in runs]
                    else:
                        y = [returned(r, suffix, cap) for r in runs]
                    ax.plot(DAMPINGS, y, color=color, linestyle=linestyle, marker="o", markersize=4, linewidth=1.4)
                    if measure == "return":
                        missing = [
                            d
                            for d, r in zip(DAMPINGS, runs, strict=True)
                            if not np.isfinite(r[f"return_time_s{suffix}"])
                        ]
                        ax.scatter(missing, [cap] * len(missing), marker="x", s=60, color=color, zorder=3)
            ax.set_xticks(DAMPINGS)
            ax.set_ylim(0, None)
            if row == 0:
                ax.set_title(name, color="#0b0b0b", fontsize=10)
            else:
                ax.set_xlabel("damping ratio", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel(
                    "ringing over the next 2 s (mm)" if measure == "ringing" else "back within 5 mm\nof the path (s)",
                    color=TEXT_COLOR,
                )
    return fig


def plot_control_period() -> Figure:
    """After each disturbance, the ringing and the return with the torque computed every 2 ms or every 10 ms.

    Given a zero reference velocity; the runs given the reference's are much the same
    (Section 3.4). Solid lines: every 2 ms; dashed: every 10 ms.
    """
    fig = Figure(figsize=(14, 6.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Joint PD, zero reference velocity: the chosen ESN (blue) and the replay (orange),"
        " the torque computed every 2 ms (solid) or every 10 ms and held (dashed)",
        color="#0b0b0b",
    )
    axes = fig.subplots(2, len(DISTURBANCES), sharex=True)
    for col, (scenario, suffix, name, cap) in enumerate(DISTURBANCES):
        for row, measure in enumerate(("ringing", "return")):
            ax = axes[row, col]
            style(ax)
            for control, linestyle in zip(CONTROLS, ("-", "--"), strict=True):
                for arm, color in COLORS.items():
                    runs = [run_of(scenario, "zero", damping, arm, control=control) for damping in DAMPINGS]
                    if measure == "ringing":
                        y = [1000 * r[f"ringing_m{suffix}"] for r in runs]
                    else:
                        y = [returned(r, suffix, cap) for r in runs]
                    ax.plot(DAMPINGS, y, color=color, linestyle=linestyle, marker="o", markersize=4, linewidth=1.4)
                    if measure == "return":
                        missing = [
                            d
                            for d, r in zip(DAMPINGS, runs, strict=True)
                            if not np.isfinite(r[f"return_time_s{suffix}"])
                        ]
                        ax.scatter(missing, [cap] * len(missing), marker="x", s=60, color=color, zorder=3)
            ax.set_xticks(DAMPINGS)
            ax.set_ylim(0, None)
            if row == 0:
                ax.set_title(name, color="#0b0b0b", fontsize=10)
            else:
                ax.set_xlabel("damping ratio", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel(
                    "ringing over the next 2 s (mm)" if measure == "ringing" else "back within 5 mm\nof the path (s)",
                    color=TEXT_COLOR,
                )
    return fig


def plot_undisturbed() -> Figure:
    """Undisturbed, the path RMSE and the arrival; from the offset starts, the peak torques; against the damping."""
    fig = Figure(figsize=(14, 4.0), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Joint PD: the chosen ESN (blue) and the replay (orange), against the damping ratio;"
        " solid: zero reference velocity, dashed: reference velocity",
        color="#0b0b0b",
    )
    axes = fig.subplots(1, 3)
    titles = (
        "undisturbed: path RMSE (mm)",
        "undisturbed: arrival (s)",
        "offset starts: peak torque (N m), median and largest (▲)",
    )
    for ax, title in zip(axes, titles, strict=True):
        style(ax)
        ax.set_title(title, color="#0b0b0b", fontsize=10)
        ax.set_xticks(DAMPINGS)
        ax.set_xlabel("damping ratio", color=TEXT_COLOR)
    for velocity, linestyle in LINESTYLES.items():
        for arm, color in COLORS.items():
            nominal = [run_of("nominal", velocity, damping, arm) for damping in DAMPINGS]
            style_kwargs = {"color": color, "linestyle": linestyle, "marker": "o", "markersize": 4, "linewidth": 1.4}
            axes[0].plot(DAMPINGS, [1000 * r["taught_path_rmse_m"] for r in nominal], **style_kwargs)
            axes[1].plot(DAMPINGS, [r["arrival_time_s"] for r in nominal], **style_kwargs)
            peaks = [
                [r["peak_torque_nm"] for r in rows_of("offsets", velocity) if r["damping"] == d and r["arm"] == arm]
                for d in DAMPINGS
            ]
            axes[2].plot(DAMPINGS, [np.median(p) for p in peaks], **style_kwargs)
            axes[2].scatter(DAMPINGS, [max(p) for p in peaks], color=color, marker="^", s=24)
    for ax, ticks in ((axes[0], (0.3, 0.5, 1, 2, 5)), (axes[2], (10, 20, 50, 100, 200))):
        ax.set_yscale("log")
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    return fig


# ---------------------------------------------------------------------------- figures from the logs


def path_distances(stem: str, setting: str, arm: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The times (every 10 ms) and the hand's distance (mm) from the taught path in a run's first start."""
    log = StateLog.load(resolve_run_path(f"results/{run_name(stem)}") / setting / f"{arm}_00.sklog.npz")
    task = log.times >= 0
    skeleton = Skeleton.from_toml(TAKE_RUN / "config.toml")
    hand = endpoint_positions(skeleton, log.channel("q").reshape(len(log.times), -1)[task])
    every = round(DT / float(np.median(np.diff(log.times))))
    return log.times[task][::every], 1000 * distances_to_path(hand[::every], taught_hand())


def plot_distances(scenario: str) -> Figure:
    """Each arm's distance from the taught path over time in a scenario, one panel per tracker."""
    fig = Figure(figsize=(14, 6.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        f"{scenario.capitalize()}, joint PD: how far each arm goes from the taught path, and how it comes back",
        color="#0b0b0b",
    )
    axes = fig.subplots(len(VELOCITIES), len(DAMPINGS), sharex=True, sharey=True)
    for row, (velocity, velocity_name) in enumerate(VELOCITIES.items()):
        stem = stem_of(scenario, velocity)
        with (RESULTS / run_name(stem) / "config.toml").open("rb") as f:
            spans = disturbance_spans(tomllib.load(f)["disturbance"])
        for col, damping in enumerate(DAMPINGS):
            ax = axes[row, col]
            style(ax)
            for onset, end in spans:
                ax.axvspan(onset, end, color=DISTURBANCE_COLOR, zorder=0)
            for arm, color in COLORS.items():
                times, distance = path_distances(stem, f"pd_w20_z{damping:g}", arm)
                ax.plot(times, distance, color=color, linewidth=1.2, label=ARMS[arm])
            ax.axhline(BACK_WITHIN, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
            ax.set_title(f"damping ratio {damping:g}, {velocity_name}", color="#0b0b0b", fontsize=10)
            ax.set_xlim(0, 20)
            ax.set_ylim(0, None)
            if col == 0:
                ax.set_ylabel("from the taught path (mm)", color=TEXT_COLOR)
            if row == len(VELOCITIES) - 1:
                ax.set_xlabel("time (s)", color=TEXT_COLOR)
    axes[0, 0].legend(frameon=False, labelcolor=TEXT_COLOR)
    return fig


# ---------------------------------------------------------------------------- the animation


def player_export(log: Path, out: Path, fps: float) -> None:
    """Render a run's log into ``out`` (a GIF) with skelarm's player, without a window."""
    player = REPO_ROOT / "third_party" / "skelarm" / "tools" / "player.py"
    subprocess.run(
        [sys.executable, str(player), str(log), "--export", str(out), "--fps", f"{fps:g}"],
        check=True,
        capture_output=True,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen"},  # render without a window
    )


def export_animation(name: str, stem: str, title: str) -> None:
    """Animate a run's demonstrated start into ``<name>.gif``: the arms of PANELS side by side, rendered by skelarm.

    The player exports each arm's run as a GIF; its frames, which the GIF merges
    where the arm rests, are spread back over time, cropped to where the arms move,
    labeled, and tiled into one GIF with the task time and the disturbance acting.
    """
    import tempfile

    from PIL import Image, ImageDraw, ImageFont, ImageSequence

    with (RESULTS / run_name(stem) / "config.toml").open("rb") as f:
        disturbance = tomllib.load(f).get("disturbance")
    tables = [] if disturbance is None else [disturbance] if isinstance(disturbance, dict) else disturbance
    spans = disturbance_spans(disturbance)
    frames, starts = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (_, setting, arm, _) in enumerate(PANELS):
            path = resolve_run_path(f"results/{run_name(stem)}") / setting / f"{arm}_00.sklog.npz"
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
        return f"{title}, zero reference velocity   t = {t:+.1f} s{status}"

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
    print("== 3.1 Undisturbed: arm, reference velocity, damping ratio: arrival (s), holds, path RMSE (mm), tracking")
    print("error (deg RMS), peak torque (N m)")
    for velocity in VELOCITIES:
        for damping in DAMPINGS:
            for arm in ARMS:
                r = run_of("nominal", velocity, damping, arm)
                print(
                    f"  {ARMS[arm]}, {velocity}, {damping:g}: {r['arrival_time_s']:.2f}, {r['success']},"
                    f" {1000 * r['taught_path_rmse_m']:.2f}, {r['tracking_error_deg']:.2f}, {r['peak_torque_nm']:.2f}"
                )

    print("\n== 3.2 After each disturbance (the block's release; the pushes at 3 s, 7 s, 15 s): arm, reference")
    print("velocity, damping ratio: gap (deg) | ringing (mm) | farthest from the path (mm) | back within 5 mm (s);")
    print("then the arrival (s) and whether it holds, under the block and under the pushes")
    for velocity in VELOCITIES:
        for damping in DAMPINGS:
            for arm in ARMS:
                cells = []
                for scenario, suffix, _, _ in DISTURBANCES:
                    r = run_of(scenario, velocity, damping, arm)
                    back = r[f"return_time_s{suffix}"]
                    cells.append(
                        f"{r[f'end_gap_deg{suffix}']:.1f} | {1000 * r[f'ringing_m{suffix}']:.1f} |"
                        f" {1000 * r[f'off_path_m{suffix}']:.0f} | {f'{back:.2f}' if np.isfinite(back) else 'not back'}"
                    )
                block, pushes = run_of("block", velocity, damping, arm), run_of("pushes", velocity, damping, arm)
                print(
                    f"  {ARMS[arm]}, {velocity}, {damping:g}: {'  ||  '.join(cells)}"
                    f"  ||  block {block['arrival_time_s']:.2f} {block['success']},"
                    f" pushes {pushes['arrival_time_s']:.2f} {pushes['success']},"
                    f" block peak torque {block['peak_torque_nm']:.1f}"
                )

    print("\n== 3.3 Offset starts: arm, reference velocity, damping ratio: starts that hold of 49, median path RMSE")
    print("(mm), median and largest peak torque (N m), the start of the largest")
    for velocity in VELOCITIES:
        for damping in DAMPINGS:
            for arm in ARMS:
                group = [r for r in rows_of("offsets", velocity) if r["damping"] == damping and r["arm"] == arm]
                largest = max(group, key=lambda r: r["peak_torque_nm"])
                print(
                    f"  {ARMS[arm]}, {velocity}, {damping:g}: {sum(r['success'] for r in group)},"
                    f" {1000 * np.median([r['taught_path_rmse_m'] for r in group]):.1f},"
                    f" {np.median([r['peak_torque_nm'] for r in group]):.1f}, {largest['peak_torque_nm']:.0f}"
                    f" ({largest['origin'].split(' ', 1)[-1]})"
                )

    print("\n== 3.4 The torque computed every 2 ms or every 10 ms: arm, reference velocity, damping ratio, then for")
    print("each: after the block's release and each push, ringing (mm) / back within 5 mm (s); whether it holds under")
    print("the block and the pushes; undisturbed path RMSE (mm); offset starts: holds of 49, median path RMSE (mm),")
    print("median and largest peak torque (N m)")
    for velocity in VELOCITIES:
        for damping in DAMPINGS:
            for arm in ARMS:
                print(f"  {ARMS[arm]}, {velocity}, {damping:g}:")
                for control, control_name in CONTROLS.items():
                    cells = []
                    for scenario, suffix, _, _ in DISTURBANCES:
                        r = run_of(scenario, velocity, damping, arm, control=control)
                        back = r[f"return_time_s{suffix}"]
                        back_text = f"{back:.2f}" if np.isfinite(back) else "not back"
                        cells.append(f"{1000 * r[f'ringing_m{suffix}']:.1f} / {back_text}")
                    holds = [
                        run_of(scenario, velocity, damping, arm, control=control)["success"]
                        for scenario in ("block", "pushes")
                    ]
                    nominal = run_of("nominal", velocity, damping, arm, control=control)
                    group = [
                        r for r in rows_of("offsets", velocity, control) if r["damping"] == damping and r["arm"] == arm
                    ]
                    peaks = [r["peak_torque_nm"] for r in group]
                    rmse = 1000 * np.median([r["taught_path_rmse_m"] for r in group])
                    print(
                        f"    {control_name}: {' | '.join(cells)} || holds {holds[0]}, {holds[1]}"
                        f" || {1000 * nominal['taught_path_rmse_m']:.2f}"
                        f" || {sum(r['success'] for r in group)}, {rmse:.1f},"
                        f" {np.median(peaks):.1f}, {max(peaks):.0f}"
                    )


if __name__ == "__main__":
    main()
