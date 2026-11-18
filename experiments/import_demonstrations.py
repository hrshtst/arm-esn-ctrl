# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Import demonstrations taught by hand into a demonstration run.

    uv run python experiments/import_demonstrations.py experiments/demonstrations/reach_manual_single.toml

The take recorded with skelarm's ``tools/trajectory_recorder.py`` (``take`` in
``[recording]``, relative to the storage root) is checked first, with
:func:`arm_esn_ctrl.demonstrations.check_take`: it must have been recorded with the
configuration's arm, begin at the start posture, arrive at the target and hold
there, and never jump. A take that fails a check is not imported: the problems are
printed, no run is created, and the take must be recorded again.

Several takes of one task, each from its own start posture, are listed as ``takes``
instead, in the order of the start postures of ``[demonstrations]`` ``start_q``.
They are imported together, as ``demo_00``, ``demo_01``, ..., each with its filtered
copy, only if every one passes; the figure then draws them together. A task without
a target, such as a periodic motion, has no arrival to check or to measure.

Takes of several targets have a ``[task]`` of type ``multi_target_reaching``, whose
``targets`` the recorder shows, and ``takes`` is a pattern instead of a list, such
as ``data/reach_manual_v2_targets/take_*.sklog.npz``: every file under the storage
root that matches it is read, in the order of their names. Each take reaches the
target that its log records as active, from where the arm was posed in the
recorder, its own first sample, so there is no start posture to check. A take whose
log records no multi-target task, or other targets than the configuration's, is not
imported, as a take that fails a check. The takes are imported into one run per
target that has takes, named after the configuration and the target's label
(``reach_manual_v2_targets_a``, ...), each as a configuration of that target alone
would import them: its ``config.toml`` has a ``reaching`` ``[task]`` toward the
target, lists the target's takes in ``[recording]`` ``takes`` and where each begins
in ``[demonstrations]`` ``start_q`` (deg), and keeps the rest. An ESN's
``[[demonstrations]]`` tables then name the runs as any others; the import prints
each run's directory.

A take that passes is imported, its pause before moving included, into a new run
directory, as two demonstrations that a configuration's ``[demonstrations]`` names
as any others:

- ``demo_00.sklog.npz``: a copy of the take as recorded, with its uneven sample
  times; the ESN resamples it at its own period;
- ``demo_00_filtered.sklog.npz``: the take smoothed by the filter of
  ``[recording.filter]`` (see :func:`arm_esn_ctrl.demonstrations.smooth_take`),
  which removes the steps of the cursor's whole screen pixels; the same checks
  apply to it;

and their measurements:

- ``metrics.csv``: one row per demonstration: its reach metrics (see
  :func:`arm_esn_ctrl.metrics.reach_metrics`), when the hand starts to move (5 mm
  from where it started) and when it arrives, its joint jitter, how the take was
  sampled (the number of samples, the length, the requested and achieved rates,
  and the longest interval), and the take's file, relative to the storage root, and
  its SHA-256, which ties the run to it;
- ``demonstrations.png``: the hand path, the joint angles and the hand speed over
  time of both, and the sampling intervals of the take.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import tomli_w
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog, Task, active_target_index, multi_target_specs

from arm_esn_ctrl.demonstrations import check_take, endpoint_positions, smooth_take
from arm_esn_ctrl.metrics import hand_speed, hold_metrics, jitter, onset_index, reach_metrics
from arm_esn_ctrl.postures import joint_columns
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path, start_run, storage_root

TAKE_COLOR = "#2a78d6"
RECORDED_COLOR = "#a3a29d"
# Color, line width, and legend label of each demonstration: the take as recorded, under the filtered one.
DEMO_STYLES = {"demo_00": (RECORDED_COLOR, 2.2, "as recorded"), "demo_00_filtered": (TAKE_COLOR, 1.2, "filtered")}
# The colors of several takes, one each, in turn.
TAKE_COLORS = ["#2a78d6", "#eb6834", "#2e8b57", "#8b5fbf", "#c2185b", "#00838f", "#9e7b00", "#5d6d7e", "#6d4c41"]
JITTER_WINDOW = 5  # samples of the moving average the jitter is measured from, as in the autonomous runs
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
# The line style of each joint's angle, and its name in the panel's title.
JOINT_STYLES = [("-", "solid"), ("--", "dashed"), (":", "dotted"), ("-.", "dash-dot")]
MULTI_TARGET_TASK = "multi_target_reaching"  # the [task] type of takes of several targets
# The keys of a multi-target [task] that the [task] of one of its targets does not have.
MULTI_TARGET_KEYS = ("type", "active", "targets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="demonstration configuration file (TOML)")
    args = parser.parse_args()

    # Check the takes before creating a run, so that a take that fails leaves nothing behind.
    with args.config.open("rb") as f:
        config = tomllib.load(f)
    skeleton = Skeleton.from_toml(args.config)
    several_targets = config["task"]["type"] == MULTI_TARGET_TASK
    if several_targets:
        takes = sorted(storage_root().glob(config["recording"]["takes"]))
        if not takes:
            msg = f"no take under {storage_root()} matches [recording] takes = {config['recording']['takes']!r}"
            raise FileNotFoundError(msg)
        runs, problems = target_configs(config, takes, args.config.stem)
    else:
        takes = listed_takes(config["recording"])
        runs, problems = {args.config.stem: config}, {}
    # What tells one run's demonstrations from another's where they are printed: the target.
    of = {name: f" of target {run['task']['target']['label']}" if several_targets else "" for name, run in runs.items()}
    imports = {name: load_demos(run, skeleton) for name, run in runs.items()}
    for name, (demos, start_of, take_of) in imports.items():
        checked = check_demos(demos, start_of, skeleton, runs[name])
        problems |= {f"{demo}{of[name]} ({take_of[demo].name})": checked[demo] for demo in demos}
    if any(problems.values()):
        print("Not imported: " + ", ".join(str(take) for take in takes))
        for where, found in problems.items():
            for problem in found:
                print(f"  - {where}: {problem}")
        raise SystemExit(1)

    run_dirs = {}
    for name, (demos, start_of, take_of) in imports.items():
        run = runs[name]
        _, run_dir = start_run(args.config, name=name, text=None if run is config else target_text(args.config, run))
        run_dirs[name] = run_dir
        for demo, log in demos.items():
            if demo.endswith("_filtered"):
                log.save(run_dir / f"{demo}.sklog.npz")
            else:
                shutil.copy2(take_of[demo], run_dir / f"{demo}.sklog.npz")
        task = Task.from_dict(run["task"])
        hands = {demo: endpoint_positions(skeleton, joint_angles(log)) for demo, log in demos.items()}
        rows = [
            {"demo": demo}
            | demo_metrics(log, hands[demo], start_of[demo], task, run["recording"]["hold"])
            | sampling_metrics(demos[demo.removesuffix("_filtered")])
            | {"take": str(take_of[demo].relative_to(storage_root())), "take_sha256": sha256_of(take_of[demo])}
            for demo, log in demos.items()
        ]
        with (run_dir / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if several_targets:
            print(f"\nThe takes{of[name]}:")
        print_metrics(rows)
        plot_take(demos, hands, task, rows, title=name).savefig(run_dir / "demonstrations.png", dpi=150)

    print()
    for name, run_dir in run_dirs.items():
        n_takes = len(imports[name][0]) // 2
        print(f"Imported {n_takes} take{'s' if n_takes > 1 else ''}{of[name]} to {run_dir}")
    first = next(iter(run_dirs.values()))
    player = (REPO_ROOT / "third_party/skelarm/tools/player.py").relative_to(REPO_ROOT)
    print(f"Replay them with:\n  uv run python {player} {first / 'demo_00.sklog.npz'}")
    print(f"  uv run python {player} {first / 'demo_00_filtered.sklog.npz'}")


def listed_takes(recording: dict[str, Any]) -> list[Path]:
    """The takes that ``[recording]`` names: one ``take``, or a list of ``takes``, relative to the storage root."""
    if ("take" in recording) == ("takes" in recording):
        msg = "[recording] names one take, or lists several takes, not both"
        raise ValueError(msg)
    return [resolve_run_path(path) for path in recording.get("takes", [recording.get("take")])]


def target_configs(
    config: dict[str, Any], takes: list[Path], stem: str
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """A configuration of one target for each target that has takes, by its run's name; and why a take has none.

    ``config`` has a multi-target ``[task]``. A take reaches the target its log
    records as active, from its own first sample. Each target's configuration is
    ``config`` with a ``reaching`` ``[task]`` toward that target, its takes in
    ``[recording]`` ``takes`` (relative to the storage root, in the order of
    ``takes``), and where each begins in ``[demonstrations]`` ``start_q`` (deg). A take
    whose log records no multi-target task, or other targets than ``config``'s, has
    no target: why is given by the take's file name instead.
    """
    targets = config["task"]["targets"]
    if any("label" not in target for target in targets):
        msg = "each [task] target needs a label, which names its run"
        raise ValueError(msg)
    places = target_places(config["task"])
    by_target: dict[int, list[Path]] = {}
    start_of: dict[Path, list[float]] = {}
    problems: dict[str, list[str]] = {}
    for take in takes:
        log = StateLog.load(take)
        recorded = log.extra.get("playback", {}).get("task", {})
        if recorded.get("type") != MULTI_TARGET_TASK:
            problems[take.name] = ["the take records no multi-target [task], so not which target it reaches"]
        elif target_places(recorded) != places:
            problems[take.name] = ["the take records other targets than the configuration's [task]"]
        else:
            by_target.setdefault(active_target_index(Task.from_dict(recorded)), []).append(take)
            start_of[take] = np.degrees(joint_angles(log)[0]).tolist()
    rest = {key: value for key, value in config["task"].items() if key not in MULTI_TARGET_KEYS}
    root = storage_root()
    configs = {}
    for index, target_takes in sorted(by_target.items()):
        configs[f"{stem}_{targets[index]['label'].lower()}"] = config | {
            "recording": config["recording"] | {"takes": [str(take.relative_to(root)) for take in target_takes]},
            "demonstrations": {"start_q": [start_of[take] for take in target_takes]},
            "task": {"type": "reaching", "target": targets[index]} | rest,
        }
    return configs, problems


def target_places(task: dict[str, Any]) -> list[tuple[float | None, ...]]:
    """The position and the tolerance of each target of a multi-target ``[task]`` table."""
    return [(*pos.tolist(), tolerance) for pos, _, _, tolerance in multi_target_specs(Task.from_dict(task))]


def target_text(config_path: Path, config: dict[str, Any]) -> str:
    """The ``config.toml`` of one target's run: its configuration, under a comment that says where it comes from."""
    return (
        f"# The takes of target {config['task']['target']['label']} of {config_path.name}, as a configuration of\n"
        "# that target alone, written by experiments/import_demonstrations.py: [demonstrations]\n"
        "# start_q lists where each take of [recording] takes begins (deg), as recorded.\n\n"
    ) + tomli_w.dumps(config)


def load_demos(
    config: dict[str, Any], skeleton: Skeleton
) -> tuple[dict[str, StateLog], dict[str, list[float]], dict[str, Path]]:
    """The demonstrations of a configuration of one target, by name: each take as recorded and filtered.

    Also each one's start posture, from ``[demonstrations]`` ``start_q`` (deg), and its take.
    """
    recording = config["recording"]
    takes = listed_takes(recording)
    starts_deg = config["demonstrations"]["start_q"]
    if len(starts_deg) != len(takes):
        msg = f"[demonstrations] start_q lists {len(starts_deg)} start postures for {len(takes)} takes"
        raise ValueError(msg)
    demos: dict[str, StateLog] = {}
    start_of: dict[str, list[float]] = {}
    take_of: dict[str, Path] = {}
    for i, (take, start_deg) in enumerate(zip(takes, starts_deg, strict=True)):
        recorded = StateLog.load(take)
        for name, log in (
            (f"demo_{i:02d}", recorded),
            (f"demo_{i:02d}_filtered", smooth_take(recorded, skeleton, config["task"], recording["filter"])),
        ):
            demos[name], start_of[name], take_of[name] = log, start_deg, take
    return demos, start_of, take_of


def check_demos(
    demos: dict[str, StateLog], start_of: dict[str, list[float]], skeleton: Skeleton, config: dict[str, Any]
) -> dict[str, list[str]]:
    """Why each demonstration of a configuration of one target cannot be one, by name: nothing if it can.

    See :func:`arm_esn_ctrl.demonstrations.check_take`. A configuration without
    ``start_tolerance_deg``, as that of each target of takes of several targets,
    whose takes begin where the arm was posed, has no start posture to check.
    """
    recording = config["recording"]
    task = Task.from_dict(config["task"])
    if task.target is not None and task.tolerance is None:
        msg = "the [task] target needs a tolerance, which is the goal radius"
        raise ValueError(msg)
    return {
        name: check_take(
            log,
            skeleton,
            start_q=np.radians(start_of[name]),
            target=task.target,
            radius=task.tolerance or 0.0,
            hold=recording["hold"],
            start_tolerance=float(np.radians(recording.get("start_tolerance_deg", np.inf))),
            max_tip_speed=recording["max_tip_speed"],
        )
        for name, log in demos.items()
    }


def joint_angles(log: StateLog) -> NDArray[np.float64]:
    """The joint angles of a log, shape ``(n, joints)`` (rad)."""
    return log.channel("q").reshape(len(log.times), -1)


def demo_metrics(
    log: StateLog, hand: NDArray[np.float64], start_deg: list[float], task: Task, hold: float
) -> dict[str, Any]:
    """A demonstration's start, reach metrics, onset and arrival, and joint jitter (5 samples).

    A task without a target has no reach to measure: only the start, the onset, and the jitter.
    """
    times = log.times
    onset = onset_index(hand)
    reach: dict[str, Any] = {}
    arrival = float("nan")
    if task.target is not None:
        reach = reach_metrics(times, hand, task.target)
        arrival = hold_metrics(times, hand, task.target, task.tolerance or 0.0, hold)["arrival_time_s"]
    return (
        joint_columns("start_q{j}_deg", start_deg)
        | reach
        | {
            "onset_time_s": float("nan") if onset is None else float(times[onset]),
            "arrival_time_s": arrival,
            "jitter_deg": float(np.degrees(jitter(joint_angles(log), JITTER_WINDOW))),
        }
    )


def sampling_metrics(log: StateLog) -> dict[str, Any]:
    """How the take was sampled: the number of samples, the length, the rates, and the longest interval."""
    times = log.times
    acquisition = log.extra.get("acquisition", {})
    return {
        "samples": len(times),
        "length_s": float(times[-1] - times[0]),
        "requested_rate_hz": acquisition.get("requested_rate_hz", float("nan")),
        "achieved_rate_hz": acquisition.get("achieved_rate_hz", float("nan")),
        "longest_interval_s": float(np.diff(times).max()),
    }


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file, in hexadecimal."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def print_metrics(rows: list[dict[str, Any]]) -> None:
    """Print each demonstration's timing and reach metrics, then how each take was sampled."""
    for row in rows:
        if "movement_time" not in row:  # a task without a target
            print(
                f"{row['demo']}: moves at {row['onset_time_s']:.2f} s of {row['length_s']:.2f} s;"
                f" joint jitter {row['jitter_deg']:.3f} deg"
            )
            continue
        print(
            f"{row['demo']}: moves at {row['onset_time_s']:.2f} s and arrives at {row['arrival_time_s']:.2f} s"
            f" of {row['length_s']:.2f} s; movement {row['movement_time']:.2f} s, peak speed"
            f" {row['peak_speed']:.2f} m/s, {int(row['speed_peaks'])} speed peaks, joint jitter"
            f" {row['jitter_deg']:.3f} deg, path deviation {row['path_deviation']:.1%},"
            f" final error {1000 * row['final_error']:.1f} mm"
        )
    recorded = [row for row in rows if not row["demo"].endswith("_filtered")]
    for row in recorded:
        name = "The take" if len(recorded) == 1 else f"The take of {row['demo']}"
        print(
            f"{name}: {row['samples']} samples at {row['achieved_rate_hz']:.1f} Hz of {row['requested_rate_hz']:g} Hz"
            f" requested; longest interval {1000 * row['longest_interval_s']:.1f} ms"
        )


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def plot_take(
    demos: dict[str, StateLog],
    hands: dict[str, NDArray[np.float64]],
    task: Task,
    rows: list[dict[str, Any]],
    title: str,
) -> Figure:
    """The hand path, the joint angles and hand speed over time of each demonstration, and the takes' sampling.

    With one take, the dotted lines mark when its hand starts to move and when it
    arrives. With several, each take has its own color, its recorded hand path
    gray under the filtered one, and only the filtered joint angles and speeds are
    drawn.
    """
    several = len(demos) > 2
    fig = Figure(figsize=(12, 8.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"Demonstration{'s' if several else ''} taught by hand: {title}", color="#0b0b0b")
    ax_hand, ax_joint, ax_speed, ax_interval = fig.subplots(2, 2).flat
    for ax in (ax_hand, ax_joint, ax_speed, ax_interval):
        style(ax)

    for name, log in demos.items():
        times, hand = log.times, hands[name]
        filtered = name.endswith("_filtered")
        take = int(name.split("_")[1])
        if several:
            color = TAKE_COLORS[take % len(TAKE_COLORS)] if filtered else RECORDED_COLOR
            width = 1.2 if filtered else 2.2
            label = f"take {take:02d}" if filtered else None
        else:
            color, width, label = DEMO_STYLES[name]
        q = np.degrees(joint_angles(log))
        ax_hand.plot(hand[:, 0], hand[:, 1], color=color, linewidth=width, label=label)
        if several and not filtered:
            ax_hand.plot(*hand[0], marker="o", markersize=5, color="#0b0b0b")
            ax_interval.plot(times[1:], 1000 * np.diff(times), color=TAKE_COLORS[take % len(TAKE_COLORS)], linewidth=1)
            continue
        for j, (linestyle, _) in zip(range(q.shape[1]), JOINT_STYLES, strict=False):
            ax_joint.plot(times, q[:, j], color=color, linestyle=linestyle, linewidth=width)
        ax_speed.plot(times, hand_speed(times, hand), color=color, linewidth=width)
    if not several:
        ax_hand.plot(*hands["demo_00"][0], marker="o", markersize=6, color="#0b0b0b")
    if task.target is not None:
        ax_hand.plot(*task.target, marker="+", markersize=12, color="#0b0b0b", markeredgewidth=1.5)
    if task.target is not None and task.tolerance is not None:
        ring = np.linspace(0.0, 2 * np.pi, 100)
        ax_hand.plot(
            task.target[0] + task.tolerance * np.cos(ring),
            task.target[1] + task.tolerance * np.sin(ring),
            color="#0b0b0b",
            linewidth=0.8,
        )
    marks = "circle: start; cross: target" if task.target is not None else "circle: start"
    ax_hand.set(title=f"Hand path ({marks})", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax_hand.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=7 if several else None)
    n_joints = joint_angles(demos["demo_00"]).shape[1]
    styles = ", ".join(name for _, name in JOINT_STYLES[:n_joints])
    ax_joint.set(title=f"Joint angles (joints 1 to {n_joints}: {styles})", xlabel="time (s)", ylabel="angle (deg)")
    ax_speed.set(title="Hand speed", xlabel="time (s)", ylabel="speed (m/s)")
    if not several:
        for ax in (ax_joint, ax_speed):
            for key in ("onset_time_s", "arrival_time_s"):
                if np.isfinite(rows[0][key]):
                    ax.axvline(rows[0][key], color=TEXT_COLOR, linewidth=0.8, linestyle=":")
        times = demos["demo_00"].times
        ax_interval.plot(times[1:], 1000 * np.diff(times), color=TAKE_COLOR, linewidth=1.0)
    requested = rows[0]["requested_rate_hz"]
    if np.isfinite(requested):
        ax_interval.axhline(1000 / requested, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax_interval.set(
        title=f"Sampling interval of the take{'s' if several else ''} (dotted: requested)",
        xlabel="time (s)",
        ylabel="interval (ms)",
    )
    return fig


if __name__ == "__main__":
    main()
