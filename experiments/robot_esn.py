# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Run a trained ESN as the reference generator of the simulated robot arm (Stage 2).

    uv run python experiments/robot_esn.py configs/robot/nominal.toml

The arm tracks a reference generated every reference period (the ESN's 10 ms)
from its measured joint angles (see :mod:`arm_esn_ctrl.tracking`). Three arms
reach from each start posture:

- ``esn``: the ESN, driven by the arm's measured joint angles;
- ``replay``: the demonstration that starts nearest, replayed by time (the
  time-indexed baseline);
- ``demonstrator``: the controller that made the demonstrations, on its own.

The ESN and the replay run with every tracking law and natural frequency of the
tracking error listed in ``[tracker]``. Their runs start with the arm holding its
start posture for the ESN's warm-up (at negative times), and the task starts at
t = 0. Every run is compared with the demonstrator's reach from the same start
posture, and the run directory receives:

- ``<law>_w<omega>/esn_00.sklog.npz``, ``replay_00.sklog.npz``, ...: the arm's runs,
  with the reference (``q_ref``) and the tracking error (``error``);
- ``demonstrator_00.sklog.npz``, ...: the demonstrator's reaches;
- ``metrics.csv``: for each run, the reach and hold metrics of Stage 1 (see
  :func:`arm_esn_ctrl.autonomous.run_metrics`), the RMS tracking error, and the
  peak joint torque, both over the task (t >= 0);
- ``metrics.png``: those metrics against the natural frequency, for each law;
- ``paths.png``: the hand paths of every run.

Replay a run with ``uv run python third_party/skelarm/tools/player.py <file>``.
"""

from __future__ import annotations

import argparse
import csv
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.autonomous import Run, Setup, load_setup, rms_degrees, run_metrics
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.storage import start_run, storage_root
from arm_esn_ctrl.tracking import (
    EsnSource,
    ReferenceSource,
    ReplaySource,
    TrackerConfig,
    nearest_demonstration,
    task_joint_angles,
    track,
    tracking_gains,
)

ARM_COLORS = {"esn": "#2a78d6", "replay": "#eb6834", "demonstrator": "#52514e"}
ARM_LABELS = {
    "esn": "ESN on the measured posture",
    "replay": "demonstration replayed by time",
    "demonstrator": "demonstrator",
}
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    esn_path = storage_root() / config["esn"]["model"]
    esn = ReachingEsn.load(esn_path)
    # The demonstrations the ESN was trained on, which the replay replays.
    with (esn_path.parent / "config.toml").open("rb") as f:
        demonstrations = tomllib.load(f)["demonstrations"]
    with (storage_root() / demonstrations["run"] / "config.toml").open("rb") as f:
        simulator = tomllib.load(f)["simulator"]
    evaluation = config["evaluation"]
    setup = load_setup({"demonstrations": demonstrations, "esn": {"dt": esn.config.dt}, "evaluation": evaluation})
    print(f"ESN {config['esn']['model']}, trained on {len(setup.demos)} demonstrations")

    rows = []
    for i, (start, log) in enumerate(zip(setup.starts, setup.demonstrator_logs, strict=True)):
        log.save(run_dir / f"demonstrator_{i:02d}.sklog.npz")
        rows.append(
            {
                "law": "",  # the demonstrator's own controller, without a reference
                "omega": "",
                "arm": "demonstrator",
                "start": i,
                "origin": start.origin,
                "replayed": "",
            }
            | arm_metrics(log, i, setup, esn.config.dt, tracked=False)
        )

    tracker = config["tracker"]
    end_posture = np.mean([q[-1] for q in setup.demos.values()], axis=0)  # where the reaches end
    for law in tracker["laws"]:
        for omega in tracker["omegas"]:
            setting = TrackerConfig(law, float(omega), tracker["acceleration_filter"])
            gains = tracking_gains(setting, setup.skeleton, end_posture)
            setting_dir = run_dir / f"{law}_w{omega:g}"
            setting_dir.mkdir()
            for i, start in enumerate(setup.starts):
                replayed = nearest_demonstration(start.q, setup.demos)
                sources: dict[str, ReferenceSource] = {
                    "esn": EsnSource(esn),
                    "replay": ReplaySource(setup.demos[replayed]),
                }
                for arm, source in sources.items():
                    log = track(
                        posed(setup.skeleton, start.q),
                        source,
                        setting,
                        gains,
                        period=esn.config.dt,
                        warmup_steps=esn.config.warmup_steps,
                        duration=evaluation["duration"],
                        dt=simulator["dt"],
                        enforce_limits=simulator.get("enforce_limits", True),
                        extra={
                            "playback": {"task": setup.task},
                            "tracking": {"reference": arm, "law": law, "omega": float(omega), "start": start.origin},
                        },
                    )
                    log.save(setting_dir / f"{arm}_{i:02d}.sklog.npz")
                    rows.append(
                        {"law": law, "omega": float(omega), "arm": arm, "start": i, "origin": start.origin}
                        | {"replayed": replayed if arm == "replay" else ""}
                        | arm_metrics(log, i, setup, esn.config.dt, tracked=True)
                    )
            print(f"Ran {law} at omega = {omega:g} rad/s (kp = {np.round(gains[0], 2).tolist()})")

    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows, len(setup.starts))

    title = f"ESN on the robot: {args.config.stem}"
    plot_metrics(rows, tracker["laws"], title).savefig(run_dir / "metrics.png", dpi=150)
    plot_paths(run_dir, setup, tracker["laws"], tracker["omegas"], title).savefig(run_dir / "paths.png", dpi=150)
    print(f"\nWrote the results to {run_dir}")
    first = f"{tracker['laws'][0]}_w{tracker['omegas'][0]:g}"
    print(f"Replay with:\n  uv run python third_party/skelarm/tools/player.py {run_dir / first / 'esn_00.sklog.npz'}")


def posed(skeleton: Skeleton, q: NDArray[np.float64]) -> Skeleton:
    """A copy of the robot at the joint angles ``q``, at rest."""
    arm = skeleton.clone()
    arm.q = q
    arm.dq = np.zeros_like(q)
    return arm


def arm_metrics(log: StateLog, i: int, setup: Setup, period: float, *, tracked: bool) -> dict[str, Any]:
    """The metrics of one arm's run from start posture ``i``, compared with the demonstrator's reach.

    Besides the reach and hold metrics of Stage 1, the RMS tracking error (NaN for
    the demonstrator, which tracks no reference) and the peak joint torque, both
    over the task (t >= 0).
    """
    q = task_joint_angles(log, period, setup.times[-1])
    run = Run(setup.starts[i], q, setup.q_refs[i], endpoint_positions(setup.skeleton, q), setup.hand_refs[i])
    task = log.times > -1e-9
    tracking_error = rms_degrees(log.channel("error")[task]) if tracked else float("nan")
    return run_metrics(run, setup) | {
        "tracking_error_deg": tracking_error,
        "peak_torque_nm": float(np.abs(log.channel("tau")[task]).max()),
    }


def print_summary(rows: list[dict[str, Any]], n_starts: int) -> None:
    """Print the metrics of each arm and tracker setting, over all start postures."""
    print("\n                         --------------- reach ---------------  hold     tracking   peak")
    print("law              omega   arm           joint error  path dist.  arrival  success  error      torque")
    print("                 (rad/s)               (deg RMS)    (mm)        delay (s)         (deg RMS)  (N m)")
    for key in dict.fromkeys((r["law"], r["omega"], r["arm"]) for r in rows):
        group = [r for r in rows if (r["law"], r["omega"], r["arm"]) == key]
        law, omega, arm = key
        print(
            f"{law or '-':<16} {f'{omega:g}' if omega != '' else '-':>5}   {arm:<12}"
            f"  {np.mean([r['reach_joint_error_deg'] for r in group]):11.2f}"
            f"  {1000 * np.mean([r['reach_path_distance_m'] for r in group]):10.1f}"
            f"  {np.nanmean([r['arrival_delay_s'] for r in group]):+8.2f}"
            f"  {sum(r['success'] for r in group):3d} of {n_starts}"
            f"  {np.mean([r['tracking_error_deg'] for r in group]):9.3f}"
            f"  {np.max([r['peak_torque_nm'] for r in group]):6.1f}"
        )
    print("Means over the start postures, except the peak torque (the largest).")


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def plot_metrics(rows: list[dict[str, Any]], laws: list[str], title: str) -> Figure:
    """Plot each metric's mean over the start postures (and its range) against the natural frequency."""
    metrics = [
        ("tracking_error_deg", "Tracking error (deg RMS)"),
        ("reach_joint_error_deg", "Joint error from the demonstrator\nduring the reach (deg RMS)"),
        ("arrival_delay_s", "Arrival delay (s)"),
        ("peak_torque_nm", "Peak joint torque (N m)"),
    ]
    fig = Figure(figsize=(13, 3.2 * len(laws) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color="#0b0b0b")
    axes = np.atleast_2d(fig.subplots(len(laws), len(metrics)))
    demonstrator = [r for r in rows if r["arm"] == "demonstrator"]
    for law, ax_row in zip(laws, axes, strict=True):
        for (key, label), ax in zip(metrics, ax_row, strict=True):
            style(ax)
            for arm in ("esn", "replay"):
                group = [r for r in rows if r["law"] == law and r["arm"] == arm]
                omegas = sorted({r["omega"] for r in group})
                values = [[r[key] for r in group if r["omega"] == omega] for omega in omegas]
                mean = [np.nanmean(v) for v in values]
                color = ARM_COLORS[arm]
                ax.fill_between(
                    omegas,
                    [np.nanmin(v) for v in values],
                    [np.nanmax(v) for v in values],
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
                ax.plot(omegas, mean, marker="o", markersize=5, linewidth=2, color=color, label=ARM_LABELS[arm])
                ax.set_xscale("log")
                ax.set_xticks(omegas, [f"{omega:g}" for omega in omegas])
                ax.minorticks_off()
            if key in ("reach_joint_error_deg", "arrival_delay_s", "peak_torque_nm"):
                level = np.nanmean([r[key] for r in demonstrator])
                ax.axhline(
                    level,
                    color=ARM_COLORS["demonstrator"],
                    linestyle="--",
                    linewidth=1.2,
                    label=ARM_LABELS["demonstrator"],
                )
            ax.set_title(label, color=TEXT_COLOR, fontsize=10)
            ax.set_xlabel("natural frequency ω (rad/s)", color=TEXT_COLOR)
        ax_row[0].set_ylabel(law.replace("_", " "), color="#0b0b0b", fontsize=11)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_paths(run_dir: Path, setup: Setup, laws: list[str], omegas: list[float], title: str) -> Figure:
    """Plot the hand paths of the ESN's and the replay's runs over the demonstrator's, for every setting."""
    fig = Figure(figsize=(3.2 * len(omegas), 3.3 * len(laws) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color="#0b0b0b")
    axes = np.atleast_2d(fig.subplots(len(laws), len(omegas), sharex=True, sharey=True))
    n_starts = len(setup.starts)
    for law, ax_row in zip(laws, axes, strict=True):
        for omega, ax in zip(omegas, ax_row, strict=True):
            style(ax)
            for i, hand_ref in enumerate(setup.hand_refs):
                label = ARM_LABELS["demonstrator"] if i == 0 else None
                ax.plot(
                    *hand_ref.T,
                    color=ARM_COLORS["demonstrator"],
                    linewidth=4,
                    alpha=0.3,
                    label=label,
                    solid_capstyle="round",
                )
            for arm in ("replay", "esn"):
                for i in range(n_starts):
                    log = StateLog.load(run_dir / f"{law}_w{omega:g}" / f"{arm}_{i:02d}.sklog.npz")
                    hand = endpoint_positions(setup.skeleton, log.channel("q").reshape(len(log.times), -1))
                    label = ARM_LABELS[arm] if i == 0 else None
                    ax.plot(*hand.T, color=ARM_COLORS[arm], linewidth=1.2, label=label)
            ax.plot(*setup.target, marker="+", markersize=12, color="#0b0b0b", markeredgewidth=1.5)
            ax.set_title(f"{law.replace('_', ' ')}, ω = {omega:g} rad/s", color=TEXT_COLOR, fontsize=10)
            ax.set_aspect("equal")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = 3 if len(omegas) > 2 else 1  # a narrow figure stacks the legend
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, labelcolor=TEXT_COLOR)
    return fig


if __name__ == "__main__":
    main()
