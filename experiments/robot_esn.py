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
t = 0. An optional ``[disturbance]`` table pushes or blocks all three arms alike
(see :mod:`arm_esn_ctrl.disturbances`); start postures away from the demonstrated
ones (``start_offsets_deg`` in ``[evaluation]``) make the initial-offset scenario.

Every run is compared with the demonstrator's undisturbed reach from the same
start posture, and the run directory receives:

- ``<law>_w<omega>/esn_00.sklog.npz``, ``replay_00.sklog.npz``, ...: the arm's runs,
  with the reference (``q_ref``), the tracking error (``error``), and the
  disturbance (``ext_force``);
- ``demonstrator_00.sklog.npz``, ...: the demonstrator's reaches, under the same disturbance;
- ``metrics.csv``: the metrics of every run (see :func:`arm_metrics`);
- ``metrics.png``: those metrics against the natural frequency, for each law;
- ``paths.png``: the hand paths of every run;
- ``timeline.png``: the hand's distance to the target and the joint torque over
  time, from the first start posture.

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
from arm_esn_ctrl.demonstrations import endpoint_positions, simulate_disturbed_reach
from arm_esn_ctrl.disturbances import make_disturbance
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.metrics import hand_speed
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

ARMS = ("esn", "replay", "demonstrator")
ARM_COLORS = {"esn": "#2a78d6", "replay": "#eb6834", "demonstrator": "#52514e"}
ARM_LABELS = {
    "esn": "ESN on the measured posture",
    "replay": "demonstration replayed by time",
    "demonstrator": "demonstrator",
}
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"


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
        demo_config = tomllib.load(f)
    evaluation = config["evaluation"]
    setup = load_setup({"demonstrations": demonstrations, "esn": {"dt": esn.config.dt}, "evaluation": evaluation})
    disturbance = config.get("disturbance")
    window = evaluation.get("effort_window", [0.0, evaluation["duration"]])
    print(f"ESN {config['esn']['model']}, trained on {len(setup.demos)} demonstrations")
    print(f"Disturbance: {disturbance or 'none'}; {len(setup.starts)} start postures")

    def disturbance_from(i: int) -> Any:
        """A fresh disturbance for a run from start posture ``i``."""
        return make_disturbance(disturbance, setup.hand_refs[i][0], setup.target)

    rows = []
    for i, start in enumerate(setup.starts):
        log = setup.demonstrator_logs[i]
        if disturbance is not None:
            log = simulate_disturbed_reach(demo_config, start.q, evaluation["duration"], disturbance_from(i))
        log.save(run_dir / f"demonstrator_{i:02d}.sklog.npz")
        rows.append(
            {"law": "", "omega": "", "arm": "demonstrator", "start": i, "origin": start.origin, "replayed": ""}
            | arm_metrics(log, i, setup, esn.config.dt, window)
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
                        dt=demo_config["simulator"]["dt"],
                        enforce_limits=demo_config["simulator"].get("enforce_limits", True),
                        extra={
                            "playback": {"task": setup.task},
                            "tracking": {"reference": arm, "law": law, "omega": float(omega), "start": start.origin},
                        },
                        external_force=disturbance_from(i),
                    )
                    log.save(setting_dir / f"{arm}_{i:02d}.sklog.npz")
                    rows.append(
                        {"law": law, "omega": float(omega), "arm": arm, "start": i, "origin": start.origin}
                        | {"replayed": replayed if arm == "replay" else ""}
                        | arm_metrics(log, i, setup, esn.config.dt, window)
                    )
            print(f"Ran {law} at omega = {omega:g} rad/s (kp = {np.round(gains[0], 2).tolist()})")

    with (run_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows, len(setup.starts), window)

    title = f"ESN on the robot: {args.config.stem}"
    laws, omegas = tracker["laws"], tracker["omegas"]
    plot_metrics(rows, laws, len(setup.starts), window, title).savefig(run_dir / "metrics.png", dpi=150)
    plot_paths(run_dir, setup, laws, omegas, title).savefig(run_dir / "paths.png", dpi=150)
    span = disturbance_span(disturbance)
    plot_timeline(run_dir, setup, laws, omegas, span, title).savefig(run_dir / "timeline.png", dpi=150)
    print(f"\nWrote the results to {run_dir}")
    first = f"{laws[0]}_w{omegas[0]:g}"
    print(f"Replay with:\n  uv run python third_party/skelarm/tools/player.py {run_dir / first / 'esn_00.sklog.npz'}")


def posed(skeleton: Skeleton, q: NDArray[np.float64]) -> Skeleton:
    """A copy of the robot at the joint angles ``q``, at rest."""
    arm = skeleton.clone()
    arm.q = q
    arm.dq = np.zeros_like(q)
    return arm


def arm_metrics(log: StateLog, i: int, setup: Setup, period: float, window: list[float]) -> dict[str, Any]:
    """The metrics of one arm's run from start posture ``i``, compared with the demonstrator's undisturbed reach.

    Besides the reach and hold metrics of Stage 1 (see
    :func:`arm_esn_ctrl.autonomous.run_metrics`), over the task (t >= 0):

    - ``tracking_error_deg``: RMS of the tracking error ``q_ref - q``;
    - ``peak_reference_speed_dps``: the reference's fastest joint speed, which
      shows a jump of the reference (as a jump of a few degrees in one period);
    - ``peak_hand_speed_mps``: the hand's fastest speed, sampled every period;

    and over the effort window (``effort_window`` in ``[evaluation]``):

    - ``peak_torque_nm``: the largest joint torque;
    - ``effort_n2m2s``: the integral of the squared joint torques, summed over the joints;
    - ``peak_external_force_n``: the largest disturbance force at the tip (a
      block's holding force shows how hard the arm pushes against it).

    The demonstrator tracks no reference, so its tracking metrics are NaN.
    """
    q = task_joint_angles(log, period, setup.times[-1])
    hand = endpoint_positions(setup.skeleton, q)
    run = Run(setup.starts[i], q, setup.q_refs[i], hand, setup.hand_refs[i])
    times = log.times
    task = times > -1e-9
    in_window = (times > window[0] - 1e-9) & (times < window[1] + 1e-9)
    tracked = "q_ref" in log.channel_names
    nan = float("nan")
    tau = log.channel("tau")
    force = log.channel("ext_force") if "ext_force" in log.channel_names else np.zeros((len(times), 2))
    reference_speed = nan
    if tracked:
        q_ref = log.channel("q_ref")[task]
        reference_speed = float(np.degrees(np.abs(np.diff(q_ref, axis=0)).max() / np.diff(times[task]).min()))
    return run_metrics(run, setup) | {
        "tracking_error_deg": rms_degrees(log.channel("error")[task]) if tracked else nan,
        "peak_reference_speed_dps": reference_speed,
        "peak_hand_speed_mps": float(hand_speed(setup.times, hand).max()),
        "peak_torque_nm": float(np.abs(tau[in_window]).max()),
        "effort_n2m2s": float(np.sum(tau[in_window] ** 2) * np.diff(times).mean()),
        "peak_external_force_n": float(np.linalg.norm(force[in_window], axis=1).max()),
    }


def disturbance_span(disturbance: dict[str, Any] | None) -> tuple[float, float] | None:
    """When the disturbance acts on the task clock, or None without one."""
    if disturbance is None:
        return None
    if disturbance["type"] == "push":
        return disturbance["onset"], disturbance["onset"] + disturbance["duration"]
    return disturbance["onset"], disturbance["release"]


def print_summary(rows: list[dict[str, Any]], n_starts: int, window: list[float]) -> None:
    """Print the metrics of each arm and tracker setting: means over the start postures."""
    print(f"\nMeans over the {n_starts} start postures (torque, effort, force: t = {window[0]:g} to {window[1]:g} s)")
    print("                        ------- reach -------  hold   tracking  ref.   hand   peak    effort    peak")
    print(
        "law              omega arm          error  path  delay  succ.  error     speed  speed  torque  (N2 m2 s) force"
    )
    print(
        "                 (rad/s)            (deg)  (mm)  (s)           (deg)     (dps)  (m/s)  (N m)             (N)"
    )
    for key in dict.fromkeys((r["law"], r["omega"], r["arm"]) for r in rows):
        group = [r for r in rows if (r["law"], r["omega"], r["arm"]) == key]
        law, omega, arm = key
        m = {name: mean_of(group, name) for name in group[0] if isinstance(group[0][name], float)}
        print(
            f"{law or '-':<16} {f'{omega:g}' if omega != '' else '-':>5} {arm:<12}"
            f" {m['reach_joint_error_deg']:5.2f} {1000 * m['reach_path_distance_m']:5.1f} {m['arrival_delay_s']:+5.2f}"
            f" {sum(r['success'] for r in group):2d}/{n_starts:<3d} {m['tracking_error_deg']:6.3f}"
            f" {m['peak_reference_speed_dps']:8.0f} {m['peak_hand_speed_mps']:6.2f} {m['peak_torque_nm']:7.1f}"
            f" {m['effort_n2m2s']:9.1f} {m['peak_external_force_n']:6.1f}"
        )


def mean_of(rows: list[dict[str, Any]], name: str) -> float:
    """The mean of a metric over ``rows``, ignoring NaN (NaN if all are)."""
    values = np.array([r[name] for r in rows], dtype=np.float64)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float("nan")


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def plot_metrics(rows: list[dict[str, Any]], laws: list[str], n_starts: int, window: list[float], title: str) -> Figure:
    """Plot each metric's mean over the start postures (and its range) against the natural frequency.

    Each law takes two rows of panels. The demonstrator, which has no gains, is a
    dashed level.
    """
    effort = f"over {window[0]:g}-{window[1]:g} s"
    metrics = [
        ("reach_joint_error_deg", "Joint error from the demonstrator\nduring the reach (deg RMS)"),
        ("reach_path_distance_m", "Path distance from the demonstrator\nduring the reach (m)"),
        ("arrival_delay_s", "Arrival delay (s)"),
        ("success", f"Successes: arrive and hold\n(of {n_starts} start postures)"),
        ("peak_reference_speed_dps", "Peak reference joint speed (deg/s)"),
        ("peak_hand_speed_mps", "Peak hand speed (m/s)"),
        ("peak_torque_nm", f"Peak joint torque (N m)\n{effort}"),
        ("effort_n2m2s", f"Integral of squared torque (N² m² s)\n{effort}"),
    ]
    per_row = 4
    fig = Figure(figsize=(14, 6.2 * len(laws) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color="#0b0b0b")
    axes = np.asarray(fig.subplots(2 * len(laws), per_row)).reshape(len(laws), 2 * per_row)
    for law, law_axes in zip(laws, axes, strict=True):
        for (key, label), ax in zip(metrics, law_axes, strict=True):
            style(ax)
            for arm in ("esn", "replay"):
                group = [r for r in rows if r["law"] == law and r["arm"] == arm]
                omegas = sorted({r["omega"] for r in group})
                values = [[float(r[key]) for r in group if r["omega"] == omega] for omega in omegas]
                if key == "success":
                    low = high = mean = [sum(v) for v in values]
                else:
                    low, high = [np.nanmin(v) for v in values], [np.nanmax(v) for v in values]
                    mean = [np.nanmean(v) for v in values]
                color = ARM_COLORS[arm]
                ax.fill_between(omegas, low, high, color=color, alpha=0.15, linewidth=0)
                ax.plot(omegas, mean, marker="o", markersize=5, linewidth=2, color=color, label=ARM_LABELS[arm])
                ax.set_xscale("log")
                ax.set_xticks(omegas, [f"{omega:g}" for omega in omegas])
                ax.minorticks_off()
            demonstrator = [float(r[key]) for r in rows if r["arm"] == "demonstrator"]
            if np.isfinite(demonstrator).any():
                level = sum(demonstrator) if key == "success" else np.nanmean(demonstrator)
                ax.axhline(
                    level,
                    color=ARM_COLORS["demonstrator"],
                    linestyle="--",
                    linewidth=1.2,
                    label=ARM_LABELS["demonstrator"],
                )
            if key == "success":
                ax.set_ylim(-0.5, n_starts + 0.5)
            ax.set_title(label, color=TEXT_COLOR, fontsize=9)
            ax.set_xlabel("natural frequency ω (rad/s)", color=TEXT_COLOR, fontsize=8)
        law_axes[0].set_ylabel(law.replace("_", " "), color="#0b0b0b", fontsize=11)
        law_axes[per_row].set_ylabel(law.replace("_", " "), color="#0b0b0b", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def load_hand(path: Path, setup: Setup) -> tuple[NDArray[np.float64], NDArray[np.float64], StateLog]:
    """The times and hand positions of a saved run, every simulation step, and the log."""
    log = StateLog.load(path)
    return log.times, endpoint_positions(setup.skeleton, log.channel("q").reshape(len(log.times), -1)), log


def plot_paths(run_dir: Path, setup: Setup, laws: list[str], omegas: list[float], title: str) -> Figure:
    """Plot the hand paths of every run, over the demonstrator's undisturbed reaches, for every setting."""
    fig = Figure(figsize=(3.2 * len(omegas), 3.3 * len(laws) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color="#0b0b0b")
    axes = np.atleast_2d(fig.subplots(len(laws), len(omegas), sharex=True, sharey=True))
    n_starts = len(setup.starts)
    for law, ax_row in zip(laws, axes, strict=True):
        for omega, ax in zip(omegas, ax_row, strict=True):
            style(ax)
            for i, hand_ref in enumerate(setup.hand_refs):
                label = "demonstrator, undisturbed" if i == 0 else None
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
                    _, hand, _ = load_hand(run_dir / f"{law}_w{omega:g}" / f"{arm}_{i:02d}.sklog.npz", setup)
                    ax.plot(*hand.T, color=ARM_COLORS[arm], linewidth=1.2, label=ARM_LABELS[arm] if i == 0 else None)
            ax.plot(*setup.target, marker="+", markersize=12, color="#0b0b0b", markeredgewidth=1.5)
            ax.set_title(f"{law.replace('_', ' ')}, ω = {omega:g} rad/s", color=TEXT_COLOR, fontsize=10)
            ax.set_aspect("equal")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = 3 if len(omegas) > 2 else 1  # a narrow figure stacks the legend
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_timeline(
    run_dir: Path, setup: Setup, laws: list[str], omegas: list[float], span: tuple[float, float] | None, title: str
) -> Figure:
    """Plot the hand's distance to the target and the joint torque over time, from the first start posture.

    The shaded band is when the disturbance acts.
    """
    fig = Figure(figsize=(3.4 * len(omegas), 5.0 * len(laws) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"{title}, from {setup.starts[0].origin}", color="#0b0b0b")
    axes = np.asarray(fig.subplots(2 * len(laws), len(omegas), sharex=True)).reshape(2 * len(laws), len(omegas))
    reference = setup.demonstrator_logs[0]
    ref_times = reference.times
    ref_hand = endpoint_positions(setup.skeleton, reference.channel("q").reshape(len(ref_times), -1))
    demonstrator = load_hand(run_dir / "demonstrator_00.sklog.npz", setup)
    for row, law in enumerate(laws):
        for col, omega in enumerate(omegas):
            ax_distance, ax_torque = axes[2 * row, col], axes[2 * row + 1, col]
            runs = {
                arm: load_hand(run_dir / f"{law}_w{omega:g}" / f"{arm}_00.sklog.npz", setup)
                for arm in ("esn", "replay")
            }
            runs["demonstrator"] = demonstrator
            for ax in (ax_distance, ax_torque):
                style(ax)
                if span is not None:
                    ax.axvspan(*span, color=DISTURBANCE_COLOR, zorder=0)
            ax_distance.plot(
                ref_times,
                1000 * np.linalg.norm(ref_hand - setup.target, axis=1),
                color=ARM_COLORS["demonstrator"],
                linewidth=4,
                alpha=0.3,
                label="demonstrator, undisturbed",
            )
            for arm in ARMS:
                times, hand, log = runs[arm]
                color = ARM_COLORS[arm]
                ax_distance.plot(
                    times,
                    1000 * np.linalg.norm(hand - setup.target, axis=1),
                    color=color,
                    linewidth=1.3,
                    label=ARM_LABELS[arm],
                )
                ax_torque.plot(times, np.linalg.norm(log.channel("tau"), axis=1), color=color, linewidth=1.3)
            ax_distance.axhline(1000 * setup.radius, color="#0b0b0b", linewidth=0.8, linestyle=":")
            ax_distance.set(xlim=(-0.3, 3.0), ylim=(0, None))
            ax_distance.set_title(f"{law.replace('_', ' ')}, ω = {omega:g} rad/s", color=TEXT_COLOR, fontsize=10)
            ax_torque.set_xlabel("time (s)", color=TEXT_COLOR, fontsize=8)
            if col == 0:
                ax_distance.set_ylabel("hand to target (mm)", color=TEXT_COLOR)
                ax_torque.set_ylabel("joint torque norm (N m)", color=TEXT_COLOR)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = 4 if len(omegas) > 2 else 2  # a narrow figure wraps the legend
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, labelcolor=TEXT_COLOR)
    return fig


if __name__ == "__main__":
    main()
