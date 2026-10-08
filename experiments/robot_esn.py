# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Run a trained ESN as the reference generator of the simulated robot arm (Stage 2).

    uv run python experiments/robot_esn.py experiments/multi_demonstration_robot_tracking/nominal.toml

The arm tracks a reference generated every reference period (the ESN's 10 ms)
from its measured joint angles (see :mod:`arm_esn_ctrl.tracking`). Three arms
reach from each start posture:

- ``esn``: the ESN, driven by the arm's measured joint angles;
- ``replay``: the demonstration that starts nearest, replayed by time (the
  time-indexed baseline);
- ``demonstrator``: the controller that made the demonstrations, on its own.

The ESN and the replay run with every tracking law, natural frequency of the
tracking error, and damping ratio listed in ``[tracker]`` (``dampings`` is
optional; its default, 1, is critically damped). ``omegas`` is a list for every
law, or a table with a list for each law, such as
``omegas = {computed_torque = [10.0], pd = [20.0]}``. With ``reference_velocity =
false`` (optional; the default is true), the tracking laws are given a zero
reference velocity, so that their derivative term damps the arm's own velocity
rather than the velocity error. Their runs start with the arm holding its
start posture for the ESN's warm-up (at negative times), and the task starts at
t = 0. An optional ``[disturbance]`` table, or several ``[[disturbance]]`` tables,
push or block all three arms alike (see :mod:`arm_esn_ctrl.disturbances`): from the
same start posture, the same forces at the same times. Start postures away from the demonstrated
ones (``start_offsets_deg`` in ``[evaluation]``) make the initial-offset scenario.

Every run is compared with the demonstrator's undisturbed reach from the same
start posture, and the run directory receives:

- ``<law>_w<omega>/esn_00.sklog.npz``, ``replay_00.sklog.npz``, ...: the arm's runs
  (``<law>_w<omega>_z<damping>`` below critical damping),
  with the reference (``q_ref``), the tracking error (``error``), and the
  disturbance (``ext_force``);
- ``demonstrator_00.sklog.npz``, ...: the demonstrator's reaches, under the same disturbance;
- ``metrics.csv``: the metrics of every run (see :func:`arm_metrics`);
- ``metrics.png``: those metrics against the natural frequency (or the damping
  ratio, if that is what varies), for each law;
- ``paths.png``: the hand paths of every run;
- ``timeline.png``: the hand's distance to the target, the progress along the
  demonstrated path of the arm and of its reference, and the joint torque over
  time, from the first start posture;
- ``joints.png``: the joint angles over time from the first start posture: the
  demonstrated motion, the ESN's output (its reference), and the arms;
- ``torques.png``: the arms' joint torques over time from the first start posture;
- ``grid.png``, with an ``[evaluation.start_grid]``: maps over the start offsets
  of the outcome, the path distance, the training path ratio, the peak reference
  speed, the peak torque, and the tracking error, for each tracker setting.

Replay a run with ``uv run python third_party/skelarm/tools/player.py <file>``.
"""

from __future__ import annotations

import argparse
import csv
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.autonomous import Reference, Run, Setup, load_setup, rms_degrees, run_metrics
from arm_esn_ctrl.demonstrations import endpoint_positions, simulate_disturbed_reach
from arm_esn_ctrl.disturbances import disturbance_spans, make_disturbance
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.metrics import hand_speed, path_progress, path_rmse
from arm_esn_ctrl.storage import resolve_run_path, start_run
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
    esn_path = resolve_run_path(config["esn"]["model"])
    esn = ReachingEsn.load(esn_path)
    # The demonstrations the ESN was trained on, which the replay replays.
    with (esn_path.parent / "config.toml").open("rb") as f:
        demonstrations = tomllib.load(f)["demonstrations"]
    with (resolve_run_path(demonstrations["run"]) / "config.toml").open("rb") as f:
        demo_config = tomllib.load(f)
    evaluation = config["evaluation"]
    setup = load_setup({"demonstrations": demonstrations, "esn": {"dt": esn.config.dt}, "evaluation": evaluation})
    disturbance = config.get("disturbance")
    spans = disturbance_spans(disturbance)
    window = evaluation.get("effort_window", [0.0, evaluation["duration"]])
    print(f"ESN {config['esn']['model']}, trained on {len(setup.demos)} demonstrations")
    print(f"Disturbance: {disturbance or 'none'}; {len(setup.starts)} start postures")
    if not config["tracker"].get("reference_velocity", True):
        print("The tracking laws are given a zero reference velocity")

    def disturbance_from(i: int) -> Any:
        """A fresh disturbance for a run from start posture ``i``."""
        start_hand = endpoint_positions(setup.skeleton, setup.starts[i].q[np.newaxis])[0]
        return make_disturbance(disturbance, start_hand, setup.target)

    rows = []
    for i, start in enumerate(setup.starts):
        if not setup.demonstrator_logs:  # a demonstration taught by hand: no demonstrator to run
            break
        log = setup.demonstrator_logs[i]
        if disturbance is not None:
            log = simulate_disturbed_reach(demo_config, start.q, evaluation["duration"], disturbance_from(i))
        log.save(run_dir / f"demonstrator_{i:02d}.sklog.npz")
        rows.append(
            {"law": "", "omega": "", "damping": "", "arm": "demonstrator", "start": i, "origin": start.origin}
            | {"replayed": ""}
            | arm_metrics(log, i, setup, esn.config.dt, window, spans)
        )

    tracker = config["tracker"]
    settings = tracker_settings(tracker)
    end_posture = np.mean([q[-1] for q in setup.demos.values()], axis=0)  # where the reaches end
    for setting in settings:
        law, omega = setting.law, setting.omega
        tracker_config = TrackerConfig(
            law,
            omega,
            tracker["acceleration_filter"],
            damping=setting.damping,
            reference_velocity=tracker.get("reference_velocity", True),
        )
        gains = tracking_gains(tracker_config, setup.skeleton, end_posture)
        setting_dir = run_dir / setting.name
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
                    tracker_config,
                    gains,
                    period=esn.config.dt,
                    warmup_steps=esn.config.warmup_steps,
                    duration=evaluation["duration"],
                    dt=demo_config["simulator"]["dt"],
                    enforce_limits=demo_config["simulator"].get("enforce_limits", True),
                    extra={
                        "playback": {"task": setup.task},
                        "tracking": {
                            "reference": arm,
                            "law": law,
                            "omega": omega,
                            "damping": setting.damping,
                            "reference_velocity": tracker_config.reference_velocity,
                            "start": start.origin,
                        },
                    },
                    external_force=disturbance_from(i),
                )
                log.save(setting_dir / f"{arm}_{i:02d}.sklog.npz")
                rows.append(
                    {"law": law, "omega": omega, "damping": setting.damping, "arm": arm, "start": i}
                    | {"origin": start.origin}
                    | {"replayed": replayed if arm == "replay" else ""}
                    | arm_metrics(log, i, setup, esn.config.dt, window, spans)
                )
        print(f"Ran {setting.label} (kp = {np.round(gains[0], 2).tolist()}, kd = {np.round(gains[1], 2).tolist()})")

    with (run_dir / "metrics.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows, len(setup.starts), window, setup.reference)

    title = f"ESN on the robot: {args.config.stem}"
    swept = "damping" if len(tracker.get("dampings", [1.0])) > 1 else "omega"
    plot_metrics(rows, tracker["laws"], len(setup.starts), window, swept, setup.reference, title).savefig(
        run_dir / "metrics.png", dpi=150
    )
    plot_paths(run_dir, setup, settings, title).savefig(run_dir / "paths.png", dpi=150)
    plot_timeline(run_dir, setup, settings, spans, title).savefig(run_dir / "timeline.png", dpi=150)
    plot_joints(run_dir, setup, settings, spans, title).savefig(run_dir / "joints.png", dpi=150)
    plot_torques(run_dir, setup, settings, spans, title).savefig(run_dir / "torques.png", dpi=150)
    if "start_grid" in evaluation:
        plot_grid(rows, setup, settings, title).savefig(run_dir / "grid.png", dpi=150)
    print(f"\nWrote the results to {run_dir}")
    first = run_dir / settings[0].name / "esn_00.sklog.npz"
    print(f"Replay with:\n  uv run python third_party/skelarm/tools/player.py {first}")


@dataclass(frozen=True)
class Setting:
    """One tracker setting: a tracking law, and the natural frequency and damping ratio of its error."""

    law: str
    omega: float  # rad/s
    damping: float  # 1 is critically damped

    @property
    def name(self) -> str:
        """Its directory in the run: ``pd_w10``, or ``pd_w10_z0.3`` below critical damping."""
        return f"{self.law}_w{self.omega:g}" + ("" if self.damping == 1.0 else f"_z{self.damping:g}")

    @property
    def label(self) -> str:
        """Its name in titles and messages."""
        label = f"{self.law.replace('_', ' ')}, ω = {self.omega:g} rad/s"
        return label + ("" if self.damping == 1.0 else f", ζ = {self.damping:g}")


def tracker_settings(tracker: dict[str, Any]) -> list[Setting]:
    """Every combination of the laws, natural frequencies, and damping ratios of ``[tracker]``, law by law.

    ``omegas`` lists the natural frequencies of every law, or maps each law to its
    own list. Only one of the natural frequency and the damping ratio may take
    several values, so that the metrics plot against it.
    """
    laws, omegas, dampings = tracker["laws"], tracker["omegas"], tracker.get("dampings", [1.0])
    by_law: dict[str, list[float]] = omegas if isinstance(omegas, dict) else {law: omegas for law in laws}
    if set(by_law) != set(laws):
        msg = f"[tracker] omegas must give the natural frequencies of exactly the laws {', '.join(laws)}"
        raise ValueError(msg)
    if any(len(values) > 1 for values in by_law.values()) and len(dampings) > 1:
        msg = "[tracker] can sweep the natural frequencies (omegas) or the damping ratios (dampings), not both"
        raise ValueError(msg)
    return [Setting(law, float(omega), float(damping)) for law in laws for omega in by_law[law] for damping in dampings]


def posed(skeleton: Skeleton, q: NDArray[np.float64]) -> Skeleton:
    """A copy of the robot at the joint angles ``q``, at rest."""
    arm = skeleton.clone()
    arm.q = q
    arm.dq = np.zeros_like(q)
    return arm


def arm_metrics(
    log: StateLog, i: int, setup: Setup, period: float, window: list[float], spans: list[tuple[float, float]]
) -> dict[str, Any]:
    """The metrics of one arm's run from start posture ``i``, compared with the demonstrator's undisturbed reach.

    Besides the reach and hold metrics of Stage 1 (see
    :func:`arm_esn_ctrl.autonomous.run_metrics`), over the task (t >= 0):

    - ``tracking_error_deg``: RMS of the tracking error ``q_ref - q``;
    - ``reference_path_rmse_m``: RMS distance between the hand path of the
      reference (what the ESN generates) and the taught hand path, regardless of
      timing, as ``taught_path_rmse_m`` measures the arm's;
    - ``peak_reference_speed_dps``: the reference's fastest joint speed, which
      shows a jump of the reference (as a jump of a few degrees in one period);
    - ``peak_hand_speed_mps``: the hand's fastest speed, sampled every period;
    - ``settling_time_s``: the time from which the hand stays within the goal
      radius until the end of the run (NaN if it ends outside);
    - ``final_distance_m``: the hand's distance to the target at the end of the run;

    and over the effort window (``effort_window`` in ``[evaluation]``):

    - ``peak_torque_nm``: the largest joint torque;
    - ``effort_n2m2s``: the integral of the squared joint torques, summed over the joints;
    - ``peak_external_force_n``: the largest disturbance force at the tip (a
      block's holding force shows how hard the arm pushes against it);

    and at the end of each disturbance (``spans``, when each acts on the task clock):

    - ``reference_lead``: how far the reference is ahead of the arm along the
      demonstrated path, as a fraction of the path (see :func:`progress`); NaN
      without a disturbance. With several disturbances, ``reference_lead_1``,
      ``reference_lead_2``, ... at the end of each, in the order configured.

    The demonstrator tracks no reference, so its tracking metrics are NaN.
    """
    q = task_joint_angles(log, period, setup.times[-1])
    hand = endpoint_positions(setup.skeleton, q)
    q_ref = None if setup.q_refs is None else setup.q_refs[i]
    hand_ref = None if setup.hand_refs is None else setup.hand_refs[i]
    run = Run(setup.starts[i], q, q_ref, hand, hand_ref)
    times = log.times
    task = times > -1e-9
    in_window = (times > window[0] - 1e-9) & (times < window[1] + 1e-9)
    tracked = "q_ref" in log.channel_names
    nan = float("nan")
    tau = log.channel("tau")
    force = log.channel("ext_force") if "ext_force" in log.channel_names else np.zeros((len(times), 2))
    distance = np.linalg.norm(hand - setup.target, axis=1)
    outside = np.flatnonzero(distance > setup.radius)
    settling_time = nan
    if len(outside) == 0:
        settling_time = float(setup.times[0])
    elif outside[-1] < len(distance) - 1:
        settling_time = float(setup.times[outside[-1] + 1])
    reference_speed = nan
    leads = [nan] * len(spans)
    if tracked:
        q_ref = log.channel("q_ref")[task]
        reference_speed = float(np.degrees(np.abs(np.diff(q_ref, axis=0)).max() / np.diff(times[task]).min()))
        for n, (_, end) in enumerate(spans):
            k = int(np.argmin(np.abs(times - end)))
            ahead = progress(log.channel("q_ref")[k : k + 1], setup.starts[i].q, setup)
            leads[n] = float(ahead[0] - progress(log.channel("q")[k : k + 1], setup.starts[i].q, setup)[0])
    if len(spans) > 1:
        lead_metrics = {f"reference_lead_{n + 1}": lead for n, lead in enumerate(leads)}
    else:
        lead_metrics = {"reference_lead": leads[0] if leads else nan}
    reference_path_rmse = nan
    if tracked:
        reference_hand = endpoint_positions(setup.skeleton, task_joint_angles(log, period, setup.times[-1], "q_ref"))
        taught_hand = setup.demo_hands[nearest_demonstration(setup.starts[i].q, setup.demos)]
        reference_path_rmse = path_rmse(reference_hand, taught_hand)
    return run_metrics(run, setup) | {
        "tracking_error_deg": rms_degrees(log.channel("error")[task]) if tracked else nan,
        "reference_path_rmse_m": reference_path_rmse,
        "peak_reference_speed_dps": reference_speed,
        "peak_hand_speed_mps": float(hand_speed(setup.times, hand).max()),
        "settling_time_s": settling_time,
        "final_distance_m": float(distance[-1]),
        "peak_torque_nm": float(np.abs(tau[in_window]).max()),
        "effort_n2m2s": float(np.sum(tau[in_window] ** 2) * np.diff(times).mean()),
        "peak_external_force_n": float(np.linalg.norm(force[in_window], axis=1).max()),
        **lead_metrics,
    }


def progress(q: NDArray[np.float64], start_q: NDArray[np.float64], setup: Setup) -> NDArray[np.float64]:
    """How far along the demonstrated joint path the joint angles ``q`` are, from 0 at its start to 1 at its end.

    The path is that of the training demonstration that starts nearest to
    ``start_q``. A reference generator that keeps the demonstration's timing, as the
    replay does, advances along it whatever the arm does; one that adapts to the
    arm stays with the arm's progress.
    """
    return path_progress(q, setup.demos[nearest_demonstration(start_q, setup.demos)])


def print_summary(rows: list[dict[str, Any]], n_starts: int, window: list[float], reference: Reference) -> None:
    """Print the metrics of each arm and tracker setting: means over the start postures.

    The joint error and path distance are from the ``reference``: the demonstrator's
    undisturbed reach, or the taught motion. The settling time is the mean over the
    runs that settle.
    """
    print(f"\nMeans over the {n_starts} start postures (torque, effort, force: t = {window[0]:g} to {window[1]:g} s);")
    print(f"joint error and path distance from the {reference.name}")
    # Columns: reach joint error and path distance, hold successes, settling time, final distance, peak reference
    # joint speed, peak hand speed, peak torque, effort, and peak disturbance force.
    names = ["error", "path", "hold", "settle", "final", "ref.", "hand", "torque", "effort", "force"]
    units = ["(deg)", "(mm)", "", "(s)", "(mm)", "(dps)", "(m/s)", "(N m)", "(N2m2s)", "(N)"]
    widths = [6, 6, 7, 7, 7, 7, 6, 7, 9, 7]
    print(f"{'law':<16}{'ω':<6}{'ζ':<5}{'arm':<12}" + "".join(f"{n:>{w}}" for n, w in zip(names, widths, strict=True)))
    print(f"{'':<16}{'rad/s':<6}{'':<17}" + "".join(f"{u:>{w}}" for u, w in zip(units, widths, strict=True)))
    keys = ("law", "omega", "damping", "arm")
    for key in dict.fromkeys(tuple(r[name] for name in keys) for r in rows):
        group = [r for r in rows if tuple(r[name] for name in keys) == key]
        law, omega, damping, arm = key
        m = {name: mean_of(group, name) for name in group[0] if isinstance(group[0][name], float)}
        setting = "-" if law == "" else f"{law.replace('_', ' '):<15} {omega:<5g} {damping:<4g}"
        print(
            f"{setting:<26} {arm:<12}"
            f" {m[reference.joint_error]:5.2f} {1000 * m[reference.path_distance]:5.1f}"
            f" {sum(r['success'] for r in group):2d}/{n_starts:<3d} {m['settling_time_s']:6.2f}"
            f" {1000 * m['final_distance_m']:6.1f} {m['peak_reference_speed_dps']:6.0f} {m['peak_hand_speed_mps']:5.2f}"
            f" {m['peak_torque_nm']:6.1f} {m['effort_n2m2s']:8.1f} {m['peak_external_force_n']:6.1f}"
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


def plot_metrics(
    rows: list[dict[str, Any]],
    laws: list[str],
    n_starts: int,
    window: list[float],
    swept: str,
    reference: Reference,
    title: str,
) -> Figure:
    """Plot each metric's mean over the start postures (and its range) against ``swept``: "omega" or "damping".

    Each law takes two rows of panels. The demonstrator, which has no gains, is a
    dashed level. The settling time is the mean over the runs that settle. The joint
    error, path distance, and arrival delay are from the ``reference``.
    """
    effort = f"over {window[0]:g}-{window[1]:g} s"
    during = "\nduring the reach" if reference.name == "demonstrator" else ""
    metrics = [
        (reference.joint_error, f"Joint error from the {reference.name}{during} (deg RMS)"),
        (reference.path_distance, f"Path distance from the {reference.name}{during} (m)"),
        (reference.arrival_delay, f"Arrival delay after the {reference.name} (s)"),
        ("success", f"Successes: arrive and hold\n(of {n_starts} start postures)"),
        ("settling_time_s", "Settling time: in the goal\nfrom then on (s)"),
        ("final_distance_m", "Final distance to the target (m)"),
        ("peak_reference_speed_dps", "Peak reference joint speed (deg/s)"),
        ("peak_hand_speed_mps", "Peak hand speed (m/s)"),
        ("peak_torque_nm", f"Peak joint torque (N m)\n{effort}"),
        ("effort_n2m2s", f"Integral of squared torque (N² m² s)\n{effort}"),
    ]
    per_row = 5
    x_label = {"omega": "natural frequency ω (rad/s)", "damping": "damping ratio ζ"}[swept]
    fig = Figure(figsize=(17, 6.2 * len(laws) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color="#0b0b0b")
    axes = np.asarray(fig.subplots(2 * len(laws), per_row)).reshape(len(laws), 2 * per_row)
    for law, law_axes in zip(laws, axes, strict=True):
        for (key, label), ax in zip(metrics, law_axes, strict=True):
            style(ax)
            for arm in ("esn", "replay"):
                group = [r for r in rows if r["law"] == law and r["arm"] == arm]
                xs = sorted({r[swept] for r in group})
                values = [[float(r[key]) for r in group if r[swept] == x] for x in xs]
                if key == "success":
                    low = high = mean = [sum(v) for v in values]
                else:
                    low = [np.nanmin(v) if np.isfinite(v).any() else np.nan for v in values]
                    high = [np.nanmax(v) if np.isfinite(v).any() else np.nan for v in values]
                    mean = [np.nanmean(v) if np.isfinite(v).any() else np.nan for v in values]
                color = ARM_COLORS[arm]
                ax.fill_between(xs, low, high, color=color, alpha=0.15, linewidth=0)
                ax.plot(xs, mean, marker="o", markersize=5, linewidth=2, color=color, label=ARM_LABELS[arm])
                ax.set_xscale("log")
                ax.set_xticks(xs, [f"{x:g}" for x in xs])
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
            ax.set_xlabel(x_label, color=TEXT_COLOR, fontsize=8)
        law_axes[0].set_ylabel(law.replace("_", " "), color="#0b0b0b", fontsize=11)
        law_axes[per_row].set_ylabel(law.replace("_", " "), color="#0b0b0b", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def load_hand(path: Path, setup: Setup) -> tuple[NDArray[np.float64], NDArray[np.float64], StateLog]:
    """The times and hand positions of a saved run, every simulation step, and the log."""
    log = StateLog.load(path)
    return log.times, endpoint_positions(setup.skeleton, log.channel("q").reshape(len(log.times), -1)), log


def by_law(settings: list[Setting]) -> list[list[Setting]]:
    """The settings grouped by law: one row of the figures per law, one column per setting."""
    laws = list(dict.fromkeys(setting.law for setting in settings))
    return [[setting for setting in settings if setting.law == law] for law in laws]


def plot_paths(run_dir: Path, setup: Setup, settings: list[Setting], title: str) -> Figure:
    """Plot the hand paths of every run, over the demonstrator's undisturbed reaches (or the taught motion)."""
    rows = by_law(settings)
    columns = len(rows[0])
    fig = Figure(figsize=(3.2 * columns, 3.3 * len(rows) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color="#0b0b0b")
    axes = fig.subplots(len(rows), columns, sharex=True, sharey=True, squeeze=False)
    n_starts = len(setup.starts)
    if setup.hand_refs is not None:
        references, reference_label = setup.hand_refs, "demonstrator, undisturbed"
    else:
        references, reference_label = list(setup.demo_hands.values()), "taught motion"
    for row, ax_row in zip(rows, axes, strict=True):
        for setting, ax in zip(row, ax_row, strict=True):
            style(ax)
            for i, hand_ref in enumerate(references):
                label = reference_label if i == 0 else None
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
                    _, hand, _ = load_hand(run_dir / setting.name / f"{arm}_{i:02d}.sklog.npz", setup)
                    ax.plot(*hand.T, color=ARM_COLORS[arm], linewidth=1.2, label=ARM_LABELS[arm] if i == 0 else None)
            ax.plot(*setup.target, marker="+", markersize=12, color="#0b0b0b", markeredgewidth=1.5)
            ax.set_title(setting.label, color=TEXT_COLOR, fontsize=10)
            ax.set_aspect("equal")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = 3 if columns > 2 else 1  # a narrow figure stacks the legend
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_timeline(
    run_dir: Path, setup: Setup, settings: list[Setting], spans: list[tuple[float, float]], title: str
) -> Figure:
    """Plot the hand's distance to the target, the progress, and the joint torque over time, from the first start.

    The progress along the demonstrated path (see :func:`progress`) is drawn for the
    arms (solid) and for their references (dashed). The shaded bands are when the
    disturbances act.
    """
    rows = by_law(settings)
    columns = len(rows[0])
    fig = Figure(figsize=(3.6 * columns + 1.0, 7.2 * len(rows) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"{title}, from {setup.starts[0].origin}", color="#0b0b0b")
    axes = np.asarray(fig.subplots(3 * len(rows), columns, sharex=True)).reshape(3 * len(rows), columns)
    start_q = setup.starts[0].q
    ref_times, ref_q, ref_label = demonstrated_motion(setup)
    ref_hand = endpoint_positions(setup.skeleton, ref_q)
    demonstrator = run_dir / "demonstrator_00.sklog.npz"
    for row, law_settings in enumerate(rows):
        for col, setting in enumerate(law_settings):
            ax_distance, ax_progress, ax_torque = axes[3 * row : 3 * row + 3, col]
            runs = {arm: load_hand(run_dir / setting.name / f"{arm}_00.sklog.npz", setup) for arm in ("esn", "replay")}
            if demonstrator.exists():
                runs["demonstrator"] = load_hand(demonstrator, setup)
            for ax in (ax_distance, ax_progress, ax_torque):
                style(ax)
                for span in spans:
                    ax.axvspan(*span, color=DISTURBANCE_COLOR, zorder=0)
            ax_distance.plot(
                ref_times,
                1000 * np.linalg.norm(ref_hand - setup.target, axis=1),
                color=ARM_COLORS["demonstrator"],
                linewidth=4,
                alpha=0.3,
                label=ref_label,
            )
            undisturbed = progress(ref_q, start_q, setup)
            ax_progress.plot(ref_times, undisturbed, color=ARM_COLORS["demonstrator"], linewidth=4, alpha=0.3)
            for arm in [arm for arm in ARMS if arm in runs]:
                times, hand, log = runs[arm]
                color = ARM_COLORS[arm]
                ax_distance.plot(
                    times,
                    1000 * np.linalg.norm(hand - setup.target, axis=1),
                    color=color,
                    linewidth=1.3,
                    label=ARM_LABELS[arm],
                )
                ax_progress.plot(times, progress(log.channel("q"), start_q, setup), color=color, linewidth=1.3)
                if "q_ref" in log.channel_names:
                    reference_progress = progress(log.channel("q_ref"), start_q, setup)
                    ax_progress.plot(times, reference_progress, color=color, linewidth=1.3, linestyle="--")
                ax_torque.plot(times, np.linalg.norm(log.channel("tau"), axis=1), color=color, linewidth=1.3)
            ax_distance.axhline(1000 * setup.radius, color="#0b0b0b", linewidth=0.8, linestyle=":")
            ax_distance.set(xlim=(-0.3, float(setup.times[-1])), ylim=(0, None))
            ax_distance.set_title(setting.label, color=TEXT_COLOR, fontsize=10)
            ax_progress.set_ylim(-0.05, 1.05)
            ax_torque.set_xlabel("time (s)", color=TEXT_COLOR, fontsize=8)
            if col == 0:
                ax_distance.set_ylabel("hand to target (mm)", color=TEXT_COLOR)
                ax_progress.set_ylabel("progress along the\ndemonstrated path", color=TEXT_COLOR)
                ax_torque.set_ylabel("joint torque norm (N m)", color=TEXT_COLOR)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(axes[1, 0].plot([], [], color=TEXT_COLOR, linestyle="--", linewidth=1.3)[0])
    labels.append("its reference (progress)")
    ncol = 4 if columns > 2 else columns  # a narrow figure wraps the legend
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def demonstrated_motion(setup: Setup) -> tuple[NDArray[np.float64], NDArray[np.float64], str]:
    """The motion the runs from the first start are compared with: its times, joint angles, and label.

    It is the demonstrator's undisturbed reach from that start, or, without a
    demonstrator, the taught motion nearest to it.
    """
    if setup.demonstrator_logs:
        reference = setup.demonstrator_logs[0]
        return reference.times, reference.channel("q").reshape(len(reference.times), -1), "demonstrator, undisturbed"
    q = setup.demos[nearest_demonstration(setup.starts[0].q, setup.demos)]
    return (setup.times[1] - setup.times[0]) * np.arange(len(q)), q, "taught motion"


def plot_joints(
    run_dir: Path, setup: Setup, settings: list[Setting], spans: list[tuple[float, float]], title: str
) -> Figure:
    """Plot the joint angles over time from the first start: one column per tracker setting, one row per joint.

    Each panel draws the demonstrated motion (thick gray), the ESN's output, the
    reference it gives the tracker (dashed), the arm driven by the ESN, the arm
    replaying the demonstration by time, and the demonstrator's arm when there is
    one. The shaded bands are when the disturbances act.
    """
    ref_times, ref_q, ref_label = demonstrated_motion(setup)

    def draw(ax: Axes, logs: dict[str, StateLog], joint: int) -> None:
        ax.plot(
            ref_times,
            np.degrees(ref_q[:, joint]),
            color=ARM_COLORS["demonstrator"],
            linewidth=4,
            alpha=0.3,
            label=ref_label,
        )
        esn = logs["esn"]
        ax.plot(
            esn.times,
            np.degrees(esn.channel("q_ref").reshape(len(esn.times), -1)[:, joint]),
            color=ARM_COLORS["esn"],
            linewidth=1.3,
            linestyle="--",
            label="ESN output (its reference)",
        )
        for arm in [arm for arm in ARMS if arm in logs]:
            log = logs[arm]
            q = np.degrees(log.channel("q").reshape(len(log.times), -1)[:, joint])
            ax.plot(log.times, q, color=ARM_COLORS[arm], linewidth=1.3, label=ARM_LABELS[arm])

    return per_joint_figure(run_dir, setup, settings, spans, f"{title}: joint angles", draw, "joint {joint} (deg)")


def plot_torques(
    run_dir: Path, setup: Setup, settings: list[Setting], spans: list[tuple[float, float]], title: str
) -> Figure:
    """Plot the joint torques over time from the first start: one column per tracker setting, one row per joint.

    Each panel draws the torque of the arm driven by the ESN, of the arm replaying
    the demonstration by time, and of the demonstrator's arm when there is one. The
    shaded bands are when the disturbances act.
    """

    def draw(ax: Axes, logs: dict[str, StateLog], joint: int) -> None:
        for arm in [arm for arm in ARMS if arm in logs]:
            log = logs[arm]
            tau = log.channel("tau").reshape(len(log.times), -1)[:, joint]
            ax.plot(log.times, tau, color=ARM_COLORS[arm], linewidth=1.0, label=ARM_LABELS[arm])

    return per_joint_figure(run_dir, setup, settings, spans, f"{title}: joint torques", draw, "joint {joint} (N m)")


def per_joint_figure(
    run_dir: Path,
    setup: Setup,
    settings: list[Setting],
    spans: list[tuple[float, float]],
    title: str,
    draw: Callable[[Axes, dict[str, StateLog], int], None],
    ylabel: str,
) -> Figure:
    """A figure of one panel per tracker setting (columns, one row of panels per law) and joint, from the first start.

    ``draw(ax, logs, joint)`` draws a panel from the logs of the arms (``esn``,
    ``replay``, and ``demonstrator`` when there is one); ``ylabel`` names the joint's
    axis, with ``{joint}`` its number. The shaded bands are when the disturbances act.
    """
    rows = by_law(settings)
    columns = len(rows[0])
    n_joints = setup.skeleton.num_joints
    width = max(4.6 * columns + 1.0, 9.0)  # wide enough for the title with a single column
    fig = Figure(figsize=(width, 2.6 * n_joints * len(rows) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"{title}, from {setup.starts[0].origin}", color="#0b0b0b")
    axes = np.asarray(fig.subplots(n_joints * len(rows), columns, sharex=True)).reshape(n_joints * len(rows), columns)
    demonstrator = run_dir / "demonstrator_00.sklog.npz"
    for row, law_settings in enumerate(rows):
        for col, setting in enumerate(law_settings):
            logs = {arm: StateLog.load(run_dir / setting.name / f"{arm}_00.sklog.npz") for arm in ("esn", "replay")}
            if demonstrator.exists():
                logs["demonstrator"] = StateLog.load(demonstrator)
            for joint in range(n_joints):
                ax = axes[n_joints * row + joint, col]
                style(ax)
                for span in spans:
                    ax.axvspan(*span, color=DISTURBANCE_COLOR, zorder=0)
                draw(ax, logs, joint)
                ax.set_xlim(-0.3, float(setup.times[-1]))
                ax.set_title(f"{setting.label}: joint {joint + 1}", color=TEXT_COLOR, fontsize=9)
                if col == 0:
                    ax.set_ylabel(ylabel.format(joint=joint + 1), color=TEXT_COLOR)
            axes[n_joints * row + n_joints - 1, col].set_xlabel("time (s)", color=TEXT_COLOR, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, labelcolor=TEXT_COLOR)
    return fig


OUTCOME_COLORS = ["#2a78d6", "#eb6834", "#52514e"]  # arrive and hold, leave the goal, never arrive
OUTCOMES = ["arrive and hold", "leave the goal", "never arrive"]
# The maps of grid.png: metric, title, factor to its display unit, and the top of a
# linear color scale from 0, or None for a logarithmic scale over the values.
# The first two compare with the reference, the demonstrator or the taught motion (see grid_maps).
GRID_MAPS: list[tuple[str, str, float, float | None]] = [
    ("peak_reference_speed_dps", "Peak reference joint speed (deg/s)", 1.0, None),
    ("peak_torque_nm", "Peak joint torque (N m)", 1.0, None),
    ("tracking_error_deg", "Tracking error (deg RMS)", 1.0, None),
]


def grid_maps(reference: Reference) -> list[tuple[str, str, float, float | None]]:
    """The maps of grid.png: the two that compare with the ``reference``, then :data:`GRID_MAPS`."""
    where = "demonstrator's reach" if reference.name == "demonstrator" else "taught motion"
    course = reference.course_label.replace(": ", ":\n", 1)
    return [
        (reference.path_distance, f"Path distance from the\n{where} (mm)", 1000.0, None),
        (reference.course, course, 1.0, 1.5),
        *GRID_MAPS,
    ]


def plot_grid(rows: list[dict[str, Any]], setup: Setup, settings: list[Setting], title: str) -> Figure:
    """Map the outcome and the main metrics over the start offsets: one row per tracker setting and arm.

    Each metric's colors are shared by its column, so the ESN and the replay compare.
    """
    offsets = {}
    for i, start in enumerate(setup.starts):
        demonstrated = setup.demos[nearest_demonstration(start.q, setup.demos)][0]
        offsets[i] = tuple(np.round(np.degrees(start.q - demonstrated), 6))
    xs = np.unique([offset[0] for offset in offsets.values()])
    ys = np.unique([offset[1] for offset in offsets.values()])
    step_x, step_y = xs[1] - xs[0], ys[1] - ys[0]
    extent = (xs[0] - step_x / 2, xs[-1] + step_x / 2, ys[0] - step_y / 2, ys[-1] + step_y / 2)
    panels = [(setting, arm) for setting in settings for arm in ("esn", "replay")]
    maps = grid_maps(setup.reference)
    columns = 1 + len(maps)
    fig = Figure(figsize=(3.3 * columns, 3.0 * len(panels) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"{title}: start offsets", color="#0b0b0b")
    axes = fig.subplots(len(panels), columns, squeeze=False)

    def grid_of(group: list[dict[str, Any]], key: str, factor: float) -> NDArray[np.float64]:
        cell = {offsets[r["start"]][:2]: float(r[key]) * factor for r in group}
        return np.array([[cell.get((x, y), np.nan) for x in xs] for y in ys])

    groups = [
        [r for r in rows if (r["law"], r["omega"], r["damping"], r["arm"]) == (s.law, s.omega, s.damping, arm)]
        for s, arm in panels
    ]
    for ax_row, (setting, arm), group in zip(axes, panels, groups, strict=True):
        outcome = [{**r, "outcome": 0 if r["success"] else 1 if r["arrived"] else 2} for r in group]
        ax_row[0].imshow(
            grid_of(outcome, "outcome", 1.0),
            origin="lower",
            extent=extent,
            cmap=ListedColormap(OUTCOME_COLORS),
            vmin=-0.5,
            vmax=2.5,
        )
        ax_row[0].set_ylabel(f"{setting.label}\n{ARM_LABELS[arm]}\njoint 2 offset (deg)", color=TEXT_COLOR, fontsize=8)
        ax_row[0].set_title("Outcome", color=TEXT_COLOR, fontsize=9)
    handles = [Patch(color=color, label=text) for color, text in zip(OUTCOME_COLORS, OUTCOMES, strict=True)]
    axes[0, 0].legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.8)

    for column, (key, label, factor, top) in enumerate(maps, start=1):
        data = [grid_of(group, key, factor) for group in groups]
        finite = np.concatenate([d[np.isfinite(d)] for d in data])
        if top is not None:
            norm: Normalize = Normalize(vmin=0.0, vmax=top)
        elif finite.size:
            norm = LogNorm(vmin=max(float(finite.min()), 1e-3), vmax=max(float(finite.max()), 1e-3) * 1.0001)
        else:
            norm = Normalize(0.0, 1.0)
        for ax, grid in zip(axes[:, column], data, strict=True):
            ax.imshow(grid, origin="lower", extent=extent, cmap="Blues", norm=norm)
            ax.set_title(label, color=TEXT_COLOR, fontsize=9)
        extend = "max" if top is not None else "neither"
        fig.colorbar(
            ScalarMappable(norm=norm, cmap="Blues"), ax=axes[:, column].tolist(), location="bottom", extend=extend
        )
    for ax in axes.flat:
        ax.plot(0.0, 0.0, marker="+", markersize=10, color="#0b0b0b", markeredgewidth=1.5)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("joint 1 offset (deg)", color=TEXT_COLOR, fontsize=8)
    return fig


if __name__ == "__main__":
    main()
