# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Record a trained ESN's reservoir states and compare them by principal components (Stage 1).

    uv run python experiments/reservoir_states.py \\
        experiments/single_demonstration_autonomous_reaching/states_multi_demo_settings.toml

The ESN is loaded from a run of ``experiments/autonomous_esn.py`` (``model`` in
``[esn]``), and its training demonstrations from that run's configuration. The
reservoir state is recorded at every step:

- while each training demonstration is fed in sample by sample (teacher forcing),
  after the same warm-up as in training: the states the readout was fitted on;
- in the autonomous runs from the start postures of ``[evaluation]``, as in
  ``autonomous_esn.py``.

Times are on the task clock: the warm-up runs at negative times, and the state at
time t has seen the input up to t. The principal components of the demonstrations'
states over the task (t >= 0) are the axes onto which every state is projected.
Distances are measured in the full state space, relative to the spread of the
demonstrations' states over the task (the RMS distance of those states from their
mean), and compare each run with the demonstration whose start is nearest:

- the **time-matched distance**: from the run's state at t to the demonstration's
  state at t;
- the **phase**: the time of the demonstration's state nearest to the run's state
  at t. A run that keeps the demonstration's timing has a phase equal to t; the
  **phase lead** is the phase minus t. The phase locates a run along the
  demonstration only while the run is close to the demonstration's states: a run
  that lies before the demonstration's start has phase 0 and a lead of -t.

The run directory receives:

- ``states.csv``: for every start posture, its offset from the demonstrated start,
  the time-matched distance at the end of the warm-up (t = 0) and at 0.25, 0.5,
  and 1 s, the **join time** from which that distance stays below
  ``join_threshold``, and the median phase lead over ``phase_window``;
- ``projections.npz``: every state projected onto the first ``n_components``
  principal components, for the demonstrations and the runs, with the explained
  variance of all components;
- ``pca.png``: the projections onto the first three components;
- ``convergence.png``: the distances and the phase lead over time, and, with an
  ``[evaluation.start_grid]``, maps over the start offsets of the distance at the
  end of the warm-up, the join time, and the phase lead, with an arrow along the
  demonstration's change of joint angles from start to end.
"""

from __future__ import annotations

import argparse
import csv
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray

from arm_esn_ctrl.autonomous import Start, start_postures
from arm_esn_ctrl.demonstrations import load_joint_angles
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.storage import resolve_run_path, start_run
from arm_esn_ctrl.tracking import nearest_demonstration

ESN_COLOR = "#2a78d6"
TRAINING_COLOR = "#eb6834"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
REPORT_TIMES = (0.25, 0.5, 1.0)  # times at which states.csv reports the time-matched distance (s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    esn_path = resolve_run_path(config["esn"]["model"])
    esn = ReachingEsn.load(esn_path)
    with (esn_path.parent / "config.toml").open("rb") as f:
        demonstrations = tomllib.load(f)["demonstrations"]
    demo_dir = resolve_run_path(demonstrations["run"])
    dt, warmup_steps = esn.config.dt, esn.config.warmup_steps
    demos = {name.split(".")[0]: load_joint_angles(demo_dir / name, dt)[1] for name in demonstrations["train"]}
    evaluation, options = config["evaluation"], config["states"]
    starts = start_postures(evaluation, demos)
    n_steps = round(evaluation["duration"] / dt)
    times = dt * np.arange(-warmup_steps, n_steps + 1)
    print(f"ESN {config['esn']['model']}: {esn.config.n_neurons} neurons, trained on {len(demos)} demonstrations")

    # The states the readout was fitted on, and their principal components over the task.
    taught = {name: teacher_forced_states(esn, q) for name, q in demos.items()}
    task_states = np.vstack([states[warmup_steps:] for states in taught.values()])
    mean = task_states.mean(axis=0)
    _, singular, axes = np.linalg.svd(task_states - mean, full_matrices=False)
    explained = singular**2 / np.sum(singular**2)
    basis = axes[: options["n_components"]]
    spread = float(np.sqrt(np.mean(np.sum((task_states - mean) ** 2, axis=1))))
    print(f"Explained variance of the first components: {np.round(explained[:5], 4).tolist()}")

    rows, runs, distances, phases = [], [], [], []
    for i, start in enumerate(starts):
        name = nearest_demonstration(start.q, demos)
        states = autonomous_states(esn, start.q, n_steps)
        reference = taught[name]
        distance, phase = compare(states, reference, warmup_steps, spread)
        runs.append((states - mean) @ basis.T)
        distances.append(distance)
        phases.append(phase)
        rows.append(start_row(i, start, demos[name][0], times, distance, phase, warmup_steps, options))
    distances_array = np.array(distances)
    phases_array = np.array(phases)

    with (run_dir / "states.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    taught_projections = {name: (states - mean) @ basis.T for name, states in taught.items()}
    arrays: dict[str, Any] = {
        "times": times,
        "explained": explained,
        "runs": np.array(runs),
        "start_q": np.array([start.q for start in starts]),
    }
    arrays |= {f"demonstration_{name}": projection for name, projection in taught_projections.items()}
    np.savez_compressed(run_dir / "projections.npz", **arrays)
    print_summary(rows, spread, options)

    title = f"Reservoir states: {args.config.stem}"
    shown = [
        i for i, row in enumerate(rows) if [row["offset_q1_deg"], row["offset_q2_deg"]] in options["shown_offsets_deg"]
    ]
    plot_pca(taught_projections, np.array(runs), rows, shown, explained, warmup_steps, title).savefig(
        run_dir / "pca.png", dpi=150
    )
    motion = np.degrees(np.mean([q[-1] - q[0] for q in demos.values()], axis=0))
    plot_convergence(times, distances_array, phases_array, rows, shown, warmup_steps, motion, options, title).savefig(
        run_dir / "convergence.png", dpi=150
    )
    print(f"\nWrote the results to {run_dir}")


def teacher_forced_states(esn: ReachingEsn, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """The reservoir states while the demonstration ``q`` is fed in after the warm-up, as in training.

    One state per input: the held start posture for the warm-up, then every sample.
    """
    esn.reset()
    states = []
    for posture in [q[0]] * esn.config.warmup_steps + list(q):
        esn.step(posture)
        states.append(esn.state())
    return np.array(states)


def autonomous_states(esn: ReachingEsn, start_q: NDArray[np.float64], n_steps: int) -> NDArray[np.float64]:
    """The reservoir states of an autonomous run from ``start_q``, as :meth:`ReachingEsn.generate` runs it.

    One state per input: the held start posture for the warm-up and once more at
    time 0, then each generated posture, for ``n_steps`` steps.
    """
    esn.reset()
    states = []
    for _ in range(esn.config.warmup_steps):
        esn.step(start_q)
        states.append(esn.state())
    output = esn.step(start_q)
    states.append(esn.state())
    for _ in range(n_steps):
        output = esn.step(output)
        states.append(esn.state())
    return np.array(states)


def compare(
    states: NDArray[np.float64], reference: NDArray[np.float64], warmup_steps: int, spread: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The time-matched distance from ``reference`` at every time, and the phase from t = 0 (NaN before).

    Both cover the run's times; beyond the reference's last sample, the time-matched
    distance compares with that last state. Distances are relative to ``spread``.
    """
    matched = reference[np.minimum(np.arange(len(states)), len(reference) - 1)]
    distance = np.linalg.norm(states - matched, axis=1) / spread
    task, taught = states[warmup_steps:], reference[warmup_steps:]
    squared = np.sum(task**2, axis=1)[:, None] + np.sum(taught**2, axis=1)[None, :] - 2.0 * task @ taught.T
    phase = np.full(len(states), np.nan)
    phase[warmup_steps:] = np.argmin(squared, axis=1)
    return distance, phase


def start_row(
    i: int,
    start: Start,
    demonstrated: NDArray[np.float64],
    times: NDArray[np.float64],
    distance: NDArray[np.float64],
    phase: NDArray[np.float64],
    warmup_steps: int,
    options: dict[str, Any],
) -> dict[str, Any]:
    """The row of ``states.csv`` for start posture ``i``."""
    dt = float(times[1] - times[0])
    offset = np.degrees(start.q - demonstrated)
    task = times > -1e-9
    below = distance[task] < options["join_threshold"]
    above = np.flatnonzero(~below)
    join = float(times[task][0]) if len(above) == 0 else float("nan")
    if len(above) and above[-1] < len(below) - 1:
        join = float(times[task][above[-1] + 1])
    begin, end = options["phase_window"]
    window = (times >= begin - 1e-9) & (times <= end + 1e-9)
    steps = np.arange(len(times)) - warmup_steps  # time in steps, so that a lead of 0 is exactly 0
    lead = float(np.median((phase[window] - steps[window]) * dt))
    row = {
        "start": i,
        "origin": start.origin,
        "offset_q1_deg": round(float(offset[0]), 6),
        "offset_q2_deg": round(float(offset[1]), 6),
        "offset_deg": float(np.linalg.norm(offset)),
        "warmup_distance": float(distance[warmup_steps]),
    }
    for t in REPORT_TIMES:
        row[f"distance_{t:g}s"] = float(distance[warmup_steps + round(t / dt)])
    return row | {"join_time_s": join, "phase_lead_s": lead}


def print_summary(rows: list[dict[str, Any]], spread: float, options: dict[str, Any]) -> None:
    """Print how far the runs' states start from the demonstration's, and how they join it."""
    offset = [r for r in rows if r["offset_deg"] > 1e-9]
    if not offset:
        return
    joined = [r["join_time_s"] for r in offset if np.isfinite(r["join_time_s"])]
    print(f"Spread of the demonstration's states over the task: {spread:.3f}")
    print(
        f"{len(offset)} offset starts: time-matched distance, relative to that spread, median "
        f"{np.median([r['warmup_distance'] for r in offset]):.3f} at t = 0, "
        + ", ".join(f"{np.median([r[f'distance_{t:g}s'] for r in offset]):.3f} at {t:g} s" for t in REPORT_TIMES)
    )
    joined_text = f"median join time {np.median(joined):.2f} s" if joined else "no run joins"
    print(
        f"{len(joined)} of {len(offset)} join the demonstration's states (below {options['join_threshold']:g}),"
        f" {joined_text}; median phase lead over {options['phase_window']} s:"
        f" {np.median([r['phase_lead_s'] for r in offset]):+.3f} s"
    )


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def plot_pca(
    taught: dict[str, NDArray[np.float64]],
    runs: NDArray[np.float64],
    rows: list[dict[str, Any]],
    shown: list[int],
    explained: NDArray[np.float64],
    warmup_steps: int,
    title: str,
) -> Figure:
    """Project the states onto pairs of the first three principal components.

    The demonstration's states are orange (dashed during the warm-up); the dots are
    every run's state at the end of its warm-up, darker for a larger start offset;
    the blue lines are the runs from ``shown`` start offsets, from the warm-up on.
    """
    fig = Figure(figsize=(16, 5.6), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color="#0b0b0b")
    pairs = [(0, 1), (0, 2), (1, 2)]
    offsets = np.array([r["offset_deg"] for r in rows])
    norm = Normalize(vmin=0.0, vmax=max(float(offsets.max()), 1e-9))
    fig.colorbar(
        ScalarMappable(norm=norm, cmap="Blues"),
        ax=fig.subplots(1, 3),
        shrink=0.8,
        label="start offset (deg): states at the end of the warm-up",
    )
    for ax, (a, b) in zip(fig.axes[:3], pairs, strict=True):
        style(ax)
        for states in taught.values():
            ax.plot(states[: warmup_steps + 1, a], states[: warmup_steps + 1, b], color=TRAINING_COLOR, linestyle="--")
            ax.plot(states[warmup_steps:, a], states[warmup_steps:, b], color=TRAINING_COLOR, linewidth=4, alpha=0.8)
            ax.plot(*states[warmup_steps, [a, b]], marker="o", markersize=8, color=TRAINING_COLOR)
        for k, i in enumerate(shown):
            label = "runs from shown start offsets" if k == 0 else None
            ax.plot(runs[i, :, a], runs[i, :, b], color=ESN_COLOR, linewidth=1.0, alpha=0.8, label=label)
        ax.scatter(
            runs[:, warmup_steps, a], runs[:, warmup_steps, b], c=offsets, cmap="Blues", norm=norm, s=14, zorder=3
        )
        ax.set_xlabel(f"PC{a + 1} ({100 * explained[a]:.1f}% of the variance)", color=TEXT_COLOR)
        ax.set_ylabel(f"PC{b + 1} ({100 * explained[b]:.1f}%)", color=TEXT_COLOR)
    handles = [
        fig.axes[0].plot([], [], color=TRAINING_COLOR, linewidth=4, label="demonstration, teacher forced")[0],
        fig.axes[0].plot([], [], color=TRAINING_COLOR, linestyle="--", label="its warm-up")[0],
        fig.axes[0].plot([], [], color=ESN_COLOR, label="autonomous runs from shown start offsets")[0],
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_convergence(
    times: NDArray[np.float64],
    distances: NDArray[np.float64],
    phases: NDArray[np.float64],
    rows: list[dict[str, Any]],
    shown: list[int],
    warmup_steps: int,
    motion: NDArray[np.float64],
    options: dict[str, Any],
    title: str,
) -> Figure:
    """Distances and phase lead over time, and maps over the start offsets.

    The maps, drawn with an ``[evaluation.start_grid]``, show the distance at the end
    of the warm-up, the join time, and the phase lead; their arrow points along
    ``motion``, the demonstration's change of joint angles from start to end (deg).
    """
    dt = float(times[1] - times[0])
    fig = Figure(figsize=(17, 10), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color="#0b0b0b")
    # D: distance, L: phase lead over time; maps of W: the warm-up distance, J: the join time, P: the phase lead.
    panels = fig.subplot_mosaic("DDDLLL;WWJJPP")
    ax_distance, ax_phase = panels["D"], panels["L"]
    for ax in (ax_distance, ax_phase):
        style(ax)
    offset = np.array([r["offset_deg"] > 1e-9 for r in rows])
    for d in distances[offset]:
        ax_distance.plot(times, d, color=ESN_COLOR, linewidth=0.6, alpha=0.15)
    ax_distance.plot(times, np.median(distances[offset], axis=0), color="#0b0b0b", linewidth=2, label="median")
    ax_distance.axhline(options["join_threshold"], color=TEXT_COLOR, linestyle=":", label="join threshold")
    ax_distance.axvline(0.0, color=TEXT_COLOR, linewidth=0.8)
    ax_distance.set(yscale="log", xlabel="time (s)", ylabel="relative to the spread of the demonstration's states")
    ax_distance.set_title(
        "Time-matched distance from the demonstration's states (blue: every offset start)",
        color=TEXT_COLOR,
        fontsize=10,
    )
    ax_distance.legend(fontsize=8)

    lead = (phases - (np.arange(len(times)) - warmup_steps)) * dt  # in steps, so that 0 is exactly 0
    for i in shown:
        ax_phase.plot(times, lead[i], color=ESN_COLOR, linewidth=1.0, alpha=0.8)
    task = times > -1e-9  # the phase is defined from time 0
    median_lead = np.median(lead[offset][:, task], axis=0)
    ax_phase.plot(times[task], median_lead, color="#0b0b0b", linewidth=2, label="median, all offsets")
    ax_phase.axhline(0.0, color=TEXT_COLOR, linewidth=0.8)
    begin, end = options["phase_window"]
    ax_phase.axvspan(begin, end, color=GRID_COLOR, alpha=0.6, label="phase window")
    ax_phase.set(xlim=(0.0, float(times[-1])), xlabel="time (s)", ylabel="phase lead (s)")
    ax_phase.set_title(
        "Phase lead: the time of the nearest demonstration state, minus t\n(blue: runs from shown start offsets)",
        color=TEXT_COLOR,
        fontsize=10,
    )
    ax_phase.legend(fontsize=8)

    if len({r["offset_q1_deg"] for r in rows}) < 2 or len({r["offset_q2_deg"] for r in rows}) < 2:
        for name in "WJP":
            panels[name].set_axis_off()
        return fig
    leads = np.array([r["phase_lead_s"] for r in rows])
    # A range of a few steps is the quantization of the phase: the scale spans at least 10 steps.
    reach = max(float(np.max(np.abs(leads))), 10.0 * dt)
    maps = [
        ("W", "warmup_distance", "Blues", None, "relative distance", "Distance at the end of the warm-up"),
        ("J", "join_time_s", "Blues", None, "s", "Join time (blank: never joins)"),
        (
            "P",
            "phase_lead_s",
            "RdBu_r",
            Normalize(vmin=-reach, vmax=reach),
            "s",
            f"Median phase lead over {begin:g}-{end:g} s",
        ),
    ]
    for name, key, cmap, norm, label, map_title in maps:
        draw_map(fig, panels[name], rows, key, cmap, norm, label, motion)
        panels[name].set_title(map_title, color=TEXT_COLOR, fontsize=10)
    return fig


def draw_map(
    fig: Figure,
    ax: Axes,
    rows: list[dict[str, Any]],
    key: str,
    cmap: str,
    norm: Normalize | None,
    label: str,
    motion: NDArray[np.float64],
) -> None:
    """Draw ``key`` of ``rows`` over the grid of start offsets, with an arrow along ``motion``."""
    xs = np.unique([r["offset_q1_deg"] for r in rows])
    ys = np.unique([r["offset_q2_deg"] for r in rows])
    cell = {(r["offset_q1_deg"], r["offset_q2_deg"]): r[key] for r in rows}
    data = np.array([[cell.get((x, y), np.nan) for x in xs] for y in ys])
    step_x, step_y = xs[1] - xs[0], ys[1] - ys[0]
    extent = (xs[0] - step_x / 2, xs[-1] + step_x / 2, ys[0] - step_y / 2, ys[-1] + step_y / 2)
    image = ax.imshow(data, origin="lower", extent=extent, cmap=cmap, norm=norm)
    fig.colorbar(image, ax=ax, shrink=0.85, label=label)
    arrow = 0.4 * min(xs[-1], ys[-1]) * motion / np.linalg.norm(motion)
    ax.annotate(
        "", xy=(float(arrow[0]), float(arrow[1])), xytext=(0.0, 0.0), arrowprops={"arrowstyle": "->", "linewidth": 1.5}
    )
    ax.plot(0.0, 0.0, marker="+", markersize=12, color="#0b0b0b", markeredgewidth=1.5)
    ax.set(xlabel="joint 1 offset (deg)", ylabel="joint 2 offset (deg)")


if __name__ == "__main__":
    main()
