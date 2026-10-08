# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Plot a trained ESN's reservoir states while it drives the robot: neurons, and principal components (Stage 2).

    uv run python experiments/robot_states.py \\
        experiments/manual_demonstration_v2_robot_tracking/states_robot_fine_best_filtered.toml

The runs of ``[[run]]`` are runs of ``experiments/robot_esn.py`` that share their
ESN, each shown from one of its start postures, with each tracker setting of
``[states] settings``. They do not store the reservoir states:
:func:`arm_esn_ctrl.states.run_states` recomputes them from each run's log, from
the reset at the warm-up's start, and refuses a log whose reference the ESN does
not reproduce. For comparison, the take the ESN was trained on is fed in as in
training (teacher forcing). For each tracker setting, the run directory receives:

- ``neurons_<setting>.png``: the states of ``[neurons] count`` neurons picked at
  random (``seed``), one row each, over the whole run, from the reset, for every
  run, one column each;
- ``neurons_warmup_<setting>.png``: the same neurons over ``[neurons] span``, such
  as the warm-up and the task's first seconds, for the runs marked
  ``neurons = true``: how the warm-up brings the states from the reset to where
  the task starts;
- ``pca_<setting>.png``: every run's states along the first ``[pca] n_components``
  principal components over the whole run, from the reset, one column each;
- ``pca_planes_<setting>.png``: the same, in the planes of pairs of components.

The principal components are those of the states of every run with every tracker
setting, warm-up included, so that the figures share their axes. In every figure,
the take's states are drawn beneath each run's, and, over time, the warm-up and
the run's disturbances are shaded.
"""

from __future__ import annotations

import argparse
import itertools
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import NDArray
from skelarm import StateLog

from arm_esn_ctrl.demonstrations import load_joint_angles
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.states import PrincipalComponents, principal_components, run_states, teacher_forced_states
from arm_esn_ctrl.storage import resolve_run_path, start_run
from arm_esn_ctrl.tracking import nearest_demonstration

ESN_COLOR = "#2a78d6"
ESN_WARMUP_COLOR = "#93bfec"  # the ESN's states during the warm-up, in the planes
TAKE_COLOR = "#a3a29d"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
WARMUP_COLOR = "#ecebe6"
DISTURBANCE_COLOR = "#fbe0d2"


@dataclass(frozen=True)
class Column:
    """A run from one start posture with one tracker setting: one column of the figures."""

    label: str
    offset: NDArray[np.float64]  # the start posture's offset from the demonstrated start (deg)
    times: NDArray[np.float64]  # the ESN's instants, from the warm-up's start (s)
    traces: NDArray[np.float64]  # one row per instant: states, or their projections
    spans: list[tuple[float, float]]  # when its disturbances act (s)

    def heading(self) -> str:
        return f"{self.label}\nstart offset ({self.offset[0]:+.0f}, {self.offset[1]:+.0f}) deg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    settings, runs = config["states"]["settings"], config["run"]
    robot_configs = []
    for run in runs:
        with (resolve_run_path(run["path"]) / "config.toml").open("rb") as f:
            robot_configs.append(tomllib.load(f))
    models = sorted({robot["esn"]["model"] for robot in robot_configs})
    if len(models) != 1:
        msg = f"the runs must share their ESN, but they use {models}"
        raise ValueError(msg)
    model = resolve_run_path(models[0])
    esn = ReachingEsn.load(model)
    dt, warmup_steps = esn.config.dt, esn.config.warmup_steps
    with (model.parent / "config.toml").open("rb") as f:
        demonstrations = tomllib.load(f)["demonstrations"]
    demo_dir = resolve_run_path(demonstrations["run"])
    demos = {name.split(".")[0]: load_joint_angles(demo_dir / name, dt)[1] for name in demonstrations["train"]}
    taught = {name: teacher_forced_states(esn, q) for name, q in demos.items()}
    print(f"ESN {models[0]}: {esn.config.n_neurons} neurons")

    recomputed: dict[str, list[Column]] = {}
    for setting in settings:
        recomputed[setting] = []
        for run, robot in zip(runs, robot_configs, strict=True):
            log = StateLog.load(resolve_run_path(run["path"]) / setting / f"esn_{run['start']:02d}.sklog.npz")
            times, states = run_states(esn, log)
            start_q = log.channel("q").reshape(len(log.times), -1)[0]
            offset = np.degrees(start_q - demos[nearest_demonstration(start_q, demos)][0])
            spans = disturbance_spans(robot.get("disturbance"))
            recomputed[setting].append(Column(run["label"], offset, times, states, spans))
            print(f"  {setting}, {run['label']}: {len(states)} states recomputed; the logged reference reproduced")

    # The principal components of every run with every tracker setting, from the reset on.
    components = principal_components(np.vstack([c.traces for columns in recomputed.values() for c in columns]))
    n_components = config["pca"]["n_components"]
    print(f"Principal components: explained variance {np.round(components.explained[:5], 4).tolist()}")

    options = config["neurons"]
    rng = np.random.default_rng(options["seed"])
    neurons = np.sort(rng.choice(esn.config.n_neurons, options["count"], replace=False))
    print(f"Neurons picked at random (seed {options['seed']}): {neurons.tolist()}")
    taught_neurons = {name: states[:, neurons] for name, states in taught.items()}
    rows = [f"neuron {neuron}" for neuron in neurons]
    for setting, columns in recomputed.items():
        whole = [min(float(c.times[0]) for c in columns), max(float(c.times[-1]) for c in columns)]
        picked = f"{len(neurons)} of {esn.config.n_neurons} neurons picked at random (seed {options['seed']})"
        title = f"{args.config.stem}, tracker {setting}: {picked}, over the whole run"
        selected = [Column(c.label, c.offset, c.times, c.traces[:, neurons], c.spans) for c in columns]
        plot_traces(selected, taught_neurons, rows, dt, warmup_steps, whole, title).savefig(
            run_dir / f"neurons_{setting}.png", dpi=150
        )
        shown = [c for c, run in zip(selected, runs, strict=True) if run.get("neurons", False)]
        if shown:
            title = f"{args.config.stem}, tracker {setting}: {picked}, from the reset"
            plot_traces(shown, taught_neurons, rows, dt, warmup_steps, options["span"], title).savefig(
                run_dir / f"neurons_warmup_{setting}.png", dpi=150
            )

        projected = [
            Column(c.label, c.offset, c.times, components.project(c.traces, n_components), c.spans) for c in columns
        ]
        taught_projected = {name: components.project(states, n_components) for name, states in taught.items()}
        labels = [f"PC{i + 1} ({100 * components.explained[i]:.1f}%)" for i in range(n_components)]
        title = (
            f"{args.config.stem}, tracker {setting}: the principal components of every run's states, warm-up included"
        )
        plot_traces(projected, taught_projected, labels, dt, warmup_steps, whole, title).savefig(
            run_dir / f"pca_{setting}.png", dpi=150
        )
        plot_planes(projected, taught_projected, components, warmup_steps, title).savefig(
            run_dir / f"pca_planes_{setting}.png", dpi=150
        )
    print(f"\nWrote the results to {run_dir}")


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def plot_traces(
    columns: list[Column],
    taught: dict[str, NDArray[np.float64]],
    rows: list[str],
    dt: float,
    warmup_steps: int,
    span: list[float],
    title: str,
) -> Figure:
    """One row per trace (a neuron, or a component) and one column per run, over time, over the take's traces."""
    fig = Figure(
        figsize=(3.6 * len(columns) + 1.0, 1.5 * len(rows) + 1.4), facecolor=SURFACE_COLOR, layout="constrained"
    )
    fig.suptitle(title, color="#0b0b0b")
    axes = np.asarray(fig.subplots(len(rows), len(columns), sharex=True, sharey="row", squeeze=False))
    for col, column in enumerate(columns):
        axes[0, col].set_title(column.heading(), color=TEXT_COLOR)
        for row, label in enumerate(rows):
            ax: Any = axes[row, col]
            style(ax)
            ax.axvspan(column.times[0], 0.0, color=WARMUP_COLOR, zorder=0)
            for begin, end in column.spans:
                ax.axvspan(begin, end, color=DISTURBANCE_COLOR, zorder=0)
            for take in taught.values():
                ax.plot(dt * (np.arange(len(take)) - warmup_steps), take[:, row], color=TAKE_COLOR, linewidth=3)
            ax.plot(column.times, column.traces[:, row], color=ESN_COLOR, linewidth=1.0)
            if col == 0:
                ax.set_ylabel(label, color=TEXT_COLOR)
        axes[-1, col].set(xlim=span)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    handles = [
        Patch(color=TAKE_COLOR, label="the take fed in, as in training"),
        Patch(color=ESN_COLOR, label="the ESN driving the robot"),
        Patch(color=WARMUP_COLOR, label="the warm-up"),
    ]
    if any(column.spans for column in columns):
        handles.append(Patch(color=DISTURBANCE_COLOR, label="a disturbance acting"))
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles), frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_planes(
    columns: list[Column],
    taught: dict[str, NDArray[np.float64]],
    components: PrincipalComponents,
    warmup_steps: int,
    title: str,
) -> Figure:
    """One row per pair of components and one column per run: the states in that plane, over the take's.

    The warm-up is dashed and lighter; a hollow dot marks the first state after the
    reset, a filled one the state at the task's start.
    """
    pairs = list(itertools.combinations(range(columns[0].traces.shape[1]), 2))
    fig = Figure(
        figsize=(3.6 * len(columns) + 1.0, 3.2 * len(pairs) + 1.4), facecolor=SURFACE_COLOR, layout="constrained"
    )
    fig.suptitle(title, color="#0b0b0b")
    axes = np.asarray(fig.subplots(len(pairs), len(columns), sharex="row", sharey="row", squeeze=False))
    for col, column in enumerate(columns):
        axes[0, col].set_title(column.heading(), color=TEXT_COLOR)
        for row, (a, b) in enumerate(pairs):
            ax: Any = axes[row, col]
            style(ax)
            for take in taught.values():
                ax.plot(take[: warmup_steps + 1, a], take[: warmup_steps + 1, b], color=TAKE_COLOR, linestyle="--")
                ax.plot(take[warmup_steps:, a], take[warmup_steps:, b], color=TAKE_COLOR, linewidth=3)
            traces = column.traces
            ax.plot(
                traces[: warmup_steps + 1, a], traces[: warmup_steps + 1, b], color=ESN_WARMUP_COLOR, linestyle="--"
            )
            ax.plot(traces[warmup_steps:, a], traces[warmup_steps:, b], color=ESN_COLOR, linewidth=1.0)
            ax.plot(traces[0, a], traces[0, b], marker="o", markerfacecolor="none", color=ESN_COLOR, markersize=7)
            ax.plot(traces[warmup_steps, a], traces[warmup_steps, b], marker="o", color=ESN_COLOR, markersize=6)
            ax.set_xlabel(f"PC{a + 1} ({100 * components.explained[a]:.1f}%)", color=TEXT_COLOR)
            if col == 0:
                ax.set_ylabel(f"PC{b + 1} ({100 * components.explained[b]:.1f}%)", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=TAKE_COLOR, linewidth=3, label="the take fed in, as in training"),
        Line2D([], [], color=TAKE_COLOR, linestyle="--", label="its warm-up"),
        Line2D([], [], color=ESN_COLOR, label="the ESN driving the robot"),
        Line2D([], [], color=ESN_WARMUP_COLOR, linestyle="--", label="its warm-up"),
        Line2D([], [], color=ESN_COLOR, marker="o", markerfacecolor="none", linestyle="none", label="after the reset"),
        Line2D([], [], color=ESN_COLOR, marker="o", linestyle="none", label="at the task's start"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=6, frameon=False, labelcolor=TEXT_COLOR)
    return fig


if __name__ == "__main__":
    main()
