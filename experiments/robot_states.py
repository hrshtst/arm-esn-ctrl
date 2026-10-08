# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Plot a trained ESN's reservoir states while it drives the robot, for neurons picked at random (Stage 2).

    uv run python experiments/robot_states.py \\
        experiments/manual_demonstration_v2_robot_tracking/states_robot_fine_best_filtered.toml

The runs of ``[[run]]`` are runs of ``experiments/robot_esn.py`` that share their
ESN, each shown from one of its start postures. They do not store the reservoir
states: :func:`arm_esn_ctrl.states.run_states` recomputes them from each run's log,
and refuses a log whose reference the ESN does not reproduce. For comparison, the
take the ESN was trained on is fed in as in training (teacher forcing). For each
tracker setting of ``[states] settings``, the run directory receives
``states_<setting>.png``: the states of ``count`` neurons picked at random
(``seed``), one row each, and one column per run, over the take's states, from the
warm-up on, with the warm-up and the run's disturbances shaded.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray
from skelarm import StateLog

from arm_esn_ctrl.demonstrations import load_joint_angles
from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.states import run_states, teacher_forced_states
from arm_esn_ctrl.storage import resolve_run_path, start_run
from arm_esn_ctrl.tracking import nearest_demonstration

ESN_COLOR = "#2a78d6"
TAKE_COLOR = "#a3a29d"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
WARMUP_COLOR = "#ecebe6"
DISTURBANCE_COLOR = "#fbe0d2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    options, runs = config["states"], config["run"]
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
    with (model.parent / "config.toml").open("rb") as f:
        demonstrations = tomllib.load(f)["demonstrations"]
    demo_dir = resolve_run_path(demonstrations["run"])
    demos = {
        name.split(".")[0]: load_joint_angles(demo_dir / name, esn.config.dt)[1] for name in demonstrations["train"]
    }
    taught = {name: teacher_forced_states(esn, q) for name, q in demos.items()}
    rng = np.random.default_rng(options["seed"])
    neurons = np.sort(rng.choice(esn.config.n_neurons, options["count"], replace=False))
    print(f"ESN {models[0]}: {esn.config.n_neurons} neurons; shown: {neurons.tolist()}")

    for setting in options["settings"]:
        columns = []
        for run, robot in zip(runs, robot_configs, strict=True):
            log = StateLog.load(resolve_run_path(run["path"]) / setting / f"esn_{run['start']:02d}.sklog.npz")
            times, states = run_states(esn, log)
            start_q = log.channel("q").reshape(len(log.times), -1)[0]
            offset = np.degrees(start_q - demos[nearest_demonstration(start_q, demos)][0])
            columns.append((run["label"], offset, times, states, disturbance_spans(robot.get("disturbance"))))
            print(f"  {setting}, {run['label']}: {len(states)} states recomputed; the logged reference reproduced")
        title = (
            f"{args.config.stem}: the reservoir states with the tracker {setting},"
            f" {len(neurons)} of {esn.config.n_neurons} neurons picked at random (seed {options['seed']})"
        )
        fig = plot_states(columns, taught, neurons, esn.config.dt, esn.config.warmup_steps, options["span"], title)
        fig.savefig(run_dir / f"states_{setting}.png", dpi=150)
    print(f"\nWrote the results to {run_dir}")


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def plot_states(
    columns: list[tuple[str, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], list[tuple[float, float]]]],
    taught: dict[str, NDArray[np.float64]],
    neurons: NDArray[np.int64],
    dt: float,
    warmup_steps: int,
    span: list[float],
    title: str,
) -> Figure:
    """One row per neuron and one column per run: the run's states over the take's, fed in as in training.

    Each column is a run's label, its start posture's offset (deg) from the
    demonstrated start, its times, its states, and its disturbances' spans.
    """
    fig = Figure(
        figsize=(3.6 * len(columns) + 1.0, 1.3 * len(neurons) + 1.4), facecolor=SURFACE_COLOR, layout="constrained"
    )
    fig.suptitle(title, color="#0b0b0b")
    axes = np.asarray(fig.subplots(len(neurons), len(columns), sharex=True, sharey="row", squeeze=False))
    for col, (label, offset, times, states, spans) in enumerate(columns):
        axes[0, col].set_title(f"{label}\nstart offset ({offset[0]:+.0f}, {offset[1]:+.0f}) deg", color=TEXT_COLOR)
        for row, neuron in enumerate(neurons):
            ax: Any = axes[row, col]
            style(ax)
            ax.axvspan(span[0], 0.0, color=WARMUP_COLOR, zorder=0)
            for begin, end in spans:
                ax.axvspan(begin, end, color=DISTURBANCE_COLOR, zorder=0)
            for take in taught.values():
                ax.plot(dt * (np.arange(len(take)) - warmup_steps), take[:, neuron], color=TAKE_COLOR, linewidth=3)
            ax.plot(times, states[:, neuron], color=ESN_COLOR, linewidth=1.0)
            if col == 0:
                ax.set_ylabel(f"neuron {neuron}", color=TEXT_COLOR)
        axes[-1, col].set(xlim=span)
        axes[-1, col].set_xlabel("time (s)", color=TEXT_COLOR)
    handles = [
        Patch(color=TAKE_COLOR, label="the take fed in, as in training"),
        Patch(color=ESN_COLOR, label="the ESN driving the robot"),
        Patch(color=WARMUP_COLOR, label="the warm-up"),
        Patch(color=DISTURBANCE_COLOR, label="a disturbance acting"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False, labelcolor=TEXT_COLOR)
    return fig


if __name__ == "__main__":
    main()
