# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Run a trained ESN with warm-ups other than the one it was trained with (Stage 1).

    uv run python experiments/warmup_esn.py \\
        experiments/single_demonstration_autonomous_reaching/warmup_multi_demo_settings.toml

During the warm-up, the held start posture is the ESN's only input, so the ESN
can tell how long ago its reservoir was reset only from how far the reservoir's
transient has gone. If the ESN times its reach from the reset, like a clock, a
warm-up longer by some time makes it arrive earlier by that time. If it times its
reach from the end of the warm-up, its arrival does not move.

The ESN is loaded from a run of ``experiments/autonomous_esn.py`` (``model`` in
``[esn]``), and its training demonstrations from that run's configuration. It runs
from the start postures of ``[evaluation]``, as in ``autonomous_esn.py``, once for
each warm-up duration in ``[warmup]``; the trained model is not changed. The run
directory receives:

- ``warmup.csv``: one row per warm-up and start posture, with the start's offset
  from the demonstrated start and the metrics of
  :func:`arm_esn_ctrl.autonomous.run_metrics` that tell the timing and the course:
  whether the run arrives and holds, the arrival time and its delay after the
  demonstrator's arrival, the first step, the path distance, and the training path
  ratio;
- ``warmup.png``: the arrival delay, the failed runs, and the training path ratio
  against the warm-up, and the hand's distance to the target over time from the
  demonstrated start with each warm-up.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from arm_esn_ctrl.autonomous import Run, Setup, load_setup, run_autonomously, run_metrics
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.storage import resolve_run_path, start_run
from arm_esn_ctrl.tracking import nearest_demonstration

ESN_COLOR = "#2a78d6"
DEMONSTRATOR_COLOR = "#52514e"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    esn_path = resolve_run_path(config["esn"]["model"])
    esn = ReachingEsn.load(esn_path)
    with (esn_path.parent / "config.toml").open("rb") as f:
        demonstrations = tomllib.load(f)["demonstrations"]
    setup = load_setup(
        {"demonstrations": demonstrations, "esn": {"dt": esn.config.dt}, "evaluation": config["evaluation"]}
    )
    trained = esn.config.warmup
    durations = sorted(set(config["warmup"]["durations"]) | {trained})
    print(f"ESN {config['esn']['model']}: trained with a {trained:g} s warm-up")

    rows, demonstrated_runs = [], {}
    for warmup in durations:
        esn.config = dataclasses.replace(esn.config, warmup=warmup)
        runs = run_autonomously(esn, setup)
        rows += [run_row(warmup, i, run, setup) for i, run in enumerate(runs)]
        demonstrated_runs[warmup] = next(run for run in runs if run.start.demonstrated)

    with (run_dir / "warmup.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows, durations, trained)
    title = f"Warm-up: {args.config.stem} (trained with {trained:g} s)"
    plot_warmup(rows, durations, trained, demonstrated_runs, setup, title).savefig(run_dir / "warmup.png", dpi=150)
    print(f"\nWrote the results to {run_dir}")


def run_row(warmup: float, i: int, run: Run, setup: Setup) -> dict[str, Any]:
    """The row of ``warmup.csv`` for start posture ``i`` run with ``warmup``."""
    metrics = run_metrics(run, setup)
    demonstrated = setup.demos[nearest_demonstration(run.start.q, setup.demos)][0]
    offset = np.degrees(run.start.q - demonstrated)
    keys = ("success", "arrival_time_s", "arrival_delay_s", "first_step_m", "reach_path_distance_m")
    return {
        "warmup_s": warmup,
        "start": i,
        "origin": run.start.origin,
        "group": run.start.group,
        "offset_q1_deg": round(float(offset[0]), 6),
        "offset_q2_deg": round(float(offset[1]), 6),
    } | {key: metrics[key] for key in (*keys, "training_path_ratio")}


def by_warmup(rows: list[dict[str, Any]], warmup: float, demonstrated: bool) -> list[dict[str, Any]]:
    """The rows of one warm-up, from the demonstrated start or from the others."""
    return [r for r in rows if r["warmup_s"] == warmup and (r["group"] == "demonstrated") == demonstrated]


def median_of(rows: list[dict[str, Any]], key: str) -> float:
    """The median of ``key`` over ``rows``, leaving out NaN (NaN if nothing is left)."""
    values = [r[key] for r in rows if np.isfinite(r[key])]
    return float(np.median(values)) if values else float("nan")


def print_summary(rows: list[dict[str, Any]], durations: list[float], trained: float) -> None:
    """Print, for each warm-up, the failures, the arrival delay, and the training path ratio."""
    reference = median_of(by_warmup(rows, trained, demonstrated=False), "arrival_delay_s")
    print(
        "Arrival delay: after the demonstrator's arrival, median over the runs that arrive;"
        " shift: from the trained warm-up's (a clock from the reset shifts by minus the change of warm-up)"
    )
    print(
        f"{'warm-up':>8}  {'fail':>8}  {'delay, demonstrated':>19}  {'delay, others':>13}  {'shift':>6}  {'ratio':>5}"
    )
    for warmup in durations:
        others = by_warmup(rows, warmup, demonstrated=False)
        demonstrated = by_warmup(rows, warmup, demonstrated=True)
        failed = sum(not r["success"] for r in others + demonstrated)
        delay = median_of(others, "arrival_delay_s")
        mark = " (trained)" if warmup == trained else ""
        print(
            f"{warmup:7.2f}s  {failed:3d}/{len(others) + len(demonstrated):<4d}"
            f"  {median_of(demonstrated, 'arrival_delay_s'):+18.2f}s  {delay:+12.2f}s  {delay - reference:+5.2f}s"
            f"  {median_of(others, 'training_path_ratio'):5.2f}{mark}"
        )


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def plot_warmup(
    rows: list[dict[str, Any]],
    durations: list[float],
    trained: float,
    demonstrated_runs: dict[float, Run],
    setup: Setup,
    title: str,
) -> Figure:
    """The arrival delay, failures, and training path ratio against the warm-up, and the hand's approach."""
    fig = Figure(figsize=(13, 9), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color="#0b0b0b")
    ax_delay, ax_fail, ax_approach, ax_ratio = fig.subplots(2, 2).flat
    for ax in (ax_delay, ax_fail, ax_approach, ax_ratio):
        style(ax)
    x = np.array(durations)

    others = [by_warmup(rows, w, demonstrated=False) for w in durations]
    delays = [[r["arrival_delay_s"] for r in group if np.isfinite(r["arrival_delay_s"])] for group in others]
    median = np.array([np.median(d) if d else np.nan for d in delays])
    low = np.array([np.percentile(d, 25) if d else np.nan for d in delays])
    high = np.array([np.percentile(d, 75) if d else np.nan for d in delays])
    ax_delay.fill_between(x, low, high, color=ESN_COLOR, alpha=0.2, label="offset starts, middle half")
    ax_delay.plot(x, median, color=ESN_COLOR, marker="o", label="offset starts, median")
    demonstrated = [by_warmup(rows, w, demonstrated=True)[0]["arrival_delay_s"] for w in durations]
    ax_delay.plot(x, demonstrated, color="#0b0b0b", marker="s", linestyle="none", label="demonstrated start")
    at_trained = float(median[durations.index(trained)])
    ax_delay.plot(x, at_trained - (x - trained), color=TEXT_COLOR, linestyle="--", label="a clock from the reset")
    ax_delay.axvline(trained, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    shown = np.concatenate([low, high, demonstrated])
    shown = shown[np.isfinite(shown)]
    margin = 0.1 * max(float(np.ptp(shown)), 0.1)
    # The clock's line leaves the plot where the runs no longer follow it.
    ax_delay.set_ylim(float(shown.min()) - margin, float(shown.max()) + margin)
    ax_delay.set(xlabel="warm-up (s)", ylabel="arrival delay after the demonstrator (s)")
    ax_delay.set_title("Arrival delay against the warm-up (dotted: trained)", color=TEXT_COLOR, fontsize=10)
    ax_delay.legend(fontsize=8)

    n_runs = sum(r["warmup_s"] == trained for r in rows)
    failed = [sum(not r["success"] for r in rows if r["warmup_s"] == w) for w in durations]
    ax_fail.plot(x, failed, color=ESN_COLOR, marker="o")
    ax_fail.axvline(trained, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax_fail.set(xlabel="warm-up (s)", ylabel="runs", ylim=(-0.05 * n_runs, 1.05 * n_runs))
    ax_fail.set_title(f"Failed runs (never arrive or leave the goal), of {n_runs}", color=TEXT_COLOR, fontsize=10)

    norm = Normalize(vmin=-0.3 * max(durations), vmax=max(durations))  # the shortest warm-up stays visible
    reference = demonstrated_runs[trained]
    to_target = np.linalg.norm(reference.hand_ref - setup.target, axis=1)
    ax_approach.plot(setup.times, to_target, color=DEMONSTRATOR_COLOR, linewidth=3, label="demonstrator")
    for warmup, run in demonstrated_runs.items():
        width = 2.0 if warmup == trained else 1.0
        label = f"{warmup:g} s" + (" (trained)" if warmup == trained else "")
        color = "#0b0b0b" if warmup == trained else ESN_COLOR
        alpha = 1.0 if warmup == trained else float(0.25 + 0.75 * norm(warmup))
        to_target = np.linalg.norm(run.hand - setup.target, axis=1)
        ax_approach.plot(setup.times, to_target, color=color, alpha=alpha, linewidth=width, label=label)
    ax_approach.axhline(setup.radius, color=TEXT_COLOR, linestyle=":", linewidth=0.8)
    ax_approach.set(xlabel="time (s)", ylabel="hand distance to the target (m)", xlim=(0.0, 2.0))
    ax_approach.set_title(
        "Hand distance to the target from the demonstrated start, by warm-up (dotted: goal radius)",
        color=TEXT_COLOR,
        fontsize=10,
    )
    ax_approach.legend(fontsize=8, ncol=2)

    ratio = [median_of(group, "training_path_ratio") for group in others]
    ax_ratio.plot(x, ratio, color=ESN_COLOR, marker="o")
    ax_ratio.axvline(trained, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    ax_ratio.set(xlabel="warm-up (s)", ylabel="median ratio", ylim=(0.0, max(1.0, float(np.nanmax(ratio)) * 1.05)))
    ax_ratio.set_title(
        "Training path ratio of the offset starts:\n0 returns onto the demonstration, 1 reaches as the demonstrator",
        color=TEXT_COLOR,
        fontsize=10,
    )
    return fig


if __name__ == "__main__":
    main()
