# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Sweep ESN hyperparameters and summarize the autonomous runs (Stage 1).

    uv run python experiments/sweep_esn.py experiments/multi_demonstration_autonomous_reaching/sweep_tvs_all.toml

The configuration is that of ``experiments/autonomous_esn.py`` plus a ``[sweep]``
table, which lists values for two or three of the ``[esn]`` hyperparameters. Every
combination of those values is trained on the same demonstrations and run from
the same start postures as in ``autonomous_esn.py``. The run directory receives:

- ``sweep.csv``: one row per combination, with its hyperparameters, the metrics of
  :func:`arm_esn_ctrl.autonomous.run_metrics` that compare a run with its
  reference (the demonstrator's reach, or the taught motion when there is no
  demonstrator) averaged over the demonstrated start postures and over the other
  ones, the median of the other ones' training path ratio (or offset retained),
  and the hold over all runs: how many runs fail (never arrive or leave the goal),
  and the median and largest hold error of the runs that arrive;
- ``sweep.png``: heatmaps of the main metrics over the swept values.

The combinations printed as the best are those without failures that stay closest
to the demonstrator's reach from the other starts. With a ``[ranking]`` table, they
are ranked by robustness instead: the fewest failed runs, then, if the table has a
``max_detour`` (m), the fewest swings: runs whose hand strays from the taught path
by more than that beyond where it started; then the fewest jumps: runs whose first
step is longer than ``max_first_step`` (m), rather than a mild return; then the
smallest mean first step from the other starts. ``sweep.csv`` then also counts the
swings and jumps and gives the largest detour and first step, and ``runs.csv`` holds
every run's outcome, first step, and detour, so that other thresholds can be applied
without running again.

To look at one combination in detail, copy its values into a configuration for
``autonomous_esn.py`` and run that.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

from arm_esn_ctrl.autonomous import (
    DEMONSTRATOR,
    Run,
    Setup,
    load_setup,
    rms_degrees,
    run_autonomously,
    run_metrics,
)
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.metrics import distances_to_path
from arm_esn_ctrl.storage import start_run
from arm_esn_ctrl.tracking import nearest_demonstration

TEXT_COLOR = "#52514e"
COLORMAP = "Blues"  # sequential: darker is larger
SURFACE_COLOR = "#fcfcfb"

# The heatmaps: metric column in sweep.csv, title, factor to its display unit, and
# whether its colors follow a logarithmic scale. HEATMAPS compare with the
# demonstrator; TAUGHT_HEATMAPS with the taught motion, when there is no demonstrator.
HEATMAPS = [
    ("other_reach_path_distance_m", "Reach path distance, other starts (mm)", 1000.0, True),
    ("other_reach_joint_error_deg", "Reach joint error, other starts (deg RMS)", 1.0, True),
    ("demonstrated_reach_joint_error_deg", "Reach joint error, demonstrated starts (deg RMS)", 1.0, True),
    ("other_first_step_m", "First step, other starts (mm)", 1000.0, True),
    ("other_median_training_path_ratio", "Training path ratio, other starts (median)", 1.0, False),
    ("failures", "Failed runs: never arrive or leave the goal", 1.0, False),
    ("max_hold_error_m", "Largest hold error of the runs that arrive (mm)", 1000.0, True),
]

TAUGHT_HEATMAPS = [
    ("other_taught_path_distance_m", "Path distance from the taught motion,\nother starts (mm)", 1000.0, True),
    ("other_taught_joint_error_deg", "Joint error from the taught motion,\nother starts (deg RMS)", 1.0, True),
    (
        "demonstrated_taught_joint_error_deg",
        "Joint error from the taught motion,\ndemonstrated starts (deg RMS)",
        1.0,
        True,
    ),
    ("other_first_step_m", "First step, other starts (mm)", 1000.0, True),
    ("other_median_offset_retained", "Offset retained, other starts (median)", 1.0, False),
    ("failures", "Failed runs: never arrive or leave the goal", 1.0, False),
    ("max_hold_error_m", "Largest hold error of the runs that arrive (mm)", 1000.0, True),
]

# Fixed tops of color scales. Far from the training path, ratios grow without bound in
# failed runs; the scale stops at 1.5, as in the grid maps of autonomous_esn.py.
COLOR_TOPS = {"other_median_training_path_ratio": 1.5, "other_median_offset_retained": 1.5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="sweep configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    setup = load_setup(config)
    reference = setup.reference
    sweep: dict[str, list[Any]] = config["sweep"]
    ranking: dict[str, float] | None = config.get("ranking")
    if len(sweep) not in (2, 3):
        msg = f"[sweep] must list two or three hyperparameters, not {len(sweep)}"
        raise ValueError(msg)

    combinations = [dict(zip(sweep, values, strict=True)) for values in itertools.product(*sweep.values())]
    rows, run_rows = [], []
    for i, combination in enumerate(combinations):
        esn = ReachingEsn(EsnConfig(**(config["esn"] | combination)))
        esn.fit(list(setup.demos.values()))
        one_step = rms_degrees(np.vstack([esn.one_step_predictions(q) - q[1:] for q in setup.demos.values()]))
        summary, per_run = summarize(run_autonomously(esn, setup), setup, ranking)
        row = combination | {"one_step_error_deg": one_step} | summary
        rows.append(row)
        run_rows += [combination | r for r in per_run]
        values = ", ".join(f"{name}={value:g}" for name, value in combination.items())
        jumps = "" if ranking is None else f" {row['jumps']} first steps over {1000 * ranking['max_first_step']:g} mm;"
        if ranking is not None and "max_detour" in ranking:
            jumps = f" {row['swings']} swings;{jumps}"
        hold = f"largest hold error {1000 * row['max_hold_error_m']:.1f} mm"
        print(
            f"[{i + 1:3d}/{len(combinations)}] {values}:"
            f" path distance from the {reference.name}, other starts"
            f" {1000 * row[f'other_{reference.path_distance}']:6.1f} mm;"
            f" {row['failures']} of {row['runs']} runs fail;{jumps} {hold}"
        )

    with (run_dir / "sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if run_rows:
        with (run_dir / "runs.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(run_rows[0]))
            writer.writeheader()
            writer.writerows(run_rows)
    if ranking is None:
        print_best(rows, list(sweep))
    else:
        print_ranking(rows, list(sweep), ranking)
    maps = HEATMAPS if reference is DEMONSTRATOR else TAUGHT_HEATMAPS
    if ranking is not None:  # the swings and jumps, before the hold error
        ranked_maps = []
        if "max_detour" in ranking:
            title = f"Runs that stray over {1000 * ranking['max_detour']:g} mm beyond their start"
            ranked_maps.append(("swings", title, 1.0, False))
        title = f"Runs whose first step is over {1000 * ranking['max_first_step']:g} mm"
        ranked_maps.append(("jumps", title, 1.0, False))
        maps = [*maps[:-1], *ranked_maps, maps[-1]]
    title = f"ESN sweep: {args.config.stem}"
    plot_sweep(rows, sweep, maps, title).savefig(run_dir / "sweep.png", dpi=150)
    print(f"\nWrote the results to {run_dir}")


def summarize(
    runs: list[Run], setup: Setup, ranking: dict[str, float] | None = None
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Average the metrics against the reference by kind of start posture, and count and measure the holds of all runs.

    The training path ratio (or offset retained) of the other start postures is
    their median: near the training path, a few ratios grow large. Those without a
    ratio are left out. With a ``ranking``, the runs whose first step is longer than
    its ``max_first_step`` are counted as jumps and, if it has a ``max_detour``, the
    runs whose :func:`detour` is longer than that as swings; the second result then
    lists every run's outcome, first step, and detour (otherwise it is empty).
    """
    reference = setup.reference
    metrics = [run_metrics(run, setup) for run in runs]
    summary: dict[str, float] = {}
    for kind, demonstrated in (("demonstrated", True), ("other", False)):
        group = [m for m, run in zip(metrics, runs, strict=True) if run.start.demonstrated == demonstrated]
        for name in ("first_step_m", reference.path_distance, reference.joint_error):
            summary[f"{kind}_{name}"] = float(np.mean([m[name] for m in group])) if group else float("nan")
    ratios = [m[reference.course] for m, run in zip(metrics, runs, strict=True) if not run.start.demonstrated]
    ratios = [r for r in ratios if np.isfinite(r)]
    summary[f"other_median_{reference.course}"] = float(np.median(ratios)) if ratios else float("nan")
    per_run: list[dict[str, Any]] = []
    if ranking is not None:
        first_steps = [m["first_step_m"] for m in metrics]
        detours = [detour(run, setup) for run in runs] if "max_detour" in ranking else [float("nan")] * len(runs)
        if "max_detour" in ranking:
            summary["swings"] = sum(d > ranking["max_detour"] for d in detours)
            summary["largest_detour_m"] = float(np.max(detours))
        summary["jumps"] = sum(step > ranking["max_first_step"] for step in first_steps)
        summary["largest_first_step_m"] = float(np.max(first_steps))
        per_run = [
            {"start": i, "origin": run.start.origin, "success": m["success"], "first_step_m": m["first_step_m"]}
            | {"detour_m": d, "hold_error_m": m["hold_error_m"]}
            for i, (run, m, d) in enumerate(zip(runs, metrics, detours, strict=True))
        ]
    hold_errors = [m["hold_error_m"] for m in metrics if m["arrived"]]
    summary["runs"] = len(metrics)
    summary["failures"] = sum(not m["success"] for m in metrics)
    summary["median_hold_error_m"] = float(np.median(hold_errors)) if hold_errors else float("nan")
    summary["max_hold_error_m"] = float(np.max(hold_errors)) if hold_errors else float("nan")
    return summary, per_run


def detour(run: Run, setup: Setup, every: int = 5) -> float:
    """How much farther from the taught hand path the run's hand strays than where it started (m).

    The taught path is that of the training demonstration that starts nearest; both
    paths are compared every ``every`` samples, which is plenty for detours of
    centimeters and keeps a sweep fast.
    """
    taught = setup.demo_hands[nearest_demonstration(run.start.q, setup.demos)]
    distances = distances_to_path(run.hand[::every], taught[::every])
    return float(distances.max() - distances[0])


def print_best(rows: list[dict[str, Any]], names: list[str], count: int = 5) -> None:
    """Print the combinations without failures that have the smallest reach path distance from the other starts."""
    succeeding = [r for r in rows if r["failures"] == 0]
    best = sorted(succeeding, key=lambda r: r["other_reach_path_distance_m"])[:count]
    print(f"\n{len(succeeding)} of {len(rows)} combinations arrive and stay at the goal from every start.")
    print("Smallest reach path distance from the other starts among them:")
    for r in best:
        values = ", ".join(f"{name}={r[name]:g}" for name in names)
        print(
            f"  {values}: reach path distance {1000 * r['other_reach_path_distance_m']:.1f} mm (other),"
            f" {1000 * r['demonstrated_reach_path_distance_m']:.1f} mm (demonstrated);"
            f" reach joint error {r['other_reach_joint_error_deg']:.2f} deg (other),"
            f" {r['demonstrated_reach_joint_error_deg']:.2f} deg (demonstrated);"
            f" first step {1000 * r['other_first_step_m']:.1f} mm (other);"
            f" training path ratio {r['other_median_training_path_ratio']:.2f} (other, median);"
            f" hold error median {1000 * r['median_hold_error_m']:.1f} mm,"
            f" largest {1000 * r['max_hold_error_m']:.1f} mm"
        )


def print_ranking(rows: list[dict[str, Any]], names: list[str], ranking: dict[str, float], count: int = 5) -> None:
    """Print the most robust combinations: the fewest failed runs, swings, and jumps, then the mildest first step.

    A swing is a run that strays from the taught path by more than ``max_detour``
    (m), if the ``ranking`` has one; a jump, a first step longer than
    ``max_first_step`` (m); the mildest first step is the smallest mean first step
    from the other starts.
    """
    ranked = sorted(rows, key=lambda r: (r["failures"], r.get("swings", 0), r["jumps"], r["other_first_step_m"]))
    robust = sum(r["failures"] == 0 for r in rows)
    print(f"\n{robust} of {len(rows)} combinations arrive and stay at the goal from every start.")
    swings = f"swings over {1000 * ranking['max_detour']:g} mm, then " if "max_detour" in ranking else ""
    print(f"The fewest failed runs, then {swings}first steps over {1000 * ranking['max_first_step']:g} mm:")
    for r in ranked[:count]:
        values = ", ".join(f"{name}={r[name]:g}" for name in names)
        swung = f"{r['swings']} swing (largest {1000 * r['largest_detour_m']:.0f} mm), " if "swings" in r else ""
        print(
            f"  {values}: {r['failures']} of {r['runs']} runs fail, {swung}{r['jumps']} jump;"
            f" first step {1000 * r['other_first_step_m']:.1f} mm (other, mean),"
            f" {1000 * r['largest_first_step_m']:.1f} mm (largest);"
            f" offset retained {r['other_median_offset_retained']:.2f} (other, median);"
            f" hold error median {1000 * r['median_hold_error_m']:.1f} mm,"
            f" largest {1000 * r['max_hold_error_m']:.1f} mm"
        )


def plot_sweep(
    rows: list[dict[str, Any]],
    sweep: dict[str, list[Any]],
    maps: list[tuple[str, str, float, bool]],
    title: str,
) -> Figure:
    """Heatmaps of the ``maps``: the first two swept values on the axes, one column per third value."""
    names = list(sweep)
    x_name, y_name = names[0], names[1]
    panel_values = sweep[names[2]] if len(names) == 3 else [None]
    fig = Figure(
        figsize=(3.6 * len(panel_values) + 0.8, 3.2 * len(maps)), facecolor=SURFACE_COLOR, layout="constrained"
    )
    fig.suptitle(title, color="#0b0b0b")
    axes = fig.subplots(len(maps), len(panel_values), squeeze=False)

    for row_axes, (metric, label, factor, log) in zip(axes, maps, strict=True):
        values = np.array([r[metric] for r in rows], dtype=float) * factor
        finite = values[np.isfinite(values)]
        top = float(finite.max()) if finite.size else 1.0
        if metric in COLOR_TOPS:
            top = COLOR_TOPS[metric]  # larger values take the darkest color; the cells print them
        # One color scale per metric, shared by its panels.
        norm = LogNorm(vmin=max(float(finite.min()), 1e-3), vmax=top) if log and finite.size else Normalize(0.0, top)
        if metric in ("failures", "jumps"):
            label = f"{label} (of {rows[0]['runs']})"
        for ax, panel_value in zip(row_axes, panel_values, strict=True):
            grid = np.full((len(sweep[y_name]), len(sweep[x_name])), np.nan)
            for r in rows:
                if panel_value is None or r[names[2]] == panel_value:
                    grid[sweep[y_name].index(r[y_name]), sweep[x_name].index(r[x_name])] = r[metric] * factor
            ax.imshow(grid, cmap=COLORMAP, norm=norm, origin="lower", aspect="auto")
            for (y, x), value in np.ndenumerate(grid):
                dark = np.isfinite(value) and norm(value) > 0.6
                text = f"{value:.3g}" if np.isfinite(value) else "-"
                ax.text(x, y, text, ha="center", va="center", fontsize=7, color="white" if dark else "#0b0b0b")
            ax.set_xticks(range(len(sweep[x_name])), [f"{v:g}" for v in sweep[x_name]])
            ax.set_yticks(range(len(sweep[y_name])), [f"{v:g}" for v in sweep[y_name]])
            ax.set_xlabel(x_name, color=TEXT_COLOR)
            ax.set_ylabel(y_name, color=TEXT_COLOR)
            ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR, labelsize=8)
            panel = "" if panel_value is None else f"{names[2]} = {panel_value:g}\n"
            ax.set_title(f"{panel}{label}", color="#0b0b0b", fontsize=9)
        fig.colorbar(ScalarMappable(norm=norm, cmap=COLORMAP), ax=row_axes.tolist(), shrink=0.9)
    return fig


if __name__ == "__main__":
    main()
