# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Sweep ESN hyperparameters and summarize the autonomous runs (Stage 1).

    uv run python experiments/sweep_esn.py configs/esn/sweep_tvs_all.toml

The configuration is that of ``experiments/autonomous_esn.py`` plus a ``[sweep]``
table, which lists values for two or three of the ``[esn]`` hyperparameters. Every
combination of those values is trained on the same demonstrations and run from
the same start postures as in ``autonomous_esn.py``. The run directory receives:

- ``sweep.csv``: one row per combination, with its hyperparameters and the
  metrics of :func:`arm_esn_ctrl.autonomous.run_metrics`, averaged over the
  demonstrated start postures and over the other ones;
- ``sweep.png``: heatmaps of the main metrics over the swept values.

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
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from arm_esn_ctrl.autonomous import Run, Setup, load_setup, rms_degrees, run_autonomously, run_metrics
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.storage import start_run

TEXT_COLOR = "#52514e"
COLORMAP = "Blues"  # sequential: darker is larger
SURFACE_COLOR = "#fcfcfb"
# A run counts as reaching the goal if its hand ends within this distance of the target.
GOAL_TOLERANCE = 0.01  # m

# The heatmaps: metric column in sweep.csv, title, and factor to its display unit.
HEATMAPS = [
    ("other_path_distance_m", "Path distance, other starts (mm)", 1000.0),
    ("other_joint_rms_error_deg", "Joint error, other starts (deg RMS)", 1.0),
    ("demonstrated_joint_rms_error_deg", "Joint error, demonstrated starts (deg RMS)", 1.0),
    ("worst_final_error_m", "Worst final error, all starts (mm)", 1000.0),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="sweep configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    setup = load_setup(config)
    sweep: dict[str, list[Any]] = config["sweep"]
    if len(sweep) not in (2, 3):
        msg = f"[sweep] must list two or three hyperparameters, not {len(sweep)}"
        raise ValueError(msg)

    combinations = [dict(zip(sweep, values, strict=True)) for values in itertools.product(*sweep.values())]
    rows = []
    for i, combination in enumerate(combinations):
        esn = ReachingEsn(EsnConfig(**(config["esn"] | combination)))
        esn.fit(list(setup.demos.values()))
        one_step = rms_degrees(np.vstack([esn.one_step_predictions(q) - q[1:] for q in setup.demos.values()]))
        row = combination | {"one_step_error_deg": one_step} | summarize(run_autonomously(esn, setup), setup)
        rows.append(row)
        values = ", ".join(f"{name}={value:g}" for name, value in combination.items())
        print(
            f"[{i + 1:3d}/{len(combinations)}] {values}:"
            f" path distance, other starts {1000 * row['other_path_distance_m']:6.1f} mm;"
            f" worst final error {1000 * row['worst_final_error_m']:.1f} mm"
        )

    with (run_dir / "sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_best(rows, list(sweep))
    plot_sweep(rows, sweep, title=f"ESN sweep: {args.config.stem}").savefig(run_dir / "sweep.png", dpi=150)
    print(f"\nWrote the results to {run_dir}")


def summarize(runs: list[Run], setup: Setup) -> dict[str, float]:
    """Average the run metrics over the demonstrated start postures and over the other ones."""
    metrics = [run_metrics(run, setup) for run in runs]
    summary = {}
    for kind, demonstrated in (("demonstrated", True), ("other", False)):
        group = [m for m, run in zip(metrics, runs, strict=True) if run.start.demonstrated == demonstrated]
        for name in ("first_step_m", "joint_rms_error_deg", "path_distance_m"):
            summary[f"{kind}_{name}"] = float(np.mean([m[name] for m in group])) if group else float("nan")
    summary["worst_final_error_m"] = max(m["final_error_m"] for m in metrics)
    return summary


def print_best(rows: list[dict[str, Any]], names: list[str], count: int = 5) -> None:
    """Print the combinations with the smallest path distance from the other starts that reach the goal."""
    reaching = [r for r in rows if r["worst_final_error_m"] < GOAL_TOLERANCE]
    best = sorted(reaching, key=lambda r: r["other_path_distance_m"])[:count]
    print(
        f"\n{len(reaching)} of {len(rows)} combinations end within {1000 * GOAL_TOLERANCE:g} mm"
        " of the target from every start."
    )
    print("Smallest path distance from the other starts among them:")
    for r in best:
        values = ", ".join(f"{name}={r[name]:g}" for name in names)
        print(
            f"  {values}: path distance {1000 * r['other_path_distance_m']:.1f} mm (other),"
            f" {1000 * r['demonstrated_path_distance_m']:.1f} mm (demonstrated);"
            f" joint error {r['other_joint_rms_error_deg']:.2f} deg (other),"
            f" {r['demonstrated_joint_rms_error_deg']:.2f} deg (demonstrated);"
            f" first step {1000 * r['other_first_step_m']:.1f} mm (other)"
        )


def plot_sweep(rows: list[dict[str, Any]], sweep: dict[str, list[Any]], title: str) -> Figure:
    """Heatmaps of the main metrics: the first two swept values on the axes, one column per third value."""
    names = list(sweep)
    x_name, y_name = names[0], names[1]
    panel_values = sweep[names[2]] if len(names) == 3 else [None]
    fig = Figure(
        figsize=(3.6 * len(panel_values) + 0.8, 3.2 * len(HEATMAPS)), facecolor=SURFACE_COLOR, layout="constrained"
    )
    fig.suptitle(title, color="#0b0b0b")
    axes = fig.subplots(len(HEATMAPS), len(panel_values), squeeze=False)

    for row_axes, (metric, label, factor) in zip(axes, HEATMAPS, strict=True):
        values = np.array([r[metric] for r in rows], dtype=float) * factor
        norm = LogNorm(vmin=max(values.min(), 1e-3), vmax=values.max())  # one scale per metric
        for ax, panel_value in zip(row_axes, panel_values, strict=True):
            grid = np.full((len(sweep[y_name]), len(sweep[x_name])), np.nan)
            for r in rows:
                if panel_value is None or r[names[2]] == panel_value:
                    grid[sweep[y_name].index(r[y_name]), sweep[x_name].index(r[x_name])] = r[metric] * factor
            ax.imshow(grid, cmap=COLORMAP, norm=norm, origin="lower", aspect="auto")
            for (y, x), value in np.ndenumerate(grid):
                dark = norm(value) > 0.6
                ax.text(
                    x, y, f"{value:.3g}", ha="center", va="center", fontsize=7, color="white" if dark else "#0b0b0b"
                )
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
