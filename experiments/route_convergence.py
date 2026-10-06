# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Measure how the autonomous runs of an ESN gather onto a common route (Stage 1).

    uv run python experiments/route_convergence.py \\
        experiments/manual_demonstration_autonomous_reaching/route_candidate_a_raw.toml

The configuration names a run directory of ``experiments/autonomous_esn.py``
(``[runs] grid``): an ESN run from a grid of start postures. Their hand paths are
compared every ``step`` seconds (``[convergence]``), regardless of timing:

- the **spread** of a run at a time is the median distance of its hand from the
  routes the other runs take (:func:`arm_esn_ctrl.metrics.route_spread`). Runs that
  gather onto a common route lose their spread from where they join it, however
  late or early they get there;
- a run **joins** the common route when its spread drops below ``join_distance``
  for good (:func:`arm_esn_ctrl.metrics.join_index`). Since every route that
  arrives ends at the target, joining only says something when it happens away
  from the target, so the hand's distance to the target when it joins is recorded;
- the **common route** is the route of the run with the smallest mean spread
  before it reaches the goal (afterwards, every run that arrives is at the
  target); how far it strays from the taught motion tells whether the runs gather
  onto the taught route or onto another one.

The run directory receives:

- ``convergence.csv``: per start posture, when the run joins the common route, how
  far its hand then is from the target and from the taught path, and the largest
  distance of its hand from the taught path;
- ``convergence.png``: the hand paths with the common route and the taught path;
  over time, across the runs, the spread, the distance from the taught path, and
  the distance to the target (median and 10th to 90th percentiles); and, over the
  start offsets, how far each run strays from the taught path.
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
from skelarm import StateLog

from arm_esn_ctrl.autonomous import Setup, load_setup
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.metrics import distances_to_path, join_index, route_spread
from arm_esn_ctrl.storage import resolve_run_path, start_run

RUN_COLOR = "#2a78d6"
ROUTE_COLOR = "#0b0b0b"
TAUGHT_COLOR = "#eb6834"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    grid_dir = resolve_run_path(config["runs"]["grid"])
    with (grid_dir / "config.toml").open("rb") as f:
        setup = load_setup(tomllib.load(f))
    options = config["convergence"]
    every = round(options["step"] / (setup.times[1] - setup.times[0]))
    times = setup.times[::every]
    hands = [
        endpoint_positions(setup.skeleton, StateLog.load(grid_dir / f"esn_{i:02d}.sklog.npz").channel("q"))[::every]
        for i in range(len(setup.starts))
    ]
    (taught_q,) = setup.demos.values()  # one taught motion
    taught = endpoint_positions(setup.skeleton, taught_q)
    print(f"Runs of {config['runs']['grid']}: {len(hands)} start postures, compared every {options['step']:g} s")

    spread = route_spread(hands)
    from_taught = np.array([distances_to_path(hand, taught) for hand in hands])
    to_target = np.array([np.linalg.norm(hand - setup.target, axis=1) for hand in hands])
    outside = to_target > setup.radius  # before reaching the goal
    before = [float(row[out].mean()) if out.any() else np.inf for row, out in zip(spread, outside, strict=True)]
    common = int(np.argmin(before))
    route = hands[common]

    rows = []
    for i, start in enumerate(setup.starts):
        joined = join_index(spread[i], options["join_distance"])
        rows.append(
            {"start": i, "origin": start.origin}
            | {
                "join_time_s": float("nan") if joined is None else float(times[joined]),
                "target_distance_at_join_m": float("nan") if joined is None else float(to_target[i, joined]),
                "taught_distance_at_join_m": float("nan") if joined is None else float(from_taught[i, joined]),
                "largest_taught_distance_m": float(from_taught[i].max()),
            }
        )
    with (run_dir / "convergence.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows, route, taught, setup, options["join_distance"], common)
    title = f"Route convergence: {args.config.stem}"
    plot_convergence(times, hands, route, taught, spread, from_taught, to_target, setup, title).savefig(
        run_dir / "convergence.png", dpi=150
    )
    print(f"\nWrote the results to {run_dir}")


def print_summary(
    rows: list[dict[str, Any]],
    route: NDArray[np.float64],
    taught: NDArray[np.float64],
    setup: Setup,
    join_distance: float,
    common: int,
) -> None:
    """Print when and where the runs join the common route, and how far that route is from the taught one."""
    joined = [r for r in rows if np.isfinite(r["join_time_s"])]
    away = [r for r in joined if r["target_distance_at_join_m"] > setup.radius]

    def quartiles(name: str, factor: float = 1.0) -> str:
        values = factor * np.array([r[name] for r in joined])
        return "-" if not len(values) else "{:.2f} / {:.2f} / {:.2f}".format(*np.percentile(values, [25, 50, 75]))

    print(f"Common route: the run from {setup.starts[common].origin}, the smallest mean spread")
    print(
        f"  it strays up to {1000 * float(distances_to_path(route, taught).max()):.0f} mm from the taught path;"
        f" the taught path strays up to {1000 * float(distances_to_path(taught, route).max()):.0f} mm from it"
    )
    print(
        f"{len(joined)} of {len(rows)} runs join it (spread below {1000 * join_distance:g} mm for good),"
        f" {len(away)} of them before reaching the goal"
    )
    print(f"  join time (s), quartiles: {quartiles('join_time_s')}")
    print(f"  distance to the target when joining (m), quartiles: {quartiles('target_distance_at_join_m')}")
    print(f"  distance from the taught path when joining (m), quartiles: {quartiles('taught_distance_at_join_m')}")
    largest = np.array([r["largest_taught_distance_m"] for r in rows])
    print(f"Largest distance from the taught path (m), quartiles: {np.percentile(largest, [25, 50, 75]).round(3)}")


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def band(ax: Axes, times: NDArray[np.float64], values: NDArray[np.float64], label: str) -> None:
    """The median of ``values`` across runs (rows) over ``times``, with its 10th to 90th percentiles shaded."""
    low, median, high = np.percentile(values, [10, 50, 90], axis=0)
    ax.fill_between(times, 1000 * low, 1000 * high, color=RUN_COLOR, alpha=0.25, linewidth=0)
    ax.plot(times, 1000 * median, color=RUN_COLOR, linewidth=1.5)
    ax.set(title=label, xlabel="time (s)", ylabel="mm", ylim=(0, None))


def plot_convergence(
    times: NDArray[np.float64],
    hands: list[NDArray[np.float64]],
    route: NDArray[np.float64],
    taught: NDArray[np.float64],
    spread: NDArray[np.float64],
    from_taught: NDArray[np.float64],
    to_target: NDArray[np.float64],
    setup: Setup,
    title: str,
) -> Figure:
    """The hand paths with the common route and the taught path, and the spread and distances over time."""
    fig = Figure(figsize=(18, 9), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color="#0b0b0b")
    ax_paths, ax_spread, ax_map, ax_taught, ax_target, ax_free = fig.subplots(2, 3).flat
    ax_free.set_axis_off()
    for ax in (ax_paths, ax_spread, ax_taught, ax_target):
        style(ax)
    for k, hand in enumerate(hands):
        ax_paths.plot(*hand.T, color=RUN_COLOR, linewidth=0.6, alpha=0.35, label="runs" if k == 0 else None)
    ax_paths.plot(*taught.T, color=TAUGHT_COLOR, linewidth=3, label="taught path")
    ax_paths.plot(*route.T, color=ROUTE_COLOR, linewidth=1.5, label="common route")
    ax_paths.plot(*setup.target, marker="+", markersize=12, color=ROUTE_COLOR, markeredgewidth=1.5)
    ax_paths.set(title="Hand paths", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax_paths.legend(frameon=False, labelcolor=TEXT_COLOR, fontsize=8)
    band(ax_spread, times, spread, "Spread: distance from the other runs' routes")
    band(ax_taught, times, from_taught, "Distance from the taught path")
    band(ax_target, times, to_target, "Distance to the target")
    ax_target.axhline(1000 * setup.radius, color=TEXT_COLOR, linewidth=0.8, linestyle=":")
    plot_offset_map(ax_map, setup, 1000 * from_taught.max(axis=1), "Largest distance from the taught path (mm)")
    return fig


def plot_offset_map(ax: Axes, setup: Setup, values: NDArray[np.float64], title: str) -> None:
    """``values`` of the runs over their start's offset from the demonstrated start (deg), one cell per start."""
    demo_start = next(iter(setup.demos.values()))[0]
    offsets = np.degrees(np.array([start.q - demo_start for start in setup.starts])).round(6)
    x_values, y_values = np.unique(offsets[:, 0]), np.unique(offsets[:, 1])
    grid = np.full((len(y_values), len(x_values)), np.nan)
    for (dx, dy), value in zip(offsets, values, strict=True):
        grid[np.searchsorted(y_values, dy), np.searchsorted(x_values, dx)] = value
    step = x_values[1] - x_values[0] if len(x_values) > 1 else 1.0
    extent = (x_values[0] - step / 2, x_values[-1] + step / 2, y_values[0] - step / 2, y_values[-1] + step / 2)
    image = ax.imshow(grid, origin="lower", extent=extent, cmap="Blues", aspect="equal")
    ax.figure.colorbar(image, ax=ax, shrink=0.85)
    ax.plot(0.0, 0.0, marker="+", markersize=12, color=ROUTE_COLOR, markeredgewidth=1.5)
    ax.set(title=title, xlabel="joint 1 offset (deg)", ylabel="joint 2 offset (deg)")


if __name__ == "__main__":
    main()
