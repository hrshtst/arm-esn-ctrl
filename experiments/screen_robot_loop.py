# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Take every fixed point of the screens with the arm: how the robot changes how they attract, and what that tells.

    uv run python experiments/screen_robot_loop.py \\
        experiments/manual_demonstration_v2_consistent_robot_loop/screen_robot_loop.toml --workers 30

The configuration names runs of ``screen_esns.py`` and, for each, the run of
``screen_against_robot.py`` that set it against the robot (``[runs]``), and a run of
``release_robot.py`` of the first screen's ESNs. Each ESN of the screens is trained as
the screen trained it and drives the arm with one tracker setting (``[tracker]``, as in
``robot_esn.py``); the ESN and the arm are read as one map of the reservoir's state and
the arm's joint angles and velocities (:class:`arm_esn_ctrl.dynamics.RobotLoop`). Each
fixed point the screen found, the arm at rest at its posture, is one of the system's, and
the eigenvalues of the system's Jacobian there tell whether it attracts with the arm, and
how fast, against the ESN's own. The screen keeps each fixed point's posture, not its
state, which is found again (``found``): the held reservoir's fixed point at the posture,
from the reset (``reset``); else, as the screen found it, by Newton's method on R from
the grid's cells around the posture (``cells``, :func:`arm_esn_ctrl.screening.map_zero`);
else from random states (``random``); each refined by Newton's method on the ESN's own
map, and kept only if it gives the screen's posture and the slowest mode's growth rate.
A held reservoir with several fixed points at a posture, as at large spectral radii,
can settle on another from the reset. A fixed point not found again (``found`` empty)
has no eigenvalues with the arm: rule (a) with the arm takes it as on its own, and the
releases and the changes leave it out. A fixed point beyond the arm's joint stops (not
``within_stops``), where the ESN on its own can rest, is none of the system's: the arm
cannot rest there, and rule (a) with the arm and the changes leave it out. From them:

- **the releases**: at each of the release run's releases, how long the arm stays at the
  fixed point, against the slowest mode's growth rate on its own and with the arm: the
  rank correlation, and the area under the ROC curve for the arm staying the whole
  release (an infinite stay), over all the releases and over the leak rates up to and
  from ``fast_from`` (``[release]``);
- **rule (a) with the arm**: an ESN is flagged if it has a fixed point that attracts
  with the arm and whose hand is outside the goal radius, against rule (a), which takes
  the ESN on its own; each flag's area under the ROC curve for failing on the robot, as
  ``screen_against_robot.py`` counts it, over all the ESNs and within each leak rate;
- **the changes**: within each leak rate, the fixed points whose stability the arm
  changes, and the slowest mode's growth rate with the arm against on its own.

The run directory receives ``fixed_points_<n>.csv``, the screen ``n``'s fixed points
(from 0, in ``[runs]``' order) with the system's eigenvalues; ``release.csv``, the
release run's rows with the growth rate with the arm; ``esns_<n>.csv``, each ESN's
outcome and flags; ``changes.csv``; and ``scores.csv``. The ESNs are spread over
``--workers`` processes, each computing with one thread.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing
import os
import tomllib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from robot_esn import tracker_settings
from screen_against_robot import auc, spearman

from arm_esn_ctrl.autonomous import Setup, load_setup
from arm_esn_ctrl.dynamics import ClosedLoop, RobotLoop
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.postures import joint_columns, joint_values, map_grid
from arm_esn_ctrl.screening import PostureMap, cell_zeros, held_fixed_points
from arm_esn_ctrl.storage import resolve_run_path, start_run
from arm_esn_ctrl.tracking import TrackerConfig, gain_posture

SWEPT = ("leak_rate", "spectral_radius", "input_scaling", "seed")
KEPT = ("target_distance_m", "path_distance_deg", "progress", "n_unstable", "growth_per_s")  # and the posture
POSTURE_TOLERANCE_DEG = 1e-6  # a state found again gives the screen's posture this closely ...
GROWTH_TOLERANCE = 1e-6  # ... and its slowest mode's growth rate (1/s)
RANDOM_STARTS = 20  # the random states a fixed point's state is sought from, last
Key = tuple[float, float, float, int]


@dataclass(frozen=True)
class Context:
    """What every ESN needs, loaded once per process."""

    esn: dict[str, Any]  # the screen's [esn] table
    setup: Setup
    take: NDArray[np.float64]
    simulator: dict[str, Any]
    tracker: TrackerConfig
    gains: tuple[NDArray[np.float64], NDArray[np.float64]]
    grids_deg: list[NDArray[np.float64]]  # the screen's grid of R, each joint's angles


_contexts: dict[str, Context] = {}  # this process's, by screen run, set by init_worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="configuration file (TOML)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="worker processes")
    args = parser.parse_args()
    config, run_dir = start_run(args.config)
    screens, outcomes = config["runs"]["screens"], config["runs"]["outcomes"]
    if len(screens) != len(outcomes):
        msg = "[runs] must name one outcomes run per screen"
        raise ValueError(msg)
    radius = load_context(config, screens[0]).setup.radius

    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
    fixed: list[list[dict[str, Any]]] = []
    with ProcessPoolExecutor(
        args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(config,)
    ) as pool:
        for n, screen in enumerate(screens):
            by_esn: dict[Key, list[dict[str, str]]] = defaultdict(list)
            for p in read_csv(resolve_run_path(screen) / "fixed_points.csv"):
                by_esn[esn_key(p)].append(p)
            jobs = [(screen, k, points) for k, points in sorted(by_esn.items())]
            print(f"Screen {n}, {screen}: {sum(len(p) for p in by_esn.values())} fixed points of {len(jobs)} ESNs")
            rows: list[dict[str, Any]] = []
            for k, result in enumerate(pool.map(esn_job, jobs, chunksize=8)):
                rows += result
                if (k + 1) % 500 == 0 or k + 1 == len(jobs):
                    print(f"  [{k + 1:4d}/{len(jobs)}]", flush=True)
            write_csv(run_dir / f"fixed_points_{n}.csv", rows)
            fixed.append(rows)
            print_checks(rows)

    scores: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for n, (rows, outcome_run) in enumerate(zip(fixed, outcomes, strict=True)):
        esns = flags(rows, read_csv(resolve_run_path(outcome_run) / "esns.csv"), radius)
        write_csv(run_dir / f"esns_{n}.csv", esns)
        scores += rule_scores(esns, n)
        changes += stability_changes(rows, n)
    releases = with_growth(read_csv(resolve_run_path(config["runs"]["release"]) / "release.csv"), fixed[0])
    write_csv(run_dir / "release.csv", releases)
    scores += release_scores(releases, config["release"]["fast_from"])
    write_csv(run_dir / "changes.csv", changes)
    write_csv(run_dir / "scores.csv", scores)
    summarize(scores, changes)
    print(f"\nWrote the results to {run_dir}")


def load_context(config: dict[str, Any], screen: str) -> Context:
    """The take as ``screen`` loaded it, its ESNs' table, and the tracker."""
    screen_config = read_toml(resolve_run_path(screen) / "config.toml")
    setup = load_setup(
        {
            "demonstrations": screen_config["demonstrations"],
            "esn": {"dt": screen_config["esn"]["dt"]},
            "evaluation": {"duration": 0.0, "hold": 0.0, "start_offsets_deg": [[0.0]]},
        }
    )
    with (resolve_run_path(screen_config["demonstrations"]["run"]) / "config.toml").open("rb") as f:
        simulator = tomllib.load(f)["simulator"]
    settings, table = tracker_settings(config["tracker"]), config["tracker"]
    if len(settings) != 1:
        msg = "[tracker] must give one setting"
        raise ValueError(msg)
    setting = settings[0]
    tracker = TrackerConfig(
        setting.law,
        setting.omega,
        table["acceleration_filter"],
        damping=setting.damping,
        reference_velocity=table.get("reference_velocity", True),
        control_period=table.get("control_period"),
        mass_matrix=table.get("mass_matrix", False),
    )
    gains = setting.gains(tracker, setup.skeleton, gain_posture(setup.demos.values()))
    take = next(iter(setup.demos.values()))
    grids = map_grid(setup.skeleton, screen_config["screen"]["grid_step_deg"])
    return Context(screen_config["esn"], setup, take, simulator, tracker, gains, grids)


def init_worker(config: dict[str, Any]) -> None:
    for screen in config["runs"]["screens"]:
        _contexts[screen] = load_context(config, screen)


def esn_job(job: tuple[str, Key, list[dict[str, str]]]) -> list[dict[str, Any]]:
    """One ESN's fixed points in this worker."""
    screen, key, points = job
    return esn_fixed_points(_contexts[screen], key, points)


def esn_fixed_points(context: Context, key: Key, points: list[dict[str, str]]) -> list[dict[str, Any]]:
    """The ESN ``key``, trained as the screen trained it: each of its ``points``, on its own and with the arm."""
    combination: dict[str, Any] = dict(zip(SWEPT, key, strict=True))
    esn = ReachingEsn(EsnConfig(**(context.esn | combination)))
    esn.fit([context.take])
    loop = ClosedLoop.from_esn(esn)
    robot = RobotLoop.from_tracker(
        loop,
        context.setup.skeleton,
        context.tracker,
        context.gains,
        context.simulator["dt"],
        enforce_limits=context.simulator.get("enforce_limits", True),
        stops=context.simulator.get("stops", "projected"),
    )
    rows = []
    nan = float("nan")
    for p in points:
        posture = joint_columns("q{j}_deg", joint_values(p, "q{j}_deg"))
        q = np.radians(joint_values(p, "q{j}_deg"))
        x, found = screen_state(loop, q, float(p["growth_per_s"]), context.grids_deg)
        row = combination | {"fixed_point": int(p["fixed_point"])} | posture | {c: float(p[c]) for c in KEPT}
        within = bool(np.all(q >= robot.lower) and np.all(q <= robot.upper)) if robot.lower is not None else True
        row |= {"from_grid": p["from_grid"] == "True", "found": found, "within_stops": within}
        if not found or not within:
            rows.append(row | {"residual": nan, "system_n_unstable": nan, "system_growth_per_s": nan})
            continue
        z = robot.at_rest(x)
        system = np.abs(np.linalg.eigvals(robot.jacobian(z)))
        rows.append(
            row
            | {
                "residual": float(np.abs(robot.advance(z) - z).max()),
                "system_n_unstable": int(np.count_nonzero(system > 1.0)),
                "system_growth_per_s": float(np.log(system.max()) / loop.dt),
            }
        )
    return rows


def screen_state(
    loop: ClosedLoop, q: NDArray[np.float64], growth: float, grids_deg: list[NDArray[np.float64]]
) -> tuple[NDArray[np.float64], str]:
    """The state of the screen's fixed point at the posture ``q`` (rad) with the slowest mode's ``growth`` rate.

    ``grids_deg`` is the screen's grid of R, each joint's angles. Returns the state and
    how it was found (see the module), or the first state tried and an empty string if
    none gives the screen's posture and growth rate.
    """

    def matches(x: NDArray[np.float64], converged: bool) -> bool:
        if not converged or np.degrees(np.abs(loop.posture(x) - q).max()) > POSTURE_TOLERANCE_DEG:
            return False
        own = np.log(np.abs(np.linalg.eigvals(loop.jacobian(x))).max()) / loop.dt
        return bool(abs(own - growth) <= GROWTH_TOLERANCE)

    held, _ = held_fixed_points(loop, q[np.newaxis])
    first, converged = loop.fixed_point(held[0])
    if matches(first, converged):
        return first, "reset"
    # As the screen found it: Newton's method on R from the cells of its grid around q, where R changes sign.
    local = []
    for grid, angle in zip(grids_deg, q, strict=True):
        i = int(np.clip(np.searchsorted(np.radians(grid), angle) - 1, 0, len(grid) - 2))
        local.append(grid[max(i - 1, 0) : min(i + 3, len(grid))])
    corners_q = np.array(list(itertools.product(*(np.radians(g) for g in local))))
    corners_x, converged_x = held_fixed_points(loop, corners_q)
    shape = tuple(len(g) for g in local)
    residual = (loop.posture(corners_x) - corners_q).reshape(*shape, -1)
    cells = PostureMap(tuple(local), corners_x.reshape(*shape, -1), residual, converged_x.reshape(shape))
    for x in cell_zeros(loop, cells):
        if x is not None:
            x, converged = loop.fixed_point(x)
            if matches(x, converged):
                return x, "cells"
    rng = np.random.default_rng(0)
    for _ in range(RANDOM_STARTS):
        held, _ = held_fixed_points(loop, q[np.newaxis], rng.uniform(-1.0, 1.0, (1, loop.n_neurons)))
        x, converged = loop.fixed_point(held[0])
        if matches(x, converged):
            return x, "random"
    return first, ""


def flags(fixed: list[dict[str, Any]], outcomes: list[dict[str, str]], radius: float) -> list[dict[str, Any]]:
    """Each ESN's outcome on the robot, and rule (a)'s flag on its own and with the arm."""
    by_esn: dict[Key, list[dict[str, Any]]] = defaultdict(list)
    for p in fixed:
        by_esn[esn_key(p)].append(p)
    rows = []
    for r in outcomes:
        points = by_esn.get(esn_key(r), [])
        off = [p for p in points if p["target_distance_m"] > radius]
        own = any(p["n_unstable"] == 0 for p in off)
        # A fixed point not found again is taken as on its own; one beyond the stops is none of the system's.
        with_arm = any(
            (p["system_n_unstable"] if p["found"] else p["n_unstable"]) == 0 for p in off if p["within_stops"]
        )
        if own != (r["flag_a"] == "True"):
            msg = f"rule (a) on its own differs from the outcomes' run for {esn_key(r)}"
            raise RuntimeError(msg)
        rows.append(
            {c: float(r[c]) for c in SWEPT[:3]}
            | {"seed": int(float(r["seed"])), "passes": r["passes"] == "True"}
            | {"flag_a": own, "flag_a_arm": with_arm}
        )
    return rows


def rule_scores(esns: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Each flag's area under the ROC curve for failing, over all the ESNs and within each leak rate."""
    lr = np.array([r["leak_rate"] for r in esns])
    failing = ~np.array([r["passes"] for r in esns])
    scores = []
    for flag in ("flag_a", "flag_a_arm"):
        flagged = np.array([r[flag] for r in esns], dtype=float)
        for name, group in [("all", np.ones(len(esns), dtype=bool))] + [(f"{v:g}", lr == v) for v in np.unique(lr)]:
            value, count = auc(flagged[group], failing[group])
            scores.append(
                {"screen": n, "measure": flag, "against": "failing", "leak_rate": name, "statistic": "auc"}
                | {"value": value, "n": count, "flagged": int(flagged[group].sum())}
            )
    return scores


def stability_changes(fixed: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Within each leak rate, the fixed points whose stability the arm changes, and the growth rate with it."""
    rows = []
    for lr in sorted({p["leak_rate"] for p in fixed}):
        points = [p for p in fixed if p["leak_rate"] == lr and p["found"] and p["within_stops"]]
        own = np.array([p["growth_per_s"] for p in points])
        system = np.array([p["system_growth_per_s"] for p in points])
        own_unstable = np.array([p["n_unstable"] > 0 for p in points])
        system_unstable = np.array([p["system_n_unstable"] > 0 for p in points])
        rows.append(
            {"screen": n, "leak_rate": lr, "fixed_points": len(points)}
            | {"not_found": sum(p["leak_rate"] == lr and not p["found"] for p in fixed)}
            | {"beyond_stops": sum(p["leak_rate"] == lr and not p["within_stops"] for p in fixed)}
            | {
                "unstable": int(own_unstable.sum()),
                "unstable_to_attracting": int((own_unstable & ~system_unstable).sum()),
                "attracting_to_unstable": int((~own_unstable & system_unstable).sum()),
                "growth_ratio_median": float(np.median(np.abs(system) / np.abs(own))),
                "growth_ratio_p10": float(np.quantile(np.abs(system) / np.abs(own), 0.1)),
                "growth_ratio_p90": float(np.quantile(np.abs(system) / np.abs(own), 0.9)),
            }
        )
    return rows


def with_growth(releases: list[dict[str, str]], fixed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The release rows, each with the growth rate and unstable modes with the arm at its fixed point."""
    at = {(esn_key(p), tuple(joint_values(p, "q{j}_deg"))): p for p in fixed}
    rows = []
    for r in releases:
        p = at[(esn_key(r), tuple(joint_values(r, "q{j}_deg")))]
        rows.append(
            {c: (int(float(r[c])) if c == "seed" else float(r[c])) for c in SWEPT}
            | joint_columns("q{j}_deg", joint_values(r, "q{j}_deg"))
            | {c: float(r[c]) for c in ("path_distance_deg", "growth_per_s", "stays_s")}
            | {c: float(r[c]) for c in ("arrival_s", "own_arrival_s")}
            | {"system_growth_per_s": p["system_growth_per_s"], "system_n_unstable": p["system_n_unstable"]}
        )
    return rows


def release_scores(releases: list[dict[str, Any]], fast_from: float) -> list[dict[str, Any]]:
    """How long the arm stays against the growth rate, on its own and with the arm, by leak rate group."""
    lr = np.array([r["leak_rate"] for r in releases])
    stays = np.array([r["stays_s"] for r in releases])
    stays_put = np.isinf(stays)
    found = np.isfinite([r["system_growth_per_s"] for r in releases])  # both growth rates over the same releases
    groups = [("all", found), (f"<{fast_from:g}", found & (lr < fast_from))]
    groups.append((f">={fast_from:g}", found & (lr >= fast_from)))
    scores = []
    for measure, column in (("own growth", "growth_per_s"), ("growth with the arm", "system_growth_per_s")):
        growth = np.array([r[column] for r in releases])
        for name, group in groups:
            scores.append(
                {"screen": 0, "measure": measure, "against": "stay", "leak_rate": name, "statistic": "spearman"}
                | dict(zip(("value", "n"), spearman(growth, stays, group), strict=True))
                | {"flagged": ""}
            )
            value, count = auc(-growth[group], stays_put[group])
            scores.append(
                {"screen": 0, "measure": measure, "against": "stays put", "leak_rate": name, "statistic": "auc"}
                | {"value": value, "n": count, "flagged": int(stays_put[group].sum())}
            )
    attracting = np.array([r["system_n_unstable"] == 0 for r in releases])
    for name, group in groups:
        scores.append(
            {"screen": 0, "measure": "attracting with the arm", "against": "stays put", "leak_rate": name}
            | {"statistic": "count", "value": int((attracting & stays_put & group).sum()), "n": int(group.sum())}
            | {"flagged": int((attracting & group).sum())}
        )
    return scores


def print_checks(rows: list[dict[str, Any]]) -> None:
    """Print how the screen's fixed points were found again, and the system's residual there."""
    ways = {way: sum(r["found"] == way for r in rows) for way in ("reset", "cells", "random", "")}
    residual = max(r["residual"] for r in rows if r["found"] and r["within_stops"])
    beyond = sum(not r["within_stops"] for r in rows)
    print(
        f"  found from the reset {ways['reset']}, from the cells {ways['cells']}, from random states"
        f" {ways['random']}, not found {ways['']}; {beyond} beyond the arm's stops; the system's residual at"
        f" most {residual:.1g} at the others"
    )


def summarize(scores: list[dict[str, Any]], changes: list[dict[str, Any]]) -> None:
    """Print the changes by leak rate and the scores."""
    print("\nThe arm's changes, by leak rate (screen, leak rate: fixed points, unstable, changed, growth ratio):")
    for c in changes:
        print(
            f"  {c['screen']}, {c['leak_rate']:g}: {c['fixed_points']}, {c['unstable']} unstable;"
            f" {c['unstable_to_attracting']} unstable to attracting, {c['attracting_to_unstable']} attracting to"
            f" unstable; growth with the arm / on its own {c['growth_ratio_median']:.2f}"
            f" ({c['growth_ratio_p10']:.2f} to {c['growth_ratio_p90']:.2f})"
        )
    print("\nScores:")
    for s in scores:
        if s["leak_rate"] in ("all",) or s["against"] != "failing":
            print(
                f"  screen {s['screen']}, {s['measure']} against {s['against']}, {s['leak_rate']}:"
                f" {s['statistic']} {s['value']:.3f} over {s['n']} (flagged {s['flagged']})"
            )


def esn_key(row: dict[str, Any]) -> Key:
    return (
        float(row["leak_rate"]),
        float(row["spectral_radius"]),
        float(row["input_scaling"]),
        int(float(row["seed"])),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` to the CSV file ``path``, one column per key."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
