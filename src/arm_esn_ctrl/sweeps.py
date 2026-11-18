# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Summaries of ESN sweeps that try several random reservoirs (seeds) for each setting.

A seed draws an ESN's random reservoir, input weights, and biases, and the other
settings only rescale them: the spectral radius scales the reservoir, and the
input scaling the input weights. So the same seed gives the same random draws in
every setting, and settings compare seed by seed. Over seeds, a setting's spread
shows how much its result depends on the draw.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np


def sweep_combinations(
    sweep: dict[str, list[Any]], settings: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """The combinations of hyperparameters a sweep tries, in its order.

    ``sweep`` lists values for some hyperparameters: every combination of them is
    tried, the last one varying fastest. With ``settings``, a list of chosen
    settings that each give the same other hyperparameters, each of them is tried
    with every combination of ``sweep``, such as with every seed.
    """
    products = [dict(zip(sweep, values, strict=True)) for values in itertools.product(*sweep.values())]
    if settings is None:
        return products
    names = set(settings[0])
    if any(set(setting) != names for setting in settings):
        msg = "[[settings]] must all give the same hyperparameters"
        raise ValueError(msg)
    if names & set(sweep):
        msg = f"[[settings]] and [sweep] both give {sorted(names & set(sweep))}"
        raise ValueError(msg)
    return [setting | product for setting in settings for product in products]


def summarize_seeds(rows: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    """Summarize each setting of a sweep over its seeds.

    ``rows`` are the rows of a sweep's ``sweep.csv`` with ``seed`` among its swept
    keys: each holds its swept values, its failed runs (``failures``), its worst
    path RMSE (``worst_path_rmse_m``), its latest arrival (``latest_arrival_s``,
    NaN if a run never arrives), and its largest distance from the target at the
    end of a run (``worst_final_distance_m``). A setting is a combination of the
    values of ``names``, the swept keys other than ``seed``.

    Returns one row per setting, in the order of their first rows: its values,
    then ``seeds`` (how many), ``failing_seeds`` (the seeds with a failed run),
    the median, mean, standard deviation, smallest, and largest worst path RMSE
    over the seeds, the median latest arrival over the seeds that arrive in every
    run (NaN if none), and the largest final distance.

    Rows of a sweep with disturbances also hold the runs whose hand is not back on
    the path after one (``unreturned``), the latest return (``latest_return_s``,
    NaN if a run is not back), and the farthest from the path after one
    (``worst_off_path_m``). Their settings then also give ``unreturned_seeds`` (the
    seeds with a run not back), the median latest return over the seeds whose runs
    are all back (NaN if none), and the median farthest from the path.
    """
    settings: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        settings.setdefault(tuple(row[name] for name in names), []).append(row)
    summary = []
    for values, group in settings.items():
        rmse = np.array([row["worst_path_rmse_m"] for row in group], dtype=np.float64)
        setting = dict(zip(names, values, strict=True)) | {
            "seeds": len(group),
            "failing_seeds": sum(row["failures"] > 0 for row in group),
            "median_worst_path_rmse_m": float(np.median(rmse)),
            "mean_worst_path_rmse_m": float(np.mean(rmse)),
            "sd_worst_path_rmse_m": float(np.std(rmse, ddof=1)) if len(group) > 1 else 0.0,
            "min_worst_path_rmse_m": float(rmse.min()),
            "max_worst_path_rmse_m": float(rmse.max()),
            "median_latest_arrival_s": _finite_median([row["latest_arrival_s"] for row in group]),
            "worst_final_distance_m": max(float(row["worst_final_distance_m"]) for row in group),
        }
        if "unreturned" in group[0]:
            setting |= {
                "unreturned_seeds": sum(row["unreturned"] > 0 for row in group),
                "median_latest_return_s": _finite_median([row["latest_return_s"] for row in group]),
                "median_worst_off_path_m": float(np.median([row["worst_off_path_m"] for row in group])),
            }
        summary.append(setting)
    return summary


def _finite_median(values: list[float]) -> float:
    """The median of the finite ``values`` (NaN if none is)."""
    array = np.array(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float("nan")


def score_settings(runs: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    """Score each setting of a sweep by how many of its runs arrive and hold, then by how directly they arrive.

    ``runs`` are the rows of a sweep's ``runs.csv``: each holds its swept values,
    whether it failed (``failed``: it never arrives, or does not hold), and its
    detour ratio (``detour_ratio``, NaN for a run that never arrives; see
    :func:`arm_esn_ctrl.metrics.directness`). A setting is a combination of the
    values of ``names``, the swept keys other than ``seed``; its runs are those of
    all its seeds.

    Returns one row per setting, the best first: its values, then ``runs`` (how
    many), ``holding_runs`` (those that arrive and hold), and
    ``median_detour_ratio`` (over its runs that arrive; NaN if none does). The
    settings with the most holding runs come first, and among equals, the smallest
    median detour ratio; a NaN comes last, and settings equal in both keep the
    sweep's order.

    With ``seed`` in the runs, each row also holds ``seeds`` (how many) and
    ``typical_seed``: of the setting's seeds with the most holding runs, the one
    whose own median detour ratio is nearest the setting's (the first, if none of
    them arrives).
    """
    settings: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for run in runs:
        settings.setdefault(tuple(run[name] for name in names), []).append(run)
    scores = []
    for values, group in settings.items():
        median = _finite_median([run["detour_ratio"] for run in group])
        score = dict(zip(names, values, strict=True)) | {
            "runs": len(group),
            "holding_runs": sum(not run["failed"] for run in group),
            "median_detour_ratio": median,
        }
        if "seed" in group[0]:
            seeds: dict[Any, list[dict[str, Any]]] = {}
            for run in group:
                seeds.setdefault(run["seed"], []).append(run)
            holding = {seed: sum(not run["failed"] for run in each) for seed, each in seeds.items()}
            gaps = {
                seed: abs(_finite_median([run["detour_ratio"] for run in each]) - median)
                for seed, each in seeds.items()
                if holding[seed] == max(holding.values())
            }
            typical = min(gaps, key=lambda seed: _nan_last(gaps[seed]))
            score |= {"seeds": len(seeds), "typical_seed": typical}
        scores.append(score)
    return sorted(scores, key=lambda score: (-score["holding_runs"], _nan_last(score["median_detour_ratio"])))


def score_cycles(runs: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    """Score each setting of a sweep of a periodic task by how many of its runs cycle, then by their curve RMS.

    ``runs`` are the rows of a sweep's ``runs.csv``: each holds its swept values,
    whether it cycles (``cycles``), and its curve RMS (``curve_rms_m``; see
    :func:`arm_esn_ctrl.metrics.cycle_metrics`). A setting is a combination of the
    values of ``names``, the swept keys other than ``seed``; its runs are those of
    all its seeds.

    Returns one row per setting, the best first: its values, then ``runs`` (how
    many), ``cycling_runs`` (those that cycle), and ``median_curve_rms_m`` (over its
    runs that cycle; NaN if none does). The settings with the most cycling runs come
    first, and among equals, the smallest median curve RMS; a NaN comes last, and
    settings equal in both keep the sweep's order.

    With ``seed`` in the runs, each row also holds ``seeds`` (how many) and
    ``typical_seed``: of the setting's seeds with the most cycling runs, the one
    whose own median curve RMS, over its runs that cycle, is nearest the setting's
    (the first, if none of them cycles), as :func:`score_settings` chooses one.
    """

    def median_rms(group: list[dict[str, Any]]) -> float:
        return _finite_median([run["curve_rms_m"] for run in group if run["cycles"]])

    settings: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for run in runs:
        settings.setdefault(tuple(run[name] for name in names), []).append(run)
    scores = []
    for values, group in settings.items():
        median = median_rms(group)
        score = dict(zip(names, values, strict=True)) | {
            "runs": len(group),
            "cycling_runs": sum(bool(run["cycles"]) for run in group),
            "median_curve_rms_m": median,
        }
        if "seed" in group[0]:
            seeds: dict[Any, list[dict[str, Any]]] = {}
            for run in group:
                seeds.setdefault(run["seed"], []).append(run)
            cycling = {seed: sum(bool(run["cycles"]) for run in each) for seed, each in seeds.items()}
            gaps = {
                seed: abs(median_rms(each) - median)
                for seed, each in seeds.items()
                if cycling[seed] == max(cycling.values())
            }
            typical = min(gaps, key=lambda seed: _nan_last(gaps[seed]))
            score |= {"seeds": len(seeds), "typical_seed": typical}
        scores.append(score)
    return sorted(scores, key=lambda score: (-score["cycling_runs"], _nan_last(score["median_curve_rms_m"])))


def _nan_last(value: float) -> float:
    """``value``, or infinity for a NaN, so that it sorts last."""
    return value if np.isfinite(value) else float("inf")
