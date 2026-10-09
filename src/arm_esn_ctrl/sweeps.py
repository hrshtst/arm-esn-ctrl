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

from typing import Any

import numpy as np


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
