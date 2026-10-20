# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Set the screen of ``experiments/screen_esns.py`` against the robot's outcomes of the same ESNs.

    uv run python experiments/screen_against_robot.py \\
        experiments/manual_demonstration_v2_esn_screening/against_robot.toml

The configuration names a screen run and the robot sweeps of
``experiments/sweep_robot_esn.py`` that ran the same ESNs (``[runs]``), one run per
leak rate and scenario: undisturbed, under a block (one ``[disturbance]``), or under
pushes (several). Each ESN's outcome is report 011's rule: undisturbed and under the
block, it passes if its hand arrives, holds for the whole hold window, and ends within
the goal; under the pushes, if it arrives and ends within it. The rules
(``[rules]``) flag ESNs from the screen alone:

- **(a)** an attracting fixed point whose hand is outside the goal radius, against
  ending away from the target under the pushes; and, apart from the rules, only
  those more than ``far_m`` off;
- **(b)** a median lead near the path under ``lead_deg``, against the same, and the
  lead's area under the ROC curve for it;
- **(c)** an unstable fixed point within ``path_deg`` of the take's joint path, or a
  release arrival later than ``release_s`` (or none), against never arriving in some
  scenario, and the release arrival's rank correlation with the robot's arrival
  time undisturbed and under the block;
- **(d)** the rank correlations of the rates along the ESN's run with the return
  time (the mean over the pushes, of the ESNs whose arm returns after each) and the
  following (the mean following gain over the block and the pushes).

Each count and score is taken over all the ESNs and within each leak rate; the
counts also over the ESNs of the qualifying settings (``[qualifying]``), and the
flags of the named ESNs (``[named]``) are listed. The run directory receives:

- ``esns.csv``: one row per ESN, its outcomes on the robot, its flags, and the
  screen's measures they come from;
- ``rules.csv``: per rule, over all the ESNs and per leak rate, the ESNs flagged,
  the failing ones and those of the rule's own outcome caught, the passing and the
  qualifying ones rejected, and the share failing among the flagged and the
  unflagged;
- ``scores.csv``: the areas under the ROC curve and the rank correlations, with how
  many ESNs each is over.
"""

from __future__ import annotations

import argparse
import csv
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from arm_esn_ctrl.autonomous import load_setup
from arm_esn_ctrl.storage import resolve_run_path, start_run

Key = tuple[float, float, float, int]  # leak rate, spectral radius, input scaling, seed
SETTING = "pd_w20_z0.4"  # the sweeps' one tracker setting
HOLD_S = 2.0  # the sweeps' hold window after arrival (s)
RATES = ("across_rate_median_per_s", "across_rate_max_per_s", "posture_scale_median", "posture_scale_max")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="configuration file (TOML)")
    args = parser.parse_args()
    config, run_dir = start_run(args.config)
    rules = config["rules"]

    screen_dir = resolve_run_path(config["runs"]["screen"])
    screen_config = read_toml(screen_dir / "config.toml")
    setup = load_setup(
        {
            "demonstrations": screen_config["demonstrations"],
            "esn": {"dt": screen_config["esn"]["dt"]},
            "evaluation": {"duration": 0.0, "hold": 0.0, "start_offsets_deg": [[0.0, 0.0]]},
        }
    )
    radius = setup.radius
    screen = {esn_key(r): r for r in read_csv(screen_dir / "screen.csv")}
    fixed: dict[Key, list[dict[str, str]]] = {k: [] for k in screen}
    for p in read_csv(screen_dir / "fixed_points.csv"):
        fixed[esn_key(p)].append(p)

    robot = robot_runs(config["runs"]["robot"])
    if set(robot) != set(screen):
        msg = "the robot sweeps and the screen must hold the same ESNs"
        raise ValueError(msg)
    keys = sorted(screen)
    qualifying = {tuple(s) for s in config["qualifying"]["settings"]}
    named = {tuple(v): name for name, v in config["named"].items()}

    rows = []
    for k in keys:
        s, outcome = screen[k], outcomes(robot[k], radius)
        attracting_off = [p for p in fixed[k] if p["n_unstable"] == "0" and num(p, "target_distance_m") > radius]
        rows.append(
            dict(zip(("leak_rate", "spectral_radius", "input_scaling", "seed"), k, strict=True))
            | {"qualifying": k[:3] in qualifying, "named": named.get(k, "")}
            | outcome
            | {
                "flag_a": bool(attracting_off),
                "flag_a_far": any(num(p, "target_distance_m") > rules["far_m"] for p in attracting_off),
                "flag_b": num(s, "lead_median_deg") < rules["lead_deg"],
                "flag_c": num(s, "nearest_unstable_path_distance_deg") <= rules["path_deg"]
                or (int(s["unstable"]) > 0 and not num(s, "release_arrival_s") <= rules["release_s"]),
            }
            | {
                c: s[c]
                for c in (
                    "attracting_off_target",
                    "lead_median_deg",
                    "unstable",
                    "nearest_unstable_path_distance_deg",
                    "nearest_unstable_growth_per_s",
                    "release_arrival_s",
                    *RATES,
                )
            }
        )
    write_csv(run_dir / "esns.csv", rows)

    table = {c: np.array([row[c] for row in rows]) for c in rows[0]}
    lr = table["leak_rate"].astype(float)
    groups = [("all", np.ones(len(keys), dtype=bool))] + [(f"{v:g}", lr == v) for v in np.unique(lr)]
    failing = ~table["passes"].astype(bool)
    measure = {c: table[c].astype(float) for c in ("lead_median_deg", "release_arrival_s", *RATES)}

    rule_rows = []
    for rule, flag, target in (
        ("a", "flag_a", "ends_away"),
        ("a_far", "flag_a_far", "ends_away"),
        ("b", "flag_b", "ends_away"),
        ("c", "flag_c", "never_arrives"),
    ):
        flagged, own = table[flag].astype(bool), table[target].astype(bool)
        for name, group in groups:
            rule_rows.append(counts(rule, target, name, group, flagged, failing, own, table["qualifying"].astype(bool)))
    write_csv(run_dir / "rules.csv", rule_rows)

    scores = []
    ends_away, never = table["ends_away"].astype(bool), table["never_arrives"].astype(bool)
    has_unstable = table["unstable"].astype(int) > 0
    nearest = table["nearest_unstable_path_distance_deg"].astype(float)
    growth = table["nearest_unstable_growth_per_s"].astype(float)
    release = measure["release_arrival_s"]  # NaN without an unstable fixed point, infinite if it never arrives
    for name, group in groups:
        if name == "all":
            scores.append(score("leak rate", "failing", name, "auc", auc(lr, failing)))
            scores.append(score("leak rate", "following", name, "spearman", *spearman(lr, table["following"], group)))
        scores.append(
            score("smaller lead", "ends_away", name, "auc", auc(-measure["lead_median_deg"][group], ends_away[group]))
        )
        for arrival in ("arrival_nominal_s", "arrival_block_s"):
            scores.append(
                score(
                    "release arrival",
                    arrival,
                    name,
                    "spearman",
                    *spearman(release, table[arrival], group & has_unstable),
                )
            )
        for rate in RATES:
            for against in ("return_s", "following"):
                scores.append(score(rate, against, name, "spearman", *spearman(measure[rate], table[against], group)))
        u = group & has_unstable
        for against, positive in (("failing", failing), ("never_arrives", never)):
            scores.append(score("nearer unstable fixed point", against, name, "auc", auc(-nearest[u], positive[u])))
            scores.append(score("slower growth there", against, name, "auc", auc(-growth[u], positive[u])))
            scores.append(
                score(
                    "later release arrival",
                    against,
                    name,
                    "auc",
                    auc(np.nan_to_num(release[u], posinf=1e9), positive[u]),
                )
            )
    scores.append(
        score(
            "posture_scale_max",
            "following",
            "within each leak rate",
            "spearman",
            *pooled_within(measure["posture_scale_max"], table["following"].astype(float), lr),
        )
    )
    write_csv(run_dir / "scores.csv", scores)

    summarize(rule_rows, scores, rows)
    print(f"\nWrote the results to {run_dir}")


def robot_runs(references: list[str]) -> dict[Key, dict[str, dict[str, str]]]:
    """Each ESN's row of ``runs.csv`` in each scenario, from the sweeps' run directories."""
    robot: dict[Key, dict[str, dict[str, str]]] = {}
    for reference in references:
        run = resolve_run_path(reference)
        config = read_toml(run / "config.toml")
        disturbance = config.get("disturbance")
        scenario = "nominal" if disturbance is None else "block" if isinstance(disturbance, dict) else "pushes"
        leak_rate = float(config["esn"]["leak_rate"])
        for r in read_csv(run / "runs.csv"):
            if r["setting"] != SETTING or r["start"] != "0":
                msg = f"{run.name}: expected one run per ESN, of {SETTING} from the demonstrated start"
                raise ValueError(msg)
            scenarios = robot.setdefault(
                (leak_rate, num(r, "spectral_radius"), num(r, "input_scaling"), int(num(r, "seed"))), {}
            )
            if scenario in scenarios:
                msg = f"{run.name}: a second {scenario} run of an ESN"
                raise ValueError(msg)
            scenarios[scenario] = r
    return robot


def outcomes(runs: dict[str, dict[str, str]], radius: float) -> dict[str, Any]:
    """Report 011's rule for one ESN, and what the rules are set against."""
    nominal, block, pushes = runs["nominal"], runs["block"], runs["pushes"]

    def holds(r: dict[str, str]) -> bool:
        # success: arrived without leaving the goal after; the rule also asks for the whole hold and the end within it
        hold = float(r["hold_observed_s"]) if r["hold_observed_s"] != "nan" else 0.0
        return yes(r["arrived"]) and yes(r["success"]) and hold >= HOLD_S and num(r, "final_distance_m") <= radius

    returns = [num(pushes, f"return_time_s_{i}") for i in (1, 2, 3)]
    gains = [num(block, "following_gain")] + [num(pushes, f"following_gain_{i}") for i in (1, 2, 3)]
    return {
        "passes": holds(nominal)
        and holds(block)
        and yes(pushes["arrived"])
        and num(pushes, "final_distance_m") <= radius,
        "ends_away": num(pushes, "final_distance_m") > radius,
        "never_arrives": not all(yes(r["arrived"]) for r in runs.values()),
        "arrival_nominal_s": num(nominal, "arrival_time_s") if yes(nominal["arrived"]) else np.inf,
        "arrival_block_s": num(block, "arrival_time_s") if yes(block["arrived"]) else np.inf,
        "return_s": float(np.mean(returns)) if np.all(np.isfinite(returns)) else np.nan,
        "following": float(np.mean(gains)),
    }


def counts(
    rule: str,
    target: str,
    group: str,
    members: NDArray[np.bool_],
    flagged: NDArray[np.bool_],
    failing: NDArray[np.bool_],
    own: NDArray[np.bool_],
    qualifying: NDArray[np.bool_],
) -> dict[str, Any]:
    """One row of ``rules.csv``: a rule's counts over the ESNs of ``members``."""
    f, fl, o, q = (a[members] for a in (flagged, failing, own, qualifying))
    return {
        "rule": rule,
        "outcome": target,
        "leak_rate": group,
        "esns": int(members.sum()),
        "flagged": int(f.sum()),
        "failing": int(fl.sum()),
        "failing_caught": int((f & fl).sum()),
        "outcome_esns": int(o.sum()),
        "outcome_caught": int((f & o).sum()),
        "passing": int((~fl).sum()),
        "passing_rejected": int((f & ~fl).sum()),
        "qualifying": int(q.sum()),
        "qualifying_rejected": int((f & q).sum()),
        "failing_share_flagged": float(fl[f].mean()) if f.any() else np.nan,
        "failing_share_unflagged": float(fl[~f].mean()) if (~f).any() else np.nan,
    }


def auc(score_: NDArray[np.float64], positive: NDArray[np.bool_]) -> tuple[float, int]:
    """The chance that a positive's score exceeds a negative's, ties counted half, and how many it is over."""
    pos, neg = score_[positive], score_[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, len(score_)
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), len(score_)


def spearman(x: NDArray[Any], y: NDArray[Any], members: NDArray[np.bool_]) -> tuple[float, int]:
    """The rank correlation of ``x`` and ``y`` over ``members``, without NaN; an infinity ranks last."""
    x, y = x.astype(float), y.astype(float)
    ok = members & ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 5:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1]), int(ok.sum())  # Spearman's


def pooled_within(x: NDArray[np.float64], y: NDArray[np.float64], group: NDArray[np.float64]) -> tuple[float, int]:
    """The correlation of ``x`` and ``y``'s ranks within each value of ``group``, pooled."""
    rx, ry = np.empty(len(x)), np.empty(len(y))
    for value in np.unique(group):
        members = group == value
        rx[members], ry[members] = rankdata(x[members]) / members.sum(), rankdata(y[members]) / members.sum()
    return float(np.corrcoef(rx, ry)[0, 1]), len(x)


def score(
    measure: str, against: str, group: str, statistic: str, value: tuple[float, int] | float, n: int | None = None
) -> dict[str, Any]:
    """One row of ``scores.csv``."""
    if isinstance(value, tuple):
        value, n = value
    return {"measure": measure, "against": against, "leak_rate": group, "statistic": statistic, "value": value, "n": n}


def summarize(rule_rows: list[dict[str, Any]], scores: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Print the counts and the scores."""
    print("rule, leak rate: flagged; failing caught; its outcome caught; passing rejected; qualifying rejected;")
    print("  the share failing among the flagged / the unflagged")
    for r in rule_rows:
        print(
            f"{r['rule']}, {r['leak_rate']}: {r['flagged']}; {r['failing_caught']}/{r['failing']};"
            f" {r['outcome_caught']}/{r['outcome_esns']} {r['outcome']}; {r['passing_rejected']}/{r['passing']};"
            f" {r['qualifying_rejected']}/{r['qualifying']};"
            f" {r['failing_share_flagged']:.2f} / {r['failing_share_unflagged']:.2f}"
        )
    print("\nmeasure | against | leak rate | statistic: value (ESNs)")
    for s in scores:
        print(f"{s['measure']} | {s['against']} | {s['leak_rate']} | {s['statistic']}: {s['value']:+.3f} ({s['n']})")
    print("\nnamed ESNs:")
    for row in rows:
        if row["named"]:
            flags = ", ".join(c[5:] for c in ("flag_a", "flag_a_far", "flag_b", "flag_c") if row[c]) or "none"
            print(f"  {row['named']}: passes {row['passes']}; flagged by {flags}")


def esn_key(row: dict[str, str]) -> Key:
    return (num(row, "leak_rate"), num(row, "spectral_radius"), num(row, "input_scaling"), int(num(row, "seed")))


def num(row: dict[str, str], column: str) -> float:
    return float(row[column])


def yes(value: str) -> bool:
    return value == "True"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
