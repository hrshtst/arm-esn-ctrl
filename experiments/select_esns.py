# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Choose the ESNs of a robot search by its rule, from the runs of its sweeps (Stage 2).

    uv run python experiments/select_esns.py experiments/manual_demonstration_v2_consistent_search/select.toml

The configuration's ``[runs]`` names runs of ``experiments/sweep_robot_esn.py``,
relative to the storage root, that ran the same ESNs with the same tracker, each a
list, such as one run per leak rate: undisturbed from the demonstrated start
(``nominal``), under one block (``block``), under pushes (``pushes``), and
undisturbed from offset starts (``offsets``). ``replay`` names runs of
``experiments/robot_esn.py`` with the same tracker, under the same block and the
same pushes: their replay arm sets the guard. An ESN is its run's ``[esn]`` table
with the values its ``[sweep]`` gave it, and a setting is an ESN but for its seed.
Every run must share the ``[demonstrations]`` and ``[tracker]`` tables and the
measures of ``[evaluation]``, and every ESN must have run undisturbed, under the
block, and under the pushes.

The rule (report 011's, with the offset starts counted for every ESN):

1. **An ESN passes** if its hand arrives within the goal radius, holds there for
   the whole hold window, and ends within it undisturbed, under the block, and
   from every offset start; and under the pushes, if it arrives and ends within
   it: leaving the goal during the hold is excused there, since a push can carry
   the arm into the goal early, still swinging. An ESN not run from the offset
   starts does not pass.
2. **Its measures:** after each disturbance, the block's then each push's, the
   following gain, the ringing, the farthest from the taught path, and the return
   onto it, a hand not back counting as back at the end of the disturbance's window
   (when the next one starts, or the run ends); their means over the disturbances;
   the undisturbed path RMSE; under the block, the reference lead and the peak
   torque; from the offset starts, the median first step and peak torque; and in
   each scenario, how hard the motors work over the task (``robot_esn.arm_metrics``'s
   ``task_*`` columns, their medians over the offset starts).
3. **A setting qualifies** if all its seeds pass. Its measures are their medians
   over its seeds.
4. **The guard:** a setting whose median ringing exceeds the replay's, its mean
   over the disturbances, is not chosen.
5. **The ranking:** the qualifying settings within the guard, by their median
   following, the largest first.
6. **The typical seed** of a setting: of its passing seeds, the one nearest its
   medians of the undisturbed path RMSE, the following, and the ringing, each
   scaled by its interquartile range over the seeds.
7. **The chosen ESNs,** ``chosen`` of them (``[rule]``), ESN #1, ESN #2, ...: the
   typical seeds of the top settings. If fewer settings qualify, the rest are taken
   from the settings with at least ``fallback_seeds`` passing seeds, ranked alike.

The run directory receives:

- ``esns.csv``: one row per ESN: its hyperparameters, whether it passes and how
  its runs fail, and its measures;
- ``settings.csv``: one row per setting: its hyperparameters, its passing seeds,
  whether it qualifies and is within the guard, its rank, its typical seed, and
  its medians of the measures;
- ``chosen.csv``: the chosen ESNs, with their settings' ranks.
"""

from __future__ import annotations

import argparse
import csv
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from robot_esn import per_disturbance

from arm_esn_ctrl.disturbances import disturbance_spans
from arm_esn_ctrl.storage import resolve_run_path, start_run

SCENARIOS = ("nominal", "block", "pushes", "offsets")
# The [evaluation] keys every run must share: what the measures and the rule depend on.
SHARED_EVALUATION = ("duration", "hold", "return_tolerance", "ringing_cutoff_hz", "ringing_window", "following_window")
EFFORT = ("task_torque_nms", "task_effort_n2m2s", "task_work_j")  # robot_esn.arm_metrics' columns
# How the hyperparameters are listed; others follow, and the seed last.
ORDER = ("leak_rate", "spectral_radius", "input_scaling", "n_neurons", "sparsity", "ridge", "warmup")

Key = tuple[tuple[str, Any], ...]  # an ESN or a setting: its hyperparameters, sorted by name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="selection configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    runs = {scenario: [resolve_run_path(path) for path in config["runs"][scenario]] for scenario in SCENARIOS}
    configs = {path: read_toml(path / "config.toml") for paths in runs.values() for path in paths}
    check_shared(list(configs.values()))
    first = next(iter(configs.values()))
    radius = goal_radius(first)
    disturbances = [
        *disturbance_windows(configs[runs["block"][0]], "block"),
        *disturbance_windows(configs[runs["pushes"][0]], "push"),
    ]
    by_scenario = {scenario: esn_runs(paths, configs) for scenario, paths in runs.items()}
    missing = [s for s in ("block", "pushes") if set(by_scenario[s]) != set(by_scenario["nominal"])]
    if missing:
        msg = f"the ESNs run undisturbed must also run under the {' and the '.join(missing)}"
        raise ValueError(msg)

    esns = {
        key: esn_measures(
            by_scenario["nominal"][key][0],
            by_scenario["block"][key][0],
            by_scenario["pushes"][key][0],
            by_scenario["offsets"].get(key),
            disturbances,
            radius,
        )
        for key in by_scenario["nominal"]
    }
    guard = replay_ringing(config["runs"]["replay"], first["tracker"], len(disturbances))
    rule = config["rule"]
    settings = rank_settings(esns, guard, rule["chosen"], rule["fallback_seeds"])

    varying = varying_names(list(esns))
    esn_rows = [dict(key_values(key, varying)) | measures for key, measures in esns.items()]
    setting_rows = [dict(key_values(key, varying)) | row for key, row in settings.items()]
    chosen = [
        {"esn": f"ESN #{row['chosen']}"} | dict(key_values(key, varying)) | {"seed": row["typical_seed"]}
        for key, row in sorted(settings.items(), key=lambda kv: kv[1]["chosen"] or np.inf)
        if row["chosen"]
    ]
    for name, table in (("esns.csv", esn_rows), ("settings.csv", setting_rows), ("chosen.csv", chosen)):
        write_csv(run_dir / name, table)
    print_summary(esns, settings, guard, varying)
    print(f"\nWrote the results to {run_dir}")


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def value(text: str) -> Any:
    """A CSV field as a bool, a float, or the text itself."""
    if text in ("True", "False"):
        return text == "True"
    try:
        return float(text)
    except ValueError:
        return text


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [{key: value(text) for key, text in row.items()} for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(name for row in rows for name in row)))
        writer.writeheader()
        writer.writerows(rows)


def check_shared(configs: list[dict[str, Any]]) -> None:
    """Check that the runs share their demonstrations, their tracker, and what the measures depend on."""
    first = configs[0]
    for config in configs[1:]:
        for table in ("demonstrations", "tracker"):
            if config[table] != first[table]:
                msg = f"the runs must share their [{table}] table"
                raise ValueError(msg)
        for name in SHARED_EVALUATION:
            if config["evaluation"].get(name) != first["evaluation"].get(name):
                msg = f"the runs must share [evaluation] {name}"
                raise ValueError(msg)


def goal_radius(config: dict[str, Any]) -> float:
    """The goal radius (m): the target tolerance of the task of the run's demonstrations."""
    take = read_toml(resolve_run_path(config["demonstrations"]["run"]) / "config.toml")
    return float(take["task"]["target"]["tolerance"])


def disturbance_windows(config: dict[str, Any], name: str) -> list[tuple[str, str, float]]:
    """Each disturbance of a run: its name, its columns' suffix, and how long its window lasts after its end (s).

    A window lasts until the next disturbance starts, or the run ends.
    """
    spans = disturbance_spans(config.get("disturbance"))
    duration = float(config["evaluation"]["duration"])
    suffixes = [name.removeprefix("x") for name in per_disturbance("x", len(spans))]
    names = [name] if len(spans) == 1 else [f"{name}{k}" for k in range(1, len(spans) + 1)]
    ends = [onset for onset, _ in spans[1:]] + [duration]
    return [(n, s, stop - end) for n, s, (_, end), stop in zip(names, suffixes, spans, ends, strict=True)]


def esn_runs(paths: list[Path], configs: dict[Path, dict[str, Any]]) -> dict[Key, list[dict[str, Any]]]:
    """Each ESN's runs in some sweeps, from its start postures in order, by its hyperparameters."""
    out: dict[Key, list[dict[str, Any]]] = {}
    for path in paths:
        config = configs[path]
        for row in read_csv(path / "runs.csv"):
            esn = config["esn"] | {name: row[name] for name in config["sweep"]}
            out.setdefault(esn_key(esn), []).append(row)
    for rows in out.values():
        rows.sort(key=lambda row: row["start"])
    return out


def esn_key(esn: dict[str, Any]) -> Key:
    """An ESN's hyperparameters as a key, numbers as floats, whether read from a TOML file or a CSV file."""
    return tuple(
        sorted(
            (name, float(v) if isinstance(v, int | float) and not isinstance(v, bool) else v) for name, v in esn.items()
        )
    )


def setting_of(key: Key) -> Key:
    return tuple((name, v) for name, v in key if name != "seed")


def varying_names(keys: list[Key]) -> list[str]:
    """The hyperparameters whose values differ between the ESNs, and the seed, in the order of ORDER."""
    names = [name for name, _ in keys[0] if name == "seed" or len({dict(key)[name] for key in keys}) > 1]
    return sorted(names, key=lambda name: ORDER.index(name) if name in ORDER else len(ORDER))


def key_values(key: Key, names: list[str]) -> list[tuple[str, Any]]:
    values = dict(key)
    return [(name, int(values[name]) if name == "seed" else values[name]) for name in names if name in values]


def failure(run: dict[str, Any], scenario: str, radius: float) -> str:
    """How a sweep's run fails by the rule, or "" if it does not: under the pushes, leaving the goal is excused."""
    if not np.isfinite(run["arrival_time_s"]):
        return "never arrives"
    if run["final_distance_m"] > radius:
        return "ends away"
    if scenario == "pushes":
        return ""
    if run["left_goal"]:
        return "leaves the goal during the hold"
    return "hold cut short" if run["failed"] else ""


def esn_measures(
    nominal: dict[str, Any],
    block: dict[str, Any],
    pushes: dict[str, Any],
    offsets: list[dict[str, Any]] | None,
    disturbances: list[tuple[str, str, float]],
    radius: float,
) -> dict[str, Any]:
    """One ESN's outcome by the rule and its measures, from its runs (see the module).

    ``disturbances`` are the block's and the pushes' names, column suffixes, and
    windows after their ends, the block's first.
    """
    failures = [
        f"{scenario}: {how}"
        for scenario, run in (("nominal", nominal), ("block", block), ("pushes", pushes))
        if (how := failure(run, scenario, radius))
    ]
    if offsets is None:
        failures.append("offsets: not run")
    else:
        failures += [f"{run['origin']}: {how}" for run in offsets if (how := failure(run, "offsets", radius))]
    measures: dict[str, Any] = {"passes": not failures, "failures": "; ".join(failures)}
    measures["path_rmse_mm"] = 1000 * nominal["taught_path_rmse_m"]
    per: dict[str, list[float]] = {"following": [], "ringing_mm": [], "return_s": [], "off_path_mm": []}
    for k, (name, suffix, window) in enumerate(disturbances):
        run = block if k == 0 else pushes
        back = run[f"return_time_s{suffix}"]
        values = {
            "following": run[f"following_gain{suffix}"],
            "ringing_mm": 1000 * run[f"ringing_m{suffix}"],
            "return_s": back if np.isfinite(back) else window,
            "off_path_mm": 1000 * run[f"off_path_m{suffix}"],
        }
        for measure, v in values.items():
            per[measure].append(v)
            measures[f"{measure}_{name}"] = v
        measures[f"back_{name}"] = bool(np.isfinite(back))
    for measure in ("following", "ringing_mm", "return_s"):
        measures[measure] = float(np.mean(per[measure]))
    measures["block_lead_pct"] = 100 * block["reference_lead"]
    measures["block_peak_torque_nm"] = block["peak_torque_nm"]
    moved = [run for run in offsets or [] if str(run["origin"]).endswith(" deg")]  # not the demonstrated start
    measures["offsets_first_step_mm"] = float(np.median([1000 * r["first_step_m"] for r in moved])) if moved else np.nan
    measures["offsets_peak_torque_nm"] = float(np.median([r["peak_torque_nm"] for r in moved])) if moved else np.nan
    for scenario, run in (("nominal", nominal), ("block", block), ("pushes", pushes)):
        for name in EFFORT:
            measures[f"{name}_{scenario}"] = run.get(name, np.nan)
    for name in EFFORT:
        measures[f"{name}_offsets"] = float(np.median([r.get(name, np.nan) for r in moved])) if moved else np.nan
    return measures


def typical_seed(seeds: list[tuple[int, dict[str, Any]]]) -> int:
    """Of the passing seeds, the one nearest the medians of the path RMSE, the following, and the ringing.

    ``seeds`` are a setting's seeds and their measures. Each distance is scaled by
    the measure's interquartile range over all the seeds.
    """
    x = np.array([[m["path_rmse_mm"], m["following"], m["ringing_mm"]] for _, m in seeds])
    iqr = np.subtract(*np.percentile(x, [75, 25], axis=0))
    iqr[iqr == 0] = 1.0
    distance = np.sum(np.abs(x - np.median(x, axis=0)) / iqr, axis=1)
    passing = [i for i, (_, m) in enumerate(seeds) if m["passes"]]
    return seeds[min(passing, key=lambda i: distance[i])][0]


def rank_settings(
    esns: dict[Key, dict[str, Any]], guard: float, chosen: int, fallback_seeds: int
) -> dict[Key, dict[str, Any]]:
    """Each setting's seeds, qualification, guard, rank, typical seed, and medians; the chosen (see the module)."""
    grouped: dict[Key, list[tuple[int, dict[str, Any]]]] = {}
    for key, measures in esns.items():
        grouped.setdefault(setting_of(key), []).append((int(dict(key)["seed"]), measures))
    names = [name for name, v in next(iter(esns.values())).items() if isinstance(v, float)]
    settings: dict[Key, dict[str, Any]] = {}
    for setting, seeds in grouped.items():
        seeds.sort()
        passing = sum(m["passes"] for _, m in seeds)
        medians = {name: float(np.median([m[name] for _, m in seeds])) for name in names}
        settings[setting] = {
            "seeds": len(seeds),
            "passing_seeds": passing,
            "qualifies": passing == len(seeds),
            "within_guard": medians["ringing_mm"] <= guard,
            "rank": None,
            "chosen": None,
            "typical_seed": typical_seed(seeds) if passing else None,
        } | medians

    def ranked(eligible: list[Key]) -> list[Key]:
        return sorted(eligible, key=lambda k: -settings[k]["following"])

    qualifying = ranked([k for k, row in settings.items() if row["qualifies"] and row["within_guard"]])
    for rank, k in enumerate(qualifying, start=1):
        settings[k]["rank"] = rank
    picks = qualifying[:chosen]
    if len(picks) < chosen:
        relaxed = [
            k
            for k, row in settings.items()
            if not row["qualifies"] and row["passing_seeds"] >= fallback_seeds and row["within_guard"]
        ]
        picks += ranked(relaxed)[: chosen - len(picks)]
    for n, k in enumerate(picks, start=1):
        settings[k]["chosen"] = n
    return settings


def replay_ringing(paths: list[str], tracker: dict[str, Any], count: int) -> float:
    """The replay's ringing (mm), its mean over the disturbances, in runs of ``robot_esn.py``: the rule's guard.

    The runs must have the sweeps' ``tracker`` and, all together, ``count`` disturbances.
    """
    ringings: list[float] = []
    for path in map(resolve_run_path, paths):
        config = read_toml(path / "config.toml")
        if config["tracker"] != tracker:
            msg = f"the replay's run {path.name} must have the sweeps' [tracker]"
            raise ValueError(msg)
        n = len(disturbance_spans(config.get("disturbance")))
        (row,) = [r for r in read_csv(path / "metrics.csv") if r["arm"] == "replay" and r["start"] == 0]
        ringings += [1000 * row[name] for name in per_disturbance("ringing_m", n)]
    if len(ringings) != count:
        msg = f"the replay's runs have {len(ringings)} disturbances, the sweeps {count}"
        raise ValueError(msg)
    return float(np.mean(ringings))


def print_summary(
    esns: dict[Key, dict[str, Any]], settings: dict[Key, dict[str, Any]], guard: float, varying: list[str]
) -> None:
    names = [name for name in varying if name != "seed"]

    def label(key: Key) -> str:
        values = dict(key)
        return ", ".join(f"{name} {values[name]:g}" for name in names)

    print(f"{sum(m['passes'] for m in esns.values())} of {len(esns)} ESNs pass;", end=" ")
    print(f"{sum(row['qualifies'] for row in settings.values())} of {len(settings)} settings qualify")
    kinds: dict[str, int] = {}
    for measures in esns.values():
        for failure_text in filter(None, measures["failures"].split("; ")):
            where, how = failure_text.rsplit(": ", 1)
            kind = f"{'offsets' if where not in ('nominal', 'block', 'pushes') else where}: {how}"
            kinds[kind] = kinds.get(kind, 0) + 1
    print("Failing runs:", dict(sorted(kinds.items(), key=lambda kv: -kv[1])))
    print(f"The guard: the replay's ringing, {guard:.2f} mm")
    beyond = [k for k, row in settings.items() if row["qualifies"] and not row["within_guard"]]
    print(f"Qualifying settings beyond the guard: {', '.join(map(label, beyond)) or 'none'}")
    print("\nThe ranking, by the median following (ringing mm, return s, undisturbed path RMSE mm; typical seed):")
    for k, row in sorted(((k, r) for k, r in settings.items() if r["rank"]), key=lambda kv: kv[1]["rank"]):
        print(
            f"  {row['rank']:3d}. {label(k)}: following {row['following']:.2f}, ringing {row['ringing_mm']:.2f},"
            f" return {row['return_s']:.2f}, RMSE {row['path_rmse_mm']:.2f}; seed {row['typical_seed']}"
        )
    print("\nThe qualifying settings' best by the earlier reports' measures, each the smallest:")
    qualifying = [k for k, row in settings.items() if row["rank"]]
    for measure in ("path_rmse_mm", "ringing_mm", "return_s"):
        best = sorted(qualifying, key=lambda k: settings[k][measure])[:3]
        print(f"  {measure}: " + "; ".join(f"{label(k)} ({settings[k][measure]:.2f})" for k in best))
    print("\nChosen:")
    for k, row in sorted(((k, r) for k, r in settings.items() if r["chosen"]), key=lambda kv: kv[1]["chosen"]):
        how = f"rank {row['rank']}" if row["rank"] else f"{row['passing_seeds']} of {row['seeds']} seeds pass"
        print(f"  ESN #{row['chosen']}: {label(k)}, seed {row['typical_seed']} ({how})")


if __name__ == "__main__":
    main()
