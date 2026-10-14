# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Make the summary figures and tables of report 010.

    uv run python reports/010-noisy-search-a-negative-result/make_figures.py
    uv run python reports/010-noisy-search-a-negative-result/make_figures.py --logs

From the copies of the runs under ``results/`` (configurations, run records, metrics,
and sweep tables), with report 009's copies of its grid, its noise sweep, and its
validation, it draws into ``results/summary``:

- ``spectrum.png``: stage A, each noise's settings, the median ringing against the
  median return over their ten seeds, joined in the order of the noise level, with
  report 009's pink and brown noise beside (Section 3.1);
- ``grid.png``: the grid without noise (report 009's) and with white noise of 1°,
  stage B's and its edge's, likewise (Section 3.2);

and prints the tables of the report. With ``--logs``, it reads the shortlisted ESNs'
logs, which are not kept in Git, from their runs under the storage root, and prints
where each one's failing offset starts end (Section 3.3).
"""

from __future__ import annotations

import argparse
import csv
import functools
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from skelarm import StateLog

from arm_esn_ctrl.storage import resolve_run_path

REPORT = Path(__file__).resolve().parent
RESULTS = REPORT / "results"
SUMMARY = RESULTS / "summary"
# Report 009's copies: the grid and the noise sweep without the noises of stage A, and the validation's replay.
REPORT_009_RESULTS = REPORT.parent / "009-noisy-training-for-a-faster-return" / "results"
# Report 007's copies, where its chosen ESN's trained run is.
REPORT_007_RESULTS = REPORT.parent / "007-ringing-with-joint-pd" / "results"
GOAL = 0.02  # the goal's radius (m): a run ending farther from the target ends away from it

SCENARIOS = ("nominal", "block", "pushes")
# Report 009's grid, and stage B's, which repeats it with noise: its leak rates, one sweep each per scenario.
LEAKS = (0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1)
# The edge check's leak rates, likewise.
EDGE_LEAKS = (0.1, 0.15, 0.2, 0.3)
# Report 007's chosen setting (leak rate, spectral radius, input scaling) without noise, the score's reference.
REFERENCE = (0.02, 1.35, 2.0)
# The noises, by the exponent of their power spectrum: stage A's, then report 009's; and their levels (deg).
NOISES = {-2.0: "violet", -1.0: "blue", 0.0: "white", 1.0: "pink", 2.0: "brown"}
LEVELS = (0.5, 1.0, 2.0, 4.0, 8.0)
# After each disturbance (the block's release, then the three pushes), a hand not back on the path before the next
# one or the end of the 24 s run counts as back then (s after its end).
RETURN_CAPS = (24.0 - 5.5, 7.0 - 3.5, 15.0 - 7.5, 24.0 - 15.5)
# The ESNs tried in the robot app: the name, the directory of the copies, and the trained run's stem; none was chosen
# over report 007's chosen ESN, given beside them.
SHORTLIST = [
    ("lr 0.1, sr 0.8, is 2.5, seed 9", RESULTS, "grid_lr0.1_sr0.8_is2.5_seed9_filtered"),
    ("lr 0.15, sr 0.9, is 2, seed 4", RESULTS, "grid_lr0.15_sr0.9_is2_seed4_filtered"),
    ("lr 0.15, sr 0.8, is 2, seed 7", RESULTS, "grid_lr0.15_sr0.8_is2_seed7_filtered"),
    ("report 009's chosen ESN", REPORT_009_RESULTS, "noise_white1deg_ridge4e-2_seed3_filtered"),
    ("report 007's chosen ESN", REPORT_007_RESULTS, "grid_lr0.02_sr1.35_is2_seed7_filtered"),
]
# Report 009's validation runs with the search's tracker (joint PD at 0.4, reference velocity), for the replay.
VALIDATION = "{}_robot_white1deg_seed3_reference_velocity_filtered"
SEARCH_DAMPING = 0.4

REPLAY_COLOR = "#eb6834"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
NOISE_COLORS = {-2.0: "#7b3fb8", -1.0: "#2a78d6", 0.0: "#6b6b6b", 1.0: "#d6457a", 2.0: "#8b5a2b"}
LABEL_OFFSETS = {-2.0: (-17, 4), -1.0: (5, -11), 0.0: (-8, -13), 1.0: (7, -3), 2.0: (5, -11)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", action="store_true", help="also read the shortlisted ESNs' logs in the storage root")
    args = parser.parse_args()

    SUMMARY.mkdir(parents=True, exist_ok=True)
    figures = {"spectrum.png": plot_spectrum(), "grid.png": plot_grid()}
    for name, fig in figures.items():
        fig.savefig(SUMMARY / name, dpi=150)
    print_tables()
    if args.logs:
        print_resting_postures()
    print(f"\nWrote {', '.join(figures)} to {SUMMARY}")


# ---------------------------------------------------------------------------- reading the runs


def value(text: str) -> Any:
    """A CSV field as a bool, a float, or the text itself."""
    if text in ("True", "False"):
        return text == "True"
    try:
        return float(text)
    except ValueError:
        return text


@functools.cache
def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [{key: value(text) for key, text in row.items()} for row in csv.DictReader(f)]


def run_name(stem: str, results: Path = RESULTS) -> str:
    """The name of the one run of a configuration, by its stem."""
    matches = sorted(p.name for p in results.glob(f"*-{stem}") if p.is_dir())
    if len(matches) != 1:
        msg = f"expected one run of {stem}, found {matches}"
        raise FileNotFoundError(msg)
    return matches[0]


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


# ---------------------------------------------------------------------------- the search


def failure_of(run: dict[str, Any]) -> str:
    """How a run of a sweep fails, or "" if it does not."""
    if not np.isfinite(run["arrival_time_s"]):
        return "never arrives"
    if run["final_distance_m"] > GOAL:
        return "ends away"
    if run["left_goal"]:
        return "leaves the goal during the hold"
    return "hold cut short" if run["failed"] else ""


def seed_measures(n: dict[str, Any], b: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    """One seed's measures from its runs undisturbed (``n``), under the block (``b``), and under the pushes (``p``).

    As report 009's: ``failures``, how each scenario's run fails, if it does; ``rmse``,
    the undisturbed path RMSE (mm); ``ringing`` and ``ret``, the means over the four
    disturbances (the block's release, then the three pushes) of the ringing (mm) and
    of the return onto the path (s, a hand not back counting as back at RETURN_CAPS);
    ``offs``, after each, the farthest from the path (mm); and ``gaps``, at the end of
    each, the gap between the reference and the arm's joint angles (deg).
    """
    runs = dict(zip(SCENARIOS, (n, b, p), strict=True))
    ringings = [1000 * b["ringing_m"]] + [1000 * p[f"ringing_m_{k}"] for k in (1, 2, 3)]
    returns = [b["return_time_s"]] + [p[f"return_time_s_{k}"] for k in (1, 2, 3)]
    returns = [r if np.isfinite(r) else cap for r, cap in zip(returns, RETURN_CAPS, strict=True)]
    return {
        "seed": int(n["seed"]),
        "failures": {scenario: failure_of(r) for scenario, r in runs.items() if failure_of(r)},
        "rmse": 1000 * n["taught_path_rmse_m"],
        "ringings": ringings,
        "ringing": float(np.mean(ringings)),
        "ret": float(np.mean(returns)),
        "offs": [1000 * b["off_path_m"]] + [1000 * p[f"off_path_m_{k}"] for k in (1, 2, 3)],
        "gaps": [b["end_gap_deg"]] + [p[f"end_gap_deg_{k}"] for k in (1, 2, 3)],
    }


Setting = tuple[float, float, float]


def sweep_seeds(stems: list[str], key: tuple[str, str], results: Path) -> dict[Any, list[dict[str, Any]]]:
    """Each seed of every setting of sweeps, by the setting's two swept values, then the stem's index (``key``).

    ``stems`` holds each sweep's stem with ``{}`` for its scenario.
    """
    out: dict[Any, list[dict[str, Any]]] = {}
    for i, stem in enumerate(stems):
        runs = {
            scenario: {
                (r[key[0]], r[key[1]], r["seed"]): r
                for r in read_csv(results / run_name(stem.format(scenario), results) / "runs.csv")
            }
            for scenario in SCENARIOS
        }
        for a, b, seed in sorted(runs["nominal"]):
            out.setdefault((i, a, b), []).append(seed_measures(*(runs[s][a, b, seed] for s in SCENARIOS)))
    return out


def grid_of(prefix: str, leaks: tuple[float, ...], results: Path) -> dict[Setting, list[dict[str, Any]]]:
    """A grid's seeds by (leak rate, spectral radius, input scaling), one sweep per leak rate and scenario."""
    stems = [f"{prefix}_lr{leak:g}_{{}}_filtered" for leak in leaks]
    seeds = sweep_seeds(stems, ("spectral_radius", "input_scaling"), results)
    return {(leaks[i], radius, scaling): s for (i, radius, scaling), s in seeds.items()}


@functools.cache
def plain_grid() -> dict[Setting, list[dict[str, Any]]]:
    """Report 009's grid, without noise."""
    return grid_of("sweep_robot_damped", LEAKS, REPORT_009_RESULTS)


@functools.cache
def noisy_grid() -> dict[Setting, list[dict[str, Any]]]:
    """Stage B's grid, report 009's with white noise of 1°."""
    return grid_of("sweep_robot_noisy", LEAKS, RESULTS)


@functools.cache
def edge_grid() -> dict[Setting, list[dict[str, Any]]]:
    """The edge check's grid, beyond stage B's corner, with the same noise."""
    return grid_of("sweep_robot_noisy_edge", EDGE_LEAKS, RESULTS)


@functools.cache
def noise_seeds() -> dict[tuple[float, float], list[dict[str, Any]]]:
    """Each seed at report 007's setting by (noise level, exponent): stage A's, report 009's pink and brown at ridge
    4e-2, and its noiseless setting there (level 0)."""
    key = ("teacher_noise_deg", "teacher_noise_exponent")
    spectrum = sweep_seeds(["sweep_robot_noise_spectrum_{}_filtered"], key, RESULTS)
    earlier = sweep_seeds(["sweep_robot_damped_noise_ridge4e-2_{}_filtered"], key, REPORT_009_RESULTS)
    out = {(level, exponent): s for (_, level, exponent), s in spectrum.items()}
    out |= {(level, exponent): s for (_, level, exponent), s in earlier.items() if exponent > 0 or level == 0}
    return out


def qualifies(seeds: list[dict[str, Any]]) -> bool:
    """A setting whose seeds all arrive, hold, and end in the goal in the three scenarios (the rule)."""
    return not any(s["failures"] for s in seeds)


def median(seeds: list[dict[str, Any]], key: str, i: int | None = None) -> float:
    """The median over seeds of a measure, or of its ``i``-th entry."""
    return float(np.median([s[key] if i is None else s[key][i] for s in seeds]))


def score(seeds: list[dict[str, Any]]) -> float:
    """A setting's score: the mean of the ratios of its medians of the ringing and the return to report 007's."""
    reference = plain_grid()[REFERENCE]
    return 0.5 * (
        median(seeds, "ringing") / median(reference, "ringing") + median(seeds, "ret") / median(reference, "ret")
    )


def typical_seed(seeds: list[dict[str, Any]]) -> int:
    """The seed nearest the medians of the RMSE, the ringing, and the return, each by its interquartile range."""
    x = np.array([[s["rmse"], s["ringing"], s["ret"]] for s in seeds])
    iqr = np.subtract(*np.percentile(x, [75, 25], axis=0))
    iqr[iqr == 0] = 1.0
    return int(seeds[int(np.argmin(np.sum(np.abs(x - np.median(x, axis=0)) / iqr, axis=1)))]["seed"])


@functools.cache
def replay() -> dict[str, list[float]]:
    """The replay with the search's tracker, from report 009's validation: after the block's release and each push,
    its ringing (mm), its return (s), and the gap between its reference and the arm at the disturbance's end (deg)."""
    runs = {
        scenario: next(
            r
            for r in read_csv(
                REPORT_009_RESULTS / run_name(VALIDATION.format(scenario), REPORT_009_RESULTS) / "metrics.csv"
            )
            if r["damping"] == SEARCH_DAMPING and r["arm"] == "replay" and r["start"] == 0
        )
        for scenario in ("block", "pushes")
    }
    b, p = runs["block"], runs["pushes"]
    returns = [b["return_time_s"]] + [p[f"return_time_s_{k}"] for k in (1, 2, 3)]
    return {
        "ringings": [1000 * b["ringing_m"]] + [1000 * p[f"ringing_m_{k}"] for k in (1, 2, 3)],
        "returns": [r if np.isfinite(r) else cap for r, cap in zip(returns, RETURN_CAPS, strict=True)],
        "gaps": [b["end_gap_deg"]] + [p[f"end_gap_deg_{k}"] for k in (1, 2, 3)],
    }


def gap_ratio(seeds: list[dict[str, Any]]) -> list[float]:
    """After the block's release and each push, the median over seeds of the reference-arm gap over the replay's."""
    return [median(seeds, "gaps", i) / g for i, g in enumerate(replay()["gaps"])]


# ---------------------------------------------------------------------------- figures


def tradeoff_axes(ax: Axes, xmax: float, ylim: tuple[float, float]) -> None:
    """Report 007's setting, the replay, and the lines of equal score on a plot of the ringing against the return."""
    reference = plain_grid()[REFERENCE]
    ringing, ret = median(reference, "ringing"), median(reference, "ret")
    for level in (0.6, 0.8, 1.0):
        x = np.linspace(0.0, 2 * level * ret, 50)
        ax.plot(x, (2 * level - x / ret) * ringing, color=TEXT_COLOR, linewidth=0.6, linestyle=":", zorder=1)
        start = (
            (0.0, 2 * level * ringing)
            if 2 * level * ringing <= ylim[1]
            else ((2 * level - ylim[1] / ringing) * ret, ylim[1])
        )
        if ylim[0] <= start[1] and start[0] <= xmax:
            ax.annotate(
                f"score {level:g}",
                start,
                xytext=(4, -2),
                textcoords="offset points",
                fontsize=7,
                color=TEXT_COLOR,
                va="top",
            )
    ax.scatter([ret], [ringing], marker="*", s=220, color="#0b0b0b", zorder=5)
    ax.scatter(
        [np.mean(replay()["returns"])], [np.mean(replay()["ringings"])], marker="D", s=60, color=REPLAY_COLOR, zorder=5
    )
    ax.set_xlim(0, xmax)
    ax.set_ylim(*ylim)
    ax.set_xlabel("return onto the path (s): median over the seeds of the mean after the four", color=TEXT_COLOR)


MARKERS = [
    Line2D(
        [],
        [],
        color=TEXT_COLOR,
        marker="o",
        markerfacecolor=SURFACE_COLOR,
        linestyle="none",
        label="a seed fails (hollow)",
    ),
    Line2D(
        [], [], color="#0b0b0b", marker="*", markersize=12, linestyle="none", label="report 007's setting, no noise"
    ),
    Line2D([], [], color=REPLAY_COLOR, marker="D", linestyle="none", label="the replay"),
]


def plot_spectrum() -> Figure:
    """Every noise at report 007's setting: the median over its seeds of the ringing against that of the return.

    Each noise's settings are joined in the order of the noise level, from the
    noiseless setting, which they share; filled: the settings that qualify.
    """
    settings = noise_seeds()
    fig = Figure(figsize=(10.0, 6.5), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "Stage A, at report 007's setting, the take fed in four times with noise on the input:"
        "\nringing against return after the block and the pushes, each setting over its ten seeds",
        color="#0b0b0b",
    )
    ax = fig.subplots()
    style(ax)
    plain = settings[0.0, 0.0]
    for exponent, color in NOISE_COLORS.items():
        group = [plain] + [settings[level, exponent] for level in LEVELS]
        x, y = [median(s, "ret") for s in group], [median(s, "ringing") for s in group]
        ax.plot(x, y, color=color, linewidth=1.3, zorder=2)
        for level, seeds, xi, yi in zip(LEVELS, group[1:], x[1:], y[1:], strict=True):
            ax.scatter(
                xi,
                yi,
                s=40,
                color=color if qualifies(seeds) else SURFACE_COLOR,
                edgecolor=color,
                linewidth=1.4,
                zorder=3,
            )
            ax.annotate(
                f"{level:g}°",
                (xi, yi),
                xytext=LABEL_OFFSETS[exponent],
                textcoords="offset points",
                fontsize=7,
                color=color,
            )
    top = max(max(median(s, "ringing") for s in settings.values()), float(np.mean(replay()["ringings"])))
    tradeoff_axes(ax, 1.06 * max(median(s, "ret") for s in settings.values()), (5.0, 1.04 * top))
    ax.set_ylabel("ringing (mm): median over the seeds of the mean after the four", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=c, marker="o", label=f"{NOISES[e]} noise" + (" (report 009)" if e > 0 else ""))
        for e, c in NOISE_COLORS.items()
    ]
    ax.legend(handles=handles + MARKERS, frameon=False, labelcolor=TEXT_COLOR, fontsize=8, loc="lower left")
    return fig


def plot_grid() -> Figure:
    """Every setting of report 009's grid without noise, and of stage B's and the edge's with white noise of 1°.

    Each point is the median over a setting's seeds of the ringing against that of
    the return; filled: the settings that qualify. Circles: report 009's grid and
    stage B's; squares: the edge's, beyond stage B's corner.
    """
    plain, noisy, edge = plain_grid(), noisy_grid(), edge_grid()
    leaks = sorted(set(LEAKS) | set(EDGE_LEAKS))
    ramp = colormaps["viridis"]
    colors = {leak: ramp(0.92 * i / (len(leaks) - 1)) for i, leak in enumerate(leaks)}
    fig = Figure(figsize=(13.0, 5.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(
        "The grid without noise (report 009) and with white noise of 1° (stage B and its edge):"
        " ringing against return, each setting over its ten seeds",
        color="#0b0b0b",
    )
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    panels = [("without noise (report 009)", [(plain, "o")]), ("with white noise of 1°", [(noisy, "o"), (edge, "s")])]
    everything = [*plain.values(), *noisy.values(), *edge.values()]
    xmax = 1.06 * max(median(s, "ret") for s in everything)
    top = max(max(median(s, "ringing") for s in everything), float(np.mean(replay()["ringings"])))
    for ax, (title, groups) in zip(axes, panels, strict=True):
        style(ax)
        for settings, marker in groups:
            for k, seeds in settings.items():
                color = colors[k[0]]
                ax.scatter(
                    median(seeds, "ret"),
                    median(seeds, "ringing"),
                    s=34,
                    marker=marker,
                    color=color if qualifies(seeds) else SURFACE_COLOR,
                    edgecolor=color,
                    linewidth=1.2,
                    zorder=3,
                )
        tradeoff_axes(ax, xmax, (0.0, 1.08 * top))
        ax.set_title(title, color="#0b0b0b", fontsize=10)
    axes[0].set_ylabel("ringing (mm): median over the seeds of the mean after the four", color=TEXT_COLOR)
    handles = [
        Line2D([], [], color=c, marker="o", linestyle="none", label=f"leak rate {k:g}") for k, c in colors.items()
    ]
    handles.append(Line2D([], [], color=TEXT_COLOR, marker="s", linestyle="none", label="the edge"))
    axes[1].legend(
        handles=handles + MARKERS, frameon=False, labelcolor=TEXT_COLOR, fontsize=8, ncol=2, loc="upper right"
    )
    return fig


# ---------------------------------------------------------------------------- tables


def cell(seeds: list[dict[str, Any]]) -> str:
    """A setting's score, then its median ringing and return, starred if a seed fails."""
    star = "" if qualifies(seeds) else "*"
    return f"{score(seeds):.3f}{star} ({median(seeds, 'ringing'):.1f} mm, {median(seeds, 'ret'):.2f} s)"


def failures(settings: list[list[dict[str, Any]]]) -> str:
    """How the failing runs under the pushes fail, counted over the settings' seeds."""
    counts: dict[str, int] = {}
    for seeds in settings:
        for s in seeds:
            how = s["failures"].get("pushes")
            if how:
                counts[how] = counts.get(how, 0) + 1
    return ", ".join(f"{n} {how}" for how, n in sorted(counts.items(), key=lambda kv: -kv[1])) or "none"


def print_tables() -> None:
    noise = noise_seeds()
    print("== Section 3.1: each noise at report 007's setting: score (median ringing, return); * a seed fails")
    print("| Noise | " + " | ".join(f"{level:g}°" for level in LEVELS) + " |")
    print("| --- |" + " --- |" * len(LEVELS))
    for exponent, name in NOISES.items():
        print(f"| {name} | " + " | ".join(cell(noise[level, exponent]) for level in LEVELS) + " |")
    print(f"without noise: {cell(noise[0.0, 0.0])}")
    print("\n   the undisturbed path RMSE, in the median (mm)")
    for exponent, name in NOISES.items():
        print(f"| {name} | " + " | ".join(f"{median(noise[level, exponent], 'rmse'):.2f}" for level in LEVELS) + " |")

    plain, noisy, edge = plain_grid(), noisy_grid(), edge_grid()
    print("\n== Section 3.2: settings that qualify per leak rate, with noise (without); how the pushes' runs fail")
    for leak in LEAKS:
        group = [k for k in noisy if k[0] == leak]
        print(
            f"| {leak:g} | {sum(qualifies(noisy[k]) for k in group)} ({sum(qualifies(plain[k]) for k in group)})"
            f" of {len(group)} | {failures([noisy[k] for k in group])} |"
        )
    print("   the edge")
    for leak in EDGE_LEAKS:
        group = [k for k in edge if k[0] == leak]
        print(
            f"| {leak:g} | {sum(qualifies(edge[k]) for k in group)} of {len(group)} |"
            f" {failures([edge[k] for k in group])} |"
        )
    repeated = (0.1, 0.99, 2.0)
    same = all(
        np.isclose(a["ringing"], b["ringing"]) and np.isclose(a["ret"], b["ret"]) and a["failures"] == b["failures"]
        for a, b in zip(edge[repeated], noisy[repeated], strict=True)
    )
    print(f"   the edge's {repeated} repeats stage B's: {same}")

    both = {**edge, **noisy}
    ranked = sorted((k for k in both if qualifies(both[k])), key=lambda k: score(both[k]))
    print(
        "\n   the top ten qualifying settings of both grids: score, ringing, return, RMSE, typical seed, ringing each"
    )
    for k in ranked[:10]:
        where = "edge" if k not in noisy else "B"
        s = both[k]
        print(
            f"| {k[0]:g} | {k[1]:g} | {k[2]:g} | {where} | {score(s):.3f} | {median(s, 'ringing'):.2f} mm |"
            f" {median(s, 'ret'):.2f} s | {median(s, 'rmse'):.2f} mm | {typical_seed(s)} |"
            f" {' / '.join(f'{median(s, "ringings", i):.1f}' for i in range(4))} mm |"
        )
    print(f"   report 007's setting with the noise (report 009's chosen): {cell(noisy[REFERENCE])}")
    # Seed by seed, against stage B's top setting: a seed's score, as a setting's, from its own ringing and return.
    reference = plain_grid()[REFERENCE]
    ringing, ret = median(reference, "ringing"), median(reference, "ret")
    stage_b_top = min((k for k in noisy if qualifies(noisy[k])), key=lambda k: score(noisy[k]))
    for k in ranked[:3]:
        pairs = zip(both[k], noisy[stage_b_top], strict=True)
        better = sum(
            a["ringing"] / ringing + a["ret"] / ret < b["ringing"] / ringing + b["ret"] / ret for a, b in pairs
        )
        print(f"   {k} against stage B's top {stage_b_top}: better in {better} of 10 seeds")
    r = replay()
    print(
        f"   the replay: ringing {np.mean(r['ringings']):.2f} mm, return {np.mean(r['returns']):.2f} s;"
        f" ringing after each {' / '.join(f'{x:.1f}' for x in r['ringings'])} mm"
    )

    print("\n== Section 4: the reference-arm gap at a disturbance's end over the replay's (block; the three pushes)")
    print(f"   the replay's gaps: {' / '.join(f'{g:.1f}' for g in r['gaps'])} deg")
    for name, settings in (("without noise", plain), ("stage B", noisy), ("the edge", edge)):
        ratios = np.array([gap_ratio(s) for s in settings.values()])
        pushes = ratios[:, 1:].mean(axis=1)
        print(
            f"   {name}: block, median over settings {np.median(ratios[:, 0]):.2f};"
            f" pushes, mean of the three, {pushes.min():.2f} to {pushes.max():.2f} (median {np.median(pushes):.2f})"
        )
    for name, seeds in (("report 007's setting", plain[REFERENCE]), ("with white noise of 1°", noisy[REFERENCE])):
        print(f"   {name}: {' / '.join(f'{x:.2f}' for x in gap_ratio(seeds))}")

    print("\n== Section 3.3: on its own from the 48 offset starts")
    for name, results, stem in SHORTLIST:
        rows = read_csv(results / run_name(stem, results) / "metrics.csv")
        offsets = [r for r in rows if str(r["origin"]).endswith(" deg")]
        steps = 1000 * np.array([r["first_step_m"] for r in offsets])
        ok = sum(bool(r["success"]) for r in offsets)
        print(f"| {name} | {ok} of {len(offsets)} | {np.median(steps):.0f} / {steps.max():.0f} mm |")


def print_resting_postures() -> None:
    """Where each shortlisted ESN's failing offset starts end, from the logs of its trained run."""
    print("\n== Section 3.3: where the failing offset starts end (deg)")
    for name, results, stem in SHORTLIST:
        run = run_name(stem, results)
        rows = read_csv(results / run / "metrics.csv")
        logs = resolve_run_path(f"results/{run}")
        ends = [
            np.degrees(StateLog.load(logs / f"esn_{int(r['start']):02d}.sklog.npz").channel("q")[-1])
            for r in rows
            if not r["success"]
        ]
        if ends:
            ends_array = np.array(ends)
            spread = np.ptp(ends_array, axis=0).max()
            print(
                f"   {name}: {len(ends)} end at ({ends_array[0, 0]:.1f}, {ends_array[0, 1]:.1f}), spread {spread:.2f}"
            )
        else:
            print(f"   {name}: none fails")


if __name__ == "__main__":
    main()
