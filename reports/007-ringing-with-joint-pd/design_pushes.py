# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Reproduce the design of report 007's pushes: their duration, the time of the push across, and the forward force.

    uv run python reports/007-ringing-with-joint-pd/design_pushes.py

Report 006's chosen ESN and the replay of the take run with joint PD from the
demonstrated start, as ``experiments/robot_esn.py`` runs them, under variants of the
study's pushes (``experiments/manual_demonstration_v2_robot_ringing/
pushes_robot_lr0.03_sr1.35_is1.5_seed8_filtered.toml``: 3 N nearly forward at 3 s,
5 N nearly across at 7 s, 4 N nearly across the other way at 15 s, 0.5 s each).
Nothing is saved; it prints, for each variant and arm, per push: the gap between
the reference and the arm at its end (deg), the hand's speed then (m/s), the
ringing after it (mm), the farthest from the taught path (mm), and when the hand is
back within 5 mm of it (s); the hand's distance from the path when the push across
starts (mm); and whether the arm arrives and holds. About two minutes.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from typing import Any

import numpy as np

from arm_esn_ctrl.autonomous import load_setup
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.disturbances import disturbance_spans, make_disturbance
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.metrics import distances_to_path
from arm_esn_ctrl.storage import REPO_ROOT, resolve_run_path
from arm_esn_ctrl.tracking import EsnSource, ReplaySource, TrackerConfig, nearest_demonstration, track, tracking_gains

CONFIG = (
    REPO_ROOT
    / "experiments"
    / "manual_demonstration_v2_robot_ringing"
    / "pushes_robot_lr0.03_sr1.35_is1.5_seed8_filtered.toml"
)
PD = TrackerConfig("pd", 20.0, 0.02, damping=0.1, reference_velocity=False)


def runner(name: str) -> Any:
    """An experiment runner of ``experiments/``, such as robot_esn, loaded by its path to use its functions."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "experiments" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered first, as an import does: its dataclasses look it up
    spec.loader.exec_module(module)
    return module


robot_esn = runner("robot_esn")
with CONFIG.open("rb") as f:
    config = tomllib.load(f)
esn_path = resolve_run_path(config["esn"]["model"])
esn = ReachingEsn.load(esn_path)
with (esn_path.parent / "config.toml").open("rb") as f:
    demonstrations = tomllib.load(f)["demonstrations"]
with (resolve_run_path(demonstrations["run"]) / "config.toml").open("rb") as f:
    simulator = tomllib.load(f)["simulator"]
evaluation = config["evaluation"]
setup = load_setup({"demonstrations": demonstrations, "esn": {"dt": esn.config.dt}, "evaluation": evaluation})
gains = tracking_gains(PD, setup.skeleton, np.mean([q[-1] for q in setup.demos.values()], axis=0))
start = setup.starts[0]
start_hand = endpoint_positions(setup.skeleton, start.q[np.newaxis])[0]
taught = setup.demo_hands[nearest_demonstration(start.q, setup.demos)]


def pushes(duration: float = 0.5, across: float = 7.0, forward_force: float = 3.0) -> list[dict[str, Any]]:
    """The study's pushes, with another duration, time of the push across, or forward force."""
    tables = [dict(table) for table in config["disturbance"]]
    for table in tables:
        table["duration"] = duration
    tables[0]["force"] = forward_force
    tables[1]["onset"] = across
    return tables


def measure(arm: str, disturbance: list[dict[str, Any]]) -> str:
    """One arm's run under the pushes, as a line of measures."""
    if arm == "esn":
        source: Any = EsnSource(esn)
    else:
        source = ReplaySource(setup.demos[nearest_demonstration(start.q, setup.demos)])
    log = track(
        robot_esn.posed(setup.skeleton, start.q),
        source,
        PD,
        gains,
        period=esn.config.dt,
        warmup_steps=esn.config.warmup_steps,
        duration=evaluation["duration"],
        dt=simulator["dt"],
        enforce_limits=simulator.get("enforce_limits", True),
        external_force=make_disturbance(disturbance, start_hand, setup.target),
    )
    spans = disturbance_spans(disturbance)
    ringing = robot_esn.ringing_settings(evaluation)
    m = robot_esn.arm_metrics(log, 0, setup, esn.config.dt, [0.0, evaluation["duration"]], spans, 0.005, ringing)
    t = log.times
    q, dq = (log.channel(name).reshape(len(t), -1) for name in ("q", "dq"))
    away = 1000 * distances_to_path(endpoint_positions(setup.skeleton, q), taught)
    pieces = []
    for k, (_, end) in enumerate(spans, start=1):
        i = int(np.argmin(np.abs(t - end)))
        hand_speed = (
            np.linalg.norm(
                endpoint_positions(setup.skeleton, (q[i] + 1e-6 * dq[i])[np.newaxis])[0]
                - endpoint_positions(setup.skeleton, q[i][np.newaxis])[0]
            )
            / 1e-6
        )
        pieces.append(
            f"{m[f'end_gap_deg_{k}']:4.1f}, {hand_speed:.2f}, {1000 * m[f'ringing_m_{k}']:4.1f},"
            f" {1000 * m[f'off_path_m_{k}']:3.0f}, {m[f'return_time_s_{k}']:.2f}"
        )
    before = away[int(np.argmin(np.abs(t - spans[1][0])))]
    return f"{' | '.join(pieces)} || {before:4.1f} | {m['arrival_time_s']:.2f}, {'holds' if m['success'] else 'fails'}"


def show(title: str, variants: list[tuple[str, list[dict[str, Any]]]]) -> None:
    print(f"\n== {title}")
    print("  per push (forward, across, across after the hold): gap at its end (deg), hand speed then (m/s),")
    print("  ringing (mm), farthest from the path (mm), back within 5 mm (s) || off the path when the push across")
    print("  starts (mm) | arrival (s), hold")
    for label, disturbance in variants:
        for arm in ("esn", "replay"):
            print(f"  {label:<14} {arm:<6} {measure(arm, disturbance)}")


show(
    "The duration of the pushes (forward 3 N at 3 s, across 5 N at 7 s and 4 N at 15 s)",
    [(f"{duration:g} s", pushes(duration=duration)) for duration in (0.2, 0.35, 0.5, 0.75, 1.0)],
)
show(
    "The time of the push across, 0.5 s pushes",
    [(f"across at {across:g} s", pushes(across=across)) for across in (7.0, 8.0)],
)
show(
    "The force of the push forward, 0.5 s and 0.55 s pushes",
    [
        (f"{force:g} N, {duration:g} s", pushes(duration=duration, forward_force=force))
        for force in (2.0, 3.0, 4.0)
        for duration in (0.5, 0.55)
    ],
)
