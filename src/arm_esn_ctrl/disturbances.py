# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Disturbances of Stage 2: forces at the arm's tip, scripted on the task clock.

A disturbance is configured by a ``[disturbance]`` table, and several by
``[[disturbance]]`` tables, whose forces add up:

- ``type = "push"``: a constant force of ``force`` newtons from ``onset`` for
  ``duration`` seconds, in a direction relative to the reach (the straight line
  from the start hand position to the target, fixed for the run): ``angle_deg``
  degrees counterclockwise from the direction toward the target, or a
  ``direction``: ``"forward"`` along the reach toward the target (0 deg),
  ``"across"`` it (90 deg, the default), or ``"backward"`` away from the target
  (180 deg).
- ``type = "block"``: from ``onset`` until ``release``, a stiff spring-damper
  (``stiffness`` in N/m, ``damping`` in N s/m) holds the tip where it was at
  ``onset``, like a hand gripping the arm; then it lets go.
- ``type = "recorded"``: the tip force recorded in a ``log``, such as one that
  ``tools/robot_app.py`` saved while the arm was pushed by hand, replayed by time on
  the task clock (see skelarm's ``RecordedForce``), shifted by ``time_offset``
  seconds (0 by default): the same force at the same times, whatever the arm does.
  ``log`` is a path under the storage root, such as
  ``data/recorded_forces/<log>.sklog.npz``. Its pushes, for the measures after each
  disturbance, are the spans in which the force exceeds ``threshold`` newtons
  (``RECORDED_THRESHOLD`` by default), joined across lulls shorter than ``min_gap``
  seconds (``RECORDED_MIN_GAP``). Runs under it start from the log's start posture
  (see :func:`recorded_evaluation`), with an arm like the log's
  (:func:`check_recorded_arm`), and keep a copy of the log
  (:func:`keep_recorded_logs`). Force replay happens only under this type.

Each is a skelarm external force ``f(t, skeleton) -> (fx, fy)`` with ``t`` on the
task clock (0 when the task starts). From the same start posture, every arm meets
the same pushes: the same force, in the same direction, at the same times. The initial offset needs no force: it is a
start posture away from the demonstrated ones (``start_offsets_deg`` in
``[evaluation]``).
"""

from __future__ import annotations

import functools
import hashlib
import shutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tomli_w
from numpy.typing import NDArray
from skelarm import RecordedForce, Skeleton, StateLog, compute_jacobian

from arm_esn_ctrl.storage import resolve_run_path

# The keys each type of [disturbance] table needs, and those it may have.
DISTURBANCE_KEYS = {
    "push": {"type", "force", "onset", "duration"},
    "block": {"type", "onset", "release", "stiffness", "damping"},
    "recorded": {"type", "log"},
}
OPTIONAL_KEYS = {
    "push": {"direction", "angle_deg"},
    "block": set(),
    "recorded": {"time_offset", "threshold", "min_gap"},
}
# A recorded force counts as a push above this magnitude (N), which a drag's residual pull, with the cursor near the
# tip, stays below; pushes this close together (s) count as one.
RECORDED_THRESHOLD = 0.5
RECORDED_MIN_GAP = 0.2
# The push directions by name, as angles counterclockwise from the direction toward the target (deg).
PUSH_DIRECTIONS = {"forward": 0.0, "across": 90.0, "backward": 180.0}


@dataclass
class Push:
    """A constant tip force ``force`` (N, base frame) during ``[onset, onset + duration)``."""

    force: NDArray[np.float64]
    onset: float
    duration: float

    def __call__(self, t: float, skeleton: Skeleton) -> NDArray[np.float64]:
        if self.onset <= t < self.onset + self.duration:
            return self.force
        return np.zeros(2)


@dataclass
class Block:
    """A spring-damper that holds the tip where it was at ``onset``, until ``release``."""

    onset: float
    release: float
    stiffness: float  # N/m
    damping: float  # N s/m
    held: NDArray[np.float64] | None = field(default=None, init=False)  # where the tip is held

    def __call__(self, t: float, skeleton: Skeleton) -> NDArray[np.float64]:
        if not self.onset <= t < self.release:
            return np.zeros(2)
        tip = np.array([skeleton.links[-1].xe, skeleton.links[-1].ye])
        if self.held is None:
            self.held = tip
        velocity = compute_jacobian(skeleton) @ skeleton.dq
        return -self.stiffness * (tip - self.held) - self.damping * velocity


@dataclass
class Recorded:
    """A tip force recorded in a log, replayed by time on the task clock (``force(t, skeleton)``)."""

    force: Callable[[float, Skeleton], NDArray[np.float64]]

    def __call__(self, t: float, skeleton: Skeleton) -> NDArray[np.float64]:
        return np.asarray(self.force(t, skeleton), dtype=np.float64)


@dataclass
class Disturbances:
    """Several disturbances in one run: their tip forces add up."""

    parts: list[Push | Block | Recorded]

    def __call__(self, t: float, skeleton: Skeleton) -> NDArray[np.float64]:
        return np.sum([part(t, skeleton) for part in self.parts], axis=0)


def make_disturbance(
    config: dict[str, Any] | list[dict[str, Any]] | None,
    start_hand: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    dt: float | None = None,
) -> Push | Block | Recorded | Disturbances | None:
    """A fresh disturbance for one run from the hand position ``start_hand``, or None without one.

    ``config`` is a ``[disturbance]`` table, or a list of them (``[[disturbance]]``),
    whose forces add up. Make one per run: a block remembers where it holds the tip.
    With the simulation step ``dt``, a recorded force gives each step the mean
    recorded force over it, which is the recorded sample itself when the log has the
    same step; without it, the sample at each time.
    """
    if config is None:
        return None
    if isinstance(config, dict):
        return _make_one(config, start_hand, target, dt)
    parts = []
    for index, table in enumerate(config, start=1):
        try:
            parts.append(_make_one(table, start_hand, target, dt))
        except ValueError as error:
            msg = f"disturbance {index} of {len(config)}: {error}"
            raise ValueError(msg) from error
    return Disturbances(parts) if parts else None


def disturbance_spans(config: dict[str, Any] | list[dict[str, Any]] | None) -> list[tuple[float, float]]:
    """When each disturbance of ``config`` acts on the task clock, in the order given (none without one).

    A recorded disturbance acts in each of its pushes: the spans in which its force
    exceeds its ``threshold``, joined across lulls shorter than its ``min_gap``.
    """
    spans: list[tuple[float, float]] = []
    for table in _tables(config):
        if table["type"] == "push":
            spans.append((table["onset"], table["onset"] + table["duration"]))
        elif table["type"] == "block":
            spans.append((table["onset"], table["release"]))
        else:
            threshold = table.get("threshold", RECORDED_THRESHOLD)
            spans.extend(recorded_force(table).spans(threshold, table.get("min_gap", RECORDED_MIN_GAP)))
    return spans


def recorded_force(table: dict[str, Any]) -> RecordedForce:
    """The force recorded in a ``type = "recorded"`` table's log, on the task clock shifted by its ``time_offset``.

    Raises
    ------
    ValueError
        If the log records no tip force (``ext_force``).
    """
    log = _load_log(resolve_run_path(table["log"]))
    if "ext_force" not in log.channel_names:
        msg = f"the recorded disturbance's log {table['log']} records no tip force (ext_force)"
        raise ValueError(msg)
    return RecordedForce.from_log(log, time_offset=table.get("time_offset", 0.0))


def recorded_evaluation(
    evaluation: dict[str, Any], config: dict[str, Any] | list[dict[str, Any]] | None
) -> dict[str, Any]:
    """The ``[evaluation]`` table for runs under ``config``: with a recorded disturbance, from its log's start.

    The runs then start from the log's start posture alone, as the start postures
    group ``recorded``, and must last until the recorded pushes end. Without a
    recorded disturbance, ``evaluation`` is returned unchanged.

    Raises
    ------
    ValueError
        If ``evaluation`` lists start postures of its own, if the runs end before the
        recorded pushes do, or if more than one disturbance is recorded.
    """
    tables = _tables(config)
    recorded = [table for table in tables if table.get("type") == "recorded"]
    if not recorded:
        return evaluation
    if len(recorded) > 1:
        msg = "a run can replay one recorded disturbance only"
        raise ValueError(msg)
    if len(tables) > 1:
        warnings.warn(
            "the recorded force and the scripted disturbances add up: check that the log does not hold them already",
            stacklevel=2,
        )
    listed = [key for key in ("start_offsets_deg", "start_grid", "extra_starts") if key in evaluation]
    if listed:
        msg = (
            "runs under a recorded disturbance start from its log's start posture:"
            f" [evaluation] must not list start postures ({', '.join(listed)})"
        )
        raise ValueError(msg)
    table = recorded[0]
    spans = disturbance_spans(table)
    if spans and spans[-1][1] > evaluation["duration"] + 1e-9:
        msg = f"the runs last {evaluation['duration']:g} s, but the recorded pushes go on until {spans[-1][1]:.2f} s"
        raise ValueError(msg)
    start = np.degrees(_load_log(resolve_run_path(table["log"])).channel("q")[0])
    return evaluation | {"extra_starts": {"recorded": [start.tolist()]}}


def check_recorded_arm(config: dict[str, Any] | list[dict[str, Any]] | None, skeleton: Skeleton) -> None:
    """Refuse a recorded disturbance whose log was recorded with an arm other than ``skeleton``.

    Raises
    ------
    ValueError
        If a recorded log's arm has other links.
    """
    for table in _tables(config):
        if table.get("type") == "recorded":
            log = _load_log(resolve_run_path(table["log"]))
            if not same_arm(log.build_skeleton(), skeleton):
                msg = f"the recorded disturbance's log {table['log']} was recorded with another arm"
                raise ValueError(msg)


def keep_recorded_logs(config: dict[str, Any] | list[dict[str, Any]] | None, run_dir: Path) -> None:
    """Copy each recorded disturbance's log into ``run_dir``; ``recorded_forces.toml`` lists them with their SHA-256."""
    kept = []
    for table in _tables(config):
        if table.get("type") == "recorded":
            source = resolve_run_path(table["log"])
            shutil.copy2(source, run_dir / source.name)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            kept.append({"log": str(table["log"]), "copy": source.name, "sha256": digest})
    if kept:
        with (run_dir / "recorded_forces.toml").open("wb") as f:
            tomli_w.dump({"log": kept}, f)


def same_arm(a: Skeleton, b: Skeleton) -> bool:
    """Whether two robots have the same links: lengths, masses, inertias, centers of mass, and joint limits."""
    if a.num_joints != b.num_joints:
        return False
    return all(
        np.allclose(
            [p.length, p.m, p.i, p.rgx, p.rgy, p.qmin, p.qmax],
            [q.length, q.m, q.i, q.rgx, q.rgy, q.qmin, q.qmax],
        )
        for p, q in ((x.prop, y.prop) for x, y in zip(a.links[1:], b.links[1:], strict=True))
    )


def _tables(config: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [] if config is None else [config] if isinstance(config, dict) else list(config)


@functools.cache
def _load_log(path: Path) -> StateLog:
    """A saved log, loaded once however many runs replay it."""
    return StateLog.load(path)


def _make_one(
    config: dict[str, Any], start_hand: NDArray[np.float64], target: NDArray[np.float64], dt: float | None = None
) -> Push | Block | Recorded:
    """The disturbance of one ``[disturbance]`` table."""
    kind = config.get("type")
    if kind not in DISTURBANCE_KEYS:
        msg = f"unknown disturbance type {kind!r}; choose from {', '.join(DISTURBANCE_KEYS)}"
        raise ValueError(msg)
    keys = set(config)
    if not DISTURBANCE_KEYS[kind] <= keys <= DISTURBANCE_KEYS[kind] | OPTIONAL_KEYS[kind]:
        optional = f", and may have {', '.join(sorted(OPTIONAL_KEYS[kind]))}" if OPTIONAL_KEYS[kind] else ""
        msg = f"a {kind} [disturbance] needs the keys {', '.join(sorted(DISTURBANCE_KEYS[kind]))}{optional}"
        raise ValueError(msg)
    if kind == "push":
        if "direction" in config and "angle_deg" in config:
            msg = "a push takes a direction or an angle_deg, not both"
            raise ValueError(msg)
        direction = config.get("direction", "across")
        if direction not in PUSH_DIRECTIONS:
            msg = f"unknown push direction {direction!r}; choose from {', '.join(PUSH_DIRECTIONS)}"
            raise ValueError(msg)
        angle = np.radians(config.get("angle_deg", PUSH_DIRECTIONS[direction]))
        forward = (target - start_hand) / np.linalg.norm(target - start_hand)
        turn = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])  # counterclockwise
        return Push(config["force"] * (turn @ forward), config["onset"], config["duration"])
    if kind == "recorded":
        force = recorded_force(config)
        return Recorded(force if dt is None else force.stepped(dt))
    return Block(config["onset"], config["release"], config["stiffness"], config["damping"])
