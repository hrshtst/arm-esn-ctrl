# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""
Interactive demonstration of a trained ESN generating joint-angle references.

Load a robot and its reaching task (a skelarm TOML file with ``[skeleton]`` and
``[task]``, such as ``configs/demonstrations/reach_tvs.toml``) and an ESN saved by
``experiments/autonomous_esn.py`` (``esn.toml`` in a run directory). Drag the arm tip
with the mouse to choose a start posture (inverse kinematics), then press Play: the
ESN is reset, driven by the held start posture for its warm-up (shown at negative
times), and then runs autonomously, its output fed back as its next input. The arm
shows every posture the ESN generates, and keeps running until you pause it. Reset
returns the arm to the start posture of the last run, where you can pose it again.

While the ESN runs, the side panel shows the metrics of :mod:`arm_esn_ctrl.metrics`:
the first-step jump, when the hand arrives within the target tolerance, and whether it
stays there during the hold window. If the file also holds ``[controller]`` and
``[simulator]`` tables (a demonstration configuration), the demonstrator's own reach
from the same start posture is drawn for comparison, and the panel compares the ESN
with it.

Keys, as in skelarm's player: ``Space`` play/pause, ``Right``/``F`` one step while
paused, ``R`` or ``Home`` reset, ``Q`` quit.

Usage::

    uv run python tools/esn_reference_app.py configs/demonstrations/reach_tvs.toml \\
        --model <run directory>/esn.toml
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)
from skelarm import (
    PlaybackClock,
    SkelarmCanvas,
    Skeleton,
    SpeedSpinBox,
    Task,
    TransportBar,
    bind_quit_key,
    compute_forward_kinematics,
)
from skelarm.canvas import TrailOverlay

from arm_esn_ctrl.demonstrations import endpoint_positions, resample_joint_angles, simulate_reaches
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.metrics import arrival_index, path_distance

_PANEL_WIDTH_PX = 340  # fixed side-panel width, so the changing readouts cannot resize it
_ESN_COLOR = QColor(42, 120, 214)  # the ESN's tip path
_LAST_RUN_COLOR = QColor(42, 120, 214, 60)  # faint: the tip path of the last run, after a reset
# The demonstrator's tip path: a wide, translucent band under the ESN's path, which stays visible on top.
_DEMONSTRATOR_COLOR = QColor(82, 81, 78, 90)
_DEMONSTRATOR_WIDTH_PX = 8.0
_DEMONSTRATOR_TABLES = ("skeleton", "task", "simulator", "controller")


@dataclass
class Demonstrator:
    """The demonstrator's reach from a run's start posture, at the ESN's sampling period."""

    q: NDArray[np.float64]
    hand: NDArray[np.float64]
    arrival_time: float | None


@dataclass
class LiveRun:
    """One autonomous run of the ESN, recorded and measured as it goes.

    Times are on the task clock: the warm-up runs at negative times and the
    ESN starts moving the arm at time 0.
    """

    start_q: NDArray[np.float64]
    stream: Iterator[NDArray[np.float64]]
    skeleton: Skeleton
    target: NDArray[np.float64]
    radius: float  # goal radius: the target tolerance (m)
    hold: float  # hold duration T_h after arrival (s)
    dt: float
    warmup_steps: int
    demonstrator: Demonstrator | None = None
    times: list[float] = field(default_factory=list)
    q: list[NDArray[np.float64]] = field(default_factory=list)
    hand: list[NDArray[np.float64]] = field(default_factory=list)
    arrival_time: float | None = None
    left_goal: bool = False
    hold_error: float | None = None  # distance to the target at the latest sample of the hold window (m)
    hold_observed: float = 0.0  # how much of the hold window has passed (s)
    reach_path_distance: float = 0.0  # largest distance from the demonstrator's path until arrival (m)
    _joint_error_sum: float = 0.0
    _joint_error_count: int = 0

    def take_step(self) -> None:
        """Take the next posture from the ESN and update the metrics."""
        q = next(self.stream)
        time = (len(self.times) - self.warmup_steps) * self.dt
        hand = endpoint_positions(self.skeleton, q[np.newaxis, :])[0]
        self.times.append(time)
        self.q.append(q)
        self.hand.append(hand)
        if time < 0.0:
            return
        distance = float(np.linalg.norm(hand - self.target))
        if self.arrival_time is None and distance <= self.radius:
            self.arrival_time = time
        if self.arrival_time is not None and time <= self.arrival_time + self.hold + 1e-9:
            self.left_goal = self.left_goal or distance > self.radius
            self.hold_error = distance
            self.hold_observed = time - self.arrival_time
        if self.demonstrator is not None:
            self._compare_with_demonstrator(self.demonstrator, time, q, hand)

    def _compare_with_demonstrator(
        self, demonstrator: Demonstrator, time: float, q: NDArray[np.float64], hand: NDArray[np.float64]
    ) -> None:
        if self.arrival_time is None or time <= self.arrival_time:
            distance = path_distance(hand[np.newaxis, :], demonstrator.hand)
            self.reach_path_distance = max(self.reach_path_distance, distance)
        index = round(time / self.dt)
        reaching = demonstrator.arrival_time is None or time <= demonstrator.arrival_time + 1e-9
        if index < len(demonstrator.q) and reaching:
            self._joint_error_sum += float(np.sum((q - demonstrator.q[index]) ** 2))
            self._joint_error_count += q.size

    @property
    def time(self) -> float:
        """The time of the latest posture (s)."""
        return self.times[-1] if self.times else -self.warmup_steps * self.dt

    @property
    def first_step(self) -> float | None:
        """How far the hand moved in the ESN's first step (m), once it has been taken."""
        start = self.warmup_steps
        return float(np.linalg.norm(self.hand[start + 1] - self.hand[start])) if len(self.hand) > start + 1 else None

    @property
    def reach_joint_error(self) -> float | None:
        """RMS joint-angle difference from the demonstrator until it arrives (deg)."""
        if self._joint_error_count == 0:
            return None
        return float(np.degrees(np.sqrt(self._joint_error_sum / self._joint_error_count)))

    @property
    def phase(self) -> str:
        """What the run is doing: warming up, reaching, holding, or how the hold ended."""
        if self.time < 0.0:
            return "warming up"
        if self.arrival_time is None:
            return "reaching"
        if self.left_goal:
            return "left the goal during the hold window"
        if self.hold_observed < self.hold - 1e-9:
            return "holding"
        return "held for the whole hold window"


class EsnReferenceApp(QMainWindow):
    """Pose the arm, then watch a trained ESN generate the reach from that posture, live.

    Parameters
    ----------
    skeleton : Skeleton
        The robot, posed at the first start posture.
    task : Task
        The reaching task: its target and tolerance (the goal radius).
    esn : ReachingEsn
        The trained ESN.
    hold : float
        Hold duration after arrival (s).
    demonstrator_config : dict, optional
        Skelarm scenario tables (``[skeleton]``, ``[task]``, ``[simulator]``,
        ``[controller]``) of the demonstrator, which is then simulated from each
        run's start posture for comparison.
    speed : float, optional
        Initial playback speed (task seconds per real second).
    name : str, optional
        The model's name, shown in the side panel and the title.
    """

    def __init__(
        self,
        skeleton: Skeleton,
        task: Task,
        esn: ReachingEsn,
        *,
        hold: float,
        demonstrator_config: dict[str, Any] | None = None,
        speed: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if task.tolerance is None:
            msg = "the task's target needs a tolerance, which is the goal radius"
            raise ValueError(msg)
        if len(esn.center) != skeleton.num_joints:
            msg = f"the ESN generates {len(esn.center)} joint angles, but the robot has {skeleton.num_joints} joints"
            raise ValueError(msg)
        self.skeleton = skeleton
        self.task = task
        self.target = task.require_target()
        self.radius = task.tolerance
        self.esn = esn
        self.hold = hold
        self.demonstrator_config = demonstrator_config
        self.run: LiveRun | None = None
        self._last_trail: TrailOverlay | None = None  # the last run's tip path, shown faintly after a reset
        self._pending_steps = 0.0  # fractional ESN steps owed to the playback clock
        self._last_start_q = skeleton.q.copy()

        self.setWindowTitle("ESN reference generator" + (f" - {name}" if name else ""))
        self.resize(1100, 780)
        self.canvas = SkelarmCanvas(skeleton)
        self.canvas.overlay_targets = [(self.target, QColor(task.color), self.radius, True)]
        self.canvas.pose_changed.connect(self._on_posed)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.addWidget(self.canvas, stretch=3)
        panel = QWidget()
        panel.setFixedWidth(_PANEL_WIDTH_PX)
        controls = QVBoxLayout(panel)
        if name:
            model_label = QLabel(f"Model: {name}")
            model_label.setWordWrap(True)
            controls.addWidget(model_label)

        self.time_label = QLabel()
        time_font = self.time_label.font()
        time_font.setPointSize(time_font.pointSize() + 4)
        time_font.setBold(True)
        self.time_label.setFont(time_font)
        controls.addWidget(self.time_label)
        self.phase_label = QLabel()
        self.phase_label.setWordWrap(True)
        controls.addWidget(self.phase_label)

        self.transport_bar = TransportBar(step_label="Next step", reset_label="Back to the start posture")
        self.transport_bar.play_button.toggled.connect(self._on_play_toggled)
        self.transport_bar.step_button.clicked.connect(self.step)
        self.transport_bar.reset_button.clicked.connect(self.reset)
        controls.addWidget(self.transport_bar)
        self.home_shortcut = QShortcut(QKeySequence("Home"), self)
        self.home_shortcut.activated.connect(self.transport_bar.reset_button.click)
        self.quit_shortcut = bind_quit_key(self)

        controls.addWidget(QLabel("Playback speed"))
        self.speed_spin = SpeedSpinBox(speed=speed)
        self.speed_spin.valueChanged.connect(self._on_speed_changed)
        controls.addWidget(self.speed_spin)

        self.target_checkbox = QCheckBox("Show target")
        self.target_checkbox.setChecked(True)
        self.target_checkbox.toggled.connect(self._on_overlays_toggled)
        controls.addWidget(self.target_checkbox)
        self.trail_checkbox = QCheckBox("Show tip paths")
        self.trail_checkbox.setChecked(True)
        self.trail_checkbox.toggled.connect(self._on_overlays_toggled)
        controls.addWidget(self.trail_checkbox)
        self.demonstrator_checkbox = QCheckBox("Show the demonstrator's path")
        self.demonstrator_checkbox.setChecked(True)
        self.demonstrator_checkbox.setVisible(demonstrator_config is not None)
        self.demonstrator_checkbox.toggled.connect(self._on_overlays_toggled)
        controls.addWidget(self.demonstrator_checkbox)
        self.com_checkbox = QCheckBox("Show center of mass")
        self.com_checkbox.toggled.connect(self._on_overlays_toggled)
        controls.addWidget(self.com_checkbox)

        self.metrics_label = QLabel()
        self.metrics_label.setWordWrap(True)
        controls.addWidget(self.metrics_label)
        controls.addStretch()
        layout.addWidget(panel, stretch=1)

        self.clock = PlaybackClock(self, speed=speed)
        self.clock.ticked.connect(self._advance_timeline)
        self._refresh()

    @property
    def is_playing(self) -> bool:
        """Whether the ESN is running."""
        return self.clock.is_running

    @property
    def speed(self) -> float:
        """Playback speed: task seconds per real second."""
        return self.clock.speed

    @speed.setter
    def speed(self, value: float) -> None:
        self.clock.speed = value

    def play(self) -> None:
        """Run the ESN: start a run from the current posture, or resume the paused one."""
        if self.run is None:
            self._start_run()
        self.clock.start()
        self.transport_bar.set_playing(True)

    def pause(self) -> None:
        """Pause the ESN; the run and its reservoir state are kept."""
        self.clock.stop()
        self.transport_bar.set_playing(False)

    def step(self) -> None:
        """Take one ESN step while paused (starting a run if there is none)."""
        if self.is_playing:
            return
        if self.run is None:
            self._start_run()
        assert self.run is not None
        self.run.take_step()
        self._refresh()

    def reset(self) -> None:
        """End the run and return the arm to its start posture, ready to be posed again."""
        self.pause()
        if self.run is not None:
            if len(self.run.hand) > 1:
                self._last_trail = TrailOverlay(np.array(self.run.hand), _LAST_RUN_COLOR)
            self.run = None
        self._set_posture(self._last_start_q)
        self._show_last_trail()
        self._refresh()

    def advance(self, seconds: float) -> None:
        """Advance the run by ``seconds`` of real time, scaled by :attr:`speed`."""
        self._advance_timeline(seconds * self.speed)

    def _advance_timeline(self, seconds: float) -> None:
        """Advance the run by ``seconds`` of task time, as many ESN steps as fit."""
        if self.run is None:
            return
        self._pending_steps += seconds / self.esn.config.dt
        steps = int(self._pending_steps)
        self._pending_steps -= steps
        for _ in range(steps):
            self.run.take_step()
        self._refresh()

    def _start_run(self) -> None:
        start_q = self.skeleton.q.copy()
        self._last_start_q = start_q
        self._pending_steps = 0.0
        self.canvas.clear_ik_target()
        self.run = LiveRun(
            start_q=start_q,
            stream=self.esn.stream(start_q),
            skeleton=self.skeleton,
            target=self.target,
            radius=self.radius,
            hold=self.hold,
            dt=self.esn.config.dt,
            warmup_steps=self.esn.config.warmup_steps,
            demonstrator=self._simulate_demonstrator(start_q),
        )

    def _simulate_demonstrator(self, start_q: NDArray[np.float64]) -> Demonstrator | None:
        """Simulate the demonstrator's reach from ``start_q``, if a demonstrator is configured."""
        if self.demonstrator_config is None:
            return None
        config = {name: self.demonstrator_config[name] for name in _DEMONSTRATOR_TABLES}
        config["demonstrations"] = {"start_q": [np.degrees(start_q).tolist()]}
        log = simulate_reaches(config)[0]
        _, q = resample_joint_angles(log, self.esn.config.dt)
        hand = endpoint_positions(self.skeleton, q)
        arrival = arrival_index(hand, self.target, self.radius)
        self._set_posture(start_q)  # endpoint_positions moved the shared skeleton
        return Demonstrator(q, hand, None if arrival is None else arrival * self.esn.config.dt)

    def _set_posture(self, q: NDArray[np.float64]) -> None:
        """Pose the arm at ``q`` as given, even beyond the joint limits (an ESN is not bounded by them)."""
        for link, angle in zip(self.skeleton.links[1:], q, strict=True):
            link.q = float(angle)
        compute_forward_kinematics(self.skeleton)

    def _refresh(self) -> None:
        """Show the latest posture, the overlays, and the readouts."""
        run = self.run
        self.canvas.drag_to_pose = run is None
        if run is not None and run.q:
            self._set_posture(run.q[-1])
        self._update_overlays()
        self._update_labels()

    def _update_labels(self) -> None:
        run = self.run
        hand = self.skeleton.links[-1]
        hand_xy = np.array([hand.xe, hand.ye])
        distance = float(np.linalg.norm(hand_xy - self.target))
        q_deg = np.degrees(self.skeleton.q)
        lines = [
            f"Joint angles: ({q_deg[0]:.1f}, {q_deg[1]:.1f}) deg"
            if len(q_deg) == 2
            else f"Joint angles: {np.round(q_deg, 1).tolist()} deg",
            f"Hand: ({hand_xy[0]:.3f}, {hand_xy[1]:.3f}) m, {1000 * distance:.1f} mm from the target",
        ]
        if run is None:
            self.time_label.setText("Posing")
            self.phase_label.setText("Drag the arm tip to choose a start posture, then press Play (Space).")
            self.metrics_label.setText("\n".join(lines))
            return
        self.time_label.setText(f"t = {run.time:+.2f} s")
        paused = "" if self.is_playing else " (paused)"
        self.phase_label.setText(f"{run.phase.capitalize()}{paused}. Reset (R) to pose the arm again.")
        if len(run.hand) > 1:
            speed = float(np.linalg.norm(run.hand[-1] - run.hand[-2])) / run.dt
            lines.append(f"Hand speed: {speed:.2f} m/s")
        lines.append("")
        lines.append("Reach")
        first_step = run.first_step
        lines.append(f"  First step: {'-' if first_step is None else f'{1000 * first_step:.1f} mm'}")
        arrival = "not yet" if run.arrival_time is None else f"{run.arrival_time:.2f} s"
        lines.append(f"  Arrival (within {1000 * self.radius:g} mm): {arrival}")
        demonstrator = run.demonstrator
        if demonstrator is not None:
            if demonstrator.arrival_time is not None and run.arrival_time is not None:
                lines.append(f"  Arrival delay: {run.arrival_time - demonstrator.arrival_time:+.2f} s")
            lines.append(f"  Path distance from the demonstrator: {1000 * run.reach_path_distance:.1f} mm")
            joint_error = run.reach_joint_error
            lines.append(f"  Joint error: {'-' if joint_error is None else f'{joint_error:.2f} deg RMS'}")
        lines.append("")
        lines.append(f"Hold ({self.hold:g} s after arrival)")
        if run.arrival_time is None:
            lines.append("  waiting for the arrival")
        else:
            lines.append(f"  Observed: {run.hold_observed:.2f} s of {self.hold:g} s")
            lines.append(f"  Left the goal: {'yes' if run.left_goal else 'no'}")
            assert run.hold_error is not None
            lines.append(f"  Hold error: {1000 * run.hold_error:.1f} mm")
        self.metrics_label.setText("\n".join(lines))

    def _on_posed(self) -> None:
        """Keep the readouts in step with the posture being dragged."""
        self._last_start_q = self.skeleton.q.copy()
        self._update_labels()

    def _update_overlays(self) -> None:
        """Draw the target, the demonstrator's path, and the ESN's path, as the checkboxes choose."""
        trails = []
        run = self.run
        if run is not None and run.demonstrator is not None and self.demonstrator_checkbox.isChecked():
            trails.append(TrailOverlay(run.demonstrator.hand, _DEMONSTRATOR_COLOR, _DEMONSTRATOR_WIDTH_PX))
        if run is not None and self.trail_checkbox.isChecked():
            trails.append(TrailOverlay(np.array(run.hand), _ESN_COLOR))
        self.canvas.trails = trails
        self.canvas.show_overlay_targets = self.target_checkbox.isChecked()
        self.canvas.show_com = self.com_checkbox.isChecked()
        self.canvas.update_skeleton()

    def _show_last_trail(self) -> None:
        # Reassigned only when it changes: the canvas caches static trails until the list is replaced.
        shown = [self._last_trail] if self._last_trail is not None and self.trail_checkbox.isChecked() else []
        if shown != self.canvas.static_trails:
            self.canvas.static_trails = shown

    def _on_overlays_toggled(self) -> None:
        self._show_last_trail()
        self._update_overlays()

    def _on_play_toggled(self, playing: bool) -> None:
        if playing:
            self.play()
        else:
            self.pause()
        self._update_labels()

    def _on_speed_changed(self, value: float) -> None:
        self.clock.speed = value


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Pose a robot arm and watch a trained ESN generate the reach from that posture, live."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="robot and task TOML ([skeleton], [task], optional [initial]); with [controller] and [simulator], "
        "the demonstrator is simulated for comparison",
    )
    parser.add_argument("--model", type=Path, required=True, help="trained ESN: esn.toml in a run directory")
    parser.add_argument("--hold", type=float, default=2.0, help="hold duration after arrival in s (default: 2)")
    parser.add_argument("--speed", type=float, default=1.0, help="initial playback speed (default: 1)")
    parser.add_argument(
        "--pose",
        default=None,
        help="first start posture in degrees, e.g. 18.2,119.9 (default: [initial], else the first demonstrated start)",
    )
    parser.add_argument("--no-demonstrator", action="store_true", help="do not simulate the demonstrator")
    return parser


def main() -> None:
    """Parse the arguments, load the robot, task, and ESN, and run the app."""
    parser = build_parser()
    args = parser.parse_args()
    for path in (args.config, args.model):
        if not path.exists():
            parser.error(f"file not found: {path}")
    with args.config.open("rb") as f:
        config = tomllib.load(f)
    if "task" not in config:
        parser.error(f"{args.config} has no [task] table")
    skeleton = Skeleton.from_toml(args.config)  # posed at [initial], if the file has one
    if args.pose is None and "initial" not in config and "demonstrations" in config:
        skeleton.q = np.radians(config["demonstrations"]["start_q"][0])  # the first demonstrated start
    if args.pose is not None:
        pose = np.radians([float(value) for value in args.pose.split(",")])
        if len(pose) != skeleton.num_joints:
            parser.error(f"--pose has {len(pose)} values but the arm has {skeleton.num_joints} joints")
        skeleton.q = pose
    has_demonstrator = all(name in config for name in _DEMONSTRATOR_TABLES) and not args.no_demonstrator

    app = QApplication(sys.argv)
    try:
        window = EsnReferenceApp(
            skeleton,
            Task.from_dict(config["task"]),
            ReachingEsn.load(args.model),
            hold=args.hold,
            demonstrator_config=config if has_demonstrator else None,
            speed=args.speed,
            name=str(args.model),
        )
    except ValueError as error:
        parser.error(str(error))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
