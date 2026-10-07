# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Interactive dynamics simulation of the robot arm reaching under a reference generator (Stage 2).

Load a robot, its reaching task, and its demonstrator from a demonstration
configuration (such as ``experiments/demonstrations/reach_tvs.toml``). Drag the arm
tip to choose a start posture (inverse kinematics), then press Play: the arm is
simulated in real time with skelarm's dynamics. During a run, running or paused,
dragging pulls the tip toward the cursor with a spring force (the red arrow),
which acts on top of the controller's torque. Reset returns the arm to the start
posture of the last run, ready to be posed again.

The mode is fixed at launch:

- **ESN** (``--model``): a trained ESN, driven by the arm's measured joint angles,
  generates the reference every period, and the tracker (``--law``, and ``--omega``
  or ``--kp`` and ``--kd``) tracks it, as in ``experiments/robot_esn.py``. The ESN's warm-up, with the arm
  holding its start posture, is consumed at once when a run starts, so the run is
  shown from t = 0.
- **Replay a take** (``--replay``): a recorded demonstration, such as a take taught
  by hand, sampled every period (``--period``), and every run replays it from t = 0,
  from wherever the arm was posed, as the replay of ``experiments/robot_esn.py``
  does. The configuration needs no demonstrator.
- **Replay from the start posture** (neither a model, a given posture, nor
  ``--demonstrator``): when a run starts, the demonstrator's reach is simulated
  from the arm's posture and the tracker tracks it, replayed by time.
- **Replay from a given posture** (``--pose``, or an ``[initial]`` table in the
  configuration): the demonstrator's reach is simulated once, from the given
  posture, and every run replays it from t = 0, from wherever the arm was posed.
  Posing the arm away from the given posture emulates an initial offset.
- **Demonstrator** (``--demonstrator``): the demonstrator's own controller drives
  the arm, without a reference.

With ``--omega``, ``--damping`` sets the damping ratio of the tracking error (1,
critically damped, by default). ``--zero-reference-velocity`` gives the tracking law
a zero reference velocity, so that its derivative term damps the arm's own velocity
rather than the velocity error.

External forces never act during the warm-up. The faint gray arm is the initial
posture: the given posture, or else the start posture of the last run. When the
arm tracks a reference, a faint colored arm shows the reference posture.

Keys: ``Space`` play/pause, ``Right``/``F`` one step (10 ms) while paused, ``R`` or
``Home`` reset, ``Q`` quit.

Usage::

    uv run python tools/robot_app.py experiments/demonstrations/reach_tvs.toml \\
        --law computed_torque --omega 10 --model <run directory>/esn.toml
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtGui import QCloseEvent, QColor, QKeySequence, QMouseEvent, QShortcut
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
    Controller,
    PlaybackClock,
    ShortcutFriendlySpinBox,
    SkelarmCanvas,
    Skeleton,
    SpeedSpinBox,
    Task,
    TransportBar,
    bind_quit_key,
    compute_forward_kinematics,
    compute_inverse_kinematics,
    compute_jacobian,
    integrate_with_limits,
    scenario_from_config,
)
from skelarm.canvas import TrailOverlay
from skelarm.simulator import SimulatorCanvas

from arm_esn_ctrl.demonstrations import SCENARIO_TABLES, load_joint_angles, resample_joint_angles, simulate_reaches
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.tracking import (
    LAWS,
    EsnSource,
    ReferenceTracker,
    ReplaySource,
    TrackerConfig,
    error_dynamics,
    tracking_gains,
)

_PANEL_WIDTH_PX = 360
_STEP_SECONDS = 0.01  # how far one press of Step advances a paused run (s)
_MAX_STEPS_PER_TICK = 200  # simulation steps per clock tick at most, so a slow tick cannot snowball
_DEFAULT_PERIOD = 0.01  # period of the replayed reference without a model (s): the ESN period of Stage 2
_DEFAULT_ACCELERATION_FILTER = 0.02  # as in experiments/multi_demonstration_robot_tracking/*.toml (s)
_DEFAULT_STIFFNESS = 20.0  # N/m for the mouse drag, as in skelarm's controlled simulators
_LAW_ALIASES = {"ct": "computed_torque"}  # short names of the tracking laws on the command line
_GHOST_WIDTH_PX = 9.0
_INITIAL_COLOR = QColor(82, 81, 78, 60)
_MODE_COLORS = {  # the reference's color in each mode, as in experiments/robot_esn.py
    "esn": QColor("#2a78d6"),
    "replay_from_start": QColor("#eb6834"),
    "replay_from_given": QColor("#eb6834"),
    "replay_take": QColor("#eb6834"),
    "demonstrator": QColor("#52514e"),
}
_LAST_RUN_COLOR = QColor(82, 81, 78, 110)


class RobotCanvas(SimulatorCanvas):
    """The arm canvas: a drag poses the arm while :attr:`posing`, and pulls its tip with a spring otherwise."""

    def __init__(self, skeleton: Skeleton) -> None:
        super().__init__(skeleton)
        self.posing = True

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        if self.posing:
            SkelarmCanvas.mousePressEvent(self, a0)
        else:
            super().mousePressEvent(a0)

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        if self.posing:
            SkelarmCanvas.mouseMoveEvent(self, a0)
        else:
            super().mouseMoveEvent(a0)


@dataclass
class RobotRun:
    """One simulated run, measured as it goes.

    Times are on the task clock: the warm-up runs at negative times and the task
    starts at time 0. The metrics cover the task only.
    """

    start_q: NDArray[np.float64]
    controller: Controller
    tracker: ReferenceTracker | None  # the controller, when it tracks a reference
    warmup: float  # s
    warmup_steps: int  # simulation steps of the warm-up
    target: NDArray[np.float64]
    radius: float  # goal radius: the target tolerance (m)
    hold: float  # hold duration T_h after arrival (s)
    steps: int = 0  # simulation steps taken
    sim_time: float = 0.0  # skelarm's clock, from the start of the warm-up (s)
    hand_path: list[NDArray[np.float64]] = field(default_factory=list)
    hand_speed: float = 0.0  # m/s
    arrival_time: float | None = None
    left_goal: bool = False
    hold_error: float | None = None  # distance to the target at the latest sample of the hold window (m)
    hold_observed: float = 0.0  # how much of the hold window has passed (s)
    tracking_error: NDArray[np.float64] | None = None  # q_ref - q now (rad)
    _tracking_square_sum: float = 0.0
    _tracking_count: int = 0
    reference_speed: float = 0.0  # fastest joint speed of the reference now (rad/s)
    peak_reference_speed: float = 0.0
    torque: NDArray[np.float64] | None = None  # the controller's joint torque now (N m)
    peak_torque: float = 0.0  # largest joint torque (N m)
    effort: float = 0.0  # integral of the squared joint torques (N^2 m^2 s)
    force: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))  # external tip force now (N)
    peak_force: float = 0.0

    @property
    def time(self) -> float:
        """Time on the task clock (s)."""
        return self.sim_time - self.warmup

    @property
    def warming_up(self) -> bool:
        return self.steps < self.warmup_steps

    def record(
        self,
        dt: float,
        q: NDArray[np.float64],
        hand: NDArray[np.float64],
        torque: NDArray[np.float64],
        force: NDArray[np.float64],
    ) -> None:
        """Measure the step about to be taken from the current state: joint angles ``q`` and hand position."""
        time = self.time
        if self.hand_path:
            self.hand_speed = float(np.linalg.norm(hand - self.hand_path[-1])) / dt
        self.hand_path.append(hand)
        self.torque, self.force = torque, force
        if self.warming_up:
            return
        distance = float(np.linalg.norm(hand - self.target))
        if self.arrival_time is None and distance <= self.radius:
            self.arrival_time = time
        if self.arrival_time is not None and time <= self.arrival_time + self.hold + 1e-9:
            self.left_goal = self.left_goal or distance > self.radius
            self.hold_error = distance
            self.hold_observed = time - self.arrival_time
        self.peak_torque = max(self.peak_torque, float(np.abs(torque).max()))
        self.effort += float(np.sum(torque**2)) * dt
        self.peak_force = max(self.peak_force, float(np.linalg.norm(force)))
        if self.tracker is not None:
            reference = self.tracker.reference
            self.tracking_error = reference.q - q
            self._tracking_square_sum += float(np.sum(self.tracking_error**2))
            self._tracking_count += self.tracking_error.size
            self.reference_speed = float(np.abs(reference.dq).max())
            self.peak_reference_speed = max(self.peak_reference_speed, self.reference_speed)

    @property
    def tracking_rms(self) -> float | None:
        """RMS tracking error over the task so far (rad)."""
        if self._tracking_count == 0:
            return None
        return float(np.sqrt(self._tracking_square_sum / self._tracking_count))

    @property
    def phase(self) -> str:
        """What the run is doing: warming up, reaching, holding, or how the hold ended."""
        if self.warming_up:
            return "warming up"
        if self.arrival_time is None:
            return "reaching"
        if self.left_goal:
            return "left the goal during the hold window"
        if self.hold_observed < self.hold - 1e-9:
            return "holding"
        return "held for the whole hold window"


class RobotApp(QMainWindow):
    """Pose the arm, then simulate it reaching under the chosen reference generator, and push it.

    Parameters
    ----------
    skeleton : Skeleton
        The robot, posed at its first start posture.
    config : dict
        The demonstration configuration: skelarm's ``[skeleton]``, ``[task]``,
        ``[simulator]``, and ``[controller]`` tables (the demonstrator).
    tracker : TrackerConfig, optional
        The tracking law and natural frequency; required unless ``demonstrator``.
    gains : tuple of NDArray[np.float64], optional
        The tracker's per-joint gains ``(kp, kd)``, given directly instead of by the
        tracker's natural frequency.
    esn : ReachingEsn, optional
        A trained ESN to generate the reference (the ESN mode).
    given_q : NDArray[np.float64], optional
        A given initial posture (rad): the demonstrator's reach is simulated from it
        and replayed (unless an ESN or a take is given), and it is drawn as the initial
        posture.
    take : NDArray[np.float64], optional
        A recorded demonstration's joint angles (rad), sampled every ``period``, to
        replay by time in every run (the take mode).
    demonstrator : bool, optional
        Drive the arm with the demonstrator's own controller, without a reference.
    period : float, optional
        Period of the replayed reference (s); the ESN uses its own.
    hold : float, optional
        Hold duration after arrival (s).
    stiffness : float, optional
        Force per meter of tip-to-cursor distance for the drag (N/m).
    speed : float, optional
        Initial playback speed (task seconds per real second).
    name : str, optional
        The model's or the take's name, shown in the side panel and the title.
    """

    def __init__(
        self,
        skeleton: Skeleton,
        config: dict[str, Any],
        *,
        tracker: TrackerConfig | None = None,
        gains: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
        esn: ReachingEsn | None = None,
        given_q: NDArray[np.float64] | None = None,
        take: NDArray[np.float64] | None = None,
        demonstrator: bool = False,
        period: float = _DEFAULT_PERIOD,
        hold: float = 2.0,
        stiffness: float = _DEFAULT_STIFFNESS,
        speed: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__()
        task = Task.from_dict(config["task"])
        if task.tolerance is None:
            msg = "the task's target needs a tolerance, which is the goal radius"
            raise ValueError(msg)
        if esn is not None and demonstrator:
            msg = "an ESN and the demonstrator's own controller cannot both drive the arm"
            raise ValueError(msg)
        if take is not None and (esn is not None or demonstrator):
            msg = "a take is replayed only without an ESN or the demonstrator's own controller"
            raise ValueError(msg)
        if (tracker is None) != demonstrator:
            msg = "a tracker is needed exactly when the arm tracks a reference (not with the demonstrator)"
            raise ValueError(msg)
        if esn is not None and len(esn.center) != skeleton.num_joints:
            msg = f"the ESN generates {len(esn.center)} joint angles, but the robot has {skeleton.num_joints} joints"
            raise ValueError(msg)
        if esn is None and take is None and any(name not in config for name in SCENARIO_TABLES):
            msg = (
                "without an ESN or a take, the configuration needs the demonstrator's tables:"
                f" {', '.join(SCENARIO_TABLES)}"
            )
            raise ValueError(msg)
        if take is not None and take.shape[1:] != (skeleton.num_joints,):
            msg = f"the take has joint angles of shape {take.shape[1:]}, but the robot has {skeleton.num_joints} joints"
            raise ValueError(msg)
        self.skeleton = skeleton
        self.config = config
        self.target = task.require_target()
        self.radius = task.tolerance
        self.tracker_config = tracker
        self.esn = esn
        self.given_q = None if given_q is None else np.asarray(given_q, dtype=np.float64).copy()
        self.period = esn.config.dt if esn is not None else period
        self.hold = hold
        self.dt = float(config["simulator"]["dt"])
        self.enforce_limits = bool(config["simulator"].get("enforce_limits", True))
        self._lower = np.array([link.prop.qmin for link in skeleton.links[1:]])
        self._upper = np.array([link.prop.qmax for link in skeleton.links[1:]])
        if esn is not None:
            self.mode = "esn"
        elif demonstrator:
            self.mode = "demonstrator"
        elif take is not None:
            self.mode = "replay_take"
        elif self.given_q is not None:
            self.mode = "replay_from_given"
        else:
            self.mode = "replay_from_start"
        self.gains = None
        if tracker is not None:
            self.gains = tracking_gains(tracker, skeleton, self._end_posture()) if gains is None else gains
            kp, kd = self.gains
            if kp.shape != (skeleton.num_joints,) or kd.shape != (skeleton.num_joints,):
                msg = f"the tracker needs one kp and one kd per joint ({skeleton.num_joints})"
                raise ValueError(msg)
            if np.any(kp <= 0.0) or np.any(kd < 0.0):
                msg = "the tracker's kp must be positive and its kd not negative"
                raise ValueError(msg)
        # The take, or the given posture's reach, simulated once, is replayed in every run; replays from the start
        # posture are simulated at every run.
        self._given_reference: NDArray[np.float64] | None = None
        if self.mode == "replay_take":
            assert take is not None
            self._given_reference = np.asarray(take, dtype=np.float64).copy()
        if self.mode == "replay_from_given":
            assert self.given_q is not None
            self._given_reference = self._demonstrator_reach(self.given_q)
        self.run: RobotRun | None = None
        self._previous_start: NDArray[np.float64] | None = None  # the start posture of the last run
        self._last_trail: TrailOverlay | None = None  # the last run's tip path, shown faintly after a reset
        self._pending_steps = 0.0  # fractional simulation steps owed to the playback clock
        self._ghost = skeleton.clone()  # a copy of the robot to place the faint postures

        self.setWindowTitle("Robot reaching" + (f" - {name}" if name else ""))
        self.resize(1150, 800)
        self.canvas = RobotCanvas(skeleton)
        self.canvas.overlay_targets = [(self.target, QColor(task.color), self.radius, True)]
        self.canvas.pose_changed.connect(self._on_posed)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.addWidget(self.canvas, stretch=3)
        panel = QWidget()
        panel.setFixedWidth(_PANEL_WIDTH_PX)
        controls = QVBoxLayout(panel)
        self.mode_label = QLabel(self._mode_text(name))
        self.mode_label.setWordWrap(True)
        controls.addWidget(self.mode_label)

        self.time_label = QLabel()
        time_font = self.time_label.font()
        time_font.setPointSize(time_font.pointSize() + 4)
        time_font.setBold(True)
        self.time_label.setFont(time_font)
        controls.addWidget(self.time_label)
        self.phase_label = QLabel()
        self.phase_label.setWordWrap(True)
        controls.addWidget(self.phase_label)

        self.transport_bar = TransportBar(step_label="Next 10 ms", reset_label="Back to the start posture")
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
        controls.addWidget(QLabel("Drag stiffness (N/m)"))
        self.stiffness_spin = ShortcutFriendlySpinBox()  # leaves Space and the letters to the shortcuts
        self.stiffness_spin.setToolTip("The drag pulls the tip with this force per meter from the tip to the cursor")
        self.stiffness_spin.setRange(0.0, 10000.0)
        self.stiffness_spin.setDecimals(1)
        self.stiffness_spin.setSingleStep(5.0)
        self.stiffness_spin.setValue(stiffness)
        controls.addWidget(self.stiffness_spin)

        self.target_checkbox = QCheckBox("Show target")
        self.target_checkbox.setChecked(True)
        self.trail_checkbox = QCheckBox("Show tip paths")
        self.trail_checkbox.setChecked(True)
        self.initial_checkbox = QCheckBox("Show the initial posture")
        self.initial_checkbox.setChecked(True)
        self.reference_checkbox = QCheckBox("Show the reference posture")
        self.reference_checkbox.setChecked(True)
        self.reference_checkbox.setVisible(self.mode != "demonstrator")
        self.com_checkbox = QCheckBox("Show center of mass")
        for checkbox in (
            self.target_checkbox,
            self.trail_checkbox,
            self.initial_checkbox,
            self.reference_checkbox,
            self.com_checkbox,
        ):
            checkbox.toggled.connect(self._on_overlays_toggled)
            controls.addWidget(checkbox)

        self.metrics_label = QLabel()
        self.metrics_label.setWordWrap(True)
        controls.addWidget(self.metrics_label)
        controls.addStretch()
        layout.addWidget(panel, stretch=1)

        self.clock = PlaybackClock(self, speed=speed)
        self.clock.ticked.connect(self._advance_timeline)
        self._show_static_overlays()
        self._refresh()

    @property
    def is_playing(self) -> bool:
        """Whether the simulation is running."""
        return self.clock.is_running

    @property
    def speed(self) -> float:
        """Playback speed: task seconds per real second."""
        return self.clock.speed

    @property
    def stiffness(self) -> float:
        """Force per meter of tip-to-cursor distance for the drag (N/m)."""
        return self.stiffness_spin.value()

    @property
    def initial_posture(self) -> NDArray[np.float64] | None:
        """The posture drawn faintly: the given one, or else the last run's start (None before any run)."""
        if self.given_q is not None:
            return self.given_q
        if self.run is not None:
            return self.run.start_q
        return self._previous_start

    def play(self) -> None:
        """Run the simulation: start a run from the current posture, or resume the paused one."""
        if self.run is None:
            self._start_run()
        self.clock.start()
        self.transport_bar.set_playing(True)
        self._refresh()

    def pause(self) -> None:
        """Pause the simulation; the run is kept, and a drag still shows the force it will apply."""
        self.clock.stop()
        self.transport_bar.set_playing(False)
        self._refresh()

    def step(self) -> None:
        """Advance a paused run by 10 ms; without a run, start one and show time 0."""
        if self.is_playing:
            return
        if self.run is None:
            self._start_run()
        else:
            self._simulate(round(_STEP_SECONDS / self.dt))
        self._refresh()

    def reset(self) -> None:
        """End the run and return the arm to its start posture, at rest, ready to be posed again."""
        self.pause()
        if self.run is not None:
            if len(self.run.hand_path) > 1:
                self._last_trail = TrailOverlay(np.array(self.run.hand_path), _LAST_RUN_COLOR)
            self._previous_start = self.run.start_q
            self.run = None
            self._set_state(self._previous_start)
        self.canvas.drag_point = None
        self._show_static_overlays()
        self._refresh()

    def advance(self, seconds: float) -> None:
        """Advance the run by ``seconds`` of real time, scaled by :attr:`speed`."""
        self._advance_timeline(seconds * self.speed)

    def _advance_timeline(self, seconds: float) -> None:
        """Advance the run by ``seconds`` of task time, as many simulation steps as fit."""
        if self.run is None:
            return
        self._pending_steps += seconds / self.dt
        steps = min(int(self._pending_steps), _MAX_STEPS_PER_TICK)
        self._pending_steps = min(self._pending_steps - steps, 1.0)
        self._simulate(steps)
        self._refresh()

    def _start_run(self) -> None:
        """Start a run from the arm's posture, at rest, and consume the warm-up."""
        start_q = self.skeleton.q.copy()
        self._set_state(start_q)
        self.canvas.clear_ik_target()
        self._pending_steps = 0.0
        tracker = None
        warmup_steps = 0
        if self.mode == "demonstrator":
            scenario = {name: self.config[name] for name in SCENARIO_TABLES}
            scenario["initial"] = {"q": np.degrees(start_q).tolist()}
            controller: Controller = scenario_from_config(scenario).controller
        else:
            assert self.tracker_config is not None and self.gains is not None
            if self.esn is not None:
                source: EsnSource | ReplaySource = EsnSource(self.esn)
                warmup_steps = self.esn.config.warmup_steps
            elif self._given_reference is not None:
                source = ReplaySource(self._given_reference)
            else:
                source = ReplaySource(self._demonstrator_reach(start_q))
            tracker = ReferenceTracker(
                source, self.tracker_config, *self.gains, period=self.period, warmup_steps=warmup_steps
            )
            controller = tracker
        controller.reset(self.skeleton)
        warmup = warmup_steps * self.period
        self.run = RobotRun(
            start_q=start_q,
            controller=controller,
            tracker=tracker,
            warmup=warmup,
            warmup_steps=round(warmup / self.dt),
            target=self.target,
            radius=self.radius,
            hold=self.hold,
        )
        self._simulate(self.run.warmup_steps)  # the warm-up, at once: the run is shown from time 0
        self._show_static_overlays()

    def _simulate(self, steps: int) -> None:
        """Take simulation steps of the run, as skelarm's simulate_controlled does."""
        run = self.run
        assert run is not None
        for _ in range(steps):
            t = run.sim_time
            run.controller.update(t, self.skeleton, self.dt)
            torque = run.controller.control(t, self.skeleton)
            force = np.zeros(2) if run.warming_up else self.canvas.external_force(self.stiffness)
            tip = self.skeleton.links[-1]
            run.record(self.dt, self.skeleton.q.copy(), np.array([tip.xe, tip.ye]), torque, force)
            applied = torque + compute_jacobian(self.skeleton).T @ force
            if self.enforce_limits:
                integrate_with_limits(self.skeleton, applied, self.dt, self._lower, self._upper)
            else:
                integrate_with_limits(self.skeleton, applied, self.dt)
            run.sim_time += self.dt
            run.steps += 1

    def _demonstrator_reach(self, start_q: NDArray[np.float64]) -> NDArray[np.float64]:
        """The demonstrator's reach from ``start_q``, sampled every reference period."""
        config = {name: self.config[name] for name in SCENARIO_TABLES}
        config["demonstrations"] = {"start_q": [np.degrees(start_q).tolist()]}
        return resample_joint_angles(simulate_reaches(config)[0], self.period)[1]

    def _end_posture(self) -> NDArray[np.float64]:
        """The posture with the hand at the target, near the first start posture, for joint PD's gains."""
        model = self.skeleton.clone()
        return compute_inverse_kinematics(model, tuple(self.target)).q

    def _set_state(self, q: NDArray[np.float64]) -> None:
        """Place the arm at ``q``, at rest, as given (an arm beyond its limits is shown where it is)."""
        for link, angle in zip(self.skeleton.links[1:], q, strict=True):
            link.q = float(angle)
            link.dq = 0.0
        compute_forward_kinematics(self.skeleton)

    def _arm_points(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """The base and every joint and the tip of the arm at ``q``, for drawing a faint arm."""
        for link, angle in zip(self._ghost.links[1:], q, strict=True):
            link.q = float(angle)
        compute_forward_kinematics(self._ghost)
        return np.array([[0.0, 0.0]] + [[link.xe, link.ye] for link in self._ghost.links])

    def _mode_text(self, name: str | None) -> str:
        if self.mode == "esn":
            assert self.esn is not None
            warmup = self.esn.config.warmup
            reference = (
                f"the ESN{f' {name}' if name else ''}, driven by the measured joint angles"
                f" (its {warmup:g} s warm-up is consumed at once)"
            )
        elif self.mode == "replay_from_start":
            reference = "the demonstrator's reach from each run's start posture, replayed by time"
        elif self.mode == "replay_take":
            reference = (
                f"the take{f' {name}' if name else ''}, replayed by time from t = 0;"
                " posing the arm away from its start emulates an initial offset"
            )
        elif self.mode == "replay_from_given":
            assert self.given_q is not None
            given = ", ".join(f"{angle:.1f}" for angle in np.degrees(self.given_q))
            reference = (
                f"the demonstrator's reach from the given posture ({given}) deg, replayed by time;"
                " posing the arm elsewhere emulates an initial offset"
            )
        else:
            reference = "none"
        if self.tracker_config is None:
            controller = f"the demonstrator's own ({self.config['controller']['type'].replace('_', ' ')})"
        else:
            assert self.gains is not None
            law = {"computed_torque": "computed torque", "pd": "joint PD"}[self.tracker_config.law]
            kp = ", ".join(f"{value:.4g}" for value in self.gains[0])
            kd = ", ".join(f"{value:.4g}" for value in self.gains[1])
            omega = self.tracker_config.omega
            controller = f"{law}, " + ("" if omega is None else f"ω = {omega:g} rad/s, ") + f"kp = {kp}; kd = {kd}"
            natural, damping = error_dynamics(self.tracker_config.law, self.gains, self.skeleton, self._end_posture())
            controller += (
                f"\nTracking error: natural frequency {', '.join(f'{value:.3g}' for value in natural)} rad/s,"
                f" damping ratio {', '.join(f'{value:.2f}' for value in damping)}"
                + (" (at the target posture)" if self.tracker_config.law == "pd" else "")
            )
            if not self.tracker_config.reference_velocity:
                controller += "\nReference velocity: zero (the derivative term damps the arm's own velocity)"
        return f"Reference: {reference}\nController: {controller}"

    def _refresh(self) -> None:
        """Show the overlays and the readouts of the current state."""
        self.canvas.posing = self.run is None
        self._update_overlays()
        self._update_labels()

    def _update_labels(self) -> None:
        run = self.run
        hand = self.skeleton.links[-1]
        hand_xy = np.array([hand.xe, hand.ye])
        distance = float(np.linalg.norm(hand_xy - self.target))
        q_deg = np.degrees(self.skeleton.q)
        lines = [
            f"Joint angles: {', '.join(f'{angle:.1f}' for angle in q_deg)} deg",
            f"Hand: ({hand_xy[0]:.3f}, {hand_xy[1]:.3f}) m, {1000 * distance:.1f} mm from the target",
        ]
        if run is None:
            self.time_label.setText("Posing")
            self.phase_label.setText("Drag the arm tip to choose a start posture, then press Play (Space).")
            self.metrics_label.setText("\n".join(lines))
            return
        self.time_label.setText(f"t = {run.time:+.2f} s")
        paused = "" if self.is_playing else " (paused)"
        self.phase_label.setText(f"{run.phase.capitalize()}{paused}. Drag to push the arm; Reset (R) to pose it again.")
        lines.append(f"Hand speed: {run.hand_speed:.2f} m/s")
        lines.append("")
        arrival = "not yet" if run.arrival_time is None else f"{run.arrival_time:.2f} s"
        lines.append(f"Arrival (within {1000 * self.radius:g} mm): {arrival}")
        lines.append(f"Hold ({self.hold:g} s after arrival)")
        if run.arrival_time is None:
            lines.append("  waiting for the arrival")
        else:
            assert run.hold_error is not None
            lines.append(f"  Observed: {run.hold_observed:.2f} s of {self.hold:g} s")
            lines.append(f"  Left the goal: {'yes' if run.left_goal else 'no'}")
            lines.append(f"  Hold error: {1000 * run.hold_error:.1f} mm")
        if run.tracker is not None:
            lines.append("")
            lines.append("Tracking")
            if run.tracking_error is not None and run.tracking_rms is not None:
                now = float(np.degrees(np.abs(run.tracking_error).max()))
                lines.append(f"  Error: {now:.2f} deg now, {np.degrees(run.tracking_rms):.2f} deg RMS")
            reference_speed = np.degrees([run.reference_speed, run.peak_reference_speed])
            lines.append(f"  Reference joint speed: {reference_speed[0]:.0f} deg/s now, peak {reference_speed[1]:.0f}")
        lines.append("")
        lines.append("Effort")
        if run.torque is not None:
            torque = ", ".join(f"{value:+.1f}" for value in run.torque)
            lines.append(f"  Joint torque: ({torque}) N m now, peak {run.peak_torque:.1f}")
        lines.append(f"  Integral of squared torque: {run.effort:.2f} N² m² s")
        lines.append(f"  External force: {np.linalg.norm(run.force):.1f} N now, peak {run.peak_force:.1f}")
        self.metrics_label.setText("\n".join(lines))

    def _update_overlays(self) -> None:
        """Draw the reference posture and the tip path of the run, as the checkboxes choose."""
        trails = []
        run = self.run
        color = _MODE_COLORS[self.mode]
        if run is not None and run.tracker is not None and self.reference_checkbox.isChecked():
            ghost = QColor(color)
            ghost.setAlpha(70)
            trails.append(TrailOverlay(self._arm_points(run.tracker.reference.q), ghost, _GHOST_WIDTH_PX))
        if run is not None and len(run.hand_path) > 1 and self.trail_checkbox.isChecked():
            trails.append(TrailOverlay(np.array(run.hand_path[:: max(1, round(self.period / self.dt))]), color))
        self.canvas.trails = trails
        self.canvas.show_overlay_targets = self.target_checkbox.isChecked()
        self.canvas.show_com = self.com_checkbox.isChecked()
        self.canvas.update_skeleton()

    def _show_static_overlays(self) -> None:
        """Draw the initial posture and the last run's tip path, which change only between runs."""
        shown = []
        initial = self.initial_posture
        if initial is not None and self.initial_checkbox.isChecked():
            shown.append(TrailOverlay(self._arm_points(initial), _INITIAL_COLOR, _GHOST_WIDTH_PX))
        if self.run is None and self._last_trail is not None and self.trail_checkbox.isChecked():
            shown.append(self._last_trail)
        # Reassigned only when it changes: the canvas caches static trails until the list is replaced.
        if shown != self.canvas.static_trails:
            self.canvas.static_trails = shown

    def _on_posed(self) -> None:
        self._update_labels()

    def _on_overlays_toggled(self) -> None:
        self._show_static_overlays()
        self._update_overlays()

    def _on_play_toggled(self, playing: bool) -> None:
        if playing:
            self.play()
        else:
            self.pause()

    def _on_speed_changed(self, value: float) -> None:
        self.clock.speed = value

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Stop the clock along with the window."""
        self.clock.stop()
        super().closeEvent(a0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "config",
        type=Path,
        help="demonstration configuration: the robot, the task, and the demonstrator (skelarm's scenario tables)",
    )
    parser.add_argument(
        "--law",
        type=lambda name: _LAW_ALIASES.get(name, name),
        choices=sorted(LAWS),
        help="tracking law, or ct for computed_torque (required unless --demonstrator)",
    )
    parser.add_argument(
        "--omega",
        type=float,
        help="natural frequency of the tracking error (rad/s), critically damped; or give --kp and --kd",
    )
    parser.add_argument(
        "--damping",
        type=float,
        help="damping ratio of the tracking error with --omega (default 1, critically damped; less oscillates)",
    )
    parser.add_argument(
        "--zero-reference-velocity",
        action="store_true",
        help="give the tracking law a zero reference velocity: its derivative term damps the arm's own velocity",
    )
    parser.add_argument(
        "--kp", help="the tracker's proportional gain instead of --omega: one value, or one per joint (comma-separated)"
    )
    parser.add_argument(
        "--kd", help="the tracker's derivative gain instead of --omega: one value, or one per joint (comma-separated)"
    )
    parser.add_argument("--model", type=Path, help="a trained ESN (esn.toml) to generate the reference")
    parser.add_argument(
        "--demonstrator", action="store_true", help="drive the arm with the demonstrator's own controller instead"
    )
    parser.add_argument(
        "--replay", type=Path, help="a recorded take (.sklog.npz) to replay by time, such as a take taught by hand"
    )
    parser.add_argument("--pose", help="a given initial posture: joint angles in degrees, such as 29.4,88.2")
    parser.add_argument(
        "--period",
        type=float,
        help=f"period of the replayed reference or take without a model (s; default {_DEFAULT_PERIOD:g})",
    )
    parser.add_argument(
        "--acceleration-filter",
        type=float,
        default=_DEFAULT_ACCELERATION_FILTER,
        help="time constant of the low-pass filter on the reference acceleration (s)",
    )
    parser.add_argument("--hold", type=float, default=2.0, help="hold duration after arrival (s)")
    parser.add_argument(
        "--stiffness",
        type=float,
        default=_DEFAULT_STIFFNESS,
        help="drag force per meter of tip-to-cursor distance (N/m)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="initial playback speed")
    return parser


def check_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject combinations of arguments that contradict each other."""
    if args.model is not None and args.demonstrator:
        parser.error("--model and --demonstrator exclude each other: either the ESN or the demonstrator drives the arm")
    if args.replay is not None and (args.model is not None or args.demonstrator):
        parser.error("--replay excludes --model and --demonstrator: the take is the reference")
    gains = (args.omega, args.damping, args.kp, args.kd)
    if args.demonstrator and (
        args.law is not None or any(value is not None for value in gains) or args.zero_reference_velocity
    ):
        parser.error(
            "--law, --omega, --damping, --kp, --kd, and --zero-reference-velocity set the tracker,"
            " which the demonstrator's own controller does not use"
        )
    if args.demonstrator:
        return
    if args.law is None:
        parser.error("--law is required unless --demonstrator")
    if (args.kp is None) != (args.kd is None):
        parser.error("--kp and --kd go together")
    if (args.omega is None) == (args.kp is None):
        parser.error("give the tracker either --omega or --kp and --kd (required unless --demonstrator)")
    if args.damping is not None and args.omega is None:
        parser.error("--damping sets the damping ratio with --omega; with --kp and --kd, the gains set it")
    if args.period is not None and (args.model is not None or args.demonstrator):
        parser.error(
            "--period sets the replayed reference's period, which is used only without --model or --demonstrator"
        )


def joint_values(parser: argparse.ArgumentParser, name: str, text: str, n_joints: int) -> NDArray[np.float64]:
    """One value per joint from ``text``: a single value for every joint, or one per joint, comma-separated."""
    try:
        values = np.array([float(value) for value in text.split(",")])
    except ValueError:
        parser.error(f"{name} takes numbers, such as 100 or 100,20")
    if len(values) == 1:
        return np.full(n_joints, values[0])
    if len(values) != n_joints:
        parser.error(f"{name} has {len(values)} values but the arm has {n_joints} joints")
    return values


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    check_arguments(parser, args)
    with args.config.open("rb") as f:
        config = tomllib.load(f)
    skeleton = Skeleton.from_toml(args.config)  # posed at [initial], if the file has one
    given_q = skeleton.q.copy() if "initial" in config else None
    if args.pose is not None:
        given_q = np.radians([float(value) for value in args.pose.split(",")])
        if len(given_q) != skeleton.num_joints:
            parser.error(f"--pose has {len(given_q)} values but the arm has {skeleton.num_joints} joints")
    if given_q is not None:
        skeleton.q = given_q
    elif "demonstrations" in config:
        skeleton.q = np.radians(config["demonstrations"]["start_q"][0])  # a reachable, non-singular posture
    esn = None if args.model is None else ReachingEsn.load(args.model)
    period = _DEFAULT_PERIOD if args.period is None else args.period
    take = None if args.replay is None else load_joint_angles(args.replay, period)[1]
    tracker = None
    gains = None
    if not args.demonstrator:
        tracker = TrackerConfig(
            args.law,
            args.omega,
            args.acceleration_filter,
            damping=1.0 if args.damping is None else args.damping,
            reference_velocity=not args.zero_reference_velocity,
        )
    if args.kp is not None:
        gains = (
            joint_values(parser, "--kp", args.kp, skeleton.num_joints),
            joint_values(parser, "--kd", args.kd, skeleton.num_joints),
        )

    app = QApplication(sys.argv)
    try:
        window = RobotApp(
            skeleton,
            config,
            tracker=tracker,
            gains=gains,
            esn=esn,
            given_q=given_q,
            take=take,
            demonstrator=args.demonstrator,
            period=period,
            hold=args.hold,
            stiffness=args.stiffness,
            speed=args.speed,
            name=args.model.parent.name
            if args.model is not None
            else None
            if args.replay is None
            else args.replay.name,
        )
    except ValueError as error:
        parser.error(str(error))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
