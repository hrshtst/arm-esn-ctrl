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
critically damped, by default). Joint PD's gains are set at the posture where the
reference's demonstrations end, as the experiments set them: the takes the ESN was
trained on, the replayed take, or the demonstrator's reaches. ``--zero-reference-velocity`` gives the tracking law
a zero reference velocity, so that its derivative term damps the arm's own velocity
rather than the velocity error. ``--control-period`` computes the tracker's torque
only that often and holds it between, as ``control_period`` in
``experiments/robot_esn.py``; the arm is still simulated every step.

External forces never act during the warm-up. The faint gray arm is the initial
posture: the given posture, or else the start posture of the last run. When the
arm tracks a reference, a faint colored arm shows the reference posture.

**Save log** (``Ctrl+S``) saves the run so far as a log in the format of
``experiments/robot_esn.py``'s runs: on the task clock, the warm-up at negative
times, every simulation step's joint angles ``q`` and velocities ``dq``, the
controller's torque ``tau``, the external tip force ``ext_force``, and, when the arm
tracks a reference, ``q_ref`` and ``error``; then a final frame, as skelarm ends a
run. Its metadata name the reference, the tracker, the start posture, the
simulation step, and the commit. It is saved by default under ``data/recorded_forces/``
in the storage root, where experiments can share it: a ``[disturbance]`` of
``type = "recorded"`` replays its force (see :mod:`arm_esn_ctrl.disturbances`).

**Replay recorded forces** (``--forces``): every run replays a saved log's external
force by time, on the task clock, instead of the drag, from the log's start
posture; posing and dragging are off, so that every run meets the same forces.
The replayed force is drawn as an arrow, scaled as a drag of the set stiffness
would draw it.

Keys: ``Space`` play/pause, ``Right``/``F`` one step (10 ms) while paused, ``R`` or
``Home`` reset, ``Ctrl+S`` save the run's log, ``Q`` quit.

Usage::

    uv run python tools/robot_app.py experiments/demonstrations/reach_tvs.toml \\
        --law computed_torque --omega 10 --model <run directory>/esn.toml
    uv run python tools/robot_app.py experiments/demonstrations/reach_manual_v2.toml \\
        --law pd --omega 20 --damping 0.4 --model <run directory>/esn.toml \\
        --forces <storage root>/data/recorded_forces/<log>.sklog.npz
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PyQt6.QtGui import QCloseEvent, QColor, QKeySequence, QMouseEvent, QPainter, QPaintEvent, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skelarm import (
    Controller,
    PlaybackClock,
    RecordedForce,
    ShortcutFriendlySpinBox,
    SkelarmCanvas,
    Skeleton,
    SpeedSpinBox,
    StateLog,
    Task,
    TransportBar,
    bind_quit_key,
    compute_forward_kinematics,
    compute_inverse_kinematics,
    compute_jacobian,
    integrate_with_limits,
    scenario_from_config,
)
from skelarm.canvas import TrailOverlay, draw_arrow
from skelarm.simulator import SimulatorCanvas

from arm_esn_ctrl.demonstrations import (
    SCENARIO_TABLES,
    load_joint_angles,
    resample_joint_angles,
    simulate_reaches,
    training_demonstrations,
)
from arm_esn_ctrl.disturbances import same_arm
from arm_esn_ctrl.esn import ReachingEsn
from arm_esn_ctrl.storage import git_state, storage_root
from arm_esn_ctrl.tracking import (
    LAWS,
    EsnSource,
    ReferenceTracker,
    ReplaySource,
    TrackerConfig,
    error_dynamics,
    gain_posture,
    shift_times,
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
_FORCE_COLOR = QColor(220, 0, 0)  # the replayed force's arrow, red as skelarm's drag arrow
# Where the saved logs go by default, under the storage root, for experiments to share.
RECORDED_FORCES = Path("data") / "recorded_forces"
# The channels of a saved log, as skelarm's simulate_controlled describes them.
_CHANNEL_META: dict[str, dict[str, Any]] = {
    "q": {"unit": "rad", "label": "joint angle"},
    "dq": {"unit": "rad/s", "label": "joint velocity"},
    "tau": {"unit": "N*m", "label": "applied joint torque"},
    "ext_force": {"unit": "N", "label": "external tip force", "columns": ["fx", "fy"]},
}


class RobotCanvas(SimulatorCanvas):
    """The arm canvas: a drag poses the arm while :attr:`posing`, and pulls its tip with a spring otherwise.

    While :attr:`locked`, as when recorded forces are replayed, the mouse neither
    poses the arm nor pulls it. A :attr:`shown_force` is drawn as an arrow from the
    tip, :attr:`force_scale` meters per newton.
    """

    def __init__(self, skeleton: Skeleton) -> None:
        super().__init__(skeleton)
        self.posing = True
        self.locked = False
        self.shown_force: NDArray[np.float64] | None = None
        self.force_scale = 1.0 / _DEFAULT_STIFFNESS  # m/N: as a drag of the default stiffness

    def mousePressEvent(self, a0: QMouseEvent | None) -> None:
        if self.locked:
            return
        if self.posing:
            SkelarmCanvas.mousePressEvent(self, a0)
        else:
            super().mousePressEvent(a0)

    def mouseMoveEvent(self, a0: QMouseEvent | None) -> None:
        if self.locked:
            return
        if self.posing:
            SkelarmCanvas.mouseMoveEvent(self, a0)
        else:
            super().mouseMoveEvent(a0)

    def paintEvent(self, a0: QPaintEvent | None) -> None:
        """Draw the arm and the drag, then the shown force as an arrow from the tip."""
        super().paintEvent(a0)
        force = self.shown_force
        if force is None or not np.any(force):
            return
        tip = self.skeleton.links[-1]
        center_x, center_y = self.width() / 2, self.height() / 2
        end = np.array([tip.xe, tip.ye]) + self.force_scale * force
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        start = self._world_to_screen(tip.xe, tip.ye, center_x, center_y)
        draw_arrow(painter, start, self._world_to_screen(end[0], end[1], center_x, center_y), color=_FORCE_COLOR)


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
    log: StateLog | None = None  # every simulation step, on the task clock, as experiments/robot_esn.py logs a run

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
    forces : StateLog, optional
        A saved log whose external force every run replays by time, from the log's
        start posture, instead of the drag; posing and dragging are then off.
    gains_at : NDArray[np.float64], optional
        The posture where joint PD's gains are set (rad), as the experiments set them
        where the demonstrations end (see :func:`arm_esn_ctrl.tracking.gain_posture`);
        by default, the posture with the hand at the target.
    provenance : dict, optional
        Where the reference and the forces come from (such as the model's path), saved
        in each log's metadata.
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
        forces: StateLog | None = None,
        provenance: dict[str, Any] | None = None,
        gains_at: NDArray[np.float64] | None = None,
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
        if forces is not None and not same_arm(forces.build_skeleton(), skeleton):
            msg = "the recorded forces were recorded with another arm: its links differ from this robot's"
            raise ValueError(msg)
        self.skeleton = skeleton
        self.config = config
        self.provenance = {} if provenance is None else dict(provenance)
        # The code that runs is the code at launch: its commit, and whether the repository had uncommitted changes.
        self._repository = git_state()
        self.target = task.require_target()
        self.radius = task.tolerance
        self.tracker_config = tracker
        self.esn = esn
        self.given_q = None if given_q is None else np.asarray(given_q, dtype=np.float64).copy()
        self.period = esn.config.dt if esn is not None else period
        self.hold = hold
        self.dt = float(config["simulator"]["dt"])
        control_period = None if tracker is None else tracker.control_period
        if control_period is not None and not np.isclose(round(control_period / self.dt) * self.dt, control_period):
            msg = f"the control period {control_period} s must be a whole number of simulation steps of {self.dt} s"
            raise ValueError(msg)
        self.enforce_limits = bool(config["simulator"].get("enforce_limits", True))
        # The recorded forces, replayed on the task clock: each simulation step gets the mean recorded force over it,
        # which is the recorded sample itself when the log has this step.
        self.forces = forces
        self._replayed = None if forces is None else RecordedForce.from_log(forces).stepped(self.dt)
        if forces is not None:
            self._set_state(np.asarray(forces.channel("q")[0], dtype=np.float64))  # the log's start posture
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
        self.gains_at = self._target_posture() if gains_at is None else np.asarray(gains_at, dtype=np.float64)
        self.gains = None
        if tracker is not None:
            self.gains = tracking_gains(tracker, skeleton, self.gains_at) if gains is None else gains
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
        self.canvas.locked = forces is not None
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
        self.save_button = QPushButton("Save log")
        self.save_button.setToolTip(f"Save the run so far as a log, by default under {RECORDED_FORCES} in the storage")
        self.save_button.clicked.connect(self._on_save_clicked)
        controls.addWidget(self.save_button)
        self.save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        self.save_shortcut.activated.connect(self.save_button.click)

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
        log = StateLog(self.skeleton.clone(), producer="robot_app", channel_meta=_CHANNEL_META, extra=self._log_extra())
        self.run = RobotRun(
            start_q=start_q,
            controller=controller,
            tracker=tracker,
            warmup=warmup,
            warmup_steps=round(warmup / self.dt),
            target=self.target,
            radius=self.radius,
            hold=self.hold,
            log=log,
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
            force = self._external_force(run)
            assert run.log is not None
            run.log.record(
                run.time,
                q=self.skeleton.q,
                dq=self.skeleton.dq,
                tau=torque,
                ext_force=force,
                **run.controller.log_channels(),
            )
            tip = self.skeleton.links[-1]
            run.record(self.dt, self.skeleton.q.copy(), np.array([tip.xe, tip.ye]), torque, force)
            applied = torque + compute_jacobian(self.skeleton).T @ force
            if self.enforce_limits:
                integrate_with_limits(self.skeleton, applied, self.dt, self._lower, self._upper)
            else:
                integrate_with_limits(self.skeleton, applied, self.dt)
            run.sim_time += self.dt
            run.steps += 1

    def _external_force(self, run: RobotRun) -> NDArray[np.float64]:
        """The tip force of the step about to be taken: none in the warm-up, then the replayed one or the drag."""
        if run.warming_up:
            return np.zeros(2)
        if self._replayed is not None:
            return self._replayed(run.time, self.skeleton)
        return self.canvas.external_force(self.stiffness)

    def _log_extra(self) -> dict[str, Any]:
        """The metadata of a run's log: the task for the player, the tracker, and where everything comes from."""
        reference = {"esn": "esn", "demonstrator": "demonstrator"}.get(self.mode, "replay")
        tracking: dict[str, Any] = {"reference": reference, "start": "posed in the app"}
        if self.tracker_config is not None and self.gains is not None:
            tracking |= {"law": self.tracker_config.law, "reference_velocity": self.tracker_config.reference_velocity}
            if self.tracker_config.omega is not None:
                tracking |= {"omega": self.tracker_config.omega, "damping": self.tracker_config.damping}
            if self.tracker_config.control_period is not None:
                tracking["control_period"] = self.tracker_config.control_period
            tracking |= {"kp": self.gains[0].tolist(), "kd": self.gains[1].tolist()}
        commit, uncommitted_changes = self._repository
        app = {
            "mode": self.mode,
            "dt": self.dt,
            "enforce_limits": self.enforce_limits,
            "stiffness": self.stiffness,
            "start_q_deg": np.degrees(self.skeleton.q).tolist(),
            "commit": commit,
            "uncommitted_changes": uncommitted_changes,
        } | {key: str(value) for key, value in self.provenance.items() if value is not None}
        return {"playback": {"task": self.config["task"]}, "tracking": tracking, "robot_app": app}

    def saved_log(self) -> StateLog:
        """The run so far as a log, ending with a final frame, as skelarm's simulate_controlled ends a run.

        The final frame records the state reached, the torque last applied, and the
        force that would act next, which no step applied.

        Raises
        ------
        ValueError
            If there is no run, or it has taken no step.
        """
        run = self.run
        if run is None or run.log is None or len(run.log) == 0 or run.torque is None:
            msg = "there is no run to save: press Play or Step first"
            raise ValueError(msg)
        log = shift_times(run.log, 0.0)  # a copy, so that the run can go on
        log.record(
            run.time,
            q=self.skeleton.q,
            dq=self.skeleton.dq,
            tau=run.torque,
            ext_force=self._external_force(run),
            **run.controller.log_channels(),
        )
        return log

    def default_log_path(self) -> Path:
        """Where a run's log is saved by default: under :data:`RECORDED_FORCES` in the storage, named by time."""
        name = self.provenance.get("name") or self.mode
        if self.tracker_config is not None and self.tracker_config.omega is not None:
            name += f"_{self.tracker_config.law}_w{self.tracker_config.omega:g}_z{self.tracker_config.damping:g}"
        return storage_root() / RECORDED_FORCES / f"{datetime.now():%Y%m%d-%H%M%S}-{name}.sklog.npz"

    def save_log(self, path: str | Path) -> Path:
        """Save the run so far (see :meth:`saved_log`) to ``path``, creating its directory; returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.saved_log().save(path)
        return path

    def _on_save_clicked(self) -> None:
        """Pause, ask where to save the run's log, and save it."""
        self.pause()
        if self.run is None:
            self.phase_label.setText("There is no run to save yet: press Play (Space) first.")
            return
        default = self.default_log_path()
        default.parent.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "Save the run's log", str(default), "skelarm logs (*.sklog.npz)")
        if path:
            saved = self.save_log(path if path.endswith(".sklog.npz") else f"{path}.sklog.npz")
            self.phase_label.setText(f"Saved the run's log to {saved}")

    def _demonstrator_reach(self, start_q: NDArray[np.float64]) -> NDArray[np.float64]:
        """The demonstrator's reach from ``start_q``, sampled every reference period."""
        config = {name: self.config[name] for name in SCENARIO_TABLES}
        config["demonstrations"] = {"start_q": [np.degrees(start_q).tolist()]}
        return resample_joint_angles(simulate_reaches(config)[0], self.period)[1]

    def _target_posture(self) -> NDArray[np.float64]:
        """The posture with the hand at the target, near the first start posture, for gains without demonstrations."""
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
            natural, damping = error_dynamics(self.tracker_config.law, self.gains, self.skeleton, self.gains_at)
            controller += (
                f"\nTracking error: natural frequency {', '.join(f'{value:.3g}' for value in natural)} rad/s,"
                f" damping ratio {', '.join(f'{value:.2f}' for value in damping)}"
                + (" (where the reaches end)" if self.tracker_config.law == "pd" else "")
            )
            if not self.tracker_config.reference_velocity:
                controller += "\nReference velocity: zero (the derivative term damps the arm's own velocity)"
            if self.tracker_config.control_period is not None:
                controller += f"\nTorque: computed every {1000 * self.tracker_config.control_period:g} ms, held between"
        text = f"Reference: {reference}\nController: {controller}"
        if self.forces is not None:
            name = self.provenance.get("forces")
            text += (
                f"\nForces: replayed from {Path(name).name if name else 'a saved log'}, from its start posture;"
                " posing and dragging are off"
            )
            recorded_step = float(np.median(np.diff(self.forces.times)))
            if not np.isclose(recorded_step, self.dt):
                text += f" (recorded every {1000 * recorded_step:g} ms: each step gets the mean force over it)"
            recorded = self.forces.extra.get("tracking", {})
            if "law" in recorded:
                text += f"\nRecorded with: {recorded['law']}" + (
                    f", ω = {recorded['omega']:g} rad/s, damping ratio {recorded['damping']:g}"
                    if "omega" in recorded
                    else ""
                )
        return text

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
            self.time_label.setText("Posing" if self.forces is None else "Ready")
            self.phase_label.setText(
                "Drag the arm tip to choose a start posture, then press Play (Space)."
                if self.forces is None
                else "The arm starts from the recorded posture: press Play (Space) to replay the recorded forces."
            )
            self.metrics_label.setText("\n".join(lines))
            return
        self.time_label.setText(f"t = {run.time:+.2f} s")
        paused = "" if self.is_playing else " (paused)"
        action = "Drag to push the arm" if self.forces is None else "The recorded forces push the arm"
        self.phase_label.setText(f"{run.phase.capitalize()}{paused}. {action}; Reset (R) to start again.")
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
        self.canvas.shown_force = None if run is None or self.forces is None else run.force
        self.canvas.force_scale = 1.0 / self.stiffness if self.stiffness > 0 else 0.0
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
        "--control-period",
        type=float,
        help="compute the tracker's torque this often and hold it between (s): a whole number of simulation steps;"
        " by default, at every step",
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
    parser.add_argument(
        "--forces",
        type=Path,
        help="a saved log (.sklog.npz) whose external force every run replays by time, from its start posture,"
        " instead of the drag",
    )
    return parser


def check_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject combinations of arguments that contradict each other."""
    if args.model is not None and args.demonstrator:
        parser.error("--model and --demonstrator exclude each other: either the ESN or the demonstrator drives the arm")
    if args.replay is not None and (args.model is not None or args.demonstrator):
        parser.error("--replay excludes --model and --demonstrator: the take is the reference")
    gains = (args.omega, args.damping, args.kp, args.kd)
    if args.demonstrator and (
        args.law is not None
        or any(value is not None for value in gains)
        or args.zero_reference_velocity
        or args.control_period is not None
    ):
        parser.error(
            "--law, --omega, --damping, --kp, --kd, --zero-reference-velocity, and --control-period set the tracker,"
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


def reference_demonstrations(
    config: dict[str, Any], model: Path | None, esn: ReachingEsn | None, take: NDArray[np.float64] | None, period: float
) -> list[NDArray[np.float64]]:
    """The demonstrations of the reference, whose final postures set joint PD's gains, as in the experiments.

    The takes the ESN was trained on, named in its run's configuration; else the
    replayed take; else the demonstrator's reaches of the configuration's
    ``[demonstrations]``; else none.
    """
    if model is not None and esn is not None:
        with (model.parent / "config.toml").open("rb") as f:
            demonstrations = tomllib.load(f)["demonstrations"]
        return list(training_demonstrations(demonstrations, esn.config.dt).values())
    if take is not None:
        return [take]
    if "demonstrations" in config and all(name in config for name in SCENARIO_TABLES):
        return [resample_joint_angles(log, period)[1] for log in simulate_reaches(config)]
    return []


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
    forces = None
    if args.forces is not None:
        forces = StateLog.load(args.forces)
        if "ext_force" not in forces.channel_names:
            parser.error(f"--forces: {args.forces} records no external force (ext_force)")
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
            control_period=args.control_period,
        )
    if args.kp is not None:
        gains = (
            joint_values(parser, "--kp", args.kp, skeleton.num_joints),
            joint_values(parser, "--kd", args.kd, skeleton.num_joints),
        )

    name = args.model.parent.name if args.model is not None else None if args.replay is None else args.replay.name
    demonstrations = [] if args.demonstrator else reference_demonstrations(config, args.model, esn, take, period)
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
            name=name,
            forces=forces,
            provenance={
                "config": args.config,
                "model": args.model,
                "replay": args.replay,
                "forces": args.forces,
                "name": name,
                "command": " ".join(sys.argv),
            },
            gains_at=gain_posture(demonstrations) if demonstrations else None,
        )
    except ValueError as error:
        parser.error(str(error))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
