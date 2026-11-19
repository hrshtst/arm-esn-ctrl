# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Run a trained ESN as the reference generator of the simulated robot arm (Stage 2).

    uv run python experiments/robot_esn.py experiments/multi_demonstration_robot_tracking/nominal.toml

The arm tracks a reference generated every reference period (the ESN's 10 ms)
from its measured joint angles (see :mod:`arm_esn_ctrl.tracking`). These arms
reach from each start posture:

- ``esn``: the ESN, driven by the arm's measured joint angles;
- ``replay``: the demonstration that starts nearest, replayed by time (the
  time-indexed baseline);
- ``progress``, with a ``[progress]`` table: the same demonstration replayed by
  time, but never more than ``lead_deg`` along its joint path past the arm's
  progress (the progress-based baseline), the arm's nearest sample searched within
  ``window`` seconds of the last one found, and falling back by at most a sample a
  period;
- ``demonstrator``: the controller that made the demonstrations, on its own.

The ESN and the replays run with every tracking law, natural frequency of the
tracking error, and damping ratio listed in ``[tracker]`` (``dampings`` is
optional; its default, 1, is critically damped). ``omegas`` is a list for every
law, or a table with a list for each law, such as
``omegas = {computed_torque = [10.0], pd = [20.0]}``. Instead of ``omegas`` and
``dampings``, ``kp`` and ``kd`` can give one law's gains directly, each one number
for every joint or a list of one per joint, such as ``kp = 80.0`` and
``kd = 3.0``. With ``reference_velocity =
false`` (optional; the default is true), the tracking laws are given a zero
reference velocity, so that their derivative term damps the arm's own velocity
rather than the velocity error. With ``control_period`` (optional, in seconds; a
whole number of simulation steps), their torque is computed only that often and held
between, as by a controller slower than the simulation. With ``mass_matrix = true``
(optional), joint PD's gains with ``omegas`` are the whole mass matrix at the reaches'
end, scaled (``Kp = ω² M``, ``Kd = 2ζω M``), rather than each joint's inertia, as for
an arm whose joints couple strongly (see :func:`arm_esn_ctrl.tracking.tracking_gains`).
Their runs start with the arm holding its
start posture for the ESN's warm-up (at negative times), or for ``warmup`` seconds if
``[evaluation]`` gives one, and the task starts at t = 0. An optional
``[disturbance]`` table, or several ``[[disturbance]]`` tables, push or block all
the arms alike, or replay a force recorded in a log, such as one the robot app saved
(see :mod:`arm_esn_ctrl.disturbances`): from the same start posture, the same forces
at the same times. Under a recorded force, the runs start from its log's start
posture, and the run directory keeps a copy of the log. The metrics then measure how
far each disturbance sends the hand from the taught hand path, how soon the hand
comes back within ``return_tolerance`` of it (in ``[evaluation]``; 5 mm by default),
and how much it rings afterwards: its motion above ``ringing_cutoff_hz`` (1.5 Hz by
default) over ``ringing_window`` seconds (2 s by default); and how much its reference
follows the arm from the disturbance's onset to ``following_window`` seconds past its
end (0.5 s by default).
Start postures away from the demonstrated
ones (``start_offsets_deg`` in ``[evaluation]``) make the initial-offset scenario.
A push can be placed by its ``way`` along the taught motion each run starts from,
instead of its ``onset`` (see :func:`arm_esn_ctrl.disturbances.placed`). With
``gains_from`` in ``[tracker]``, a demonstration file under the storage root, joint
PD's gains with ``omegas`` are set where that take ends rather than where the ESN's
own demonstrations do, so that ESNs trained on other takes are tracked alike.

An ESN of several tasks (see :mod:`arm_esn_ctrl.esn`) runs each task from that
task's start postures with that task's context, against the replays of that task's
demonstrations; the run directory then has a directory per task, named by it, with
the runs and figures below, and ``metrics.csv`` has a ``task`` column. Each task's
runs follow its own evaluation: ``[evaluation.tasks.<task>]`` tables give a task keys
of its own in place of ``[evaluation]``'s, such as its ``duration``, its
``start_offsets_deg``, and its ``extra_starts`` (see
:func:`arm_esn_ctrl.autonomous.task_evaluations`). ``[evaluation]``'s ``start_tasks``,
a list of the tasks' names, runs only the start postures of those tasks; every task's
without it. ``[[switch]]`` tables change the ESN's task during every run: each gives a
``time`` on the task clock and the ``task``'s name. The runs' logs record the context
at every step (``context``), and the metrics then measure the part of the run after
the last change as a run of the last task that begins at the change, and each change
on its own (see :func:`switch_metrics`; empty for the replays, which keep to their
take). The figures over time mark each change with a vertical line and name its new task.

A periodic task (a ``[task]`` of type ``periodic_curve``, taught by one take) has no
target. Its replay, and its progress replay if there is one, go on without end
through the take's taught loop, at the ESN's period (see
:func:`arm_esn_ctrl.metrics.taught_loop`): from the take's first sample, whatever the
start posture. Its pushes are turned from the direction from the start hand toward
the curve's center, and given by their ``onset``. Its hand is measured by its
distance from the task's curve, the ellipse itself: off the curve. Its metrics say
whether each run cycles, by ``[evaluation]``'s ``curve_tolerance``,
``cycle_periods``, and ``cycle_turns`` (see :func:`arm_esn_ctrl.autonomous.task_cycle`
and :func:`arm_esn_ctrl.metrics.cycle_metrics`): ``cycles``, ``turns``,
``curve_rms_m``, ``off_curve_max_m``, ``period_s``, ``period_ratio``, and
``joining_time_s``. The measures of the arrival, the hold, the detour, the settling,
the distance to the target, and the progress along the demonstrated path and the
reference's lead are empty or NaN. After a disturbance, ``off_path_m`` and
``return_time_s`` are measured from the curve, the following gain against where the
looping replay is at the same time, and two measures are added: the largest joint
torque (``torque_after_nm``) and the hand's largest distance from the curve
(``off_curve_after_m``) over ``after_window`` seconds (in ``[evaluation]``; 2 s by
default) from its end. ``paths.png`` draws the curve, ``timeline.png`` the hand's
distance from the curve and its turns about the curve's center, and ``metrics.png``
is not drawn.

Every run is compared with the demonstrator's undisturbed reach from the same
start posture, and the run directory receives:

- ``<law>_w<omega>/esn_00.sklog.npz``, ``replay_00.sklog.npz``, ``progress_00.sklog.npz``, ...: the arm's runs
  (``<law>_w<omega>_z<damping>`` below critical damping, ``<law>_kp<kp>_kd<kd>`` with
  given gains),
  with the reference (``q_ref``), the tracking error (``error``), and the
  disturbance (``ext_force``);
- ``demonstrator_00.sklog.npz``, ...: the demonstrator's reaches, under the same disturbance;
- ``metrics.csv``: the metrics of every run (see :func:`arm_metrics`);
- ``metrics.png``: those metrics against the natural frequency (or the damping
  ratio, if that is what varies), for each law; not with given gains;
- ``paths.png``: the hand paths of every run, with faint postures of the arm driven
  by the ESN from the first start posture (the first, the last, and three between);
- ``timeline.png``: the hand's distance to the target, the progress along the
  demonstrated path of the arm and of its reference, and the joint torque over
  time, from the first start posture;
- ``joints.png``: the joint angles over time from the first start posture: the
  demonstrated motion, the ESN's output (its reference), and the arms;
- ``torques.png``: the arms' joint torques over time from the first start posture;
- ``joints_offset_<dq1>_<dq2>.png`` and ``torques_offset_<dq1>_<dq2>.png``, such as
  ``joints_offset_+15_+15.png``: the same from each start posture offset by
  ``[dq1, dq2]`` degrees (one offset per joint) that an optional ``[plots]`` table
  lists in ``offsets_deg``;
- ``grid.png``, with an ``[evaluation.start_grid]``: maps over the start offsets
  of the outcome, the path distance, the training path ratio, the peak reference
  speed, the peak torque, and the tracking error, for each tracker setting (with
  more than two joints, as nested grids: see :class:`arm_esn_ctrl.drawing.OffsetMap`).

Replay a run with ``uv run python third_party/skelarm/tools/player.py <file>``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray
from skelarm import Skeleton, StateLog

from arm_esn_ctrl.autonomous import (
    AFTER_WINDOW,
    CURVE_SIDES,
    Cycle,
    Reference,
    Run,
    Setup,
    cycle_run_metrics,
    demonstration_tables,
    load_setups,
    rms_degrees,
    run_metrics,
    task_cycle,
    task_evaluations,
    task_names,
)
from arm_esn_ctrl.demonstrations import endpoint_positions, simulate_disturbed_reach, task_path
from arm_esn_ctrl.disturbances import (
    check_recorded_arm,
    disturbance_spans,
    keep_recorded_logs,
    make_disturbance,
    placed,
    recorded_evaluation,
)
from arm_esn_ctrl.drawing import OffsetMap, draw_postures, spread_along_path
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.metrics import (
    arrival_index,
    cycle_metrics,
    distances_to_path,
    disturbance_following,
    disturbance_returns,
    disturbance_ringing,
    hand_speed,
    hold_metrics,
    largest_after,
    loop_phase,
    motor_effort,
    path_progress,
    path_rmse,
)
from arm_esn_ctrl.storage import resolve_run_path, start_run
from arm_esn_ctrl.tracking import (
    EsnSource,
    ProgressReplaySource,
    ReferenceSource,
    ReplaySource,
    TrackerConfig,
    gains_posture,
    loop_sample,
    nearest_demonstration,
    task_joint_angles,
    track,
    tracking_gains,
)

TRACKED = ("esn", "replay", "progress")  # the arms that track a reference, those a run has
ARMS = (*TRACKED, "demonstrator")
ARM_COLORS = {"esn": "#2a78d6", "replay": "#eb6834", "progress": "#2e8b57", "demonstrator": "#52514e"}
ARM_LABELS = {
    "esn": "ESN on the measured posture",
    "replay": "demonstration replayed by time",
    "progress": "demonstration replayed by the arm's progress",
    "demonstrator": "demonstrator",
}
# The layers of the arms' lines over time: the ESN's on top, its reference (dashed) under it, the others below.
ARM_ZORDERS = {"esn": 3.0, "replay": 2.0, "progress": 2.0, "demonstrator": 2.0}
REFERENCE_ZORDER = 2.5
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3de"
SURFACE_COLOR = "#fcfcfb"
DISTURBANCE_COLOR = "#f0efec"
CHANGE_COLOR = "#8a6fc4"  # a change of the ESN's task
# What switch_metrics measures of the part of a run after the last change of its task: a reach's, a periodic task's.
SWITCH_REACH = (
    "arrived",
    "arrival_time_s",
    "left_goal",
    "hold_error_m",
    "hold_observed_s",
    "success",
    "final_distance_m",
)
SWITCH_CYCLE = ("cycles", "turns", "curve_rms_m", "joining_time_s")
# How near the taught hand path the hand must come back after a disturbance, unless
# [evaluation] gives a return_tolerance (m).
RETURN_TOLERANCE = 0.005
# After a disturbance, the hand's motion above this frequency is its ringing, measured over this
# window, unless [evaluation] gives a ringing_cutoff_hz (Hz) and a ringing_window (s). The cutoff lies
# between the reach, under about 0.5 Hz, and a tracker at omega = 20 rad/s, which rings at 3.2 Hz.
RINGING_CUTOFF_HZ = 1.5
RINGING_WINDOW = 2.0
# How much the reference follows the arm is measured from a disturbance's onset to this long after its
# end, unless [evaluation] gives a following_window (s): the block's effort window, 0.5 s past its release.
FOLLOWING_WINDOW = 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="experiment configuration file (TOML)")
    args = parser.parse_args()

    config, run_dir = start_run(args.config)
    esn_path = resolve_run_path(config["esn"]["model"])
    esn = ReachingEsn.load(esn_path)
    # The demonstrations the ESN was trained on, which the replay replays: of each of its tasks, if several.
    with (esn_path.parent / "config.toml").open("rb") as f:
        demonstrations = tomllib.load(f)["demonstrations"]
    names = task_names(demonstrations)
    disturbance = config.get("disturbance")
    # Under a recorded disturbance, the runs start from its log's start posture, with an arm like its log's.
    evaluation = recorded_evaluation(config["evaluation"], disturbance)
    setups = load_setups({"demonstrations": demonstrations, "esn": {"dt": esn.config.dt}, "evaluation": evaluation})
    evaluations = task_evaluations(evaluation, names)
    # What each periodic task's runs, and the part of a run after a change to it, are measured against.
    cycles = [task_cycle(setup, each) for setup, each in zip(setups, evaluations, strict=True)]
    check_recorded_arm(disturbance, setups[0].skeleton)
    keep_recorded_logs(disturbance, run_dir)
    switches = switch_schedule(config.get("switch", []), names)
    started = start_tasks(evaluation, names)
    n_demos = sum(len(setup.demos) for setup in setups)
    tasks = f" of {len(names)} tasks ({', '.join(names)})" if names else ""
    print(f"ESN {config['esn']['model']}, trained on {n_demos} demonstrations{tasks}")
    n_starts = sum(len(setups[task].starts) for task in started)
    print(f"Disturbance: {disturbance or 'none'}; {n_starts} start postures")
    if len(started) < len(setups):
        print(f"Only the start postures of {', '.join(names[task] for task in started)} are run")
    if switches:
        print("The ESN's task changes: " + ", ".join(f"to {names[task]} at {time:g} s" for time, task in switches))
    if not config["tracker"].get("reference_velocity", True):
        print("The tracking laws are given a zero reference velocity")
    if "control_period" in config["tracker"]:
        print(f"The tracking laws' torque is computed every {config['tracker']['control_period']} s and held between")
    # Where joint PD's gains are set: the same for every task.
    every_demo = [q for setup in setups for q in setup.demos.values()]
    end_posture = gains_posture(config["tracker"], every_demo, esn.config.dt)

    rows: list[dict[str, Any]] = []
    tables = demonstration_tables(demonstrations)
    for task in started:
        with (resolve_run_path(tables[task]["run"]) / "config.toml").open("rb") as f:
            demo_config = tomllib.load(f)
        task_dir = run_dir
        schedule = None
        if names:
            print(f"\nTask {names[task]}")
            task_dir = run_dir / names[task]
            task_dir.mkdir()
            schedule = [(0.0, task), *switches]
        task_rows = run_task(
            config,
            evaluations[task],
            args.config.stem,
            task_dir,
            esn,
            setups[task],
            demo_config,
            end_posture,
            schedule,
            (setups, cycles),
        )
        rows += [({"task": names[task]} if names else {}) | row for row in task_rows]

    with (run_dir / "metrics.csv").open("w", newline="") as out:
        # Every row's columns, in the order they first come: a reach's runs lack a periodic task's measures.
        writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote the results to {run_dir}")
    first = (
        (run_dir / names[started[0]] if names else run_dir)
        / tracker_settings(config["tracker"])[0].name
        / "esn_00.sklog.npz"
    )
    print(f"Replay with:\n  uv run python third_party/skelarm/tools/player.py {first}")


def start_tasks(evaluation: dict[str, Any], names: tuple[str, ...]) -> list[int]:
    """The tasks whose start postures are run, as indices: ``[evaluation]``'s ``start_tasks``, or every task.

    A single task, without ``names``, is the one task 0.

    Raises
    ------
    ValueError
        If ``start_tasks`` names a task the ESN does not have, or none, or is given for
        an ESN of one task.
    """
    if "start_tasks" not in evaluation:
        return list(range(max(len(names), 1)))
    listed = evaluation["start_tasks"]
    if not listed or any(name not in names for name in listed):
        msg = f"[evaluation] start_tasks lists some of the ESN's tasks, {', '.join(names) or 'none for one task'}"
        raise ValueError(msg)
    return [task for task, name in enumerate(names) if name in listed]


def switch_schedule(tables: list[dict[str, Any]], names: tuple[str, ...]) -> list[tuple[float, int]]:
    """When the ESN's task changes during every run, from ``[[switch]]`` tables: each change's time and task index.

    Raises
    ------
    ValueError
        If the ESN has one task, a table lacks its ``time`` or names an unknown
        ``task``, or the times are not in order.
    """
    if tables and not names:
        msg = "[[switch]] changes the task of an ESN of several tasks: this ESN has one"
        raise ValueError(msg)
    schedule = []
    for table in tables:
        if set(table) != {"time", "task"} or table["task"] not in names:
            msg = f"each [[switch]] gives a time and the task's name, one of {', '.join(names)}"
            raise ValueError(msg)
        schedule.append((float(table["time"]), names.index(table["task"])))
    if any(later[0] < earlier[0] for earlier, later in itertools.pairwise(schedule)):
        msg = "the [[switch]] tables must be in the order of their times"
        raise ValueError(msg)
    return schedule


def switch_metrics(
    log: StateLog | None,
    times: NDArray[np.float64],
    schedule: list[tuple[float, int]],
    tasks: tuple[list[Setup], list[Cycle | None]],
    period: float,
    after: float,
) -> dict[str, Any]:
    """How a run whose task changes does each task it changes to, each part measured as a run that begins at its change.

    ``times`` are the run's, every ``period`` from 0 to its end; ``schedule`` gives its
    task, its first from 0 and then each change, as a time on the task clock and an
    index into ``tasks``: every task's setup, and its cycle if it is periodic (see
    :func:`arm_esn_ctrl.autonomous.task_cycle`).

    The part after the last change is measured as a run of the last task:

    - a reach: whether the hand arrives at its target and holds there, the hold
      metrics of that part (see :func:`arm_esn_ctrl.metrics.hold_metrics`) with the
      arrival's time counted from the change, as ``switch_arrived``,
      ``switch_arrival_time_s``, ``switch_left_goal``, ``switch_hold_error_m``,
      ``switch_hold_observed_s``, and ``switch_success``, and the hand's distance to
      the target at the run's end, ``switch_final_distance_m``;
    - a periodic task: whether the hand cycles at the end of the run, by
      :func:`arm_esn_ctrl.metrics.cycle_metrics`' rule over the run's last
      ``cycle.periods`` taught periods (``switch_cycles``), the turns and the curve RMS
      it is decided by (``switch_turns``, ``switch_curve_rms_m``), and the joining time
      counted from the change (``switch_joining_time_s``).

    Then, for each change k, from 1:

    - ``cycles_before_switch_k``: for a change that leaves a periodic task, whether the
      hand cycles, by the same rule, over the ``cycle.periods`` taught periods that end
      at the change, or from the run's start if it changes sooner; empty where the task
      left is a reach;
    - ``switch_peak_reference_speed_dps_k`` and ``switch_peak_torque_nm_k``: the
      reference's fastest joint speed (see :func:`reference_speed`) and the largest
      joint torque over the ``after`` seconds from the change;
    - for a change to a reach, ``switch_arrival_time_s_k``: the time from the change
      until the hand arrives at the reach's target, NaN if it does not before the next
      change or the run's end; for a change to a periodic task,
      ``switch_joining_time_s_k``: the joining time counted from the change, over the
      part until the next change or the run's end, NaN if the hand has not joined the
      curve by then.

    Without a ``log``, as for a replay, which keeps to its take, every one is empty.

    The run's own measures (:func:`arm_metrics`) are still its first task's, over the
    whole run. Those of how it starts (``first_step_m``, ``onset_delay_s``), and a
    reach's arrival, hold, and detour if the hold ends before the first change, keep
    their meaning; those over the whole run or at its end (the final distance, the
    settling time, the path RMSEs and distances, the peaks of the reference's speed, the
    hand's speed, and the torque, the effort, and a periodic task's cycle measures over
    the run's last periods) mix the tasks, and only the tracking error keeps its own.
    """
    setups, cycles = tasks
    changes = schedule[1:]
    last_time, last = changes[-1]
    keys = SWITCH_REACH if cycles[last] is None else SWITCH_CYCLE
    names = [f"switch_{key}" for key in keys]
    for k, (_, task) in enumerate(changes, start=1):
        names += [f"cycles_before_switch_{k}", f"switch_peak_reference_speed_dps_{k}", f"switch_peak_torque_nm_{k}"]
        names.append(f"switch_arrival_time_s_{k}" if cycles[task] is None else f"switch_joining_time_s_{k}")
    if log is None:
        return dict.fromkeys(names, "")

    nan = float("nan")
    hand = endpoint_positions(setups[last].skeleton, task_joint_angles(log, period, float(times[-1])))

    def cycled(part: NDArray[np.bool_], cycle: Cycle, start: float = 0.0) -> dict[str, Any]:
        """The cycle measures of the run's ``part``, its times counted from ``start``."""
        return cycle_metrics(
            times[part] - start,
            hand[part],
            cycle.curve,
            cycle.center,
            cycle.period,
            cycle.tolerance,
            cycle.periods,
            cycle.turns,
        )

    measured: dict[str, Any] = {}
    after_last = times >= last_time - 1e-9
    last_cycle = cycles[last]
    if last_cycle is None:
        setup = setups[last]
        held = hold_metrics(times[after_last] - last_time, hand[after_last], setup.target, setup.radius, setup.hold)
        held["final_distance_m"] = float(np.linalg.norm(hand[-1] - setup.target))
    else:
        held = cycled(after_last, last_cycle, last_time)
    measured |= {f"switch_{key}": held[key] for key in keys}
    ends = [time for time, _ in changes[1:]] + [float(times[-1])]
    tau = np.abs(log.channel("tau"))
    q_ref = log.channel("q_ref")
    for k, ((time, task), end) in enumerate(zip(changes, ends, strict=True), start=1):
        left = cycles[schedule[k - 1][1]]
        measured[f"cycles_before_switch_{k}"] = "" if left is None else cycled(times <= time + 1e-9, left)["cycles"]
        window = (log.times > time - 1e-9) & (log.times < time + after + 1e-9)
        measured[f"switch_peak_reference_speed_dps_{k}"] = reference_speed(log.times[window], q_ref[window])
        measured[f"switch_peak_torque_nm_{k}"] = float(tau[window].max())
        stretch = (times >= time - 1e-9) & (times <= end + 1e-9)
        cycle = cycles[task]
        if cycle is None:
            arrival = arrival_index(hand[stretch], setups[task].target, setups[task].radius)
            measured[f"switch_arrival_time_s_{k}"] = nan if arrival is None else float(times[stretch][arrival] - time)
        else:
            measured[f"switch_joining_time_s_{k}"] = cycled(stretch, cycle, time)["joining_time_s"]
    return {name: measured[name] for name in names}


def run_task(
    config: dict[str, Any],
    evaluation: dict[str, Any],
    name: str,
    run_dir: Path,
    esn: ReachingEsn,
    setup: Setup,
    demo_config: dict[str, Any],
    end_posture: NDArray[np.float64],
    schedule: list[tuple[float, int]] | None,
    tasks: tuple[list[Setup], list[Cycle | None]],
) -> list[dict[str, Any]]:
    """Run every arm from one task's start postures with every tracker setting, and draw that task's figures.

    ``evaluation`` is this task's; ``run_dir`` receives the runs and the figures;
    ``schedule`` gives an ESN of several tasks its context, this task's from the
    warm-up on and then each change, as indices into ``tasks``: every task's setup, and
    its cycle if it is periodic, which the measures of the changes need (see
    :func:`switch_metrics`). Returns the metrics of every run.
    """
    disturbance = config.get("disturbance")
    window = evaluation.get("effort_window", [0.0, evaluation["duration"]])
    tolerance = evaluation.get("return_tolerance", RETURN_TOLERANCE)
    ringing = ringing_settings(evaluation)
    following = evaluation.get("following_window", FOLLOWING_WINDOW)
    period = esn.config.dt
    cycle = task_cycle(setup, evaluation)
    if cycle is not None:
        print_taught_loop(setup, cycle, period)
    after = evaluation.get("after_window", AFTER_WINDOW)  # after a change of the ESN's task (s)

    def disturbance_of(i: int) -> Any:
        """The disturbance's tables for a run from start posture ``i``, its pushes placed along the taught motion."""
        taught_hand = setup.demo_hands[nearest_demonstration(setup.starts[i].q, setup.demos)]
        return placed(disturbance, taught_hand, setup.target, period)

    def disturbance_from(i: int) -> Any:
        """A fresh disturbance for a run from start posture ``i``."""
        start_hand = endpoint_positions(setup.skeleton, setup.starts[i].q[np.newaxis])[0]
        aim = push_aim(setup, cycle)
        return make_disturbance(disturbance_of(i), start_hand, aim, dt=demo_config["simulator"]["dt"])

    spans = [disturbance_spans(disturbance_of(i)) for i in range(len(setup.starts))]
    changes = [] if schedule is None else schedule[1:]

    def switch_columns(log: StateLog | None) -> dict[str, Any]:
        """The measures of the changes of the ESN's task, if it changes; empty without a ``log``."""
        if schedule is None or not changes:
            return {}
        return switch_metrics(log, setup.times, schedule, tasks, period, after)

    rows = []
    for i, start in enumerate(setup.starts):
        if not setup.demonstrator_logs:  # a demonstration taught by hand: no demonstrator to run
            break
        log = setup.demonstrator_logs[i]
        if disturbance is not None:
            log = simulate_disturbed_reach(demo_config, start.q, evaluation["duration"], disturbance_from(i))
        log.save(run_dir / f"demonstrator_{i:02d}.sklog.npz")
        rows.append(
            {"law": "", "omega": "", "damping": "", "arm": "demonstrator", "start": i, "origin": start.origin}
            | {"replayed": ""}
            | arm_metrics(log, i, setup, period, window, spans[i], tolerance, ringing, following, cycle)
            | switch_columns(None)
        )

    progress = config.get("progress")
    tracker = config["tracker"]
    settings = tracker_settings(tracker)
    for setting in settings:
        law = setting.law
        tracker_config = TrackerConfig(
            law,
            setting.omega,
            tracker["acceleration_filter"],
            damping=setting.damping,
            reference_velocity=tracker.get("reference_velocity", True),
            control_period=tracker.get("control_period"),
            mass_matrix=tracker.get("mass_matrix", False),
        )
        gains = setting.gains(tracker_config, setup.skeleton, end_posture)
        # The log keeps the natural frequency and damping ratio, or the given gains.
        given = (
            {"kp": gains[0].tolist(), "kd": gains[1].tolist()}
            if setting.omega is None
            else {"omega": setting.omega, "damping": setting.damping}
        )
        held = {} if tracker_config.control_period is None else {"control_period": tracker_config.control_period}
        held |= {"mass_matrix": True} if tracker_config.mass_matrix else {}
        setting_dir = run_dir / setting.name
        setting_dir.mkdir()
        for i, start in enumerate(setup.starts):
            replayed = nearest_demonstration(start.q, setup.demos)
            loop = None if cycle is None else cycle.loop  # a periodic take is replayed without end
            sources: dict[str, ReferenceSource] = {
                "esn": EsnSource(esn, schedule),
                "replay": ReplaySource(setup.demos[replayed], loop),
            }
            if progress is not None:
                sources["progress"] = progress_source(progress, setup.demos[replayed], period, loop)
            for arm, source in sources.items():
                log = track(
                    posed(setup.skeleton, start.q),
                    source,
                    tracker_config,
                    gains,
                    period=period,
                    warmup_steps=warmup_steps(evaluation, esn.config),
                    duration=evaluation["duration"],
                    dt=demo_config["simulator"]["dt"],
                    enforce_limits=demo_config["simulator"].get("enforce_limits", True),
                    stops=demo_config["simulator"].get("stops", "projected"),
                    extra={
                        "playback": {"task": setup.task},
                        "tracking": {
                            "reference": arm,
                            "law": law,
                            **given,
                            "reference_velocity": tracker_config.reference_velocity,
                            "start": start.origin,
                        }
                        | held
                        | ({"progress": progress} if arm == "progress" else {}),
                    },
                    external_force=disturbance_from(i),
                )
                log.save(setting_dir / f"{arm}_{i:02d}.sklog.npz")
                rows.append(
                    setting.columns
                    | {"arm": arm, "start": i}
                    | {"origin": start.origin}
                    | {"replayed": replayed if arm in ("replay", "progress") else ""}
                    | arm_metrics(log, i, setup, period, window, spans[i], tolerance, ringing, following, cycle)
                    | switch_columns(log if arm == "esn" else None)
                )
        print(f"Ran {setting.label} (kp = {np.round(gains[0], 2).tolist()}, kd = {np.round(gains[1], 2).tolist()})")

    if cycle is None:
        print_summary(rows, len(setup.starts), window, setup.reference)
    else:
        print_cycle_summary(rows, len(setup.starts))
    if changes:
        print_switch_summary(rows, changes, esn.contexts)

    title = f"ESN on the robot: {name}"
    # Given gains have no natural frequency or damping ratio to plot against, and a periodic task's runs neither
    # arrive nor hold.
    if settings[0].omega is not None and cycle is None:
        swept = "damping" if len(tracker.get("dampings", [1.0])) > 1 else "omega"
        plot_metrics(rows, tracker["laws"], len(setup.starts), window, swept, setup.reference, title).savefig(
            run_dir / "metrics.png", dpi=150
        )
    plot_paths(run_dir, setup, settings, title).savefig(run_dir / "paths.png", dpi=150)
    changed = [(time, esn.contexts[task]) for time, task in changes]  # marked on the figures over time
    if cycle is None:
        timeline = plot_timeline(run_dir, setup, settings, spans[0], title)
    else:
        timeline = plot_cycle_timeline(run_dir, setup, settings, spans[0], title, cycle)
    mark_changes(timeline, changed).savefig(run_dir / "timeline.png", dpi=150)
    mark_changes(plot_joints(run_dir, setup, settings, spans[0], title), changed).savefig(
        run_dir / "joints.png", dpi=150
    )
    mark_changes(plot_torques(run_dir, setup, settings, spans[0], title), changed).savefig(
        run_dir / "torques.png", dpi=150
    )
    for offset in config.get("plots", {}).get("offsets_deg", []):
        index = offset_start(setup, offset)
        name = "offset_" + "_".join(f"{value:+g}" for value in offset)
        mark_changes(plot_joints(run_dir, setup, settings, spans[index], title, index), changed).savefig(
            run_dir / f"joints_{name}.png", dpi=150
        )
        mark_changes(plot_torques(run_dir, setup, settings, spans[index], title, index), changed).savefig(
            run_dir / f"torques_{name}.png", dpi=150
        )
    if "start_grid" in evaluation:
        plot_grid(rows, setup, settings, title).savefig(run_dir / "grid.png", dpi=150)
    return rows


def mark_changes(fig: Figure, changes: list[tuple[float, str]]) -> Figure:
    """Mark each change of the ESN's task on every panel of a figure over time, and name its new task on the first.

    ``changes`` gives each change's time on the task clock and its task's name.
    """
    for ax in fig.axes:
        for time, _ in changes:
            ax.axvline(time, color=CHANGE_COLOR, linewidth=1.0, linestyle="-.", zorder=1.0)
    for time, task in changes:
        fig.axes[0].annotate(
            f" to {task}", (time, 1.0), xycoords=("data", "axes fraction"), va="top", color=CHANGE_COLOR, fontsize=8
        )
    return fig


def print_switch_summary(rows: list[dict[str, Any]], changes: list[tuple[float, int]], names: tuple[str, ...]) -> None:
    """Print, of the ESN's runs, how many do the task of the last change, and how many cycle before each change.

    After the last change, to a reach, the runs that arrive and hold, with the median
    arrival time from the change; to a periodic task, the runs that cycle at the end,
    with the median joining time from the change. Before each change that leaves a
    periodic task, the runs that cycle (see :func:`switch_metrics`).
    """
    runs = [r for r in rows if r["arm"] == "esn"]
    time, task = changes[-1]
    if "switch_success" in runs[0]:
        count, measure, key = sum(r["switch_success"] for r in runs), "arrive and hold", "switch_arrival_time_s"
    else:
        count, measure, key = sum(r["switch_cycles"] for r in runs), "cycle at the end", "switch_joining_time_s"
    values = np.array([r[key] for r in runs], dtype=np.float64)
    median = float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else float("nan")
    when = "arrival" if key == "switch_arrival_time_s" else "joining time"
    print(
        f"\nAfter the change to {names[task]} at {time:g} s: {count} of {len(runs)} ESN runs {measure},"
        f" median {when} {median:.2f} s after the change"
    )
    for k, (time, _) in enumerate(changes, start=1):
        before = [r[f"cycles_before_switch_{k}"] for r in runs]
        if "" not in before:
            print(f"Before the change at {time:g} s: {sum(before)} of {len(runs)} ESN runs cycle")


def progress_source(
    table: dict[str, Any], q_demo: NDArray[np.float64], period: float, loop: tuple[int, int] | None = None
) -> ProgressReplaySource:
    """The progress-based replay of ``q_demo`` that a ``[progress]`` table configures: ``lead_deg``, ``window`` (s).

    A periodic take is replayed without end through its ``loop`` (see
    :class:`arm_esn_ctrl.tracking.ProgressReplaySource`).
    """
    unknown = set(table) - {"lead_deg", "window"}
    if unknown:
        msg = f"[progress] takes lead_deg and window, not {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    return ProgressReplaySource(q_demo, np.radians(table["lead_deg"]), round(table["window"] / period), loop)


def print_taught_loop(setup: Setup, cycle: Cycle, period: float) -> None:
    """Print the taught loop the replays of a periodic task go around, the step where it closes, and the period."""
    (q,) = setup.demos.values()
    first, end = cycle.loop
    step = float(np.degrees(np.linalg.norm(q[end] - q[first])))
    print(
        f"The replays go around the take's taught loop from {first * period:.2f} s to {end * period:.2f} s, then"
        f" again, a step of {step:.3f} deg where it closes; the taught period is {cycle.period:.3f} s"
    )


def push_aim(setup: Setup, cycle: Cycle | None) -> NDArray[np.float64]:
    """Where a push's direction is turned from, from the start hand: a reach's target, or a periodic task's center."""
    return setup.target if cycle is None else cycle.center


@dataclass(frozen=True)
class Setting:
    """One tracker setting: a tracking law, and the natural frequency and damping ratio of its error, or its gains.

    Given gains (``kp`` and ``kd``, one value for every joint or one per joint), the
    natural frequency is None and the damping ratio unused.
    """

    law: str
    omega: float | None  # rad/s; None with given gains
    damping: float = 1.0  # 1 is critically damped
    kp: tuple[float, ...] = ()  # given proportional gains (N m/rad)
    kd: tuple[float, ...] = ()  # given derivative gains (N m s/rad)

    @property
    def name(self) -> str:
        """Its directory in the run: ``pd_w10``, ``pd_w10_z0.3`` below critical damping, or ``pd_kp80_kd3``."""
        if self.omega is None:
            return f"{self.law}_kp{'-'.join(f'{v:g}' for v in self.kp)}_kd{'-'.join(f'{v:g}' for v in self.kd)}"
        return f"{self.law}_w{self.omega:g}" + ("" if self.damping == 1.0 else f"_z{self.damping:g}")

    @property
    def label(self) -> str:
        """Its name in titles and messages."""
        law = self.law.replace("_", " ")
        if self.omega is None:
            return f"{law}, kp = {', '.join(f'{v:g}' for v in self.kp)}, kd = {', '.join(f'{v:g}' for v in self.kd)}"
        return f"{law}, ω = {self.omega:g} rad/s" + ("" if self.damping == 1.0 else f", ζ = {self.damping:g}")

    @property
    def columns(self) -> dict[str, Any]:
        """Its columns in the metrics: the law, the natural frequency, and the damping ratio, empty with given gains."""
        if self.omega is None:
            return {"law": self.law, "omega": "", "damping": ""}
        return {"law": self.law, "omega": self.omega, "damping": self.damping}

    def gains(
        self, config: TrackerConfig, skeleton: Skeleton, posture: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The per-joint gains ``(kp, kd)``: the given ones, or those of ω and ζ at ``posture`` (tracking_gains)."""
        if self.omega is not None:
            return tracking_gains(config, skeleton, posture)
        n = skeleton.num_joints
        if len(self.kp) not in (1, n) or len(self.kd) not in (1, n):
            msg = f"[tracker] kp and kd must each give one value for every joint or one per joint ({n})"
            raise ValueError(msg)
        return np.broadcast_to(self.kp, n).astype(np.float64), np.broadcast_to(self.kd, n).astype(np.float64)


def warmup_steps(evaluation: dict[str, Any], config: EsnConfig) -> int:
    """How many periods the start posture is held before the task: ``[evaluation]``'s ``warmup`` (s), or the ESN's."""
    if "warmup" not in evaluation:
        return config.warmup_steps
    return round(float(evaluation["warmup"]) / config.dt)


def tracker_settings(tracker: dict[str, Any]) -> list[Setting]:
    """Every combination of the laws, natural frequencies, and damping ratios of ``[tracker]``, law by law.

    ``omegas`` lists the natural frequencies of every law, or maps each law to its
    own list. Only one of the natural frequency and the damping ratio may take
    several values, so that the metrics plot against it. Instead of ``omegas`` and
    ``dampings``, ``kp`` and ``kd`` give one law's gains directly: each one number for
    every joint, or a list of one per joint.
    """
    if "kp" in tracker or "kd" in tracker:
        if "kp" not in tracker or "kd" not in tracker:
            msg = "[tracker] kp and kd go together"
            raise ValueError(msg)
        if "omegas" in tracker or "dampings" in tracker:
            msg = "[tracker] gives either omegas (and dampings) or kp and kd, not both"
            raise ValueError(msg)
        if len(tracker["laws"]) != 1:
            msg = "[tracker] kp and kd are the gains of one law"
            raise ValueError(msg)
        kp, kd = (tuple(float(v) for v in np.atleast_1d(tracker[key])) for key in ("kp", "kd"))
        if any(v <= 0.0 for v in kp) or any(v < 0.0 for v in kd):
            msg = "[tracker] kp must be positive and kd not negative"
            raise ValueError(msg)
        return [Setting(tracker["laws"][0], None, kp=kp, kd=kd)]
    laws, omegas, dampings = tracker["laws"], tracker["omegas"], tracker.get("dampings", [1.0])
    by_law: dict[str, list[float]] = omegas if isinstance(omegas, dict) else {law: omegas for law in laws}
    if set(by_law) != set(laws):
        msg = f"[tracker] omegas must give the natural frequencies of exactly the laws {', '.join(laws)}"
        raise ValueError(msg)
    if any(len(values) > 1 for values in by_law.values()) and len(dampings) > 1:
        msg = "[tracker] can sweep the natural frequencies (omegas) or the damping ratios (dampings), not both"
        raise ValueError(msg)
    return [Setting(law, float(omega), float(damping)) for law in laws for omega in by_law[law] for damping in dampings]


def posed(skeleton: Skeleton, q: NDArray[np.float64]) -> Skeleton:
    """A copy of the robot at the joint angles ``q``, at rest."""
    arm = skeleton.clone()
    arm.q = q
    arm.dq = np.zeros_like(q)
    return arm


def ringing_settings(evaluation: dict[str, Any]) -> tuple[float, float]:
    """The cutoff (Hz) and the window (s) of the ringing after a disturbance, from ``[evaluation]``."""
    return evaluation.get("ringing_cutoff_hz", RINGING_CUTOFF_HZ), evaluation.get("ringing_window", RINGING_WINDOW)


def arm_metrics(
    log: StateLog,
    i: int,
    setup: Setup,
    period: float,
    window: list[float],
    spans: list[tuple[float, float]],
    tolerance: float,
    ringing: tuple[float, float],
    following: float,
    cycle: Cycle | None = None,
) -> dict[str, Any]:
    """The metrics of one arm's run from start posture ``i``, compared with the demonstrator's undisturbed reach.

    Besides the reach and hold metrics of Stage 1 (see
    :func:`arm_esn_ctrl.autonomous.run_metrics`), over the task (t >= 0):

    - ``tracking_error_deg``: RMS of the tracking error ``q_ref - q``;
    - ``reference_path_rmse_m``: RMS distance between the hand path of the
      reference (what the ESN generates) and the taught hand path, regardless of
      timing, as ``taught_path_rmse_m`` measures the arm's;
    - ``peak_reference_speed_dps``: the reference's fastest joint speed, which
      shows a jump of the reference (as a jump of a few degrees in one period);
    - ``peak_hand_speed_mps``: the hand's fastest speed, sampled every period;
    - ``settling_time_s``: the time from which the hand stays within the goal
      radius until the end of the run (NaN if it ends outside);
    - ``final_distance_m``: the hand's distance to the target at the end of the run;
    - ``task_torque_nms``, ``task_effort_n2m2s``, and ``task_work_j``: how hard the
      motors work over the whole task, whatever the effort window: the integrals of
      the joint torques' sizes, of their squares, and of the positive work the
      joints do (see :func:`arm_esn_ctrl.metrics.motor_effort`);

    and over the effort window (``effort_window`` in ``[evaluation]``):

    - ``peak_torque_nm``: the largest joint torque;
    - ``effort_n2m2s``: the integral of the squared joint torques, summed over the joints;
    - ``peak_external_force_n``: the largest disturbance force at the tip (a
      block's holding force shows how hard the arm pushes against it);

    and for each disturbance (``spans``, when each acts on the task clock):

    - ``reference_lead``: at its end, how far the reference is ahead of the arm
      along the demonstrated path, as a fraction of the path (see :func:`progress`);
    - ``off_path_m``: the hand's largest distance from the taught hand path from its
      onset until the next disturbance starts, or to the end of the run;
    - ``return_time_s``: the time from its end until the hand is back within
      ``tolerance`` (m) of the taught hand path until the next disturbance starts or
      the run ends: 0 if it never strays farther after the end, NaN if it is not
      back by then (see :func:`arm_esn_ctrl.metrics.disturbance_returns`);
    - ``end_gap_deg``: at its end, how far the reference is from the arm's joint
      angles, the norm of ``q_ref - q``: the error the tracker starts from;
    - ``ringing_m``: how much the hand rings after its end, the RMS size of its
      motion above ``ringing[0]`` (Hz) over ``ringing[1]`` seconds, or until the next
      disturbance starts (see :func:`arm_esn_ctrl.metrics.disturbance_ringing`);
    - ``following_gain``: how much of the hand's displacement its reference's hand
      shares, from its onset until ``following`` seconds past its end, or until the
      next disturbance starts, both taken from where the taught motion is at the same
      time, as the replay's reference: 0 for a reference that keeps to the taught
      motion, as the replay's does, 1 for one that goes wherever the arm goes (see
      :func:`arm_esn_ctrl.metrics.disturbance_following`).

    Without a disturbance, these are NaN. With several, ``reference_lead_1``,
    ``reference_lead_2``, ..., ``off_path_m_1``, ... measure each, in the order
    configured (see :func:`per_disturbance`).

    The demonstrator tracks no reference, so its tracking metrics are NaN.

    A periodic task's run, given its ``cycle``, is measured from the task's curve:
    its ``off_path_m`` and ``return_time_s`` are the hand's distance from the curve
    and its return within ``tolerance`` of it, and :func:`cycle_columns` adds the rest.
    """
    q = task_joint_angles(log, period, setup.times[-1])
    hand = endpoint_positions(setup.skeleton, q)
    q_ref = None if setup.q_refs is None else setup.q_refs[i]
    hand_ref = None if setup.hand_refs is None else setup.hand_refs[i]
    run = Run(setup.starts[i], q, q_ref, hand, hand_ref)
    times = log.times
    task = times > -1e-9
    in_window = (times > window[0] - 1e-9) & (times < window[1] + 1e-9)
    tracked = "q_ref" in log.channel_names
    nan = float("nan")
    tau = log.channel("tau")
    force = log.channel("ext_force") if "ext_force" in log.channel_names else np.zeros((len(times), 2))
    distance = np.linalg.norm(hand - setup.target, axis=1)
    outside = np.flatnonzero(distance > setup.radius)
    settling_time = nan
    if len(outside) == 0:
        settling_time = float(setup.times[0])
    elif outside[-1] < len(distance) - 1:
        settling_time = float(setup.times[outside[-1] + 1])
    peak_reference_speed = nan
    leads = [nan] * len(spans)
    gaps = [nan] * len(spans)
    if tracked:
        peak_reference_speed = reference_speed(times[task], log.channel("q_ref")[task])
        for n, (_, end) in enumerate(spans):
            k = int(np.argmin(np.abs(times - end)))
            ahead = progress(log.channel("q_ref")[k : k + 1], setup.starts[i].q, setup)
            leads[n] = float(ahead[0] - progress(log.channel("q")[k : k + 1], setup.starts[i].q, setup)[0])
            gaps[n] = float(np.degrees(np.linalg.norm(log.channel("q_ref")[k] - log.channel("q")[k])))
    taught_hand = setup.demo_hands[nearest_demonstration(setup.starts[i].q, setup.demos)]
    returns: list[tuple[float, float]] = []
    if spans:  # a periodic task's hand is measured from its curve
        off_path = distances_to_path(hand, taught_hand if cycle is None else cycle.curve)
        returns = disturbance_returns(setup.times, off_path, spans, tolerance)
    reference_hand = None
    followed = [nan] * len(spans)
    if tracked:
        reference_hand = endpoint_positions(setup.skeleton, task_joint_angles(log, period, setup.times[-1], "q_ref"))
        if spans:
            # Where the taught motion is at each time, as the replay's reference: its last posture once it ends,
            # or, of a periodic take, its taught loop again and again.
            taught_now = taught_hand[np.minimum(np.arange(len(setup.times)), len(taught_hand) - 1)]
            if cycle is not None:
                taught_now = taught_hand[[loop_sample(k, cycle.loop) for k in range(len(setup.times))]]
            followed = disturbance_following(setup.times, hand, reference_hand, taught_now, spans, following)
    measured = {
        "reference_lead": leads,
        "off_path_m": [farthest for farthest, _ in returns],
        "return_time_s": [back for _, back in returns],
        "end_gap_deg": gaps,
        "ringing_m": disturbance_ringing(setup.times, hand, spans, *ringing) if spans else [],
        "following_gain": followed,
    }
    disturbance_metrics: dict[str, float] = {}
    for metric, values in measured.items():
        disturbance_metrics |= dict(zip(per_disturbance(metric, len(spans)), values or [nan], strict=True))
    reference_path_rmse = nan if reference_hand is None else path_rmse(reference_hand, taught_hand)
    task_torque, task_effort, task_work = motor_effort(times[task], tau[task], log.channel("dq")[task])
    periodic = {} if cycle is None else cycle_columns(log, hand, setup, spans, cycle)  # in place of a reach's
    return run_metrics(run, setup) | {
        "tracking_error_deg": rms_degrees(log.channel("error")[task]) if tracked else nan,
        "reference_path_rmse_m": reference_path_rmse,
        "peak_reference_speed_dps": peak_reference_speed,
        "peak_hand_speed_mps": float(hand_speed(setup.times, hand).max()),
        "settling_time_s": settling_time,
        "final_distance_m": float(distance[-1]),
        "task_torque_nms": task_torque,
        "task_effort_n2m2s": task_effort,
        "task_work_j": task_work,
        "peak_torque_nm": float(np.abs(tau[in_window]).max()),
        "effort_n2m2s": float(np.sum(tau[in_window] ** 2) * np.diff(times).mean()),
        "peak_external_force_n": float(np.linalg.norm(force[in_window], axis=1).max()),
        **disturbance_metrics,
        **periodic,
    }


def reference_speed(times: NDArray[np.float64], q_ref: NDArray[np.float64]) -> float:
    """The reference's fastest joint speed (deg/s): its largest step in a joint over the shortest step in ``times``.

    ``q_ref`` holds the reference's joint angles at ``times``, a log's every simulation
    step. A jump of the reference shows as a jump of a few degrees in one step.
    """
    return float(np.degrees(np.abs(np.diff(q_ref, axis=0)).max() / np.diff(times).min()))


def cycle_columns(
    log: StateLog, hand: NDArray[np.float64], setup: Setup, spans: list[tuple[float, float]], cycle: Cycle
) -> dict[str, Any]:
    """The metrics of a periodic task's run, its ``hand`` at ``setup``'s times, that :func:`arm_metrics` adds.

    A periodic task has no target: the settling time and each disturbance's
    ``reference_lead`` are NaN, and the arrival and the hold are left empty or NaN
    (see :func:`arm_esn_ctrl.autonomous.cycle_run_metrics`). Then whether the run
    cycles and the rest of the cycle measures, and, for each disturbance (``spans``,
    when each acts on the task clock), over ``cycle.after`` seconds from its end:

    - ``torque_after_nm``: the largest joint torque;
    - ``off_curve_after_m``: the hand's largest distance from the curve.

    Without a disturbance, these are NaN. With several, ``torque_after_nm_1``, ... (see
    :func:`per_disturbance`).
    """
    nan = float("nan")
    torque_after, off_curve_after = [nan], [nan]
    if spans:
        torque = np.abs(log.channel("tau")).max(axis=1)
        torque_after = largest_after(log.times, torque, spans, cycle.after)
        off_curve_after = largest_after(setup.times, distances_to_path(hand, cycle.curve), spans, cycle.after)
    columns: dict[str, Any] = {"settling_time_s": nan}
    columns |= {name: nan for name in per_disturbance("reference_lead", len(spans))}
    columns |= cycle_run_metrics(hand, setup, cycle)
    columns |= dict(zip(per_disturbance("torque_after_nm", len(spans)), torque_after, strict=True))
    columns |= dict(zip(per_disturbance("off_curve_after_m", len(spans)), off_curve_after, strict=True))
    return columns


def per_disturbance(metric: str, count: int) -> list[str]:
    """The columns of a disturbance metric of :func:`arm_metrics` with ``count`` disturbances.

    One column, ``metric``, with one disturbance or none; ``metric_1``, ``metric_2``,
    ... with several.
    """
    return [f"{metric}_{n + 1}" for n in range(count)] if count > 1 else [metric]


def progress(q: NDArray[np.float64], start_q: NDArray[np.float64], setup: Setup) -> NDArray[np.float64]:
    """How far along the demonstrated joint path the joint angles ``q`` are, from 0 at its start to 1 at its end.

    The path is that of the training demonstration that starts nearest to
    ``start_q``. A reference generator that keeps the demonstration's timing, as the
    replay does, advances along it whatever the arm does; one that adapts to the
    arm stays with the arm's progress.
    """
    return path_progress(q, setup.demos[nearest_demonstration(start_q, setup.demos)])


def print_summary(rows: list[dict[str, Any]], n_starts: int, window: list[float], reference: Reference) -> None:
    """Print the metrics of each arm and tracker setting: means over the start postures.

    The joint error and path distance are from the ``reference``: the demonstrator's
    undisturbed reach, or the taught motion. The settling time is the mean over the
    runs that settle.
    """
    print(f"\nMeans over the {n_starts} start postures (torque, effort, force: t = {window[0]:g} to {window[1]:g} s);")
    print(f"joint error and path distance from the {reference.name}")
    # Columns: reach joint error and path distance, hold successes, settling time, final distance, peak reference
    # joint speed, peak hand speed, peak torque, effort, and peak disturbance force.
    names = ["error", "path", "hold", "settle", "final", "ref.", "hand", "torque", "effort", "force"]
    units = ["(deg)", "(mm)", "", "(s)", "(mm)", "(dps)", "(m/s)", "(N m)", "(N2m2s)", "(N)"]
    widths = [6, 6, 7, 7, 7, 7, 6, 7, 9, 7]
    print(f"{'law':<16}{'ω':<6}{'ζ':<5}{'arm':<12}" + "".join(f"{n:>{w}}" for n, w in zip(names, widths, strict=True)))
    print(f"{'':<16}{'rad/s':<6}{'':<17}" + "".join(f"{u:>{w}}" for u, w in zip(units, widths, strict=True)))
    keys = ("law", "omega", "damping", "arm")
    for key in dict.fromkeys(tuple(r[name] for name in keys) for r in rows):
        group = [r for r in rows if tuple(r[name] for name in keys) == key]
        law, omega, damping, arm = key
        m = {name: mean_of(group, name) for name in group[0] if isinstance(group[0][name], float)}
        if law == "":
            setting = "-"
        elif omega == "":  # given gains
            setting = f"{law.replace('_', ' '):<15} {'kp, kd':<10}"
        else:
            setting = f"{law.replace('_', ' '):<15} {omega:<5g} {damping:<4g}"
        print(
            f"{setting:<26} {arm:<12}"
            f" {m[reference.joint_error]:5.2f} {1000 * m[reference.path_distance]:5.1f}"
            f" {sum(r['success'] for r in group):2d}/{n_starts:<3d} {m['settling_time_s']:6.2f}"
            f" {1000 * m['final_distance_m']:6.1f} {m['peak_reference_speed_dps']:6.0f} {m['peak_hand_speed_mps']:5.2f}"
            f" {m['peak_torque_nm']:6.1f} {m['effort_n2m2s']:8.1f} {m['peak_external_force_n']:6.1f}"
        )


def print_cycle_summary(rows: list[dict[str, Any]], n_starts: int) -> None:
    """Print, for each arm and tracker setting of a periodic task, how many runs cycle, and two medians over its runs.

    The medians are of the curve RMS and of the period ratio, a run without a whole
    loop at the end left out of the latter (see :func:`arm_esn_ctrl.metrics.cycle_metrics`).
    """
    print(f"\nOf the {n_starts} start postures: the runs that cycle, and the medians over the runs")
    print(f"{'law':<16}{'ω':<6}{'ζ':<5}{'arm':<12}{'cycle':>7}{'curve RMS':>11}{'period':>8}")
    print(f"{'':<16}{'rad/s':<6}{'':<17}{'':>7}{'(mm)':>11}{'ratio':>8}")
    keys = ("law", "omega", "damping", "arm")
    for key in dict.fromkeys(tuple(r[name] for name in keys) for r in rows):
        group = [r for r in rows if tuple(r[name] for name in keys) == key]
        law, omega, damping, arm = key
        if omega == "":  # given gains
            setting = f"{law.replace('_', ' '):<15} {'kp, kd':<10}"
        else:
            setting = f"{law.replace('_', ' '):<15} {omega:<5g} {damping:<4g}"
        ratios = np.array([r["period_ratio"] for r in group], dtype=np.float64)
        ratio = float(np.median(ratios[np.isfinite(ratios)])) if np.isfinite(ratios).any() else float("nan")
        print(
            f"{setting:<26} {arm:<12} {sum(r['cycles'] for r in group):2d}/{n_starts:<3d}"
            f" {1000 * np.median([r['curve_rms_m'] for r in group]):10.1f} {ratio:7.3f}"
        )


def mean_of(rows: list[dict[str, Any]], name: str) -> float:
    """The mean of a metric over ``rows``, ignoring NaN (NaN if all are)."""
    values = np.array([r[name] for r in rows], dtype=np.float64)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float("nan")


def style(ax: Axes) -> None:
    ax.set_facecolor(SURFACE_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def plot_metrics(
    rows: list[dict[str, Any]],
    laws: list[str],
    n_starts: int,
    window: list[float],
    swept: str,
    reference: Reference,
    title: str,
) -> Figure:
    """Plot each metric's mean over the start postures (and its range) against ``swept``: "omega" or "damping".

    Each law takes two rows of panels. The demonstrator, which has no gains, is a
    dashed level. The settling time is the mean over the runs that settle. The joint
    error, path distance, and arrival delay are from the ``reference``.
    """
    effort = f"over {window[0]:g}-{window[1]:g} s"
    during = "\nduring the reach" if reference.name == "demonstrator" else ""
    metrics = [
        (reference.joint_error, f"Joint error from the {reference.name}{during} (deg RMS)"),
        (reference.path_distance, f"Path distance from the {reference.name}{during} (m)"),
        (reference.arrival_delay, f"Arrival delay after the {reference.name} (s)"),
        ("success", f"Successes: arrive and hold\n(of {n_starts} start postures)"),
        ("settling_time_s", "Settling time: in the goal\nfrom then on (s)"),
        ("final_distance_m", "Final distance to the target (m)"),
        ("peak_reference_speed_dps", "Peak reference joint speed (deg/s)"),
        ("peak_hand_speed_mps", "Peak hand speed (m/s)"),
        ("peak_torque_nm", f"Peak joint torque (N m)\n{effort}"),
        ("effort_n2m2s", f"Integral of squared torque (N² m² s)\n{effort}"),
    ]
    per_row = 5
    x_label = {"omega": "natural frequency ω (rad/s)", "damping": "damping ratio ζ"}[swept]
    fig = Figure(figsize=(17, 6.2 * len(laws) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color="#0b0b0b")
    axes = np.asarray(fig.subplots(2 * len(laws), per_row)).reshape(len(laws), 2 * per_row)
    for law, law_axes in zip(laws, axes, strict=True):
        for (key, label), ax in zip(metrics, law_axes, strict=True):
            style(ax)
            for arm in [arm for arm in TRACKED if any(r["arm"] == arm for r in rows)]:
                group = [r for r in rows if r["law"] == law and r["arm"] == arm]
                xs = sorted({r[swept] for r in group})
                values = [[float(r[key]) for r in group if r[swept] == x] for x in xs]
                if key == "success":
                    low = high = mean = [sum(v) for v in values]
                else:
                    low = [np.nanmin(v) if np.isfinite(v).any() else np.nan for v in values]
                    high = [np.nanmax(v) if np.isfinite(v).any() else np.nan for v in values]
                    mean = [np.nanmean(v) if np.isfinite(v).any() else np.nan for v in values]
                color = ARM_COLORS[arm]
                ax.fill_between(xs, low, high, color=color, alpha=0.15, linewidth=0)
                ax.plot(xs, mean, marker="o", markersize=5, linewidth=2, color=color, label=ARM_LABELS[arm])
                ax.set_xscale("log")
                ax.set_xticks(xs, [f"{x:g}" for x in xs])
                ax.minorticks_off()
            demonstrator = [float(r[key]) for r in rows if r["arm"] == "demonstrator"]
            if np.isfinite(demonstrator).any():
                level = sum(demonstrator) if key == "success" else np.nanmean(demonstrator)
                ax.axhline(
                    level,
                    color=ARM_COLORS["demonstrator"],
                    linestyle="--",
                    linewidth=1.2,
                    label=ARM_LABELS["demonstrator"],
                )
            if key == "success":
                ax.set_ylim(-0.5, n_starts + 0.5)
            ax.set_title(label, color=TEXT_COLOR, fontsize=9)
            ax.set_xlabel(x_label, color=TEXT_COLOR, fontsize=8)
        law_axes[0].set_ylabel(law.replace("_", " "), color="#0b0b0b", fontsize=11)
        law_axes[per_row].set_ylabel(law.replace("_", " "), color="#0b0b0b", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def load_hand(path: Path, setup: Setup) -> tuple[NDArray[np.float64], NDArray[np.float64], StateLog]:
    """The times and hand positions of a saved run, every simulation step, and the log."""
    log = StateLog.load(path)
    return log.times, endpoint_positions(setup.skeleton, log.channel("q").reshape(len(log.times), -1)), log


def by_law(settings: list[Setting]) -> list[list[Setting]]:
    """The settings grouped by law: one row of the figures per law, one column per setting."""
    laws = list(dict.fromkeys(setting.law for setting in settings))
    return [[setting for setting in settings if setting.law == law] for law in laws]


def plot_paths(run_dir: Path, setup: Setup, settings: list[Setting], title: str) -> Figure:
    """Plot the hand paths of every run, over the demonstrator's undisturbed reaches (or the taught motion).

    Faint postures of the arm driven by the ESN from the first start (the first, the
    last, and three between them, spread along its path) show where the arm was. A
    periodic task's curve is drawn where a reach's target would be.
    """
    rows = by_law(settings)
    columns = len(rows[0])
    curve = task_path(setup.task, CURVE_SIDES)
    width = max(3.2 * columns, 4.8)  # wide enough for the arm with a single column
    fig = Figure(figsize=(width, 3.3 * len(rows) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(title, color="#0b0b0b")
    axes = fig.subplots(len(rows), columns, sharex=True, sharey=True, squeeze=False)
    n_starts = len(setup.starts)
    if setup.hand_refs is not None:
        references, reference_label = setup.hand_refs, "demonstrator, undisturbed"
    else:
        references, reference_label = list(setup.demo_hands.values()), "taught motion"
    for row, ax_row in zip(rows, axes, strict=True):
        for setting, ax in zip(row, ax_row, strict=True):
            style(ax)
            for i, hand_ref in enumerate(references):
                label = reference_label if i == 0 else None
                ax.plot(
                    *hand_ref.T,
                    color=ARM_COLORS["demonstrator"],
                    linewidth=4,
                    alpha=0.3,
                    label=label,
                    solid_capstyle="round",
                )
            for arm in [arm for arm in TRACKED[::-1] if (run_dir / setting.name / f"{arm}_00.sklog.npz").exists()]:
                for i in range(n_starts):
                    _, hand, _ = load_hand(run_dir / setting.name / f"{arm}_{i:02d}.sklog.npz", setup)
                    ax.plot(*hand.T, color=ARM_COLORS[arm], linewidth=1.2, label=ARM_LABELS[arm] if i == 0 else None)
            _, hand, log = load_hand(run_dir / setting.name / "esn_00.sklog.npz", setup)
            q = log.channel("q").reshape(len(log.times), -1)
            draw_postures(ax, setup.skeleton, q[spread_along_path(hand)])
            ax.plot(*setup.target, marker="+", markersize=12, color="#0b0b0b", markeredgewidth=1.5)
            if curve is not None:
                ax.plot(*curve.T, color="#0b0b0b", linewidth=0.8, linestyle="--", label="the task's curve")
            ax.set_title(setting.label, color=TEXT_COLOR, fontsize=10)
            ax.set_aspect("equal")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = 3 if columns > 2 else 1  # a narrow figure stacks the legend
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_timeline(
    run_dir: Path, setup: Setup, settings: list[Setting], spans: list[tuple[float, float]], title: str
) -> Figure:
    """Plot the hand's distance to the target, the progress, and the joint torque over time, from the first start.

    The progress along the demonstrated path (see :func:`progress`) is drawn for the
    arms (solid) and for their references (dashed). The shaded bands are when the
    disturbances act.
    """
    rows = by_law(settings)
    columns = len(rows[0])
    fig = Figure(figsize=(3.6 * columns + 1.0, 7.2 * len(rows) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"{title}, from {setup.starts[0].origin}", color="#0b0b0b")
    axes = np.asarray(fig.subplots(3 * len(rows), columns, sharex=True)).reshape(3 * len(rows), columns)
    start_q = setup.starts[0].q
    ref_times, ref_q, ref_label = demonstrated_motion(setup)
    ref_hand = endpoint_positions(setup.skeleton, ref_q)
    demonstrator = run_dir / "demonstrator_00.sklog.npz"
    for row, law_settings in enumerate(rows):
        for col, setting in enumerate(law_settings):
            ax_distance, ax_progress, ax_torque = axes[3 * row : 3 * row + 3, col]
            runs = {
                arm: load_hand(run_dir / setting.name / f"{arm}_00.sklog.npz", setup)
                for arm in TRACKED
                if (run_dir / setting.name / f"{arm}_00.sklog.npz").exists()
            }
            if demonstrator.exists():
                runs["demonstrator"] = load_hand(demonstrator, setup)
            for ax in (ax_distance, ax_progress, ax_torque):
                style(ax)
                for span in spans:
                    ax.axvspan(*span, color=DISTURBANCE_COLOR, zorder=0)
            ax_distance.plot(
                ref_times,
                1000 * np.linalg.norm(ref_hand - setup.target, axis=1),
                color=ARM_COLORS["demonstrator"],
                linewidth=4,
                alpha=0.3,
                label=ref_label,
            )
            undisturbed = progress(ref_q, start_q, setup)
            ax_progress.plot(ref_times, undisturbed, color=ARM_COLORS["demonstrator"], linewidth=4, alpha=0.3)
            for arm in [arm for arm in ARMS if arm in runs]:
                times, hand, log = runs[arm]
                color = ARM_COLORS[arm]
                zorder = ARM_ZORDERS[arm]
                ax_distance.plot(
                    times,
                    1000 * np.linalg.norm(hand - setup.target, axis=1),
                    color=color,
                    linewidth=1.3,
                    label=ARM_LABELS[arm],
                    zorder=zorder,
                )
                q_progress = progress(log.channel("q"), start_q, setup)
                ax_progress.plot(times, q_progress, color=color, linewidth=1.3, zorder=zorder)
                if "q_ref" in log.channel_names:
                    reference_progress = progress(log.channel("q_ref"), start_q, setup)
                    reference_zorder = REFERENCE_ZORDER if arm == "esn" else zorder
                    ax_progress.plot(
                        times, reference_progress, color=color, linewidth=1.3, linestyle="--", zorder=reference_zorder
                    )
                tau = np.linalg.norm(log.channel("tau"), axis=1)
                ax_torque.plot(times, tau, color=color, linewidth=1.3, zorder=zorder)
            ax_distance.axhline(1000 * setup.radius, color="#0b0b0b", linewidth=0.8, linestyle=":")
            ax_distance.set(xlim=(-0.3, float(setup.times[-1])), ylim=(0, None))
            ax_distance.set_title(setting.label, color=TEXT_COLOR, fontsize=10)
            ax_progress.set_ylim(-0.05, 1.05)
            ax_torque.set_xlabel("time (s)", color=TEXT_COLOR, fontsize=8)
            if col == 0:
                ax_distance.set_ylabel("hand to target (mm)", color=TEXT_COLOR)
                ax_progress.set_ylabel("progress along the\ndemonstrated path", color=TEXT_COLOR)
                ax_torque.set_ylabel("joint torque norm (N m)", color=TEXT_COLOR)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(axes[1, 0].plot([], [], color=TEXT_COLOR, linestyle="--", linewidth=1.3)[0])
    labels.append("its reference (progress)")
    ncol = 4 if columns > 2 else columns  # a narrow figure wraps the legend
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def plot_cycle_timeline(
    run_dir: Path, setup: Setup, settings: list[Setting], spans: list[tuple[float, float]], title: str, cycle: Cycle
) -> Figure:
    """Plot a periodic task's timeline from the first start: the hand off the curve, its turns, and the torque.

    The hand's distance from the curve and its turns about the curve's center, counted
    from the run's start, are drawn at the ESN's period, the turns for the arms (solid)
    and for their references (dashed); the taught motion's turns are drawn thick and
    gray. The dotted line is the curve tolerance, which a run's curve RMS must keep
    within; the shaded bands are when the disturbances act.
    """
    rows = by_law(settings)
    columns = len(rows[0])
    fig = Figure(figsize=(3.6 * columns + 1.0, 7.2 * len(rows) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"{title}, from {setup.starts[0].origin}", color="#0b0b0b")
    axes = np.asarray(fig.subplots(3 * len(rows), columns, sharex=True)).reshape(3 * len(rows), columns)
    period, duration = float(setup.times[1] - setup.times[0]), float(setup.times[-1])
    ref_times, ref_q, ref_label = demonstrated_motion(setup)

    def turns(q: NDArray[np.float64]) -> NDArray[np.float64]:
        phase = loop_phase(endpoint_positions(setup.skeleton, q), cycle.center)
        return (phase - phase[0]) / (2 * np.pi)

    for row, law_settings in enumerate(rows):
        for col, setting in enumerate(law_settings):
            ax_distance, ax_turns, ax_torque = axes[3 * row : 3 * row + 3, col]
            for ax in (ax_distance, ax_turns, ax_torque):
                style(ax)
                for span in spans:
                    ax.axvspan(*span, color=DISTURBANCE_COLOR, zorder=0)
            ax_turns.plot(ref_times, turns(ref_q), color=ARM_COLORS["demonstrator"], linewidth=4, alpha=0.3)
            for arm in TRACKED:
                path = run_dir / setting.name / f"{arm}_00.sklog.npz"
                if not path.exists():
                    continue
                log = StateLog.load(path)
                color, zorder = ARM_COLORS[arm], ARM_ZORDERS[arm]
                q = task_joint_angles(log, period, duration)
                off = distances_to_path(endpoint_positions(setup.skeleton, q), cycle.curve)
                ax_distance.plot(
                    setup.times, 1000 * off, color=color, linewidth=1.3, label=ARM_LABELS[arm], zorder=zorder
                )
                ax_turns.plot(setup.times, turns(q), color=color, linewidth=1.3, zorder=zorder)
                reference_zorder = REFERENCE_ZORDER if arm == "esn" else zorder
                q_ref = task_joint_angles(log, period, duration, "q_ref")
                ax_turns.plot(
                    setup.times, turns(q_ref), color=color, linewidth=1.3, linestyle="--", zorder=reference_zorder
                )
                tau = np.linalg.norm(log.channel("tau"), axis=1)
                ax_torque.plot(log.times, tau, color=color, linewidth=1.3, zorder=zorder)
            ax_distance.axhline(1000 * cycle.tolerance, color="#0b0b0b", linewidth=0.8, linestyle=":")
            ax_distance.set(xlim=(-0.3, duration), ylim=(0, None))
            ax_distance.set_title(setting.label, color=TEXT_COLOR, fontsize=10)
            ax_torque.set_xlabel("time (s)", color=TEXT_COLOR, fontsize=8)
            if col == 0:
                ax_distance.set_ylabel("hand off the curve (mm)", color=TEXT_COLOR)
                ax_turns.set_ylabel("turns about the\ncurve's center", color=TEXT_COLOR)
                ax_torque.set_ylabel("joint torque norm (N m)", color=TEXT_COLOR)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(axes[1, 0].plot([], [], color=ARM_COLORS["demonstrator"], linewidth=4, alpha=0.3)[0])
    labels.append(f"{ref_label} (turns)")
    handles.append(axes[1, 0].plot([], [], color=TEXT_COLOR, linestyle="--", linewidth=1.3)[0])
    labels.append("its reference (turns)")
    ncol = 4 if columns > 2 else columns  # a narrow figure wraps the legend
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, labelcolor=TEXT_COLOR)
    return fig


def demonstrated_motion(setup: Setup, start: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64], str]:
    """The motion the runs from start posture ``start`` are compared with: its times, joint angles, and label.

    It is the demonstrator's undisturbed reach from that start, or, without a
    demonstrator, the taught motion nearest to it.
    """
    if setup.demonstrator_logs:
        reference = setup.demonstrator_logs[start]
        return reference.times, reference.channel("q").reshape(len(reference.times), -1), "demonstrator, undisturbed"
    q = setup.demos[nearest_demonstration(setup.starts[start].q, setup.demos)]
    return (setup.times[1] - setup.times[0]) * np.arange(len(q)), q, "taught motion"


def offset_start(setup: Setup, offset: Sequence[float]) -> int:
    """The index of the first start posture offset by ``offset`` (deg) from a demonstrated start, such as (15, 15)."""
    suffix = f" {','.join(f'{value:+g}' for value in offset)} deg"
    for i, start in enumerate(setup.starts):
        if start.origin.endswith(suffix):
            return i
    msg = f"no start posture is offset by {list(offset)} deg; add it to [evaluation]"
    raise ValueError(msg)


def plot_joints(
    run_dir: Path,
    setup: Setup,
    settings: list[Setting],
    spans: list[tuple[float, float]],
    title: str,
    start: int = 0,
) -> Figure:
    """Plot the joint angles over time from start posture ``start`` (the first by default): one column per tracker
    setting, one row per joint.

    Each panel draws the demonstrated motion (thick gray), the ESN's output, the
    reference it gives the tracker (dashed), the arm driven by the ESN, the arm
    replaying the demonstration by time, and the demonstrator's arm when there is
    one. The shaded bands are when the disturbances act.
    """
    ref_times, ref_q, ref_label = demonstrated_motion(setup, start)

    def draw(ax: Axes, logs: dict[str, StateLog], joint: int) -> None:
        ax.plot(
            ref_times,
            np.degrees(ref_q[:, joint]),
            color=ARM_COLORS["demonstrator"],
            linewidth=4,
            alpha=0.3,
            label=ref_label,
        )
        esn = logs["esn"]
        ax.plot(
            esn.times,
            np.degrees(esn.channel("q_ref").reshape(len(esn.times), -1)[:, joint]),
            color=ARM_COLORS["esn"],
            linewidth=1.3,
            linestyle="--",
            label="ESN output (its reference)",
            zorder=REFERENCE_ZORDER,
        )
        for arm in [arm for arm in ARMS if arm in logs]:
            log = logs[arm]
            q = np.degrees(log.channel("q").reshape(len(log.times), -1)[:, joint])
            ax.plot(log.times, q, color=ARM_COLORS[arm], linewidth=1.3, label=ARM_LABELS[arm], zorder=ARM_ZORDERS[arm])

    angles = f"{title}: joint angles"
    return per_joint_figure(run_dir, setup, settings, spans, angles, draw, "joint {joint} (deg)", start)


def plot_torques(
    run_dir: Path,
    setup: Setup,
    settings: list[Setting],
    spans: list[tuple[float, float]],
    title: str,
    start: int = 0,
) -> Figure:
    """Plot the joint torques over time from start posture ``start`` (the first by default): one column per tracker
    setting, one row per joint.

    Each panel draws the torque of the arm driven by the ESN, of the arm replaying
    the demonstration by time, and of the demonstrator's arm when there is one. The
    shaded bands are when the disturbances act.
    """

    def draw(ax: Axes, logs: dict[str, StateLog], joint: int) -> None:
        for arm in [arm for arm in ARMS if arm in logs]:
            log = logs[arm]
            tau = log.channel("tau").reshape(len(log.times), -1)[:, joint]
            ax.plot(
                log.times, tau, color=ARM_COLORS[arm], linewidth=1.0, label=ARM_LABELS[arm], zorder=ARM_ZORDERS[arm]
            )

    torques = f"{title}: joint torques"
    return per_joint_figure(run_dir, setup, settings, spans, torques, draw, "joint {joint} (N m)", start)


def per_joint_figure(
    run_dir: Path,
    setup: Setup,
    settings: list[Setting],
    spans: list[tuple[float, float]],
    title: str,
    draw: Callable[[Axes, dict[str, StateLog], int], None],
    ylabel: str,
    start: int = 0,
) -> Figure:
    """A figure of one panel per tracker setting (columns, one row of panels per law) and joint, from start ``start``.

    ``draw(ax, logs, joint)`` draws a panel from the logs of the arms (``esn``,
    ``replay``, and ``progress`` and ``demonstrator`` when there are); ``ylabel`` names the joint's
    axis, with ``{joint}`` its number. The shaded bands are when the disturbances act.
    """
    rows = by_law(settings)
    columns = len(rows[0])
    n_joints = setup.skeleton.num_joints
    width = max(4.6 * columns + 1.0, 9.0)  # wide enough for the title with a single column
    fig = Figure(figsize=(width, 2.6 * n_joints * len(rows) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"{title}, from {setup.starts[start].origin}", color="#0b0b0b")
    axes = np.asarray(fig.subplots(n_joints * len(rows), columns, sharex=True)).reshape(n_joints * len(rows), columns)
    demonstrator = run_dir / f"demonstrator_{start:02d}.sklog.npz"
    for row, law_settings in enumerate(rows):
        for col, setting in enumerate(law_settings):
            logs = {
                arm: StateLog.load(run_dir / setting.name / f"{arm}_{start:02d}.sklog.npz")
                for arm in TRACKED
                if (run_dir / setting.name / f"{arm}_{start:02d}.sklog.npz").exists()
            }
            if demonstrator.exists():
                logs["demonstrator"] = StateLog.load(demonstrator)
            for joint in range(n_joints):
                ax = axes[n_joints * row + joint, col]
                style(ax)
                for span in spans:
                    ax.axvspan(*span, color=DISTURBANCE_COLOR, zorder=0)
                draw(ax, logs, joint)
                ax.set_xlim(-0.3, float(setup.times[-1]))
                ax.set_title(f"{setting.label}: joint {joint + 1}", color=TEXT_COLOR, fontsize=9)
                if col == 0:
                    ax.set_ylabel(ylabel.format(joint=joint + 1), color=TEXT_COLOR)
            axes[n_joints * row + n_joints - 1, col].set_xlabel("time (s)", color=TEXT_COLOR, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, labelcolor=TEXT_COLOR)
    return fig


OUTCOME_COLORS = ["#2a78d6", "#eb6834", "#52514e"]  # arrive and hold, leave the goal, never arrive
OUTCOMES = ["arrive and hold", "leave the goal", "never arrive"]
# The maps of grid.png: metric, title, factor to its display unit, and the top of a
# linear color scale from 0, or None for a logarithmic scale over the values.
# The first two compare with the reference, the demonstrator or the taught motion (see grid_maps).
GRID_MAPS: list[tuple[str, str, float, float | None]] = [
    ("peak_reference_speed_dps", "Peak reference joint speed (deg/s)", 1.0, None),
    ("peak_torque_nm", "Peak joint torque (N m)", 1.0, None),
    ("tracking_error_deg", "Tracking error (deg RMS)", 1.0, None),
]


def grid_maps(reference: Reference) -> list[tuple[str, str, float, float | None]]:
    """The maps of grid.png: the two that compare with the ``reference``, then :data:`GRID_MAPS`."""
    where = "demonstrator's reach" if reference.name == "demonstrator" else "taught motion"
    course = reference.course_label.replace(": ", ":\n", 1)
    return [
        (reference.path_distance, f"Path distance from the\n{where} (mm)", 1000.0, None),
        (reference.course, course, 1.0, 1.5),
        *GRID_MAPS,
    ]


def plot_grid(rows: list[dict[str, Any]], setup: Setup, settings: list[Setting], title: str) -> Figure:
    """Map the outcome and the main metrics over the start offsets: one row per tracker setting and arm.

    Each metric's colors are shared by its column, so the ESN and the replays compare.
    """
    offsets = {}
    for i, start in enumerate(setup.starts):
        demonstrated = setup.demos[nearest_demonstration(start.q, setup.demos)][0]
        offsets[i] = tuple(np.round(np.degrees(start.q - demonstrated), 6))
    layout = OffsetMap.of(list(offsets.values()))
    extent, (xlabel, ylabel) = layout.extent, layout.labels()
    panels = [(setting, arm) for setting in settings for arm in TRACKED if any(r["arm"] == arm for r in rows)]
    maps = grid_maps(setup.reference)
    columns = 1 + len(maps)
    fig = Figure(figsize=(3.3 * columns, 3.0 * len(panels) + 0.8), facecolor=SURFACE_COLOR, layout="constrained")
    fig.suptitle(f"{title}: start offsets", color="#0b0b0b")
    axes = fig.subplots(len(panels), columns, squeeze=False)

    def grid_of(group: list[dict[str, Any]], key: str, factor: float) -> NDArray[np.float64]:
        return layout.grid({offsets[r["start"]]: float(r[key]) * factor for r in group})

    groups = [
        [r for r in rows if {key: r[key] for key in s.columns} == s.columns and r["arm"] == arm] for s, arm in panels
    ]
    for ax_row, (setting, arm), group in zip(axes, panels, groups, strict=True):
        outcome = [{**r, "outcome": 0 if r["success"] else 1 if r["arrived"] else 2} for r in group]
        ax_row[0].imshow(
            grid_of(outcome, "outcome", 1.0),
            origin="lower",
            extent=extent,
            cmap=ListedColormap(OUTCOME_COLORS),
            vmin=-0.5,
            vmax=2.5,
        )
        ax_row[0].set_ylabel(f"{setting.label}\n{ARM_LABELS[arm]}\n{ylabel}", color=TEXT_COLOR, fontsize=8)
        ax_row[0].set_title("Outcome", color=TEXT_COLOR, fontsize=9)
    handles = [Patch(color=color, label=text) for color, text in zip(OUTCOME_COLORS, OUTCOMES, strict=True)]
    axes[0, 0].legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.8)

    for column, (key, label, factor, top) in enumerate(maps, start=1):
        data = [grid_of(group, key, factor) for group in groups]
        finite = np.concatenate([d[np.isfinite(d)] for d in data])
        if top is not None:
            norm: Normalize = Normalize(vmin=0.0, vmax=top)
        elif finite.size:
            norm = LogNorm(vmin=max(float(finite.min()), 1e-3), vmax=max(float(finite.max()), 1e-3) * 1.0001)
        else:
            norm = Normalize(0.0, 1.0)
        for ax, grid in zip(axes[:, column], data, strict=True):
            ax.imshow(grid, origin="lower", extent=extent, cmap="Blues", norm=norm)
            ax.set_title(label, color=TEXT_COLOR, fontsize=9)
        extend = "max" if top is not None else "neither"
        fig.colorbar(
            ScalarMappable(norm=norm, cmap="Blues"), ax=axes[:, column].tolist(), location="bottom", extend=extend
        )
    for ax in axes.flat:
        ax.plot(*layout.zero, marker="+", markersize=10, color="#0b0b0b", markeredgewidth=1.5)
        ax.tick_params(labelsize=7)
        layout.decorate(ax, fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=8)
    return fig


if __name__ == "__main__":
    main()
