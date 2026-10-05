# 002 The ESN as the reference generator of a robot arm

**Summary.** The ESN of [report 001](../001-autonomous-reaching/README.md) was
connected to a simulated arm. Every 10 ms it receives the arm's measured joint
angles and gives the posture the arm should have next, which a tracking
controller follows. It was compared with the same demonstrations replayed by time,
through the same controller, in five scenarios: no disturbance, a push, a block,
and start postures 3° and 10° away from the demonstrated ones.

- **Initial offsets: a clear win.** From an offset start, the ESN reaches from
  where the arm is, with peak torques of 2–7 N m at every gain. The replay first
  yanks the arm to the demonstrated start, with peak torques of 65–2480 N m
  growing with the gain.
- **Push and block: it gives way.** The ESN needs less corrective effort than the
  replay: about a third of it while the arm is blocked. But it strays up to twice
  as far from the demonstrated path after a push. After a block, it overshoots the
  target and oscillates around it, so most runs fail to hold.
- **Gains:** with computed torque, any of the natural frequencies tried
  (5–40 rad/s) works without disturbances. With joint PD, the ESN needs higher
  gains than the replay to hold at the target.

## 1. Question

When the ESN is fed the robot's measured posture instead of its own prediction,
is it a more robust reference generator than the demonstration replayed by time
(research question RQ4 of the [project README](../../README.md#research-questions))?
And since its reference stays close to where the arm actually is, does it make
careful tuning of the tracker's gains unnecessary?

## 2. Setup

### The arm and the reference

The robot is the simulated two-link arm of report 001, integrated every 2 ms. The
reference is updated every 10 ms, the ESN's period:

1. At each instant t_k, a **reference source** receives the arm's measured joint
   angles q(t_k) and gives the posture the arm should have 10 ms later, q̂(k+1).
2. Between two instants, the reference moves in a straight line from q̂(k) to
   q̂(k+1). Its desired velocity is that line's slope, and its desired
   acceleration is the change of the slope, low-pass filtered (time constant
   0.02 s).
3. A **tracking controller** turns the reference into joint torques every 2 ms:
   - *computed torque*, which cancels the arm's dynamics using an exact model;
   - *joint PD*, which does not.

The gains come from one natural frequency ω of the tracking error, critically
damped: kp = ω² and kd = 2ω for computed torque, scaled by each joint's inertia
(at the posture where the reaches end) for joint PD. Every scenario runs both
laws at ω = 5, 10, 20, and 40 rad/s.

### Three arms

From each start posture, three arms reach:

| Arm | Reference | Controller |
| --- | --- | --- |
| **ESN** | the ESN of report 001, driven by the measured joint angles | the tracker |
| **replay** | the demonstration whose start is nearest, replayed by time (the *time-indexed baseline*) | the tracker |
| **demonstrator** | none | the controller that made the demonstrations, on its own |

The ESN and the replay differ only in where the next posture comes from: the ESN
generates it from the measured state, while the replay reads it off a clock and
ignores the state. Both tracked arms first hold their start posture for the ESN's
0.25 s warm-up, at negative times. The task starts at t = 0. The ESN was trained
in report 001 (`esn.toml` and `esn.rclib` in
[`data/20261002-213015-autonomous_tvs_all_distances`](data/20261002-213015-autonomous_tvs_all_distances));
nothing is trained here.

### Scenarios

| Scenario | Start postures | What happens |
| --- | ---: | --- |
| nominal | 8 | no disturbance, from the demonstrated starts |
| push | 8 | a 5 N force at the tip for 0.1 s from t = 0.4 s, near the peak speed, sideways to the reach |
| block | 8 | from t = 0.3 s to 0.8 s, a stiff spring-damper (20 kN/m, 100 N s/m) holds the tip where it was, like a hand gripping the arm; then it lets go |
| offset 3° | 32 | starts 3° away from each demonstrated start in both joints, in the four diagonal directions; no force |
| offset 10° | 32 | the same, 10° away |

All three arms meet the same disturbance. The push was set at 5 N after a probe
from one start: 20 N pushed the hand 11–75 cm off its path at ω ≤ 10 rad/s, up to
more than the 50 cm reach itself. At ω = 40 rad/s, the trackers push against the
block with peak forces of up to 640 N (means over the starts), and in the probe
the spring yielded 13–20 mm.

### Metrics

Every run is compared with the **demonstrator's undisturbed reach** from the same
start posture, using the reach and hold metrics of
[report 001](../001-autonomous-reaching/README.md#metrics): the path distance from
it until arrival, the joint error, the arrival delay, and whether the hand
**arrives and holds** within the 2 cm goal radius for 2 s. In addition:

- **peak reference joint speed**: a jump of the reference shows as a high speed;
- **peak hand speed**;
- over an **effort window**: the peak joint torque, the integral of the squared
  joint torques (∫τ²), and the peak disturbance force. The window starts with
  the disturbance and lasts 1 s: 0.4–1.4 s for the push, 0.3–1.3 s for the
  block, and 0–1 s for the offsets. For the nominal runs, it is the whole 5 s.

### Reproducing the results

The runs used the demonstrations and the ESN in [`data/`](data). Under the
storage root, they are read from `results/20261002-194644-reach_tvs` and
`results/20261002-213015-autonomous_tvs_all_distances`; place or link the copies
there to rerun.

| Run | Command |
| --- | --- |
| [`20261005-120454-nominal`](results/20261005-120454-nominal) | `uv run python experiments/robot_esn.py configs/robot/nominal.toml` |
| [`20261005-120457-push`](results/20261005-120457-push) | `uv run python experiments/robot_esn.py configs/robot/push.toml` |
| [`20261005-120459-block`](results/20261005-120459-block) | `uv run python experiments/robot_esn.py configs/robot/block.toml` |
| [`20261005-120501-offset_3deg`](results/20261005-120501-offset_3deg) | `uv run python experiments/robot_esn.py configs/robot/offset_3deg.toml` |
| [`20261005-120503-offset_10deg`](results/20261005-120503-offset_10deg) | `uv run python experiments/robot_esn.py configs/robot/offset_10deg.toml` |
| [`summary`](results/summary) | `uv run python reports/002-esn-reference-on-the-robot/make_figures.py --animations` |

All five runs are from commit `2d5b567` (recorded in each `run.toml`) and are
deterministic. Each run directory holds its configuration, its run record, and its
per-run metrics (`metrics.csv`). [`make_figures.py`](make_figures.py) draws the
summary figure and prints the tables of Section 3.1 from those metrics. With
`--animations`, it exports the animations of skelarm's player from the run logs
under the storage root; the logs are not kept in Git.

## 3. Results

### 3.1 All scenarios at a glance

![The three arms across the scenarios](results/summary/summary.png)

**Figure 1.** Means over the start postures, at two representative gains:
ω = 10 rad/s for computed torque (top), and ω = 20 rad/s for joint PD (bottom),
the lowest at which the ESN holds every time without disturbances. The torque metrics cover each
scenario's effort window, so compare the arms within a scenario rather than the
scenarios with each other. The undisturbed demonstrator is its own reference, so
its path distance (0 mm) is off the log scale.

Computed torque, ω = 10 rad/s (ESN / replay / demonstrator):

| Scenario | Starts | Path distance (mm) | Arrive and hold (%) | Peak torque (N m) | ∫τ² (N² m² s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| nominal | 8 | 1.8 / 0.6 / 0.0 | 100 / 100 / 100 | 2.5 / 2.2 / 2.0 | 2.5 / 2.3 / 2.2 |
| push | 8 | 75.5 / 41.3 / 14.3 | 100 / 100 / 100 | 3.4 / 4.5 / 4.5 | 2.6 / 3.1 / 3.3 |
| block | 8 | 49.9 / 24.8 / 9.9 | 25 / 100 / 100 | 21.9 / 36.0 / 54.2 | 142 / 407 / 724 |
| offset 3° | 32 | 4.1 / 70.9 / 0.0 | 100 / 100 / 100 | 3 / 376 / 2 | 2 / 1767 / 2 |
| offset 10° | 32 | 12 / 192 / 0 | 100 / 100 / 100 | 4 / 1165 / 2 | 3 / 17133 / 2 |

Joint PD, ω = 20 rad/s (ESN / replay / demonstrator):

| Scenario | Starts | Path distance (mm) | Arrive and hold (%) | Peak torque (N m) | ∫τ² (N² m² s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| nominal | 8 | 8.2 / 3.2 / 0.0 | 100 / 100 / 100 | 2.5 / 2.5 / 2.0 | 2.7 / 2.5 / 2.2 |
| push | 8 | 23.1 / 18.7 / 14.3 | 100 / 100 / 100 | 4.6 / 5.4 / 4.5 | 3.3 / 3.9 / 3.3 |
| block | 8 | 74.5 / 31.7 / 9.9 | 38 / 100 / 100 | 83 / 138 / 54 | 1486 / 4512 / 724 |
| offset 3° | 32 | 9.7 / 71.7 / 0.0 | 100 / 100 / 100 | 3 / 261 / 2 | 2 / 610 / 2 |
| offset 10° | 32 | 17 / 197 / 0 | 100 / 100 / 100 | 3 / 839 / 2 | 3 / 6341 / 2 |

The metrics of every run and gain are in each run's `metrics.csv`.

### 3.2 Without disturbances: how much the gains matter

![Nominal metrics against the natural frequency](results/20261005-120454-nominal/metrics.png)

**Figure 2.** Nominal runs: each metric's mean over the 8 starts (line) and range
(band) against ω, for computed torque (top two rows) and joint PD (bottom two
rows). Blue: ESN; orange: replay; dashed: demonstrator.

- **Computed torque:** both tracked arms reproduce the demonstrations at every ω,
  and all hold. The ESN's joint error ranges from 0.38° at ω = 5 to 0.02° at
  ω = 40, and its path distance from 3.9 to 0.3 mm: two to three times the
  replay's at every ω.
- **Joint PD:** the arm overshoots the target at the end of the reach, since PD
  does not compensate the arm's dynamics. The replay holds from ω = 10 on. The
  ESN needs ω = 20: at ω = 10, only 3 of 8 runs hold, and at ω = 5, 2 of 8 (as
  for the replay). In the failing runs, the ESN's reference follows the
  overshooting arm out of the goal instead of pulling it back. The arm and the
  ESN then oscillate slowly around the target, leaving the 2 cm radius and ending
  within 5 mm of the target.

### 3.3 Initial offsets: the ESN starts from where the arm is

![Offset 10° timeline](results/20261005-120503-offset_10deg/timeline.png)

**Figure 3.** From a start 10° away from demonstration 0 in both joints: the
hand's distance to the target (upper rows) and the joint torque (lower rows) over
time, for every law and ω. The thick gray line is the demonstrator's reach from
this start.

| ESN | replay |
| :---: | :---: |
| ![ESN from an offset start](results/summary/offset_10deg_esn_00.gif) | ![Replay from an offset start](results/summary/offset_10deg_replay_00.gif) |

**Animation 1.** The same start, computed torque at ω = 10 rad/s, in real time.
The purple dot is the target.

This offset puts the hand 180 mm from the target instead of 500 mm. The replay
reads the demonstration's next posture off the clock, so at t = 0 its reference
jumps back to the demonstrated start. The arm follows at up to 10.8 m/s, with up
to 1270 N m, before it reaches. The ESN reaches from where the arm is, much like
the demonstrator from the same start, with a peak torque of 5 N m. Over all
offset starts, the ESN's results barely change with the gains, while the replay's
torque grows with them (Figure 1; peak torques of 950–2480 N m at 10° for
computed torque as ω rises from 5 to 40 rad/s). With joint PD at ω ≤ 10, though,
the ESN fails to hold from many offset starts, as in Section 3.2 (8–12 of 32
hold, against all of the replay's at ω = 10).

### 3.4 Push: the ESN gives way

![Push metrics against the natural frequency](results/20261005-120457-push/metrics.png)

**Figure 4.** Push runs, as in Figure 2. Torque, ∫τ², and force are measured
from the push for 1 s.

After the push, the replay's reference stays where it was and pulls the arm back
onto the demonstrated path. The ESN instead generates its next postures from
where the push has put the arm. Its path ends up 1.2–2 times as far from the
demonstrated path as the replay's (75 against 41 mm at computed torque,
ω = 10 rad/s). It needs 3–18% less effort (∫τ²) than the replay at every gain
but one: at computed torque, ω = 5 rad/s, it needs 26% more. At low
gains, it again fails to hold more often than the replay (5 of 8 at computed
torque, ω = 5 rad/s, and 3 of 8 at joint PD, ω = 10 rad/s, against 8 of 8).

### 3.5 Block: less effort, but no waiting and no settling

![Block timeline](results/20261005-120459-block/timeline.png)

**Figure 5.** From demonstration 0's start, with the block shaded: the hand's
distance to the target (upper rows) and the joint torque (lower rows).

| ESN | replay | demonstrator |
| :---: | :---: | :---: |
| ![ESN under a block](results/summary/block_esn_00.gif) | ![Replay under a block](results/summary/block_replay_00.gif) | ![Demonstrator under a block](results/summary/block_demonstrator_00.gif) |

**Animation 2.** The same start, computed torque at ω = 10 rad/s, in real time.
The red arrow is the gripping force; each animation scales it by its own largest
force, so compare the arrows only within an animation.

- **Less effort.** While the arm is held, both references run ahead of it, and
  the tracker pushes harder the further they get. The ESN's reference runs about
  13° ahead by the release, against 23° for the replay, so the ESN fights the block
  with about a third of the replay's ∫τ² at every gain (142 against 407 N² m² s
  at computed torque, ω = 10 rad/s). It also jumps less on release at high gains
  (a peak hand speed of 3.3 against 5.0 m/s at ω = 40 rad/s).
- **No waiting.** The ESN does not wait for the arm. Its reference moves on about
  60% as far as the replay's. The ESN was never trained on a reach that stalls
  midway, and its slow leak rate gives the reservoir a memory of roughly 0.2 s,
  which likely carries the reach on.
- **No settling.** After the release, the ESN overshoots the target and oscillates
  around it (Figure 5). Only 0–3 of 8 runs hold at any gain or law, against all
  of the replay's and the demonstrator's. Every run still arrives and ends within
  7 mm of the target.

## 4. Observations

- **The ESN's reference follows the measured state.** That is its strength and
  its weakness. Wherever the arm is, the ESN generates a reach from there:
  - **Strength:** it never jumps, and it fights a disturbance less (initial
    offsets, the push, the block).
  - **Weakness:** it does not pull the arm back onto the demonstrated path, and
    once the arm's state leaves what the ESN saw in training (an overshoot under
    joint PD, or the release after a block), it can oscillate instead of settling.
- **On the gains.**
  - **Where the hypothesis holds:** with computed torque and without large
    disturbances, a rough choice of ω is enough, and from offset starts the
    ESN's results hardly depend on ω at all.
  - **Where it fails:** the hypothesis that the ESN makes gain tuning unnecessary
    does not hold in general. With joint PD, the ESN needs higher gains than the
    replay, because it amplifies the arm's overshoot instead of correcting it.
    After a block, it fails to settle at every gain.
- **Every failure is a failure to hold, not to arrive.** Every run of every arm
  arrives. The ESN's failing runs leave the 2 cm goal radius while oscillating,
  and end within 7 mm of the target.
- **Caveats.**
  - **The replay is a naive baseline.** It ignores the start posture, so its jump
    at t = 0 exaggerates the cost of an offset. A replay blended in from the
    actual start would narrow the gap.
  - **The replayed demonstration is the nearest one.** At 10°, 6 of the 32 starts
    are nearer a neighboring demonstration's start than to their own, and replay
    the neighbor's.
  - **The disturbances are single choices.** One push size, one block duration,
    and one block stiffness. The spring yields 13–20 mm at ω = 40 rad/s.
  - **One ESN, one seed, in simulation.** Computed torque used the exact model of
    the arm.

## 5. Next steps

- **Teaching the ESN to return.** Both of its failures share one cause: the ESN
  has only seen inputs on the demonstrated trajectories, so off them it does not
  produce a flow back toward them. Training with noise added to its inputs (noisy
  teacher forcing) is a standard way to teach that return. It may also stiffen
  the reference and cost some of the compliance that helped here. This is open
  for discussion before any experiment.
- **Waiting under a block** may need other data, such as demonstrations that
  pause midway, or a shorter reservoir memory.
- **A fairer baseline:** a time-indexed replay that blends in from the actual
  start posture.
- **Robustness:** more reservoir seeds and ESNs, other push sizes and block
  durations, and a stiffer block at high gains.
