# Manual-demonstration consistent search, v2 (Stage 2)

Results: report 017 (the search), with report 018 (the trackers) and report 019 (the
ESNs as dynamical systems), to be written.

Reports 005 to 016 changed their assumptions as they went: the tracker the search was
run with (computed torque and joint PD at ζ = 0.1 with a zero reference velocity and
the torque every 2 ms up to report 007, joint PD at ζ = 0.4 with the reference
velocity and the torque held 10 ms from report 009), the pushes (0.2 s long in
reports 005 and 006, 0.5 s at 3, 7, and 15 s from report 007), the measure the
search ranked by (the path RMSE, the ringing, the ringing and the return, the
following), whether a run had to hold the goal under the pushes, whether and from
how many offset starts the ESNs were searched (none, 8 for some settings, 49 only in
validation), how a hand that never came back was counted, and how the final ESN was
chosen: by hand in the robot app in reports 005, 007, 009, and 011. This study runs
the main results again under one set of assumptions, fixed here before any run, in
three parts: the search on the robot (report 017), the chosen ESNs across trackers
(report 018), and the chosen ESNs as dynamical systems (report 019). This directory
is the first part, and its README holds the assumptions of all three.

## The protocol

Every part follows it; nothing below changes once the runs start.

**The robot, the task, and the take.** The arm of
[`reach_manual_v2.toml`](../demonstrations/reach_manual_v2.toml), as in reports 005 to
016: two links, the target at (−0.6, 0.4) m with a goal radius of 20 mm, simulated
every 2 ms. The ESNs are trained on its one take, untrimmed, filtered at 2 Hz:
`results/demonstrations/20261008-170034-reach_manual_v2`, `demo_00_filtered`.

**The ESNs.** Trained by teacher forcing on the take, as `sweep_robot_esn.py` and
`autonomous_esn.py` train them, with these fixed:

| Parameter | Value |
| --- | --- |
| Neurons | 400 |
| Sparsity (nonzero reservoir connections) | 0.05 |
| Bias | yes, random |
| Ridge | 10⁻² |
| Warm-up (the start posture held before the ESN moves, in training and on the robot) | 1 s |
| Sampling period | 10 ms |
| Teacher noise | none |

and these swept, every combination, 504 settings:

| Parameter | Values |
| --- | --- |
| Leak rate | 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5 |
| Spectral radius | 0.7, 0.9, 0.99, 1.2, 1.35, 1.5, 1.75 |
| Input scaling | 0.1, 0.2, 0.5, 1, 1.5, 2 |
| Seed (the random reservoir, its input weights, and its biases) | 0 to 9 for the search; 10 to 19 held out |

5,040 ESNs for the search, and 5,040 held out. The grid holds report 011's 105
settings (leak rates 0.015 to 0.1, spectral radii 0.99 to 1.75, input scalings 1 to
2), so that their runs here repeat report 011's.

**The search's tracker.** Joint PD at ω = 20 rad/s and a damping ratio of 0.4, each
joint's gains scaled by its inertia at the take's final posture (kp = 74.96 and
8.96 N m/rad, kd = 3.00 and 0.358 N m s/rad), given the reference's velocity, its
torque computed every 10 ms, the ESN's period, and held between: report 009's and
011's search tracker. The robot app sets the same gains (since `e7ad339`).

**The scenarios.** Runs of 24 s after the 1 s warm-up, with a hold window of 2 s:

| Scenario | What happens |
| --- | --- |
| undisturbed | from the take's start |
| block | the tip held by a spring of 20,000 N/m and a damper of 100 N s/m from 4.5 s to 5.5 s, from the take's start (report 005's) |
| pushes | 3 N at 3 s, 10° from the direction toward the target, counterclockwise; 5 N at 7 s, at 80°; 4 N at 15 s, after the hold, at −80°; each 0.5 s at the tip, from the take's start (report 007's) |
| offsets | undisturbed from the 49 starts offset from the take's by −15° to 15° in steps of 5° in each joint, the take's start among them |

**The measures,** of each run (`robot_esn.arm_metrics`), as reports 006 to 016
defined them: whether the hand arrives within the goal radius, holds there, and ends
within it; the arm's path RMSE from the taught hand path; the first step; the peak
torque; and after each disturbance, the reference lead, the farthest from the taught
path, the return onto it (within 5 mm), the ringing (the hand's motion above 1.5 Hz
over 2 s), and the following gain (from the onset to 0.5 s past the end). A hand
not back by the end of a disturbance's window (when the next one starts, or the run
ends) counts as back then. New here: how hard the motors work over the whole task,
the integrals of the joint torques' sizes (N m s), of their squares (N² m² s, a
stand-in for the motors' copper losses), and of the positive work the joints do (J),
none recovered while a joint is driven backward (see `metrics.motor_effort`).

**The rule** that chooses the ESNs, coded in
[`select_esns.py`](../select_esns.py), report 011's with the offset starts counted for
every ESN:

1. **An ESN passes** if its hand arrives, holds for the whole hold window, and ends
   within the goal undisturbed, under the block, and from all 49 offset starts; and
   under the pushes, if it arrives and ends within the goal (leaving it during the
   hold is excused there, since the push at 7 s can carry the arm into the goal
   early, still swinging).
2. **A setting qualifies** if all ten of its seeds pass. Its measures are the medians
   over its seeds; a seed's following, ringing, and return are its means over the four
   disturbances, the block's and the three pushes'.
3. **The guard:** a setting whose median ringing exceeds the replay's with the same
   tracker, its mean over the four disturbances, is not chosen.
4. **The ranking:** the qualifying settings within the guard, by their median
   following, the largest first.
5. **The typical seed** of a setting: the one nearest its medians of the undisturbed
   path RMSE, the following, and the ringing, each scaled by its interquartile range
   over the seeds.
6. **ESN #1 and ESN #2** are the typical seeds of the top two settings. Nothing is
   chosen in the robot app; the app only shows them. If fewer than two settings
   qualify, the rest come from the settings of which at least nine seeds pass, ranked
   alike, the typical seed among the passing ones.

**The comparison arms** on the robot: the take replayed by time (the time-indexed
baseline), the take replayed by the arm's progress along its joint path (report 014's
progress replay: never more than 2.5° along the path past the arm's progress, its
nearest sample searched within 1 s of the last one found), and, in part 2, the replay
with computed torque.

## Part 1: the search (report 017)

### Step 0: the regression check

Before the grid, the current code runs again what reports 011 and 015 ran, and its
numbers are compared with theirs: report 011's sweep at leak rate 0.03 under the
pushes, as configured there; its ESN #1 and the replay with the search's tracker
under the block and under the pushes; and report 015's fixed points, as configured
there. The two runs of ESN #1 also give the rule's guard, the replay's ringing with
the search's tracker: the replay does not depend on the ESN it runs beside.

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_following_search/sweep_robot_following_lr0.03_pushes_filtered.toml --workers 30
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_consistent_search/guard_block_robot.toml
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_consistent_search/guard_pushes_robot.toml
uv run python experiments/fixed_points.py experiments/manual_demonstration_v2_esn_dynamics/fixed_points.toml
```

| Configurations | What they run |
| --- | --- |
| `guard_{block,pushes}_robot.toml` | report 011's ESN #1 and the replay with the search's tracker, under the block and under the pushes |

### Step 1: the grid

[`sweep_robot_esn.py`](../sweep_robot_esn.py) trains every ESN of one leak rate and
runs it with the search's tracker in one scenario, its ESNs spread over worker
processes:

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_consistent_search/sweep_lr0.03_pushes.toml --workers 30
```

| Configurations | What they sweep |
| --- | --- |
| `sweep_lr<leak rate>_{nominal,block,pushes,offsets}.toml` (48) | spectral radius 0.7 to 1.75 × input scaling 0.1 to 2 × seeds 0 to 9, at one leak rate, in one scenario |
| `sweep_lr<leak rate>_{nominal,block,pushes,offsets}_seeds10to19.toml` (48) | the same with the held-out seeds 10 to 19 |

Each ESN runs 52 times: once undisturbed, under the block, and under the pushes, and
from the 49 offset starts.

### Step 2: the choice

[`select_esns.py`](../select_esns.py) applies the rule to the runs of step 1 with
seeds 0 to 9, listed in `select.toml` once they have run, and chooses ESN #1 and
ESN #2; `select_seeds10to19.toml` applies it to the held-out seeds alone:

```bash
uv run python experiments/select_esns.py experiments/manual_demonstration_v2_consistent_search/select.toml
```

The report also ranks the qualifying settings by the measures the earlier searches
ranked by (the path RMSE, the ringing, and the return), to show how much the choice
depends on the measure, and asks of the held-out seeds whether the qualifying
settings qualify again and keep their order (the rank correlation of the following).
The two ESNs are then trained by [`autonomous_esn.py`](../autonomous_esn.py):

```bash
uv run python experiments/autonomous_esn.py experiments/manual_demonstration_v2_consistent_search/esn1_lr0.03_sr1.2_is1.5_seed8.toml
```

| Configurations | What they run |
| --- | --- |
| `select.toml` | the rule over the runs of step 1 with seeds 0 to 9, and the guard's runs of step 0 |
| `select_seeds10to19.toml` | the rule over the runs of step 1 with the held-out seeds 10 to 19 alone |
| `esn1_lr0.03_sr1.2_is1.5_seed8.toml`, `esn2_lr0.03_sr1.2_is1_seed2.toml` | ESN #1 and ESN #2, as the rule chose them, trained and run on their own from the 49 offset starts |

### Step 3: the size and the ridge

At ESN #1's setting, every scenario of step 1 with seeds 0 to 9: 100, 200, 400, and
800 neurons, at the ridge of 10⁻², and ridges of 10⁻⁴, 10⁻³, 10⁻², 10⁻¹, and 1, with
400 neurons. They are reported with the grid's measures, and do not change the
choice.

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_consistent_search/size_offsets.toml --workers 30
```

| Configurations | What they sweep |
| --- | --- |
| `size_{nominal,block,pushes,offsets}.toml` | 100 to 800 neurons × seeds 0 to 9 at ESN #1's setting, in one scenario |
| `ridge_{nominal,block,pushes,offsets}.toml` | ridges of 10⁻⁴ to 1 × seeds 0 to 9 at ESN #1's setting, in one scenario |
| `select_{size,ridge}.toml` | the rule's measures of each size and each ridge, over their runs |

```bash
uv run python experiments/select_esns.py experiments/manual_demonstration_v2_consistent_search/select_size.toml
```

### Step 4: the chosen ESNs on the robot

ESN #1 and ESN #2, as trained in step 2, against the replay and the progress replay
with the search's tracker, in every scenario of step 1, so that the report can draw
their runs over time:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_consistent_search/pushes_robot_esn1_lr0.03_sr1.2_is1.5_seed8.toml
```

| Configurations | What they run |
| --- | --- |
| `{nominal,block,pushes,offsets}_robot_esn1_lr0.03_sr1.2_is1.5_seed8.toml` | ESN #1, the replay, and the progress replay, in one scenario |
| `{nominal,block,pushes,offsets}_robot_esn2_lr0.03_sr1.2_is1_seed2.toml` | the same with ESN #2 |

## Part 2: the trackers (report 018)

ESN #1 and ESN #2 against the replay and the progress replay, by
[`robot_esn.py`](../robot_esn.py), in every scenario of the search and under the three
pushes given by hand in report 012 (`data/recorded_forces/recorded_forces_00{0,1,2}.sklog.npz`
under the storage root, replayed by time), with nine trackers, all at ω = 20 rad/s,
their torque computed every 10 ms and held:

- joint PD at damping ratios 0.2, 0.4, and 0.8, each given a zero reference velocity
  or the reference's (report 011's six);
- computed torque at damping ratios 0.2, 0.4, and 0.8, given the reference's
  velocity.

The main comparison is the ESNs with joint PD against the replays with joint PD and
the replay with computed torque; the ESNs with computed torque are noted in the
tables. Besides the search's measures, how hard the motors work over the task, and
how much more than the same arm undisturbed with the same tracker.

## Part 3: the ESNs as dynamical systems (report 019)

- ESN #1 and ESN #2 as report 015 read them: run on their own from rest at 1,296
  postures over both joints' whole range (every 10°), for 60 s; with warm-ups of 0.25
  to 4 s; pushed by short pulses all along the reach on the robot with the search's
  tracker, against both replays, in runs of 24 s; and as the map of their reservoir
  state, its fixed points found by Newton's method, the eigenvalues of its Jacobian
  there, and the rates along their run.
- The 5,040 ESNs of the search and the 5,040 held out screened as report 016 screened
  them, runs from rest 60 s long, and set against their outcomes in part 1 by the
  rule's first point, the offset starts counted; then the ESNs that pass, held still
  on the robot at their unstable fixed points near the take's path and released, and
  blocked along the reach.
- A closing section: every main claim of reports 011 to 016, confirmed, changed, or
  not tested here.
