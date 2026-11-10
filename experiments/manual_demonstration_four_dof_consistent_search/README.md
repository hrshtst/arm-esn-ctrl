# Four-link consistent search (Stage 2)

Reports 017 to 020 ran the main results of the project again under one protocol, on the
two-link arm of `reach_manual_v2.toml`
([manual_demonstration_v2_consistent_search](../manual_demonstration_v2_consistent_search/README.md)).
This study runs the same procedure on skelarm's four-link arm,
`third_party/skelarm/examples/reach_four_dof_robot.toml`: the search on the robot, the
chosen ESNs across trackers, the chosen ESNs as dynamical systems, and the ESN and the
robot as one system. This directory is the first part, and its README holds the
assumptions of all four. They are v2's, with the changes below; the configurations are
written once the take is imported, from the values this README fixes. Results of this
part: [report 021](../../reports/021-search-on-a-four-link-arm/README.md).

## The protocol

**The robot, the task, and the take.** The arm, start posture, and target of
`reach_four_dof_robot.toml`, copied into
[`reach_manual_four_dof.toml`](../demonstrations/reach_manual_four_dof.toml): four links
(0.1, 0.25, 0.25, and 0.15 m, 3 kg in all) on a 0.05 m base, joint limits of −20 to
30°, 0 to 120°, 0 to 150°, and −20 to 90°, locked hard stops in the simulation
(below); the start posture (−20, 10, 60, 50)°, joint 1 at its lower limit, the hand
0.54 m from the target at (0, 0.4) m, with a goal radius of 20 mm; simulated every
2 ms. One take taught
by hand with skelarm's trajectory recorder, untrimmed, filtered at 2 Hz as v2's
(`demo_00_filtered`); the ESNs are trained on it and the replays replay it.

**The ESNs.** As v2's: 400 neurons, sparsity 0.05, a bias per neuron drawn uniformly
from [−1, 1], ridge 10⁻², a 1 s warm-up, a 10 ms period, no teacher noise; leak rates
0.01 to 0.5, spectral radii 0.7 to 1.75, and input scalings 0.1 to 2, the same 504
settings, with seeds 0 to 9 for the search and 10 to 19 held out. Their input and
output are the four joint angles, each normalized by the take's range of that joint.

**The search's tracker.** Joint PD at ω = 20 rad/s and a damping ratio of 0.4, given the
reference's velocity, its torque computed every 10 ms and held between, as v2's, but
with **the whole mass matrix M at the take's final posture as its gains**: Kp = ω² M
and Kd = 2ζω M (`[tracker] mass_matrix = true`), rather than each joint's own inertia
M_ii. On this arm, v2's gains are unstable with the torque held: joints 1 and 2 are
nearly redundant, so that, with the other joints free, they feel 18 and 25 times less
inertia than their own M_ii, ring far faster than ω, and break report 013's bound on
the held torque; even the replay of the take is thrown 100° off and onto the stops.
With the whole matrix, at the final posture every mode of the error has ω and ζ, the
diagonal of the gains is v2's, and the replay follows the take within a degree.

**The stops.** The joints' hard stops lock a joint pressed into them
(`[simulator] stops = "locked"`, skelarm's locked stops): the stop holds the joint by its
reaction, which does no work, the other joints move as if it were welded, and a joint
that reaches its stop stops as in a plastic impact. v2's stops, skelarm's default, are a
projection: they zero the stopped joint's velocity but let the other joints accelerate
as if it were free. v2's joints stop only at ±180°; on this arm, whose joints 1
and 2 are nearly redundant and whose joint 1 starts at its stop, the projection feeds
the arm energy the motors never gave it whenever a reference presses a joint into its
stop, as an ESN's does when its output dips below joint 1's limit. With joint 1's
reference held 5° below its stop through a reach (`tests/test_four_dof.py`), the
projected stop gives the arm 2.7 J, which the tracker takes out again; the locked one
gives it none. Part 4's map of the ESN and the robot steps the arm with the same stops.

**The scenarios.** Undisturbed from the take's start; the block (a spring of 20,000 N/m
and a damper of 100 N s/m holding the tip for 1 s); three pushes of 0.5 s at the tip
(3, 5, and 4 N, at 10°, 80°, and −80° from the direction toward the target,
counterclockwise); and the offset starts: every combination of −15, 0, and 15° in each
joint, those beyond the joint limits left out, 36 from the take's start (joint 1 can
only go up, joint 2 only 10° down). v2's 49 starts, every 5°, would be 2,401 in four
joints. The times follow the take as v2's followed its own, by this rule, fixed before
the take was recorded:

| What | v2 | The rule | Here |
| --- | --- | --- | --- |
| block | 4.5 to 5.5 s: 32% of the way | from when the take's hand is 32% of the way to the target, for 1 s | 4.0 to 5.0 s (4.02 s) |
| first push | 3 s: 18% of the way | when the take's hand is 18% of the way | 2.6 s (2.64 s) |
| second push | 7 s: 62% of the way | when it is 62% of the way | 6.7 s (6.69 s) |
| third push | 15 s: 5.4 s after the take arrives, after the hold | 5.4 s after the take arrives | 14.9 s (arrives at 9.51 s) |
| runs | 24 s: the third push + 9 s | the third push + 9 s, and no shorter than the take | 23.9 s (the take: 21.43 s) |

each rounded to 0.1 s, measured on the take the ESNs learn (`demo_00_filtered`; on the
take as recorded, only the first push moves, to 2.7 s); the hold window is 2 s, and the
motors' effort under the block is measured from its onset to 0.5 s after its release
(4.0 to 5.5 s), as v2's. *The way to the target* is one less the hand's distance to the
target over its distance at the start, as v2's configurations measured it ("the take's
hand is 18% of the way"); the README first called it the hand's *path*, but the
fractions of the length along v2's hand path would move v2's times to 3.0, 4.5, and
7.1 s (filtered) or 3.1, 4.6, and 7.1 s (recorded), and those of this take to 2.8, 4.2,
and 7.0 s.

**The take** (imported as `results/demonstrations/20261103-134628-reach_manual_four_dof`):
21.43 s at 100 Hz, from the start posture exactly; its hand moves at 0.44 s, arrives
within the goal at 9.50 s (9.51 s filtered), and stays within it to the end, 11.9 s,
ending 3.3 mm from the target; it is wholly still for its last 2.1 s, and within 5.3 mm
of where it ends from 12 s on. No joint reaches a limit but joint 1, which leaves its
lower limit at the start (the joints' ranges over the take: −20 to 11°, 10 to 40°, 60
to 81°, and 50 to 58°).

**The measures, the rule, and the comparison arms** are v2's, unchanged: an ESN passes
if it arrives, holds, and ends within the goal undisturbed, under the block, and from
every offset start, and arrives and ends within it under the pushes; a setting
qualifies if all ten seeds pass; the guard is the replay's ringing with the search's
tracker; the qualifying settings are ranked by their median following; ESN #1 and
ESN #2 are the typical seeds of the top two. The replay's runs that give the guard run
beside an ESN at v2's ESN #1 setting (leak rate 0.03, spectral radius 1.2, input
scaling 1.5, seed 8), trained on the take: the replay does not depend on it.

## The parts

1. **The search** (this directory): the guard's runs, the grid of 5,040 ESNs in the four
   scenarios, the rule, the held-out seeds, and the size and ridge at ESN #1's
   setting, as v2's steps 0 to 4.
2. **The trackers**: ESN #1 and #2 against both replays with joint PD at damping ratios
   0.2, 0.4, and 0.8, given a zero reference velocity or the reference's, and computed
   torque, all with the mass matrix's gains for joint PD, in every scenario and under
   pushes given by hand in the robot app on this arm (`tools/robot_app.py --mass-matrix`).
3. **The ESNs as dynamical systems**: from rest anywhere, over each joint's range between
   its limits; with other warm-ups; pushed by short pulses; their fixed points and
   rates; the screen of the grid, its posture map over the four joints' limits, set
   against the search; the releases and blocks.
4. **The ESN and the robot as one system**: the same, with the arm's joint angles and
   velocities in the state (eight values beside the reservoir's).

The grids of parts 3 and 4 are each joint's range from its lower limit in steps that
divide every range (10° gives 6 × 13 × 16 × 12 = 14,976 postures), and are set with
their configurations, after the search, with their cost.

## Part 1: the search

### Step 0: the guard

The guard's ESN, at v2's ESN #1 setting, is trained on the take and run on its own from
the 36 offset starts, so that its ESN is saved; the replay then runs beside it with the
search's tracker, under the block and under the pushes, and its ringing there is the
rule's guard. Nothing here repeats v2's regression check (its step 0): the code that
generalized the programs to four joints gives v2's outputs bit for bit (6278507).

```bash
uv run python experiments/autonomous_esn.py experiments/manual_demonstration_four_dof_consistent_search/guard_esn_lr0.03_sr1.2_is1.5_seed8.toml
uv run python experiments/robot_esn.py experiments/manual_demonstration_four_dof_consistent_search/guard_block_robot.toml
uv run python experiments/robot_esn.py experiments/manual_demonstration_four_dof_consistent_search/guard_pushes_robot.toml
```

| Configurations | What they run |
| --- | --- |
| `guard_esn_lr0.03_sr1.2_is1.5_seed8.toml` | the guard's ESN, trained on the take and run on its own from the 36 offset starts |
| `guard_{block,pushes}_robot.toml` | the replay with the search's tracker, beside the guard's ESN (run `20261103-135144-guard_esn_lr0.03_sr1.2_is1.5_seed8`), under the block and under the pushes |

### Step 1: the grid

As v2's step 1, with the times and the tracker above: every ESN runs 39 times, once
undisturbed, under the block, and under the pushes, and from the 36 offset starts.

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_four_dof_consistent_search/sweep_lr0.03_pushes.toml --workers 30
```

The 48 with seeds 0 to 9 run as one queue (`run_queue.sh`, to be paused and resumed
while the machine is wanted), the undisturbed, block, and pushes runs of every leak rate
first, about 13 min a leak rate, then the offset starts, about 2.4 h a leak rate:

```bash
D=experiments/manual_demonstration_four_dof_consistent_search
systemd-run --user --unit=arm-grid --same-dir --collect --setenv=PATH="$PATH" \
    experiments/run_queue.sh experiments/sweep_robot_esn.py --workers 30 -- \
    $(for s in nominal block pushes offsets; do for lr in 0.01 0.015 0.02 0.03 0.04 0.05 0.07 0.1 0.15 0.2 0.3 0.5; do echo $D/sweep_lr${lr}_$s.toml; done; done)
```

| Configurations | What they sweep |
| --- | --- |
| `sweep_lr<leak rate>_{nominal,block,pushes,offsets}.toml` (48) | spectral radius 0.7 to 1.75 × input scaling 0.1 to 2 × seeds 0 to 9, at one leak rate, in one scenario |
| `sweep_lr<leak rate>_{nominal,block,pushes,offsets}_seeds10to19.toml` (48) | the same with the held-out seeds 10 to 19 |

`tests/test_four_dof_search_configs.py` holds each one to v2's configuration of the
same name, but for the take, the tracker's mass matrix, and the times and starts above.

### Step 2: the choice

As v2's: [`select_esns.py`](../select_esns.py) applies the rule to the runs of step 1
with seeds 0 to 9, listed in `select.toml`, beside the guard's runs of step 0:

```bash
uv run python experiments/select_esns.py experiments/manual_demonstration_four_dof_consistent_search/select.toml
```

| Configurations | What they run |
| --- | --- |
| `select.toml` | the rule over the runs of step 1 with seeds 0 to 9, and the guard's runs of step 0 |

### Step 2b: the grid extended below its input scalings

Over the grid of step 1, the rule chooses nothing (run `20261107-085854-select`): 83 of
the 5,040 ESNs pass, against v2's 1,049, and no setting has more than 5 passing seeds,
so that none qualifies and none passes the fallback's 9. The offset starts fail the
most ESNs, then the pushes. The passing ESNs gather at the grid's lowest input
scaling: 46 of the 83 at 0.1, 18 at 0.2, and 1 to 8 at each larger one; the four
settings with 3 to 5 passing seeds are all at 0.1. The grid is therefore extended below
it, to input scalings 0.02 and 0.05, with step 1's leak rates, spectral radii, and
seeds 0 to 9, in its four scenarios; the rule is then applied to both grids together.
The 48 run as one queue, about 11 h at 30 workers:

```bash
D=experiments/manual_demonstration_four_dof_consistent_search
systemd-run --user --unit=arm-grid --same-dir --collect --setenv=PATH="$PATH" \
    experiments/run_queue.sh experiments/sweep_robot_esn.py --workers 30 -- \
    $(for s in nominal block pushes offsets; do for lr in 0.01 0.015 0.02 0.03 0.04 0.05 0.07 0.1 0.15 0.2 0.3 0.5; do echo $D/extend_lr${lr}_$s.toml; done; done)
```

| Configurations | What they sweep |
| --- | --- |
| `extend_lr<leak rate>_{nominal,block,pushes,offsets}.toml` (48) | the grid's `sweep_lr<leak rate>_<scenario>.toml` with input scalings 0.02 and 0.05 |
| `select_extended.toml` | the rule over the runs of step 1 and of the extension together, and the guard's runs of step 0 |

```bash
uv run python experiments/select_esns.py experiments/manual_demonstration_four_dof_consistent_search/select_extended.toml
```

Over both grids (run `20261108-051827-select_extended`), 206 of the 6,720 ESNs pass
and one setting qualifies, at the extension's lowest input scaling: ESN #1 is its
typical seed, leak rate 0.01, spectral radius 1.5, input scaling 0.02, and seed 7
(median following 0.23, median ringing 5.12 mm). ESN #2 comes by the fallback, the
first of the settings with 9 of 10 passing seeds: leak rate 0.02, spectral radius 1.5,
input scaling 0.02, and seed 8 (median following 0.46, median ringing 4.87 mm). The
passing ESNs still grow in number toward the extension's edge (76 at input scaling
0.02, 47 at 0.05, 46 at 0.1), and the chosen ESNs follow the arm little: under the
block, ESN #1's reference follows it by 0.06 and runs 8.2% ahead, against the replay's
0 and 9%. The held-out seeds are deferred. The two ESNs are trained by
[`autonomous_esn.py`](../autonomous_esn.py):

```bash
uv run python experiments/autonomous_esn.py experiments/manual_demonstration_four_dof_consistent_search/esn1_lr0.01_sr1.5_is0.02_seed7.toml
```

| Configurations | What they run |
| --- | --- |
| `esn1_lr0.01_sr1.5_is0.02_seed7.toml`, `esn2_lr0.02_sr1.5_is0.02_seed8.toml` | ESN #1 and ESN #2, as the rule chose them, trained and run on their own from the 36 offset starts |

### Step 4: the chosen ESNs on the robot

As v2's: ESN #1 and ESN #2, as trained in step 2b, against the replay and the progress
replay with the search's tracker, in every scenario of step 1, so that their runs can
be drawn over time. From the offset starts, the joint angles and torques are also drawn
from the start whose hand is nearest the target (+15° in every joint, 223 mm away) and
the one whose hand is farthest (0, 0, −15, and −15°). The size and the ridge (v2's
step 3) are deferred with the held-out seeds.

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_four_dof_consistent_search/pushes_robot_esn1_lr0.01_sr1.5_is0.02_seed7.toml
```

| Configurations | What they run |
| --- | --- |
| `{nominal,block,pushes,offsets}_robot_esn1_lr0.01_sr1.5_is0.02_seed7.toml` | ESN #1, the replay, and the progress replay, in one scenario |
| `{nominal,block,pushes,offsets}_robot_esn2_lr0.02_sr1.5_is0.02_seed8.toml` | the same with ESN #2 |

### Step 5: a rule that asks the reference to follow the arm

The review of step 4 asked for something the rule did not: under the block, ESN #1's and
ESN #2's references keep to the taught motion, while the two-link arm's approach the arm's
joint angles, which lowers the torque after the release. Every ESN of the grids ran under
the disturbances, but the rule asked only that the hand reach the goal, and the following
only ranked the settings that qualified, of which there is one. Step 5 changes the rule
over the same runs, in three ways. **These changes were decided after seeing the results,
so they stand outside the protocol above.**

1. From the offset starts, leaving the goal during the hold is excused, as under the
   pushes. The hand must still arrive and end in the goal. Of the runs from offset starts
   where an ESN with a small gap fails, 55% leave the goal after arriving early (median
   5.75 s, against 9.58 s for the runs that pass) and end about 3 mm from the target.
2. An ESN also fails if its reference, at the block's release, is farther from the arm's
   joint angles than half the replay's. This fraction is `end_gap_deg` over the replay's,
   measured in every run of the grids; the two-link arm's ESN #1 and ESN #2 have 0.38 and
   0.34, and this search's ESN #1 and ESN #2 have 0.78 and 0.66.
3. A setting qualifies if at least 8 of its 10 seeds pass, and the settings are ranked by
   their median of that fraction, the smallest first.

| Configurations | What they run |
| --- | --- |
| `select_following.toml` | the rule with these changes, over the runs of steps 1 and 2b |

```bash
uv run python experiments/select_esns.py experiments/manual_demonstration_four_dof_consistent_search/select_following.toml
```

Over the same runs (run `20261109-202923-select_following`), 390 of the 6,720 ESNs pass
and 8 of the 672 settings qualify, all within the guard. The ranking's first two are the
new ESN #1, leak rate 0.015, spectral radius 1.2, input scaling 0.05, and seed 0 (median
gap 0.28, median following 0.69, median ringing 4.43 mm), and the new ESN #2, leak rate
0.02, spectral radius 1.35, input scaling 0.05, and seed 7 (median gap 0.32, median
following 0.71, median ringing 4.02 mm). They are trained as in step 2b. Because the
changes followed the results, the 8 settings are run again with the held-out seeds 10
to 19, by the same programs as the grid's and in every scenario of step 1. The rule
over those runs, alone, shows whether they hold on seeds not used to choose them.

```bash
D=experiments/manual_demonstration_four_dof_consistent_search
uv run python experiments/autonomous_esn.py $D/following_esn1_lr0.015_sr1.2_is0.05_seed0.toml
uv run python experiments/autonomous_esn.py $D/following_esn2_lr0.02_sr1.35_is0.05_seed7.toml
```

| Configurations | What they run |
| --- | --- |
| `following_esn1_lr0.015_sr1.2_is0.05_seed0.toml`, `following_esn2_lr0.02_sr1.35_is0.05_seed7.toml` | the new ESN #1 and ESN #2, trained and run on their own from the 36 offset starts |
| `heldout_lr<leak rate>_sr<spectral radius>_is<input scaling>_{nominal,block,pushes,offsets}.toml` (32) | each of the 8 qualifying settings with seeds 10 to 19, in one scenario |

As in step 4, the new ESN #1 and ESN #2 run against the replay and the progress replay
in every scenario, and then the held-out seeds run. The queue can be paused, as for the
grid:

```bash
D=experiments/manual_demonstration_four_dof_consistent_search
systemd-run --user --unit=arm-grid --same-dir --collect --setenv=PATH="$PATH" bash -c "
  experiments/run_queue.sh experiments/robot_esn.py -- $D/*_robot_following_esn?_*.toml &&
  experiments/run_queue.sh experiments/sweep_robot_esn.py --workers 30 -- $D/heldout_*.toml"
```

| Configurations | What they run |
| --- | --- |
| `{nominal,block,pushes,offsets}_robot_following_esn1_lr0.015_sr1.2_is0.05_seed0.toml` | the new ESN #1, the replay, and the progress replay, in one scenario |
| `{nominal,block,pushes,offsets}_robot_following_esn2_lr0.02_sr1.35_is0.05_seed7.toml` | the same with the new ESN #2 |

The same rule then reads the held-out runs alone:

```bash
uv run python experiments/select_esns.py experiments/manual_demonstration_four_dof_consistent_search/select_following_seeds10to19.toml
```

| Configurations | What they run |
| --- | --- |
| `select_following_seeds10to19.toml` | the rule of `select_following.toml`, over the held-out seeds 10 to 19 of the 8 settings |

**Results.** Over the held-out runs alone (run `20261109-215903-select_following_seeds10to19`),
53 of the 80 ESNs pass and 2 of the 8 settings qualify. The gap holds on the new seeds;
the passing does not, most of all at input scaling 0.05, the scaling of the new ESN #1
and ESN #2.

| Leak rate | Spectral radius | Input scaling | Seeds passing, 0 to 9 | Seeds passing, 10 to 19 | Median gap, 0 to 9 | Median gap, 10 to 19 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.015 | 1.2 | 0.02 | 9 | 8 | 0.41 | 0.41 |
| 0.01 | 0.99 | 0.02 | 8 | 8 | 0.44 | 0.43 |
| 0.02 | 1.2 | 0.02 | 9 | 7 | 0.36 | 0.33 |
| 0.02 | 1.35 | 0.02 | 8 | 7 | 0.45 | 0.47 |
| 0.01 | 1.2 | 0.05 | 8 | 7 | 0.40 | 0.37 |
| 0.015 | 1.2 | 0.05 (new ESN #1) | 8 | 6 | 0.28 | 0.29 |
| 0.015 | 1.35 | 0.05 | 8 | 5 | 0.38 | 0.38 |
| 0.02 | 1.35 | 0.05 (new ESN #2) | 8 | 5 | 0.32 | 0.27 |

Of the 27 ESNs that fail, 13 end away from the goal after the pushes, 12 fail from the
offset starts, and 5 have a gap above the limit (3 fail on both the pushes and the
offset starts). Of the 143 offset runs that never arrive, 96 start with joint 3 at -15
degrees.

The runs of the new ESN #1 and ESN #2 against the replay, beside the earlier ESNs and the
two-link arm's. "Following" is the fit of the reference's departure from the taught motion
to the arm's, in joint space, from the onset to 0.5 s after the disturbance ends.

| | Old #1 | Old #2 | New #1 | New #2 | Two-link #1 | Two-link #2 |
| --- | --- | --- | --- | --- | --- | --- |
| Block: gap at the release, over the replay's | 0.78 | 0.66 | 0.31 | 0.29 | 0.38 | 0.34 |
| Block: following, joints 1 to 4 (two-link: 1, 2) | 0.09, -0.03, -0.05, 0.06 | 0.19, -0.09, -0.11, 0.19 | 0.54, 0.32, -0.03, 0.65 | 0.57, 0.43, 0.03, 0.60 | 1.03, 0.55 | 0.96, 0.60 |
| Block: following, all joints | 0.01 | 0.02 | 0.32 | 0.36 | 0.84 | 0.79 |
| Block: peak torque, % of the replay's | 88 | 81 | 39 | 35 | 17 | 7 |
| Pushes: following, all joints, pushes 1 to 3 | 0.16, 0.15, 0.12 | 0.36, 0.39, 0.34 | 0.43, 0.56, 0.48 | 0.52, 0.61, 0.62 | 0.70, 0.61, 0.76 | 0.67, 0.62, 0.65 |
| Pushes: peak torque, % of the replay's | 105 | 120 | 167 | 169 | 89 | 77 |
| Offsets: median peak torque, % of the replay's | 82 | 70 | 59 | 67 | 49 | 54 |
| Offsets: median of the largest reference-to-arm distance in the first second, % of the replay's | 95 | 103 | 90 | 80 | 54 | 61 |
| Offsets: starts that pass the original rule, of those with an offset | 35 of 35 | 35 of 35 | 32 of 35 | 35 of 35 | 48 of 48 | 48 of 48 |

At the block's release, the new ESN #1's reference is 2.9, 2.8, 4.9, and 0.0 degrees from
the arm in joints 1 to 4, against the replay's 6.6, 6.3, 13.4, and 12.0. Joint 3 keeps to
the taught motion, with a following of about zero in both new ESNs. The new ESNs' 3 runs
that fail the original rule from the offset starts leave the goal and still end in it. All
of the four-link ESNs, old and new, leave the goal during the pushes, and end in it. The
new ESNs' peak torque under the pushes is 67% and 69% above the replay's, where the old
ESNs' is 5% and 20% above it. The reason was not examined.

The new ESN #1 and ESN #2 replace the earlier ones as this study's ESN #1 and ESN #2,
by the author's decision after these results. `presentations/four-dof-arm/make_media.py`
makes the slides' media from their runs.

## Step 6: smaller pushes and offset starts

Step 5's ESN #1 and ESN #2, no training, under smaller disturbances than those of the
search that chose them. The ESNs stay the same, so these runs describe those two ESNs,
not which ESN would be best at the smaller sizes.

- **Pushes:** step 5's three, at the same times, durations, and directions, with every
  force at half (1.5, 2.5, and 2 N) and at a quarter (0.75, 1.25, and 1 N) of its size.
- **Offset starts:** every combination of −s, 0, and s in each joint, within the joint
  limits, for s of 5° and of 2.5° (step 5's: 15°). These are 54 starts rather than 36,
  because joint 2 may now start below the take's 10°. Sizes are compared over the 36
  directions the three sets share, and the 18 with joint 2 lower are reported apart.
- **The block and the undisturbed runs** are step 5's, not run again.

```bash
D=experiments/manual_demonstration_four_dof_consistent_search
systemd-run --user --unit=arm-grid --same-dir --collect --setenv=PATH="$PATH" \
    experiments/run_queue.sh experiments/robot_esn.py -- \
    $D/pushes_{half,quarter}_robot_following_esn?_*.toml $D/offsets_{5,2.5}deg_robot_following_esn?_*.toml
```

| Configurations | What they run |
| --- | --- |
| `pushes_{half,quarter}_robot_following_esn{1,2}_*.toml` (4) | step 5's pushes at half and at a quarter of their force: the ESN, the replay, and the progress replay |
| `offsets_{5,2.5}deg_robot_following_esn{1,2}_*.toml` (4) | the same three, from the 54 starts offset by 5° and by 2.5° |

### What step 6 found

The progress replay is the replay in every run here (they differ only at the full force),
so only the replay is given.

**Pushes.** The largest hand displacement grows in proportion to the force for every arm,
and the ESNs' stays about twice the replay's. Each cell is at the full, half, and quarter
force.

| | ESN #1 | ESN #2 | Replay |
| --- | --- | --- | --- |
| Hand's largest displacement, push 1 (mm) | 66 / 38 / 21 | 77 / 45 / 25 | 37 / 19 / 9 |
| Hand's largest displacement, push 2 (mm) | 200 / 110 / 56 | 198 / 114 / 60 | 77 / 41 / 21 |
| Hand's largest displacement, push 3 (mm) | 126 / 69 / 35 | 158 / 91 / 47 | 67 / 34 / 17 |
| Path RMSE (mm) | 61.6 / 29.5 / 13.2 | 59.7 / 31.2 / 15.4 | 19.6 / 8.6 / 4.0 |
| Hand path length (mm) | 2925 / 1774 / 1110 | 2369 / 1498 / 975 | 1018 / 793 / 678 |
| Peak torque (N m) | 3.51 / 1.88 / 1.02 | 3.55 / 1.83 / 0.96 | 2.10 / 1.03 / 0.53 |
| Accumulated torque (N m s) | 12.3 / 6.6 / 3.5 | 11.0 / 6.0 / 3.2 | 7.3 / 4.7 / 3.4 |
| Leaves the goal after arriving | yes / yes / yes | yes / yes / yes | no / no / no |
| Ends in the goal (mm from the target) | 2.2 / 1.8 / 1.9 | 7.4 / 5.6 / 4.2 | 3.3 / 3.3 / 3.3 |

- The ESNs' peak torque falls less than in proportion to the force, from 167% and 169% of
  the replay's at the full force to 195% and 182% at a quarter. Their accumulated torque
  falls more: at a quarter of the force it is 103% and 95% of the replay's.
- Following does not depend much on the size. The hand's following gain stays between 0.64
  and 0.78 at every size. Joint 1 under push 1, which is 1.85 and 1.97 at the full force,
  falls to 0.98 and 1.03 at a quarter. There, joints 1 to 3 follow with gains from 0.86 to
  1.03, but for joint 2 under push 1 (0.53 and 0.61), and joint 4 with 0.3 to 0.4, as at
  every size. Taking each signal less the same arm's undisturbed run in place of the take
  changes the gains by at most 0.05, and the reading not at all.
- The slow swing after each push is still there in proportion: it depends on the ESN, not
  on the force's size. Both ESNs still leave the goal after the third push, whose
  displacement is 35 and 47 mm at a quarter of the force, above the goal's 20 mm.

**Offset starts.** Over the 35 offset starts shared by the three sizes (the 36 directions
less the take's own start); each cell is at ±15°, ±5°, and ±2.5°.

| | ESN #1 | ESN #2 | Replay |
| --- | --- | --- | --- |
| Arrive and never leave the goal (of 35) | 32 / 35 / 35 | 35 / 35 / 35 | 35 / 35 / 35 |
| Latest arrival (s) | 15.27 / 9.85 / 9.73 | 14.59 / 9.66 / 9.63 | 9.51 / 9.51 / 9.51 |
| Path RMSE, median (mm) | 58.0 / 11.4 / 5.7 | 34.4 / 9.0 / 3.8 | 17.6 / 3.5 / 1.3 |
| Path RMSE, largest (mm) | 108.6 / 25.2 / 11.5 | 138.4 / 19.4 / 8.5 | 65.5 / 10.0 / 3.4 |
| Peak torque, median (N m) | 64.1 / 18.9 / 9.2 | 72.8 / 23.8 / 12.3 | 109.5 / 36.5 / 18.2 |
| Accumulated torque, median (N m s) | 10.2 / 1.8 / 0.9 | 5.3 / 1.6 / 0.9 | 9.5 / 4.4 / 3.2 |
| Accumulated torque, largest (N m s) | 399.8 / 3.7 / 1.9 | 797.0 / 4.2 / 2.2 | 18.2 / 7.4 / 4.7 |
| Starts with joints 2 to 4 at a stop over 0.1 s | 16 / 1 / 0 | 7 / 0 / 0 | 0 / 0 / 0 |
| Starts with the reference over 5° past a limit | 23 / 1 / 0 | 9 / 0 / 0 | 0 / 0 / 0 |
| Offset retained, median | 0.55 / 0.22 / 0.24 | 0.14 / 0.13 / 0.15 | 0.02 / 0.03 / 0.05 |

- The large excursions of ±15°, where the reference runs past the joint limits and the
  arm is held against its stops, are gone at ±2.5° and all but gone at ±5°.
- **One start of the 18 with joint 2 lower** still shows one: ESN #1 from (0, −5, −5, −5)°,
  the farthest from the target, arrives at 14.56 s with a path RMSE of 96.5 mm, its
  reference 32.3° past a limit, joints 1, 2, and 4 at their stops for 5.0, 3.5, and
  2.6 s, and an accumulated torque of 227 N m s. All the other runs from these 18 starts
  arrive by 9.8 s. At ±5°, ESN #1's joint 2 touches its stop for 0.95 s from
  (5, −5, −5, −5)° and ESN #2's for 0.43 s from (0, −5, −5, 5)°, and the reference is over
  5° past a limit from 2 other starts of ESN #1 and from 2 of ESN #2. At ±2.5° none of this
  happens. The ESNs' median peak torque over these 18, from the first instants, is 104% and
  113% of the replay's at ±5° and 75% and 115% at ±2.5°, against 50% to 68% over the 35
  above, since the replay's own peak is lower there.
- At ±5° and ±2.5° the ESNs' accumulated torque is 29% to 40% of the replay's, whose
  own undisturbed run has 2.18 N m s against the ESNs' 0.27 and 0.28.
