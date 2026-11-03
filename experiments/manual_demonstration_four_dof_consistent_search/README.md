# Four-link consistent search (Stage 2)

Reports 017 to 020 ran the main results of the project again under one protocol, on the
two-link arm of `reach_manual_v2.toml`
([manual_demonstration_v2_consistent_search](../manual_demonstration_v2_consistent_search/README.md)).
This study runs the same procedure on skelarm's four-link arm,
`third_party/skelarm/examples/reach_four_dof_robot.toml`: the search on the robot, the
chosen ESNs across trackers, the chosen ESNs as dynamical systems, and the ESN and the
robot as one system. This directory is the first part, and its README holds the
assumptions of all four. They are v2's, with the changes below; the configurations are
written once the take is imported, from the values this README fixes.

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
rule's guard; `guard_{block,pushes}_robot.toml` are written once the ESN is saved, to
name its run. Nothing here repeats v2's regression check (its step 0): the code that
generalized the programs to four joints gives v2's outputs bit for bit (6278507).

```bash
uv run python experiments/autonomous_esn.py experiments/manual_demonstration_four_dof_consistent_search/guard_esn_lr0.03_sr1.2_is1.5_seed8.toml
```

| Configurations | What they run |
| --- | --- |
| `guard_esn_lr0.03_sr1.2_is1.5_seed8.toml` | the guard's ESN, trained on the take and run on its own from the 36 offset starts |

### Step 1: the grid

As v2's step 1, with the times and the tracker above: every ESN runs 39 times, once
undisturbed, under the block, and under the pushes, and from the 36 offset starts.

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_four_dof_consistent_search/sweep_lr0.03_pushes.toml --workers 30
```

| Configurations | What they sweep |
| --- | --- |
| `sweep_lr<leak rate>_{nominal,block,pushes,offsets}.toml` (48) | spectral radius 0.7 to 1.75 × input scaling 0.1 to 2 × seeds 0 to 9, at one leak rate, in one scenario |
| `sweep_lr<leak rate>_{nominal,block,pushes,offsets}_seeds10to19.toml` (48) | the same with the held-out seeds 10 to 19 |

`tests/test_four_dof_search_configs.py` holds each one to v2's configuration of the
same name, but for the take, the tracker's mass matrix, and the times and starts above.
The choice (v2's step 2), the size and the ridge (step 3), and the chosen ESNs on the
robot (step 4) are configured once the grid has run, as v2's were.
