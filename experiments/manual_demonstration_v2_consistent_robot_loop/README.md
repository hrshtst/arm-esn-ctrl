# Manual-demonstration consistent robot loop, v2 (Stages 1 and 2)

The ESN and the robot read as one dynamical system: report 019's analyses of the ESNs
on their own, run again with the arm in the loop. Results:
[report 020](../../reports/020-the-esn-and-the-robot-as-one-system/README.md).

On the robot, the ESN is fed the arm's measured joint angles, not its own output, and
the search's tracker, joint PD with its torque computed every 10 ms and held, makes the
arm follow the ESN's output. Sampled every 10 ms, the ESN and the arm are one system:
its state z = (x, q, q̇) holds the reservoir's 400 values and the arm's two joint angles
and two velocities, and nothing else carries over from one period to the next
([`RobotLoop`](../../src/arm_esn_ctrl/dynamics.py)). The system's fixed points are the
ESN's own, the arm at rest at their postures: at rest, the arm, in a plane without
gravity, needs no torque, so it rests where the ESN points, and the ESN is fed its own
output there. What can change is how they attract.

- **The ESNs:** report 017's ESN #1 (leak rate 0.03, spectral radius 1.2, input scaling
  1.5, seed 8) and ESN #2 (input scaling 1, seed 2), as the search trained them
  (`results/manual_demonstration_v2_consistent_search/20261024-205825-esn1_lr0.03_sr1.2_is1.5_seed8`
  and `20261024-205835-esn2_lr0.03_sr1.2_is1_seed2`); and the search's 5,040 ESNs
  (seeds 0 to 9) and 5,040 held out (seeds 10 to 19), as report 019's screens trained
  them.
- **The tracker,** wherever the robot runs: the search's, joint PD at ω = 20 rad/s and
  ζ = 0.4, given the reference's velocity, its torque computed every 10 ms and held.
- **Report 019's runs read here:** step 3's pulses (`20261027-180441-push_pulses`),
  step 4's fixed points (`20261027-183137-fixed_points`), step 5's screens
  (`20261027-183607-screen`, `20261028-064044-screen_seeds10to19`), step 6's outcomes
  (`20261028-044632-against_robot`, `20261028-115552-against_robot_seeds10to19`), and
  step 7's releases (`20261028-044731-release_robot`). Step 1 compares with report 019's
  step 1 (`20261027-180441-global_starts`).

Every step and claim below is fixed before its runs.

## Step 1: from rest anywhere, on the robot

[`robot_starts.py`](../robot_starts.py), report 019's step 1 on the robot: each ESN
driving the arm, from rest at 1,296 start postures every 10° over both joints' range, the
arm held there through the ESN's 1 s warm-up, for 60 s. Through the warm-up the ESN is
fed the held posture, as on its own, so its first output, and the reference's first
step, are its own.

```bash
uv run python experiments/robot_starts.py experiments/manual_demonstration_v2_consistent_robot_loop/robot_starts.toml --workers 30
```

The claims, for each ESN:

1. from each of the 1,296 starts, the arm ends at rest at the target;
2. every run joins the take's joint path for good (within 5°), as the ESN's own runs do;
3. where both the robot and the ESN on its own (report 019's step 1) arrive, their
   arrivals have a rank correlation of at least 0.9 over the starts.

## Step 2: ESN #1 and ESN #2 with the arm

[`robot_fixed_points.py`](../robot_fixed_points.py), report 019's step 4 with the arm:

- **the fixed points:** each of the ESN's fixed points that report 019's step 4 found,
  the arm at rest at its posture: the system's residual there, and its eigenvalues
  against the ESN's own;
- **the rates along the robot's run** from the take's start, for the take's duration,
  over windows of 0.5, 1, 2, and 4 s, as report 019's step 4 measured them along the
  ESN's own run, the arm's joint angles weighed as the ESN's input and their velocities
  by how far they move them in a period;
- **the pulses, linearized:** report 019's step 3 pulses on each ESN's arm (3 N and 6 N
  for 0.2 s, at 1 to 9 s and 12 s, in 8 directions), predicted for 2 s from the onset by
  the system's Jacobians along the undisturbed run, against the simulated runs.

```bash
uv run python experiments/robot_fixed_points.py experiments/manual_demonstration_v2_consistent_robot_loop/robot_fixed_points.toml
```

The claims, for each ESN:

1. each of its fixed points, the arm at rest at its posture, is a fixed point of the
   system (a residual of 10⁻¹² or less), with as many unstable modes as on its own;
2. over the 2 s and 4 s windows, the median of the largest rate across the robot's run
   is within 0.1 /s of that along the ESN's own run.

The pulses' prediction is described, not claimed (see below).

## Step 3: every fixed point of the screens with the arm

[`screen_robot_loop.py`](../screen_robot_loop.py): every fixed point of report 019's two
screens, its state found again, the arm at rest at it: the system's eigenvalues against
the ESN's own. The screens keep each fixed point's posture, not its state: the state is
the held reservoir's fixed point at the posture from the reset, else as the screen found
it, by Newton's method on R from the grid's cells around the posture, else from random
states, refined by Newton's method on the ESN's own map, and kept only if it gives the
screen's posture and growth rate. A fixed point beyond the arm's joint stops at ±180°,
where the ESN on its own can rest, is none of the system's: the arm cannot rest there.
Then:

- **the releases** of report 019's step 7 (994, at the unstable fixed points within 30°
  of the take's path of the 1,049 passing ESNs): how long the arm stays within 1° of the
  fixed point, against the slowest mode's growth rate on its own and with the arm;
- **rule (a) with the arm:** an ESN flagged if it has a fixed point that attracts with
  the arm and whose hand is outside the goal radius, against report 019's rule (a), on
  the ESN's own, for failing on the robot as report 019's step 6 counted it;
- **the changes:** within each leak rate, the fixed points whose stability the arm
  changes, and the growth rate with the arm against on its own.

```bash
uv run python experiments/screen_robot_loop.py experiments/manual_demonstration_v2_consistent_robot_loop/screen_robot_loop.toml --workers 30
```

The claims:

1. at the releases at leak rates 0.15 to 0.5, where report 019 saw the robot part from
   the ESN on its own, how long the arm stays has a more negative rank correlation with
   the growth rate with the arm than with the growth rate on its own;
2. over all 994 releases, the growth rate with the arm tells the releases where the arm
   stays the whole 60 s from the others with a larger area under the ROC curve than the
   growth rate on its own (a smaller growth, a longer stay);
3. rule (a) with the arm has an area under the ROC curve for failing at least that of
   rule (a) on its own, in both seed sets.

## Added after the claims' runs

Steps 4 and 5 were planned after steps 1 to 3 had run and printed their claims. They
test nothing of the plan above; they check two of its measures.

**Step 4: the releases, nudged.** Run again with a fixed point's posture, report 019's
releases showed the arm staying put at some, and leaving at others, as the run went: the
reservoir, held 4 s at a fixed point, can come to rest on it to the last bit, and a
simulation then adds nothing to grow. [`nudge_releases.py`](../nudge_releases.py) runs
every release of report 019's step 7 again from 10⁻⁹° off the fixed point in each joint,
on the robot and on its own.

```bash
uv run python experiments/nudge_releases.py experiments/manual_demonstration_v2_consistent_robot_loop/nudge_releases.toml --workers 30
```

**Step 5: the rates, weighed otherwise.** Step 2's rates along the robot's run weigh the
arm's joint angles and velocities as above. [`robot_rates_weighed.py`](../robot_rates_weighed.py)
measures them over the 2 s and 4 s windows with the joint angles and the joint
velocities weighed by 0.1 to 100 times more.

```bash
uv run python experiments/robot_rates_weighed.py experiments/manual_demonstration_v2_consistent_robot_loop/robot_rates_weighed.toml --workers 12
```

**The first runs of steps 1 and 3,** at `1b6d2f2`, did not stand. Step 1 ran every run
and wrote ESN #1's files (`20261102-112847-robot_starts`), then stopped on printing a
measure of report 019's step 1 it does not take. Step 3 (`20261102-121258-screen_robot_loop`)
took each state from the reset only, which gave another fixed point for 329 and 317 of
the screens' fixed points, nearly all at a spectral radius of 1.75, where the held
reservoir has several fixed points at a posture; it printed its scores with those among
them. Both ran again with the fix that finds the states as above and prints step 1's
summary without that measure. Step 3's second run (`20261102-122257-screen_robot_loop`)
took the 817 and 668 fixed points beyond the arm's stops as the system's, though the arm
cannot rest there, and printed its scores with them; it ran a third time with them left
out.

## What was seen before this plan

To check that the system can be read this way, and to write and time the scripts, these
were computed before the plan, and are not tests of it:

- **ESN #2's three fixed points** with the arm: the same stability, growth rates of
  +0.227, +0.911, and −0.678 /s on its own and +0.230, +0.902, and −0.675 /s with it;
  and the map's one-step error along 12 s of its robot run, 2 × 10⁻¹⁴;
- **30 fixed points of six ESNs** with ESN #2's spectral radius, input scaling, and seed,
  at leak rates 0.03, 0.1, 0.15, 0.2, 0.3, and 0.5, and **11 of two others**, (0.5,
  1.75, 2, seed 9) and (0.2, 0.9, 0.5, seed 4): none changes stability with the arm,
  and from a leak rate of 0.1 the arm slows growth and decay, at 0.5 by up to an order
  of magnitude;
- **ESN #2's 16 pulses at 4 s**, linearized: relative errors of 0.07 to 0.19 at 3 N and
  0.13 to 0.43 at 6 N, and 0.009 for a pulse of 0.3 N through the system's own map. The
  error grows with the force: the system is that nonlinear at these pulses. So step 2's
  pulses are described, not claimed;
- **two runs of step 1**, ESN #1 from (−180°, −180°) and (−160°, 0°): both arrive, with
  peak torques of 1,011 and 658 N m.

No release, no outcome on the robot, and no area under the ROC curve was computed with
the arm before the plan.
