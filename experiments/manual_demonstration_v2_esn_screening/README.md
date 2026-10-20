# Manual-demonstration ESN screening, v2

Report 015 read report 011's two ESNs as maps of their reservoir state and found
where they can rest: ESN #2, which passed every scenario of report 011's robot search,
has two unstable fixed points beside the take's path, where an arm held still for a
few seconds stands still long after it is let go. This study asks whether such
measures, computed from an ESN on its own in seconds, tell which ESNs the robot
search passes or fails, and why, and what it misses.

## The ESNs

Report 011's grid of 1,050 ESNs
([`sweep_robot_following_lr*_filtered.toml`](../manual_demonstration_v2_robot_following_search/README.md)),
each trained on the take as its robot sweeps trained it:

| Parameter | Value |
| --- | --- |
| Leak rate | 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1 |
| Spectral radius | 0.99, 1.2, 1.35, 1.5, 1.75 |
| Input scaling | 1, 1.5, 2 |
| Seed (the random reservoir, its input weights, and its biases) | 0 to 9 |
| Neurons | 400 |
| Sparsity (nonzero reservoir connections) | 0.05 |
| Bias | yes |
| Ridge | 10⁻² |
| Warm-up (the start posture held before the ESN moves) | 1 s |
| Sampling period | 10 ms |
| Teacher noise | none |
| Training take | `results/demonstrations/20261008-170034-reach_manual_v2`, `demo_00_filtered` (filtered at 2 Hz) |

Report 011's ESN #1 is leak rate 0.03, spectral radius 1.2, input scaling 1.5, seed 8;
its ESN #2 the same with input scaling 1 and seed 2.

## The method

Fed its own output, an ESN rests at a fixed point x\* of its update. Such an x\* is
also the fixed point of the reservoir held at its own output posture q\*. Where the
held reservoir has one fixed point x_h(q) at each posture, the fixed points are the
zeros of a map of the joint angles alone,

    R(q) = (the posture the ESN outputs from x_h(q)) − q,

how far the ESN's output leads a posture held at q once its reservoir has settled
there: on the robot, how far its reference leads an arm held still at q, as under
report 011's block ([`screening.py`](../../src/arm_esn_ctrl/screening.py)). On a grid over both joints'
range, every cell over whose corners both components of R change sign is searched by
Newton's method on R. Every 10° on report 015's two ESNs, this finds exactly the fixed
points that report 015 found by Newton's method from 2,592 states of their runs.

Near the take's end posture, though, where the ESN learned to hold still, R can be
nearly flat: less than 3° over a 10° cell. There Newton's method on R steps far from
the zero, and the first screen (`20261020-134112-screen`) missed the target's own
fixed point in 128 ESNs. So Newton's method on
the ESN's update also starts from where each run from rest comes to rest (below),
which finds the attracting fixed points the grid misses.

## Step 0: the grid's ESNs are report 011's

Report 011's sweeps at leak rate 0.03, rerun unchanged at commit `a7d159f`, give the
same outcome, arrival, settling, and return times for all 450 runs (150 ESNs,
undisturbed, under the block, and under the pushes); the other measures agree to
1.2 × 10⁻⁷ of their values. The runs are under
`results/manual_demonstration_v2_robot_following_search/20261020-131544-sweep_robot_following_lr0.03_{nominal,block,pushes}_filtered`.

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_following_search/sweep_robot_following_lr0.03_nominal_filtered.toml
```

## Step 1: the screen

```bash
uv run python experiments/screen_esns.py experiments/manual_demonstration_v2_esn_screening/screen.toml
```

| Configuration | What it runs |
| --- | --- |
| `screen.toml` | the 1,050 ESNs: the fixed points from R every 10° and where the runs from rest come to rest, the lead |R| near the path, runs from rest every 30° and around the take's end posture, runs held 4 s at each unstable fixed point, and the rates along the run from the take's start over 4 s windows |
| `screen_5deg_seed0.toml` | the first seed of each of the 105 settings, with R every 5°, to check that 10° misses no fixed point |

The runs, at commit `fe2a921` with the runs from rest 20 s long, are under
`results/manual_demonstration_v2_esn_screening/`: `20261020-151050-screen` (58 min on
30 processes, 97 s per ESN) and `20261020-160829-screen_5deg_seed0` (18 min).

- Of the 3,499 fixed points, 173 are found only where runs come to rest (153
  attracting, 134 of them within the goal radius): the first screen found the
  target's fixed point in 853 ESNs, this one in 981.
- Every 5°, the 105 ESNs have 333 fixed points; every 10°, 324. The 10° grid misses
  10 unstable ones (in 9 ESNs) and finds one that 5° misses, but whether an ESN has
  an unstable fixed point within 30° of the path, or an attracting one off the
  target, is the same for all 105, and its release arrival is later than 9.58 s at
  both or neither for 104. The median lead's rank correlation between the grids is
  0.998.
- The held reservoir has one fixed point at every posture checked, from random
  states, for every ESN with a spectral radius up to 1.35. At 1.75, in 196 of 210
  ESNs it does not settle at some postures (2,051 of the grid's 1,437,450
  postures), and in 89 another fixed point is found at some (134 of the 177,450
  postures checked); at 1.5, it does not settle at some postures in 2 ESNs.
- The ESNs trained in the two screens differ by up to 2 × 10⁻⁹ rad in R, where the
  held reservoir settles, in about half of them, and are identical in the rest;
  whether a cell holds a fixed point differs only in ESNs with a spectral radius of
  1.75, where the held reservoir does not settle.
- 49 ESNs, 36 of them at a spectral radius of 1.75, have no fixed point within the
  goal radius. Their runs from the take's start pass through the goal and leave it at
  20.6 to 29 s (the 10th to 90th percentiles; the median 23.4 s), after the 20 s runs
  end but within report 011's 24 s robot runs, and 36 of them come to rest elsewhere
  within 120 s.

So `screen.toml`'s runs from rest are 60 s long, past report 011's robot runs and
past those departures, and step 2 uses the screen run again with them. The 5° check
stands as it was run, against the screen of 20 s runs: it checks the grid.

## Step 2: what the screen predicts

The rules, fixed before the screen's results, each against report 011's robot
outcomes by its rule (undisturbed and under the block: arrives, holds, and ends within
the goal; under the pushes: arrives and ends within it; report 011's offset starts,
run only for some settings, are not counted):

- (a) an attracting fixed point whose hand is outside the goal radius predicts ending
  away from the target under the pushes;
- (b) a small lead near the path (the median of |R| over the grid's postures within 15°
  of the take's joint path, up to 0.9 of the way along it), a reference that stays
  with a held arm, predicts the same, at fast leak rates. This was first written as a
  near-zero departure speed, the smallest |R| per second; a smoke test on four ESNs
  showed that minimum to measure only how near a grid posture lies to a zero of R, so
  the median replaced it before the screen was run;
- (c) an unstable fixed point within 30° of the take's joint path, or arriving later
  than the take's 9.58 s after a 4 s hold at an unstable fixed point, predicts never
  arriving or arriving late;
- (d) the rate along the path over 4 s windows predicts the return time and the
  following gain (by rank correlation).

For each, how many of the 430 failing ESNs it catches, how many of the 620 passing
ones it rejects, and whether it keeps report 011's 11 qualifying settings and ESN #1
and #2, over the grid and within each leak rate.

How (b), (c), and (d) are scored, fixed after the screen's ESN-alone results (above)
and before they were set against the robot's outcomes:

- (a) flags an ESN with an attracting fixed point (every eigenvalue inside the unit
  circle) whose hand is more than the goal radius (20 mm) from the target, against
  ending away from the target under the pushes;
- (b) by the area under the ROC curve of the lead for ending away from the target
  under the pushes, over the grid and within each leak rate (whether the lead tells
  more than the leak rate does); and, for the counts, a lead under 10°;
- (c) flags an ESN by either of its two conditions, counted against never arriving in
  any scenario; and arriving late, by the rank correlation of the arrival after the
  release with the robot's arrival time undisturbed and under the block;
- (d) the rank correlations of the median and largest rate across the run and the
  median and largest scaling of a posture's deviation with the return time (the mean
  over the three pushes, of the runs that return) and with report 011's following
  (the mean of the following gain over the block and the three pushes).

The counts, for (a) to (c): the failing ESNs caught, in all and of its own outcome;
the passing ones rejected; the qualifying settings' 110 ESNs rejected; ESN #1 and #2;
over the grid and within each leak rate. Beyond the rules, and apart from them: (a)
for attracting fixed points more than 10 cm off the target only, and whether the
unstable fixed points' growth rates or distances from the path tell the outcomes
apart.
