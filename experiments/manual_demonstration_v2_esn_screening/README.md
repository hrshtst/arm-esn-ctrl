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

### Step 2's results

```bash
uv run python experiments/screen_against_robot.py experiments/manual_demonstration_v2_esn_screening/against_robot.toml
```

The screen, its runs from rest 60 s long, at commit `8711521`:
`20261020-165217-screen` (64 min on 30 processes); set against the robot at commit
`df5ec60`: `20261020-181402-against_robot`. On the robot, 620 ESNs pass and 430 fail,
and 13 settings pass with all ten seeds, as report 011 counted. The share failing
rises with the leak rate, from 0.17 to 0.22 at 0.015 to 0.03 to 0.76 at 0.1: the
leak rate alone has an area under the ROC curve of 0.749 for failing, and a rank
correlation of 0.75 with the following.

| Rule | Flagged | Its outcome caught | Failing caught (of 430) | Passing rejected (of 620) | Qualifying rejected (of 110) | Of ESN #1, #2, #3, and report 007's |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| (a) | 458 | ending away, 288 of 410 | 300 | 158 | 3 | none |
| (b) | 415 | ending away, 272 of 410 | 280 | 135 | 0 | none |
| (c) | 827 | never arriving, 49 of 49 | 389 | 438 | 72 | #2, #3, report 007's |

The share failing among the flagged ESNs and among the unflagged, within each leak
rate:

| Rule | 0.015 | 0.02 | 0.03 | 0.04 | 0.05 | 0.07 | 0.1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (a) | 0.68 / 0.11 | 0.89 / 0.07 | 0.45 / 0.18 | 0.26 / 0.33 | 0.53 / 0.42 | 0.75 / 0.51 | 0.84 / 0.47 |
| (b) | none flagged / 0.22 | 0.00 / 0.18 | 0.29 / 0.20 | 0.47 / 0.25 | 0.65 / 0.38 | 0.72 / 0.43 | 0.76 / none unflagged |
| (c) | 0.20 / 0.24 | 0.21 / 0.09 | 0.27 / 0.09 | 0.33 / 0.20 | 0.52 / 0.33 | 0.74 / 0.29 | 0.77 / 0.40 |

- (a) tells the slow leak rates' failures apart, and not those at 0.04 or 0.05. The
  three qualifying ESNs it rejects have their attracting fixed point more than 1 m
  from the target and pass; 122 ESNs end away without one. Only those more than
  10 cm off: 423 flagged, 262 of the 410 ending away caught, 150 passing rejected.
- (b)'s area under the ROC curve is 0.748 over the grid, the leak rate's; within each
  leak rate, 0.24, 0.32, 0.43, 0.65, 0.68, 0.75, and 0.74 from 0.015 to 0.1: a
  small lead tells failing apart at the fast leak rates, and at the slow ones those
  ending away lead more.
- (c)'s release arrival has a rank correlation with the robot's arrival under the
  block of +0.44 over the grid, and +0.23, +0.31, +0.26, +0.28, +0.16, +0.16, and
  −0.01 within each leak rate; with the undisturbed arrival, +0.03 (−0.11 to +0.18).
- (d): the largest scaling of a posture's deviation has a rank correlation with the
  following of +0.72 over the grid, and +0.67, +0.57, +0.66, +0.74, +0.74, +0.53,
  and +0.30 within each leak rate (pooled, +0.60); its median's, +0.61 to +0.74 up
  to 0.05. The median rate across the run's is −0.67 over the grid, −0.10 to −0.42
  within. The return time, of the 324 ESNs whose arm returns after each push (276 of
  the 620 passing), correlates at most +0.24 over the grid (the largest rate across
  the run; −0.11 to +0.44 within).
- Apart from the rules, an unstable fixed point nearer the path tells failing apart
  with an area under the ROC curve of 0.76 over the grid and 0.58 to 0.78 within
  each leak rate; its growth rate does not, with no sign kept across the leak rates.
  Of the 49 ESNs that never arrive, 43 are at 0.07 and 0.1, where the nearness tells
  them apart little (0.59 and 0.52).

## Step 3: the passing ESNs held still on the robot

Fixed before the runs. All 620 passing ESNs, trained as the screen trains them, on
the robot with report 011's tracker (joint PD at 20 rad/s and a damping ratio of 0.4,
given the reference's velocity, its torque computed every 10 ms):

- **Released at an unstable fixed point:** for each of the 810 unstable fixed points
  within 30° of the take's joint path (of 420 ESNs), the arm at rest at its posture,
  the ESN held there 4 s, then let go for 60 s: when the hand arrives, how long the
  arm stays within 1° of the posture, and whether it ends within the goal; against
  the ESN's own release arrival from the same hold. Whether the robot's arrival is
  the ESN's own, and how many stall.
- **Blocks along the reach:** from the take's start, the tip held for 1 s or for 3 s
  from each of 2, 3, 4, 5, 6, 7, and 8 s, one block a run, for 30 s: whether the
  arm arrives, holds, and ends within the goal (report 011's rule under the block);
  how much later it arrives than the take would from where it was held (the arrival
  after the release less the take's time from the progress at the middle of the
  block to its arrival at 9.58 s); and, at the held posture, the lead R and the
  distance to the nearest unstable fixed point. Whether the ESNs with an unstable
  fixed point near the path (420) fail or arrive late more often than those without
  (200), and whether the delay grows as the held posture nears an unstable fixed
  point or its lead shrinks, by rank correlation, over the ESNs and within each leak
  rate.

### Step 3's results

```bash
uv run python experiments/release_robot.py experiments/manual_demonstration_v2_esn_screening/release_robot.toml
```

At commit `2deb92c`: `20261020-182057-release_robot` (77 min on 30 processes).

- **Released, the robot does what the ESN does on its own.** At 710 of the 810
  unstable fixed points (420 ESNs) both arrive, and the robot's arrival is the ESN's
  own to a median of +0.05 s (the 10th to 90th percentiles −0.05 to +0.25 s; rank
  correlation +0.99); at 98 neither arrives within 60 s, at one only the robot, and
  at one only the ESN on its own. Arriving later than the take's 9.58 s: the robot
  498, the ESN on its own 490.
- **The 99 where the robot never arrives (in 86 ESNs) do not stall:** the arm
  leaves the posture within 5 s at 80 of them, and stays within 1° for the whole
  60 s at 3; 76 of the 86 ESNs also have an attracting fixed point off the target
  (rule (a)). Their unstable fixed points lie at the start of the path (the median progress 0) and 7.5° from it (the median), their
  modes growing at 1.2 per second (the median; 0.75 at the others). The 86 are 75
  ESNs at leak rates 0.04 to 0.1 and 11 at 0.015 to 0.03, 3 of them in qualifying
  settings; none of ESN #1, #2, #3, and report 007's. ESN #2 released at
  (39.8°, 97.1°) arrives 20.2 s later (on its own, 20.15 s), and at (73.7°, 79.6°)
  11.95 s later; ESN #3 at its two, 6.55 and 6.7 s later.
- **Under the blocks, an unstable fixed point near the path does not make an arm
  fail:** of the runs of the 420 ESNs with one, 8.7% fail with a hold of 1 s and
  16.9% with 3 s, and of the 200 without, 11.4% and 21.5%. The failures are late
  blocks: from 8 s, 42% and 57% of runs fail, from 2 to 4 s 2% to 7%; most arrive
  and then leave the goal or do not hold (1,086 of 1,213), 30 never arrive. 458 of
  the 620 ESNs fail at least one of the 14 blocks, 82 of the qualifying settings'
  110, and ESN #1, #3, and report 007's one each, ESN #2 three.
- **The delay after a block grows a little near an unstable fixed point and where
  the lead is small:** the median delay is −0.54 s with an unstable fixed point
  near the path and −0.65 s without (a hold of 1 s; −0.53 and −0.72 s with 3 s, the
  90th percentiles +0.78 and +0.59 s). Its rank correlation with the held posture's
  distance to the nearest unstable fixed point is −0.19 (−0.07 to −0.27 within each
  leak rate) with a hold of 1 s and −0.18 with 3 s, and with the lead there −0.40
  (−0.21 to −0.39) and −0.16 (+0.10 to −0.34).
