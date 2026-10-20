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
| `screen.toml` | the 1,050 ESNs: the fixed points from R every 10°, the lead |R| near the path, runs from rest every 30° and around the take's end posture, runs held 4 s at each unstable fixed point, and the rates along the run from the take's start over 4 s windows |
| `screen_5deg_seed0.toml` | the first seed of each of the 105 settings, with R every 5°, to check that 10° misses no fixed point |

## Step 2: what the screen predicts

The rules, fixed before the screen's results, each against report 011's robot
outcomes by its rule (undisturbed and under the block: arrives, holds, and ends within
the goal; under the pushes: arrives and ends within it):

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
