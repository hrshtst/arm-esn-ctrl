# Manual-demonstration robot following search, v2 (Stage 2)

Results: [report 011](../../reports/011-following-the-arm/README.md).

The study after [the noisy search](../manual_demonstration_v2_robot_noisy_search/README.md)
(report 010). There, the search was scored by the ringing and the return onto the
taught path after a disturbance, and the ESNs it found returned about as fast as the
replay of the take: during a push, their reference held to the taught path and the
tracker snapped the arm back. In the robot app, none did better than report 007's
chosen ESN, whose reference goes with the arm. What is wanted is a reference that
adapts to the arm's actual state when it is disturbed: one that waits for the arm
under the block and yields to it under a push, and then still completes the reach.
Here the ESN is searched again, scored by how much its reference follows the arm,
and the ESN chosen is compared with the replay across joint PD trackers.

- **The following gain** (`following_gain` in the metrics; see
  [`disturbance_following`](../../src/arm_esn_ctrl/metrics.py)). For each
  disturbance, the window runs from its onset to 0.5 s past its end
  (`following_window`), the block's effort window.
  - At each time in it, the arm's hand and its reference's hand are taken from where
    the take is at that time, as the replay's reference, less their values at the
    onset.
  - The gain is the least-squares slope of the reference's displacement on the
    arm's: 0 for a reference that keeps to the take whatever the arm does, as the
    replay's does, 1 for one that goes wherever the arm goes.
  - Under the block, it measures how far the reference waits for the held arm; under
    a push, how far it yields to the push.
  - With report 009's tracker, report 007's chosen ESN follows by 0.82 under the
    block and by 0.75, 0.50, and 0.70 under the three pushes; report 009's chosen
    ESN, trained with noise, by 0.63 and by 0.23, 0.00, and 0.19; the replay by 0.
    Both ESNs follow nearly alike with every tracker of report 008.
- **The tracker:** report 009's, joint PD at ω = 20 rad/s and a damping ratio of
  0.4, given the reference's velocity, with its torque computed every 10 ms and held
  between, while the arm is still simulated every 2 ms.
- **The scenarios:** report 007's: undisturbed, the block, and the three pushes, in
  runs of 24 s from the demonstrated start; and, for the settings that pass them,
  undisturbed from eight offset starts.

## Step 1: the grid

[`sweep_robot_esn.py`](../sweep_robot_esn.py) trains every setting with ten seeds
and runs it with the tracker from the demonstrated start, undisturbed, under the
block, and under the pushes:

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_following_search/sweep_robot_following_lr0.02_pushes_filtered.toml
```

| Configurations | What they sweep |
| --- | --- |
| `sweep_robot_following_lr{0.015,0.02,0.03,0.04,0.05,0.07,0.1}_{nominal,block,pushes}_filtered.toml` | spectral radius 0.99, 1.2, 1.35, 1.5, 1.75 × input scaling 1, 1.5, 2 × seeds 0–9, at one leak rate |

That is report 009's grid without noise, whose runs these repeat, now measured also
by the following gain: 105 settings, 400 neurons, sparsity 0.05, ridge 1e-2, and a
1 s warm-up. Report 007's chosen ESN is seed 7 of its setting (leak rate 0.02,
spectral radius 1.35, input scaling 2).

The settings that pass the three scenarios, by the rule's first point, are then run
undisturbed from eight offset starts, by configurations written once those are
known: the demonstrated start offset by −15°, 0, or 15° in each joint, but for
(0, 0).

| Configurations | What they run |
| --- | --- |
| `sweep_robot_following_offsets_lr<leak rate>_sr<spectral radius>_is<input scaling>_filtered.toml` (13) | one setting that passes the three scenarios × seeds 0–9, undisturbed from the eight offset starts |

The 13 settings: leak rate 0.015 with (spectral radius, input scaling) (0.99, 1.5),
(1.2, 1.5), (1.2, 2), and (1.35, 1.5); leak rate 0.02 with (0.99, 1.5), (1.2, 1.5),
(1.2, 2), (1.35, 1.5), and (1.35, 2); and leak rate 0.03 with (1.2, 1), (1.2, 1.5),
(1.35, 1), and (1.35, 1.5).

The rule, fixed before the runs:

1. **A setting qualifies** if every one of its ten seeds:
   - undisturbed and under the block, arrives, holds, and ends within the goal;
   - under the pushes, arrives and ends within the goal. Leaving the goal during the
     hold is excused here, since the push at 7 s can carry the arm into the goal
     early, still swinging;
   - from each of the eight offset starts, arrives, holds, and ends within the goal.
2. **The following:** a seed's is the mean of its following gain over the four
   disturbances, the block's and the three pushes'; a setting's is the median over
   its seeds. The qualifying settings are ranked by it, the largest first.
3. **The guard:** a setting whose median ringing, each seed's mean over the four
   disturbances as in report 009, exceeds the replay's with the same tracker
   (10.70 mm) is not shortlisted.
4. **Alongside,** the medians over the seeds: the following under the block and
   under each push; after each disturbance, the ringing, the farthest from the taught
   path, and the return onto it; the undisturbed path RMSE; under the block, the
   peak torque and the reference lead; and from the offset starts, the first step.
5. **The shortlist** is the top three settings, each with its typical seed (the one
   nearest its medians of the undisturbed path RMSE, the following, and the ringing,
   each scaled by its interquartile range over the seeds), and report 007's chosen
   ESN.

By the rule, 11 of the 13 settings that pass the three scenarios also pass from the
offset starts; the ringing guard excludes none (5.6 to 6.4 mm). Every setting at a
leak rate of 0.04 or more fails, mostly by ending away from the target after the
pushes, though those settings follow the most (up to 0.95).

## Step 2: the shortlist, with six trackers

The shortlisted ESNs, each trained by [`autonomous_esn.py`](../autonomous_esn.py)
with its setting's typical seed and run on its own from the 49 offset starts:

```bash
uv run python experiments/autonomous_esn.py experiments/manual_demonstration_v2_robot_following_search/grid_lr0.03_sr1.2_is1.5_seed8_filtered.toml
```

| Configuration | Following | Block | Pushes at 3 / 7 / 15 s | Ringing | Return | Undisturbed path RMSE |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `grid_lr0.03_sr1.2_is1.5_seed8_filtered.toml` | 0.76 | 0.88 | 0.79 / 0.56 / 0.77 | 5.68 mm | 2.92 s | 0.69 mm |
| `grid_lr0.03_sr1.2_is1_seed2_filtered.toml` | 0.74 | 0.84 | 0.81 / 0.62 / 0.75 | 5.72 mm | 3.27 s | 0.88 mm |
| `grid_lr0.03_sr1.35_is1.5_seed6_filtered.toml` | 0.73 | 0.85 | 0.72 / 0.55 / 0.77 | 6.14 mm | 2.84 s | 0.58 mm |

The measures are each setting's medians over its seeds. With them, report 007's
chosen ESN
([`manual_demonstration_v2_robot_ringing/grid_lr0.02_sr1.35_is2_seed7_filtered.toml`](../manual_demonstration_v2_robot_ringing/grid_lr0.02_sr1.35_is2_seed7_filtered.toml),
already trained), whose setting follows by 0.69 (block 0.84, pushes 0.78 / 0.41 /
0.70), rings 5.86 mm, and returns in 2.66 s, with a path RMSE of 0.58 mm.

Each of the four is then run against the replay with six trackers, joint PD at
damping ratios 0.2, 0.4, and 0.8, given a zero reference velocity or the
reference's, all with the torque computed every 10 ms, undisturbed, under the block
and the pushes, and from the 49 offset starts, by configurations written once the
three are trained, and tried in [the robot app](../../tools/README.md), where one
is chosen:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_following_search/pushes_robot_lr0.03_sr1.2_is1.5_seed8_reference_velocity_filtered.toml
uv run python tools/robot_app.py experiments/demonstrations/reach_manual_v2.toml \
    --law pd --omega 20 --damping 0.4 --control-period 0.01 --model <run directory>/esn.toml
```

| Configurations | What they run |
| --- | --- |
| `{nominal,block,pushes,offsets}_robot_<ESN>_{zero,reference}_velocity_filtered.toml` (32) | one ESN and the replay with joint PD at damping ratios 0.2, 0.4, and 0.8, given a zero reference velocity or the reference's; `<ESN>` is `lr0.03_sr1.2_is1.5_seed8`, `lr0.03_sr1.2_is1_seed2`, `lr0.03_sr1.35_is1.5_seed6`, or report 007's `lr0.02_sr1.35_is2_seed7` |

They are report 009's validation configurations with each ESN, and the damping
ratio 0.2 in place of 0.1.

With the torque computed every 10 ms, a damping ratio of 0.1 acts like about 0.05,
so 0.2 replaces report 008's 0.1.

**Two ESNs were chosen in the app:** the first and the second,
`grid_lr0.03_sr1.2_is1.5_seed8_filtered.toml` and
`grid_lr0.03_sr1.2_is1_seed2_filtered.toml`. Their runs above with the six trackers
are the validation; both are compared with the replay.

## Step 3: the reservoir states

[`robot_states.py`](../robot_states.py) recomputes each chosen ESN's reservoir
states from its runs with the search's tracker (damping ratio 0.4, given the
reference's velocity): undisturbed, under the block and the pushes, and from two
offset starts.

```bash
uv run python experiments/robot_states.py experiments/manual_demonstration_v2_robot_following_search/states_robot_lr0.03_sr1.2_is1.5_seed8_filtered.toml
```

| Configurations | What they draw |
| --- | --- |
| `states_robot_{lr0.03_sr1.2_is1.5_seed8,lr0.03_sr1.2_is1_seed2}_filtered.toml` | one chosen ESN's states: eight neurons over every run and through the warm-up, and the principal components of every run |
