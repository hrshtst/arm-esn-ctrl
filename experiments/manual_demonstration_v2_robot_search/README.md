# Manual-demonstration robot search, v2 (Stage 2)

A wider search for the ESN of
[manual-demonstration robot tracking, v2](../manual_demonstration_v2_robot_tracking/README.md)
(report 005), by the same measure: how closely the simulated robot arm, driven by
the ESN through the tracker, follows the taught path. Report 005 kept the spectral
radius below 1, chose input scaling 2 with one random reservoir, and found the
spread over reservoirs as large as the differences between settings. Here:

1. Does a spectral radius over 1 do better?
2. Was input scaling 2 needed, or does 1.5 or 1 do as well?
3. How do leak rates above 0.05 do?

Every setting is tried with ten random reservoirs (seeds 0 to 9). A seed draws the
reservoir, its input weights, and its biases, which the spectral radius and the
input scaling only rescale, so the settings compare seed by seed
(`arm_esn_ctrl.sweeps`).

## Stage 1: the wide grid

[`sweep_robot_esn.py`](../sweep_robot_esn.py) trains every combination on the take
filtered at 2 Hz and runs it on the robot from the demonstrated start, undisturbed,
with both trackers, as in report 005's sweeps.

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_search/sweep_robot_wide_lr0.05_filtered.toml
```

| Configurations | What they sweep |
| --- | --- |
| `sweep_robot_wide_lr{0.03,0.05,0.07,0.1,0.15,0.2,0.3}_filtered.toml` | input scaling 1–2.5 × spectral radius 0.9–1.5 × seeds 0–9, one leak rate each (280 combinations) |

Besides `sweep.csv`, `runs.csv`, and `sweep.png` (one column of heatmaps per seed),
each run writes `seeds.csv` and `seeds.png`: each setting over its ten seeds, with
the seeds that fail, the median, mean, and spread of the worst path RMSE, the
median latest arrival, and the largest distance from the target at the end of a
run, 22 s, about 12 s after an arrival like the take's.

All train with ridge 1e-2, a 1 s warm-up, 400 neurons, and sparsity 0.05, and run
22 s with report 005's tracker: computed torque and joint PD at ω = 20 rad/s,
ζ = 0.1, given a zero reference velocity.

## Judging the results

- A setting must arrive and hold with both trackers for all ten seeds, and its
  hand must still be at the target at the end of the run.
- **A spectral radius over 1 helps** only if, at its best leak rate and input
  scaling, its median worst path RMSE over the seeds beats that of spectral radius
  0.99, and it beats it seed by seed for at least 8 of the 10 seeds.
- **Input scaling 2 was needed** only if, where the best settings lie, it beats
  input scaling 1.5 for at least 8 of the 10 seeds, with a median gain over 0.2 mm.

Leak rates 0.03 and 0.05 at spectral radius 0.99 and input scaling 2 repeat report
005's seed check (stage 4) and must reproduce it.

## Candidates to try

Some settings of stage 1, each trained by
[`autonomous_esn.py`](../autonomous_esn.py) with its typical seed (the one nearest
the median over its ten seeds), so that its ESN is saved for the interactive apps
and for `robot_esn.py`:

```bash
uv run python experiments/autonomous_esn.py experiments/manual_demonstration_v2_robot_search/grid_lr0.03_sr1.5_is1.5_seed2_filtered.toml
```

| Configuration | Why |
| --- | --- |
| `grid_lr0.03_sr1.5_is1.5_seed2_filtered.toml` | stage 1's best setting (median 0.81 mm over its seeds) |
| `grid_lr0.03_sr1.35_is2_seed5_filtered.toml` | the setting steadiest over its seeds (SD 0.12 mm) |
| `grid_lr0.03_sr1.5_is1_seed7_filtered.toml` | the weakest input among the best settings |
| `grid_lr0.05_sr1.5_is1.5_seed3_filtered.toml` | report 005's leak rate with a spectral radius over 1 |
| `grid_lr0.07_sr1.5_is1.5_seed3_filtered.toml` | a faster leak rate |
| `grid_lr0.1_sr1.2_is1.5_seed4_filtered.toml` | a faster leak rate still, arriving later |
| `grid_lr0.03_sr0.99_is1.5_seed2_filtered.toml` | the first candidate's reservoir with spectral radius 0.99, below 1 |

## Step A: the kept candidates in report 005's scenarios

Candidates #1, #2 (the most balanced against disturbances in the robot app), #6
(a faster leak rate), and #7 (spectral radius below 1) run report 005's validation
with [`robot_esn.py`](../robot_esn.py), each against the replay of the take:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_search/pushes_robot_lr0.03_sr1.35_is2_seed5_filtered.toml
```

| Configurations | Scenario |
| --- | --- |
| `nominal_robot_<candidate>_filtered.toml` | no disturbance, from the demonstrated start |
| `block_robot_<candidate>_filtered.toml` | the tip held from 4.5 s to 5.5 s, in mid-reach, then let go |
| `pushes_robot_<candidate>_filtered.toml` | three pushes at the tip, at 4 s, 8 s, and 13 s |
| `offsets_robot_<candidate>_filtered.toml` | the 49 start postures, −15° to 15° from the demonstrated start in each joint |

The `<candidate>` is `lr0.03_sr1.5_is1.5_seed2` (#1), `lr0.03_sr1.35_is2_seed5`
(#2), `lr0.1_sr1.2_is1.5_seed4` (#6), or `lr0.03_sr0.99_is1.5_seed2` (#7). Each run
draws the joint angles and torques over time (`joints.png`, `torques.png`); the
offsets runs also from the offset starts that their `[plots]` table lists, (+15°,
+15°) and (0°, −10°) (`joints_offset_+15_+15.png`, `torques_offset_+0_-10.png`, ...).

## Step B: a star around candidate #2

Candidate #2 (leak rate 0.03, spectral radius 1.35, input scaling 2) felt the most
balanced against disturbances in the robot app. In Step A, it was the only candidate
whose hand came back onto the taught path after all three pushes with both trackers.
Step B varies one of its settings at a time, keeping the others, with ten random
reservoirs each. Every setting runs undisturbed, under the pushes, and under the
block of Step A, with [`sweep_robot_esn.py`](../sweep_robot_esn.py):

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_search/sweep_robot_star_sr_pushes_filtered.toml
```

| Configurations | What they sweep |
| --- | --- |
| `sweep_robot_star_sr_{nominal,pushes,block}_filtered.toml` | spectral radius 0.99, 1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.75 |
| `sweep_robot_star_lr_{nominal,pushes,block}_filtered.toml` | leak rate 0.015, 0.02, 0.025, 0.03, 0.04, 0.05 |
| `sweep_robot_star_is_{nominal,pushes,block}_filtered.toml` | input scaling 1.5, 1.75, 2, 2.25 |

Under a disturbance, the sweep also measures, for each disturbance, how far the hand
strays from the taught path and how soon it is back within 5 mm of it
(`return_tolerance`), before the next disturbance starts or the run ends. The push
at 13 s, after the hold, checks that the hand comes back to the target: in Step A,
candidate #1 (spectral radius 1.5) settled 23 to 27 mm from the target after it,
outside the goal, which the undisturbed runs of stage 1 could not show. Under the
block, the reference's lead at the release shows how long the ESN waits for the
held arm.

The undisturbed runs repeat settings of stage 1, which they must reproduce. A
setting must, for all ten seeds, arrive and hold undisturbed and under the block,
and lose no seed under the pushes. A seed is lost if a pushed run never arrives or
ends more than 20 mm from the target: the lost ones drift back along the path,
toward the start. Settings are then compared by how soon the hand is back on the
path after the push at the target, with the slower of the two trackers.

A first rule also required every seed to arrive and hold under the pushes and to be
back on the path before each next push. It proved wrong, failing nearly every
setting: the push at 13 s falls inside the 2 s hold window when the hand arrives
after 11 s, and a hand still settling 5 to 7.5 mm from the path at the next push
counted as not back. Under the pushes, the `failures` of `sweep.csv` still count
these hold failures.

### Representatives of Step B to try

Some settings of the star, trained by [`autonomous_esn.py`](../autonomous_esn.py)
as the candidates above, so that their ESNs can be tried in the apps. Each takes its
setting's typical seed: the one nearest the setting's medians over its ten seeds in
the undisturbed path RMSE, the return onto the path after the first push and after
the push at the target, and the reference lead and the farthest distance from the
path under the block. Seed 8 is typical of #2's setting, of leak rate 0.02, and of
input scaling 1.5, so those three share a reservoir.

| Configuration | Why |
| --- | --- |
| `grid_lr0.03_sr1.35_is2_seed8_filtered.toml` | candidate #2's setting (#2 itself is seed 5) |
| `grid_lr0.02_sr1.35_is2_seed8_filtered.toml` | a slower leak rate, which came back the most reliably after the push at the target |
| `grid_lr0.025_sr1.35_is2_seed3_filtered.toml` | a leak rate between 0.02 and #2's 0.03 |
| `grid_lr0.03_sr1.35_is1.5_seed8_filtered.toml` | a weaker input, which came back the fastest after the push at the target |
| `grid_lr0.03_sr1.2_is2_seed2_filtered.toml` | a smaller spectral radius, at the low end of those that lost no seed |
| `grid_lr0.03_sr1.75_is2_seed9_filtered.toml` | the largest spectral radius, whose reference ran furthest ahead under the block |
| `grid_lr0.04_sr1.35_is2_seed7_filtered.toml` | a faster leak rate, which waited longer under the block but lost two of its ten seeds |
| `grid_lr0.04_sr1.35_is2_seed8_filtered.toml` | a lost seed of that faster leak rate, with the reservoir of the leak rates 0.02 and 0.03 above: after the push at the target, its hand drifted away from the target |

### A small grid

Step B varied one setting at a time. Around #2, a slower leak rate (0.02, 0.025)
and a weaker input (input scaling 1.5) brought the hand back onto the path sooner
after the push at the target, and a smaller spectral radius (1.2) kept the
reference from running as far ahead under the block. The small grid combines them,
with ten seeds each, in the three scenarios of Step B. `sweep_robot_esn.py` sweeps
at most three hyperparameters, so each spectral radius has its own configurations:

| Configurations | What they sweep |
| --- | --- |
| `sweep_robot_small_grid_sr{1.2,1.35}_{nominal,pushes,block}_filtered.toml` | leak rate 0.02, 0.025, 0.03 × input scaling 1.5, 2 × seeds 0–9, at one spectral radius |

The grid decides the ESN of Step C. A setting of the grid replaces #2's (leak rate
0.03, spectral radius 1.35, input scaling 2) if, over its ten seeds:

1. it passes Step B's rule above: every seed arrives and holds undisturbed and
   under the block, and no seed is lost under the pushes;
2. after the push at the target, its hand is back on the path sooner than with
   #2's setting for at least 8 of the 10 seeds, seed by seed, with a median gain
   over 0.3 s (the slower of the two trackers; a hand not back by the end of the
   run counts as back at the end);
3. it gives up no more than 0.1 mm in the median worst path RMSE undisturbed, and
   no more than 10 mm in the median farthest distance from the path under the
   block (the farther of the two trackers).

If several settings do, the one with the largest median gain wins, and the ESN of
Step C is its typical seed, chosen as for the representatives above. Otherwise, the
ESN of Step C is #2 itself.

Only input scaling 1.5, at leak rate 0.03 and spectral radius 1.35, replaced #2's
setting: back sooner for 8 of the 10 seeds, by 0.41 s in the median. Its typical
seed is 8 (`grid_lr0.03_sr1.35_is1.5_seed8_filtered.toml` above).

## Step C: the chosen ESN in report 005's scenarios

The chosen ESN (leak rate 0.03, spectral radius 1.35, input scaling 1.5, seed 8)
runs report 005's validation, as the candidates of Step A, each against the replay
of the take:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_search/pushes_robot_lr0.03_sr1.35_is1.5_seed8_filtered.toml
```

| Configurations | Scenario |
| --- | --- |
| `{nominal,block,pushes,offsets}_robot_lr0.03_sr1.35_is1.5_seed8_filtered.toml` | as in Step A |

Its reservoir states in those runs, recomputed from their logs, with
[`robot_states.py`](../robot_states.py):

```bash
uv run python experiments/robot_states.py experiments/manual_demonstration_v2_robot_search/states_robot_lr0.03_sr1.35_is1.5_seed8_filtered.toml
```
