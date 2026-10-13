# Manual-demonstration robot damped search, v2 (Stage 2)

Results: [report 009](../../reports/009-noisy-training-for-a-faster-return/README.md).

The study after [the damping study](../manual_demonstration_v2_robot_damping/README.md)
(report 008). There, with joint PD at a damping ratio of 0.4 or more, the tracker
removed most of the ringing by itself, and what set report 007's chosen ESN apart
from the replay of the take was its slow return onto the taught path after a
disturbance: 2.5 to 3.6 s, against the replay's 0.5 s at most. Here the ESN is
searched again with such a tracker, and scored by the ringing and the return
together.

- **The tracker:** joint PD at ω = 20 rad/s and a damping ratio of 0.4, given the
  reference's velocity (the change of the reference over each 10 ms period), with
  its torque computed every 10 ms, the ESN's period, and held between
  (`control_period`), while the arm is still simulated every 2 ms. The search, the
  choice, and the validation all compute the torque every 10 ms.
- **The scenarios:** report 007's: undisturbed, the block, and the three pushes, in
  runs of 24 s; for the validation, also the 49 offset starts.

## Step 1: a grid

[`sweep_robot_esn.py`](../sweep_robot_esn.py) trains every setting with ten seeds
and runs it with the tracker from the demonstrated start, undisturbed, under the
block, and under the pushes:

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_damped_search/sweep_robot_damped_lr0.02_pushes_filtered.toml
```

| Configurations | What they sweep |
| --- | --- |
| `sweep_robot_damped_lr{0.015,0.02,0.03,0.04,0.05,0.07,0.1}_{nominal,block,pushes}_filtered.toml` | spectral radius 0.99, 1.2, 1.35, 1.5, 1.75 × input scaling 1, 1.5, 2 × seeds 0–9, at one leak rate |

That is report 007's grid with leak rate 0.015 added, since report 007's best
settings lay at its lowest leak rate, 0.02: 105 settings. Among them is report
007's chosen setting (leak rate 0.02, spectral radius 1.35, input scaling 2), whose
seed 7, report 007's chosen ESN, repeats report 008's runs of it with this tracker.
The other settings of the ESN are report 007's: 400 neurons, sparsity 0.05, ridge
1e-2, and a 1 s warm-up.

The rule, fixed before the runs. A seed qualifies if, in each of the three
scenarios, its run arrives and holds (`failed` is false in `runs.csv`) and ends
within the goal, 20 mm from the target. Over the four disturbances, the block's
release and the three pushes, its **ringing** is the mean of `ringing_m`, and its
**return** the mean of `return_time_s`, where a hand not back before the next
disturbance or the end of the run counts as back then.

1. A setting qualifies if all ten of its seeds do.
2. Each setting's **score** is the mean of two ratios: the median of its seeds'
   ringing to that of report 007's chosen setting, and the median of its seeds'
   return to that of report 007's chosen setting, whether or not that setting
   qualifies. Report 007's chosen setting scores 1; below 1 is better on balance,
   where a return 10% faster counts as much as a ringing 10% smaller. The
   qualifying settings are ranked by the score, the smallest first.
3. Alongside, each setting's medians over its seeds: the undisturbed path RMSE;
   after each disturbance, the ringing, the farthest from the path, and the return;
   and under the block, the reference lead and the peak torque.
4. The shortlist is the top three settings, each with its typical seed, and report
   007's chosen ESN. A setting's typical seed is the one nearest its medians over
   its seeds of the undisturbed path RMSE, the ringing, and the return, each scaled
   by its interquartile range over the seeds.

The score keeps the ringing alongside the return: ranked by the return alone, the
search would favor ESNs that follow the arm less, closer to the replay, and lose
what the ESN gives under the block.

## Step 1b: noise in the training

In the grid, report 007's chosen setting returned the fastest of all 105 settings,
in 2.66 s, the median over its seeds. The leak rate, the spectral radius, and the
input scaling shape how the reservoir responds to the posture, but none of them
teaches the ESN what to do off the taught path. Noisy teacher forcing does (see
[`esn.py`](../../src/arm_esn_ctrl/esn.py)): the take is fed in several times, each
time with noise added to the input while the target stays the clean take one step
ahead, so that from a posture off the take the ESN learns to give the take's next
posture. Report 004 tried white noise, at leak rates of 0.5 to 1. At a leak rate of
0.02, the reservoir averages its input over about 0.5 s: a first-order low-pass with
that time constant keeps about 10% of the deviation of white noise, 54% of pink
noise's, and 88% of brown noise's, which drifts as a disturbed arm does. So the
noise's color is swept with its level:

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_damped_search/sweep_robot_damped_noise_ridge4e-2_pushes_filtered.toml
```

| Configurations | What they sweep |
| --- | --- |
| `sweep_robot_damped_noise_ridge{1e-2,4e-2}_{nominal,block,pushes}_filtered.toml` | noise of 0, 0.5, 1, 2, 4, 8 deg × exponent 0, 1, 2 (white, pink, brown noise) × seeds 0–9, at one ridge |

- **The noise:** `teacher_noise_deg` is its standard deviation over each copy of the
  take, in each joint, and `teacher_noise_exponent` colors it: its power spectrum
  goes as 1/f to the exponent, 0 for white noise, 1 for pink, and 2 for brown.
- **Four copies at every level, the noiseless included.** Every copy weakens the
  ridge, so ridge 4e-2 with four copies weighs as report 007's 1e-2 with one: the
  noiseless setting at ridge 4e-2 repeats report 007's chosen setting, and the noise
  is all that changes from it. Ridge 1e-2, a quarter of that, leaves more of the
  regularization to the noise.
- **Otherwise as the grid:** report 007's chosen setting (leak rate 0.02, spectral
  radius 1.35, input scaling 2, 400 neurons, sparsity 0.05, a 1 s warm-up), and the
  grid's tracker, scenarios, and evaluation. A seed draws the reservoir and the
  noise.

That is 32 settings: at each ridge, the noiseless one is tried with each exponent,
the same each time. The rule, fixed before the runs, is step 1's:

1. A setting qualifies if all ten of its seeds do, as in step 1.
2. Its score is step 1's, against the same reference: report 007's chosen setting
   in the grid, which the noiseless setting at ridge 4e-2 repeats.
3. Alongside, step 1's medians over the seeds, and the peak torque over the run
   under the pushes.
4. For each color, the qualifying setting with noise that scores the lowest joins
   the shortlist with its typical seed, if it scores below 1: at most three more
   ESNs.

## Step 2: the shortlist, chosen in the app

The shortlisted ESNs, each trained by [`autonomous_esn.py`](../autonomous_esn.py)
with its setting's typical seed and run on its own from the 49 offset starts, so
that they can be tried in [the robot app](../../tools/README.md) with the search's
tracker:

```bash
uv run python experiments/autonomous_esn.py experiments/manual_demonstration_v2_robot_damped_search/noise_white1deg_ridge4e-2_seed3_filtered.toml
uv run python tools/robot_app.py experiments/demonstrations/reach_manual_v2.toml \
    --law pd --omega 20 --damping 0.4 --control-period 0.01 --model <run directory>/esn.toml
```

| Configuration | Score | Ringing | Return | Undisturbed path RMSE |
| --- | --- | --- | --- | --- |
| `grid_lr0.02_sr1.35_is1.5_seed0_filtered.toml` | 1.060 | 6.07 mm | 2.88 s | 0.67 mm |
| `grid_lr0.02_sr0.99_is1.5_seed6_filtered.toml` | 1.142 | 5.59 mm | 3.54 s | 0.83 mm |
| `noise_white1deg_ridge4e-2_seed3_filtered.toml` | 0.821 | 7.87 mm | 0.80 s | 0.60 mm |
| `noise_pink0.5deg_ridge4e-2_seed8_filtered.toml` | 0.919 | 8.15 mm | 1.18 s | 0.72 mm |

The score and the medians are those of each ESN's setting over its seeds, step 1's
or step 1b's. No setting with brown noise qualifies. With them, report 007's chosen
ESN
([`manual_demonstration_v2_robot_ringing/grid_lr0.02_sr1.35_is2_seed7_filtered.toml`](../manual_demonstration_v2_robot_ringing/grid_lr0.02_sr1.35_is2_seed7_filtered.toml),
already trained), whose setting scores 1: ringing 5.86 mm, return 2.66 s, path RMSE
0.58 mm.

`noise_white1deg_ridge4e-2_seed3_filtered.toml`, trained with white noise, was
chosen in the app.

## Step 3: the chosen ESN against the replay

[`robot_esn.py`](../robot_esn.py) runs the chosen ESN and the replay of the take with
joint PD at damping ratios 0.1, 0.4, and 0.8, each given a zero reference velocity
or the reference's, all with the torque computed every 10 ms and held between:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_damped_search/pushes_robot_white1deg_seed3_reference_velocity_filtered.toml
```

| Configurations | Tracker |
| --- | --- |
| `{nominal,block,pushes,offsets}_robot_white1deg_seed3_zero_velocity_filtered.toml` | joint PD at the three damping ratios, given a zero reference velocity |
| `{nominal,block,pushes,offsets}_robot_white1deg_seed3_reference_velocity_filtered.toml` | the same, given the reference's velocity |

They are report 008's configurations with the torque computed every 10 ms
(`*_control10ms_filtered.toml` of [the damping study](../manual_demonstration_v2_robot_damping/README.md)),
with the chosen ESN in place of report 007's, so that the replay repeats report
008's runs: undisturbed, under the block, under the pushes, and from the 49 offset
starts.

[`robot_states.py`](../robot_states.py) then shows the chosen ESN's reservoir states in
the runs given the reference's velocity, at the search's damping ratio of 0.4
(`states_robot_white1deg_seed3_filtered.toml`), as report 007 did for its chosen ESN.
