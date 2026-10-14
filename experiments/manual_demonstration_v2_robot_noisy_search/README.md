# Manual-demonstration robot noisy search, v2 (Stage 2)

The study after [the damped search](../manual_demonstration_v2_robot_damped_search/README.md)
(report 009). There, training report 007's chosen setting with white noise of 1° on
the input brought its return onto the taught path after a disturbance from 2.66 s to
0.80 s, the median over ten seeds; pink noise did less, and brown noise spoiled the
reach. Report 009 read this, without testing it, as the noise teaching the readout
which changes of its input to disregard, by how fast they change: white noise
changes as fast as a pushed arm, brown noise as slowly as the reach itself. Here,
stage A tests that reading with noise faster than white; stage B searches the
reservoir again, trained with the best noise; and the ESN chosen is validated
against the replay, also under disturbances that the search did not use.

- **The tracker:** report 009's search's, joint PD at ω = 20 rad/s and a damping ratio
  of 0.4, given the reference's velocity, with its torque computed every 10 ms and
  held between, while the arm is still simulated every 2 ms.
- **The scenarios:** report 007's: undisturbed, the block, and the three pushes, in
  runs of 24 s.

## Stage A: noise faster than white

[`sweep_robot_esn.py`](../sweep_robot_esn.py) trains report 007's chosen setting
(leak rate 0.02, spectral radius 1.35, input scaling 2) with violet, blue, and white
noise, ten seeds each, and runs each with the tracker undisturbed, under the block,
and under the pushes:

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_noisy_search/sweep_robot_noise_spectrum_pushes_filtered.toml
```

| Configurations | What they sweep |
| --- | --- |
| `sweep_robot_noise_spectrum_{nominal,block,pushes}_filtered.toml` | noise of 0.5, 1, 2, 4, 8 deg × exponent −2, −1, 0 (violet, blue, white noise) × seeds 0–9 |

- **The noise:** as in report 009, `teacher_noise_deg` is its standard deviation
  over each copy of the take, in each joint, and `teacher_noise_exponent` colors it:
  its power spectrum goes as 1/f to the exponent, so that a negative one gives noise
  whose power rises with the frequency, −1 for blue noise and −2 for violet.
- **Otherwise as report 009's noise sweep** at ridge 4e-2, with four copies of the
  take at every level, as report 007's ridge of 1e-2 with one copy. The white noise
  repeats report 009's runs, as a check; its pink and brown noise and the noiseless
  setting are not run again.

The rule, fixed before the runs:

1. A setting qualifies, and is scored, as in report 009's noise sweep: it qualifies
   if all ten of its seeds arrive, hold, and end within the goal in the three
   scenarios, and its score is the mean of the ratios of the medians over its seeds
   of the ringing and of the return, each the mean over the four disturbances, to
   those of report 007's chosen setting in report 009's grid (5.86 mm and 2.66 s).
2. The reading predicts that blue or violet noise, at some level, scores no higher
   than white noise at its best (0.821, at 1°) with an undisturbed path RMSE no
   larger (0.60 mm in the median). The report says whether it holds.
3. Stage B trains with the noise of the qualifying setting that scores the lowest
   here, its exponent and its level.

## Stage B: the grid, trained with the noise

Stage A's best qualifying setting was white noise of 1°, report 009's, so stage B
trains report 009's grid with it: leak rate 0.015 to 0.1, spectral radius 0.99 to
1.75, input scaling 1 to 2, ten seeds each, the take fed in four times at ridge
4e-2, each with white noise of 1° on the input:

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_noisy_search/sweep_robot_noisy_lr0.05_pushes_filtered.toml
```

| Configurations | What they sweep |
| --- | --- |
| `sweep_robot_noisy_lr{0.015,0.02,0.03,0.04,0.05,0.07,0.1}_{nominal,block,pushes}_filtered.toml` | spectral radius 0.99, 1.2, 1.35, 1.5, 1.75 × input scaling 1, 1.5, 2 × seeds 0–9, at one leak rate |

Report 007's setting (leak rate 0.02, spectral radius 1.35, input scaling 2) is in
the grid: with this noise, it is report 009's chosen setting, whose runs it repeats.

The rule, fixed before the runs:

1. A setting qualifies, and is scored, as in stage A, against the same reference:
   report 007's chosen setting without noise, in report 009's grid. Report 009's
   chosen setting scores 0.821.
2. Alongside, report 009's medians over the seeds: the undisturbed path RMSE; after
   each disturbance, the ringing, the farthest from the path, and the return; and
   under the block, the reference lead and the peak torque.
3. The shortlist is the top three settings, each with its typical seed (the one
   nearest its medians of the undisturbed path RMSE, the ringing, and the return,
   each scaled by its interquartile range over the seeds), and report 009's chosen
   ESN. They are trained by [`autonomous_esn.py`](../autonomous_esn.py) and tried in
   [the robot app](../../tools/README.md) with the tracker, where one is chosen.

## Stage B: the grid's edge

Stage B's top three settings all sat at its grid's corner, its fastest leak rate
(0.1), its largest input scaling (2), and its smallest spectral radii (0.99 to 1.35),
so its grid is extended beyond the corner before the shortlist: leak rate 0.1 to 0.3,
spectral radius 0.8 to 0.99, input scaling 2 to 3, ten seeds each, trained as in
stage B:

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_noisy_search/sweep_robot_noisy_edge_lr0.2_pushes_filtered.toml
```

| Configurations | What they sweep |
| --- | --- |
| `sweep_robot_noisy_edge_lr{0.1,0.15,0.2,0.3}_{nominal,block,pushes}_filtered.toml` | spectral radius 0.8, 0.9, 0.99 × input scaling 2, 2.5, 3 × seeds 0–9, at one leak rate |

Stage B's top setting (leak rate 0.1, spectral radius 0.99, input scaling 2) is in
the extension, whose runs of it repeat stage B's.

The rule, fixed before the runs:

1. A setting qualifies, and is scored, as in stage B, against the same reference,
   with the same medians alongside.
2. Stage B's shortlist is drawn from both grids together: the top three qualifying
   settings of stage B's grid and the extension, each with its typical seed, and
   report 009's chosen ESN.
3. The search stops here: if the top setting is again at the extension's edge, the
   report says so, and the grid is not extended further.

## The shortlist, chosen in the app

The shortlisted ESNs, each trained by [`autonomous_esn.py`](../autonomous_esn.py)
with its setting's typical seed and run on its own from the 49 offset starts, so
that they can be tried in [the robot app](../../tools/README.md) with the search's
tracker:

```bash
uv run python experiments/autonomous_esn.py experiments/manual_demonstration_v2_robot_noisy_search/grid_lr0.1_sr0.8_is2.5_seed9_filtered.toml
uv run python tools/robot_app.py experiments/demonstrations/reach_manual_v2.toml \
    --law pd --omega 20 --damping 0.4 --control-period 0.01 --model <run directory>/esn.toml
```

| Configuration | Score | Ringing | Return | Undisturbed path RMSE |
| --- | --- | --- | --- | --- |
| `grid_lr0.1_sr0.8_is2.5_seed9_filtered.toml` | 0.618 | 6.48 mm | 0.34 s | 0.57 mm |
| `grid_lr0.15_sr0.9_is2_seed4_filtered.toml` | 0.625 | 6.50 mm | 0.37 s | 0.63 mm |
| `grid_lr0.15_sr0.8_is2_seed7_filtered.toml` | 0.631 | 6.63 mm | 0.35 s | 0.69 mm |

The score and the medians are those of each ESN's setting over its seeds, all three
from the edge check; the top one is at the extension's edge, its smallest spectral
radius. With them, report 009's chosen ESN
([`manual_demonstration_v2_robot_damped_search/noise_white1deg_ridge4e-2_seed3_filtered.toml`](../manual_demonstration_v2_robot_damped_search/noise_white1deg_ridge4e-2_seed3_filtered.toml),
already trained), whose setting scores 0.821: ringing 7.87 mm, return 0.80 s, path
RMSE 0.60 mm.

## Then

- **The validation:** the chosen ESN against the replay, as in report 009, and under
  disturbances the search did not use: single pushes at other times, in other
  directions, and of other forces, and a block at another time.
