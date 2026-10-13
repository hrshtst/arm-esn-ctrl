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

## Then

- **Stage B:** report 009's grid (leak rate 0.015 to 0.1, spectral radius 0.99 to
  1.75, input scaling 1 to 2, ten seeds each), trained with stage A's noise at ridge
  4e-2 with four copies, scored as there, against the same reference; the top three
  settings, each with its typical seed, and report 009's chosen ESN are tried in the
  robot app, where one is chosen.
- **The validation:** the chosen ESN against the replay, as in report 009, and under
  disturbances the search did not use: single pushes at other times, in other
  directions, and of other forces, and a block at another time.
