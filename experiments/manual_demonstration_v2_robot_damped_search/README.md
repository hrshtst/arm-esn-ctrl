# Manual-demonstration robot damped search, v2 (Stage 2)

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

## Then

The shortlisted ESNs are trained by [`autonomous_esn.py`](../autonomous_esn.py), so
that they can be tried in [the robot app](../../tools/README.md) with the search's
tracker (`--law pd --omega 20 --damping 0.4 --control-period 0.01`), where one is
chosen by hand. The chosen ESN then runs against the replay with joint PD at damping
ratios 0.1, 0.4, and 0.8, each given a zero reference velocity or the reference's,
all with the torque computed every 10 ms: undisturbed, under the block, under the
pushes, and from the 49 offset starts.
