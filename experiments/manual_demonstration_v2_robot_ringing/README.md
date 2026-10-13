# Manual-demonstration robot ringing, v2 (Stage 2)

The study after [the wider robot search](../manual_demonstration_v2_robot_search/README.md)
(report 006). There, with joint PD, report 006's chosen ESN made the arm ring less
than the replay of the take after a block or a push: its reference follows the
measured arm, so the tracker starts from a smaller gap. But a push of 0.2 s ended
before the ESN could follow. Here the pushes last 0.5 s, and the search is scored
with joint PD, by the ringing after the block and the pushes.

## The pushes

Three pushes of 0.5 s at the tip, in runs of 24 s: 3 N nearly forward at 3 s
(report 006's backward push, reversed), 5 N nearly across at 7 s, and 4 N nearly
across the other way at 15 s, after the hold. The block stays report 006's: the tip
held from 4.5 s to 5.5 s.

- **0.5 s, not 1 s:** when a 0.5 s push ends, the arm is still moving; after 1 s,
  it has nearly settled where the push holds it, which would make the push a block
  away from the path.
- **The forces across, report 006's:** a push carries the replay farthest at its
  first overshoot, about 0.16 s in, so the replay is carried as far as by report
  006's 0.2 s pushes.
- **The push forward, 3 N:** it sends the ESN ahead on a route of its own (65 mm
  off the path, against the replay's 9 mm). With 4 N, report 006's chosen ESN is
  still 10 mm off the path when the push at 7 s comes, and its hold fails if the
  pushes last 0.55 s; with 3 N, it is 5 mm off, and it holds either way.
- **The times:** the push at 7 s ends 2.1 s before the take arrives, at 9.6 s, so
  that the replay has stopped ringing by then; at 8 s, the replay still rings, and
  the ESN is pushed into the goal early, which spoils the hold. The push at 3 s
  leaves 3.5 s to settle before it.

To watch them with report 006's chosen ESN and the replay, both with joint PD:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_ringing/pushes_robot_lr0.03_sr1.35_is1.5_seed8_filtered.toml
uv run python third_party/skelarm/tools/player.py <run directory>/pd_w20_z0.1/replay_00.sklog.npz
```

## Step 1: a grid, with joint PD

[`sweep_robot_esn.py`](../sweep_robot_esn.py) trains every setting with ten seeds
and runs it with joint PD from the demonstrated start, undisturbed, under the
block, and under the pushes, in runs of 24 s:

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_ringing/sweep_robot_ringing_lr0.03_pushes_filtered.toml
```

| Configurations | What they sweep |
| --- | --- |
| `sweep_robot_ringing_lr{0.02,0.03,0.04,0.05,0.07,0.1}_{nominal,block,pushes}_filtered.toml` | spectral radius 0.99, 1.2, 1.35, 1.5, 1.75 × input scaling 1, 1.5, 2 × seeds 0–9, at one leak rate |

That is 90 settings, among them report 006's chosen setting (leak rate 0.03,
spectral radius 1.35, input scaling 1.5). The other settings of the ESN are reports
005 and 006's: 400 neurons, sparsity 0.05, ridge 1e-2, and a 1 s warm-up.

The rule, fixed before the runs. With joint PD, a seed qualifies if, in each of the
three scenarios, its run arrives and holds (`failed` is false in `runs.csv`) and
ends within the goal, 20 mm from the target. Its **ringing** is the mean of
`ringing_m` over the four disturbances: the block's release and the three pushes.

1. A setting qualifies if all ten of its seeds do.
2. The qualifying settings are ranked by the median of the ringing over their
   seeds, the smallest first.
3. Alongside, each setting's medians over its seeds: the undisturbed path RMSE, and,
   after each disturbance, the ringing, the farthest from the path, and the return
   onto it; and the block's reference lead.
4. The shortlist is the top three settings, each with its typical seed, and report
   006's chosen ESN (leak rate 0.03, spectral radius 1.35, input scaling 1.5, seed
   8). A setting's typical seed is the one nearest its medians over its seeds of the
   undisturbed path RMSE, the ringing, and the latest return after a disturbance
   (a hand not back before the next disturbance or the end of the run counts as back
   then), each scaled by its interquartile range over the seeds.

The shortlisted ESNs are trained by [`autonomous_esn.py`](../autonomous_esn.py), so
that they can be tried in the apps, where one is chosen by hand.

The rule shortlisted three settings, all at leak rate 0.02: under the pushes, most
of the others have seeds that reach the goal while still swinging after the push at
7 s and leave it during the hold, or that end away from the target. Report 006's
setting has four such seeds, but its chosen ESN, seed 8, qualifies.

## Step 2: the shortlist, chosen in the apps

The shortlisted ESNs, each trained by [`autonomous_esn.py`](../autonomous_esn.py)
with its setting's typical seed, so that they can be tried in the apps:

```bash
uv run python experiments/autonomous_esn.py experiments/manual_demonstration_v2_robot_ringing/grid_lr0.02_sr0.99_is1.5_seed8_filtered.toml
```

| Configuration | Ringing (median over the seeds) | Undisturbed path RMSE |
| --- | --- | --- |
| `grid_lr0.02_sr0.99_is1.5_seed8_filtered.toml` | 10.1 mm | 1.22 mm |
| `grid_lr0.02_sr1.35_is2_seed7_filtered.toml` | 10.7 mm | 0.96 mm |
| `grid_lr0.02_sr1.35_is1.5_seed6_filtered.toml` | 11.0 mm | 0.94 mm |

With them, report 006's chosen ESN
(`manual_demonstration_v2_robot_search/grid_lr0.03_sr1.35_is1.5_seed8_filtered.toml`,
already trained).
