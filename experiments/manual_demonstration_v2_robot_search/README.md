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
