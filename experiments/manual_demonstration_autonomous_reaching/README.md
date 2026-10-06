# Manual-demonstration autonomous reaching (Stage 1)

The procedure of
[single-demonstration autonomous reaching](../single_demonstration_autonomous_reaching/README.md)
(report 003), with the scripted demonstration replaced by one taught by hand: an ESN
trained on one take runs on its own, its output fed back as its next input, from a
grid of start postures around the demonstrated one. The questions:

1. **Replication:** does the ESN reproduce the taught motion, the human's pause
   before moving included?
2. **Generalization:** from which start postures does it still arrive and hold,
   and does it return onto the taught motion or keep its start's offset?
3. **Noise:** the recorder moves the tip with the mouse cursor in whole screen
   pixels, about 5 mm each, so the take stalls and steps. Does training on the
   take as recorded differ from training on it filtered?

The take is a reach from report 003's start posture to its target, the hand 0.5 m
from the target toward the lower right, recorded with skelarm's trajectory recorder
and imported with
[`import_demonstrations.py`](../import_demonstrations.py)
([`reach_manual_single.toml`](../demonstrations/reach_manual_single.toml)). It
lasts 14.8 s: the hand moves 2 cm away from the target at the grab (0.8 s), waits,
reaches from about 4 s, arrives at 7.3 s, corrects inside the goal, and holds. The
import keeps it twice, as recorded (`demo_00`) and filtered by a zero-phase
first-order low-pass at 8 Hz (`demo_00_filtered`). There is no demonstrator: every
run is compared with the taught motion it was trained on and with the target.

## Phase 1: three settings over a grid of start postures

The settings are report 003's, each trained on the take as recorded (`_raw`) and
filtered (`_filtered`):

| Configurations | ESN settings |
| --- | --- |
| `grid_single_demo_settings_{raw,filtered}.toml` | those first used for a single demonstration: ridge 1, leak rate 0.3, input scaling 1, 300 neurons, spectral radius 0.5, warm-up 1 s |
| `grid_multi_demo_settings_{raw,filtered}.toml` | those found best for eight demonstrations: ridge 10⁻⁶, leak rate 0.05, input scaling 0.1, 600 neurons, spectral radius 1.3, warm-up 0.25 s |
| `grid_tuned_settings_{raw,filtered}.toml` | those report 003's sweeps found best for its scripted demonstration: ridge 10, leak rate 0.2, input scaling 3, 400 neurons, spectral radius 0.3, warm-up 1 s |

All run for 16 s from the demonstrated start offset by −15° to 15° in steps of 2.5°
in each joint (169 start postures):

```bash
uv run python experiments/autonomous_esn.py \
    experiments/manual_demonstration_autonomous_reaching/grid_single_demo_settings_raw.toml
```

The runner and its outputs are those of report 003, with the metrics against the
taught motion in place of those against the demonstrator: the joint error and the
path distance from the taught motion, the onset and arrival delays after it, the
offset retained (0 returns onto the taught motion, 1 keeps the start's offset), and
the jitter and speed peaks of the run and of the taught motion.

## Phase 2: reservoir states by principal components

As in report 003, the states while the take is fed in give the principal
components, and the runs from the grid are projected onto them and compared with
the take's states over time. The phase lead is measured over the reach, from 4.1 s,
when the hand heads for the target, to 7.3 s, when it arrives.

| Configurations | ESN |
| --- | --- |
| `states_single_demo_settings_{raw,filtered}.toml` | of `grid_single_demo_settings_{raw,filtered}.toml` |
| `states_multi_demo_settings_{raw,filtered}.toml` | of `grid_multi_demo_settings_{raw,filtered}.toml` |

```bash
uv run python experiments/reservoir_states.py \
    experiments/manual_demonstration_autonomous_reaching/states_single_demo_settings_raw.toml
```

## Phase 3: warm-up

As in report 003, each ESN runs again with warm-ups from 0 to 2.5 s: one that times
its reach from the reset of its reservoir arrives earlier by as much as the warm-up
is longer; one that times it from the end of the warm-up arrives as before.

| Configurations | ESN |
| --- | --- |
| `warmup_single_demo_settings_{raw,filtered}.toml` | of `grid_single_demo_settings_{raw,filtered}.toml` (trained with a 1 s warm-up) |
| `warmup_multi_demo_settings_{raw,filtered}.toml` | of `grid_multi_demo_settings_{raw,filtered}.toml` (trained with a 0.25 s warm-up) |

## Phase 4: sweeps

The settings of report 003 do not suit the take: the tuned ESN stays where the hand
paused, and those found best for eight demonstrations fail from every offset start.
The sweeps follow Lukoševičius, *A Practical Guide to Applying Echo State
Networks* (2012): first the three main reservoir parameters (§3.3.1), on a coarse
grid (§3.3.4), then the ridge on a logarithmic grid (§4.2), with the warm-up.

Every combination runs from the 169 start postures of Phase 1 and is ranked by
robustness: the fewest runs that fail to arrive and hold, then the fewest runs whose
first step is a jump of more than 30 mm rather than a mild return toward the taught
motion (`[ranking]`), then the smallest mean first step. How closely a run
reproduces the taught motion does not count.

| Configurations | Sweep |
| --- | --- |
| `sweep_main_ridge_{1e-6,1}_{raw,filtered}.toml` | spectral radius 0.3 to 1.5, leak rate 0.01 to 0.5, and input scaling 0.03 to 3 (150 combinations), at ridge 1e-6 or 1, with 400 neurons and a 1 s warm-up |

```bash
uv run python experiments/sweep_esn.py \
    experiments/manual_demonstration_autonomous_reaching/sweep_main_ridge_1_raw.toml
```

The ridge and the warm-up are then swept together for the reservoirs that ranked
best: the two most robust of each take, at either ridge, each trained on both takes.

| Configurations | Sweep |
| --- | --- |
| `sweep_ridge_warmup_{raw,filtered}_lr0.5_sr1.2_is0.03.toml` | ridge 1e-8 to 100 and warm-up 0 to 4 s (42 combinations), with leak rate 0.5, spectral radius 1.2, and input scaling 0.03: the only combination of the first sweeps from which no run fails and none jumps (raw take, ridge 1e-6) |
| `sweep_ridge_warmup_{raw,filtered}_lr0.2_sr0.6_is1.toml` | the same, with leak rate 0.2, spectral radius 0.6, and input scaling 1 |
| `sweep_ridge_warmup_{raw,filtered}_lr0.5_sr0.6_is1.toml` | the same, with leak rate 0.5, spectral radius 0.6, and input scaling 1 |

### Candidates and their routes

Two ESNs of the sweeps, both trained on the take as recorded and robust from every
start, return to the taught motion in opposite ways:

| Configuration | ESN | How it returns |
| --- | --- | --- |
| `grid_candidate_a_raw.toml` | leak rate 0.5, spectral radius 1.2, input scaling 0.03, ridge 1e-4 | no first step jumps, but many runs swing far before reaching the target |
| `grid_candidate_b_raw.toml` | leak rate 0.2, spectral radius 0.6, input scaling 1, ridge 1 | the first step of 73 runs jumps toward the taught path, then the runs follow it smoothly |

[`route_convergence.py`](../route_convergence.py) measures how the runs of a grid
run gather onto a common route, regardless of timing: the spread of a run is the
median distance of its hand from the routes the other runs take, and a run joins
the common route when its spread stays below 20 mm. `route_candidate_{a,b}_raw.toml`
measure the two candidates; the run directory receives `convergence.csv` and
`convergence.png`.

### Around candidate A

Candidate A returns to the taught path without jumping, but about half of its runs,
those that start with the elbow more bent, swing about 1 m away first. The next
sweeps look for settings without the swings, on the take as recorded only and from
49 start postures (every 5° up to ±15°). A run swings when its hand strays from the
taught path by more than 50 mm beyond where it started (`max_detour`); the ranking
is the fewest failures, then swings, then first steps over 30 mm, and `runs.csv`
keeps every run's detour and first step for other thresholds.

| Configurations | Sweep |
| --- | --- |
| `sweep_around_a_ridge_{1e-6,1e-5,1e-4,1e-3}_raw.toml` | input scaling 0.01 to 0.1, spectral radius 0.9 to 1.5, and leak rate 0.3 to 1 (125 combinations) |
| `sweep_noise_a_raw.toml` | noisy teacher forcing for candidate A: the noise on the input, 0 to 8°, with four noisy copies of the take, against the ridge, 1e-6 to 1e-3 (24 combinations) |
| `sweep_noise_raw_lr{0.7,0.5,0.4}_sr1.05_is0.1.toml` | the same for the three most robust reservoirs of `sweep_around_a_*`, against the ridge 1e-4 to 1e-1 |

The two best of `sweep_around_a_*`, with input scaling 0.1, spectral radius 1.05,
and ridge 1e-3, run from the 169 start postures as candidates C (leak rate 0.7) and
D (leak rate 0.5): `grid_candidate_{c,d}_raw.toml`.

## Runs

Under `results/manual_demonstration_autonomous_reaching/` in the storage, all on the
demonstration `results/demonstrations/20261006-152823-reach_manual_single`:

| Run | Configuration |
| --- | --- |
| `20261006-153058-grid_multi_demo_settings_filtered` | `grid_multi_demo_settings_filtered.toml` |
| `20261006-153139-grid_multi_demo_settings_raw` | `grid_multi_demo_settings_raw.toml` |
| `20261006-153219-grid_single_demo_settings_filtered` | `grid_single_demo_settings_filtered.toml` |
| `20261006-153239-grid_single_demo_settings_raw` | `grid_single_demo_settings_raw.toml` |
| `20261006-153258-grid_tuned_settings_filtered` | `grid_tuned_settings_filtered.toml` |
| `20261006-153325-grid_tuned_settings_raw` | `grid_tuned_settings_raw.toml` |
| `20261006-161842-states_multi_demo_settings_filtered` | `states_multi_demo_settings_filtered.toml` |
| `20261006-161843-states_multi_demo_settings_raw` | `states_multi_demo_settings_raw.toml` |
| `20261006-161918-states_single_demo_settings_filtered` | `states_single_demo_settings_filtered.toml` |
| `20261006-161919-states_single_demo_settings_raw` | `states_single_demo_settings_raw.toml` |
| `20261006-161945-warmup_multi_demo_settings_filtered` | `warmup_multi_demo_settings_filtered.toml` |
| `20261006-161946-warmup_multi_demo_settings_raw` | `warmup_multi_demo_settings_raw.toml` |
| `20261006-162318-warmup_single_demo_settings_filtered` | `warmup_single_demo_settings_filtered.toml` |
| `20261006-162355-warmup_single_demo_settings_raw` | `warmup_single_demo_settings_raw.toml` |
| `20261006-175503-sweep_main_ridge_1_filtered` | `sweep_main_ridge_1_filtered.toml` |
| `20261006-175503-sweep_main_ridge_1_raw` | `sweep_main_ridge_1_raw.toml` |
| `20261006-175503-sweep_main_ridge_1e-6_filtered` | `sweep_main_ridge_1e-6_filtered.toml` |
| `20261006-175503-sweep_main_ridge_1e-6_raw` | `sweep_main_ridge_1e-6_raw.toml` |
| `20261006-192518-sweep_ridge_warmup_filtered_lr0.2_sr0.6_is1` | `sweep_ridge_warmup_filtered_lr0.2_sr0.6_is1.toml` |
| `20261006-192518-sweep_ridge_warmup_filtered_lr0.5_sr0.6_is1` | `sweep_ridge_warmup_filtered_lr0.5_sr0.6_is1.toml` |
| `20261006-192518-sweep_ridge_warmup_filtered_lr0.5_sr1.2_is0.03` | `sweep_ridge_warmup_filtered_lr0.5_sr1.2_is0.03.toml` |
| `20261006-192518-sweep_ridge_warmup_raw_lr0.2_sr0.6_is1` | `sweep_ridge_warmup_raw_lr0.2_sr0.6_is1.toml` |
| `20261006-192518-sweep_ridge_warmup_raw_lr0.5_sr0.6_is1` | `sweep_ridge_warmup_raw_lr0.5_sr0.6_is1.toml` |
| `20261006-192518-sweep_ridge_warmup_raw_lr0.5_sr1.2_is0.03` | `sweep_ridge_warmup_raw_lr0.5_sr1.2_is0.03.toml` |
