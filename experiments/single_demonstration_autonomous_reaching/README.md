# Single-demonstration autonomous reaching (Stage 1)

An ESN trained on one demonstration runs on its own, its output fed back as its
next input, from a grid of start postures around the demonstrated one. Each run is
compared with the demonstrator's own reach from the same posture. The questions:

1. **Replication:** does the ESN reproduce the demonstration?
2. **Generalization:** from which start postures does it still arrive and hold,
   and is its reach then like the demonstrator's from there?
3. **What it learned:** a trajectory or a flow? From an offset start, does it
   return onto the demonstrated path, or reach as the demonstrator would from
   there? How do its reservoir states from offset starts relate to the
   demonstration's?
4. **What shapes it:** the hyperparameters, and the scale of the joint-angle
   normalization, which from one demonstration covers only that reach's range.

The demonstration is the time-varying-stiffness reach from one start posture, the
hand 0.5 m from the target toward the lower right
([`reach_tvs_single.toml`](../demonstrations/reach_tvs_single.toml)).

## Phase 1: two settings over a grid of start postures

| Configuration | ESN settings |
| --- | --- |
| `grid_single_demo_settings.toml` | those first used for a single demonstration: ridge 1, leak rate 0.3, input scaling 1, 300 neurons, spectral radius 0.5, warm-up 1 s |
| `grid_multi_demo_settings.toml` | those found best for eight demonstrations: ridge 10⁻⁶, leak rate 0.05, input scaling 0.1, 600 neurons, spectral radius 1.3, warm-up 0.25 s |

Both run from the demonstrated start offset by −15° to 15° in steps of 2.5° in
each joint (`[evaluation.start_grid]`, 169 start postures):

```bash
uv run python experiments/autonomous_esn.py \
    experiments/single_demonstration_autonomous_reaching/grid_multi_demo_settings.toml
```

The runner and its outputs are those of
[multi-demonstration autonomous reaching](../multi_demonstration_autonomous_reaching/README.md),
plus `grid.png`: maps over the grid of the outcome (arrive and hold, leave the
goal, or never arrive), the path distance from the demonstrator's reach from each
start, the training path ratio, the first step, and the hold error, and where the
start postures put the hand. The **training path ratio** compares how far, in joint
space, the ESN's reach stays from the demonstrated path with how far the
demonstrator's own reach from the same start does: near 0, the ESN returns onto
the demonstrated path; near 1, it reaches as the demonstrator would from there.

## Phase 2: reservoir states by principal components

`experiments/reservoir_states.py` loads a Phase 1 ESN and records its reservoir
state at every step, while the demonstration is fed in (teacher forcing) and in the
autonomous runs from the same grid of start postures. The principal components of
the demonstration's states are the axes every state is projected onto, and every
run's states are compared with the demonstration's over time, in the full state
space: how far they are at the same time, and whether they keep the
demonstration's timing (the phase of the nearest demonstration state).

| Configuration | ESN |
| --- | --- |
| `states_single_demo_settings.toml` | of `grid_single_demo_settings.toml` |
| `states_multi_demo_settings.toml` | of `grid_multi_demo_settings.toml` |

```bash
uv run python experiments/reservoir_states.py \
    experiments/single_demonstration_autonomous_reaching/states_multi_demo_settings.toml
```

The run directory receives `states.csv` (per start posture: the distance from the
demonstration's state at the end of the warm-up and later, when the run joins the
demonstration's states, and its phase lead), `projections.npz` (every state on the
first principal components), `pca.png`, and `convergence.png` (the distance and the
phase lead over time, and maps of them over the start offsets).

## Runs

Under `results/single_demonstration_autonomous_reaching/` in the storage, both on
the demonstration `results/demonstrations/20261005-185525-reach_tvs_single`:

| Run | Configuration |
| --- | --- |
| `20261005-190403-grid_single_demo_settings` | `grid_single_demo_settings.toml` |
| `20261005-190405-grid_multi_demo_settings` | `grid_multi_demo_settings.toml` |
| `20261005-191929-states_single_demo_settings` | `states_single_demo_settings.toml` |
| `20261005-191932-states_multi_demo_settings` | `states_multi_demo_settings.toml` |
