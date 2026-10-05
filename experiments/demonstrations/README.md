# Demonstrations

Reaching demonstrations of a two-link planar arm, which the ESN experiments learn
from. Demonstrations are skelarm state logs (`*.sklog.npz`). They are either
scripted with one of skelarm's reaching controllers or taught with the mouse.

## Scripted

Each configuration here runs one controller from 8 start postures, each 0.5 m
from a common target:

| Configuration | Controller | Reaches |
| --- | --- | --- |
| `reach_tvs.toml` | virtual spring-damper with time-varying stiffness | human-like: smooth bell-shaped speed peaking a little early, gently curved paths |
| `reach_pds.toml` | online reference shaping with a position-dependent ratio | human-like: nearly straight paths, speed peaking at mid-movement with a small shoulder early on |
| `reach_vsd.toml` | constant virtual spring-damper | not human-like, kept for comparison: the speed peaks almost at once |

```bash
uv run python experiments/make_demonstrations.py experiments/demonstrations/reach_tvs.toml
```

The run directory receives one log per start posture (`demo_00.sklog.npz`, ...),
their reach metrics (`metrics.csv`, defined in `src/arm_esn_ctrl/metrics.py`), and
a figure of the hand paths, joint-space paths, and speed profiles
(`demonstrations.png`). Replay a demonstration with skelarm's player:

```bash
uv run python third_party/skelarm/tools/player.py <run directory>/demo_00.sklog.npz
```

The `[skeleton]`, `[task]`, `[simulator]`, and `[controller]` tables follow
skelarm's scenario format, so skelarm's tools also read these files;
`[demonstrations]` lists the start postures in degrees.

## Taught

skelarm's trajectory recorder reads the same configuration files: it shows the arm
and the target, and you drag the arm tip with the mouse. `--pose` sets the start
posture in degrees (take one from `start_q`), and `--multi-take` numbers the saved
takes (`reach_001.sklog.npz`, ...). The recorder does not create the output
directory. With the default storage root:

```bash
mkdir -p storage/data/taught_reach
uv run python third_party/skelarm/tools/trajectory_recorder.py \
    experiments/demonstrations/reach_tvs.toml --pose 29.4,88.2 \
    --multi-take --output storage/data/taught_reach/reach.sklog.npz --show-past-trails
```

## Runs

Under `results/demonstrations/` in the storage:

| Run | Configuration | Used by |
| --- | --- | --- |
| `20261002-194644-reach_tvs` | `reach_tvs.toml` | every ESN experiment so far; reports 001 and 002 |
| `20261002-194650-reach_pds` | `reach_pds.toml` | |
| `20261002-194656-reach_vsd` | `reach_vsd.toml` | |

Each demonstration lasts 4 s: a reach of about 1.1 s, then about 3 s holding
still at the target.
