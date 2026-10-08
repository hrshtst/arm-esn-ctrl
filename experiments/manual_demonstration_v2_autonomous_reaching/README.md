# Manual-demonstration autonomous reaching, v2 (Stage 1)

The procedure of
[manual-demonstration autonomous reaching](../manual_demonstration_autonomous_reaching/README.md)
(report 004) on a remade arm: its own robot links, start posture, and target,
with a new take taught by hand. An ESN trained on the take runs on its own, its
output fed back as its next input, from a grid of start postures around the
demonstrated one.

## The take

[`reach_manual_v2.toml`](../demonstrations/reach_manual_v2.toml) holds the arm, the
start posture, and the target (its lines marked `EDIT`), and the commands to record
the take with skelarm's trajectory recorder and to import it:

```bash
uv run python experiments/import_demonstrations.py experiments/demonstrations/reach_manual_v2.toml
```

The import's run directory, `results/demonstrations/<date>-<time>-reach_manual_v2`,
is what the configurations here name in `[demonstrations] run`. Its `metrics.csv`
gives the take's length (`length_s`) and its arrival (`arrival_time_s`), which set
the run durations and the end of the reach window below; when the hand starts
toward the target is read off the take, from its hand's distance to the target
over time.

## Base configurations

Each base is a starting point: copy it to a new file and edit the copy. The lines
to set are marked `EDIT` (`grep -n EDIT <file>`); the values are report 004's.

| Base | Runner | What it runs |
| --- | --- | --- |
| `grid_base_raw.toml` | `autonomous_esn.py` | one ESN, from a grid of start postures; its settings are candidate F's, and report 003's three settings are listed in its header |
| `sweep_base_raw.toml` | `sweep_esn.py` | up to three ESN settings swept, ranked by failures, then swings, then first-step jumps |
| `route_base_raw.toml` | `route_convergence.py` | how the runs of a grid run gather onto a common route |
| `states_base_raw.toml` | `reservoir_states.py` | the reservoir states of a trained ESN, by principal components |
| `warmup_base_raw.toml` | `warmup_esn.py` | a trained ESN with other warm-ups |

```bash
uv run python experiments/autonomous_esn.py experiments/manual_demonstration_v2_autonomous_reaching/<copy>.toml
```

The bases train on the take as recorded (`demo_00.sklog.npz`, `_raw`);
`demo_00_filtered.sklog.npz` is the take filtered.

## What depends on the take

- `[demonstrations] run`: the import's run directory.
- `[evaluation] duration`: the take's length and about 1 s more (16 s for report
  004's 14.8 s take); for `states_base_raw.toml`, the take's length itself.
- `[states] phase_window`: the take's reach, from heading for the target to
  arriving (4.1 s to 7.3 s in report 004).
- `[evaluation.start_grid]`: the offsets around the new start posture; check that
  the arm can take them (joint limits) and that they keep the elbow on the same side.

## Runs

Under `results/manual_demonstration_v2_autonomous_reaching/` in the storage.

| Run | Configuration |
| --- | --- |
