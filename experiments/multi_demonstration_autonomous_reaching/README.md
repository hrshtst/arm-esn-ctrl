# Multi-demonstration autonomous reaching (Stage 1)

An ESN trained on all eight time-varying-stiffness demonstrations runs on its own,
its output fed back as its next input, from the demonstrated start postures and
from start postures no demonstration starts from. Each run is compared with the
demonstrator's own reach from the same posture. The results are in
[report 001](../../reports/001-autonomous-reaching/README.md).

## Running the ESN autonomously

`experiments/autonomous_esn.py` trains an ESN on one or more demonstrations and
runs it autonomously. Each demonstration is learned from a reset reservoir with
its own warm-up, and one readout is fitted on all of them.

| Configuration | Runs from |
| --- | --- |
| `autonomous_tvs_all.toml` | the 8 demonstrated starts, and 8 new starts halfway between them |
| `autonomous_tvs_all_distances.toml` | as above, plus starts 0.25 m and 0.75 m from the target (the demonstrations start 0.5 m away), with the best settings of the sweeps |
| `autonomous_tvs_all_ridge1_leak0.2_input0.6.toml`, `autonomous_tvs_all_ridge1e-4_leak0.05_input0.1.toml` | as `autonomous_tvs_all.toml`, with two settings an early sweep found |

```bash
uv run python experiments/autonomous_esn.py \
    experiments/multi_demonstration_autonomous_reaching/autonomous_tvs_all_distances.toml
```

The configuration names the demonstration run and the training files
(`[demonstrations]`), the ESN hyperparameters (`[esn]`), and the start postures,
run duration, and hold duration (`[evaluation]`). Start postures that no
demonstration starts from are listed in named groups under
`[evaluation.extra_starts]`, such as `between`, `nearer`, and `farther`, and the
runner summarizes the metrics for each group. To explore a hyperparameter, copy
the file, change the value, and run the copy. Each run is recorded in its own
directory.

The run directory receives the trained ESN (`esn.toml`, with the hyperparameters
and the joint-angle normalization, and `esn.rclib`, rclib's model file), the ESN's
trajectories (`esn_00.sklog.npz`, ...), and the demonstrator's
(`demonstrator_00.sklog.npz`, ...), which both replay in skelarm's player. It also
receives their reach and hold metrics (`metrics.csv`, with the origin of each
start posture) and a figure of the hand paths, joint angles, and hand speeds
(`autonomous.png`), in which filled markers show demonstrated start postures and
hollow ones the others. The trained ESN can be watched live with
`tools/esn_reference_app.py` (see [`tools/README.md`](../../tools/README.md)).

## Sweeping hyperparameters

`experiments/sweep_esn.py` trains and runs the ESN for every combination of values
listed in a configuration's `[sweep]` table (two or three `[esn]`
hyperparameters), all on the same demonstrations and start postures:

| Configuration | Sweeps |
| --- | --- |
| `sweep_tvs_all.toml` | ridge, leak rate, and input scaling |
| `sweep_tvs_all_reservoir.toml` | reservoir size, spectral radius, and warm-up |

```bash
uv run python experiments/sweep_esn.py experiments/multi_demonstration_autonomous_reaching/sweep_tvs_all.toml
```

The run directory receives `sweep.csv`, with one row per combination: its reach
metrics averaged over the demonstrated and the other start postures, the number
of failed runs (never arriving or leaving the goal), and the median and largest
hold error. `sweep.png` shows heatmaps of the main metrics. To look at a
combination in detail, copy its values into a configuration for
`autonomous_esn.py`.

## Runs

Under `results/multi_demonstration_autonomous_reaching/` in the storage, all on
the demonstrations `results/demonstrations/20261002-194644-reach_tvs`:

| Run | Configuration | What it is |
| --- | --- | --- |
| `20261002-201614-sweep_tvs_all` | `sweep_tvs_all.toml` | ridge 10⁻⁶, leak rate 0.05, and input scaling 0.1 come out best (report 001, Section 3.2) |
| `20261002-202915-sweep_tvs_all_reservoir` | `sweep_tvs_all_reservoir.toml` | 600 neurons, spectral radius 1.3, and a 0.25 s warm-up come out best (Section 3.3) |
| `20261002-213015-autonomous_tvs_all_distances` | `autonomous_tvs_all_distances.toml` | the final ESN (Section 3.4); its `esn.toml` is the reference generator of Stage 2 |

`autonomous_tvs_all.toml` and the two `autonomous_tvs_all_ridge…` configurations
have no current runs; rerun them to get one.
