# Experiments

An experiment is a set of configurations that answer one question. Each has its
own directory here, holding a README that describes it and its configuration
files. The scripts that run the configurations, the runners, are shared by the
experiments and live directly in this directory.

```text
experiments/
├── README.md                                  # this file
├── make_demonstrations.py                     # the runners
├── autonomous_esn.py
├── sweep_esn.py
├── robot_esn.py
├── reservoir_states.py
├── warmup_esn.py
├── demonstrations/                            # one directory per experiment: README.md and configurations
├── single_demonstration_autonomous_reaching/
├── single_demonstration_robot_tracking/
├── multi_demonstration_autonomous_reaching/
├── multi_demonstration_robot_tracking/
├── manual_demonstration_autonomous_reaching/
├── manual_demonstration_robot_tracking/
├── manual_demonstration_v2_autonomous_reaching/
└── manual_demonstration_v2_robot_tracking/
```

| Experiment | What it studies | Runners | Report |
| --- | --- | --- | --- |
| [demonstrations](demonstrations/README.md) | scripted reaching demonstrations of a two-link arm | `make_demonstrations.py` | |
| [single_demonstration_autonomous_reaching](single_demonstration_autonomous_reaching/README.md) | an ESN trained on one demonstration, run on its own (Stage 1) | `autonomous_esn.py`, `reservoir_states.py`, `sweep_esn.py`, `warmup_esn.py` | [003](../reports/003-single-demonstration/README.md) |
| [single_demonstration_robot_tracking](single_demonstration_robot_tracking/README.md) | that ESN as the reference generator of the simulated robot (Stage 2) | `robot_esn.py` | [003](../reports/003-single-demonstration/README.md) |
| [multi_demonstration_autonomous_reaching](multi_demonstration_autonomous_reaching/README.md) | an ESN trained on eight demonstrations, run on its own (Stage 1) | `autonomous_esn.py`, `sweep_esn.py` | [001](../reports/001-autonomous-reaching/README.md) |
| [multi_demonstration_robot_tracking](multi_demonstration_robot_tracking/README.md) | that ESN as the reference generator of the simulated robot (Stage 2) | `robot_esn.py` | [002](../reports/002-esn-reference-on-the-robot/README.md) |
| [manual_demonstration_autonomous_reaching](manual_demonstration_autonomous_reaching/README.md) | an ESN trained on one take taught by hand, run on its own (Stage 1) | `import_demonstrations.py`, `autonomous_esn.py`, `reservoir_states.py`, `sweep_esn.py`, `warmup_esn.py`, `route_convergence.py` | [004](../reports/004-manual-demonstration/README.md) |
| [manual_demonstration_robot_tracking](manual_demonstration_robot_tracking/README.md) | that ESN as the reference generator of the simulated robot (Stage 2) | `robot_esn.py` | [004](../reports/004-manual-demonstration/README.md) |
| [manual_demonstration_v2_autonomous_reaching](manual_demonstration_v2_autonomous_reaching/README.md) | as manual-demonstration autonomous reaching, on a remade arm with its own start posture and target (base configurations) | the same | |
| [manual_demonstration_v2_robot_tracking](manual_demonstration_v2_robot_tracking/README.md) | as manual-demonstration robot tracking, on that arm (base configurations) | `robot_esn.py` | |

Run a configuration with its runner:

```bash
uv run python experiments/<runner>.py experiments/<experiment>/<configuration>.toml
```

The interactive apps that run a trained ESN or the robot live are described in
[`tools/README.md`](../tools/README.md).

## Configuration and reproducibility

Every run is described by a human-readable TOML configuration file. It holds
every parameter that affects the result: the robot, the demonstrations, the ESN
hyperparameters, the tracker gains, the scenario, and the random seeds.

Every runner starts with `start_run` from `arm_esn_ctrl.storage`. It loads the
configuration file and creates a run directory, where the runner then writes its
outputs:

```python
config, run_dir = start_run("experiments/multi_demonstration_robot_tracking/nominal.toml")
```

The run directory records how the result was produced:

- `config.toml`: an exact copy of the configuration file, including the random
  seeds;
- `run.toml`: the configuration's path, the command, start time, host name, commit
  hash of this repository, and whether the working tree had uncommitted changes.

```toml
# run.toml
config = "experiments/multi_demonstration_robot_tracking/nominal.toml"
command = "experiments/robot_esn.py experiments/multi_demonstration_robot_tracking/nominal.toml"
started = 2026-10-05 12:04:54+09:00
host = "workstation"
commit = "2d5b5670dba4c5855b2fa31830d923ed00ce5b65"
uncommitted_changes = false
```

The goal is modest: running the same configuration again on the same machine
reproduces the same result. Small numerical differences on other machines or CPUs
are acceptable and are not chased. For this, `arm_esn_ctrl` limits rclib to one
OpenMP thread (`OMP_NUM_THREADS=1`, unless you set the variable yourself): a
parallel sum is rounded differently depending on how the work is split among
threads. Reproducing a result does not require checking out its recorded commit.
The commit hash is there to investigate why a reproduced result differs, by
showing what the implementation looked like when the result was produced.

## Data and results storage

Demonstrations and run outputs can grow large, so they live in a storage root,
usually outside the Git repository. The storage root is chosen in this order:

1. the `ARM_ESN_CTRL_STORAGE_ROOT` environment variable, if it is set;
2. otherwise, `storage_root` in `storage.toml` at the repository root, if that
   file exists;
3. otherwise, `storage/` in the repository.

```toml
# storage.toml
storage_root = "/path/to/storage"
```

A relative path is resolved against the repository root. Both `storage.toml`
and `storage/` are specific to one machine, so Git ignores them.

Runs are filed by experiment: a run goes into the directory named after the one
that holds its configuration.

```text
<storage root>/
├── data/                                       # demonstrations taught by hand (*.sklog.npz)
└── results/
    ├── demonstrations/
    │   └── 20261002-194644-reach_tvs/          # one directory per run: <date>-<time>-<configuration name>
    │       ├── config.toml
    │       ├── run.toml
    │       └── ...                             # outputs written by the runner
    ├── multi_demonstration_autonomous_reaching/
    └── ...
```

`data/` holds what cannot be regenerated, such as demonstrations taught with the
mouse. Everything a runner produces, including scripted demonstrations, goes into
a run directory under `results/`.

A configuration names the runs it builds on by their paths under the storage
root, such as `run = "results/demonstrations/20261002-194644-reach_tvs"` for its
demonstrations. Run names begin with the time the run started, so they are
unique, and a run is also found by its name alone, such as `results/<run>`,
wherever it is filed. The run records made before the runs were filed by
experiment (2026-10-05) were updated to the paths of this layout then; the
configuration's old path can be traced with `git log --follow`.

Git tracks only the data and results behind a specific report. They are copied
into that report's directory (see [Reports](../README.md#reports)).

## Adding an experiment

1. Make a directory here named after the experiment, with a `README.md` that says
   what it studies, which configurations it has and how to run them, and which
   runs are current.
2. Put its configurations in that directory. Reuse a runner if one fits; add a
   new runner here only for a new kind of run.
3. Add the experiment to the table above.
