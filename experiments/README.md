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
├── sweep_robot_esn.py
├── reservoir_states.py
├── robot_states.py
├── compare_arms.py
├── warmup_esn.py
├── global_starts.py
├── push_pulses.py
├── fixed_points.py
├── screen_esns.py
├── screen_against_robot.py
├── release_robot.py
├── demonstrations/                            # one directory per experiment: README.md and configurations
├── single_demonstration_autonomous_reaching/
├── single_demonstration_robot_tracking/
├── multi_demonstration_autonomous_reaching/
├── multi_demonstration_robot_tracking/
├── manual_demonstration_autonomous_reaching/
├── manual_demonstration_robot_tracking/
├── manual_demonstration_v2_autonomous_reaching/
├── manual_demonstration_v2_robot_tracking/
└── manual_demonstration_v2_robot_search/
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
| [manual_demonstration_v2_autonomous_reaching](manual_demonstration_v2_autonomous_reaching/README.md) | as manual-demonstration autonomous reaching, on a remade arm with its own start posture and target (base configurations) | the same | [005](../reports/005-tuned-on-the-robot/README.md) |
| [manual_demonstration_v2_robot_tracking](manual_demonstration_v2_robot_tracking/README.md) | as manual-demonstration robot tracking, on that arm, and ESN sweeps on the robot | `robot_esn.py`, `sweep_robot_esn.py`, `robot_states.py` | [005](../reports/005-tuned-on-the-robot/README.md) |
| [manual_demonstration_v2_robot_search](manual_demonstration_v2_robot_search/README.md) | a wider ESN search on that robot: spectral radius over 1, input scaling, larger leak rates, ten reservoirs each; then one setting's neighbors, and a small grid combining them, under pushes and a block | `sweep_robot_esn.py`, `autonomous_esn.py`, `robot_esn.py`, `robot_states.py` | [006](../reports/006-wider-search-on-the-robot/README.md) |
| [manual_demonstration_v2_robot_ringing](manual_demonstration_v2_robot_ringing/README.md) | the ringing after a block or a push, with joint PD: lengthened pushes, then an ESN search scored by it | `robot_esn.py`, `sweep_robot_esn.py`, `autonomous_esn.py`, `robot_states.py` | [007](../reports/007-ringing-with-joint-pd/README.md) |
| [manual_demonstration_v2_robot_damping](manual_demonstration_v2_robot_damping/README.md) | report 007's chosen ESN against the replay with joint PD at damping ratios 0.1 to 0.8, with and without the reference velocity | `robot_esn.py` | [008](../reports/008-damping-and-reference-velocity/README.md) |
| [manual_demonstration_v2_robot_damped_search](manual_demonstration_v2_robot_damped_search/README.md) | an ESN search with joint PD at a damping ratio of 0.4, given the reference velocity, its torque computed every 10 ms, then white, pink, and brown teacher noise, scored by the ringing and the return after a block or a push | `sweep_robot_esn.py`, `autonomous_esn.py`, `robot_esn.py`, `robot_states.py` | [009](../reports/009-noisy-training-for-a-faster-return/README.md) |
| [manual_demonstration_v2_robot_noisy_search](manual_demonstration_v2_robot_noisy_search/README.md) | the noise faster than white (blue and violet) against white, then the grid trained with the best noise, white noise of 1 deg, and beyond its corner; no ESN trained with noise did better than report 007's in the app | `sweep_robot_esn.py`, `autonomous_esn.py` | [010](../reports/010-noisy-search-a-negative-result/README.md) |
| [manual_demonstration_v2_robot_following_search](manual_demonstration_v2_robot_following_search/README.md) | report 009's grid without noise, scored by how much the reference follows the arm under the block and the pushes (the following gain), then the two chosen ESNs against the replay with joint PD at damping ratios 0.2, 0.4, and 0.8 | `sweep_robot_esn.py`, `autonomous_esn.py`, `robot_esn.py`, `robot_states.py` | [011](../reports/011-following-the-arm/README.md) |
| [manual_demonstration_v2_robot_recorded_pushes](manual_demonstration_v2_robot_recorded_pushes/README.md) | report 011's two ESNs and the replay under pushes given by hand in the robot app, recorded once and replayed by time, with joint PD at damping ratios 0.2, 0.4, and 0.8 | `robot_esn.py`, `compare_arms.py` | [012](../reports/012-pushes-by-hand/README.md) |
| [manual_demonstration_v2_robot_equal_gains](manual_demonstration_v2_robot_equal_gains/README.md) | every scenario of reports 011 and 012, report 011's two ESNs and the replay, with joint PD at the same gains for both joints, kp = 80 N m/rad and kd = 3 N m s/rad | `robot_esn.py` | [013](../reports/013-equal-gains/README.md) |
| [manual_demonstration_v2_robot_stable_equal_gains](manual_demonstration_v2_robot_stable_equal_gains/README.md) | every scenario of reports 011 and 012, report 011's two ESNs and the replay, with joint PD at equal gains estimated to keep the torque held for 10 ms stable, kp = 80 and kd = 1.2, and kp = 16 and kd = 1.4; and undisturbed at kp = 80 and kd = 1.7, just past the bound | `robot_esn.py` | [013](../reports/013-equal-gains/README.md) |
| [manual_demonstration_v2_robot_progress_replay](manual_demonstration_v2_robot_progress_replay/README.md) | every scenario of reports 011 and 012, report 011's two ESNs, the replay, and a replay that never runs more than 2.5° along the taught path past the arm's progress, with joint PD at ζ = 0.4 and at kp = 80 and kd = 1.2 for both joints | `robot_esn.py` | [014](../reports/014-progress-replay/README.md) |
| [manual_demonstration_v2_esn_dynamics](manual_demonstration_v2_esn_dynamics/README.md) | what report 011's two ESNs learned as dynamical systems: run on their own from rest at start postures all over the joint space, and pushed at the tip by short pulses all along the reach on the robot, against the replay and the progress replay, and read as the map of their reservoir state for its fixed points | `global_starts.py`, `warmup_esn.py`, `push_pulses.py`, `fixed_points.py` | [015](../reports/015-what-the-esns-learned/README.md) |
| [manual_demonstration_v2_esn_screening](manual_demonstration_v2_esn_screening/README.md) | report 011's 1,050 ESNs screened on their own, by where they can rest (the zeros of R(q) over the joint space), how far their output leads a held posture, and the rates along their run, set against report 011's robot outcomes; the passing ESNs held still on the robot at their unstable fixed points and under blocks along the reach; and 1,050 held-out ESNs screened and run on the robot alike | `screen_esns.py`, `screen_against_robot.py`, `release_robot.py`, `sweep_robot_esn.py`, `robot_esn.py` | [016](../reports/016-screening-esns/README.md) |
| [manual_demonstration_v2_consistent_search](manual_demonstration_v2_consistent_search/README.md) | the main results run again under one protocol fixed before any run, which its README holds for reports 017 to 019: an ESN search over a wider grid with one tracker, every ESN undisturbed, under the block and the pushes, and from 49 offset starts, chosen by a rule in code, and checked on held-out seeds | `sweep_robot_esn.py`, `robot_esn.py`, `select_esns.py`, `autonomous_esn.py` | [017](../reports/017-search-under-one-protocol/README.md) |

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
│   └── recorded_forces/                        # runs pushed by hand in tools/robot_app.py, to replay their forces
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
mouse and the forces of runs pushed by hand. Everything a runner produces,
including scripted demonstrations, goes into a run directory under `results/`.

A configuration names the runs it builds on by their paths under the storage
root, such as `run = "results/demonstrations/20261002-194644-reach_tvs"` for its
demonstrations. Run names begin with the time the run started, so they are
unique, and a run is also found by its name alone, such as `results/<run>`,
wherever it is filed. The run records made before the runs were filed by
experiment (2026-10-05) were updated to the paths of this layout then; the
configuration's old path can be traced with `git log --follow`.

Git tracks only the data and results behind a specific report. They are copied
into that report's directory (see [Reports](../README.md#reports)).

## Recorded forces

To compare arms under the same pushes given by hand, push the arm in
`tools/robot_app.py` with one ESN and tracker, and save the run's log (Save log,
`Ctrl+S`), by default under `data/recorded_forces/` in the storage root. Then replay
its force with `experiments/robot_esn.py`, for each ESN and with every tracker:

```toml
[disturbance]
type = "recorded"
log = "data/recorded_forces/<log>.sklog.npz"
# time_offset = 0.0   # s: shifts the recorded force on the task clock
# threshold = 0.5     # N: the force counts as a push above this, for the measures after each push
# min_gap = 0.2       # s: pushes closer than this count as one
```

- **Every arm meets the same force at the same times,** on the task clock, whatever
  it does: the recorded force is replayed, not the drag. With the log's simulation
  step, each step gets the recorded sample itself; with another, the mean recorded
  force over it, which keeps every push's impulse.
- **The runs start from the log's start posture,** so `[evaluation]` lists no start
  postures, and they must last until the recorded pushes end. A log recorded with
  another arm, or a second recorded log, is refused; scripted disturbances may be
  added, with a warning.
- **The run directory keeps a copy of the log,** with its SHA-256 in
  `recorded_forces.toml`.
- **The tracker may differ from the one recorded with:** every run, the recording's
  tracker included, meets the same force, so the runs compare with each other. The
  experiments' run with the recording's ESN and tracker retraces the recording run
  exactly, since the app sets joint PD's gains as the experiments do, where the
  ESN's training takes end.

`experiments/compare_arms.py` then overlays the arms of such runs, such as two ESNs
and the replay: its `[[arm]]` tables name each arm's run, the arm in it (`esn` or
`replay`), a label, and a color. It draws the tip force and each joint's angle
(`joints.png`) and torque (`torques.png`) over time, one column per tracker setting
the runs share, the ESNs' lines over the replay's, and refuses arms that met
different forces.

## Adding an experiment

1. Make a directory here named after the experiment, with a `README.md` that says
   what it studies, which configurations it has and how to run them, and which
   runs are current.
2. Put its configurations in that directory. Reuse a runner if one fits; add a
   new runner here only for a new kind of run.
3. Add the experiment to the table above.
