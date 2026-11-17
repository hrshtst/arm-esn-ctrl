# Many starts, v2 (Stages 1 and 2)

Reports 017 to 020 taught one reach from one start posture. This experiment is the
first stage of extending them to more tasks: one target, reached from many start
postures, taught by one take from each. Its ESNs have 800 neurons.

**Status.** The ESN's setting is being tuned by hand first, with the configurations
below. The search, its rule, and the claims of this stage will be fixed here before
any run of the search; nothing here is a result yet.

## The takes

[`reach_manual_v2_starts.toml`](../demonstrations/reach_manual_v2_starts.toml) holds the
arm and the target of reports 005 to 020 and nine start postures on a lattice in
joint space: joint 1 at −60°, 30°, and 120°, joint 2 at 30°, 90°, and 150°. The take
from (30°, 30°) is the one of reports 005 to 020; the other eight are recorded with
skelarm's recorder, as its header says, and all nine are imported into one
demonstration run:

```bash
uv run python tools/starts_figure.py experiments/demonstrations/reach_manual_v2_starts.toml --output starts.png
uv run python experiments/import_demonstrations.py experiments/demonstrations/reach_manual_v2_starts.toml
```

`starts_figure.py` draws the start postures in the workspace and in joint space, to
judge them before recording; with `--starts experiments/manual_demonstration_v2_many_starts/hand_esn.toml`
it adds the start postures no take starts from.

## Tuning an ESN by hand

1. **Name the import's run** in `hand_esn.toml` (`[demonstrations]` `run`), once.
2. **Edit `[esn]`** in `hand_esn.toml`: the leak rate, the spectral radius, the input
   scaling, the ridge, the seed, and the rest. Leave takes out of `train` to train
   on fewer.
3. **Train**, about half a minute:

   ```bash
   uv run python experiments/autonomous_esn.py experiments/manual_demonstration_v2_many_starts/hand_esn.toml
   ```

   The run trains the ESN and runs it on its own from each take's start and from 36
   start postures no take starts from. It prints how many arrive and stay, writes
   each run's measures to `metrics.csv` (among them `first_step_m`, `detour_ratio`,
   and `off_line_m`), and draws `autonomous.png`.
4. **Watch it live.** The run ends by printing the two commands, with its own
   `esn.toml`:

   ```bash
   uv run python tools/esn_reference_app.py experiments/demonstrations/reach_manual_v2_starts.toml --model <run directory>/esn.toml
   uv run python tools/robot_app.py experiments/demonstrations/reach_manual_v2_starts.toml \
       --law pd --omega 20 --damping 0.4 --control-period 0.01 \
       --gains-from reports/005-tuned-on-the-robot/data/20261008-170034-reach_manual_v2/demo_00_filtered.sklog.npz \
       --model <run directory>/esn.toml
   ```

   The first runs the ESN on its own; the second simulates the robot with the
   consistent search's tracker, to pose and to push by hand (see
   [`tools/README.md`](../../tools/README.md)). `--gains-from` sets joint PD's gains
   where the take of reports 005 to 020 ends (kp = 74.96 and 8.96 N m/rad, kd = 3.00
   and 0.358 N m s/rad), whatever takes the ESN was trained on, so that every ESN is
   tracked alike. `[apps]` in `hand_esn.toml` holds these arguments.
5. **On the robot without the app,** name the ESN's run in `hand_robot.toml` or
   `hand_pushes_robot.toml` (`[esn]` `model`) and run it:

   ```bash
   uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_many_starts/hand_robot.toml
   ```

Each training is its own run directory, so earlier ESNs stay; to keep a setting, copy
`hand_esn.toml` under a name of its own.

| Configurations | What they run |
| --- | --- |
| `hand_esn.toml` | an ESN trained on the nine takes, run on its own from their starts and from 36 start postures no take starts from |
| `hand_robot.toml` | that ESN on the robot, undisturbed, from the same starts, against the replay of the take that starts nearest (about 8 minutes) |
| `hand_pushes_robot.toml` | that ESN on the robot under report 007's three pushes, the first two placed along each take, from each take's start, against the replays of that take (about 2 minutes) |
