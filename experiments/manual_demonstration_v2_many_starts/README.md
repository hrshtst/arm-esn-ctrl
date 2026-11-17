# Many starts, v2 (Stages 1 and 2)

Reports 017 to 020 taught one reach from one start posture. This experiment is the
first stage of extending them to more tasks: one target, reached from many start
postures, taught by one take from each, on v2's arm with no joint limits. Its ESNs
have 800 neurons.

**Status.** The ESN's setting is being tuned by hand first, with the configurations
below. The search, its rule, and the claims of this stage will be fixed here before
any run of the search; nothing here is a result yet.

## The robot and the takes

[`reach_manual_v2_no_limits.toml`](../demonstrations/reach_manual_v2_no_limits.toml)
holds the robot, the target, and the nine start postures. The robot is the arm of
reports 005 to 020 (v2: its links and its target) with no joint limits: neither joint
has a range to stay in (the limits are ten turns each way, which nothing comes near).
With v2's limits of ±180°, joint 1's stop is a line across the workspace that the hand
cannot cross, so some hands would have to go around the base the long way; without
it, the hand goes to the target either way around.

All nine takes are recorded with skelarm's recorder, as the configuration's header
says, and imported into one demonstration run:

```bash
uv run python tools/starts_figure.py experiments/demonstrations/reach_manual_v2_no_limits.toml --output starts.png
uv run python experiments/import_demonstrations.py experiments/demonstrations/reach_manual_v2_no_limits.toml
```

`starts_figure.py` draws the start postures in the workspace and in joint space, to
judge them before recording; with `--starts experiments/manual_demonstration_v2_many_starts/hand_esn.toml`
it adds the start postures no take starts from.

## The start postures

The start postures are laid out by the hand, spread evenly over the workspace. Start
postures a step apart in joint space put their hands unevenly: crowded where the arm
is nearly straight or folded, and thin between.

The hand reaches the ring from 0.1 to 0.9 m around the base. It is split into rings
whose areas are as their numbers of hands, so that every hand stands for the same
area; a ring's hands lie on the circle that halves its area, equally spaced, between
the hands of the ring outside it. [`tools/tip_starts.py`](../../tools/tip_starts.py)
turns them into postures, rounded to whole degrees, which moves a hand less than
1 cm:

| Starts | Rings (hands, distance from the base) | Command |
| --- | --- | --- |
| the nine takes' | 6 at 0.74 m, 3 at 0.38 m | `tools/tip_starts.py <takes> --counts 6 3 --first-angle 43.3` |
| 36 untaught | 18 at 0.78 m, 12 at 0.53 m, 6 at 0.28 m | `tools/tip_starts.py <takes> --counts 18 12 6 --first-angle 53.3` |

- **The first hand** is in the direction of the start of reports 005 to 020's take,
  43.3° from the x axis.
- **The untaught hands** are turned 10° from the takes' rings: none is nearer than
  0.10 m to a take's start or to the target.
- **The elbow.** A hand position has two postures. Every start, taught or not, has
  joint 2 positive, the elbow on the side of reports 005 to 020's take.
- **Two rest postures.** Without limits, a take that goes around the base clockwise
  ends with joint 1 a whole turn below one that goes counterclockwise: at −246°
  instead of 114°, the hand at the same place. Dragged straight, the hands of takes
  03 and 04 go clockwise; that of take 08 would pass 0.06 m from the base, inside
  the ring the hand cannot enter, so it goes around the base on either side. The
  ESN reads joint angles, so it is taught two rest postures for the one target.

## Tuning an ESN by hand

1. **Name the import's run** in `hand_esn.toml` (`[demonstrations]` `run`), once.
2. **Edit `[esn]`** in `hand_esn.toml`: the leak rate, the spectral radius, the input
   scaling, the ridge, the seed, and the rest. Leave takes out of `train` to train
   on fewer.
3. **Train**, about half a minute:

   ```bash
   uv run python experiments/autonomous_esn.py experiments/manual_demonstration_v2_many_starts/hand_esn.toml
   ```

   The run trains the ESN and runs it on its own from each take's start and from the
   36 start postures no take starts from. It prints how many arrive and stay, for
   the takes' starts and the untaught ones apart, writes each run's measures to
   `metrics.csv` (among them `first_step_m`, `detour_ratio`, and `off_line_m`), and
   draws `autonomous.png`.
4. **Watch it live.** The run ends by printing the two commands, with its own
   `esn.toml`:

   ```bash
   uv run python tools/esn_reference_app.py experiments/demonstrations/reach_manual_v2_no_limits.toml --model <run directory>/esn.toml
   uv run python tools/robot_app.py experiments/demonstrations/reach_manual_v2_no_limits.toml \
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
| `hand_esn.toml` | an ESN trained on the nine takes, run on its own from their starts and from 36 hand positions no take starts from |
| `hand_robot.toml` | that ESN on the robot, undisturbed, from the same starts, against the replay of the take that starts nearest (90 runs, about 8 minutes) |
| `hand_pushes_robot.toml` | that ESN on the robot under report 007's three pushes, the first two placed along each take, from each take's start, against the replays of that take (about 2 minutes) |
