# Many starts, v2 (Stages 1 and 2)

Reports 017 to 020 taught one reach from one start posture. This experiment is the
first stage of extending them to more tasks: one target, reached from many start
postures, taught by one take from each, on v2's arm with no joint limits. Its ESNs
have 800 neurons.

**Status.** The ESN's setting was tuned by hand first, with the configurations
below. The search's rule, [below](#the-search), was fixed on 2026-11-17 before any
run of the search. The claims of this stage and the runs that test them,
[below](#the-analysis), were fixed on 2026-11-18 before any of those runs.

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

## The search

The search chooses two ESNs by how reliably the hand arrives and holds from the
takes' starts, and then by how directly it goes there. Its rule is given as it was
fixed; a step added afterwards, [below](#scoring-again-with-the-untaught-starts),
chose ESN #1 and ESN #2.

1. **The runs.** Each ESN is trained on the nine takes and generates the reference
   of the simulated robot, undisturbed, from each take's start: nine runs of 24 s,
   with the consistent search's tracker (report 017's), as `hand_robot.toml`'s.
2. **A run holds** if the hand arrives within 20 mm of the target and stays within
   it for the 2 s after it arrives. Nothing later is required: a hand that leaves
   after those 2 s still holds.
3. **The detour ratio** of a run that arrives is the length of the hand's path until
   it arrives, over the straight distance from its start to the goal: 1 for a
   straight reach, never less.
4. **A setting's score** is how many of its runs hold, the most first; then the
   median detour ratio of its runs that arrive, the smallest first.
5. **The coarse search:** 108 settings with one reservoir each (seed 0):

   | Parameter | Values |
   | --- | --- |
   | leak rate | 0.01, 0.03, 0.1, 0.3 |
   | spectral radius | 0.9, 1.2, 1.5 |
   | input scaling | 0.1, 0.5, 2 |
   | ridge | 10⁻⁴, 10⁻², 1 |

   The rest is `coarse_search.toml`'s `[esn]`: 800 neurons, sparsity 0.05, a bias,
   a warm-up of 1 s, 10 ms.
6. **The center** is the coarse setting with the best score.
7. **The fine search:** 81 settings around the center, with five reservoirs each
   (seeds 1 to 5). Each parameter takes three values: the center's, and the one
   halfway to its coarse neighbor on each side. Halfway is in ratio for the leak
   rate, the input scaling, and the ridge, whose coarse values step in ratio (rounded
   to two digits), and in difference for the spectral radius. Where the center is at
   an edge of the coarse grid, the value beyond the edge is as far from the center
   as the one on the other side. A setting's runs are the 45 of its five seeds.
8. **The chosen ESNs** are the typical seeds of the two fine settings with the
   best scores. A setting's typical seed is, of its seeds with the most runs that
   hold, the one whose own median detour ratio is nearest the setting's.

The search leaves out the start postures no take starts from, pushes, held-out
seeds, and an ESN trained on one take; the one-take search waits until a good ESN of
the nine takes is found. Before the rule was fixed, the 36 coarse settings with a
ridge of 10⁻² were run once to measure the search's time; that run was not kept.

```bash
uv run python experiments/sweep_robot_esn.py \
    experiments/manual_demonstration_v2_many_starts/coarse_search.toml --workers 30
uv run python experiments/sweep_robot_esn.py \
    experiments/manual_demonstration_v2_many_starts/fine_search.toml --workers 30
```

Each run writes every setting's score to `scores.csv`, the best first, and every
run's measures to `runs.csv`.

**The center.** Of the coarse search's 108 settings, 30 hold from all nine starts;
the smallest median detour ratio among them, 1.0096, is that of leak rate 0.1,
spectral radius 1.2, input scaling 0.5, and ridge 1. The fine search's values around
it:

| Parameter | Values |
| --- | --- |
| leak rate | 0.055, 0.1, 0.17 |
| spectral radius | 1.05, 1.2, 1.35 |
| input scaling | 0.22, 0.5, 1 |
| ridge | 0.1, 1, 10 |

The center's ridge is the coarse grid's largest, so the fine search goes as far
above it as below.

**The first pair.** Of the fine search's 81 settings, 20 hold in all 45 runs of
their five seeds. The two with the smallest median detour ratios, and their typical
seeds, were the ESNs the rule chose. The step below replaced them, so they are named
by the starts they were scored from:

| | Leak rate | Spectral radius | Input scaling | Ridge | Seed | Runs that hold | Median detour ratio of the setting |
| --- | --- | --- | --- | --- | --- | --- | --- |
| first | 0.1 | 1.2 | 1 | 1 | 2 | 45 of 45 | 1.0068 |
| second | 0.055 | 1.2 | 1 | 0.1 | 4 | 45 of 45 | 1.0082 |

Their configurations train them, run them on their own from the takes' starts and
from the 36 start postures no take starts from, and save `esn.toml` for the apps, as
`hand_esn.toml` does:

```bash
uv run python experiments/autonomous_esn.py \
    experiments/manual_demonstration_v2_many_starts/taught1_lr0.1_sr1.2_is1_ridge1_seed2.toml
uv run python experiments/autonomous_esn.py \
    experiments/manual_demonstration_v2_many_starts/taught2_lr0.055_sr1.2_is1_ridge0.1_seed4.toml
```

### Scoring again with the untaught starts

On their own from the 36 start postures no take starts from, which the search did
not see, the first pair differ: the first arrives and stays from 27 of them and ends
1.61 m from the target from six; the second arrives and stays from 35, but its
median detour ratio there is 1.19 and four of its runs exceed 2. The scores from the
takes' starts did not tell the two apart. So a step was added on 2026-11-17, after
those results were seen and before any run of it:

1. **The settings:** the fine search's 20 whose 45 runs all hold, each with its
   five reservoirs again (seeds 1 to 5).
2. **The runs:** on the robot, undisturbed, as in the search, from each take's
   start and from the 36 untaught ones of `hand_esn.toml`: 45 runs an ESN, 225 a
   setting.
3. **The score** is the search's, over all 225 runs: how many hold, the most first,
   then the median detour ratio of the runs that arrive, the smallest first. A
   setting's typical seed is chosen as in the search.

```bash
uv run python experiments/sweep_robot_esn.py \
    experiments/manual_demonstration_v2_many_starts/untaught_search.toml --workers 30
```

**ESN #1 and ESN #2.** No run from a take's start fails. Three of the 20 settings
hold in all 225 runs; the first pair's settings hold in 215 and 203, and the worst
setting in 168. The typical seeds of the two best are ESN #1 and ESN #2, in place of
the first pair (decided on 2026-11-18):

| ESN | Leak rate | Spectral radius | Input scaling | Ridge | Seed | Runs that hold | Median detour ratio of the setting |
| --- | --- | --- | --- | --- | --- | --- | --- |
| #1 | 0.055 | 1.2 | 1 | 1 | 4 | 225 of 225 | 1.054 |
| #2 | 0.1 | 1.05 | 0.5 | 10 | 3 | 225 of 225 | 1.063 |

Their configurations train them and run them on their own, as the first pair's do:

```bash
uv run python experiments/autonomous_esn.py \
    experiments/manual_demonstration_v2_many_starts/esn1_lr0.055_sr1.2_is1_ridge1_seed4.toml
uv run python experiments/autonomous_esn.py \
    experiments/manual_demonstration_v2_many_starts/esn2_lr0.1_sr1.05_is0.5_ridge10_seed3.toml
```

## The analysis

ESN #1 and ESN #2 are tested on the robot with the search's tracker, against the
replay of a take by time and against ESNs trained on fewer takes. No size is fixed in
advance: every claim is a count or a direction, and the sizes are reported. A claim
holds if it holds for ESN #1 and for ESN #2.

**The measures.** A run holds as in the search: the hand arrives within 20 mm of the
target and stays within it for 2 s. Its detour ratio is the search's. How a run
starts is measured by the reference's fastest joint speed and by the run's largest
joint torque: a reference that begins away from the arm jumps to it, and the tracker
kicks.

**The unseen starts** are 36 more hand positions, on the rings of the 36 untaught
ones turned 15° back (`tools/tip_starts.py <takes> --counts 18 12 6 --first-angle
38.3`), joint 2 positive: none is nearer than 0.09 m to a take's start, 0.07 m to an
untaught start of the scoring, or 0.12 m to the target. No search or scoring ran
from them.

**The one-take baseline** is trained on take 00 alone, whose hand starts in the
direction of the start of reports 005 to 020's take, and chosen as ESN #1 and ESN #2
were: the coarse search (seed 0), the fine search around its best (seeds 1 to 5),
and the fine settings whose every run holds scored again with the 36 untaught starts
of `hand_esn.toml`. The typical seeds of its two best settings are one-take ESN #1
and one-take ESN #2.

### The claims

1. **The takes' starts.** Undisturbed, from each of the nine takes' starts, the hand
   arrives and holds, as the replay's does.
2. **The unseen starts.** From the 36 unseen starts, the hand arrives and holds from
   as many as the replay of the take that starts nearest.
3. **Starting where the arm is.** From the 36 unseen starts, the reference's fastest
   joint speed and the largest joint torque are smaller than the replay's, in their
   medians over the starts.
4. **Going directly.** From the 36 unseen starts, the median detour ratio is smaller
   than the replay's.
5. **Pushes.** Under report 007's three pushes at their forces, at twice, and at four
   times them, from each take's start, the hand arrives and ends within 20 mm of
   the target in at least as many of the nine runs as the replay's and as the
   progress replay's, at each size. Leaving the goal during the hold is excused, as
   in report 017: a push can carry the arm into the goal early.
6. **Nine takes against one.** From the 44 starts take 00 does not start from (the
   other eight takes' and the 36 unseen), ESN #1 and ESN #2 each hold from more
   starts than one-take ESN #1 and than one-take ESN #2, with a smaller median detour
   ratio. The eight and the 36 are also reported apart, with the reference's fastest
   joint speed and the largest torque.
7. **More takes.** At ESN #1's setting and at ESN #2's, over five reservoirs (seeds
   1 to 5), the runs that hold from the 36 unseen starts are more with nine takes
   than with one, and do not decrease from one take to three, to six, to nine.

The plan had sets of three and five takes, proposed for start postures a step apart
in joint space. With the hands on rings, the sets are one, three, six, and nine, each
holding the one before: take 00; takes 00, 02, and 04, every other hand of the outer
ring; the outer ring; and all nine.

### The runs

| Claims | Configurations | What they run |
| --- | --- | --- |
| 1 to 4 | `robot_esn1.toml`, `robot_esn2.toml` | ESN #1 and ESN #2 against the replay, undisturbed, from the nine takes' starts and the 36 unseen ones (`robot_esn.py`; 90 runs, about 8 minutes each) |
| 5 | `pushes1x_robot_esn1.toml`, `pushes2x_…`, `pushes4x_…`, and those of ESN #2 | ESN #1 and ESN #2 against the replay and the progress replay under the three pushes at 1, 2, and 4 times their forces, from each take's start (`robot_esn.py`; 27 runs, about 2 minutes each) |
| 6 | `one_take_coarse_search.toml`, `one_take_fine_search.toml`, `one_take_untaught_search.toml`, the two chosen ESNs' configurations, and `robot_one_take_esn1.toml`, `robot_one_take_esn2.toml`, written as each step gave the next | the one-take baseline's search, and one-take ESN #1 and ESN #2 against the replay of take 00, undisturbed, from take 00's start, the other eight takes' starts, and the 36 unseen ones (`robot_esn.py`; 90 runs, about 8 minutes each) |
| 7 | `takes1.toml`, `takes3.toml`, `takes6.toml`, `takes9.toml` | ESN #1's and ESN #2's settings, seeds 1 to 5, trained on one, three, six, and nine takes, each ESN on the robot, undisturbed, from every take's start and the 36 unseen ones (`sweep_robot_esn.py`; about 4 minutes each with 30 workers) |

The pushes are `hand_pushes_robot.toml`'s: the first two placed along each take,
where its hand is 18% and 62% of the way to the target, the third at 15 s.

| Configurations | What they run |
| --- | --- |
| `coarse_search.toml` | the coarse search: 108 settings, seed 0, each on the robot from each take's start (about 7 minutes with 30 workers) |
| `fine_search.toml` | the fine search: 81 settings around the coarse search's best, seeds 1 to 5, alike (about 25 minutes) |
| `untaught_search.toml` | the fine search's 20 settings that hold in every run, seeds 1 to 5, each on the robot from each take's start and from the 36 untaught ones (about 30 minutes) |
| `esn1_lr0.055_sr1.2_is1_ridge1_seed4.toml`, `esn2_lr0.1_sr1.05_is0.5_ridge10_seed3.toml` | ESN #1 and ESN #2, the typical seeds of the two best settings of `untaught_search.toml`, trained and run on their own as `hand_esn.toml`'s |
| `taught1_lr0.1_sr1.2_is1_ridge1_seed2.toml`, `taught2_lr0.055_sr1.2_is1_ridge0.1_seed4.toml` | the first pair, the typical seeds of the two best settings of `fine_search.toml`, trained and run on their own alike |
| `one_take_coarse_search.toml`, `one_take_fine_search.toml`, `one_take_untaught_search.toml` | the one-take baseline's search, as the three above, trained on take 00 alone and run from its start (and from the 36 untaught ones in the last): 108 settings, 81, and the 70 of those that hold in every run (about 2, 3, and 70 minutes) |
| `one_take_esn1_lr0.017_sr1.05_is0.045_ridge0.1_seed5.toml`, `one_take_esn2_lr0.055_sr0.75_is0.22_ridge1_seed3.toml` | one-take ESN #1 and ESN #2, the typical seeds of the two best settings of `one_take_untaught_search.toml`, trained on take 00 and run on their own from its start and the 36 untaught ones |
| `hand_esn.toml` | an ESN trained on the nine takes, run on its own from their starts and from 36 hand positions no take starts from |
| `hand_robot.toml` | that ESN on the robot, undisturbed, from the same starts, against the replay of the take that starts nearest (90 runs, about 8 minutes) |
| `hand_pushes_robot.toml` | that ESN on the robot under report 007's three pushes, the first two placed along each take, from each take's start, against the replays of that take (about 2 minutes) |
