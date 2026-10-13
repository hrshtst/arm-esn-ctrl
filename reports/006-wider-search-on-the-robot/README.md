# 006 A wider search on the robot, under disturbances

**Summary.** Report 005's search was widened, with ten random reservoirs (seeds) for
every setting, and then narrowed around a setting chosen by hand for how it yields,
with the pushes and the block of report 005's validation run for every setting and
seed.

- **A spectral radius over 1 helps** at leak rate 0.03: spectral radius 1.5 beats
  0.99 for 8 to 10 of the 10 seeds at every input scaling. **Input scaling 1.5 does
  as well as 2,** and **faster leak rates do worse:** none of 0.07 to 0.3 does as well
  as 0.03, and none holds at 0.3.
- **Seven candidates were tried in the apps;** candidate #2 (leak rate 0.03,
  spectral radius 1.35, input scaling 2) felt the most balanced against
  disturbances. In report 005's scenarios, the candidates at leak rate 0.03 arrive
  from all 49 offset starts, where report 005's ESN arrived from 30, but one, with
  spectral radius 1.5, settles 23 to 27 mm from the target after the push at the
  target, outside the goal.
- **Around #2, every setting holds undisturbed and under the block;** the pushes
  separate them. Spectral radius 1.2 to 1.75, leak rate 0.015 to 0.03, and input
  scaling 1.5 to 2 lose no seed; at the edges (spectral radius 0.99 and 1.1, leak
  rate 0.04 and 0.05, input scaling 2.25), some seeds lose the target after the
  push at the target, their hand mostly sliding back along the taught path toward
  the start. A slower leak rate or a larger spectral radius keeps the target, but
  lets the reference run further ahead of a held arm.
- **The chosen ESN** (leak rate 0.03, spectral radius 1.35, input scaling 1.5, seed
  8) is the typical seed of the one setting that, by a rule fixed before a small
  grid around #2, came back onto the path after the push at the target sooner than
  #2's setting: for 8 of the 10 seeds, by 0.41 s in the median. In the apps, it felt
  no different from its neighbors.
- **Against the replay of the take,** it follows the taught path within 0.8 to
  0.9 mm undisturbed (the replay 0.1 to 0.9 mm) and arrives with the take; it
  arrives and holds from all 49 offset starts, as the replay does; pushed, it gives
  way, carried 1.2 to 2.5 times as far off the path as the replay, and is back on
  it within 1.7 to 3.4 s after every push, at the target too. Held in mid-reach, it
  waits only in part: its reference is 36 mm ahead of the held hand at the release
  (the replay's 155 mm), so it presses with 3 to 5 N (the replay 18 to 20 N), and
  then lurches 44 to 51 mm off the path, as far as the replay's arm.
- **Its reservoir states,** recomputed from the runs' logs, follow the take's
  undisturbed, hold still while the tip is held, swing off and back after each
  push, and, from both offset starts shown, join the take's course, ahead of it or
  behind.

## 1. Question

Report 005 kept the spectral radius below 1, chose input scaling 2 with one random
reservoir, found the spread over reservoirs as large as the differences between
settings, and met the disturbances only after choosing its ESN. Here:

1. **A spectral radius over 1:** does it do better?
2. **Input scaling:** was 2 needed, or does 1.5 or 1 do as well?
3. **Leak rates above 0.05:** how do they do?
4. **Disturbances in the search:** with the pushes and the block run for every
   setting and seed, which settings keep the hand on the path and at the target,
   and how does an ESN chosen that way compare with the replay of the take?

## 2. Setup

### The arm, the take, the ESN, and the tracker

As in report 005 ([Section 2](../005-tuned-on-the-robot/README.md#2-setup)): the
remade arm, the take filtered at 2 Hz, an ESN of 400 neurons with sparsity 0.05,
ridge 1e-2, and a 1 s warm-up, driven by the arm's measured joint angles, and the
tracker, computed torque and joint PD at ω = 20 rad/s with damping ratio 0.1, given
a zero reference velocity. Every robot run lasts 22 s, from the demonstrated start
unless stated.

Every setting is tried with ten seeds, 0 to 9. A seed draws the reservoir, its
input weights, and its biases, which the spectral radius and the input scaling only
rescale, so the settings compare seed by seed (`arm_esn_ctrl.sweeps`).

### Measures

- **Path RMSE** (`taught_path_rmse_m`): as in report 005, the RMS distance between
  the robot's hand path and the taught one, regardless of timing.
- **Holds:** the hand arrives within the goal (20 mm) and stays there for the 2 s
  hold window, with both trackers; a setting also needs every seed's hand still in
  the goal at the end of the run.
- **Under a disturbance** (`arm_esn_ctrl.metrics.disturbance_returns`): the
  reference's lead over the arm along the taught path at the disturbance's end
  (`reference_lead`, as a fraction of the path, 1.26 m long); the farthest the hand
  strays from the taught path (`off_path_m`), and when it is back within 5 mm of it
  (`return_time_s`, counted from the disturbance's end), each before the next
  disturbance starts or the run ends.
- **A lost seed:** a pushed run that never arrives, or ends more than 20 mm from the
  target.

### The search

| Step | What | Settings |
| --- | --- | --- |
| Stage 1 | leak rate 0.03–0.3 × spectral radius 0.9–1.5 × input scaling 1–2.5, undisturbed | 196, ten seeds each |
| Candidates | seven settings of stage 1, each with its typical seed, tried in the apps | 7 ESNs |
| Step A | four of them in report 005's scenarios | 4 ESNs |
| Step B | a star around candidate #2: spectral radius 0.99–1.75, leak rate 0.015–0.05, input scaling 1.5–2.25, one at a time; undisturbed, pushed, and blocked | 16, ten seeds each |
| Representatives | eight seeds of the star, tried in the apps | 8 ESNs |
| Small grid | leak rate 0.02–0.03 × input scaling 1.5–2 × spectral radius 1.2–1.35; undisturbed, pushed, and blocked | 12, ten seeds each |
| Step C | the chosen ESN in report 005's scenarios, against the replay | 1 ESN |

A setting's **typical seed** is the one nearest its medians over its ten seeds: in
stage 1, of the worst path RMSE; from Step B on, of the undisturbed path RMSE, the
returns after the first push and after the push at the target, and the lead and
the farthest from the path under the block, each scaled by its interquartile range.

### Scenarios

Report 005's ([Section 2](../005-tuned-on-the-robot/README.md#scenarios)):
undisturbed (nominal); the tip held by a stiff spring-damper from 4.5 s to 5.5 s,
in mid-reach (block); three pushes of 0.2 s at the tip, 4 N nearly backward at
4 s, 5 N nearly across at 8 s, and 4 N nearly across the other way at 13 s, after
the arrival and its hold (pushes); and the demonstrated start offset by −15° to 15°
in each joint, every 5° (offsets, 49 starts).

### Reproducing the results

The runs read the take as `results/demonstrations/20261008-170034-reach_manual_v2`
under the storage root ([report 005's copy](../005-tuned-on-the-robot/data)), and
the robot runs read the trained ESNs (`esn.toml` and `esn.rclib`, copied into their
runs' directories here) by their run names. A run is found by its name anywhere
under `results/`; place or link the copies there to rerun. All configurations are
in [`manual_demonstration_v2_robot_search`](../../experiments/manual_demonstration_v2_robot_search/README.md).

| Runs | Command | Commit |
| --- | --- | --- |
| `sweep_robot_wide_lr<leak rate>_filtered` (7) | `uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_search/<name>.toml` | `3d97b68` |
| `grid_<candidate>_filtered` (7) | `uv run python experiments/autonomous_esn.py experiments/manual_demonstration_v2_robot_search/<name>.toml` | `8189fb9` |
| `<scenario>_robot_<candidate>_filtered` (16) | `uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_search/<name>.toml` | `00c3424` |
| `sweep_robot_star_<line>_<scenario>_filtered` (9) | as stage 1 | `7c85003` (undisturbed), `d0b6c1d` |
| `grid_<representative>_filtered` (8) | as the candidates | `731897e` |
| `sweep_robot_small_grid_sr<spectral radius>_<scenario>_filtered` (6) | as stage 1 | `0526792` |
| `<scenario>_robot_lr0.03_sr1.35_is1.5_seed8_filtered` (4) | as Step A | `aa053ea` |
| `states_robot_lr0.03_sr1.35_is1.5_seed8_filtered` (1) | `uv run python experiments/robot_states.py experiments/manual_demonstration_v2_robot_search/<name>.toml` | `09b7793` |
| [`summary`](results/summary) | `uv run python reports/006-wider-search-on-the-robot/make_figures.py --logs --animations` | |

A `<candidate>` or `<representative>` is named by its setting and seed, such as
`lr0.03_sr1.35_is2_seed5` (candidate #2). All runs are deterministic, and every run
record (`run.toml`) shows no uncommitted changes. Each run directory here holds its
configuration, its run record, and its metrics (`metrics.csv`, or `sweep.csv`,
`runs.csv`, and `seeds.csv`); the grid runs also hold their map of the 49 starts
(`grid.png`), and those of the four candidates of Step A and of the chosen ESN its
trained ESN; the offsets runs hold their map, the chosen ESN's also its hand paths
(`paths.png`), and the states run only its figures.
[`make_figures.py`](make_figures.py) draws `stage1.png` and `star.png` and prints the
tables of Section 3 from those copies. With `--logs`, it reads the runs' logs under
the storage root, which are not kept in Git, to draw the chosen ESN's joint angles
and torques (`<scenario>_joints.png`, `<scenario>_torques.png`, with `robot_esn.py`'s
own functions) and the arms' distance from the taught path under the pushes
(`flexibility.png`, `flexibility.csv`), and to print the forces under the block.
With `--animations`, it exports `nominal.gif`, `block.gif`, `pushes.gif`,
`offset.gif`, and `offset_straight.gif`, each arm rendered by skelarm's player. With
`--videos`, it exports the runs of those five animations, for both arms and both
trackers, as 20 MP4s for slides into
`<storage root>/reports/006-wider-search-on-the-robot/videos/`, explained in a
`README.md` there; they are not kept in Git.

## 3. Results

### 3.1 Stage 1: the wide grid

![Stage 1](results/summary/stage1.png)

The map gives each setting's median over its ten seeds; a cross marks a setting
with a seed that fails, or whose hand has left the goal by the end of the run.

| Leak rate | Settings that hold (of 28) | Best setting | Median | Mean ± SD | Range over seeds | Arrival |
| --- | --- | --- | --- | --- | --- | --- |
| 0.03 | 28 | spectral radius 1.5, input scaling 1.5 | 0.81 mm | 0.82 ± 0.21 mm | 0.61–1.27 mm | 9.65 s |
| 0.05 | 28 | 1.5, 1.5 | 0.93 mm | 1.30 ± 0.68 mm | 0.74–2.55 mm | 9.75 s |
| 0.07 | 23 (4 more leave the goal later) | 1.5, 1.5 | 1.02 mm | 1.30 ± 0.62 mm | 0.72–2.69 mm | 9.91 s |
| 0.1 | 19 (2 more leave the goal later) | 1.2, 1.5 | 1.29 mm | 1.55 ± 0.61 mm | 1.03–3.06 mm | 10.51 s |
| 0.15 | 18 | 1.1, 1 | 1.73 mm | 1.94 ± 0.47 mm | 1.51–2.98 mm | 11.57 s |
| 0.2 | 7 | 1.05, 1 | 2.22 mm | 2.55 ± 0.93 mm | 1.54–4.10 mm | 13.39 s |
| 0.3 | 0 | | | | | |

The worst path RMSE of the two trackers; the arrival is the median of each seed's
latest. The settings at leak rate 0.03 and 0.05 with spectral radius 0.99 and input
scaling 2 repeat report 005's seed check (its stage 4) exactly.

- **A spectral radius over 1 helps at leak rate 0.03.** At every input scaling
  there, the best spectral radius over 1, 1.5, beats 0.99 in the median (0.81 to
  0.92 mm against 1.07 to 1.36 mm) and seed by seed for 8 to 10 of the 10 seeds. At
  leak rate 0.05 it does so only with input scaling 1 (10 of 10), elsewhere for 7
  of 10; at 0.07, with input scaling 1.5 (8 of 10).
- **Input scaling 2 was not needed.** At leak rate 0.03, it beats 1.5 for 5 or 6 of
  the 10 seeds, by 0.03 mm or less in the median, at every spectral radius from 1.1
  to 1.5.
- **Faster leak rates do worse,** and fewer of their settings hold; at 0.07 and 0.1,
  some settings whose seeds all hold leave the goal later in the run. They also
  arrive later: 10.5 s at 0.1, 13.4 s at 0.2, against the take's 9.6 s.
- **The best settings** all have leak rate 0.03 and spectral radius 1.2 to 1.5:
  0.81 to 0.92 mm in the median. Spectral radius 1.35 with input scaling 2 is the
  steadiest over its seeds (SD 0.12 mm).

### 3.2 Candidates in the apps

Seven settings of stage 1, each with its typical seed, were trained by
`autonomous_esn.py` and tried in the ESN app and the robot app:

| # | Leak rate | Spectral radius | Input scaling | Seed | Why | Worst path RMSE, this seed / its setting's median | Starts that arrive and hold, on its own |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.03 | 1.5 | 1.5 | 2 | stage 1's best setting | 0.80 / 0.81 mm | 49 / 49 |
| 2 | 0.03 | 1.35 | 2 | 5 | the setting steadiest over its seeds | 0.85 / 0.86 mm | 49 / 49 |
| 3 | 0.03 | 1.5 | 1 | 7 | the weakest input among the best settings | 0.93 / 0.91 mm | 49 / 49 |
| 4 | 0.05 | 1.5 | 1.5 | 3 | report 005's leak rate, with a spectral radius over 1 | 0.97 / 0.93 mm | 49 / 49 |
| 5 | 0.07 | 1.5 | 1.5 | 3 | a faster leak rate | 0.94 / 1.02 mm | 49 / 49 |
| 6 | 0.1 | 1.2 | 1.5 | 4 | a faster leak rate still | 1.33 / 1.29 mm | 28 / 49 |
| 7 | 0.03 | 0.99 | 1.5 | 2 | #1's reservoir with spectral radius 0.99 | 1.02 / 1.14 mm | 49 / 49 |

Pushing the arm by hand in the robot app, **#2 felt the most balanced against
disturbances.** #1, #2, #6 (a fast leak rate), and #7 (a spectral radius below 1)
were kept for Step A, and the search was then narrowed around #2's setting alone.

### 3.3 Step A: the kept candidates in report 005's scenarios

| # | Undisturbed path RMSE | Block: reference lead at the release | Pushes: distance from the target at the end | Offset starts that arrive and hold |
| --- | --- | --- | --- | --- |
| 1 | 0.7 / 0.8 mm | 3.5 / 3.5% | **27 / 23 mm** | 49 / 49 |
| 2 | 0.8 / 0.9 mm | 2.8 / 2.8% | 3.1 / 3.2 mm | 49 / 49 |
| 6 | 1.2 / 1.3 mm | 0.7 / 0.7% | **1060 / 39 mm** | 28 / 26 |
| 7 | 0.9 / 1.0 mm | 2.3 / 2.3% | 2.8 / 2.8 mm | 49 / 49 |

Computed torque / joint PD. Report 005's ESN: 0.9 / 1.0 mm, a lead of 1.1%
(14 mm), 2.5 mm after the pushes, and 30 / 30 offset starts.

- **At leak rate 0.03, the ESN arrives from every offset start,** with either
  tracker.
- **The push at the target separates them.** After it, #1 (spectral radius 1.5)
  settles 23 to 27 mm from the target, outside the goal, though it arrived and held;
  #6 ends 1.06 m from the target with computed torque, and 39 mm with joint PD. #2
  and #7 return to within 3 mm.
- **Under the block, the candidates at leak rate 0.03 wait less** than report 005's
  ESN: their reference is 2.3 to 3.5% of the path ahead at the release, against
  1.1%. #6, with a fast leak rate, waits the most (0.7%).

### 3.4 Step B: a star around candidate #2

![Step B](results/summary/star.png)

Each line of the star changes one of #2's settings, keeping the others, with ten
seeds each, undisturbed, under the pushes, and under the block.

| Setting | Lost seeds | Undisturbed path RMSE | Back after the push at the target: median / slowest | Block: lead | Block: farthest from the path |
| --- | --- | --- | --- | --- | --- |
| Spectral radius 0.99 | 1 (seed 9) | 1.07 mm | 4.1 / not back | 1.9% | 40 mm |
| 1.1 | 1 (seed 3) | 0.96 mm | 2.6 / 6.6 s | 2.1% | 43 mm |
| 1.2 | 0 | 0.91 mm | 2.3 / 6.4 s | 2.4% | 48 mm |
| 1.3 | 0 | 0.87 mm | 2.5 / 7.1 s | 2.6% | 47 mm |
| **1.35 (#2)** | 0 | 0.86 mm | 2.5 / 7.4 s | 2.8% | 44 mm |
| 1.4 | 0 | 0.84 mm | 2.4 / 7.2 s | 2.8% | 42 mm |
| 1.5 | 0 | 0.84 mm | 2.9 / 4.6 s | 2.9% | 37 mm |
| 1.75 | 0 | 0.91 mm | 3.1 / 8.2 s | 3.7% | 38 mm |
| Leak rate 0.015 | 0 | 1.39 mm | 2.3 / 5.8 s | 3.2% | 62 mm |
| 0.02 | 0 | 0.96 mm | 2.3 / 3.0 s | 3.1% | 58 mm |
| 0.025 | 0 | 0.88 mm | 2.3 / 4.5 s | 3.0% | 52 mm |
| **0.03 (#2)** | 0 | 0.86 mm | 2.5 / 7.4 s | 2.8% | 44 mm |
| 0.04 | 2 (seeds 4, 8) | 0.92 mm | 2.7 / 5.7 s | 2.2% | 31 mm |
| 0.05 | 5 (seeds 0, 4, 6, 8, 9) | 1.03 mm | 3.8 / 4.6 s | 1.6% | 28 mm |
| Input scaling 1.5 | 0 | 0.86 mm | 2.0 / 3.4 s | 2.6% | 41 mm |
| 1.75 | 0 | 0.88 mm | 2.3 / 4.3 s | 2.7% | 42 mm |
| **2 (#2)** | 0 | 0.86 mm | 2.5 / 7.4 s | 2.8% | 44 mm |
| 2.25 | 1 (seed 1) | 0.91 mm | 2.9 / 7.0 s | 2.7% | 47 mm |

Medians over the seeds; the return after the push at the target is the slower
tracker's, over the seeds not lost; a lead and a distance under the block, the
larger of the two trackers'. Every setting holds for all ten seeds undisturbed and
under the block.

- **A first rule failed nearly every setting under the pushes.** It required every
  seed to arrive and hold under the pushes, and to be back within 5 mm of the path
  before each next push. But the push at 13 s falls inside the 2 s hold window when
  the hand arrives after 11 s, and a hand still settling 5 to 7.5 mm from the path
  at the next push counted as not back. The real failures are the lost seeds, after
  the push at the target: in 11 of their 15 lost runs, the hand is back on the
  taught path, but has slid back along it toward the start, and stops 25 mm to 1.2 m
  from the target; in 3, it ends off the path, and in one it never arrives. The rule
  was revised to the measures above.
- **#2 lies on a plateau.** Spectral radius 1.2 to 1.75, leak rate 0.015 to 0.03,
  and input scaling 1.5 to 2 lose no seed, and their median returns lie within 2.0
  to 3.1 s. Spectral radius 0.99 and 1.1, leak rate 0.04 and 0.05, and input scaling
  2.25 lose seeds.
- **Memory keeps the target, waiting holds the arm.** A slower leak rate or a larger
  spectral radius keeps the target after the push, but lets the reference run
  further ahead under the block: from 1.9% at spectral radius 0.99 to 3.7% at 1.75,
  and from 1.6% at leak rate 0.05 to 3.2% at 0.015. Along the leak rate, the hand
  then strays further after the release, from 28 mm at 0.05 to 62 mm at 0.015;
  along the spectral radius, it does not. A faster leak rate waits, and loses
  seeds.
- **The seeds differ as much as the settings.** Within the plateau, the slowest seed
  of a setting takes 3 to 8 s to return, the medians 2 to 3 s.

### 3.5 Representatives, and a small grid

Eight seeds of the star were trained and tried in the apps: the typical seeds of
#2's setting, of leak rates 0.02, 0.025, and 0.04, of input scaling 1.5, and of
spectral radii 1.2 and 1.75, and a lost seed of leak rate 0.04. **In the apps, they
felt no different from one another.**

A small grid then combined the values that did better in the star: leak rate 0.02,
0.025, and 0.03, input scaling 1.5 and 2, and spectral radius 1.2 and 1.35, ten
seeds each, in the three scenarios of Step B. A rule fixed before the grid ran
decided whether a setting replaced #2's as the setting of the chosen ESN: if, over
its ten seeds,

1. no seed fails undisturbed or under the block, or is lost under the pushes;
2. after the push at the target, its hand is back on the path sooner than with
   #2's setting for at least 8 of the 10 seeds, seed by seed, with a median gain
   over 0.3 s (the slower tracker; a hand not back by the end of the run counts as
   back then);
3. it gives up no more than 0.1 mm in the median undisturbed path RMSE, and no more
   than 10 mm in the median farthest distance from the path under the block.

| Spectral radius | Leak rate | Input scaling | Undisturbed path RMSE | Back after the push at the target: median / slowest | Sooner than #2's | Median gain | Block: farthest from the path | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.2 | 0.02 | 1.5 | 1.03 mm | 2.3 / 4.8 s | 7 | +0.18 s | 53 mm | fails 2 and 3 |
| 1.2 | 0.02 | 2 | 1.01 mm | 2.6 / 3.6 s | 5 | +0.01 s | 62 mm | fails 2 and 3 |
| 1.2 | 0.025 | 1.5 | 0.98 mm | 2.0 / 4.0 s | 8 | +0.66 s | 50 mm | fails 3 |
| 1.2 | 0.025 | 2 | 0.97 mm | 2.5 / 4.5 s | 4 | −0.01 s | 53 mm | fails 2 and 3 |
| 1.2 | 0.03 | 1.5 | 0.92 mm | 2.0 / 3.4 s | 7 | +0.61 s | 44 mm | fails 2 |
| 1.2 | 0.03 | 2 | 0.91 mm | 2.3 / 6.4 s | 5 | +0.03 s | 48 mm | fails 2 |
| 1.35 | 0.02 | 1.5 | 0.94 mm | 2.2 / 3.9 s | 7 | +0.35 s | 54 mm | fails 2 |
| 1.35 | 0.02 | 2 | 0.96 mm | 2.3 / 3.0 s | 7 | +0.22 s | 58 mm | fails 2 and 3 |
| 1.35 | 0.025 | 1.5 | 0.87 mm | 2.4 / 3.3 s | 6 | +0.17 s | 48 mm | fails 2 |
| 1.35 | 0.025 | 2 | 0.88 mm | 2.3 / 4.5 s | 6 | +0.20 s | 52 mm | fails 2 |
| **1.35** | **0.03** | **1.5** | 0.86 mm | 2.0 / 3.4 s | 8 | +0.41 s | 41 mm | **replaces #2's** |
| 1.35 | 0.03 | 2 | 0.86 mm | 2.5 / 7.4 s |  |  | 44 mm | #2's setting |

No seed is lost or fails in any setting of the grid. The settings it shares with
the star repeat it exactly.

- **Only input scaling 1.5, at #2's leak rate and spectral radius, replaces #2's
  setting,** and narrowly: sooner for 8 of the 10 seeds, by 0.41 s in the median.
  Spectral radius 1.2 with leak rate 0.025 and input scaling 1.5 gained more (0.66
  s, for 8 seeds) but gave up 0.12 mm undisturbed.
- **The chosen ESN is its typical seed, 8,** which had been tried in the apps as a
  representative of input scaling 1.5.

### 3.6 Validation: the chosen ESN against the replay

The ESN validated (`grid_lr0.03_sr1.35_is1.5_seed8_filtered`), its parameters
([`esn.toml`](results/20261013-104954-grid_lr0.03_sr1.35_is1.5_seed8_filtered/esn.toml)):

| Parameter | Value |
| --- | --- |
| Neurons | 400, tanh |
| Sparsity | 0.05, the fraction of nonzero reservoir connections |
| Spectral radius | 1.35 |
| Leak rate | 0.03 |
| Input | the arm's measured joint angles every 10 ms, normalized by the range each covers in the filtered take, (q − (67.80°, 71.74°)) / (46.54°, 41.74°), so that the take spans [−1, 1] |
| Input scaling | 1.5, the scale of the random input weights |
| Bias | a random constant input to each neuron |
| Readout | ridge regression on the reservoir state, with a bias; its output, converted back to joint angles, is the next posture, 10 ms on |
| Ridge | 1e-2, the regularization of the readout |
| Training | teacher forcing on the filtered take alone, without noise |
| Warm-up | 1 s of the held start posture, from a reset reservoir |
| Seed | 8, of the reservoir, its input weights, and its biases |

The trackers are report 005's ([Section 3.7](../005-tuned-on-the-robot/README.md#37-validation-the-chosen-esn-against-the-replay)):
computed torque with K_p = 400 s⁻² and K_d = 4 s⁻¹ on both joints, and joint PD with
K_p = 74.96 and 8.96 N m/rad and K_d = 0.750 and 0.0896 N m s/rad, both given a
zero reference velocity.

| Scenario | Chosen ESN | Replay |
| --- | --- | --- |
| Nominal: path RMSE (computed torque / joint PD) | 0.80 / 0.92 mm | 0.13 / 0.85 mm |
| Nominal: arrival | 9.67 / 9.67 s | 9.60 / 9.59 s |
| Block: path RMSE | 12.6 / 13.3 mm | 7.5 / 16.6 mm |
| Block: force at the grip (peak) | 10.7 / 10.6 N | 9.4 / 11.3 N |
| Block: force at the release (pressing) | 2.8 / 4.7 N | 18.5 / 20.0 N |
| Block: reference ahead of the hand at the release | 36 / 36 mm | 154 / 156 mm |
| Block: after the release, farthest from the path | 51 / 44 mm | 39 / 53 mm |
| Block: after the release, back within 5 mm of it | 2.3 / 2.9 s | 1.3 / 1.6 s |
| Block: arrival | 10.19 / 10.20 s | 9.60 / 9.59 s |
| Pushes: path RMSE | 47.9 / 42.1 mm | 34.1 / 32.7 mm |
| Pushes: distance from the target at the end | 3.2 / 3.2 mm | 2.8 / 2.8 mm |
| Offsets: starts that arrive and hold | 49 / 49 of 49 | 49 / 49 of 49 |
| Offsets: median path RMSE | 17.5 / 22.8 mm | 14.6 / 27.0 mm |
| Offsets: median peak torque | 81 / 7.0 N m | 125 / 12 N m |

The force at the grip is the damper stopping the moving hand; the force at the
release is the mean over the block's last 0.1 s, how hard the arm then presses.
Offset runs: [the map of the 49 starts](results/20261013-114518-offsets_robot_lr0.03_sr1.35_is1.5_seed8_filtered/grid.png).

- **Undisturbed, the ESN follows the taught path within 0.8 to 0.9 mm** and arrives
  0.07 s after the take; the replay follows it within 0.13 mm with computed torque
  and 0.85 mm with joint PD.
- **Under the block, the ESN waits only in part.** While the tip is held, its
  output creeps on, 36 mm ahead of the hand at the release, so its arm presses with
  3 to 5 N, against the replay's 18 to 20 N, whose reference runs 155 mm ahead. The
  peak force, about 10 N for both, is the grip's.
- **After the release, the ESN's arm lurches as far as the replay's,** 44 to 51 mm
  off the taught path (the replay 39 to 53 mm, ringing), and is back within 5 mm in
  2.3 to 2.9 s (the replay 1.3 to 1.6 s). It then carries on about 0.6 s behind the
  take and arrives at 10.2 s. Report 005's ESN waited more: 14 mm ahead at the
  release, lurching 20 to 23 mm.
- **From every offset start, the ESN arrives and holds,** with either tracker, as
  the replay does; report 005's ESN failed from 19 of the 21 starts with the elbow
  straighter. With computed torque, the reference's first step from an offset costs
  a median peak torque of 81 N m, against the replay's 125 N m (7.0 and 12 N m with
  joint PD).

### 3.7 Pushes: how the ESN yields

![Distance from the taught path under the pushes](results/summary/flexibility.png)

The farthest each arm is carried off the taught path after each push, and when it
settles back within 5 mm (counted from the push's end, until the next push or 4 s
after its onset):

| Arm | Tracker | Push at 4 s | Push at 8 s | Push at 13 s | Path RMSE |
| --- | --- | --- | --- | --- | --- |
| Chosen ESN | computed torque | 49 mm, 3.3 s | 198 mm, 2.5 s | 165 mm, 1.8 s | 47.9 mm |
| Chosen ESN | joint PD | 35 mm, 3.4 s | 173 mm, 2.5 s | 149 mm, 1.7 s | 42.1 mm |
| Replay | computed torque | 34 mm, 1.4 s | 149 mm, 1.6 s | 116 mm, 1.3 s | 34.1 mm |
| Replay | joint PD | 14 mm, 1.7 s | 150 mm, 2.0 s | 115 mm, 2.2 s | 32.7 mm |

Both arms arrive (the ESN's at 10.5 to 10.6 s) and hold, and end within 3.2 mm of
the target.

- **The ESN yields.** Every push carries its arm farther off the path than the
  replay's, 1.2 to 2.5 times as far, and it settles back more slowly, in 1.7 to
  3.4 s, against the replay's 1.3 to 2.2 s. Its reference gives way at once: at the
  end of the pushes at 8 and 13 s, it is within 0.8% of the path of the arm, and
  after the backward push at 4 s 7 to 9% ahead, as the replay's (9%).
- **But it always comes back,** after the push at the target too, in 1.7 to 1.8 s,
  where report 005's ESN wandered off for about 4 s with computed torque. That
  return is what the search tuned for (Sections 3.4 and 3.5).
- **The replay is the stiffer.** Its reference ignores the arm, so the tracker drags
  the arm straight back onto the take's course, ringing at ζ = 0.1.

### 3.8 The chosen ESN, scenario by scenario

The animations show three arms side by side on one clock, each rendered by
skelarm's player; the purple dot is the target, the red arrow the force at the
tip. Each panel names the arm's reference, the chosen ESN or the replay of the
take by time, and its tracker: the chosen ESN and the replay through joint PD, and
the chosen ESN through computed torque. The figures show the chosen ESN's joint
angles (the taught motion, thick gray; the ESN's output, dashed blue; the ESN's
arm, blue; the replay's arm, orange) and the joint torques, with both trackers;
the disturbances are shaded.

#### Nominal

![Animation: nominal](results/summary/nominal.gif)

![Joint angles: nominal](results/summary/nominal_joints.png)

![Joint torques: nominal](results/summary/nominal_torques.png)

The ESN's arm follows the taught joint path, its output on it, and arrives at
9.67 s, with the take. Its torques stay within about 0.1 N m after its first step
(0.57 N m at the start with computed torque); the replay's, tracking the take's
remaining pixel steps, jitter and ring, up to 0.86 N m with computed torque.

#### Block

![Animation: block](results/summary/block.gif)

![Joint angles: block](results/summary/block_joints.png)

![Joint torques: block](results/summary/block_torques.png)

While the tip is held, from 4.5 s to 5.5 s, the ESN's output for joint 1 waits
with the arm, but joint 2's creeps on, from 96° to about 102°: 36 mm ahead of the
hand at the release. Its torques stay under 1.5 N m; the replay's climb to 12 N m
until the release, and its arm snaps forward and rings for about 1.5 s. The ESN's
arm lurches 44 to 51 mm off the path at the release, then carries on about 0.6 s
behind the take and arrives at 10.2 s.

#### Pushes

![Animation: pushes](results/summary/pushes.gif)

![Joint angles: pushes](results/summary/pushes_joints.png)

![Joint torques: pushes](results/summary/pushes_torques.png)

After each push, the chosen ESN's output goes with the arm and carries on from
where it was pushed; its torques peak at 2.2 to 4.5 N m, as high as the replay's
or lower, and, with joint PD, die out within about 1 s, while the replay's ring for
2 to 3 s.

#### Offset starts

![Animation: an offset start](results/summary/offset.gif)

![Joint angles: an offset start](results/summary/offset_joints.png)

![Joint torques: an offset start](results/summary/offset_torques.png)

From the start offset by +15° in both joints, the arm turned counterclockwise and
the elbow more bent, the ESN's output brings the hand onto the taught path within
its first 0.2 m, farther along it than the take: the arm then runs about 0.5 s
ahead of the take and arrives at 8.9 s, its path 32 mm off the taught one with
joint PD (36 mm with computed torque), having skipped the path's first part. From
every offset start, the hand joins the taught path near its start in the same way
([the hand paths](results/20261013-114518-offsets_robot_lr0.03_sr1.35_is1.5_seed8_filtered/paths.png)).
The replay's reference jumps back to the take's start: its arm swings there and
rings for about 2 s with joint PD (68 mm off the path), and with computed torque
the jump costs a peak torque of 257 N m, against the ESN's 121 N m.

![Animation: the offset start from which report 005's ESN failed](results/summary/offset_straight.gif)

![Joint angles: the offset start from which report 005's ESN failed](results/summary/offset_straight_joints.png)

![Joint torques: the offset start from which report 005's ESN failed](results/summary/offset_straight_torques.png)

From the start offset by −10° in joint 2, the elbow straighter, one of the starts
from which report 005's ESN failed, the chosen ESN barely moves the arm for about
0.8 s, then reaches along a path close to the taught one, about 1 s behind the
take, and arrives at 10.7 s, its path 10 mm off the taught one. The replay's reference
jumps to the take's start, and its arm follows it there before replaying the take
(7 to 8 mm off the path).

### 3.9 The reservoir states

As in report 005 ([Section 3.10](../005-tuned-on-the-robot/README.md#310-the-reservoir-states)),
[`robot_states.py`](../../experiments/robot_states.py) recomputes the chosen ESN's
reservoir states from the runs' logs, and refuses a log whose reference the
recomputed ESN does not reproduce; every one here passes. The figures
([run](results/20261013-120124-states_robot_lr0.03_sr1.35_is1.5_seed8_filtered))
draw the states of the ESN driving the robot (blue) over those of the take fed in as
in training (thick gray), with the warm-up and the disturbances shaded.

#### Eight neurons over the whole run

![Neurons over the whole run, joint PD](results/20261013-120124-states_robot_lr0.03_sr1.35_is1.5_seed8_filtered/neurons_pd_w20_z0.1.png)

![Neurons over the whole run, computed torque](results/20261013-120124-states_robot_lr0.03_sr1.35_is1.5_seed8_filtered/neurons_computed_torque_w20_z0.1.png)

The same eight of the 400 neurons as in report 005 (picked at random, seed 0),
from the reset to the run's end, in this ESN's reservoir. Most switch between
plateaus near ±1 during the reach and settle once the arm holds at the target.

- **Undisturbed,** each neuron follows the take's states.
- **Under the block,** the states hold nearly still while the tip is held, then
  resume the take's course about 0.5 s late.
- **Each push** makes some of them swing; after the push at 13 s, neurons 16 and 106
  swing the farthest and settle back within 2 to 3 s.
- **From (+15°, +15°),** they leave the take's course early and stay about 0.5 s
  ahead of it to the end of the reach; **from (0°, −10°),** they stay about 1 s
  behind it.

#### The warm-up

![Neurons over the warm-up, joint PD](results/20261013-120124-states_robot_lr0.03_sr1.35_is1.5_seed8_filtered/neurons_warmup_pd_w20_z0.1.png)

![Neurons over the warm-up, computed torque](results/20261013-120124-states_robot_lr0.03_sr1.35_is1.5_seed8_filtered/neurons_warmup_computed_torque_w20_z0.1.png)

The same neurons over the warm-up and the task's first 3 s, for the undisturbed run
and the two offset starts.

- **From the demonstrated start,** the warm-up brings every neuron onto the take's
  states, and the run then follows them.
- **From (+15°, +15°),** it leaves several neurons elsewhere, 201 and 334 the
  farthest; after the start, they cut across the take's course, ahead of it.
- **From (0°, −10°),** from which report 005's ESN failed, the warm-up ends close
  to the take's states, neurons 250 and 334 a little off; after the start, the
  neurons lag the take's while the arm waits at its start, then move on behind
  them.

#### The principal components

![Principal components over time, joint PD](results/20261013-120124-states_robot_lr0.03_sr1.35_is1.5_seed8_filtered/pca_pd_w20_z0.1.png)

![Principal components over time, computed torque](results/20261013-120124-states_robot_lr0.03_sr1.35_is1.5_seed8_filtered/pca_computed_torque_w20_z0.1.png)

![Planes of the principal components, joint PD](results/20261013-120124-states_robot_lr0.03_sr1.35_is1.5_seed8_filtered/pca_planes_pd_w20_z0.1.png)

![Planes of the principal components, computed torque](results/20261013-120124-states_robot_lr0.03_sr1.35_is1.5_seed8_filtered/pca_planes_computed_torque_w20_z0.1.png)

The principal components are those of all five runs' states with both trackers,
warm-up included: PC1, PC2, and PC3 hold 73.7%, 20.7%, and 2.5% of the variance. In
the planes, the warm-up is dashed, a hollow dot marks the state just after the
reset, and a filled one the state at the task's start.

- **The take's states trace an arc** in the plane of PC1 and PC2, from the task's
  start, at the upper right, down and around to the target, at the left. The
  warm-up jumps from the reset to the arc's start, and the undisturbed run follows
  the arc.
- **Under the block, the states pause** along the arc while the tip is held (flat
  over time), then go on behind the take.
- **Each push throws the states off the arc** and back, the push at 13 s in a small
  loop near its end.
- **From both offset starts,** the warm-up ends off the arc's start, below it from
  (+15°, +15°) and above it from (0°, −10°), and the states join the arc within the
  first seconds: unlike report 005's ESN from (0°, −10°), whose states moved to a
  fixed point and never started the reach.

## 4. Observations

- **A spectral radius over 1 pays at a slow leak rate.** At leak rate 0.03, every
  input scaling does better with spectral radius 1.2 to 1.5 than with 0.99, seed by
  seed; report 005's cap below 1 had kept the search out of where the best settings
  lie.
- **Robust from many starts, though scored from one.** The ESNs at leak rate 0.03
  arrive from all 49 offset starts on the robot, where report 005's, at 0.05, failed
  from 19, although offsets were never in the score. Not every seed does: on its own,
  the representative of spectral radius 1.2 (seed 2) never arrives from 11 of the
  49.
- **The pushes, not the undisturbed runs, separate the settings.** Undisturbed and
  under the block, every setting of the star holds for every seed; after the push at
  the target, some seeds lose the target, a failure the undisturbed score cannot
  see.
- **Keeping the target and waiting pull the leak rate apart.** A slower leak rate,
  or a larger spectral radius, keeps the target after a push, but lets the
  reference run ahead of a held arm; a faster one waits, and loses the target. The
  chosen ESN keeps the target and waits less than report 005's.
- **Below what a person feels.** Within the plateau around #2, the settings'
  medians differ by at most a few tenths of a millimeter undisturbed and of a second
  in the return after the push at the target, which the apps did not show: the
  representatives felt alike. The choice among them rests on a rule fixed
  beforehand, over ten seeds, and it was a narrow one.
- **A rule is a measurement too.** The first rule under the pushes failed nearly
  every setting for reasons of its own, a hold window overlapping a push and a
  return deadline too tight for an underdamped tracker, until the runs were looked
  at.

## 5. Next steps

1. **The pushes and the hold:** move the push at the target after the hold window
   ends for every arrival, or measure the hold before it, so that the hold check
   and the push no longer overlap.
2. **Waiting, scored:** if the ESN should wait for a held arm as report 005's did,
   put the block's lead into the score next to the return after the push at the
   target, and search between leak rates 0.03 and 0.04, where the two pull apart.
3. **Offset starts in the score:** add a few offset starts to the sweeps on the
   robot, as report 005 proposed: a representative of the star (spectral radius 1.2,
   seed 2) never arrives from 11 of the 49 starts on its own.
4. **Report 004's comparisons,** still open: the zero reference velocity against a
   tracker given the reference velocity, now with an ESN tuned under disturbances.
