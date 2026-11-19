# The reach and the periodic motion in one ESN, v2 (stage 3b of the multi-task study)

One ESN does two tasks on the two-link arm with no joint limits: stage 1's reach to
(−0.6, 0.4) m and stage 3a's periodic motion around the ellipse. A context added to the
ESN's input names the task, as stage 2's context names a target, and changing the context
during a run changes the task. [PLAN.md](../../PLAN.md) gives the study's stages; the
reach alone is report 023's, three targets by context report 025's, and the periodic
motion alone reports 026's and 027's.

**Status.** The rule and the claims below were proposed on 2026-11-18 in
[stage 3a's README](../manual_demonstration_v2_periodic/README.md), before any run of 3a.
On 2026-11-19, after 3a's reports and before any run of 3b, the maintainer confirmed four
points: the reach has one target, stage 1's, and not stage 2's three; the ellipse is
taught by the take by hand alone; 3b has no pushes and no block; and claim 9's second
stretch on the ellipse lasts 40 s. The maintainer then had 3b started on this text. What
was changed from the proposal is dated where it stands. No run of 3b had been made when
this file was committed.

## The robot, the tasks, and the takes

**The robot** is stage 1's: v2's arm with no joint limits
([`reach_manual_v2_no_limits.toml`](../demonstrations/reach_manual_v2_no_limits.toml)).

| Task | Context | Takes | From |
| --- | --- | --- | --- |
| the reach to (−0.6, 0.4) m, within 20 mm | `reach` | stage 1's nine takes by hand, one from each of nine start postures, filtered at 2 Hz (`results/demonstrations/20261117-164658-reach_manual_v2_no_limits`) | report 023 |
| the periodic motion around the ellipse centered at (−0.1, 0.55) m, 0.20 m by 0.12 m | `ellipse` | the one take by hand, ten loops, filtered at 2 Hz (`results/demonstrations/20261119-093746-ellipse_manual_v2_no_limits`); its taught period is 5.12 s | report 027 |

No take is new. The scripted take of the ellipse is not used. The nearest the ellipse
comes to the reach's target is 0.33 m.

**The ESN** is trained on all ten takes. Its input holds, after the two joint angles, one
value per task: 1 for the task being done and 0 for the other. The context is fed in as
in stage 2 (report 025): divided by the input scaling, so that its input weights have the
scale 1 whatever the input scaling is (`context_scaling = 1.0`,
`context_times_input_scaling = false`; proposed on 2026-11-19 after report 025, and not
searched). The rest is stage 1's: 800 neurons, sparsity 0.05, a bias, a warm-up of 1 s on
the held start posture with the run's first context, 10 ms. The joint angles are
normalized by the range all ten takes cover.

**The tracker** is stage 1's: report 017's joint PD (ω = 20 rad/s, damping ratio 0.4,
given the reference's velocity, its torque computed every 10 ms), its gains set where the
take of reports 005 to 020 ends.

## The start postures

| Starts | What they are | Used with |
| --- | --- | --- |
| the nine reach takes' | where each take of the reach starts | `reach` |
| eight on the curve | the hand on the ellipse at its angles 0°, 45°, …, 315°; the first is the ellipse take's start | `ellipse` |
| 36 untaught | stage 1's untaught starts, hands spread evenly over the workspace. The search's last step scores from them with each context | both |
| 36 unseen | stage 1's unseen starts, the untaught rings turned 15° back. No search or scoring runs from them | `reach`, in claim 6 |

## The runs and their measures

A run is given a context from its warm-up on. Its measures are its task's:

- **With `reach`:** stage 1's. A run **holds** if the hand arrives within 20 mm of the
  target and stays within it for the 2 s after. Its detour ratio is the length of the
  hand's path until it arrives over the straight distance from its start.
- **With `ellipse`:** stage 3a's, taken from the ellipse itself. A run **cycles** if,
  over its last five taught periods (25.6 s), the hand's angle about the curve's center
  grows by four to six turns and the curve RMS is at most 20 mm.
- **A run passes** if it holds, with `reach`, or cycles, with `ellipse`.
- **How a run starts,** as in stage 1: the reference's fastest joint speed and the
  largest joint torque.

**When the context changes during a run,** the part after the change is measured as a
run of the new task that begins at the change: after a change to `reach`, whether the
hand arrives and holds, and the time from the change to its arrival; after a change to
`ellipse`, whether the hand cycles at the end of the run, and the joining time counted
from the change. Reported with them, without a claim: the reference's fastest joint
speed and the largest joint torque in the 2 s after each change.

**The replays** keep to their take and their task, so the claims about a change of
context are counts of the ESN's runs alone.

## The search

The search chooses two ESNs by how reliably they do both tasks, and then by how well.
It is 3a's, with both tasks in the score.

1. **The runs.** Each ESN runs on the robot, undisturbed, with the context `reach` from
   each of the nine reach takes' starts for 24 s, and with the context `ellipse` from
   each of the eight starts on the curve for 40 s: 17 runs.
2. **A setting's score** is how many of its runs pass, the most first. Settings with as
   many are ranked twice among themselves: by the median detour ratio of their reach
   runs that arrive, and by the median curve RMS of their ellipse runs that cycle, each
   the smallest first, a setting with no such run last. The worse of a setting's two
   ranks decides, the smaller first, so that neither task's measure is weighed against
   the other's in its own unit. Settings still alike are ordered by the better of
   their two ranks (this last sentence added on 2026-11-19, before any run, to make
   the order complete).
3. **The coarse search:** stage 1's 108 settings with one reservoir each (seed 0):

   | Parameter | Values |
   | --- | --- |
   | leak rate | 0.01, 0.03, 0.1, 0.3 |
   | spectral radius | 0.9, 1.2, 1.5 |
   | input scaling | 0.1, 0.5, 2 |
   | ridge | 10⁻⁴, 10⁻², 1 |

4. **The center** is the coarse setting with the best score.
5. **The fine search:** 81 settings around the center, with five reservoirs each (seeds
   1 to 5), laid out as in stage 1 and 3a: each parameter takes the center's value and
   the one halfway to its coarse neighbor on each side, in ratio for the leak rate, the
   input scaling, and the ridge, and in difference for the spectral radius; beyond an
   edge of the coarse grid, as far as on the other side. A setting's runs are the 85 of
   its five seeds.
6. **Scoring again with the untaught starts.** The 20 fine settings with the best
   scores, each with its five reservoirs again, run from the starts of step 1 and from
   the 36 untaught ones with each context: 89 runs an ESN, 445 a setting, scored as in
   step 2.
7. **The chosen ESNs,** two-task ESN #1 and ESN #2, are the typical seeds of the two
   settings with the best scores in step 6: of a setting's seeds with the most runs
   that pass, the one whose median curve RMS is nearest the setting's.

The search has no change of context during a run, no pushes, no block, and no held-out
seeds.

### The search's outcome

Added after each step's run.

**The center.** Of the coarse search's 108 settings (`20261119-190537-coarse_search`),
none passes all 17 runs. The best passes 15: leak rate 0.1, spectral radius 1.5, input
scaling 0.5, and ridge 1, with 8 of its 9 reaches holding and 7 of its 8 ellipse runs
cycling, a median detour ratio of 1.040, and a median curve RMS of 8.55 mm. Two settings
pass 13, and all others 10 or fewer.

The reach is the easier of the two in one ESN: 552 of the 972 reach runs hold, and 23
settings hold from all nine starts. Of the 864 ellipse runs 44 cycle, where the ellipse
alone had 205 in 3a, and no setting cycles from all eight starts. Of the 820 ellipse
runs that do not cycle, the hand of 574 stays within half a turn over the last five
taught periods, that of 74 goes around backwards, that of 36 goes around too slowly,
and that of 136 keeps the pace but is more than 20 mm RMS from the ellipse. Of the 44
that cycle, 36 have the ridge of 1, the grid's largest.

The fine search's values around the center:

| Parameter | Values |
| --- | --- |
| leak rate | 0.055, 0.1, 0.17 |
| spectral radius | 1.35, 1.5, 1.65 |
| input scaling | 0.22, 0.5, 1 |
| ridge | 10⁻¹, 1, 10 |

The center's spectral radius and ridge are the coarse grid's largest, so the fine search
goes as far above them as below.

**The fine search** (`20261119-193314-fine_search`, 61 minutes). None of its 81 settings
passes all 85 runs. The best:

| Leak rate | Spectral radius | Input scaling | Ridge | Runs that pass, of 85 | Reaches that hold, of 45 | Ellipse runs that cycle, of 40 | Median detour ratio | Median curve RMS (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.055 | 1.65 | 0.5 | 10 | 69 | 33 | 36 | 1.048 | 8.65 |
| 0.17 | 1.65 | 0.22 | 10 | 69 | 35 | 34 | 1.165 | 13.14 |
| 0.055 | 1.5 | 0.5 | 10 | 68 | 41 | 27 | 1.044 | 7.91 |
| 0.1 | 1.5 | 1 | 10 | 63 | 41 | 22 | 1.024 | 11.21 |
| 0.055 | 1.35 | 1 | 10 | 59 | 43 | 16 | 1.015 | 6.46 |

The 20th setting passes 51 runs. Four settings pass 51, and the ranks leave the fourth
out. Of the 20 best, 16 have the ridge of 10 and nine the spectral radius of 1.65, each
the fine grid's largest, as the center's were the coarse grid's. The rule has no step
beyond the fine grid, and the scoring goes on with these 20.

The two tasks again differ. Of the 3,645 reach runs 3,205 hold; 393 arrive and leave
the target, 42 never arrive, and 5 arrive too late for their 2 s to be seen. Of the
3,240 ellipse runs 545 cycle, 427 of them with the ridge of 10. Of the 2,695 that do not,
the hand of 1,317 stays within half a turn over the last five taught periods, that of
1,076 keeps the pace but is more than 20 mm RMS from the ellipse, that of 241 goes
around too slowly, that of 45 backwards, and that of 16 too fast.

The reservoir decides as much as the setting. Of the 405 ESNs, 184 hold all nine
reaches and 29 cycle from all eight starts; 12 do both, 11 of them with the ridge of 10,
and 11 more fail one run. A setting's five seeds seldom agree on the ellipse: the fifth
setting above has two seeds that cycle from all eight starts and three that cycle from
none, and the fourth has two of each. The first two settings are the only ones whose
five seeds all cycle from at least six starts, and they lose reaches instead: their
seeds hold five to eight of the nine.

**The scoring with the untaught starts** (`20261119-205228-untaught_search`, 84
minutes). No setting passes all 445 runs, and none of the 100 ESNs all of its 89. The
best:

| Leak rate | Spectral radius | Input scaling | Ridge | Runs that pass, of 445 | Reaches that hold, of 225 | Ellipse runs that cycle, of 220 | Median detour ratio | Median curve RMS (mm) | Typical seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.17 | 1.65 | 0.22 | 10 | 343 | 167 | 176 | 1.221 | 12.93 | 4 |
| 0.055 | 1.5 | 0.5 | 10 | 312 | 197 | 115 | 1.131 | 8.36 | 4 |
| 0.1 | 1.65 | 0.22 | 10 | 311 | 135 | 176 | 1.227 | 10.20 | 5 |
| 0.055 | 1.65 | 0.5 | 10 | 301 | 157 | 144 | 1.183 | 8.95 | 5 |
| 0.055 | 1.5 | 1 | 10 | 291 | 208 | 83 | 1.115 | 13.36 | 3 |

No two of the first ten settings pass as many runs, so the ranks decide nothing among
them. Over all 20 settings, 3,791 of the 4,500 reach runs hold (781 of 900 from the
takes' starts, 3,010 of 3,600 from the untaught) and 1,381 of the 4,400 ellipse runs
cycle (364 of 800 from the starts on the curve, 1,017 of 3,600 from the untaught). Five
ESNs hold every reach and seven cycle from every start; none does both.

**The chosen ESNs:**

| | Leak rate | Spectral radius | Input scaling | Ridge | Seed | Its runs that pass, of 89 | Reach, of 9 takes' and 36 untaught starts | Ellipse, of 8 starts on the curve and 36 untaught |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-task ESN #1 | 0.17 | 1.65 | 0.22 | 10 | 4 | 86 | 8 and 34 | 8 and 36 |
| two-task ESN #2 | 0.055 | 1.5 | 0.5 | 10 | 4 | 81 | 8 and 30 | 8 and 35 |

They are trained and kept as `20261119-222323-esn1_lr0.17_sr1.65_is0.22_ridge10_seed4`
and `20261119-222327-esn2_lr0.055_sr1.5_is0.5_ridge10_seed4`.

Each is its setting's seed with the most runs that pass, which no other seed of the
setting equals. In every reach either fails but one, the hand arrives, leaves the
20 mm around the target within the 2 s, and ends within 5 mm of the target; in the
one, ESN #1's from an untaught start, it arrives at 22.7 s, too late for the 2 s to be
seen in 24 s. ESN #2's one ellipse run that does not cycle keeps the pace 45 mm RMS
from the ellipse. The seeds of a setting differ here too: of ESN #1's setting, seed 3
cycles from one of the 36 untaught starts, and of ESN #2's, seed 2 from none.

The ESN with the fewest failing runs is neither: leak rate 0.055, spectral radius 1.5,
input scaling 1, ridge 10, seed 3 fails two of its 89, and its setting is fifth. The
rule chooses by the setting, so it is not one of the two.

## The claims

They keep the numbers of the proposal, after 3a's claims 1 to 5. A claim holds if it
holds for two-task ESN #1 and for two-task ESN #2.

6. **Each task by its context.** With the context `reach`, from the nine reach takes'
   starts and stage 1's 36 unseen starts, the hand arrives and holds from as many
   starts as it did for report 023's ESN #1 and ESN #2, which is all 45. With the context
   `ellipse`, from the eight starts on the curve and the 36 untaught starts, in runs of
   60 s, the hand cycles at the end from at least as many starts as for the fewer of
   3a's ESN #1 and ESN #2 of the take by hand, which is all 44.
7. **From the periodic motion to the reach.** The run starts with the context
   `ellipse` from each of the eight starts on the curve, and the context changes to
   `reach` at 16 s, which meets a hand that went on from where it started at eight
   places around the curve. The hand arrives at the target and holds, in all eight runs
   of 40 s.
8. **From the reach to the periodic motion.** The run starts with the context `reach`
   from each of the nine reach takes' starts, and the context changes to `ellipse` at
   20 s, after every reach of stage 1 has arrived and held. The hand cycles at the end
   of the run, in all nine runs of 60 s.
9. **Back and forth.** From each of the eight starts on the curve, the context goes
   from `ellipse` to `reach` at 16 s, back to `ellipse` at 36 s, and to `reach` again
   at 76 s. The hand cycles over the five taught periods before the last change, and
   arrives and holds after it, in all eight runs of 100 s. (Changed on 2026-11-19,
   before any run: the second stretch on the ellipse was 30 s, to 66 s, in runs of
   90 s. It was written for 4 s a loop. The take by hand's five taught periods are
   25.6 s, and in 3a its ESNs took up to 5.1 s to join the curve.)

Reported beside the claims, without a claim: each task's measures with its context
alone, set beside those of the ESNs that do one task (report 023's and report 027's);
and after each change of context, the time to arrive or to join and how the reference
starts.

**Two rest postures.** Stage 1's takes 03 and 04 go around the base clockwise and rest
at the target with joint 1 at −246°, a whole turn below the others' 114°. The ellipse
is taught with joint 1 from 34° to 79°. A reach that ends at −246° is then asked to go
to a curve it was taught a whole turn away. What the hand does there is reported with
claim 8.

## The runs

The configurations are written before their runs, each chosen ESN's when the search
has given it.

| Claims | What runs |
| --- | --- |
| the search | the coarse search, the fine search, and the scoring with the untaught starts, on both tasks (`sweep_robot_esn.py`) |
| the chosen ESNs | each trained once and kept: `esn1_lr0.17_sr1.65_is0.22_ridge10_seed4.toml`, `esn2_lr0.055_sr1.5_is0.5_ridge10_seed4.toml` (`autonomous_esn.py`) |
| 6 | each two-task ESN with one context for the whole run: `reach` from the nine reach takes' starts and stage 1's 36 unseen ones for 24 s, `ellipse` from the eight starts on the curve and the 36 untaught ones for 60 s (`robot_esn.py`): `robot_esn1.toml`, `robot_esn2.toml` |
| 7 | each two-task ESN with the change of context above (`robot_esn.py`, `[[switch]]`): `to_reach_robot_esn1.toml`, `to_reach_robot_esn2.toml` |
| 8 | the same: `to_ellipse_robot_esn1.toml`, `to_ellipse_robot_esn2.toml` |
| 9 | the same, with three changes: `back_and_forth_robot_esn1.toml`, `back_and_forth_robot_esn2.toml` |

**How the claims' runs are counted** (added on 2026-11-19 after the search and a
rehearsal of these configurations with another ESN in a scratch directory, before any
of the claims' runs). The counts are of the ESN's rows of each run's `metrics.csv`:

- Claim 6: a reach holds if `success` is true and `hold_observed_s` is 2 s, as the
  search counts it; an ellipse run cycles by `cycles`.
- Claim 7: after the change, `switch_success` is true and `switch_hold_observed_s` is
  2 s; the time from the change to the arrival is `switch_arrival_time_s`.
- Claim 8: `switch_cycles`, over the run's last five taught periods; the joining time
  from the change is `switch_joining_time_s`.
- Claim 9: `cycles_before_switch_3`, over the five taught periods before the change at
  76 s, and after it `switch_success` with `switch_hold_observed_s` of 2 s.

In a run with a change, the columns without `switch` measure the whole run as one of
the task it starts with, and are not used. Nor is the count of runs that cycle before
the change at 16 s: 16 s is shorter than the five taught periods a cycling run is
measured over.

Sweeps here take 30 workers. In 3a's sweeps a worker took up to 9 GB for moments, in
one measure of a run whose arm runs away; that measure now takes its points in pieces
(9f3855c), with the same values, and a worker of the coarse search here peaks at
0.36 GB.
