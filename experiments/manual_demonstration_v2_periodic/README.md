# A periodic motion, v2 (stage 3 of the multi-task study)

Reports 023 and 024 taught one reach from many start postures. This experiment is the
third stage of the multi-task study ([PLAN.md](../../PLAN.md)): the hand goes around an
ellipse, again and again, on v2's arm with no joint limits. Its ESNs have 800 neurons.
It has two parts:

- **3a, the periodic motion alone.** One take teaches the motion. Is it a limit cycle
  of the ESN and the robot together: does the hand draw the curve, join it from
  elsewhere, and come back to it after a push or a block?
- **3b, the reach and the periodic motion in one ESN.** Does the context choose the
  task, and does changing it during a run switch between them?

**Status.** The maintainer decided the curve, its pace, and the takes on 2026-11-18.
The measures, the search's rule, and the claims below were written on 2026-11-18,
before any take of this experiment was recorded and before any ESN was trained on one.
The maintainer then changed them, the same day and before any search or claim was
run ([what was changed](#what-the-maintainer-changed)), and confirmed 3a's as they
stand here. 3b's are a proposal: the maintainer confirms them before 3b starts.
The scripted take is made and imported
(`results/demonstrations/20261118-171132-ellipse_scripted_v2_no_limits`). A first take
by hand was refused by the import and is recorded again. No search or claim has been
run.

**Checks of the runners.** Before the search, the runners were tried in a scratch
directory, on the scripted take and on a made-up take in place of the hand's, with
ESNs of 200 neurons and seed 99 at settings off the search's grid (leak rates 0.05 and
0.2, spectral radius 1.0, input scaling 0.3, ridge 10⁻³), and once with 800 neurons at
such a setting to time a run. Those runs are not results and are not kept.

## The robot, the curve, and the takes

**The robot** is stage 1's: v2's arm with no joint limits
([`reach_manual_v2_no_limits.toml`](../demonstrations/reach_manual_v2_no_limits.toml)).

**The curve** is an ellipse centered at (−0.1, 0.55) m, with semi-axes of 0.20 m along
x and 0.12 m along y: 0.40 m wide, 0.24 m high, and 1.02 m around. The hand goes around
it counterclockwise, at 4 s a loop. On it the hand is 0.44 to 0.68 m from the base,
joint 1 goes from 34° to 79° and joint 2 from 82° to 123°, and the nearest the hand
comes to stage 1's target at (−0.6, 0.4) m is 0.33 m.

**The takes** are two, each imported into a demonstration run of its own:

| Take | Configuration | What it is |
| --- | --- | --- |
| by hand | [`ellipse_manual_v2_no_limits.toml`](../demonstrations/ellipse_manual_v2_no_limits.toml) | recorded with skelarm's recorder, which draws the curve as a guide. The hand starts at rest on the curve's point at angle 0, (0.1, 0.55) m, and goes around ten times or more, at about 4 s a loop |
| by a script | [`ellipse_scripted_v2_no_limits.toml`](../demonstrations/ellipse_scripted_v2_no_limits.toml) | the joint angles that put the hand on the curve at an even angle rate, 4 s a loop, for 40 s, from the same start posture (`make_periodic_take.py`). It moves at its full pace from its first sample |

Each configuration's header gives its commands. Neither take starts away from the
curve: joining the curve from elsewhere is tested, not taught. An ESN normalizes the
joint angles by the range its take covers. It is trained on the take by hand filtered
at 2 Hz, as in stage 1, and on the scripted take as it is made: the filter removes the
steps of the cursor's whole screen pixels, which the scripted take does not have, and
it would pull that take's first sample 1.2° from the start posture and its loops about
3 mm inside the curve (corrected on 2026-11-18, before any run, when the import
refused the filtered scripted take).

A take by hand ends while the hand still goes around: saved after the hand has
stopped, it would teach the ESN to stop there. The import refuses a take whose hand
is at rest over its last half second.

**A take's loops.** A loop is whole when the hand's angle about the curve's center has
grown by a turn since the last one ended, counted from the take's first sample. The
first loop holds the start from rest. The **taught loops** are the whole loops after
the first, and the **taught period** is the median of their times.

## The start postures

All have joint 2 positive, as every start of stage 1.

| Starts | What they are |
| --- | --- |
| eight on the curve | the hand on the ellipse at the curve's angles 0°, 45°, …, 315°, which the scripted take grows at an even rate: an eighth of a loop apart. The first is the takes' start. A take passes through all of them |
| 36 untaught | stage 1's untaught starts, hands spread evenly over the workspace (`tools/tip_starts.py <takes> --counts 18 12 6 --first-angle 53.3`). The search scores from them in its last step, and the claims are tested from them too |

## The runs and their measures

**The tracker** is stage 1's: report 017's joint PD (ω = 20 rad/s, damping ratio 0.4,
given the reference's velocity, its torque computed every 10 ms), its gains set where
the take of reports 005 to 020 ends.

**The arms** of a run are the ESN, fed the arm's measured joint angles, and the replay
of the take, which goes on without end. The replay repeats the take by time: it plays
the take from its first sample, and after the end of a later loop it goes back to the
end of the first loop. Of the later loops' ends, it is the one whose posture is nearest
that at the end of the first, so that the step at the seam is the smallest the take
offers. The step is reported.

**The measures** of a run are taken from the curve itself, the ellipse, not from the
take: the task is to draw the ellipse, and a take by hand misses it at times.

- **Off the curve** is the hand's distance from the ellipse (m), wherever along it the
  hand is and whenever.
- **The curve RMS** of a run is the RMS of off the curve over its last five taught
  periods (20 s at 4 s a loop).
- **A run cycles** if, over those five periods, the hand's angle about the curve's
  center grows by four to six turns (five at the taught pace, so within a fifth of
  it), and its curve RMS is at most 20 mm, stage 1's goal radius. The hand may leave
  the curve for a moment; it may not go around elsewhere or at another size.
- **The period** of a run is the median time of its whole loops in those five periods,
  reported over the taught period.
- **The joining time** is the earliest time from which the RMS of off the curve over
  every following taught period is at most 20 mm, to the end of the run.
- **How a run starts** is measured as in stage 1, by the reference's fastest joint
  speed and by the run's largest joint torque.

## The search (3a)

The search chooses two ESNs by how reliably the hand cycles, and then by how closely it
draws the ellipse. It is stage 1's one-take search with the cycle in place of
the arrival, and it is run twice by the same rule: once for the take by hand and once
for the scripted take.

1. **The runs.** Each ESN is trained on the take and generates the reference of the
   simulated robot, undisturbed, from each of the eight starts on the curve: eight runs
   of 40 s.
2. **A setting's score** is how many of its runs cycle, the most first; then the median
   curve RMS of its runs that cycle, the smallest first.
3. **The coarse search:** stage 1's 108 settings with one reservoir each (seed 0):

   | Parameter | Values |
   | --- | --- |
   | leak rate | 0.01, 0.03, 0.1, 0.3 |
   | spectral radius | 0.9, 1.2, 1.5 |
   | input scaling | 0.1, 0.5, 2 |
   | ridge | 10⁻⁴, 10⁻², 1 |

   The rest is stage 1's: 800 neurons, sparsity 0.05, a bias, a warm-up of 1 s, 10 ms.
   Report 023's ESNs (leak rates 0.055 and 0.1, spectral radii 1.2 and 1.05, input
   scalings 1 and 0.5, ridges 1 and 10) lie inside this grid but for the ridge of 10.
4. **The center** is the coarse setting with the best score.
5. **The fine search:** 81 settings around the center, with five reservoirs each (seeds
   1 to 5), laid out as in stage 1: each parameter takes the center's value and the one
   halfway to its coarse neighbor on each side, in ratio for the leak rate, the input
   scaling, and the ridge, and in difference for the spectral radius; beyond an edge of
   the coarse grid, as far as on the other side. A setting's runs are the 40 of its
   five seeds.
6. **Scoring again with the untaught starts.** The 20 fine settings with the best
   scores, each with its five reservoirs again, run from the eight starts on the curve
   and from the 36 untaught ones: 44 runs an ESN, 220 a setting, scored as in step 2.
7. **The chosen ESNs** are the typical seeds of the two settings with the best scores
   in step 6: ESN #1 and ESN #2 of that take. A setting's typical seed is, of its
   seeds with the most runs that cycle, the one whose own median curve RMS is nearest
   the setting's.

The search leaves out the pushes, the block, and held-out seeds.

### The search on the scripted take

**The center.** Of the coarse search's 108 settings
(`20261118-191114-scripted_coarse_search`), 65 cycle from all eight starts and 24 from
none. The smallest median curve RMS among the 65, 0.80 mm, is that of leak rate 0.03,
spectral radius 1.5, input scaling 2, and ridge 10⁻². The fine search's values around
it:

| Parameter | Values |
| --- | --- |
| leak rate | 0.017, 0.03, 0.055 |
| spectral radius | 1.35, 1.5, 1.65 |
| input scaling | 1, 2, 4 |
| ridge | 10⁻³, 10⁻², 10⁻¹ |

The center's spectral radius and input scaling are the coarse grid's largest, so the
fine search goes as far above them as below.

## The analysis (3a)

The chosen ESNs are tested on the robot against the replay. No size is fixed in
advance: every claim is a count or a direction against the replay, and the sizes are
reported. Each claim is tested for the ESNs of the take by hand and for the ESNs of
the scripted take, and its outcome is given for each take: it holds for a take if it
holds for that take's ESN #1 and ESN #2. No direction is claimed between the two
takes; their settings and their measures are reported side by side.

### The claims

1. **On the curve.** Undisturbed, from each of the eight starts on the curve, the hand
   still cycles at the end of 120 s, 30 taught loops, as the replay's does.
2. **Starting where the arm is.** From the seven starts on the curve other than the
   take's, and from the 36 untaught starts, the reference's fastest joint speed and the
   largest joint torque are smaller than the replay's, in their medians over each
   group of starts. The replay begins at the take's start wherever the arm is.
3. **Joining the curve.** From the 36 untaught starts, in runs of 60 s, the hand cycles
   at the end from as many starts as the replay's. The search's last step scored the
   settings from these starts, in runs of 40 s.
4. **Pushes.** Under three pushes at report 007's forces, at twice, and at four times
   them, from each of the eight starts on the curve, in runs of 80 s, the hand cycles
   at the end in at least as many of the eight runs as the replay's, at each size.
5. **A block.** With the tip held for 3 s, three quarters of a loop, from each of the
   eight starts on the curve, in runs of 60 s: the hand cycles at the end in at least
   as many of the eight runs as the replay's; and in the 2 s after the release, the
   largest joint torque and the largest off the curve are smaller than the replay's,
   in their medians over the eight runs.

Reported beside the claims, without a claim: each arm's period and curve RMS; the
joining times; how far each push sends the hand off the curve and when it is back
within 20 mm of it; and
the delay a push or the block leaves, which is how far behind its own undisturbed run
the hand is at the end of the run, in seconds. A replay by time leaves none, and an
arm that goes on from where it is keeps the delay.

**The pushes** are report 007's three, of 0.5 s at the tip: 3 N at 10°, 5 N at 80°,
and 4 N at −80°, at 8.0 s, 20.5 s, and 33.0 s. A reach's push is turned from the
direction toward its target. The periodic motion has no target, so its pushes are
turned, counterclockwise, from the direction from the run's start hand toward the
curve's center, fixed for the run. The pushes are 12.5 s apart, three loops and an
eighth: the replay is pushed at three places an eighth of a loop apart, and over the
eight starts the pushes' directions turn once around the curve. A run goes on for 47 s
after the last push, since stage 1's pushed hands took 5 to 18 s to come back.

**The block** is a stiff spring-damper, the consistent search's (report 017: 20,000 N/m,
100 N s/m), that holds the tip from 10 s to 13 s where it was at 10 s.

### The runs

The configurations are written before their runs, each chosen ESN's when the search
has given it.

| Claims | What runs |
| --- | --- |
| the search | the coarse search, the fine search, and the scoring with the untaught starts, for each take (`sweep_robot_esn.py`) |
| 1, 2 | each ESN against the replay, undisturbed, from the eight starts on the curve, for 120 s (`robot_esn.py`) |
| 2, 3 | each ESN against the replay, undisturbed, from the 36 untaught starts, for 60 s |
| 4 | each ESN against the replay under the three pushes at 1, 2, and 4 times their forces, from the eight starts on the curve, for 80 s |
| 5 | each ESN against the replay under the block, from the eight starts on the curve, for 60 s |

### What the maintainer changed

On 2026-11-18, after the first version of this README was committed (4df041d) and
before any search or claim was run, the maintainer changed 3a's rule and claims. The
text above is the changed one.

- **The error is the distance from the ellipse,** not from the take's own path: it is
  geometric and has no time in it, and drawing the target ellipse matters more than
  copying the take. The first version measured the hand from the path of the take's
  taught loops.
- **A run cycles** by its pace and by a curve RMS of at most 20 mm. The first version
  asked the hand to stay within 20 mm of the taught path at every moment. The first
  take by hand was itself 9 to 11 mm RMS from the ellipse and 33 mm at most.
- **The search ranks by the curve RMS,** in place of the RMS distance from the taught
  path.
- **The 36 unseen starts are dropped.** Claims 2 and 3 are tested from the 36 untaught
  starts, which the search's last step also scores from.
- **The progress replay is dropped.** Claims 4 and 5 compare with the replay alone.
- **The take by hand aims at 4 s a loop.** The first take went around in about 2.9 s.

## The reach and the periodic motion in one ESN (3b)

This part is a proposal. The maintainer confirms its rule and claims before 3b starts,
and what is changed then is dated.

**The ESN** is trained on two tasks: stage 1's nine takes of the reach to (−0.6, 0.4) m
(context `reach`) and the take of the ellipse by hand (context `ellipse`). Its input
holds, after the joint angles, one value per task: 1 for the task being done and 0 for
the other (context scaling 1, not searched). The scripted take is not used in 3b.

**The search** is 3a's, with both tasks in the score:

1. **The runs.** Each ESN runs on the robot, undisturbed, with the context `reach` from
   each of the nine reach takes' starts for 24 s, and with the context `ellipse` from
   each of the eight starts on the curve for 40 s: 17 runs.
2. **A run passes** if it holds, for a reach (the hand arrives within 20 mm of the
   target and stays within it for 2 s, as in stage 1), or if it cycles, for the ellipse.
3. **A setting's score** is how many of its runs pass, the most first. Settings with as
   many are ranked twice among themselves: by the median detour ratio of their reach
   runs that arrive, and by the median curve RMS of their ellipse runs that cycle. The
   worse of a setting's two ranks decides, the smaller first, so that neither task's
   measure is weighed against the other's in its own unit.
4. **The coarse search, the center, and the fine search** are 3a's: the same 108
   settings with seed 0, and 81 settings around the best with seeds 1 to 5.
5. **Scoring again with the untaught starts.** The 20 fine settings with the best
   scores, each with its five reservoirs again, run from the starts of step 1 and from
   the 36 untaught ones with each context: 89 runs an ESN, scored as in step 3.
6. **The chosen ESNs,** two-task ESN #1 and ESN #2, are the typical seeds of the two
   settings with the best scores in step 5: of a setting's seeds with the most runs
   that pass, the one whose median curve RMS is nearest the setting's.

### The claims

A claim holds if it holds for two-task ESN #1 and for two-task ESN #2.

6. **Each task by its context.** With the context `reach`, from the nine reach takes'
   starts and stage 1's 36 unseen starts, the hand arrives and holds from as many
   starts as it did for report 023's ESN #1 and ESN #2, which is all 45. With the context
   `ellipse`, from the eight starts on the curve and the 36 untaught starts, in runs of
   60 s, the hand cycles at the end from at least as many starts as for the fewer of
   3a's ESN #1 and ESN #2 of the take by hand.
7. **From the periodic motion to the reach.** The run starts with the context
   `ellipse` from each of the eight starts on the curve, and the context changes to
   `reach` at 16 s, which meets a hand that went on from where it started at eight
   places around the curve. The hand arrives at the target and holds, in all eight runs of 40 s.
8. **From the reach to the periodic motion.** The run starts with the context `reach`
   from each of the nine reach takes' starts, and the context changes to `ellipse` at
   20 s, after every reach of stage 1 has arrived and held. The hand cycles at the end
   of the run, in all nine runs of 60 s.
9. **Back and forth.** From each of the eight starts on the curve, the context goes
   from `ellipse` to `reach` at 16 s, back to `ellipse` at 36 s, and to `reach` again
   at 66 s. The hand cycles over the five taught periods before the last change, and
   arrives and holds after it, in all eight runs of 90 s.

The replays keep to their take, so these claims are counts of the ESN's runs alone.

**Two rest postures.** Stage 1's takes 03 and 04 go around the base clockwise and rest
at the target with joint 1 at −246°, a whole turn below the others' 114°. The ellipse
is taught with joint 1 from 34° to 79°. A reach that ends at −246° is then asked to go
to a curve it was taught a whole turn away. What the hand does there is reported with
claim 8.

### The runs

| Claims | What runs |
| --- | --- |
| the search | the coarse search, the fine search, and the scoring with the untaught starts, on both tasks |
| 6 | each two-task ESN with one context for the whole run: `reach` from the nine reach takes' starts and stage 1's 36 unseen ones for 24 s, `ellipse` from the eight starts on the curve and the 36 untaught ones for 60 s (`robot_esn.py`) |
| 7, 8, 9 | each two-task ESN with the changes of context above (`robot_esn.py`, `[[switch]]`) |

## The configurations

Each is listed here when it is written, before its run. Sweeps take `--workers 15`:
the machine is shared.

| Configurations | What they run |
| --- | --- |
| `scripted_coarse_search.toml` | the coarse search on the scripted take: 108 settings, seed 0, each on the robot for 40 s from the eight starts on the curve (about 10 minutes with 15 workers) |
| `scripted_fine_search.toml` | the fine search on the scripted take: 81 settings around the coarse search's best, seeds 1 to 5, alike |
| `coarse_search.toml` | the same on the take by hand; its `run` names the take's import |
| `hand_esn.toml` | an ESN to tune by hand, trained on the take by hand and run on its own from the eight starts on the curve and the 36 untaught ones; it prints the apps' commands |
| `hand_robot.toml` | that ESN on the robot, undisturbed, from the same starts, against the replay (88 runs) |
