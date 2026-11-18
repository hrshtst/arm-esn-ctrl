# Several targets by context, v2 (Stage 2 of the multi-task study)

Stage 1 taught one target from many start postures
([`manual_demonstration_v2_many_starts`](../manual_demonstration_v2_many_starts/README.md),
reports 023 and 024). This experiment is the second stage: one ESN reaches three
targets, on the same arm with no joint limits. The target is named by a context added
to the ESN's input, not by its coordinates. Its ESNs have 800 neurons.

**Status.** The steps, the search's rule, and the claims below were fixed on 2026-11-18,
before any take was recorded and before any run. On the same day, after the takes were
recorded and before any run, the maintainer revised the claims: two were dropped and
one became a demonstration ([below](#the-revision-of-the-claims)). What is added later
is labelled so.

## The robot, the targets, and the takes

[`reach_manual_v2_targets.toml`](../demonstrations/reach_manual_v2_targets.toml) holds
the robot and the three targets. The robot is stage 1's: the arm of reports 005 to 020
with no joint limits.

| Target | Hand (m) | Rest posture, joint 2 positive (°) |
| --- | --- | --- |
| A | (−0.6, 0.4), stage 1's | (114.1, 74.0) |
| B | (0, 0.7) | (56.0, 78.5) |
| C | (0.45, 0.2) | (−23.6, 114.8) |

A and B, and B and C, are 0.67 m apart; A and C 1.07 m. Each has a goal radius of
20 mm.

**All takes are new, and neither their number nor their start postures are fixed in
advance.** The maintainer teaches by hand in skelarm's recorder, one take after
another in one session. Before each take, in the recorder's preparation phase, he
poses the arm where the take is to start and chooses which of the three targets it
reaches to; the take records that choice. Stage 1's nine takes to A are not used.

The import reads every take, takes each one's target from the take itself, checks it
as stage 1's takes were checked (it arrives at its target, holds 2 s, and never
jumps), and writes one demonstration run per target:

```bash
uv run python experiments/import_demonstrations.py experiments/demonstrations/reach_manual_v2_targets.toml
```

The configuration's header gives the recorder's command. The takes that were
recorded, their starts, and their number per target are reported here after the
import, before the search.

## The context

The ESN's input holds, after the two joint angles, one value per target: 1 for the
target to reach and 0 for the other two (A = (1, 0, 0), B = (0, 1, 0), C = (0, 0, 1)).
Nothing else tells the ESN which target a take reaches to, and its output is the next
joint angles alone.

**The context's size does not follow the input scaling.** The joint angles enter the
reservoir through input weights multiplied by the input scaling. The context is fed
in divided by the input scaling, so that its weights have the scale 1 whatever the
input scaling is (`context_scaling = 1.0`, `context_times_input_scaling = false`).
This scale is not swept. If the search finds no setting that works, sweeping it is
the next thing to try, and is the maintainer's to decide.

## The search

The search is stage 1's (report 023's), with each run given a context. It chooses two
ESNs by how reliably the hand arrives and holds at the target its context names, and
then by how directly it goes there.

1. **The runs.** Each ESN is trained on all takes of the three targets and generates
   the reference of the simulated robot, undisturbed, from each take's start with the
   context of that take's target: one run of 24 s per take, with the consistent
   search's tracker (report 017's: joint PD at ω = 20 rad/s and a damping ratio of
   0.4, given the reference's velocity, its torque computed every 10 ms, its gains
   set where the take of reports 005 to 020 ends).
2. **A run holds** if the hand arrives within 20 mm of the target its context names
   and stays within it for the 2 s after it arrives. Nothing later is required.
3. **The detour ratio** of a run that arrives is the length of the hand's path until
   it arrives, over the straight distance from its start to that target.
4. **A setting's score** is how many of its runs hold, over the three targets
   together, the most first; then the median detour ratio of its runs that arrive,
   the smallest first. Each target's holding runs are reported beside it.
5. **The coarse search:** stage 1's 108 settings, with one reservoir each (seed 0):

   | Parameter | Values |
   | --- | --- |
   | leak rate | 0.01, 0.03, 0.1, 0.3 |
   | spectral radius | 0.9, 1.2, 1.5 |
   | input scaling | 0.1, 0.5, 2 |
   | ridge | 10⁻⁴, 10⁻², 1 |

   The rest is `coarse_search.toml`'s `[esn]`: 800 neurons, sparsity 0.05, a bias, a
   warm-up of 1 s, 10 ms, and the context as above. The grid is stage 1's, so the two
   stages' centers can be compared; report 023's ESN #1 and ESN #2 lie within it but
   for ESN #2's ridge of 10.
6. **The center** is the coarse setting with the best score.
7. **The fine search:** 81 settings around the center, with five reservoirs each
   (seeds 1 to 5). Each parameter takes three values: the center's, and the one
   halfway to its coarse neighbor on each side, in ratio for the leak rate, the input
   scaling, and the ridge (rounded to two digits), and in difference for the spectral
   radius. Where the center is at an edge of the coarse grid, the value beyond the
   edge is as far from the center as the one on the other side.
8. **Scoring again with untaught starts.** The fine settings whose every run holds,
   at most the 20 with the best scores (the 20 best if none holds in every run), are
   run again with their five reservoirs: from each take's start as before, and from
   stage 1's 36 untaught start postures with each of the three contexts (`tools/tip_starts.py
   <takes> --counts 18 12 6 --first-angle 53.3`; none is nearer than 0.085 m to a
   target). The score is the same, over all these runs.
9. **ESN #1 and ESN #2** are the typical seeds of the two settings with the best
   scores of step 8. A setting's typical seed is, of its seeds with the most runs
   that hold, the one whose own median detour ratio is nearest the setting's.

The search has no pushes and no change of the context during a run.

**If three targets do not work:** if step 8 finds no setting whose runs hold for all
three targets, the outcome is reported as it is, and the maintainer decides between
sweeping the context's scale and falling back to two targets.

```bash
uv run python experiments/sweep_robot_esn.py \
    experiments/manual_demonstration_v2_targets/coarse_search.toml --workers 15
uv run python experiments/sweep_robot_esn.py \
    experiments/manual_demonstration_v2_targets/fine_search.toml --workers 15
uv run python experiments/sweep_robot_esn.py \
    experiments/manual_demonstration_v2_targets/untaught_search.toml --workers 15
```

## The analysis

ESN #1 and ESN #2 are tested on the robot with the search's tracker. No size is fixed
in advance: every claim is a count or a direction, and the sizes are reported. A claim
holds if it holds for ESN #1 and for ESN #2.

**The measures** are stage 1's. A run holds as in the search, at the target its
context names. Its detour ratio is the search's. How a run starts is measured by the
reference's fastest joint speed and by the run's largest joint torque.

**The unseen starts** are stage 1's 36 (`tools/tip_starts.py <takes> --counts 18 12 6
--first-angle 38.3`), joint 2 positive; none is nearer than 0.067 m to a target. No
search or scoring of this stage runs from them.

**The replay** of a run is the replay by time of the take, among those of the target
the run's context names, that starts nearest the run's start.

### The claims

1. **The takes' starts.** Undisturbed, from each take's start with the context of its
   target, the hand arrives and holds at that target, as the replay's does.
2. **The unseen starts.** From the 36 unseen starts with each of the three contexts,
   the hand arrives and holds at the target the context names from as many starts as
   the replay's, for each target.
3. **Starting where the arm is, and going directly.** From the 36 unseen starts, for
   each target, the reference's fastest joint speed, the largest joint torque, and the
   detour ratio are smaller than the replay's, in their medians over the starts.
4. **Changing the context at rest.** From each take's start with the context of its
   target, the context changes at 16 s to each of the other two targets, and the run
   goes on for 24 s more. In every such run the hand arrives and holds at the new
   target.
5. **Changing the context on the way.** As claim 4, with the change at 4 s, while the
   hand is on its way. In every such run the hand arrives and holds at the new
   target.

### A demonstration, without a claim

**Pushes by hand while the target changes.** The maintainer runs ESN #1 in the robot
app (`tools/robot_app.py`, with the search's tracker), pushes the arm by dragging its
tip, and changes the target with the app's keys, as he likes. The app saves the run
as a log, with the force and the context at every step (`Ctrl+S`). The report shows
that one run: the hand's path, the joint angles and torques, the force, and the
context over time. Nothing is counted and nothing is compared.

### What is reported without a claim

- Where each run of claims 4 and 5 is when the context changes, how long it takes to
  the new target from the change, and the reference's fastest joint speed and the
  largest joint torque after the change.
- How far the chosen settings are from stage 1's (report 023's ESN #1 and ESN #2).

### The revision of the claims

The claims were first fixed with eight. On 2026-11-18, after the takes were recorded
and before any run, the maintainer revised them to the five above:

- **Dropped: the context names the target, not the start** (from each take's start
  with each of the other two contexts, the hand goes to the target the context names;
  and no run ends at a target its context does not name). Claim 2 covers it: every
  unseen start is run with all three contexts.
- **Made a demonstration: pushes** (report 007's three pushes at one, two, and four times their
  forces over 60 s, against the replay and the progress replay). Comparing arms under
  one schedule of forces is not what this stage is to show; the demonstration above
  takes its place.
- **Dropped: three targets against one** (for each target, as many unseen starts
  held as with an ESN trained on that target's takes alone). An ESN of one target
  cannot reach the other two, so the comparison shows nothing about the context.
- With them, the ESNs of one target and the ESN with no context, which were to be
  reported beside those claims, are not trained.

### The runs

The configurations are written as each step gives the next: the takes' runs, the
center, and the chosen ESNs are not known yet.

| Claims | Configurations | What they run |
| --- | --- | --- |
| the search | `coarse_search.toml`, `fine_search.toml`, `untaught_search.toml` | steps 5, 7, and 8 (`sweep_robot_esn.py`) |
| | `esn1_….toml`, `esn2_….toml` | ESN #1 and ESN #2, trained and run on their own; they save `esn.toml` for the apps (`autonomous_esn.py`) |
| 1 to 3 | `robot_esn1.toml`, `robot_esn2.toml` | each context from the starts of its target's takes and from the 36 unseen starts, against the replay, undisturbed (`robot_esn.py`) |
| 4 | `switch_to_a_robot_esn1.toml`, `switch_to_b_…`, `switch_to_c_…`, and those of ESN #2 | the context changes at 16 s to A, to B, or to C, from each take's start; 40 s |
| 5 | `switch_early_to_a_robot_esn1.toml`, …, and those of ESN #2 | the same with the change at 4 s; 28 s |
| the demonstration | the robot app, with `esn1_….toml`'s `esn.toml` | one run pushed by hand while the target changes, saved as a log |

In a run of claims 4 and 5 whose change names the context it already has, nothing
changes; those runs are kept as they come and are not part of the claims.
