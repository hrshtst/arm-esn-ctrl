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

### The takes as recorded (added after the import, 2026-11-18)

The maintainer recorded 27 takes, `take_001` to `take_027`, nine to each target. All
pass the import's checks. They are in three runs in the storage root:
`results/demonstrations/20261118-175915-reach_manual_v2_targets_a`, `…-175916-…_b`, and
`…-175917-…_c`.

| Target | Demonstration | Take | Start (°) | Hand at the start (m) | To the target (m) | Arrives (s) | Length (s) | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | demo_00 | 001 | (13, 71) | (0.53, 0.51) | 1.13 | 8.2 | 18.5 | |
| A | demo_01 | 002 | (−19, 86) | (0.63, 0.21) | 1.24 | 9.1 | 16.8 | |
| A | demo_02 | 003 | (−60, 121) | (0.44, −0.08) | 1.15 | 10.9 | 18.5 | |
| A | demo_03 | 004 | (174, 150) | (−0.17, −0.18) | 0.72 | 7.5 | 19.5 | |
| A | demo_04 | 005 | (38, 46) | (0.43, 0.71) | 1.08 | 7.0 | 16.5 | |
| A | demo_05 | 006 | (83, 44) | (−0.18, 0.81) | 0.59 | 6.0 | 17.3 | |
| A | demo_06 | 007 | (156, 89) | (−0.63, −0.16) | 0.56 | 5.4 | 15.5 | |
| A | demo_07 | 022 | (55, 80) | (0.00, 0.69) | 0.67 | 5.8 | 18.9 | starts at B |
| A | demo_08 | 023 | (−23, 115) | (0.45, 0.20) | 1.07 | 9.4 | 18.8 | starts at C |
| B | demo_00 | 008 | (155, 95) | (−0.59, −0.16) | 1.05 | 7.1 | 17.3 | |
| B | demo_01 | 009 | (150, 138) | (−0.31, −0.13) | 0.89 | 7.6 | 18.5 | |
| B | demo_02 | 010 | (−62, 139) | (0.33, −0.05) | 0.82 | 7.5 | 18.6 | |
| B | demo_03 | 011 | (−18, 88) | (0.61, 0.22) | 0.78 | 7.4 | 18.1 | |
| B | demo_04 | 012 | (16, 53) | (0.63, 0.51) | 0.65 | 5.7 | 16.6 | |
| B | demo_05 | 013 | (36, 37) | (0.53, 0.67) | 0.53 | 5.7 | 17.1 | |
| B | demo_06 | 014 | (75, 57) | (−0.14, 0.78) | 0.16 | 3.2 | 13.7 | |
| B | demo_07 | 024 | (115, 73) | (−0.60, 0.40) | 0.67 | 6.1 | 18.5 | starts at A |
| B | demo_08 | 025 | (−24, 115) | (0.45, 0.20) | 0.67 | 7.6 | 20.1 | starts at C |
| C | demo_00 | 015 | (154, 99) | (−0.57, −0.16) | 1.08 | 9.4 | 20.1 | |
| C | demo_01 | 016 | (147, 136) | (−0.33, −0.12) | 0.84 | 10.6 | 20.1 | |
| C | demo_02 | 017 | (−65, 134) | (0.36, −0.08) | 0.30 | 4.3 | 15.1 | |
| C | demo_03 | 018 | (−14, 65) | (0.73, 0.20) | 0.28 | 3.9 | 15.6 | |
| C | demo_04 | 019 | (16, 61) | (0.57, 0.53) | 0.35 | 4.0 | 16.6 | |
| C | demo_05 | 020 | (34, 60) | (0.39, 0.68) | 0.48 | 6.5 | 17.7 | |
| C | demo_06 | 021 | (76, 57) | (−0.15, 0.78) | 0.83 | 7.7 | 21.8 | |
| C | demo_07 | 026 | (114, 75) | (−0.60, 0.40) | 1.06 | 9.3 | 20.0 | starts at A |
| C | demo_08 | 027 | (56, 78) | (0.00, 0.70) | 0.67 | 6.5 | 19.4 | starts at B |

- **The starts are shared between the targets.** The takes to different targets start
  from about the same seven places, within about 0.15 m, and the last two takes to each target start at
  the other two targets: all six reaches from one target to another are taught. So
  the start posture does not tell the ESN which target a take reaches to.
- **The hand arrives** 3.2 to 10.9 s into a take (median 7.1 s), and every take then
  holds still for 7 s or more. At 4 s, where claim 5 changes the context, the hands of
  three takes have arrived (B's demo_06, C's demo_03 and demo_04); at 16 s, claim 4's
  change, all have.
- **Every take rests at one posture a target:** (114°, 75°) at A, (55°, 79°) at B, and
  (−24°, 115°) at C, within 1.5°. No take goes around the base the other way, so stage
  1's two rest postures for one target do not occur here.

**Joint 2 a whole turn up.** As recorded, takes 004 to 027 have joint 2 between 396°
and 521°, where takes 001 to 003 have 71° to 166°: while the arm was posed after take
003, joint 2 went once around, which the arm without limits allows and which looks the
same, and every later take kept that turn. The ESN reads joint angles, so the same
motion a turn apart would be different data. The maintainer decided, before any run,
that the import moves each take by whole turns so that it starts within −180° to 180°
in every joint (`[recording] start_within_a_turn`): takes 004 to 027 come down by
360° in joint 2, and all 27 then have joint 2 between 37° and 166°. The hand paths are
unchanged, to rounding. A first import without the move
(`…-174509-…_a`, `…-174510-…_b`, `…-174511-…_c`) is in the storage root and is not
used. The recorder was then changed so that leaving the preparation brings each joint
back within a turn.

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

**The center (added after the coarse search, 2026-11-18).** The coarse search ran as
`20261118-181307-coarse_search`, in 26 minutes with 15 workers. Of its 108 settings,
18 hold in all 27 runs, nine at each target; the smallest median detour ratio among
them, 1.0105, is that of leak rate 0.1, spectral radius 1.2, input scaling 2, and
ridge 1. The fine search's values around it:

| Parameter | Values |
| --- | --- |
| leak rate | 0.055, 0.1, 0.17 |
| spectral radius | 1.05, 1.2, 1.35 |
| input scaling | 1, 2, 4 |
| ridge | 0.1, 1, 10 |

The center's input scaling and ridge are the coarse grid's largest, so the fine search
goes as far above them as below. Stage 1's center was leak rate 0.1, spectral radius
1.2, input scaling 0.5, and ridge 1, so the two differ in the input scaling only.

**The fine search, and what step 8 runs (added after the fine search, 2026-11-18).**
The fine search ran as `20261118-184536-fine_search`, in two hours with 15 workers.

- **One of its 81 settings holds in all 135 runs** (five reservoirs, 27 takes' starts
  each): leak rate 0.17, spectral radius 1.05, input scaling 1, ridge 10. Ten settings
  hold in 130 runs or more. The center holds in 119 (39 of 45 at A, 35 at B, 45 at C)
  and is 24th. In stage 1's fine search, 20 of the 81 settings held in every run.
- **C is the easy target.** 39 settings hold in all 45 runs at C, 5 at B, and 2 at A.
- **By parameter,** the mean holding runs of 135: leak rate 0.055, 0.1, 0.17: 116.8,
  103.6, 88.1; input scaling 1, 2, 4: 116.1, 112.8, 79.6; ridge 0.1, 1, 10: 94.9,
  108.4, 105.2; spectral radius 1.05, 1.2, 1.35: 104.9, 102.7, 100.9.
- **Most failing runs reach their target and do not stay the 2 s.** Of the 10,935
  runs, 2,606 fail: in 897 the hand never comes within 20 mm, in 1,693 it does and
  is outside again within the 2 s after, and in 16 it arrives too late for the 2 s to
  be seen; 1,384 of those that leave end within 20 mm. In the 20 best settings 128 of
  2,700 runs fail (69 at A, 55 at B, 4 at C): 5 never arrive, all at A, one arrives
  too late, and 122 arrive and leave, of which 120 end within 11 mm of the target
  (median 4 mm). (The counts of this item were corrected after step 8's run: they
  first left out the runs that arrive too late.)

**Step 8 runs the 20 settings with the best scores.** Its rule names two cases, the
settings whose every run holds and none such. With one such setting, step 9 would
have no second setting to choose from. So step 8 runs the 20 best, as when none
holds; the setting that holds in every run is the first of them. This was the
session's reading of the rule, made before step 8's run. The maintainer confirmed it
while the run was going, before its outcome was read: ESN #1 and ESN #2 are the best
and the second best of step 8's ranking.

**ESN #1 and ESN #2 (added after step 8's run, 2026-11-18).** Step 8 ran as
`20261118-211532-untaught_search`, in 2 hours 40 minutes with 15 workers: 20 settings,
five reservoirs each, 135 runs a reservoir. No setting holds in all 675 runs. The two
best:

| | Leak rate | Spectral radius | Input scaling | Ridge | Runs that hold, of 675 | At A, B, C, of 225 | Median detour ratio | Typical seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ESN #1 | 0.17 | 1.05 | 1 | 10 | 650 | 224, 207, 219 | 1.045 | 1 |
| ESN #2 | 0.17 | 1.2 | 1 | 10 | 639 | 214, 207, 218 | 1.053 | 5 |

- **ESN #1** is the setting that held in every run of the fine search. From the
  takes' starts it holds in all 135 runs again, and from the untaught starts in 515
  of 540: 179 of 180 at A, 162 at B, and 174 at C. Its typical seed, 1, holds in all
  its 135 runs.
- **ESN #2** differs from it in the spectral radius only. It holds in 134 of 135 from
  the takes' starts and in 505 of 540 from the untaught ones: 170 at A, 162 at B, and
  173 at C. Its typical seed, 5, holds in 131 of its 135 runs: all 27 from the takes'
  starts, and 35, 36, and 33 of 36 untaught starts at A, B, and C.
- **The untaught starts fail otherwise than the takes' starts.** Over the 20 settings,
  2,371 of the 10,800 runs from untaught starts fail, and in 1,542 of them the hand
  never comes within 20 mm of its target; from the takes' starts 128 of 2,700 fail,
  and 5 of them never arrive. Of ESN #1's setting's 25 failing runs, 23 never arrive,
  and 21 of these end 1.24 to 1.28 m from the target. One reservoir (seed 2) of each
  of the two settings fails from 15 of the 36 untaught starts at B and from none
  elsewhere.
- **The third setting** (leak rate 0.055, spectral radius 1.35, input scaling 1,
  ridge 10) holds in 626; the last of the 20, in 382.

Both ESNs have stage 1's ESN #2's ridge of 10 and a spectral radius near 1; their
leak rate is the fine grid's largest, where stage 1's two ESNs have 0.055 and 0.1.
The setting second in the fine search (leak rate 0.055, spectral radius 1.35, input
scaling 2, ridge 1; 134 of 135 there) is 14th here, with 533.

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
