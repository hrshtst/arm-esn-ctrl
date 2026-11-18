# Many starts, v2: a search with the pushes in the score (Stages 1 and 2)

[Report 023](../../reports/023-one-target-from-many-starts/README.md) taught one target
from nine start postures and chose two ESNs by undisturbed runs alone. Both arrive from
everywhere and go directly, but under report 007's pushes they end within 20 mm of the
target in one or two of nine runs: the reference goes with the pushed arm, the hand is
sent far away, and it comes back too late for the 24 s. This experiment searches again
with the pushed runs in the score, and tests report 023's seven claims again with the
ESNs it chooses.

**The question.** Does a search that also scores the pushed runs find ESNs that pass
the pushes, and what does that cost undisturbed: in arriving from the unseen starts, in
starting where the arm is, and in going directly?

**Status.** The search's rule and the claims, below, were fixed on 2026-11-18 before
any run of this experiment, but for one check of the configuration: two of the coarse
settings (leak rates 0.03 and 0.1 at spectral radius 1.2, input scaling 0.5, ridge 1)
were run once under the pushes, into a scratch directory.

## What stays as in the many-starts experiment

Everything but the score and the ESNs it chooses is that of
[`manual_demonstration_v2_many_starts`](../manual_demonstration_v2_many_starts/README.md):

- **The robot and the takes:** v2's arm with no joint limits and the nine takes of
  `reach_manual_v2_no_limits.toml`, filtered at 2 Hz.
- **The ESN:** 800 neurons, sparsity 0.05, a bias, a warm-up of 1 s, 10 ms; the leak
  rate, the spectral radius, the input scaling, the ridge, and the seed are searched.
- **The tracker:** report 017's joint PD (ω = 20 rad/s, damping ratio 0.4, given the
  reference's velocity, its torque computed every 10 ms), its gains set where the take
  of reports 005 to 020 ends.
- **The runs:** 24 s on the simulated robot; a run holds if the hand arrives within
  20 mm of the target and stays within it for the 2 s after it arrives.
- **The start postures:** the nine takes', the 36 untaught ones of the scoring, and
  the 36 unseen ones, which no search or scoring runs from.
- **The pushes:** report 007's three, of 0.5 s at the tip: 3 N where the take's hand
  is 18% of the way to the target, 5 N at 62% of the way, and 4 N at 15 s, each in a
  direction fixed for the run by its start.

## The search

1. **The runs.** Each ESN is trained on the nine takes and generates the reference of
   the simulated robot from each take's start, twice: undisturbed, and under the three
   pushes at report 007's forces. That is 18 runs an ESN.
2. **An undisturbed run passes** if it holds.
3. **A pushed run passes** if the hand arrives and ends the run within 20 mm of the
   target. Leaving the goal during the hold is excused, as in report 017 and in claim
   5: a push can carry the arm into the goal early.
4. **A setting's score** is how many of its runs pass, the most first; then the median
   detour ratio of its undisturbed runs that arrive, the smallest first. A pushed
   run's path, which the pushes lengthen, does not count as a detour.
5. **The coarse search:** the many-starts coarse search's 108 settings with one
   reservoir each (seed 0):

   | Parameter | Values |
   | --- | --- |
   | leak rate | 0.01, 0.03, 0.1, 0.3 |
   | spectral radius | 0.9, 1.2, 1.5 |
   | input scaling | 0.1, 0.5, 2 |
   | ridge | 10⁻⁴, 10⁻², 1 |

   Its undisturbed runs are those of the many-starts coarse search
   (`20261117-205256-coarse_search`), which are not run again; only the pushed runs
   are new.
6. **The center** is the coarse setting with the best score.
7. **The fine search:** 81 settings around the center, with five reservoirs each
   (seeds 1 to 5), by the many-starts rule: each parameter takes the center's value
   and the one halfway to its coarse neighbor on each side, in ratio for the leak
   rate, the input scaling, and the ridge (rounded to two digits), in difference for
   the spectral radius; at an edge of the coarse grid, the value beyond the edge is as
   far from the center as the one on the other side. A setting's runs are the 90 of
   its five seeds.
8. **Scoring again with the untaught starts:** the fine search's 20 best settings,
   each with its five reservoirs again, undisturbed from each take's start and from
   the 36 untaught ones. A setting's score is then over those 225 undisturbed runs
   and its 45 pushed runs of the fine search: 270 runs.
9. **The chosen ESNs,** ESN #1 and ESN #2 of this experiment, are the typical seeds
   of the two settings with the best scores of step 8. A setting's typical seed is, of
   its seeds with the most passing runs, the one whose own median detour ratio is
   nearest the setting's.

Report 023's chosen ESNs are called report 023's ESN #1 and ESN #2 here.

**What was known when the rule was fixed.** The many-starts coarse search's undisturbed
scores: 30 of the 108 settings hold from all nine starts, the best at leak rate 0.1,
spectral radius 1.2, input scaling 0.5, ridge 1, and the second at leak rate 0.03,
spectral radius 1.2, input scaling 2, ridge 1. Report 023's ESN #1 and ESN #2 under the
pushes, which follow the pushed arm by 0.94 to 0.96. And report 017's result for one
take with joint limits: the following rises with the leak rate, and the ESNs pass the
pushes most at leak rates 0.015 to 0.03.

`score_sweeps.py` scores a step from its two sweeps:

```bash
uv run python experiments/sweep_robot_esn.py \
    experiments/manual_demonstration_v2_many_starts_push_search/coarse_pushes.toml --workers 30
uv run python experiments/score_sweeps.py \
    experiments/manual_demonstration_v2_many_starts_push_search/coarse_score.toml
```

The scoring configurations name the runs they score, so each is written when its runs
exist; the fine search's and the rescoring's sweeps are written when the step before
gives the center and the 20 settings.

**The center.** Of the coarse search's 108 settings, 30 hold undisturbed from all nine
starts, eight pass the pushes from all nine, and five do both
(`20261118-110123-coarse_pushes`, `20261118-111042-coarse_score`). The smallest median
detour ratio among the five, 1.0135, is that of leak rate 0.03, spectral radius 0.9,
input scaling 2, and ridge 1. The pushed runs that pass, of 243 at each leak rate, are 94
at 0.01, 81 at 0.03, 27 at 0.1, and 11 at 0.3. The fine search's values around the
center:

| Parameter | Values |
| --- | --- |
| leak rate | 0.017, 0.03, 0.055 |
| spectral radius | 0.75, 0.9, 1.05 |
| input scaling | 1, 2, 4 |
| ridge | 0.1, 1, 10 |

The center's spectral radius is the coarse grid's smallest, and its input scaling and
ridge the largest, so the fine search goes as far beyond each as on the other side.

```bash
uv run python experiments/sweep_robot_esn.py \
    experiments/manual_demonstration_v2_many_starts_push_search/fine_search.toml --workers 30
uv run python experiments/sweep_robot_esn.py \
    experiments/manual_demonstration_v2_many_starts_push_search/fine_pushes.toml --workers 30
```

**The one-take center.** Of the 108 one-take settings, 67 hold undisturbed from take
00's start, 45 pass the pushes, and 44 do both (`20261118-110940-one_take_coarse_pushes`,
`20261118-111042-one_take_coarse_score`). The best of the 44 is the many-starts one-take
search's center: leak rate 0.03, spectral radius 0.9, input scaling 0.1, ridge 1. So the
one-take fine search's undisturbed runs are those of the many-starts one-take fine
search (`20261118-012638-one_take_fine_search`), not run again; only its pushed runs
are new.

**The fine search.** Of its 81 settings, 36 hold undisturbed in all 45 runs, 15 pass the
pushes in all 45, and six do both (`20261118-114046-fine_search`,
`20261118-111415-fine_pushes`, `20261118-120351-fine_score`). The pushed runs that pass,
of 1,215 at each leak rate, are 1,147 at 0.017, 889 at 0.03, and 391 at 0.055. The 20th
best setting passes 86 of its 90 runs.

**ESN #1 and ESN #2.** Scored again with the untaught starts
(`20261118-120421-untaught_search`, `20261118-125450-untaught_score`), one of the 20
settings passes all 270 runs; the others fail undisturbed from untaught starts, pushed, or
both, and the worst passes 248. The typical seeds of the two best:

| ESN | Leak rate | Spectral radius | Input scaling | Ridge | Seed | Runs that hold undisturbed | Pushed runs that pass | Median detour ratio of the setting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| #1 | 0.03 | 0.9 | 2 | 1 | 5 | 225 of 225 | 45 of 45 | 1.090 |
| #2 | 0.017 | 1.05 | 4 | 10 | 5 | 221 of 225 | 45 of 45 | 1.278 |

ESN #1's setting is the fine search's center. Their configurations train them and run
them on their own, as the many-starts experiment's do:

```bash
uv run python experiments/autonomous_esn.py \
    experiments/manual_demonstration_v2_many_starts_push_search/esn1_lr0.03_sr0.9_is2_ridge1_seed5.toml
uv run python experiments/autonomous_esn.py \
    experiments/manual_demonstration_v2_many_starts_push_search/esn2_lr0.017_sr1.05_is4_ridge10_seed5.toml
```

**The one-take ESNs.** Of the one-take fine search's 81 settings, 70 hold undisturbed in
all five runs, 36 pass the pushes in all five, and 32 do both
(`20261118-111145-one_take_fine_pushes`, `20261118-120352-one_take_fine_score`). Scored
again with the untaught starts (`20261118-123329-one_take_untaught_search`,
`20261118-125450-one_take_untaught_score`), every one of the 20 passes its five pushed
runs, and one holds in all 185 undisturbed runs:

| ESN | Leak rate | Spectral radius | Input scaling | Ridge | Seed | Runs that hold undisturbed | Pushed runs that pass | Median detour ratio of the setting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one-take #1 | 0.017 | 1.05 | 0.045 | 0.1 | 5 | 185 of 185 | 5 of 5 | 3.08 |
| one-take #2 | 0.017 | 1.05 | 0.1 | 0.1 | 1 | 183 of 185 | 5 of 5 | 2.57 |

One-take ESN #1 is the many-starts one-take ESN #1: the same setting and seed.

## The analysis

ESN #1 and ESN #2 are tested as report 023's were, on the robot with the search's
tracker, against the replay of a take by time and against ESNs trained on fewer takes.
The measures, the unseen starts, the sets of takes, and the seven claims are those of
the many-starts experiment, word for word, so that the outcomes compare. No size is
fixed in advance: every claim is a count or a direction, and the sizes are reported. A
claim holds if it holds for ESN #1 and for ESN #2.

**The one-take baseline** is trained on take 00 alone and chosen as ESN #1 and ESN #2
are, by this experiment's rule: the coarse search (seed 0; its undisturbed runs are
the many-starts one-take coarse search's, `20261118-005826-one_take_coarse_search`),
the fine search around its best (seeds 1 to 5), and the fine search's 20 best settings
scored again with the 36 untaught starts. Its runs start from take 00's start,
undisturbed and under the pushes. The typical seeds of its two best settings are
one-take ESN #1 and one-take ESN #2.

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

Report 023 found that claims 6 and 7 fail as stated, because an ESN of one take
already arrives from everywhere; they are kept as stated here. The search scores the
pushes at report 007's forces only, so twice and four times those of claim 5 are sizes
no ESN was chosen by.

**Reported without a claim:**

- **A longer run.** The pushes at report 007's forces run for 48 s, as report 023's
  were afterwards: when each hand is in the goal for good.
- **Against report 023's ESNs.** The measures of claims 2 to 5 are set beside those
  report 023 gives for its ESN #1 and ESN #2, and the following gain under each push
  beside theirs.
- **The search's map.** How the pushed runs' passing and the following gain depend on
  the four parameters, over the coarse and the fine search.

### The runs

| Claims | Configurations | What they run |
| --- | --- | --- |
| 1 to 4 | `robot_esn1.toml`, `robot_esn2.toml` | ESN #1 and ESN #2 against the replay, undisturbed, from the nine takes' starts and the 36 unseen ones (`robot_esn.py`; 90 runs, about 8 minutes each) |
| 5 | `pushes1x_robot_esn1.toml`, `pushes2x_…`, `pushes4x_…`, and those of ESN #2 | ESN #1 and ESN #2 against the replay and the progress replay under the three pushes at 1, 2, and 4 times their forces, from each take's start (`robot_esn.py`; 27 runs, about 2 minutes each) |
| 6 | the one-take search's configurations (`one_take_…`), the two chosen one-take ESNs', and `robot_one_take_esn1.toml`, `robot_one_take_esn2.toml` | the one-take baseline's search, and one-take ESN #1 and ESN #2 against the replay of take 00, undisturbed, from take 00's start, the other eight takes' starts, and the 36 unseen ones (`robot_esn.py`; 90 runs, about 8 minutes each) |
| 7 | `takes1.toml`, `takes3.toml`, `takes6.toml`, `takes9.toml` | ESN #1's and ESN #2's settings, seeds 1 to 5, trained on one, three, six, and nine takes, each ESN on the robot, undisturbed, from every take's start and the 36 unseen ones (`sweep_robot_esn.py`; about 4 minutes each with 30 workers) |
| none | `pushes1x_long_robot_esn1.toml`, `pushes1x_long_robot_esn2.toml` | the pushes at report 007's forces, run for 48 s |

These are written as the search gives the ESNs, as copies of the many-starts
experiment's with the new ESNs.

## The configurations

| Configurations | What they run |
| --- | --- |
| `coarse_pushes.toml` | the coarse search's pushed runs: 108 settings, seed 0, each on the robot under the pushes from each take's start (about 7 minutes with 30 workers) |
| `one_take_coarse_pushes.toml` | the one-take baseline's alike, trained on take 00 alone and run from its start (about 2 minutes) |
| `coarse_score.toml`, `one_take_coarse_score.toml` | the coarse searches' scores, over the many-starts coarse searches' undisturbed runs and the pushed runs above |
| `fine_search.toml`, `fine_pushes.toml` | the fine search: 81 settings around the coarse score's best, seeds 1 to 5, each on the robot from each take's start, undisturbed and under the pushes (about 25 minutes each) |
| `one_take_fine_pushes.toml` | the one-take fine search's pushed runs: 81 settings around its coarse score's best, seeds 1 to 5, from take 00's start (about 3 minutes) |
| `fine_score.toml`, `one_take_fine_score.toml` | the fine searches' scores |
| `untaught_search.toml`, `one_take_untaught_search.toml` | the fine searches' 20 best settings, seeds 1 to 5, undisturbed from the takes' starts (take 00's) and from the 36 untaught ones (about 30 and 25 minutes) |
| `untaught_score.toml`, `one_take_untaught_score.toml` | their scores, over those undisturbed runs and the fine searches' pushed runs |
| `esn1_lr0.03_sr0.9_is2_ridge1_seed5.toml`, `esn2_lr0.017_sr1.05_is4_ridge10_seed5.toml` | ESN #1 and ESN #2, the typical seeds of the two best settings of `untaught_score.toml`, trained and run on their own as the many-starts ones |
| `one_take_esn1_lr0.017_sr1.05_is0.045_ridge0.1_seed5.toml`, `one_take_esn2_lr0.017_sr1.05_is0.1_ridge0.1_seed1.toml` | one-take ESN #1 and ESN #2, the typical seeds of the two best settings of `one_take_untaught_score.toml`, trained on take 00 and run on their own from its start and the 36 untaught ones |
| `takes1.toml`, `takes3.toml`, `takes6.toml`, `takes9.toml` | ESN #1's and ESN #2's settings, seeds 1 to 5, trained on one, three, six, and nine takes, each on the robot from every take's start and the 36 unseen ones (about 4 minutes each) |
