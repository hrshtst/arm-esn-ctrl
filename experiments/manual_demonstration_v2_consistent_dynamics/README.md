# Manual-demonstration consistent dynamics, v2 (Stages 1 and 2)

The third part of the study run under one protocol, which
[the search's README](../manual_demonstration_v2_consistent_search/README.md#the-protocol)
holds with this part's plan
([Part 3](../manual_demonstration_v2_consistent_search/README.md#part-3-the-esns-as-dynamical-systems-report-019)):
report 015's analyses of the two ESNs that [the search](../manual_demonstration_v2_consistent_search/README.md)
chose ([report 017](../../reports/017-search-under-one-protocol/README.md)), and report 016's
screen of all the search's ESNs, set against their outcomes on the robot with the offset
starts counted. Results: [report 019](../../reports/019-dynamics-under-one-protocol/README.md),
which also closes the study with the claims of reports 011 to 016.

- **The ESNs:** ESN #1 (leak rate 0.03, spectral radius 1.2, input scaling 1.5,
  seed 8) and ESN #2 (input scaling 1, seed 2), as the search trained them
  (`results/manual_demonstration_v2_consistent_search/20261024-205825-esn1_lr0.03_sr1.2_is1.5_seed8`
  and `20261024-205835-esn2_lr0.03_sr1.2_is1_seed2`); and the search's 5,040 ESNs
  (seeds 0 to 9) and 5,040 held out (seeds 10 to 19), trained as its sweeps trained
  them.
- **The tracker,** wherever the robot runs: the search's, joint PD at ω = 20 rad/s and
  ζ = 0.4, given the reference's velocity, its torque computed every 10 ms and held.
- **What changes from reports 015 and 016:** the ESNs; the runs from rest 60 s long and
  the warm-up runs 24 s (report 015: 20 s); the pulses' runs 24 s (20 s); the grid of
  504 settings (105); and the outcomes on the robot, the search's, with every ESN tried
  from the 49 offset starts (report 016: report 011's, without them). Everything else is
  as those reports configured it.

Every step below is fixed before its runs; a step that reads another's run names it in a
configuration written once that run has run.

## Step 1: from rest anywhere

[`global_starts.py`](../global_starts.py), report 015's step 1: each ESN on its own, fed
its own output, from 1,296 start postures every 10° over both joints' range, each held
for the 1 s warm-up first, for 60 s; and each start posture held 4 s.

```bash
uv run python experiments/global_starts.py experiments/manual_demonstration_v2_consistent_dynamics/global_starts.toml
```

## Step 2: the warm-up

[`warmup_esn.py`](../warmup_esn.py), report 015's step 1b: each ESN with warm-ups of
0.25, 0.5, 1, 2, and 4 s, from the 49 offset starts and three starts on the take's path,
for 24 s.

```bash
uv run python experiments/warmup_esn.py experiments/manual_demonstration_v2_consistent_dynamics/warmup_esn1_lr0.03_sr1.2_is1.5_seed8.toml
```

## Step 3: pushed along the reach

[`push_pulses.py`](../push_pulses.py), report 015's step 2: on the robot, each ESN, the
replay, and the progress replay, pushed at the tip by one pulse a run, 3 N or 6 N for
0.2 s, every second from 1 s to 9 s and at 12 s, in 8 directions (160 pulses and one
undisturbed run per arm), in runs of 24 s.

```bash
uv run python experiments/push_pulses.py experiments/manual_demonstration_v2_consistent_dynamics/push_pulses.toml --workers 16
```

## Step 4: where they can rest

[`fixed_points.py`](../fixed_points.py), report 015's step 3, from step 1's run: each
ESN's fixed points by Newton's method from the states of its runs, their eigenvalues,
the held reservoir's fixed point at each start posture, and the rates along its run
from the take's start over windows of 0.5, 1, 2, and 4 s.

```bash
uv run python experiments/fixed_points.py experiments/manual_demonstration_v2_consistent_dynamics/fixed_points.toml
```

| Configuration | What it runs |
| --- | --- |
| `fixed_points.toml` | ESN #1 and ESN #2, from the states of step 1's run `20261027-180441-global_starts` |

## Step 5: the screen

[`screen_esns.py`](../screen_esns.py), report 016's step 1 with its `screen.toml`
unchanged but for the ESNs: each of the search's ESNs on its own, its fixed points from
the zeros of R(q) every 10° and where its runs from rest come to rest, the lead |R| near
the path, runs from rest every 30° and around the take's end posture for 60 s, runs held
4 s at each unstable fixed point, and the rates along its run from the take's start.

```bash
uv run python experiments/screen_esns.py experiments/manual_demonstration_v2_consistent_dynamics/screen.toml --workers 30
```

| Configurations | What they run |
| --- | --- |
| `screen.toml` | the search's 5,040 ESNs, seeds 0 to 9 |
| `screen_seeds10to19.toml` | its 5,040 held-out ESNs, seeds 10 to 19 |

The held-out screen's first run stopped at its 4,609th ESN (leak rate 0.3, spectral
radius 1.75, input scaling 1.5, seed 18), whose run from the take's start comes to rest
on a fixed point to the last bit: the rates across a run at rest were 0/0, and are now
those of every change. It ran again (`20261028-064044-screen_seeds10to19`). Of the 4,608
ESNs both runs screened, 19, all at spectral radii of 1.5 and 1.75, whose runs from rest
are chaotic, print a different count of fixed points or of starts ending at the target:
the ESNs, trained again, agree with the first run's to rounding, not bit for bit.

## Step 6: what the screen predicts

[`screen_against_robot.py`](../screen_against_robot.py), report 016's steps 2 and 4: each
screen against the search's sweeps of the same ESNs (part 1's step 1), with report
016's rules and thresholds:

- **(a)** an attracting fixed point whose hand is outside the goal radius, against
  ending away from the target under the pushes;
- **(b)** a median lead near the path under 10°, against the same;
- **(c)** an unstable fixed point within 30° of the take's joint path, or arriving later
  than the take's 9.58 s after a 4 s hold at one, against never arriving;
- **(d)** the rates along the run, against the return and the following.

An ESN's outcome is the search's rule's first point: it fails if it fails undisturbed,
under the block, under the pushes, or from any of the 49 offset starts, and it never
arrives if it never arrives in one of those runs (`screen_against_robot.py`, given the
offset sweeps, counts them; over the search's sweeps it gives the rule's 1,049 and 1,101
passing ESNs exactly). The qualifying settings are the search's: its 11 for seeds 0 to
9, and the 8 that qualify with seeds 10 to 19 for those. The named ESNs: ESN #1 and
ESN #2, and report 016's ESN #3 (leak rate 0.03, spectral radius 1.35, input scaling 1.5,
seed 6) and report 007's (0.02, 1.35, 2, seed 7).

```bash
uv run python experiments/screen_against_robot.py experiments/manual_demonstration_v2_consistent_dynamics/against_robot.toml
```

| Configuration | What it runs |
| --- | --- |
| `against_robot.toml` | step 5's screen of seeds 0 to 9, `20261027-183607-screen`, against the search's 48 sweeps of the same ESNs (its `select.toml`'s runs) |
| `against_robot_seeds10to19.toml` | step 5's screen of seeds 10 to 19, `20261028-064044-screen_seeds10to19`, against the search's 48 sweeps of the same ESNs (its `select_seeds10to19.toml`'s runs) |

The claims, report 016's of its held-out ESNs, now of both sets, over the 12 leak rates,
where a leak rate has both passing and failing ESNs among those counted:

1. **(a)** at each leak rate, the share failing is larger among the flagged ESNs than
   among the unflagged;
2. **(d)** within each leak rate from 0.015 to 0.05, the largest scaling of a posture's
   deviation has a rank correlation of at least 0.5 with the following;
3. an unstable fixed point nearer the path tells the failing ESNs apart, with an area
   under the ROC curve above 0.5 within each leak rate (of the ESNs with one);
4. over all the ESNs, no rule's flag, (a), (b), or (c), nor the lead, has a larger area
   under the ROC curve for failing than the leak rate alone.

## Step 7: the passing ESNs held still on the robot

[`release_robot.py`](../release_robot.py), report 016's step 3: the passing ESNs of
seeds 0 to 9 (1,049), as step 6 counts them, with the search's tracker:

- **released** at each unstable fixed point within 30° of the take's joint path, the
  arm at rest there and the ESN held there 4 s, then let go for 60 s;
- **blocked** from the take's start, the tip held for 1 s or for 3 s from each of 2, 3,
  4, 5, 6, 7, and 8 s, one block a run, for 30 s.

```bash
uv run python experiments/release_robot.py experiments/manual_demonstration_v2_consistent_dynamics/release_robot.toml --workers 30
```

| Configuration | What it runs |
| --- | --- |
| `release_robot.toml` | the passing ESNs of step 6's run `20261028-044632-against_robot`, trained as step 5's screen `20261027-183607-screen` trained them |

Shown over time, as report 016 showed them,
[`robot_esn.py`](../robot_esn.py) runs ESN #1 and ESN #2 each against the replay and the
progress replay, with the search's tracker, under two of the blocks and released at
ESN #2's unstable fixed point beside the path:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_consistent_dynamics/released_robot_esn2_lr0.03_sr1.2_is1_seed2.toml
```

| Configurations | What they run |
| --- | --- |
| `{block5to8,block8to11}_robot_{esn1_lr0.03_sr1.2_is1.5_seed8,esn2_lr0.03_sr1.2_is1_seed2}.toml` (4) | one ESN and both replays, the tip held from 5 s to 8 s or from 8 s to 11 s, from the take's start, for 30 s |
| `released_robot_{esn1_lr0.03_sr1.2_is1.5_seed8,esn2_lr0.03_sr1.2_is1_seed2}.toml` (2) | one ESN and both replays, the arm at rest at ESN #2's unstable fixed point beside the path, (39.85°, 97.12°), the ESN held there 4 s, then let go for 30 s |

The claims, report 016's findings of its step 3:

1. released, the robot does what the ESN does on its own: where both arrive, the rank
   correlation of the robot's arrival with the ESN's own is at least 0.9;
2. under the blocks, the ESNs with an unstable fixed point within 30° of the path fail
   no more often than those without, with either hold;
3. the delay after a block grows as the held posture nears an unstable fixed point: its
   rank correlation with the distance to the nearest one is negative, with either hold.

## Step 8: the closing table

Report 019 closes the study: every main claim of reports 011 to 016, confirmed, changed,
or not tested by reports 017 to 019.
