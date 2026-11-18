# The multi-task study: planned stages

Reports 017 to 020 taught one reach from one start posture. This study extends them to
more tasks on the same two-link arm, to show that one framework handles many tasks while
the demonstrator teaches motions only. This note gives the stages as they are thought of
now and where the study stands; each stage's own rule, claims, and runs are in its
experiment's README.

**This note is a map, not an instruction.** What the maintainer asks for in a session
decides what is done, whether or not it is in this note. Stages can be reordered, changed,
dropped, or added, and work outside the stages is normal. When that happens, the note is
brought up to date afterwards; nothing waits for it. An agent that finds the request and
this note apart follows the request and says so.

Last updated: 2026-11-18.

## What the stages have in common, so far

These are the choices made up to now. They are defaults for the next stage, not
conditions on it.

- **The ESN learns joint-position demonstrations with no task knowledge,** is fed the
  arm's measured joint angles, and generates the reference of the simulated robot.
- **The robot** is the two-link arm of reports 005 to 020 without joint limits
  (`experiments/demonstrations/reach_manual_v2_no_limits.toml`).
- **The tracker** is report 017's only: joint PD at ω = 20 rad/s and a damping ratio of
  0.4, given the reference's velocity, its torque computed every 10 ms.
- **ESNs have 800 neurons.** Each stage searches its own setting, coarse and then finer
  around what the coarse search finds. The setting may differ from stage to stage, and
  how far it differs is reported.
- **Reaching takes are taught by hand.** No take is held out: from a start no take starts
  from, a run succeeds if the hand ends and stays at the commanded target, whatever path
  it takes.
- **Start postures are laid out by the hand,** spread evenly over the workspace
  (`tools/tip_starts.py`), for recording and for evaluation.
- **No size is fixed in advance.** A claim is a count or a direction against a baseline
  (the replay, an ESN of fewer takes), and the sizes are reported.

## How a stage has gone, usually

1. Its steps and claims are written into its experiment's README and committed.
2. The maintainer records the takes it needs.
3. The search, then the runs that test the claims.
4. A report, with the next free number. Negative results are reported as they are.
5. The maintainer looks at it before the next stage starts.

Only the first step is a rule (it is in [CLAUDE.md](CLAUDE.md)): what a run is to show
is written down before the run. The rest is a habit that the maintainer changes as needed.

## The stages

Stages 0 and 1 are a record of what was done. Stages 2 to 4 are the present thinking: a
sentence or two each, to be worked out, and changed, when a stage starts.

**Stages 2 and 3 are worked on side by side,** each by its own session in its own git
worktree and branch, made from the latest commit of `main`. Each searches its own ESN
setting, as every stage does, so neither waits for stage 1's open questions. The
maintainer brings the two branches together.

| Stage | What it adds | Status |
| --- | --- | --- |
| 0 | the code every later stage needs | implemented |
| 1 | one target from many start postures | reports 023 and 024 written and reviewed; its open questions are deferred |
| 2 | several targets, named by a one-hot context | report 025 written, awaiting review |
| 3 | a periodic motion, and switching between it and reaching | next, beside stage 2 |
| 4 | all tasks in one ESN | not started |

### Stage 0: the code

A context input for the ESN, the import of several takes, runs from many starts, the
detour measures, pushes placed along each take and scaled, the figure of the starts, the
measures of a periodic motion, and a progress replay on a closed path. Checked against
report 017: its ESN #1's runs are reproduced bit for bit.

Left for the stage that needs it:

- the periodic measures and the looping replay are not wired into a runner (stage 3);
- the sweep and the dynamics runners take one task (stage 2);
- there are no demonstration configurations for stages 2 and 3;
- skelarm's recorder draws no curve to follow (stage 3, upstream);
- `robot_starts.py` and `global_starts.py` refuse ESNs of several takes.

### Stage 1: one target from many start postures

Nine takes of the reach to (−0.6, 0.4) m, one from each of nine start postures. Tested
from 36 unseen starts, under report 007's pushes at one, two, and four times their
forces, against an ESN of one take, and against one, three, six, and nine takes.

- **[Report 023](reports/023-one-target-from-many-starts/README.md):** ESNs chosen by
  undisturbed runs arrive from everywhere, start where the arm is, and go directly. A push
  sends the hand far away, and it comes back 5 to 18 s later: within 20 mm of the target
  in 1 or 2 of nine runs at 24 s, as claimed, and in all nine at 60 s. One take is already
  enough to arrive; more takes make the path direct and the start gentle.
- **[Report 024](reports/024-many-starts-with-pushes-in-the-score/README.md):** a search
  with the pushed runs in the score finds ESNs that pass report 007's pushes. They pass
  by coming back in time, not by keeping to the take, they fail at larger pushes, and one
  of them starts with a jump.

Open questions, deferred on 2026-11-18. Stages 2 and 3 do not wait for them:

1. Whether to search once more, with a pushed run scored by how far the hand is sent or
   by the following gain, and with the start's jump in the score.
2. Which of the two reports' ESNs is taken as this stage's. The maintainer prefers
   report 023's.

### Stage 2: several targets by context

One ESN reaches several targets from many starts. The target is named by a one-hot
context vector added to the ESN's input, not by its coordinates.

- **Targets (proposed):** three, A at (−0.6, 0.4) m as in stage 1, B at (0, 0.7) m, and C
  at (0.45, 0.2) m. If three do not work, two.
- **To show:** the hand goes to the target the context names; and changing the context
  during a run changes the target.

Three targets were kept. The maintainer taught 27 takes, nine to each, choosing each
take's start and target in the recorder.

- **[Report 025](reports/025-three-targets-by-context/README.md):** both chosen ESNs
  reach the target the context names from the takes' starts, and a change of context at
  rest or on the way takes the hand to the new target in every run. From 36 unseen starts
  one ESN holds everywhere and the other misses four runs. The references begin at the
  arm from starts near a take's and jump from starts far from every take's. Pushes under
  a fixed schedule were left out; a run pushed by hand while the target changes is to be
  recorded.

### Stage 3: a periodic motion

- **3a, the periodic motion alone:** the hand goes around an ellipse or a circle, as a
  limit cycle. The curve is taught both by hand and by a script.
- **3b, reaching and the periodic motion by context:** changing the context during a run
  switches between them.
- **The curve (proposed):** an ellipse centered at (−0.1, 0.55) m, 0.2 by 0.12 m, about
  3 s a loop. It is shown to the maintainer when the stage starts, who decides it.
- **A block** was planned for this stage only, beside the pushes.

### Stage 4: all tasks in one ESN

One ESN holds every task of stages 1 to 3, chosen by context. The report ends with a
table of all claims of the study and their outcomes.

## Other work

Work outside the stages, one line each, added after it is done: the date, what, and where
it is.

- (none yet)
