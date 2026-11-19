# Working in this repository

An ESN generates the joint reference of a simulated arm from demonstrations. The
[README](README.md) gives the motivation, the layout, and the list of reports; each
experiment's README under `experiments/` gives its rule, claims, and runs.

[PLAN.md](PLAN.md) is a map of where the multi-task study is heading. It is not an
instruction: the maintainer's request in the session decides what is done.

## Commands

```bash
uv run pytest                                # tests
uv run ruff format . && uv run ruff check .  # formatting and linting
uv run basedpyright && uv run mypy           # type checking
```

## Code

- **Simple code.** Plain functions on NumPy arrays, few layers. Add an abstraction only
  when a second real use appears. No options or error handling for cases that do not
  occur.
- **Every parameter that affects a result is in a TOML configuration,** not in a script.
  A configuration's header comment says what it runs and gives its command.
- **Comments, docstrings, and READMEs are plain prose** in the style of the files around
  them: short sentences, what the thing is and does.
- **Tests are named as sentences** (`test_a_pushed_run_passes_if_it_arrives_and_...`).
  Configurations are tested against each other (`tests/test_*_configs.py`).
- ESN numerics belong to rclib and arm simulation to skelarm (`third_party/`). Do not
  work around them here.

## Runs and storage

- Runs go to the storage root (`storage.toml`), which is the maintainer's drive. A run
  directory records its configuration and commit.
- **Test and dry runs go to a scratch storage root:** set `ARM_ESN_CTRL_STORAGE_ROOT` to a
  scratch directory that links the demonstrations it needs.
- Never delete or overwrite a run directory in the storage root without asking.
- Sweeps take `--workers 30`. Run sweeps one after another, not side by side.

## Experiments and reports

- **Claims, rules, and the runs that test them are written into the experiment's README
  and committed before the runs.** What is added after seeing results is labelled so.
- Negative results are reported as they are. A claim is not reworded after its runs.
- **The maintainer approves before the next piece of work starts.** When a stage, or
  whatever was asked for, is finished and reported, stop there. Do not start the next
  stage, another search, or a report's proposed next steps until the maintainer asks.
- A report is `reports/NNN-name/`: `README.md`, `make_figures.py`, copies of the runs'
  records and tables under `results/`, and figures under `results/summary/`. Its
  `make_figures.py` prints every table the text quotes.
- A report states the ESNs' parameters, swept and fixed, and their training takes. It
  compares the chosen ESNs with the replay, and shows joint angles and torques over time
  for every scenario, the undisturbed one too.

## Videos and figures for slides

- **Where:** `presentations/<name>/make_media.py` makes them and writes them to the
  storage root under `presentations/<name>/`, not into the repository. Start from the
  nearest existing script (`many-starts`, `periodic-motion-from-a-script`,
  `periodic-motion-by-hand`).
- **A single run or take:** skelarm's player,
  `player.py <log> --export out.mp4 --fps 30 --panel` with `QT_QPA_PLATFORM=offscreen`.
  `--speed N` makes the export N times as fast.
- **Runs one after another, or text drawn on the frames:** imageio and PIL
  (`sequence_video` in those scripts). Keep `quality=9.5` or lower and
  `macro_block_size=16`. Never `quality=10`: that is lossless H.264, which PowerPoint
  shows as a black frame while VLC plays it.
- **Keep every clip as its own file,** and build the concatenated videos from the kept
  clips.
- **Render into a scratch storage root first** and look at frames, then render into the
  storage root.
- **Recommended, to adapt to what the videos show:** cut a long run to a window before
  exporting it; give the script options to render a part (`--only`, `--starts`,
  `--all-starts-only` in the periodic-motion scripts).

## Git

- Commit each change before producing results from it or asking for review. Fixes after
  review are new commits; do not amend.
- Commits are local, on `main` or on the branch of a session's own worktree. **Never
  push,** and do not merge branches; the maintainer does both.
- **A new git worktree** needs three things before anything runs in it:
  `git submodule update --init --recursive`, `uv sync`, and a copy of the main worktree's
  `storage.toml`. That file is not tracked, and without it runs go to `storage/` inside
  the worktree instead of the storage root.
- No attribution or session trailers in commit messages.
- Subjects follow the log: `feat(experiments): ...`, `docs(reports): ...`, `fix: ...`.

## Roles

- **The main session** plans, briefs sub-agents, reviews their work against the tests,
  commits, runs the sweeps and analyses, and writes the reports.
- **A sub-agent** edits files and runs the tests, the linters, and dry runs into a scratch
  storage root. It does not commit, push, run into the storage root, or widen its task.
  It reports what it changed, what it ran, and anything it left undone.
- **Blockers.** A sub-agent that meets something ambiguous stops and reports it instead of
  guessing. The main session settles what the fixed plan or the repository answers, and
  asks the maintainer about what is the maintainer's to decide: a rule, a claim, a score,
  or anything that changes what a result means.
