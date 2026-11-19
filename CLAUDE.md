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

- **A presentation's media** are made by `presentations/<name>/make_media.py` and written
  to the storage root under `presentations/<name>/`, with a `README.md` that says what the
  files show and at which commit they were made, and a `media.csv` that lists every file
  with the logs it shows. The script is kept in Git; its output is not. Start a new one
  as a copy of the nearest: `many-starts` (reaches, report 023), or
  `periodic-motion-from-a-script` and `periodic-motion-by-hand` (reports 026 and 027).
- **A single run or take:** skelarm's player,
  `player.py <log> --export out.mp4 --fps 30 --panel` with `QT_QPA_PLATFORM=offscreen`
  (`player_export` in those scripts). `--speed N` makes the export N times as fast.
- **A window of a long run:** the player exports a whole log. Cut the log to the window
  into a temporary log, keeping the samples' times and every channel, and export that
  (`cut` in the periodic-motion scripts).
- **Runs one after another, or text drawn on the frames:** `sequence_video` (imageio +
  PIL). Every single clip is kept as its own file, and the concatenated videos are made
  from the kept clips; the faster ones (`SPEEDS`) keep every N-th frame. The
  periodic-motion scripts' `--all-starts-only` makes the concatenated videos alone, from
  the clips already rendered.
- **Encoding:** keep `quality=9.5` or lower and `macro_block_size=16`. Never
  `quality=10`: that is lossless H.264 (High 4:4:4 Predictive), which PowerPoint shows as
  a black frame while VLC plays it. A good file is H.264 High, yuv420p.
- **Order of work:** render a few runs into a scratch storage root first (the scripts'
  `--only` and `--starts`; a scratch root whose `results` links the storage root's), look
  at frames and figures, and commit the script. Then render into the storage root, and
  tell the maintainer which file to try in PowerPoint before rendering the rest. More
  files from a script whose videos the maintainer has already confirmed need no new
  check.
- A script overwrites its own files and deletes nothing.

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
