# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Where data and results are stored, and how each run is recorded.

The storage root holds two directories: ``data/`` for demonstrations and
``results/`` for run outputs. It is chosen in this order:

1. the ``ARM_ESN_CTRL_STORAGE_ROOT`` environment variable, if it is set;
2. otherwise, ``storage_root`` in ``storage.toml`` at the repository root;
3. otherwise, ``storage/`` in the repository.

A relative path is resolved against the repository root.

Every experiment starts with :func:`start_run`, which loads the configuration
file and creates a run directory that records how the result was produced. Run
directories are filed by experiment: ``results/<experiment>/<run>/``, where the
experiment is the directory that holds the configuration file, such as
``experiments/multi_demonstration_robot_tracking/``. A configuration names
another run, such as its demonstrations, by its path under the storage root;
:func:`resolve_run_path` finds it.
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

import tomli_w

ENV_VAR = "ARM_ESN_CTRL_STORAGE_ROOT"
REPO_ROOT = Path(__file__).resolve().parents[2]


def storage_root(repo_root: Path = REPO_ROOT) -> Path:
    """Return the storage root (see the module docstring for the lookup order)."""
    value = os.environ.get(ENV_VAR) or None
    if value is None:
        storage_file = repo_root / "storage.toml"
        if storage_file.exists():
            with storage_file.open("rb") as f:
                value = tomllib.load(f)["storage_root"]
    if value is None:
        return repo_root / "storage"
    return (repo_root / Path(value).expanduser()).resolve()


def start_run(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load a configuration file and create a run directory that records it.

    The run directory is ``<storage root>/results/<experiment>/<date>-<time>-<config name>/``,
    where the experiment is the name of the directory that holds the configuration file.
    It receives two files:

    - ``config.toml``: an exact copy of the configuration file, as loaded;
    - ``run.toml``: when, where, and with which code the run was started.

    Returns the loaded configuration and the run directory, where the
    experiment then writes its outputs.
    """
    config_path = Path(config_path)
    text = config_path.read_text(encoding="utf-8")
    config = tomllib.loads(text)

    started = datetime.now().astimezone().replace(microsecond=0)
    run_dir = _new_run_dir(started, config_path.parent.resolve().name, config_path.stem)
    (run_dir / "config.toml").write_text(text, encoding="utf-8")

    commit, uncommitted_changes = _git_state()
    record = {
        "config": str(_relative_to_repo(config_path)),
        "command": shlex.join(sys.argv),
        "started": started,
        "host": socket.gethostname(),
        "commit": commit,
        "uncommitted_changes": uncommitted_changes,
    }
    (run_dir / "run.toml").write_text(tomli_w.dumps(record), encoding="utf-8")
    return config, run_dir


def resolve_run_path(reference: str | Path) -> Path:
    """The path under the storage root that ``reference`` names, such as a run directory or a file in one.

    ``reference`` is relative to the storage root:
    ``results/<experiment>/<run>[/<file>]``, or ``results/<run>[/<file>]`` as
    recorded before runs were filed by experiment. Run names are unique, since they
    begin with the time the run started, so a run is also found by its name alone,
    in whichever experiment directory it is.

    Raises
    ------
    FileNotFoundError
        If no run, or more than one, has that name.
    """
    root = storage_root()
    path = root / reference
    if path.exists():
        return path
    parts = Path(reference).parts
    if len(parts) < 2 or parts[0] != "results":
        return path  # not a run: whatever the caller makes of a missing path
    name, rest = parts[1], parts[2:]
    matches = sorted(run for run in (root / "results").glob(f"*/{name}") if run.is_dir())
    if not matches:
        msg = f"no run named {name!r} under {root / 'results'}"
        raise FileNotFoundError(msg)
    if len(matches) > 1:
        msg = f"several runs named {name!r}: {', '.join(str(match) for match in matches)}"
        raise FileNotFoundError(msg)
    return matches[0].joinpath(*rest)


def _new_run_dir(started: datetime, experiment: str, name: str) -> Path:
    """Create a new, empty run directory of ``experiment``, adding a suffix if the name is taken."""
    results = storage_root() / "results" / experiment
    base = f"{started:%Y%m%d-%H%M%S}-{name}"
    suffix = 1
    while True:
        run_dir = results / (base if suffix == 1 else f"{base}-{suffix}")
        try:
            run_dir.mkdir(parents=True)
        except FileExistsError:
            suffix += 1
        else:
            return run_dir


def _git_state() -> tuple[str, bool]:
    """Return the current commit hash and whether there are uncommitted changes."""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)

    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        return "unknown", True
    status = git("status", "--porcelain")
    return head.stdout.strip(), status.stdout.strip() != ""


def _relative_to_repo(path: Path) -> Path:
    """Return ``path`` relative to the repository root if it is inside it."""
    path = path.resolve()
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
