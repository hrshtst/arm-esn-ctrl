# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the storage root lookup and the run records."""

import tomllib
from pathlib import Path

import pytest

from arm_esn_ctrl.storage import ENV_VAR, start_run, storage_root


def test_storage_root_defaults_to_storage_in_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert storage_root(repo_root=tmp_path) == tmp_path / "storage"


def test_storage_root_is_read_from_storage_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    (tmp_path / "storage.toml").write_text('storage_root = "../elsewhere"\n')
    assert storage_root(repo_root=tmp_path) == (tmp_path.parent / "elsewhere").resolve()


def test_environment_variable_overrides_storage_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "storage.toml").write_text('storage_root = "from-file"\n')
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "from-env"))
    assert storage_root(repo_root=tmp_path) == tmp_path / "from-env"


def test_start_run_records_config_and_code_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "store"))
    config_text = "# ESN settings\n[esn]\nseed = 7\nleak_rate = 0.3\n"
    config_path = tmp_path / "example.toml"
    config_path.write_text(config_text)

    config, run_dir = start_run(config_path)

    assert config == {"esn": {"seed": 7, "leak_rate": 0.3}}
    assert run_dir.parent == tmp_path / "store" / "results"
    assert run_dir.name.endswith("-example")
    assert (run_dir / "config.toml").read_text() == config_text
    record = tomllib.loads((run_dir / "run.toml").read_text())
    assert len(record["commit"]) == 40
    assert isinstance(record["uncommitted_changes"], bool)


def test_start_run_never_reuses_a_run_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "store"))
    config_path = tmp_path / "example.toml"
    config_path.write_text("")

    run_dirs = {start_run(config_path)[1] for _ in range(3)}

    assert len(run_dirs) == 3
