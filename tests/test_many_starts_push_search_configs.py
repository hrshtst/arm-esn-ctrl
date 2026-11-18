# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The push search's configurations are the many-starts experiment's, but for the pushes and the ESNs."""

import tomllib

import pytest

from arm_esn_ctrl.storage import REPO_ROOT

MANY_STARTS = REPO_ROOT / "experiments/manual_demonstration_v2_many_starts"
EXPERIMENT = REPO_ROOT / "experiments/manual_demonstration_v2_many_starts_push_search"
SWEPT = ("leak_rate", "spectral_radius", "input_scaling", "ridge")


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def assert_pushed_as(pushed, undisturbed):
    """``pushed`` is the sweep ``undisturbed`` under the pushes of the many-starts analysis at their forces."""
    pushes = load(MANY_STARTS / "pushes1x_robot_esn1.toml")
    for table in ("demonstrations", "esn", "sweep", "tracker"):
        assert pushed[table] == undisturbed[table]
    assert pushed.get("settings") == undisturbed.get("settings")
    assert "ranking" not in pushed  # score_sweeps.py scores the pushed runs with the undisturbed ones
    assert pushed["disturbance"] == pushes["disturbance"]
    assert [push["force"] for push in pushed["disturbance"]] == [3.0, 5.0, 4.0]
    measures = ("return_tolerance", "ringing_cutoff_hz", "ringing_window", "following_window")
    assert pushed["evaluation"] == undisturbed["evaluation"] | {name: pushes["evaluation"][name] for name in measures}


@pytest.mark.parametrize("prefix", ["", "one_take_"])
def test_the_coarse_searchs_pushed_runs_are_the_many_starts_coarse_search_under_the_pushes(prefix):
    pushed = load(EXPERIMENT / f"{prefix}coarse_pushes.toml")
    undisturbed = load(MANY_STARTS / f"{prefix}coarse_search.toml")

    assert_pushed_as(pushed, undisturbed)
    assert pushed["esn"]["seed"] == 0
    assert len(pushed["demonstrations"]["train"]) == (1 if prefix else 9)


def test_the_fine_search_runs_81_settings_around_the_coarse_scores_best_undisturbed_and_pushed():
    undisturbed = load(EXPERIMENT / "fine_search.toml")
    old = load(MANY_STARTS / "fine_search.toml")

    assert_pushed_as(load(EXPERIMENT / "fine_pushes.toml"), undisturbed)
    # The center, and halfway to its coarse neighbors: mirrored where the center is at the coarse grid's edge.
    assert undisturbed["sweep"] == {
        "leak_rate": [0.017, 0.03, 0.055],
        "spectral_radius": [0.75, 0.9, 1.05],
        "input_scaling": [1.0, 2.0, 4.0],
        "ridge": [0.1, 1.0, 10.0],
        "seed": [1, 2, 3, 4, 5],
    }
    assert [undisturbed["esn"][name] for name in SWEPT] == [0.03, 0.9, 2.0, 1.0]
    for name in SWEPT:
        assert undisturbed["sweep"][name][1] == undisturbed["esn"][name]
    # Otherwise the many-starts fine search.
    for table in ("demonstrations", "tracker", "evaluation", "ranking"):
        assert undisturbed[table] == old[table]
    assert {k: v for k, v in undisturbed["esn"].items() if k not in SWEPT} == {
        k: v for k, v in old["esn"].items() if k not in SWEPT
    }


def test_the_one_take_fine_searchs_pushed_runs_are_the_many_starts_one_take_fine_search_under_the_pushes():
    assert_pushed_as(load(EXPERIMENT / "one_take_fine_pushes.toml"), load(MANY_STARTS / "one_take_fine_search.toml"))


@pytest.mark.parametrize(
    ("name", "undisturbed", "pushes"),
    [
        ("coarse_score.toml", "manual_demonstration_v2_many_starts/20261117-205256-coarse_search", "coarse_pushes"),
        (
            "one_take_coarse_score.toml",
            "manual_demonstration_v2_many_starts/20261118-005826-one_take_coarse_search",
            "one_take_coarse_pushes",
        ),
        (
            "fine_score.toml",
            "manual_demonstration_v2_many_starts_push_search/20261118-114046-fine_search",
            "fine_pushes",
        ),
        (
            "one_take_fine_score.toml",
            "manual_demonstration_v2_many_starts/20261118-012638-one_take_fine_search",
            "one_take_fine_pushes",
        ),
        (
            "untaught_score.toml",
            "manual_demonstration_v2_many_starts_push_search/20261118-120421-untaught_search",
            "fine_pushes",
        ),
        (
            "one_take_untaught_score.toml",
            "manual_demonstration_v2_many_starts_push_search/20261118-123329-one_take_untaught_search",
            "one_take_fine_pushes",
        ),
    ],
)
def test_a_score_names_its_steps_undisturbed_run_and_its_pushed_run(name, undisturbed, pushes):
    runs = load(EXPERIMENT / name)["runs"]

    assert runs["undisturbed"] == f"results/{undisturbed}"
    assert runs["pushes"].startswith(f"results/{EXPERIMENT.name}/")
    assert runs["pushes"].endswith(f"-{pushes}")


@pytest.mark.parametrize(("prefix", "fine"), [("", EXPERIMENT / "fine_search.toml"), ("one_take_", None)])
def test_the_untaught_search_runs_20_settings_of_the_fine_search_from_the_untaught_starts_too(prefix, fine):
    search = load(EXPERIMENT / f"{prefix}untaught_search.toml")
    old = load(MANY_STARTS / f"{prefix}untaught_search.toml")
    grid = load(fine or MANY_STARTS / "one_take_fine_search.toml")["sweep"]

    settings = search["settings"]
    assert len(settings) == 20
    assert len({tuple(setting[name] for name in SWEPT) for setting in settings}) == 20
    for setting in settings:
        assert all(setting[name] in grid[name] for name in SWEPT)
    assert search["sweep"] == {"seed": [1, 2, 3, 4, 5]}
    assert "disturbance" not in search
    # Otherwise the many-starts search with the untaught starts.
    for table in ("demonstrations", "esn", "tracker", "evaluation", "ranking"):
        assert search[table] == old[table]
    assert len(search["evaluation"]["extra_starts"]["untaught"]) == 36


CHOSEN = {
    "esn1_lr0.03_sr0.9_is2_ridge1_seed5": ([0.03, 0.9, 2.0, 1.0], 5, "untaught_search.toml"),
    "esn2_lr0.017_sr1.05_is4_ridge10_seed5": ([0.017, 1.05, 4.0, 10.0], 5, "untaught_search.toml"),
    "one_take_esn1_lr0.017_sr1.05_is0.045_ridge0.1_seed5": (
        [0.017, 1.05, 0.045, 0.1],
        5,
        "one_take_untaught_search.toml",
    ),
    "one_take_esn2_lr0.017_sr1.05_is0.1_ridge0.1_seed1": ([0.017, 1.05, 0.1, 0.1], 1, "one_take_untaught_search.toml"),
}


@pytest.mark.parametrize("name", CHOSEN)
def test_a_chosen_esn_is_a_setting_and_a_seed_of_its_search_run_as_the_many_starts_ones(name):
    values, seed, search_name = CHOSEN[name]
    chosen = load(EXPERIMENT / f"{name}.toml")
    search = load(EXPERIMENT / search_name)
    old = "one_take_esn1_lr0.017_sr1.05_is0.045_ridge0.1_seed5" if name.startswith("one_take") else None
    old = load(MANY_STARTS / f"{old or 'esn1_lr0.055_sr1.2_is1_ridge1_seed4'}.toml")

    assert [chosen["esn"][key] for key in SWEPT] == values
    assert dict(zip(SWEPT, values, strict=True)) in search["settings"]
    assert chosen["esn"]["seed"] == seed
    assert seed in search["sweep"]["seed"]
    assert {k: v for k, v in chosen["esn"].items() if k not in (*SWEPT, "seed")} == {
        k: v for k, v in search["esn"].items() if k not in (*SWEPT, "seed")
    }
    for table in ("demonstrations", "evaluation", "apps"):
        assert chosen[table] == old[table]


@pytest.mark.parametrize("n_takes", [1, 3, 6, 9])
def test_the_take_counts_are_the_many_starts_ones_at_the_chosen_settings(n_takes):
    takes = load(EXPERIMENT / f"takes{n_takes}.toml")
    old = load(MANY_STARTS / f"takes{n_takes}.toml")

    assert takes.pop("settings") == [
        dict(zip(SWEPT, CHOSEN[name][0], strict=True))
        for name in ("esn1_lr0.03_sr0.9_is2_ridge1_seed5", "esn2_lr0.017_sr1.05_is4_ridge10_seed5")
    ]
    old.pop("settings")
    assert takes == old
