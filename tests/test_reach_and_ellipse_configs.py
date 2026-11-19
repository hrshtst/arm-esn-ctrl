# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The reach and ellipse experiment's configurations agree with stage 1's, stage 2's, and stage 3a's."""

import itertools
import re
import sys
import tomllib

import numpy as np
import pytest

from arm_esn_ctrl.autonomous import start_postures, task_evaluations, task_names
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.sweeps import sweep_combinations

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from robot_esn import start_tasks, switch_schedule

EXPERIMENT = REPO_ROOT / "experiments/manual_demonstration_v2_reach_and_ellipse"
MANY_STARTS = REPO_ROOT / "experiments/manual_demonstration_v2_many_starts"  # stage 1's
TARGETS = REPO_ROOT / "experiments/manual_demonstration_v2_targets"  # stage 2's
PERIODIC = REPO_ROOT / "experiments/manual_demonstration_v2_periodic"  # stage 3a's
CONTEXT = ("context_scaling", "context_times_input_scaling")  # the [esn] keys of the context


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def header_command(path):
    """The command a configuration's header comment gives, its continued lines joined, as its words."""
    header = itertools.takewhile(lambda line: line.startswith("#"), path.read_text().splitlines())
    lines = [line.lstrip("#").strip() for line in header]
    first = next(i for i, line in enumerate(lines) if line.startswith("uv run "))
    words = []
    for line in lines[first:]:
        words += line.removesuffix("\\").split()
        if not line.endswith("\\"):
            break
    return words


def test_the_coarse_search_trains_on_stage_1s_nine_takes_as_reach_and_stage_3as_take_by_hand_as_ellipse():
    reach, ellipse = load(EXPERIMENT / "coarse_search.toml")["demonstrations"]

    assert reach.pop("context") == "reach"
    assert reach == load(MANY_STARTS / "coarse_search.toml")["demonstrations"]
    assert reach["run"] == "results/demonstrations/20261117-164658-reach_manual_v2_no_limits"
    assert reach["train"] == [f"demo_{i:02d}_filtered.sklog.npz" for i in range(9)]
    assert ellipse.pop("context") == "ellipse"
    assert ellipse == load(PERIODIC / "coarse_search.toml")["demonstrations"]
    assert ellipse == {
        "run": "results/demonstrations/20261119-093746-ellipse_manual_v2_no_limits",
        "train": ["demo_00_filtered.sklog.npz"],
    }


def test_the_coarse_search_parses_as_a_sweep_of_stage_1s_108_settings_of_one_esn_of_two_tasks_seed_0():
    config = load(EXPERIMENT / "coarse_search.toml")

    assert task_names(config["demonstrations"]) == ("reach", "ellipse")
    assert config["sweep"] == load(MANY_STARTS / "coarse_search.toml")["sweep"]
    assert config["sweep"] == load(PERIODIC / "coarse_search.toml")["sweep"]
    combinations = sweep_combinations(config["sweep"], config.get("settings"))
    assert len(combinations) == 108
    for combination in combinations:
        EsnConfig(**config["esn"] | combination)
    esn = EsnConfig(**config["esn"])
    assert (esn.n_neurons, esn.seed) == (800, 0)


def test_the_coarse_searchs_esn_is_stage_1s_and_3as_with_stage_2s_context():
    config = load(EXPERIMENT / "coarse_search.toml")
    targets = load(TARGETS / "coarse_search.toml")

    esn = {key: value for key, value in config["esn"].items() if key not in CONTEXT}
    assert esn == load(MANY_STARTS / "coarse_search.toml")["esn"]
    assert esn == load(PERIODIC / "coarse_search.toml")["esn"]
    assert {key: config["esn"][key] for key in CONTEXT} == {key: targets["esn"][key] for key in CONTEXT}
    assert (config["esn"]["context_scaling"], config["esn"]["context_times_input_scaling"]) == (1.0, False)
    assert not set(CONTEXT) & set(config["sweep"])
    # Whatever the input scaling, the context's input weights, times it, have the scale 1.
    for input_scaling in config["sweep"]["input_scaling"]:
        model = ReachingEsn(EsnConfig(**config["esn"] | {"input_scaling": input_scaling}))
        assert model.context_gain * input_scaling == pytest.approx(1.0)


def test_the_coarse_search_tracks_as_stage_1s_and_3as_and_ranks_by_the_worse_of_the_two_tasks_ranks():
    config = load(EXPERIMENT / "coarse_search.toml")

    assert config["tracker"] == load(MANY_STARTS / "coarse_search.toml")["tracker"]
    assert config["tracker"] == load(PERIODIC / "coarse_search.toml")["tracker"]
    assert config["ranking"] == {"by": "worse_rank"}
    assert "disturbance" not in config


def test_each_tasks_runs_are_those_of_the_coarse_search_of_the_stage_it_comes_from():
    config = load(EXPERIMENT / "coarse_search.toml")

    reach, ellipse = task_evaluations(config["evaluation"], task_names(config["demonstrations"]))

    # The reach from the nine takes' starts for 24 s with stage 1's hold; the ellipse from the eight starts on the
    # curve for 40 s with 3a's cycle: 17 runs.
    assert reach == load(MANY_STARTS / "coarse_search.toml")["evaluation"]
    assert reach == {"duration": 24.0, "hold": 2.0, "start_offsets_deg": [[0.0, 0.0]]}
    assert ellipse == load(PERIODIC / "coarse_search.toml")["evaluation"]
    assert ellipse["duration"] == 40.0
    assert (ellipse["curve_tolerance"], ellipse["cycle_periods"], ellipse["cycle_turns"]) == (0.02, 5, [4.0, 6.0])
    assert len(ellipse["start_offsets_deg"]) + len(ellipse["extra_starts"]["on_curve"]) == 8


# The search's three sweeps: the coarse search, the fine search, and the scoring with the untaught starts.
SEARCHES = ["coarse_search.toml", "fine_search.toml", "untaught_search.toml"]


@pytest.mark.parametrize("name", SEARCHES)
def test_each_searchs_header_runs_it_with_sweep_robot_esn_and_a_number_of_workers(name):
    words = header_command(EXPERIMENT / name)

    assert words[:-1] == [
        "uv",
        "run",
        "python",
        "experiments/sweep_robot_esn.py",
        f"experiments/manual_demonstration_v2_reach_and_ellipse/{name}",
        "--workers",
    ]
    assert int(words[-1]) > 0


# The coarse search's best setting (run 20261119-190537-coarse_search): the fine search's center.
CENTER = {"leak_rate": 0.1, "spectral_radius": 1.5, "input_scaling": 0.5, "ridge": 1.0}


def test_the_fine_search_is_the_coarse_search_but_for_its_values_and_seeds():
    fine, coarse = load(EXPERIMENT / "fine_search.toml"), load(EXPERIMENT / "coarse_search.toml")

    fine_sweep, coarse_sweep = fine.pop("sweep"), coarse.pop("sweep")
    assert fine_sweep.pop("seed") == [1, 2, 3, 4, 5]
    assert list(fine_sweep) == list(coarse_sweep)
    fine_esn, coarse_esn = fine.pop("esn"), coarse.pop("esn")
    assert fine_esn == coarse_esn | CENTER | {"seed": 1}  # the center's values, the first seed, and the same context
    assert fine == coarse  # the same takes and contexts, tracker, runs, and score


def test_the_fine_search_takes_the_center_and_the_values_halfway_to_its_coarse_neighbors():
    fine, coarse = load(EXPERIMENT / "fine_search.toml"), load(EXPERIMENT / "coarse_search.toml")

    for name, center in CENTER.items():
        grid = coarse["sweep"][name]
        k = grid.index(center)
        if name == "spectral_radius":  # halfway in difference
            # Beyond an edge of the grid, as far from the center as the neighbor on the other side.
            lower = grid[k - 1] if k > 0 else 2 * center - grid[k + 1]
            upper = grid[k + 1] if k + 1 < len(grid) else 2 * center - grid[k - 1]
            expected = [round((lower + center) / 2, 2), center, round((center + upper) / 2, 2)]
        else:  # halfway in ratio, to two digits
            lower = grid[k - 1] if k > 0 else center**2 / grid[k + 1]
            upper = grid[k + 1] if k + 1 < len(grid) else center**2 / grid[k - 1]
            expected = [float(f"{np.sqrt(lower * center):.2g}"), center, float(f"{np.sqrt(center * upper):.2g}")]
        assert fine["sweep"][name] == expected, name


ESN_NAMES = ["leak_rate", "spectral_radius", "input_scaling", "ridge"]
# The fine search's two best settings (run 20261119-193314-fine_search), each passing 69 of its 85 runs.
TWO_BEST = [(0.055, 1.65, 0.5, 10.0), (0.17, 1.65, 0.22, 10.0)]


def test_the_untaught_search_is_the_fine_search_but_for_its_settings_seeds_and_untaught_starts():
    again, fine = load(EXPERIMENT / "untaught_search.toml"), load(EXPERIMENT / "fine_search.toml")

    settings, sweep, fine_sweep = again.pop("settings"), again.pop("sweep"), fine.pop("sweep")
    assert len(settings) == 20
    assert sweep == {"seed": fine_sweep["seed"]}  # seeds 1 to 5, the other values listed in settings
    tasks = again["evaluation"]["tasks"]
    assert list(tasks["reach"].pop("extra_starts")) == ["untaught"]  # the fine search's reach has none
    assert list(tasks["ellipse"]["extra_starts"]) == ["on_curve", "untaught"]
    tasks["ellipse"]["extra_starts"].pop("untaught")
    assert again == fine  # the same takes and contexts, ESN, tracker, runs, and score


def test_the_untaught_searchs_settings_are_distinct_and_on_the_fine_searchs_grid_its_two_best_first():
    again, fine = load(EXPERIMENT / "untaught_search.toml"), load(EXPERIMENT / "fine_search.toml")

    settings = [tuple(setting[name] for name in ESN_NAMES) for setting in again["settings"]]
    assert len(set(settings)) == len(settings) == 20
    for setting in again["settings"]:
        assert list(setting) == ESN_NAMES
        assert all(setting[name] in fine["sweep"][name] for name in ESN_NAMES)
    assert settings[:2] == TWO_BEST


def test_the_untaught_search_runs_each_task_from_stage_1s_36_untaught_starts_as_stage_1s_and_3as_scorings_do():
    config = load(EXPERIMENT / "untaught_search.toml")
    untaught = load(MANY_STARTS / "hand_esn.toml")["evaluation"]["extra_starts"]["untaught"]
    on_curve = load(EXPERIMENT / "fine_search.toml")["evaluation"]["tasks"]["ellipse"]["extra_starts"]["on_curve"]

    reach, ellipse = task_evaluations(config["evaluation"], task_names(config["demonstrations"]))

    assert len(untaught) == 36
    assert reach["extra_starts"] == {"untaught": untaught}
    assert ellipse["extra_starts"] == {"on_curve": on_curve, "untaught": untaught}
    # Each task's runs are those of the scoring with the untaught starts of the stage it comes from.
    assert reach == load(MANY_STARTS / "untaught_search.toml")["evaluation"]
    assert ellipse == load(PERIODIC / "untaught_search.toml")["evaluation"]


def test_the_untaught_search_parses_as_a_sweep_of_100_esns_of_two_tasks_with_89_runs_each():
    config = load(EXPERIMENT / "untaught_search.toml")

    assert task_names(config["demonstrations"]) == ("reach", "ellipse")
    combinations = sweep_combinations(config["sweep"], config["settings"])
    assert len(combinations) == 100  # each setting with its five seeds in turn
    assert combinations[:5] == [config["settings"][0] | {"seed": seed} for seed in range(1, 6)]
    for combination in combinations:
        EsnConfig(**config["esn"] | combination)
    # Each task's start postures, from each of its takes' starts (here zeros, as only their number counts) and from
    # its extra starts: the reach's 9 and 36, the ellipse's 1, 7, and 36.
    evaluations = task_evaluations(config["evaluation"], task_names(config["demonstrations"]))
    starts = [
        start_postures(evaluation, {name: np.zeros((1, 2)) for name in table["train"]})
        for table, evaluation in zip(config["demonstrations"], evaluations, strict=True)
    ]
    assert [len(each) for each in starts] == [45, 44]  # 89 runs an ESN, 445 a setting


# The two-task ESNs, the typical seeds of the two best settings of the scoring with the untaught starts (run
# 20261119-205228-untaught_search). For ESN #1 and ESN #2: the stem of its configuration, the place of its setting in
# the scoring's list, which is in the fine search's order, its setting, and its seed. The fine search's best setting
# is fourth in the scoring, so ESN #1's setting is the second in the list and ESN #2's the third.
CHOSEN_ESNS = {
    1: ("esn1_lr0.17_sr1.65_is0.22_ridge10_seed4", 1, (0.17, 1.65, 0.22, 10.0), 4),
    2: ("esn2_lr0.055_sr1.5_is0.5_ridge10_seed4", 2, (0.055, 1.5, 0.5, 10.0), 4),
}


@pytest.mark.parametrize("n", CHOSEN_ESNS)
def test_each_two_task_esn_is_trained_as_the_untaught_searchs_esns_with_the_setting_and_seed_the_scoring_chose(n):
    stem, place, _, seed = CHOSEN_ESNS[n]
    esn, search = load(EXPERIMENT / f"{stem}.toml"), load(EXPERIMENT / "untaught_search.toml")

    assert list(esn) == ["demonstrations", "esn", "evaluation", "apps"]  # no sweep, and no tracker
    assert esn["demonstrations"] == search["demonstrations"]  # the takes and their contexts, as the search's
    assert esn["evaluation"] == search["evaluation"]  # each task's runs, as the scoring's
    assert esn["esn"] == search["esn"] | search["settings"][place] | {"seed": seed}


@pytest.mark.parametrize("n", CHOSEN_ESNS)
def test_each_two_task_esns_setting_is_listed_by_the_untaught_search_and_its_seed_is_one_of_its(n):
    stem, place, setting, seed = CHOSEN_ESNS[n]
    esn, search = load(EXPERIMENT / f"{stem}.toml")["esn"], load(EXPERIMENT / "untaught_search.toml")

    assert tuple(esn[name] for name in ESN_NAMES) == setting
    assert search["settings"].index({name: esn[name] for name in ESN_NAMES}) == place
    assert esn["seed"] == seed
    assert seed in search["sweep"]["seed"]


@pytest.mark.parametrize("n", CHOSEN_ESNS)
def test_each_two_task_esns_file_name_gives_its_number_setting_and_seed(n):
    stem, _, _, _ = CHOSEN_ESNS[n]
    esn = load(EXPERIMENT / f"{stem}.toml")["esn"]

    leak_rate, spectral_radius, input_scaling, ridge = (esn[name] for name in ESN_NAMES)
    seed = esn["seed"]
    assert stem == f"esn{n}_lr{leak_rate:g}_sr{spectral_radius:g}_is{input_scaling:g}_ridge{ridge:g}_seed{seed}"


@pytest.mark.parametrize("n", CHOSEN_ESNS)
def test_each_two_task_esns_header_trains_and_runs_it_with_autonomous_esn(n):
    stem, _, _, _ = CHOSEN_ESNS[n]

    assert header_command(EXPERIMENT / f"{stem}.toml") == [
        "uv",
        "run",
        "python",
        "experiments/autonomous_esn.py",
        f"experiments/manual_demonstration_v2_reach_and_ellipse/{stem}.toml",
    ]


# The claims' runs on the robot, each with a configuration for two-task ESN #1 and one for ESN #2. Of each run with a
# change of context, as the README's claims 7 to 9 state it: the task whose starts run, their number, the run's
# duration (s), and each change's time on the task clock (s) and new task. After the last change, a reach goes on for
# 24 s and the ellipse for 40 s.
SWITCHED_RUNS = {
    "to_reach_robot": ("ellipse", 8, 40.0, [(16.0, "reach")]),  # claim 7
    "to_ellipse_robot": ("reach", 9, 60.0, [(20.0, "ellipse")]),  # claim 8
    "back_and_forth_robot": ("ellipse", 8, 100.0, [(16.0, "reach"), (36.0, "ellipse"), (76.0, "reach")]),  # claim 9
}
CLAIM_RUNS = ["robot", *SWITCHED_RUNS]  # claim 6's, with one context for the whole run, then claims 7 to 9's
ROBOT_RUNS = [f"{name}_esn{n}.toml" for name in CLAIM_RUNS for n in CHOSEN_ESNS]
CYCLE = ("curve_tolerance", "cycle_periods", "cycle_turns")  # the ellipse's keys of a run that cycles


def esn_number(name):
    """The number of the two-task ESN a robot run's file name states."""
    return int(name.removesuffix(".toml")[-1])


def starts_of_each_task(evaluation, demonstrations):
    """Each task's start postures, from each of its takes' starts (here zeros, as only their number counts)."""
    evaluations = task_evaluations(evaluation, task_names(demonstrations))
    return [
        start_postures(each, {name: np.zeros((1, 2)) for name in table["train"]})
        for table, each in zip(demonstrations, evaluations, strict=True)
    ]


def test_the_claims_have_eight_runs_on_the_robot_one_of_each_esn_for_each_of_claims_6_to_9():
    assert sorted(path.name for path in EXPERIMENT.glob("*robot_esn*.toml")) == sorted(ROBOT_RUNS)
    assert len(ROBOT_RUNS) == 8


@pytest.mark.parametrize("name", ROBOT_RUNS)
def test_each_robot_run_loads_the_model_trained_by_the_configuration_of_the_esn_its_file_name_states(name):
    stem, _, _, _ = CHOSEN_ESNS[esn_number(name)]

    esn = load(EXPERIMENT / name)["esn"]
    assert list(esn) == ["model"]  # esn.toml in the run directory of the ESN's training
    run = rf"results/manual_demonstration_v2_reach_and_ellipse/\d{{8}}-\d{{6}}-{re.escape(stem)}"
    assert re.fullmatch(rf"{run}/esn\.toml", esn["model"])
    assert (EXPERIMENT / f"{stem}.toml").is_file()


@pytest.mark.parametrize("name", CLAIM_RUNS)
def test_each_esn_2_run_is_the_esn_1_run_but_for_the_model(name):
    esn1, esn2 = load(EXPERIMENT / f"{name}_esn1.toml"), load(EXPERIMENT / f"{name}_esn2.toml")

    assert esn1.pop("esn") != esn2.pop("esn")
    assert esn1 == esn2


@pytest.mark.parametrize("name", ROBOT_RUNS)
def test_each_robot_run_tracks_holds_and_cycles_as_the_search_undisturbed(name):
    config, search = load(EXPERIMENT / name), load(EXPERIMENT / "untaught_search.toml")

    assert config["tracker"] == search["tracker"]
    assert not {"disturbance", "progress"} & set(config)
    tasks, searched = config["evaluation"]["tasks"], search["evaluation"]["tasks"]
    assert list(tasks) == ["reach", "ellipse"]
    assert tasks["reach"]["hold"] == searched["reach"]["hold"] == 2.0
    assert {key: tasks["ellipse"][key] for key in CYCLE} == {key: searched["ellipse"][key] for key in CYCLE}


@pytest.mark.parametrize("n", CHOSEN_ESNS)
def test_claim_6_runs_the_reach_as_the_search_but_from_stage_1s_36_unseen_starts_in_place_of_the_untaught(n):
    config, search = load(EXPERIMENT / f"robot_esn{n}.toml"), load(EXPERIMENT / "untaught_search.toml")
    unseen = load(MANY_STARTS / "robot_esn1.toml")["evaluation"]["extra_starts"]["unseen"]

    assert list(config) == ["esn", "tracker", "evaluation"]  # no change of context
    assert list(config["evaluation"]) == ["tasks"]  # every task's starts run
    reach, searched = config["evaluation"]["tasks"]["reach"], search["evaluation"]["tasks"]["reach"]
    assert reach.pop("extra_starts") == {"unseen": unseen}
    assert list(searched.pop("extra_starts")) == ["untaught"]
    assert reach == searched  # for 24 s with stage 1's hold, from each take's own start


def test_stage_1s_36_unseen_starts_are_none_of_the_untaught_ones():
    unseen = load(EXPERIMENT / "robot_esn1.toml")["evaluation"]["tasks"]["reach"]["extra_starts"]["unseen"]
    untaught = load(EXPERIMENT / "untaught_search.toml")["evaluation"]["tasks"]["reach"]["extra_starts"]["untaught"]

    assert len(unseen) == len(untaught) == 36
    assert not [posture for posture in unseen if posture in untaught]


@pytest.mark.parametrize("n", CHOSEN_ESNS)
def test_claim_6_runs_the_ellipse_as_the_search_but_for_60_s(n):
    config, search = load(EXPERIMENT / f"robot_esn{n}.toml"), load(EXPERIMENT / "untaught_search.toml")

    # From the eight starts on the curve and the 36 untaught ones, with 3a's cycle.
    assert config["evaluation"]["tasks"]["ellipse"] == search["evaluation"]["tasks"]["ellipse"] | {"duration": 60.0}


@pytest.mark.parametrize("n", CHOSEN_ESNS)
def test_claim_6_runs_the_reach_for_24_s_from_45_starts_and_the_ellipse_for_60_s_from_44(n):
    stem, _, _, _ = CHOSEN_ESNS[n]
    config = load(EXPERIMENT / f"robot_esn{n}.toml")
    demonstrations = load(EXPERIMENT / f"{stem}.toml")["demonstrations"]
    names = task_names(demonstrations)

    assert names == ("reach", "ellipse")
    assert start_tasks(config["evaluation"], names) == [0, 1]
    assert [each["duration"] for each in task_evaluations(config["evaluation"], names)] == [24.0, 60.0]
    reach, ellipse = starts_of_each_task(config["evaluation"], demonstrations)
    assert [start.group for start in reach] == ["demonstrated"] * 9 + ["unseen"] * 36
    assert [start.group for start in ellipse] == ["demonstrated"] + ["on_curve"] * 7 + ["untaught"] * 36


@pytest.mark.parametrize("n", CHOSEN_ESNS)
@pytest.mark.parametrize("name", SWITCHED_RUNS)
def test_each_change_of_context_runs_for_the_claims_duration_and_changes_the_task_at_the_claims_times(name, n):
    stem, _, _, _ = CHOSEN_ESNS[n]
    config = load(EXPERIMENT / f"{name}_esn{n}.toml")
    names = task_names(load(EXPERIMENT / f"{stem}.toml")["demonstrations"])
    _, _, duration, changes = SWITCHED_RUNS[name]

    assert config["switch"] == [{"time": time, "task": task} for time, task in changes]
    assert switch_schedule(config["switch"], names) == [(time, names.index(task)) for time, task in changes]
    evaluation = config["evaluation"]
    # One duration for the whole run, which both tasks take, and the reference's start measured over 2 s after each
    # change.
    assert (evaluation["duration"], evaluation["after_window"]) == (duration, 2.0)
    assert not [task for task in evaluation["tasks"].values() if "duration" in task]
    assert [each["duration"] for each in task_evaluations(evaluation, names)] == [duration, duration]


@pytest.mark.parametrize("n", CHOSEN_ESNS)
@pytest.mark.parametrize("name", SWITCHED_RUNS)
def test_each_change_of_context_runs_from_the_starts_of_its_first_task_in_the_search_before_the_untaught(name, n):
    stem, _, _, _ = CHOSEN_ESNS[n]
    config = load(EXPERIMENT / f"{name}_esn{n}.toml")
    demonstrations = load(EXPERIMENT / f"{stem}.toml")["demonstrations"]
    fine = load(EXPERIMENT / "fine_search.toml")["evaluation"]["tasks"]
    names = task_names(demonstrations)
    first, runs, _, _ = SWITCHED_RUNS[name]

    assert config["evaluation"]["start_tasks"] == [first]
    assert start_tasks(config["evaluation"], names) == [names.index(first)]
    # The nine reach takes' starts, or the eight starts on the curve: the fine search's, which has no untaught ones.
    tasks = config["evaluation"]["tasks"]
    for key in ("start_offsets_deg", "extra_starts"):
        assert tasks[first].get(key) == fine[first].get(key)
    starts = starts_of_each_task(config["evaluation"], demonstrations)[names.index(first)]
    assert len(starts) == runs


@pytest.mark.parametrize("name", ROBOT_RUNS)
def test_each_robot_runs_header_names_its_esn_with_its_setting_and_seed(name):
    n = esn_number(name)
    _, _, setting, seed = CHOSEN_ESNS[n]
    leak_rate, spectral_radius, input_scaling, ridge = setting

    header = itertools.takewhile(lambda line: line.startswith("#"), (EXPERIMENT / name).read_text().splitlines())
    text = " ".join(line.lstrip("#").strip() for line in header)
    assert text.startswith(
        f"Two-task ESN #{n} (leak rate {leak_rate:g}, spectral radius {spectral_radius:g}, input scaling"
        f" {input_scaling:g}, ridge {ridge:g}, seed {seed}) on the robot"
    )


@pytest.mark.parametrize("name", ROBOT_RUNS)
def test_each_robot_runs_header_runs_it_with_robot_esn(name):
    assert header_command(EXPERIMENT / name) == [
        "uv",
        "run",
        "python",
        "experiments/robot_esn.py",
        f"experiments/manual_demonstration_v2_reach_and_ellipse/{name}",
    ]
