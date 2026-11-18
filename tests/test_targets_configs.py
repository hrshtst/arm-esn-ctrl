# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""The targets experiment's configurations agree with each other, with the takes, and with the many-starts ones."""

import itertools
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
from skelarm import Skeleton, Task, multi_target_specs

from arm_esn_ctrl.autonomous import start_postures, task_names
from arm_esn_ctrl.demonstrations import endpoint_positions
from arm_esn_ctrl.esn import EsnConfig, ReachingEsn
from arm_esn_ctrl.storage import REPO_ROOT
from arm_esn_ctrl.sweeps import sweep_combinations

# The runners import each other as modules of their own directory, as they run.
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from robot_esn import switch_schedule

TARGETS = REPO_ROOT / "experiments/demonstrations/reach_manual_v2_targets.toml"
EXPERIMENT = REPO_ROOT / "experiments/manual_demonstration_v2_targets"
MANY_STARTS = REPO_ROOT / "experiments/manual_demonstration_v2_many_starts"  # stage 1's
CONTEXT = ("context_scaling", "context_times_input_scaling")  # the [esn] keys of the context, which stage 1 has not
TAKES = [f"demo_{i:02d}_filtered.sklog.npz" for i in range(9)]  # each target's nine takes, filtered


def load(path):
    with path.open("rb") as f:
        return tomllib.load(f)


def without(table, keys):
    """``table`` without ``keys``."""
    return {k: v for k, v in table.items() if k not in keys}


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


def test_the_coarse_search_parses_as_a_sweep_of_108_settings_of_one_esn_of_three_tasks():
    config = load(EXPERIMENT / "coarse_search.toml")

    assert task_names(config["demonstrations"]) == ("A", "B", "C")
    esn = EsnConfig(**config["esn"])
    assert (esn.n_neurons, esn.seed) == (800, 0)
    combinations = sweep_combinations(config["sweep"], config.get("settings"))
    assert len(combinations) == 108
    for combination in combinations:
        EsnConfig(**config["esn"] | combination)


def test_the_coarse_search_trains_on_the_nine_filtered_takes_of_each_target_a_b_and_c_in_that_order():
    tasks = load(EXPERIMENT / "coarse_search.toml")["demonstrations"]
    labels = [label for _, label, _, _ in multi_target_specs(Task.from_dict(load(TARGETS)["task"]))]

    assert [table["context"] for table in tasks] == labels == ["A", "B", "C"]
    for table in tasks:
        assert list(table) == ["context", "run", "train"]
        # The demonstration run the import writes for the target its context names.
        assert table["run"].startswith("results/demonstrations/")
        assert table["run"].endswith(f"-reach_manual_v2_targets_{table['context'].lower()}")
        assert table["train"] == TAKES


@pytest.mark.parametrize("table", ["sweep", "ranking", "tracker", "evaluation"])
def test_the_coarse_search_sweeps_scores_tracks_and_runs_as_stage_1s_coarse_search(table):
    targets, many_starts = load(EXPERIMENT / "coarse_search.toml"), load(MANY_STARTS / "coarse_search.toml")

    assert targets[table] == many_starts[table]


def test_the_coarse_searchs_esn_is_stage_1s_coarse_searchs_with_a_context():
    targets, many_starts = load(EXPERIMENT / "coarse_search.toml"), load(MANY_STARTS / "coarse_search.toml")

    assert not set(CONTEXT) & set(many_starts["esn"])
    assert without(targets["esn"], CONTEXT) == many_starts["esn"]
    assert "disturbance" not in targets


def test_the_context_is_fed_in_divided_by_the_input_scaling_with_a_scale_of_1_and_is_not_swept():
    config = load(EXPERIMENT / "coarse_search.toml")

    assert (config["esn"]["context_scaling"], config["esn"]["context_times_input_scaling"]) == (1.0, False)
    assert not set(CONTEXT) & set(config["sweep"])
    # Whatever the input scaling, the context's input weights, times it, have the scale 1.
    for input_scaling in config["sweep"]["input_scaling"]:
        esn = ReachingEsn(EsnConfig(**config["esn"] | {"input_scaling": input_scaling}))
        assert esn.context_gain * input_scaling == pytest.approx(1.0)


CENTER = {"leak_rate": 0.1, "spectral_radius": 1.2, "input_scaling": 2.0, "ridge": 1.0}  # the coarse search's best


def two_digits(x):
    """``x`` rounded to two significant digits."""
    return float(f"{x:.2g}")


def test_the_fine_search_parses_as_a_sweep_of_81_settings_of_five_seeds_1_to_5_of_three_tasks():
    config = load(EXPERIMENT / "fine_search.toml")

    assert task_names(config["demonstrations"]) == ("A", "B", "C")
    sweep = dict(config["sweep"])
    assert sweep.pop("seed") == [1, 2, 3, 4, 5]
    assert len(sweep_combinations(sweep)) == 81
    combinations = sweep_combinations(config["sweep"], config.get("settings"))
    assert len(combinations) == 81 * 5
    for combination in combinations:
        EsnConfig(**config["esn"] | combination)


def test_the_fine_searchs_middle_values_are_the_coarse_searchs_best_setting():
    coarse, fine = load(EXPERIMENT / "coarse_search.toml"), load(EXPERIMENT / "fine_search.toml")

    assert list(fine["sweep"]) == [*CENTER, "seed"]
    for name, center in CENTER.items():
        assert fine["sweep"][name][1] == fine["esn"][name] == center
        assert center in coarse["sweep"][name]


def test_the_fine_searchs_outer_values_are_halfway_to_the_coarse_neighbors_and_as_far_beyond_an_edge():
    coarse, fine = load(EXPERIMENT / "coarse_search.toml"), load(EXPERIMENT / "fine_search.toml")

    # The center's input scaling and ridge are the coarse grid's largest.
    assert [name for name, center in CENTER.items() if center == coarse["sweep"][name][-1]] == [
        "input_scaling",
        "ridge",
    ]
    for name, center in CENTER.items():
        grid = coarse["sweep"][name]
        below, _, above = fine["sweep"][name]
        k = grid.index(center)
        # Beyond the grid's edge, as far from the center as the coarse neighbor on the other side.
        if name == "spectral_radius":  # halfway in difference
            lower = grid[k - 1] if k > 0 else 2 * center - grid[k + 1]
            upper = grid[k + 1] if k + 1 < len(grid) else 2 * center - grid[k - 1]
            assert (below, above) == pytest.approx(((lower + center) / 2, (center + upper) / 2))
        else:  # halfway in ratio, rounded to two digits
            lower = grid[k - 1] if k > 0 else center**2 / grid[k + 1]
            upper = grid[k + 1] if k + 1 < len(grid) else center**2 / grid[k - 1]
            assert (below, above) == (two_digits((lower * center) ** 0.5), two_digits((center * upper) ** 0.5))


@pytest.mark.parametrize("table", ["demonstrations", "ranking", "tracker", "evaluation"])
def test_the_fine_search_trains_on_the_same_takes_and_scores_tracks_and_runs_as_the_coarse_search(table):
    coarse, fine = load(EXPERIMENT / "coarse_search.toml"), load(EXPERIMENT / "fine_search.toml")

    assert fine[table] == coarse[table]


def test_the_fine_searchs_esn_is_the_coarse_searchs_with_its_context_but_for_the_swept_values():
    coarse, fine = load(EXPERIMENT / "coarse_search.toml"), load(EXPERIMENT / "fine_search.toml")

    swept = {*fine["sweep"]}
    assert without(fine["esn"], swept) == without(coarse["esn"], swept)
    assert set(CONTEXT) <= set(fine["esn"])
    assert list(fine) == list(coarse)  # the same tables, and no disturbance


def test_the_untaught_search_parses_as_a_sweep_of_20_settings_of_five_seeds_of_three_tasks():
    config = load(EXPERIMENT / "untaught_search.toml")

    assert task_names(config["demonstrations"]) == ("A", "B", "C")
    combinations = sweep_combinations(config["sweep"], config["settings"])
    assert len(combinations) == 20 * 5
    for combination in combinations:
        EsnConfig(**config["esn"] | combination)


def test_the_untaught_search_lists_20_settings_of_the_fine_searchs_values_with_its_seeds():
    again, fine = load(EXPERIMENT / "untaught_search.toml"), load(EXPERIMENT / "fine_search.toml")
    names = list(CENTER)

    settings = [tuple(setting[name] for name in names) for setting in again["settings"]]
    assert len(set(settings)) == len(settings) == 20
    for setting in again["settings"]:
        assert list(setting) == names
        assert all(setting[name] in fine["sweep"][name] for name in names)
    assert settings[0] == (0.17, 1.05, 1.0, 10.0)  # the only one of the fine search's that holds in every run
    assert again["sweep"] == {"seed": fine["sweep"]["seed"]}


def test_the_untaught_search_is_the_fine_search_but_for_its_settings_its_seeds_and_the_untaught_starts():
    again, fine = load(EXPERIMENT / "untaught_search.toml"), load(EXPERIMENT / "fine_search.toml")

    again.pop("settings")
    again.pop("sweep")
    fine.pop("sweep")
    again["evaluation"].pop("extra_starts")
    # The same takes, ESN, score, tracker, and runs from the takes' starts, and no disturbance.
    assert again == fine


def test_the_untaught_starts_are_stage_1s_36_and_none_is_nearer_than_85_mm_to_a_target():
    extra = load(EXPERIMENT / "untaught_search.toml")["evaluation"]["extra_starts"]
    skeleton = Skeleton.from_toml(TARGETS)
    targets = [pos for pos, _, _, _ in multi_target_specs(Task.from_dict(load(TARGETS)["task"]))]

    assert list(extra) == ["untaught"]
    assert extra == load(MANY_STARTS / "untaught_search.toml")["evaluation"]["extra_starts"]
    assert len(extra["untaught"]) == 36
    hands = endpoint_positions(skeleton, np.radians(extra["untaught"]))
    assert min(np.linalg.norm(hands - target, axis=1).min() for target in targets) >= 0.085


@pytest.mark.parametrize("name", ["coarse_search.toml", "fine_search.toml", "untaught_search.toml"])
def test_the_searchs_header_runs_it_with_15_workers_as_the_machine_is_shared(name):
    assert header_command(EXPERIMENT / name) == [
        "uv",
        "run",
        "python",
        "experiments/sweep_robot_esn.py",
        f"experiments/manual_demonstration_v2_targets/{name}",
        "--workers",
        "15",
    ]


# ESN #1 and ESN #2, the typical seeds of the two best settings of the untaught search: their
# files, the place of their setting in its list, and their setting and seed.
CHOSEN = [
    ("esn1_lr0.17_sr1.05_is1_ridge10_seed1.toml", 0, (0.17, 1.05, 1.0, 10.0, 1)),
    ("esn2_lr0.17_sr1.2_is1_ridge10_seed5.toml", 2, (0.17, 1.2, 1.0, 10.0, 5)),  # third in the fine search's order
]
SWEPT = (*CENTER, "seed")
STAGE_1_ESN = MANY_STARTS / "esn1_lr0.055_sr1.2_is1_ridge1_seed4.toml"


@pytest.mark.parametrize("name", [name for name, _, _ in CHOSEN])
def test_each_chosen_esn_parses_as_one_esn_of_three_tasks_run_from_each_takes_start_and_36_untaught_ones(name):
    config = load(EXPERIMENT / name)

    assert task_names(config["demonstrations"]) == ("A", "B", "C")
    EsnConfig(**config["esn"])
    # Of each task: its nine takes' starts, then the 36 untaught starts, run with its context.
    takes = {take.split(".")[0]: np.zeros((1, 2)) for take in TAKES}
    starts = start_postures(config["evaluation"], takes)
    assert [start.group for start in starts] == ["demonstrated"] * 9 + ["untaught"] * 36


@pytest.mark.parametrize("name", [name for name, _, _ in CHOSEN])
def test_each_chosen_esn_is_the_untaught_searchs_in_its_takes_its_esn_and_its_runs_but_for_its_setting_and_seed(
    name,
):
    chosen, again = load(EXPERIMENT / name), load(EXPERIMENT / "untaught_search.toml")

    assert list(chosen) == ["demonstrations", "esn", "evaluation", "apps"]  # no sweep, and no disturbance
    assert chosen["demonstrations"] == again["demonstrations"]
    assert without(chosen["esn"], SWEPT) == without(again["esn"], SWEPT)
    assert set(CONTEXT) <= set(chosen["esn"])
    # Its duration, its hold, the takes' own starts, and the untaught starts.
    assert chosen["evaluation"] == again["evaluation"]


@pytest.mark.parametrize(("n", "chosen"), list(enumerate(CHOSEN, start=1)), ids=[name for name, _, _ in CHOSEN])
def test_each_chosen_esns_setting_is_listed_by_the_untaught_search_its_seed_is_one_of_its_and_its_name_states_them(
    n, chosen
):
    name, place, values = chosen
    esn, again = load(EXPERIMENT / name)["esn"], load(EXPERIMENT / "untaught_search.toml")

    assert tuple(esn[key] for key in SWEPT) == values
    setting = {key: esn[key] for key in CENTER}
    assert again["settings"].index(setting) == place  # ESN #1's is the first, the one that held in every run
    assert esn["seed"] in again["sweep"]["seed"]
    leak_rate, spectral_radius, input_scaling, ridge, seed = values
    assert name == f"esn{n}_lr{leak_rate:g}_sr{spectral_radius:g}_is{input_scaling:g}_ridge{ridge:g}_seed{seed}.toml"


def test_the_two_chosen_esns_differ_only_in_their_spectral_radius_and_seed():
    esn1, esn2 = (load(EXPERIMENT / name) for name, _, _ in CHOSEN)
    different = ("spectral_radius", "seed")

    assert without(esn1, {"esn"}) == without(esn2, {"esn"})
    assert without(esn1["esn"], different) == without(esn2["esn"], different)
    for key in different:
        assert esn1["esn"][key] != esn2["esn"][key]


@pytest.mark.parametrize("name", [name for name, _, _ in CHOSEN])
def test_each_chosen_esns_apps_get_stage_1s_esns_tracker_and_a_take_in_the_repository(name):
    apps = load(EXPERIMENT / name)["apps"]

    assert apps == load(STAGE_1_ESN)["apps"]
    arguments = apps["robot_app"].split()
    assert (REPO_ROOT / arguments[arguments.index("--gains-from") + 1]).exists()


@pytest.mark.parametrize("name", [name for name, _, _ in CHOSEN])
def test_each_chosen_esns_header_trains_and_runs_it_with_autonomous_esn(name):
    assert header_command(EXPERIMENT / name) == [
        "uv",
        "run",
        "python",
        "experiments/autonomous_esn.py",
        f"experiments/manual_demonstration_v2_targets/{name}",
    ]


# The analysis's runs on the robot: claims 1 to 3 from the takes' starts and the unseen ones, and
# claims 4 and 5 with the context changed to each target. Each kind of change, its time and its
# duration (s): it changes at 16 s or at 4 s on the task clock, and the run goes on 24 s after it.
SWITCHES = {"switch_to": (16.0, 40.0), "switch_early_to": (4.0, 28.0)}
SWITCHED = [(kind, target) for kind in SWITCHES for target in ("A", "B", "C")]
PREFIXES = ["", *(f"{kind}_{target.lower()}_" for kind, target in SWITCHED)]
ROBOT_RUNS = [f"{prefix}robot_esn{n}.toml" for prefix in PREFIXES for n in (1, 2)]


def test_the_analysis_has_14_runs_on_the_robot_two_for_claims_1_to_3_and_six_each_for_claims_4_and_5():
    assert sorted(path.name for path in EXPERIMENT.glob("*robot_esn*.toml")) == sorted(ROBOT_RUNS)
    assert len(ROBOT_RUNS) == 14


@pytest.mark.parametrize("name", ROBOT_RUNS)
def test_each_robot_run_names_the_model_saved_by_the_training_of_the_esn_its_file_name_states(name):
    n = int(name.removesuffix(".toml")[-1])
    chosen = Path(CHOSEN[n - 1][0])

    esn = load(EXPERIMENT / name)["esn"]
    assert list(esn) == ["model"]
    assert esn["model"].startswith("results/manual_demonstration_v2_targets/")
    assert esn["model"].endswith(f"-{chosen.stem}/esn.toml")
    assert chosen.name.startswith(f"esn{n}_")


@pytest.mark.parametrize("name", ROBOT_RUNS)
def test_each_robot_run_tracks_holds_and_starts_from_each_takes_own_start_as_the_untaught_search(name):
    config, again = load(EXPERIMENT / name), load(EXPERIMENT / "untaught_search.toml")

    assert config["tracker"] == again["tracker"]
    for key in ("hold", "start_offsets_deg"):
        assert config["evaluation"][key] == again["evaluation"][key]
    assert not {"disturbance", "progress"} & set(config)


@pytest.mark.parametrize("n", [1, 2])
def test_claims_1_to_3_run_each_context_for_24_s_from_its_nine_takes_starts_and_stage_1s_36_unseen_starts(n):
    config = load(EXPERIMENT / f"robot_esn{n}.toml")

    assert list(config) == ["esn", "tracker", "evaluation"]  # no change of the context
    evaluation = config["evaluation"]
    assert evaluation["duration"] == 24.0
    assert list(evaluation["extra_starts"]) == ["unseen"]
    assert evaluation["extra_starts"] == load(MANY_STARTS / "robot_esn1.toml")["evaluation"]["extra_starts"]
    # Of each task: its nine takes' starts, then the 36 unseen starts, run with its context.
    takes = {take.split(".")[0]: np.zeros((1, 2)) for take in TAKES}
    starts = start_postures(evaluation, takes)
    assert [start.group for start in starts] == ["demonstrated"] * 9 + ["unseen"] * 36


def test_the_unseen_starts_are_none_of_the_untaught_ones_and_none_is_nearer_than_67_mm_to_a_target():
    unseen = load(EXPERIMENT / "robot_esn1.toml")["evaluation"]["extra_starts"]["unseen"]
    untaught = load(EXPERIMENT / "untaught_search.toml")["evaluation"]["extra_starts"]["untaught"]
    skeleton = Skeleton.from_toml(TARGETS)
    targets = [pos for pos, _, _, _ in multi_target_specs(Task.from_dict(load(TARGETS)["task"]))]

    assert len(unseen) == 36
    assert not [posture for posture in unseen if posture in untaught]
    hands = endpoint_positions(skeleton, np.radians(unseen))
    assert min(np.linalg.norm(hands - target, axis=1).min() for target in targets) >= 0.067


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize(("kind", "target"), SWITCHED)
def test_each_switch_changes_the_context_once_to_its_files_target_and_is_otherwise_claims_1_to_3s_without_unseen_starts(
    kind, target, n
):
    switched = load(EXPERIMENT / f"{kind}_{target.lower()}_robot_esn{n}.toml")
    robot = load(EXPERIMENT / f"robot_esn{n}.toml")
    time, duration = SWITCHES[kind]

    assert switched["switch"] == [{"time": time, "task": target}]
    names = task_names(load(EXPERIMENT / CHOSEN[n - 1][0])["demonstrations"])
    assert switch_schedule(switched.pop("switch"), names) == [(time, names.index(target))]
    assert switched["evaluation"].pop("duration") == duration == time + 24.0
    assert "extra_starts" not in switched["evaluation"]  # each take's own start only
    robot["evaluation"].pop("duration")
    robot["evaluation"].pop("extra_starts")
    assert switched == robot


@pytest.mark.parametrize("prefix", PREFIXES)
def test_each_esn_2_run_is_the_esn_1_run_but_for_the_model(prefix):
    esn1, esn2 = load(EXPERIMENT / f"{prefix}robot_esn1.toml"), load(EXPERIMENT / f"{prefix}robot_esn2.toml")

    assert esn1.pop("esn") != esn2.pop("esn")
    assert esn1 == esn2


@pytest.mark.parametrize("name", ROBOT_RUNS)
def test_each_robot_runs_header_runs_it_with_robot_esn(name):
    assert header_command(EXPERIMENT / name) == [
        "uv",
        "run",
        "python",
        "experiments/robot_esn.py",
        f"experiments/manual_demonstration_v2_targets/{name}",
    ]
