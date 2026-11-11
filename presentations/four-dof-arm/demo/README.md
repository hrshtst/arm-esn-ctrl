# The four-link arm, live in the simulator

Step 5's ESN #1 and ESN #2 and the replay of the take drive skelarm's four-link arm live
in `tools/robot_app.py`, with the experiments' tracker, under the experiments'
disturbances or your own pushes; `tools/esn_reference_app.py` runs an ESN alone, without
the robot. Everything they read is in the repository, so the storage drive need not be
mounted:

- the arm, its start posture, and the goal: `experiments/demonstrations/reach_manual_four_dof.toml`;
- the ESNs: `esn.toml` and `esn.rclib` in report 021's runs of ESN #1 and ESN #2, in
  `reports/021-search-on-a-four-link-arm/results/`;
- the take filtered at 2 Hz, which the ESNs learned, the replay replays, and whose final
  posture sets the tracker's gains: report 021's copy of `demo_00_filtered.sklog.npz`;
- the forces of the pushes and the block, recorded in the experiments' runs:
  `presentations/four-dof-arm/demo/forces/`.

## Setup

In bash or zsh, at the repository root:

    CFG=experiments/demonstrations/reach_manual_four_dof.toml
    RUNS=reports/021-search-on-a-four-link-arm/results
    ESN1=$RUNS/20261109-203149-following_esn1_lr0.015_sr1.2_is0.05_seed0/esn.toml
    ESN2=$RUNS/20261109-203201-following_esn2_lr0.02_sr1.35_is0.05_seed7/esn.toml
    TAKE=reports/021-search-on-a-four-link-arm/data/20261103-134628-reach_manual_four_dof/demo_00_filtered.sklog.npz
    FORCES=presentations/four-dof-arm/demo/forces
    robot() {
        uv run python tools/robot_app.py "$CFG" --law pd --omega 20 --damping 0.4 --mass-matrix \
            --control-period 0.01 --gains-from "$TAKE" "$@"
    }

`robot` opens the robot app with the experiments' tracker: joint PD at omega = 20 rad/s
and a damping ratio of 0.4, its gains the arm's whole mass matrix where the take ends,
its torque computed every 10 ms and held between. Give it `--model "$ESN1"` or
`--model "$ESN2"` for an ESN, or `--replay "$TAKE"` for the replay.

## The ESN alone

    uv run python tools/esn_reference_app.py "$CFG" --model "$ESN1"

The ESN generates the reach from its own output, with no robot to follow. It opens at
the take's start; drag the tip to choose another start, then press Space.

## On the robot, pushed by hand

    robot --model "$ESN1"
    robot --replay "$TAKE"

Press Space: the arm reaches from the take's start, (-20, 10, 60, 50) deg, undisturbed,
within 0.005 deg of the experiments' undisturbed run. While it moves, drag with the mouse
to pull the tip with a spring (20 N/m by default, "Drag stiffness" in the panel). Before
a run, dragging the tip poses the arm at another start.

## The experiments' disturbances

`--forces` replays a force recorded in an experiment's run, by time, from that run's
start; the red arrow at the tip shows it, and posing and dragging are off. Each command
gives the experiment's run, the same to the last digit. For example:

    robot --model "$ESN1" --forces "$FORCES/pushes.sklog.npz"
    robot --model "$ESN2" --forces "$FORCES/pushes_quarter.sklog.npz"
    robot --replay "$TAKE" --forces "$FORCES/block_replay.sklog.npz"

| File in `forces/` | Disturbance | For |
| --- | --- | --- |
| `pushes.sklog.npz` | 3 N at 2.6 s, 5 N at 6.7 s, and 4 N at 14.9 s, each for 0.5 s | every arm |
| `pushes_half.sklog.npz` | the same pushes at half force: 1.5, 2.5, and 2 N | every arm |
| `pushes_quarter.sklog.npz` | the same pushes at a quarter force: 0.75, 1.25, and 1 N | every arm |
| `block_esn1.sklog.npz` | the tip held where it was from 4.0 to 5.0 s | ESN #1 only |
| `block_esn2.sklog.npz` | the same | ESN #2 only |
| `block_replay.sklog.npz` | the same | the replay only |

A push is the same force at the same times whatever the arm does, so one file serves
every arm. The block is a stiff spring (20000 N/m, damped at 100 N s/m) that holds the tip
where it was at 4.0 s: its force depends on how the arm moves, so each arm has its own
file, and another arm's would not hold it.

## Offset starts

    robot --model "$ESN1" --pose=-19.9997,5.0005,55.0027,45.0017

`--pose` sets the start in degrees (the `=` is needed, since it begins with a minus
sign); `robot --replay "$TAKE"` and the ESN alone take it the same way. The experiments
start from the take's first sample, (-19.9997, 10.0005, 60.0027, 50.0017) deg, plus the
offset. Two of the 15 deg runs (ESN #1 from +15 deg in every joint, ESN #2 from -15 deg in
joints 3 and 4) end differently from a start rounded to whole degrees, so the starts
below keep four decimals; from them, every run is within 0.002 deg of the experiment's.
They are the starts of the slides' animations:

| Offset (deg) | `--pose=` | Animation |
| --- | --- | --- |
| (+15, +15, +15, +15) | `-4.9997,25.0005,75.0027,65.0017` | `four-dof-arm/04_offset_+15_+15_+15_+15` |
| (0, 0, -15, -15) | `-19.9997,10.0005,45.0027,35.0017` | `four-dof-arm/05_offset_+0_+0_-15_-15` |
| (+5, +5, +5, +5) | `-14.9997,15.0005,65.0027,55.0017` | `four-dof-arm_smaller-disturbances/03_offset_+5_+5_+5_+5` |
| (0, 0, -5, -5) | `-19.9997,10.0005,55.0027,45.0017` | `four-dof-arm_smaller-disturbances/04_offset_+0_+0_-5_-5` |
| (0, -5, -5, -5) | `-19.9997,5.0005,55.0027,45.0017` | `four-dof-arm_smaller-disturbances/05_offset_+0_-5_-5_-5` |
| (+2.5, +2.5, +2.5, +2.5) | `-17.4997,12.5005,62.5027,52.5017` | `four-dof-arm_smaller-disturbances/06_offset_+2.5_+2.5_+2.5_+2.5` |
| (0, 0, -2.5, -2.5) | `-19.9997,10.0005,57.5027,47.5017` | `four-dof-arm_smaller-disturbances/07_offset_+0_+0_-2.5_-2.5` |
| (0, -2.5, -2.5, -2.5) | `-19.9997,7.5005,57.5027,47.5017` | `four-dof-arm_smaller-disturbances/08_offset_+0_-2.5_-2.5_-2.5` |

## Keys and notes

- Space plays and pauses; Right or F takes 10 ms while paused; R returns to the start;
  Q quits.
- Maximize the window: at its default size, the side panel's readouts are cut off at
  the bottom.
- Save log (Ctrl+S) writes under the storage drive by default. Without the drive
  mounted, it stops the app, so leave it alone then.

## Where the forces come from

Each file is a whole run's log, copied unchanged from step 5's and step 6's runs of the
four-link consistent search, under the storage root's
`results/manual_demonstration_four_dof_consistent_search/`; the app uses only its force
and its first posture, the take's start.

| File | Run, and its log in `pd_w20_z0.4/` |
| --- | --- |
| `pushes.sklog.npz` | `20261109-205926-pushes_robot_following_esn1_lr0.015_sr1.2_is0.05_seed0`, `replay_00.sklog.npz` |
| `pushes_half.sklog.npz` | `20261110-233931-pushes_half_robot_following_esn1_lr0.015_sr1.2_is0.05_seed0`, `replay_00.sklog.npz` |
| `pushes_quarter.sklog.npz` | `20261110-234039-pushes_quarter_robot_following_esn1_lr0.015_sr1.2_is0.05_seed0`, `replay_00.sklog.npz` |
| `block_esn1.sklog.npz` | `20261109-203232-block_robot_following_esn1_lr0.015_sr1.2_is0.05_seed0`, `esn_00.sklog.npz` |
| `block_esn2.sklog.npz` | `20261109-203306-block_robot_following_esn2_lr0.02_sr1.35_is0.05_seed7`, `esn_00.sklog.npz` |
| `block_replay.sklog.npz` | `20261109-203232-block_robot_following_esn1_lr0.015_sr1.2_is0.05_seed0`, `replay_00.sklog.npz` |

The pushes' files are the replay's logs: ESN #1's and ESN #2's logs of the same runs hold
the same force, from the same start. The replay's block is the same in ESN #1's run and
ESN #2's.

Every command here was run headless, with the storage root pointed at an empty folder,
for each arm under each disturbance and from each start in the table, and compared with
the experiment's run as above.
