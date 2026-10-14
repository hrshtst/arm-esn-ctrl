# Manual-demonstration robot recorded pushes, v2 (Stage 2)

The study after [the following search](../manual_demonstration_v2_robot_following_search/README.md)
(report 011), whose two chosen ESNs met report 007's scripted pushes. Here they meet
pushes given by hand: the arm was pushed in [the robot app](../../tools/README.md),
the run's log saved with its force, and the same force replayed by time for each ESN
and the replay of the take, with every tracker (see
[Recorded forces](../README.md#recorded-forces)). There is no search and no rule:
every arm is shown.

- **The recordings:** three runs of report 011's ESN #2 with joint PD at ω = 20 rad/s
  and a damping ratio of 0.4, given the reference velocity, its torque computed every
  10 ms, from the demonstrated start, pushed by hand by dragging the tip; saved as
  `data/recorded_forces/recorded_forces_{000,001,002}.sklog.npz` under the storage
  root. They were recorded with the app's former joint PD gains, about 0.4% stiffer in
  joint 1, so that the replay of a recording with its own ESN and tracker does not
  retrace it exactly; every run here meets the same force.

| Log | Pushes (above 0.5 N, joined across lulls under 0.2 s) | Lengths | Peak force | Impulse |
| --- | --- | --- | --- | --- |
| `recorded_forces_000` | 4, from 2.4, 6.1, 8.6, and 14.0 s | 0.38 to 0.60 s | 2.8 to 7.8 N | 0.9 to 2.4 N s |
| `recorded_forces_001` | 3, from 2.3, 7.2, and 14.4 s | 1.08, 1.26, and 0.46 s | 7.0, 14.4, and 6.2 N | 5.5, 10.6, and 2.0 N s |
| `recorded_forces_002` | 3, from 2.6, 6.0, and 13.6 s | 0.52 to 0.62 s | 3.7 to 9.1 N | 1.3 to 3.7 N s |

- **The arms:** report 011's ESN #1 (leak rate 0.03, spectral radius 1.2, input
  scaling 1.5, seed 8) and ESN #2 (input scaling 1, seed 2), and the replay of the take.
- **The trackers:** joint PD at ω = 20 rad/s and damping ratios 0.2, 0.4, and 0.8,
  given a zero reference velocity or the reference's, all with the torque computed
  every 10 ms, as in report 011.

## Step 1: the runs

[`robot_esn.py`](../robot_esn.py) runs each ESN and the replay with the three damping
ratios, under one recorded force, from the log's start posture, for 24 s:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_recorded_pushes/recorded001_robot_lr0.03_sr1.2_is1.5_seed8_reference_velocity_filtered.toml
```

| Configurations | What they run |
| --- | --- |
| `recorded{000,001,002}_robot_{lr0.03_sr1.2_is1.5_seed8,lr0.03_sr1.2_is1_seed2}_{zero,reference}_velocity_filtered.toml` (12) | one ESN and the replay, joint PD at damping ratios 0.2, 0.4, and 0.8, under one recorded force |

They are report 011's pushes configurations with the recorded force in place of the
scripted pushes. Each run keeps a copy of its log, with its SHA-256 in
`recorded_forces.toml`; the measures after each push (following, ringing, return,
farthest from the path) take the pushes above.

## Step 2: the comparisons

[`compare_arms.py`](../compare_arms.py) overlays ESN #1, ESN #2, and the replay of
each log and reference velocity: the tip force, the joint angles and torques over
time, and the tip's path in the plane with an arrow at the start of each push. Its
configurations name step 1's runs, so they are written once those are run:

```bash
uv run python experiments/compare_arms.py experiments/manual_demonstration_v2_robot_recorded_pushes/compare_recorded001_reference_velocity.toml
```

| Configurations | What they compare |
| --- | --- |
| `compare_recorded{000,001,002}_{zero,reference}_velocity.toml` (6) | ESN #1, ESN #2, and the replay of one log, with one reference velocity |
