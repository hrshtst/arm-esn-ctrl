# Manual-demonstration robot tracking, v2 (Stage 2)

The procedure of
[manual-demonstration robot tracking](../manual_demonstration_robot_tracking/README.md)
(report 004) on the remade arm of
[manual-demonstration autonomous reaching, v2](../manual_demonstration_v2_autonomous_reaching/README.md):
an ESN trained on the take generates the reference of the simulated robot arm,
driven by the arm's measured posture, and a tracker follows it. Each run is
compared with the take replayed by time. The robot is the arm of the take's
configuration ([`reach_manual_v2.toml`](../demonstrations/reach_manual_v2.toml)),
which the ESN's run names.

## Base configurations

One base per scenario: copy it to a new file and edit the copy. The lines to set
are marked `EDIT` (`grep -n EDIT <file>`); the values are report 004's.

| Base | Scenario |
| --- | --- |
| `nominal_base_raw.toml` | no disturbance, from the demonstrated start |
| `offsets_base_raw.toml` | no disturbance, from the grid of start postures |
| `push_across_base_raw.toml` | a push at the tip for 0.1 s, across the reach |
| `push_forward_base_raw.toml` | the same, along the reach toward the target |
| `push_backward_base_raw.toml` | the same, along the reach away from the target |
| `block_base_raw.toml` | the tip held by a stiff spring-damper, then let go |

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_tracking/<copy>.toml
```

Besides the metrics and the figures of report 004's runs, each run draws
`joints.png`: the joint angles over time from the first start posture, for each
tracker setting, of the taught motion, the ESN's output, the arm driven by the
ESN, and the arm replaying the take.

Report 004 named its copies `<scenario>_<ESN>[_pd_gains|_damping][_zero_velocity]_raw.toml`:

| Suffix | `[tracker]` |
| --- | --- |
| none | `laws = ["computed_torque", "pd"]`, `omegas = [10.0]` |
| `_pd_gains` | `laws = ["pd"]`, `omegas = [10.0, 20.0, 40.0]` |
| `_damping` | `omegas = { computed_torque = [10.0], pd = [20.0] }`, `dampings = [1.0, 0.5, 0.3, 0.1]`; its `offsets` from `[[10.0, -10.0], [-10.0, -10.0]]` only |
| `_zero_velocity` | as `_pd_gains` or `_damping` with joint PD only, and `reference_velocity = false` |

## ESN sweeps on the robot

[`sweep_robot_esn.py`](../sweep_robot_esn.py) tunes the ESN by how the robot follows
the taught path with it, rather than by its autonomous runs: every combination of
the swept `[esn]` values is trained on the take and generates the reference of the
arm from the demonstrated start, with each tracker setting, undisturbed. The
combinations are ranked by their failed runs (never arriving, leaving the goal, or
holding under 2 s), then by the worst path RMSE of their runs from the taught
motion, regardless of timing (`taught_path_rmse_m`): arriving late or moving
slowly costs nothing.

```bash
uv run python experiments/sweep_robot_esn.py experiments/manual_demonstration_v2_robot_tracking/sweep_robot_main_lr0.7_filtered.toml
```

| Configurations | What they sweep |
| --- | --- |
| `sweep_robot_baseline_filtered.toml` | stage 0: `grid_test_filtered.toml`'s settings alone |
| `sweep_robot_main_lr{0.1,0.2,0.3,0.5,0.7,1}_filtered.toml` | stage 1: input scaling 0.03–1.5 × spectral radius 0.5–0.99, one leak rate each |
| `sweep_robot_ridge_warmup_filtered_lr<lr>_sr<sr>_is1.5.toml` | stage 2: ridge 1e-6–10 × warm-up 0.25–2 s, around the best three of stage 1 (leak rate and spectral radius 0.2 and 0.99, 0.1 and 0.99, 0.1 and 0.5; input scaling 1.5) |

All train on the take filtered at 2 Hz, with ridge 1e-2, a 1 s warm-up, 400
neurons, and sparsity 0.05, and run 22 s with the tracker of
`nominal_test_filtered.toml`: computed torque and joint PD at ω = 20 rad/s,
ζ = 0.1, given a zero reference velocity.

## What depends on the take and the arm

- `[esn] model`: the trained ESN of a grid run of Stage 1.
- `[evaluation] duration`: the take's length and about 1 s more.
- The disturbances' times (`onset`, `release`, `effort_window`): report 004 struck
  the take where report 003 struck its scripted reach, the block when the take's
  hand was 3.4% of the way to the target and the pushes at 9.9%.
- The push forces and the trackers' natural frequencies, if the new arm is much
  heavier or lighter: joint PD scales its gains by the arm's inertia, but a push of
  a given force moves a lighter arm farther.

## Runs

Under `results/manual_demonstration_v2_robot_tracking/` in the storage.

| Run | Configuration |
| --- | --- |
