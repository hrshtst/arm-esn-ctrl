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
