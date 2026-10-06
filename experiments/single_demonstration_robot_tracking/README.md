# Single-demonstration robot tracking (Stage 2)

An ESN trained on one demonstration generates the reference of the simulated robot
arm, driven by the arm's measured posture, and a tracker follows it, as in
[multi-demonstration robot tracking](../multi_demonstration_robot_tracking/README.md).
Each run is compared with the demonstration replayed by time and with the
demonstrator's own controller.
The results are in [report 003](../../reports/003-single-demonstration/README.md).

[Single-demonstration autonomous reaching](../single_demonstration_autonomous_reaching/README.md)
found two ESNs trained on the same demonstration that run in different ways:

| ESN | How it runs on its own |
| --- | --- |
| `tuned`: the best settings of the single-demonstration sweeps (`grid_tuned_settings.toml`, run `20261005-212407`) | it follows the demonstrated path; the start posture sets how far along it the run begins, and it goes on from there |
| `multi_demo_settings`: the settings found best for eight demonstrations (`grid_multi_demo_settings.toml`, run `20261005-190405`) | it keeps time like a clock running from the reset of its reservoir |

The questions:

1. **Adaptation:** does the reference follow the arm's actual progress, waiting
   while the arm is held back and skipping ahead when it is pushed forward, or does
   it keep its own timing, as the replay does?
2. **Initial offsets:** can the trackers follow the ESN's jump back toward the
   demonstration in its first step (108 to 182 mm on its own)?

## Configurations

Each scenario runs with both ESNs: `<scenario>_tuned.toml` and
`<scenario>_multi_demo_settings.toml`. The tracker settings are the same in all:
computed torque and joint PD, both critically damped with ω = 10 rad/s.

| Scenario | Starts | Disturbance |
| --- | --- | --- |
| `nominal` | the demonstrated start | none |
| `offsets` | the demonstrated start, then the 169-start grid (−15° to 15° in each joint) | none |
| `push_across` | the demonstrated start | 5 N at the tip for 0.1 s at 0.4 s, across the reach |
| `push_forward` | the demonstrated start | the same, along the reach toward the target |
| `push_backward` | the demonstrated start | the same, along the reach away from the target |
| `block` | the demonstrated start | the tip held from 0.3 s to 0.8 s by a stiff spring-damper |

```bash
uv run python experiments/robot_esn.py experiments/single_demonstration_robot_tracking/block_tuned.toml
```

At ω = 10 rad/s, joint PD lets the hand drift out of the goal after it arrives
with either ESN, though it settles. `<scenario>_<ESN>_pd_gains.toml` runs each
scenario again with joint PD only, at ω = 10, 20, and 40 rad/s.

`<scenario>_<ESN>_damping.toml` runs each scenario with the tracker underdamped:
computed torque at ω = 10 rad/s and joint PD at ω = 20 rad/s, each with the damping
ratios 1, 0.5, 0.3, and 0.1. Its `offsets` scenario starts from two of the grid's
offsets only, (+10°, −10°) and (−10°, −10°).

The runner and its outputs are those of
[multi-demonstration robot tracking](../multi_demonstration_robot_tracking/README.md).
Two outputs answer the questions here:

- **Progress along the demonstrated path:** for a posture, how far along the
  demonstrated joint path its nearest point lies, from 0 at the start to 1 at the
  end. `timeline.png` draws it over time for each arm (solid) and its reference
  (dashed); `metrics.csv` has the reference's lead over the arm at the end of the
  disturbance (`reference_lead`). A reference that keeps its own timing advances
  whatever the arm does; one that adapts stays with the arm.
- **`grid.png`** (the `offsets` scenario): maps over the start offsets of the
  outcome, the path distance, the training path ratio, the peak reference speed,
  the peak torque, and the tracking error, for the ESN and the replay.

## Runs

Under `results/single_demonstration_robot_tracking/` in the storage:

| Run | Configuration |
| --- | --- |
| `20261005-231058-nominal_tuned` | `nominal_tuned.toml` |
| `20261005-231107-nominal_multi_demo_settings` | `nominal_multi_demo_settings.toml` |
| `20261005-231058-offsets_tuned` | `offsets_tuned.toml` |
| `20261005-231058-offsets_multi_demo_settings` | `offsets_multi_demo_settings.toml` |
| `20261005-231114-push_across_tuned` | `push_across_tuned.toml` |
| `20261005-231123-push_across_multi_demo_settings` | `push_across_multi_demo_settings.toml` |
| `20261005-231131-push_forward_tuned` | `push_forward_tuned.toml` |
| `20261005-231139-push_forward_multi_demo_settings` | `push_forward_multi_demo_settings.toml` |
| `20261005-231148-push_backward_tuned` | `push_backward_tuned.toml` |
| `20261005-231156-push_backward_multi_demo_settings` | `push_backward_multi_demo_settings.toml` |
| `20261005-231204-block_tuned` | `block_tuned.toml` |
| `20261005-231213-block_multi_demo_settings` | `block_multi_demo_settings.toml` |
| `20261005-234328-nominal_tuned_pd_gains` | `nominal_tuned_pd_gains.toml` |
| `20261005-234338-nominal_multi_demo_settings_pd_gains` | `nominal_multi_demo_settings_pd_gains.toml` |
| `20261005-234328-offsets_tuned_pd_gains` | `offsets_tuned_pd_gains.toml` |
| `20261005-234328-offsets_multi_demo_settings_pd_gains` | `offsets_multi_demo_settings_pd_gains.toml` |
| `20261005-234346-push_across_tuned_pd_gains` | `push_across_tuned_pd_gains.toml` |
| `20261005-234356-push_across_multi_demo_settings_pd_gains` | `push_across_multi_demo_settings_pd_gains.toml` |
| `20261005-234405-push_forward_tuned_pd_gains` | `push_forward_tuned_pd_gains.toml` |
| `20261005-234415-push_forward_multi_demo_settings_pd_gains` | `push_forward_multi_demo_settings_pd_gains.toml` |
| `20261005-234424-push_backward_tuned_pd_gains` | `push_backward_tuned_pd_gains.toml` |
| `20261005-234434-push_backward_multi_demo_settings_pd_gains` | `push_backward_multi_demo_settings_pd_gains.toml` |
| `20261005-234443-block_tuned_pd_gains` | `block_tuned_pd_gains.toml` |
| `20261005-234453-block_multi_demo_settings_pd_gains` | `block_multi_demo_settings_pd_gains.toml` |
| `20261006-121120-nominal_tuned_damping` | `nominal_tuned_damping.toml` |
| `20261006-121145-nominal_multi_demo_settings_damping` | `nominal_multi_demo_settings_damping.toml` |
| `20261006-121208-offsets_tuned_damping` | `offsets_tuned_damping.toml` |
| `20261006-121731-offsets_multi_demo_settings_damping` | `offsets_multi_demo_settings_damping.toml` |
| `20261006-121812-push_across_tuned_damping` | `push_across_tuned_damping.toml` |
| `20261006-121838-push_across_multi_demo_settings_damping` | `push_across_multi_demo_settings_damping.toml` |
| `20261006-121901-push_forward_tuned_damping` | `push_forward_tuned_damping.toml` |
| `20261006-121927-push_forward_multi_demo_settings_damping` | `push_forward_multi_demo_settings_damping.toml` |
| `20261006-121951-push_backward_tuned_damping` | `push_backward_tuned_damping.toml` |
| `20261006-122017-push_backward_multi_demo_settings_damping` | `push_backward_multi_demo_settings_damping.toml` |
| `20261006-122041-block_tuned_damping` | `block_tuned_damping.toml` |
| `20261006-122107-block_multi_demo_settings_damping` | `block_multi_demo_settings_damping.toml` |
