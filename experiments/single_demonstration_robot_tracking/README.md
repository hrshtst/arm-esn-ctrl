# Single-demonstration robot tracking (Stage 2)

An ESN trained on one demonstration generates the reference of the simulated robot
arm, driven by the arm's measured posture, and a tracker follows it, as in
[multi-demonstration robot tracking](../multi_demonstration_robot_tracking/README.md).
Each run is compared with the demonstration replayed by time and with the
demonstrator's own controller.

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

None yet.
