# Manual-demonstration robot tracking (Stage 2)

The procedure of
[single-demonstration robot tracking](../single_demonstration_robot_tracking/README.md)
(report 003), with the demonstration taught by hand: an ESN trained on the take
generates the reference of the simulated robot arm, driven by the arm's measured
posture, and a tracker follows it. Each run is compared with the take replayed by
time. There is no demonstrator to run.

The ESNs are those of
[manual-demonstration autonomous reaching](../manual_demonstration_autonomous_reaching/README.md)
with report 003's two robot ESNs' settings, each trained on the take as recorded
(`_raw`) and filtered (`_filtered`):

| ESN | Settings |
| --- | --- |
| `tuned` | the best settings of report 003's single-demonstration sweeps (`grid_tuned_settings_{raw,filtered}.toml`) |
| `multi_demo_settings` | the settings found best for eight demonstrations (`grid_multi_demo_settings_{raw,filtered}.toml`) |

The replay replays the take its ESN was trained on, so the runs with the raw ESN
compare it with the replayed raw take, and those with the filtered ESN with the
replayed filtered take. Since the replay does not depend on the ESN, the two cover
both ESNs and both replays.

The questions are report 003's (does the reference adapt to the arm's progress, and
can the trackers follow the ESN from offset starts), and one more: how does the
take's stall-and-step, from the recorder's whole-pixel cursor, reach the robot,
through the replay and through the ESN?

## Configurations

`<scenario>_<ESN>_<take>.toml` runs each scenario with each ESN and take. The
scenarios are report 003's; the tracker settings too: computed torque and joint PD,
both critically damped with ω = 10 rad/s. All runs last 16 s; the take lasts 14.8 s.

| Scenario | Starts | Disturbance |
| --- | --- | --- |
| `nominal` | the demonstrated start | none |
| `offsets` | the demonstrated start, then the 169-start grid (−15° to 15° in each joint) | none |
| `push_across` | the demonstrated start | 5 N at the tip for 0.1 s at 4.32 s, across the reach |
| `push_forward` | the demonstrated start | the same, along the reach toward the target |
| `push_backward` | the demonstrated start | the same, along the reach away from the target |
| `block` | the demonstrated start | the tip held from 4.1 s to 4.6 s by a stiff spring-damper |

Report 003 struck its scripted reach at 0.3 s (the block) and 0.4 s (the pushes),
when the hand was 3.4 % and 9.9 % of the way to the target. The take's hand is that
far at 4.10 s and 4.32 s, after its pause, so the disturbances keep their force and
length and start there.

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_robot_tracking/block_tuned_raw.toml
```

`<scenario>_<ESN>_pd_gains_<take>.toml` runs each scenario with joint PD only, at
ω = 10, 20, and 40 rad/s, and `<scenario>_<ESN>_damping_<take>.toml` with the
tracker underdamped: computed torque at ω = 10 rad/s and joint PD at ω = 20 rad/s,
each with the damping ratios 1, 0.5, 0.3, and 0.1, its `offsets` scenario from
(+10°, −10°) and (−10°, −10°) only, as in report 003.
