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

## Candidate F

Report 003's ESNs do not suit the take: the tuned one stays where the hand paused,
and the one with the settings for eight demonstrations diverges on the robot.
The sweeps of manual-demonstration autonomous reaching found candidate F (leak rate
0.7, spectral radius 1.05, input scaling 0.3, ridge 1e-2, warm-up 1 s, trained on
the take as recorded), which returns mildly onto the taught path from every start
of the grid, without swinging or jumping.

`<scenario>_candidate_f{,_pd_gains,_damping}_raw.toml` run it through the same
scenarios and tracker settings as above, with the forward and backward pushes at 10 N
instead of 5 N; the replay replays the take as recorded.
The disturbances strike at the same times, when the take's hand starts toward the
target; candidate F, on its own, reaches about 2 s earlier, so they strike it
mid-reach.

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_robot_tracking/block_candidate_f_raw.toml
```

`<scenario>_candidate_f_{pd_gains,damping}_zero_velocity_raw.toml` repeat its joint
PD runs with the tracking law given a zero reference velocity
(`reference_velocity = false` in `[tracker]`), so that the derivative term damps the
arm's own velocity rather than the velocity error: joint PD at ω = 10, 20, and
40 rad/s, and at ω = 20 rad/s with the damping ratios 1, 0.5, 0.3, and 0.1.

## Runs

Under `results/manual_demonstration_robot_tracking/` in the storage, with the ESNs of
the grid runs of manual-demonstration autonomous reaching:

| Run | Configuration |
| --- | --- |
| `20261006-153505-offsets_multi_demo_settings_damping_filtered` | `offsets_multi_demo_settings_damping_filtered.toml` |
| `20261006-153505-offsets_multi_demo_settings_damping_raw` | `offsets_multi_demo_settings_damping_raw.toml` |
| `20261006-153505-offsets_multi_demo_settings_filtered` | `offsets_multi_demo_settings_filtered.toml` |
| `20261006-153505-offsets_multi_demo_settings_pd_gains_filtered` | `offsets_multi_demo_settings_pd_gains_filtered.toml` |
| `20261006-153505-offsets_multi_demo_settings_pd_gains_raw` | `offsets_multi_demo_settings_pd_gains_raw.toml` |
| `20261006-153505-offsets_multi_demo_settings_raw` | `offsets_multi_demo_settings_raw.toml` |
| `20261006-153505-offsets_tuned_damping_filtered` | `offsets_tuned_damping_filtered.toml` |
| `20261006-153505-offsets_tuned_damping_raw` | `offsets_tuned_damping_raw.toml` |
| `20261006-153734-offsets_tuned_filtered` | `offsets_tuned_filtered.toml` |
| `20261006-153735-offsets_tuned_pd_gains_filtered` | `offsets_tuned_pd_gains_filtered.toml` |
| `20261006-153736-offsets_tuned_pd_gains_raw` | `offsets_tuned_pd_gains_raw.toml` |
| `20261006-153738-offsets_tuned_raw` | `offsets_tuned_raw.toml` |
| `20261006-162519-block_multi_demo_settings_damping_filtered` | `block_multi_demo_settings_damping_filtered.toml` |
| `20261006-162522-block_multi_demo_settings_damping_raw` | `block_multi_demo_settings_damping_raw.toml` |
| `20261006-162558-block_multi_demo_settings_filtered` | `block_multi_demo_settings_filtered.toml` |
| `20261006-162621-block_multi_demo_settings_pd_gains_filtered` | `block_multi_demo_settings_pd_gains_filtered.toml` |
| `20261006-162621-block_multi_demo_settings_pd_gains_raw` | `block_multi_demo_settings_pd_gains_raw.toml` |
| `20261006-162649-block_multi_demo_settings_raw` | `block_multi_demo_settings_raw.toml` |
| `20261006-162650-block_tuned_damping_filtered` | `block_tuned_damping_filtered.toml` |
| `20261006-162651-block_tuned_damping_raw` | `block_tuned_damping_raw.toml` |
| `20261006-162651-block_tuned_filtered` | `block_tuned_filtered.toml` |
| `20261006-162712-block_tuned_pd_gains_filtered` | `block_tuned_pd_gains_filtered.toml` |
| `20261006-162713-block_tuned_pd_gains_raw` | `block_tuned_pd_gains_raw.toml` |
| `20261006-162716-block_tuned_raw` | `block_tuned_raw.toml` |
| `20261006-162740-nominal_multi_demo_settings_damping_filtered` | `nominal_multi_demo_settings_damping_filtered.toml` |
| `20261006-162742-nominal_multi_demo_settings_damping_raw` | `nominal_multi_demo_settings_damping_raw.toml` |
| `20261006-162743-nominal_multi_demo_settings_filtered` | `nominal_multi_demo_settings_filtered.toml` |
| `20261006-162744-nominal_multi_demo_settings_pd_gains_filtered` | `nominal_multi_demo_settings_pd_gains_filtered.toml` |
| `20261006-162806-nominal_multi_demo_settings_pd_gains_raw` | `nominal_multi_demo_settings_pd_gains_raw.toml` |
| `20261006-162813-nominal_multi_demo_settings_raw` | `nominal_multi_demo_settings_raw.toml` |
| `20261006-162823-nominal_tuned_damping_filtered` | `nominal_tuned_damping_filtered.toml` |
| `20261006-162824-nominal_tuned_damping_raw` | `nominal_tuned_damping_raw.toml` |
| `20261006-162834-nominal_tuned_filtered` | `nominal_tuned_filtered.toml` |
| `20261006-162836-nominal_tuned_pd_gains_filtered` | `nominal_tuned_pd_gains_filtered.toml` |
| `20261006-162859-nominal_tuned_pd_gains_raw` | `nominal_tuned_pd_gains_raw.toml` |
| `20261006-162906-nominal_tuned_raw` | `nominal_tuned_raw.toml` |
| `20261006-162909-push_across_multi_demo_settings_damping_filtered` | `push_across_multi_demo_settings_damping_filtered.toml` |
| `20261006-162911-push_across_multi_demo_settings_damping_raw` | `push_across_multi_demo_settings_damping_raw.toml` |
| `20261006-162928-push_across_multi_demo_settings_filtered` | `push_across_multi_demo_settings_filtered.toml` |
| `20261006-162930-push_across_multi_demo_settings_pd_gains_filtered` | `push_across_multi_demo_settings_pd_gains_filtered.toml` |
| `20261006-162952-push_across_multi_demo_settings_pd_gains_raw` | `push_across_multi_demo_settings_pd_gains_raw.toml` |
| `20261006-162956-push_across_multi_demo_settings_raw` | `push_across_multi_demo_settings_raw.toml` |
| `20261006-162957-push_across_tuned_damping_filtered` | `push_across_tuned_damping_filtered.toml` |
| `20261006-162959-push_across_tuned_damping_raw` | `push_across_tuned_damping_raw.toml` |
| `20261006-163019-push_across_tuned_filtered` | `push_across_tuned_filtered.toml` |
| `20261006-163021-push_across_tuned_pd_gains_filtered` | `push_across_tuned_pd_gains_filtered.toml` |
| `20261006-163040-push_across_tuned_pd_gains_raw` | `push_across_tuned_pd_gains_raw.toml` |
| `20261006-163040-push_across_tuned_raw` | `push_across_tuned_raw.toml` |
| `20261006-163044-push_backward_multi_demo_settings_damping_filtered` | `push_backward_multi_demo_settings_damping_filtered.toml` |
| `20261006-163049-push_backward_multi_demo_settings_damping_raw` | `push_backward_multi_demo_settings_damping_raw.toml` |
| `20261006-163049-push_backward_multi_demo_settings_filtered` | `push_backward_multi_demo_settings_filtered.toml` |
| `20261006-163052-push_backward_multi_demo_settings_pd_gains_filtered` | `push_backward_multi_demo_settings_pd_gains_filtered.toml` |
| `20261006-163105-push_backward_multi_demo_settings_pd_gains_raw` | `push_backward_multi_demo_settings_pd_gains_raw.toml` |
| `20261006-163111-push_backward_multi_demo_settings_raw` | `push_backward_multi_demo_settings_raw.toml` |
| `20261006-163113-push_backward_tuned_damping_filtered` | `push_backward_tuned_damping_filtered.toml` |
| `20261006-163122-push_backward_tuned_damping_raw` | `push_backward_tuned_damping_raw.toml` |
| `20261006-163130-push_backward_tuned_filtered` | `push_backward_tuned_filtered.toml` |
| `20261006-163133-push_backward_tuned_pd_gains_filtered` | `push_backward_tuned_pd_gains_filtered.toml` |
| `20261006-163134-push_backward_tuned_pd_gains_raw` | `push_backward_tuned_pd_gains_raw.toml` |
| `20261006-163135-push_backward_tuned_raw` | `push_backward_tuned_raw.toml` |
| `20261006-163154-push_forward_multi_demo_settings_damping_filtered` | `push_forward_multi_demo_settings_damping_filtered.toml` |
| `20261006-163159-push_forward_multi_demo_settings_damping_raw` | `push_forward_multi_demo_settings_damping_raw.toml` |
| `20261006-163203-push_forward_multi_demo_settings_filtered` | `push_forward_multi_demo_settings_filtered.toml` |
| `20261006-163204-push_forward_multi_demo_settings_pd_gains_filtered` | `push_forward_multi_demo_settings_pd_gains_filtered.toml` |
| `20261006-163214-push_forward_multi_demo_settings_pd_gains_raw` | `push_forward_multi_demo_settings_pd_gains_raw.toml` |
| `20261006-163219-push_forward_multi_demo_settings_raw` | `push_forward_multi_demo_settings_raw.toml` |
| `20261006-163226-push_forward_tuned_damping_filtered` | `push_forward_tuned_damping_filtered.toml` |
| `20261006-163233-push_forward_tuned_damping_raw` | `push_forward_tuned_damping_raw.toml` |
| `20261006-163243-push_forward_tuned_filtered` | `push_forward_tuned_filtered.toml` |
| `20261006-163244-push_forward_tuned_pd_gains_filtered` | `push_forward_tuned_pd_gains_filtered.toml` |
| `20261006-163247-push_forward_tuned_pd_gains_raw` | `push_forward_tuned_pd_gains_raw.toml` |
| `20261006-163254-push_forward_tuned_raw` | `push_forward_tuned_raw.toml` |
| `20261006-225553-block_candidate_f_damping_raw` | `block_candidate_f_damping_raw.toml` |
| `20261006-225553-block_candidate_f_pd_gains_raw` | `block_candidate_f_pd_gains_raw.toml` |
| `20261006-225553-block_candidate_f_raw` | `block_candidate_f_raw.toml` |
| `20261006-225553-nominal_candidate_f_damping_raw` | `nominal_candidate_f_damping_raw.toml` |
| `20261006-225553-nominal_candidate_f_pd_gains_raw` | `nominal_candidate_f_pd_gains_raw.toml` |
| `20261006-225553-offsets_candidate_f_damping_raw` | `offsets_candidate_f_damping_raw.toml` |
| `20261006-225553-offsets_candidate_f_pd_gains_raw` | `offsets_candidate_f_pd_gains_raw.toml` |
| `20261006-225553-offsets_candidate_f_raw` | `offsets_candidate_f_raw.toml` |
| `20261006-225614-nominal_candidate_f_raw` | `nominal_candidate_f_raw.toml` |
| `20261006-225619-push_across_candidate_f_damping_raw` | `push_across_candidate_f_damping_raw.toml` |
| `20261006-225620-push_across_candidate_f_pd_gains_raw` | `push_across_candidate_f_pd_gains_raw.toml` |
| `20261006-225635-push_across_candidate_f_raw` | `push_across_candidate_f_raw.toml` |
| `20261007-003452-push_backward_candidate_f_damping_raw` | `push_backward_candidate_f_damping_raw.toml` |
| `20261007-003452-push_backward_candidate_f_pd_gains_raw` | `push_backward_candidate_f_pd_gains_raw.toml` |
| `20261007-003452-push_backward_candidate_f_raw` | `push_backward_candidate_f_raw.toml` |
| `20261007-003452-push_forward_candidate_f_damping_raw` | `push_forward_candidate_f_damping_raw.toml` |
| `20261007-003452-push_forward_candidate_f_pd_gains_raw` | `push_forward_candidate_f_pd_gains_raw.toml` |
| `20261007-003452-push_forward_candidate_f_raw` | `push_forward_candidate_f_raw.toml` |
| `20261007-103933-block_candidate_f_damping_zero_velocity_raw` | `block_candidate_f_damping_zero_velocity_raw.toml` |
| `20261007-103933-block_candidate_f_pd_gains_zero_velocity_raw` | `block_candidate_f_pd_gains_zero_velocity_raw.toml` |
| `20261007-103933-nominal_candidate_f_damping_zero_velocity_raw` | `nominal_candidate_f_damping_zero_velocity_raw.toml` |
| `20261007-103933-nominal_candidate_f_pd_gains_zero_velocity_raw` | `nominal_candidate_f_pd_gains_zero_velocity_raw.toml` |
| `20261007-103933-offsets_candidate_f_damping_zero_velocity_raw` | `offsets_candidate_f_damping_zero_velocity_raw.toml` |
| `20261007-103933-offsets_candidate_f_pd_gains_zero_velocity_raw` | `offsets_candidate_f_pd_gains_zero_velocity_raw.toml` |
| `20261007-103933-push_across_candidate_f_damping_zero_velocity_raw` | `push_across_candidate_f_damping_zero_velocity_raw.toml` |
| `20261007-103933-push_across_candidate_f_pd_gains_zero_velocity_raw` | `push_across_candidate_f_pd_gains_zero_velocity_raw.toml` |
| `20261007-103933-push_backward_candidate_f_damping_zero_velocity_raw` | `push_backward_candidate_f_damping_zero_velocity_raw.toml` |
| `20261007-103933-push_backward_candidate_f_pd_gains_zero_velocity_raw` | `push_backward_candidate_f_pd_gains_zero_velocity_raw.toml` |
| `20261007-103933-push_forward_candidate_f_damping_zero_velocity_raw` | `push_forward_candidate_f_damping_zero_velocity_raw.toml` |
| `20261007-103933-push_forward_candidate_f_pd_gains_zero_velocity_raw` | `push_forward_candidate_f_pd_gains_zero_velocity_raw.toml` |
