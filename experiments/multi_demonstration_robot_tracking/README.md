# Multi-demonstration robot tracking (Stage 2)

The ESN of [multi-demonstration autonomous reaching](../multi_demonstration_autonomous_reaching/README.md)
becomes the reference generator of the simulated robot arm, compared with the
same demonstrations replayed by time, with and without disturbances. The results
are in [report 002](../../reports/002-esn-reference-on-the-robot/README.md).

## Running the ESN on the robot

`experiments/robot_esn.py` connects a trained ESN to the simulated arm
(`src/arm_esn_ctrl/tracking.py`). Every 10 ms, the ESN's period, the ESN receives
the arm's measured joint angles and gives the posture the arm should have 10 ms
later. Between those instants the reference moves in a straight line, and
skelarm's computed-torque or joint PD law tracks it. The time-indexed baseline
replays the nearest demonstration through the same tracker, and the
demonstrator's own controller reaches from the same posture for comparison. Both
tracked arms hold their start posture during the ESN's warm-up, at negative
times, and the task starts at t = 0.

Every configuration runs the 8 demonstrated starts (or starts offset from them):

| Configuration | Scenario | Tracker |
| --- | --- | --- |
| `nominal.toml` | no disturbance | computed torque and joint PD at ω = 5, 10, 20, and 40 rad/s, critically damped |
| `push.toml` | a 5 N push at the tip, sideways to the reach, for 0.1 s from t = 0.4 s | as above |
| `block.toml` | the tip held by a stiff spring-damper (20 kN/m) from t = 0.3 s to 0.8 s, then released | as above |
| `offset_3deg.toml` | starts 3 deg away from the demonstrated ones, in the four diagonal directions | as above |
| `offset_10deg.toml` | the same, 10 deg away | as above |
| `nominal_damping.toml`, `push_damping.toml`, `block_damping.toml` | as `nominal`, `push`, and `block` | both laws at ω = 10 rad/s, with the damping ratio ζ = 1, 0.5, 0.3, and 0.1 |

```bash
uv run python experiments/robot_esn.py experiments/multi_demonstration_robot_tracking/nominal.toml
```

The configuration names the trained ESN (`model` in `[esn]`, an `esn.toml` saved
by Stage 1, whose run directory names the demonstrations), the tracking laws, the
natural frequencies, and the damping ratios (`[tracker]`), and the start postures
and the run and hold durations (`[evaluation]`, as in Stage 1). The gains come
from the natural frequency ω and the damping ratio ζ of the tracking error
(`omegas` and the optional `dampings`, whose default 1 is critically damped):
kp = ω² and kd = 2ζω for computed torque, scaled by each joint's inertia for joint
PD. A configuration sweeps either ω or ζ. An optional `[disturbance]` table pushes
or blocks all three arms alike (`src/arm_esn_ctrl/disturbances.py`), and
`effort_window` in `[evaluation]` sets when the torque is measured.

The run directory receives the arm's runs for each tracker setting
(`computed_torque_w10/esn_00.sklog.npz`, `replay_00.sklog.npz`, ...;
`pd_w10_z0.3/...` below critical damping), which also record the reference
`q_ref`, the tracking error, and the disturbance force `ext_force` (drawn as an
arrow by the player), and the demonstrator's reaches under the same disturbance
(`demonstrator_00.sklog.npz`, ...). Every run is compared with the demonstrator's
undisturbed reach, and `metrics.csv` holds, besides the reach and hold metrics of
Stage 1:

- over the task: the RMS tracking error, the peak joint speed of the reference (a
  jump of the reference shows as a high speed), the peak hand speed, the settling
  time (from when the hand stays within the goal radius until the end), and the
  final distance to the target;
- over the effort window: the peak joint torque, the integral of the squared joint
  torques, and the peak disturbance force (for a block, how hard the arm pushes
  against it).

`metrics.png` shows those metrics against ω (or ζ), `paths.png` the hand paths,
and `timeline.png` the hand's distance to the target and the joint torque over
time from the first start posture. `tools/robot_app.py` runs the same arms live
(see [`tools/README.md`](../../tools/README.md)).

## Runs

Under `results/multi_demonstration_robot_tracking/` in the storage, all with the
ESN `results/multi_demonstration_autonomous_reaching/20261002-213015-autonomous_tvs_all_distances`:

| Run | Configuration | Report 002 |
| --- | --- | --- |
| `20261005-120454-nominal` | `nominal.toml` | Section 3.2 |
| `20261005-120457-push` | `push.toml` | Section 3.4 |
| `20261005-120459-block` | `block.toml` | Section 3.5 |
| `20261005-120501-offset_3deg` | `offset_3deg.toml` | Section 3.3 |
| `20261005-120503-offset_10deg` | `offset_10deg.toml` | Section 3.3 |
| `20261005-160737-nominal_damping` | `nominal_damping.toml` | Section 3.6 |
| `20261005-160739-push_damping` | `push_damping.toml` | Section 3.6 |
| `20261005-160741-block_damping` | `block_damping.toml` | Section 3.6 |
