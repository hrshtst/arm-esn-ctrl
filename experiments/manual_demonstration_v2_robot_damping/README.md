# Manual-demonstration robot damping, v2 (Stage 2)

Results: [report 008](../../reports/008-damping-and-reference-velocity/README.md).

The study after [the ringing study](../manual_demonstration_v2_robot_ringing/README.md)
(report 007). There, with joint PD at a damping ratio of 0.1, given a zero reference
velocity, the chosen ESN made the arm ring less after a block or a push than the
replay of the take. Here the same ESN and the replay run with joint PD better
damped, and given the reference's velocity, to see whether the ESN still helps when
the tracker itself rings less.

- **The ESN:** report 007's chosen ESN (`grid_lr0.02_sr1.35_is2_seed7_filtered.toml`
  of the ringing study: leak rate 0.02, spectral radius 1.35, input scaling 2, seed
  7), unchanged.
- **The trackers:** joint PD at ω = 20 rad/s with damping ratios 0.1, 0.4, and 0.8,
  each given a zero reference velocity (the derivative term damps the arm's own
  velocity, as in reports 005 to 007) or the reference's velocity (the change of the
  reference over each 10 ms period; the derivative term damps the velocity error).
  Damping ratio 0.1 with a zero reference velocity repeats report 007.
- **The scenarios:** report 007's: undisturbed, the block, the three pushes, and the
  49 offset starts, in runs of 24 s.

The ringing keeps report 007's measure, the hand's motion above 1.5 Hz over 2 s
after a disturbance; the tracker's own ringing falls from 3.2 Hz at a damping ratio
of 0.1 to 2.9 Hz at 0.4 and 1.9 Hz at 0.8, where it hardly oscillates.

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_damping/pushes_robot_pd_reference_velocity_filtered.toml
```

| Configurations | Scenario |
| --- | --- |
| `{nominal,block,pushes,offsets}_robot_pd_zero_velocity_filtered.toml` | joint PD at the three damping ratios, given a zero reference velocity |
| `{nominal,block,pushes,offsets}_robot_pd_reference_velocity_filtered.toml` | the same, given the reference's velocity |
| `{nominal,block,pushes,offsets}_robot_pd_{zero,reference}_velocity_control10ms_filtered.toml` | the same eight, but with the tracker's torque computed every 10 ms, the ESN's period, and held between (`control_period`), while the arm is still simulated every 2 ms |
