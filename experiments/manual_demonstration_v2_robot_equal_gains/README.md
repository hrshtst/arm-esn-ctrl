# Manual-demonstration robot equal gains, v2 (Stage 2)

Reports 011 and 012 ran joint PD with each joint's gains scaled by its inertia at the
end of the reach, so that both joints' tracking errors have the same natural frequency
and damping ratio: at ω = 20 rad/s, kp = 74.97 and 8.96 N m/rad, and kd = 1.50 to 6.00
and 0.18 to 0.72 N m s/rad over the damping ratios 0.2 to 0.8. Here both joints take
the same gains instead, kp = 80 N m/rad and kd = 3 N m s/rad, given directly in
`[tracker]`: about the same for joint 1, and about nine times stiffer for joint 2,
whose inertia is about an eighth of joint 1's. Every scenario of reports 011 and 012 is
run again with this tracker. There is no search and no rule: every arm is shown. With
the torque held for 10 ms, joint 2's torque chatters from the start of every run;
[report 013](../../reports/013-equal-gains/README.md) explains why, from the
undisturbed runs, and runs equal gains that do not chatter
([the stable equal gains](../manual_demonstration_v2_robot_stable_equal_gains/README.md)).
Its earlier version, at commit `55072ed`, reports every run here.

- **The arms:** report 011's ESN #1 (leak rate 0.03, spectral radius 1.2, input
  scaling 1.5, seed 8) and ESN #2 (input scaling 1, seed 2), and the replay of the take.
- **The tracker:** joint PD at kp = 80 N m/rad and kd = 3 N m s/rad for both joints,
  given a zero reference velocity or the reference's, with the torque computed every
  10 ms, as in reports 011 and 012.
- **The scenarios:** report 011's undisturbed, block, pushes, and 49 offset starts,
  and report 012's three logs of pushes given by hand, replayed by time.

## Step 1: the runs

[`robot_esn.py`](../robot_esn.py) runs each ESN and the replay with the tracker:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_equal_gains/pushes_robot_lr0.03_sr1.2_is1.5_seed8_reference_velocity_filtered.toml
```

| Configurations | What they run |
| --- | --- |
| `{nominal,block,pushes,offsets}_robot_{lr0.03_sr1.2_is1.5_seed8,lr0.03_sr1.2_is1_seed2}_{zero,reference}_velocity_filtered.toml` (16) | one ESN and the replay in one of report 011's scenarios |
| `recorded{000,001,002}_robot_{lr0.03_sr1.2_is1.5_seed8,lr0.03_sr1.2_is1_seed2}_{zero,reference}_velocity_filtered.toml` (12) | one ESN and the replay under one of report 012's recorded forces |

Each is the configuration of the same name in
[the following search](../manual_demonstration_v2_robot_following_search/README.md) or
[the recorded pushes](../manual_demonstration_v2_robot_recorded_pushes/README.md),
with the equal gains in place of the three damping ratios. Their runs' tracker
setting is `pd_kp80_kd3`.
