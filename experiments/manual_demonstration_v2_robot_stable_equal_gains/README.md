# Manual-demonstration robot stable equal gains, v2 (Stage 2)

Joint PD at kp = 80 N m/rad and kd = 3 N m s/rad for both joints, its torque computed
every 10 ms and held between ([the equal gains](../manual_demonstration_v2_robot_equal_gains/README.md)),
chatters at 50 Hz from the start of every run until joint 2 passes about 77°. Here the
same arms run with equal gains estimated not to, in every scenario of reports 011 and
012. The results are in [report 013](../../reports/013-equal-gains/README.md).

- **The estimate:** with the same gains for both joints, kp I and kd I, the tracking
  error q̈ = −M⁻¹ (kp e + kd ė) splits along the two eigenvectors of the mass matrix
  M, each a single inertia μ under PD. With the torque held for T = 10 ms, and the arm
  simulated in five semi-implicit Euler steps of 2 ms, each mode's error grows unless
  kd < 2 μ / T − kp T / 10. Over the reach, the smaller eigenvalue of M is least at the
  start, μ = 0.00809 kg m² with joint 2 at 30°, so that kd must be under 1.54 N m s/rad
  at kp = 80, and under 1.60 at kp = 16. kd = 3 grows below joint 2 at 71.5°.
- **The gains:**
  - kp = 80, kd = 1.2: the stiffness of the equal gains, inside the bound down to joint
    2 at 10.2°. Joint 1's mode is lightly damped, a damping ratio of 0.13 to 0.15.
  - kp = 16, kd = 1.4: softer, joint 1's mode at about 8 to 9 rad/s and a damping
    ratio of 0.35 to 0.40, inside the bound down to joint 2 at 20.5°, so not from the
    offset starts with joint 2 at 15° and 20°.
  - kp = 80, kd = 1.7, undisturbed only: just past the bound, a check of it; joint 2's
    mode should grow from the start until joint 2 passes 36°.
- **The arms:** report 011's ESN #1 (leak rate 0.03, spectral radius 1.2, input
  scaling 1.5, seed 8) and ESN #2 (input scaling 1, seed 2), and the replay of the take,
  given a zero reference velocity or the reference's.
- **The scenarios:** report 011's undisturbed, block, pushes, and 49 offset starts,
  and report 012's three logs of pushes given by hand, replayed by time.

## Step 1: the runs

[`robot_esn.py`](../robot_esn.py) runs each ESN and the replay with the tracker:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_stable_equal_gains/nominal_kp80_kd1.2_robot_lr0.03_sr1.2_is1.5_seed8_reference_velocity_filtered.toml
```

| Configurations | What they run |
| --- | --- |
| `{nominal,block,pushes,offsets,recorded000,recorded001,recorded002}_{kp80_kd1.2,kp16_kd1.4}_robot_{lr0.03_sr1.2_is1.5_seed8,lr0.03_sr1.2_is1_seed2}_{zero,reference}_velocity_filtered.toml` (56) | one ESN and the replay in one scenario, with one pair of gains |
| `nominal_kp80_kd1.7_robot_{lr0.03_sr1.2_is1.5_seed8,lr0.03_sr1.2_is1_seed2}_{zero,reference}_velocity_filtered.toml` (4) | the check of the bound, undisturbed |

Each is the configuration of the same scenario, ESN, and reference velocity in
[the equal gains](../manual_demonstration_v2_robot_equal_gains/README.md) with other
gains. Their runs' tracker settings are `pd_kp80_kd1.2`, `pd_kp16_kd1.4`, and
`pd_kp80_kd1.7`. The undisturbed runs were made first, at commit `c421488`, whose
configurations differ from these only in their comments on the bound, then 2 μ / T =
1.62 N m s/rad, before the simulation's own steps were taken into account.
