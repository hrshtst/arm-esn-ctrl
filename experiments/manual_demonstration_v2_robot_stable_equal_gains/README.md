# Manual-demonstration robot stable equal gains, v2 (Stage 2)

In [report 013](../../reports/013-equal-gains/README.md), joint PD at kp = 80 N m/rad
and kd = 3 N m s/rad for both joints, its torque computed every 10 ms and held
between, chattered at 50 Hz from the start of every run until joint 2 passed about
70°. Here the same arms run undisturbed with equal gains estimated not to.

- **The estimate:** with the same gains for both joints, kp I and kd I, the tracking
  error q̈ = −M⁻¹ (kp e + kd ė) splits along the two eigenvectors of the mass matrix
  M, each a single inertia μ under PD. With the torque held for T = 10 ms, each
  mode's error grows unless kp T / 2 < kd < 2 μ / T. Over the reach, the smaller
  eigenvalue of M is least at the start, μ = 0.00809 kg m² with joint 2 at 30°, so
  every kd under 1.62 N m s/rad is stable throughout, whatever kp. kd = 3 grows below
  joint 2 at 69.5°, as report 013 found.
- **The gains:**
  - kp = 80, kd = 1.2: the stiffness of report 013, and the gains that put both of
    joint 2's mode's poles at zero at the start, kp = μ / T² and kd = 1.5 μ / T. Joint
    1's mode is lightly damped, a damping ratio of 0.13 to 0.15.
  - kp = 16, kd = 1.4: softer, joint 1's mode at about 8 to 9 rad/s and a damping
    ratio of 0.35 to 0.40.
  - kp = 80, kd = 1.7: just past the bound, a check of it; joint 2's mode should grow
    from the start until joint 2 passes 33°.
- **The arms:** report 011's ESN #1 (leak rate 0.03, spectral radius 1.2, input
  scaling 1.5, seed 8) and ESN #2 (input scaling 1, seed 2), and the replay of the take,
  given a zero reference velocity or the reference's.
- **The scenario:** undisturbed, from the demonstrated start.

## Step 1: the runs

[`robot_esn.py`](../robot_esn.py) runs each ESN and the replay with the tracker:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_stable_equal_gains/nominal_kp80_kd1.2_robot_lr0.03_sr1.2_is1.5_seed8_reference_velocity_filtered.toml
```

| Configurations | What they run |
| --- | --- |
| `nominal_{kp80_kd1.2,kp16_kd1.4,kp80_kd1.7}_robot_{lr0.03_sr1.2_is1.5_seed8,lr0.03_sr1.2_is1_seed2}_{zero,reference}_velocity_filtered.toml` (12) | one ESN and the replay, undisturbed, with one pair of gains |

Each is the nominal configuration of the same ESN and reference velocity in
[the equal gains](../manual_demonstration_v2_robot_equal_gains/README.md) with other
gains. Their runs' tracker settings are `pd_kp80_kd1.2`, `pd_kp16_kd1.4`, and
`pd_kp80_kd1.7`.
