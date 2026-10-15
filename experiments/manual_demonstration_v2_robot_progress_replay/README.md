# Manual-demonstration robot progress replay, v2 (Stage 2)

Reports 011 to 013 compared the ESNs with the take replayed by time. That replay
ignores the arm: held, the arm falls behind a reference that runs on, and the tracker
presses harder and harder against what holds it. A replay that follows the arm's
progress along the taught path also waits for a held arm, without an ESN. Here it is
a third arm, the baseline the ESNs' waiting and following should be measured against.

- **The progress replay** ([`ProgressReplaySource`](../../src/arm_esn_ctrl/tracking.py)):
  every 10 ms, the arm's progress is the length of the taught joint path up to the
  sample nearest the measured joint angles, searched within 1 s of the last one
  found. The replayed sample advances by one each period, as the replay's does, but
  never to where the path is more than 2.5° past the arm's progress: it waits for a
  held arm, and falls back with an arm pushed back along the path, by at most one
  sample a period, the take's pace in reverse. Where the take rests, its path does not
  lengthen, so the replayed sample passes through the rest. Pushed across the path or
  forward along it, the arm is pulled back as by the replay.
- **The lead:** undisturbed, the replay's arm lags its reference by at most about
  2.2° along the path (at the reach's fastest, 50°/s), with either tracker and either
  reference velocity, so that with 2.5° the progress replay is the replay: in a check
  run, their joint angles agree to within 10⁻⁴°. Under the block, it presses with
  1.0 to 1.1 N m at the damping ratio 0.4 and 3.2 to 3.3 N m at kp = 80 and kd = 1.2,
  against the replay's 12 to 13 and 17 N m.
- **The arms:** report 011's ESN #1 (leak rate 0.03, spectral radius 1.2, input
  scaling 1.5, seed 8) and ESN #2 (input scaling 1, seed 2), the replay of the take,
  and the progress replay.
- **The trackers:** joint PD at ω = 20 rad/s and ζ = 0.4, each joint's gains scaled by
  its inertia, as in reports 011 and 012, and kp = 80 N m/rad and kd = 1.2 N m s/rad
  for both joints, as in report 013; given a zero reference velocity or the
  reference's, the torque computed every 10 ms.
- **The scenarios:** report 011's undisturbed, block, pushes, and 49 offset starts,
  and report 012's three logs of pushes given by hand, replayed by time.

## Step 1: the runs

The first runs, at `a0ff165`, let the reference fall back at once to 2.5° past the arm's
progress. Under the strongest pushes by hand, the arm's nearest sample jumped back along
the path, and so did the reference, at up to 1,556°/s, throwing the arm. From `98c1a25`,
it falls back by at most one sample a period, and the same configurations are run
again.

[`robot_esn.py`](../robot_esn.py) runs each ESN, the replay, and the progress replay
(its `[progress]` table) with the tracker:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_progress_replay/pushes_w20_z0.4_robot_lr0.03_sr1.2_is1.5_seed8_reference_velocity_filtered.toml
```

| Configurations | What they run |
| --- | --- |
| `{nominal,block,pushes,offsets,recorded000,recorded001,recorded002}_{w20_z0.4,kp80_kd1.2}_robot_{lr0.03_sr1.2_is1.5_seed8,lr0.03_sr1.2_is1_seed2}_{zero,reference}_velocity_filtered.toml` (56) | one ESN, the replay, and the progress replay in one scenario, with one tracker |

Each is the configuration of the same scenario, ESN, and reference velocity in
[the stable equal gains](../manual_demonstration_v2_robot_stable_equal_gains/README.md)
with a `[progress]` table, and, for `w20_z0.4`, reports 011's and 012's joint PD at
the damping ratio 0.4 in place of the equal gains. Their runs' tracker settings are
`pd_w20_z0.4` and `pd_kp80_kd1.2`, and the progress replay's runs
`progress_<start>.sklog.npz`.
