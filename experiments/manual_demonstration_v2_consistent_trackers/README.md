# Manual-demonstration consistent trackers, v2 (Stage 2)

The second part of the study run under one protocol, which
[the search's README](../manual_demonstration_v2_consistent_search/README.md#the-protocol)
holds with this part's plan
([Part 2](../manual_demonstration_v2_consistent_search/README.md#part-2-the-trackers-report-018)):
the two ESNs that [the search](../manual_demonstration_v2_consistent_search/README.md)
chose ([report 017](../../reports/017-search-under-one-protocol/README.md)), with every
tracker of the protocol, in every scenario of the search and under the pushes given by
hand in report 012. Results: report 018, to be written.

- **The arms:** ESN #1 (leak rate 0.03, spectral radius 1.2, input scaling 1.5,
  seed 8) and ESN #2 (input scaling 1, seed 2), as the search trained them
  (`results/manual_demonstration_v2_consistent_search/20261024-205825-esn1_lr0.03_sr1.2_is1.5_seed8`
  and `20261024-205835-esn2_lr0.03_sr1.2_is1_seed2`); the take replayed by time; and
  the take replayed by the arm's progress (report 014's progress replay).
- **The trackers:** nine, all at ω = 20 rad/s, their torque computed every 10 ms and
  held between: joint PD at damping ratios 0.2, 0.4, and 0.8, each given a zero
  reference velocity or the reference's; and computed torque at the same damping
  ratios, given the reference's velocity. Joint PD at 0.4, given the reference's
  velocity, is the search's tracker, so its runs repeat the search's step 4.
- **The scenarios:** the search's four (undisturbed, the block, the pushes, and the 49
  offset starts), and report 012's three recordings of pushes given by hand in the
  robot app, replayed by time from the log's start posture, (30°, 30°), within 0.002°
  of the take's:

| Log (under the storage root's `data/recorded_forces/`) | Pushes (above 0.5 N, joined across lulls under 0.2 s) | Lengths | Peak force | Impulse |
| --- | --- | --- | --- | --- |
| `recorded_forces_000.sklog.npz` | 4, from 2.4, 6.1, 8.6, and 14.0 s | 0.38 to 0.60 s | 2.8 to 7.8 N | 0.9 to 2.4 N s |
| `recorded_forces_001.sklog.npz` | 3, from 2.3, 7.2, and 14.4 s | 1.08, 1.26, and 0.46 s | 7.0, 14.4, and 6.2 N | 5.5, 10.6, and 2.0 N s |
| `recorded_forces_002.sklog.npz` | 3, from 2.6, 6.0, and 13.6 s | 0.52 to 0.62 s | 3.7 to 9.1 N | 1.3 to 3.7 N s |

- **The measures:** the search's, of each arm with each tracker (see
  [The protocol](../manual_demonstration_v2_consistent_search/README.md#the-protocol)).
  An arm passes a scenario as the rule's first point says, the recorded pushes as the
  scripted ones: it arrives and ends within the goal. How hard the motors work is
  measured over the whole task (`task_torque_nms`, `task_effort_n2m2s`, and
  `task_work_j` in `metrics.csv`), and each disturbed run's is set against the same
  arm's undisturbed run with the same tracker.

## The runs

[`robot_esn.py`](../robot_esn.py) runs one ESN, the replay, and the progress replay
with three or six trackers, in one scenario:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_consistent_trackers/pushes_robot_esn1_lr0.03_sr1.2_is1.5_seed8_reference_velocity.toml
```

| Configurations | What they run |
| --- | --- |
| `{nominal,block,pushes,offsets,recorded000,recorded001,recorded002}_robot_{esn1_lr0.03_sr1.2_is1.5_seed8,esn2_lr0.03_sr1.2_is1_seed2}_reference_velocity.toml` (14) | one ESN and both replays, joint PD and computed torque at damping ratios 0.2, 0.4, and 0.8, given the reference's velocity, in one scenario |
| `{nominal,block,pushes,offsets,recorded000,recorded001,recorded002}_robot_{esn1_lr0.03_sr1.2_is1.5_seed8,esn2_lr0.03_sr1.2_is1_seed2}_zero_velocity.toml` (14) | the same with joint PD alone, given a zero reference velocity |

The search's scenarios are configured as in its step 4, but for the trackers, and the
recorded pushes as its pushes, but for the force and the start posture. The replays
do not depend on the ESN they run beside, so each runs twice, once beside each ESN.
