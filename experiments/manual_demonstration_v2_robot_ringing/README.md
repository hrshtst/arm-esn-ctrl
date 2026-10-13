# Manual-demonstration robot ringing, v2 (Stage 2)

The study after [the wider robot search](../manual_demonstration_v2_robot_search/README.md)
(report 006). There, with joint PD, report 006's chosen ESN made the arm ring less
than the replay of the take after a block or a push: its reference follows the
measured arm, so the tracker starts from a smaller gap. But a push of 0.2 s ended
before the ESN could follow. Here the pushes last 0.5 s, and the search is scored
with joint PD, by the ringing after the block and the pushes.

## The pushes

Three pushes of 0.5 s at the tip, in runs of 24 s: 3 N nearly forward at 3 s
(report 006's backward push, reversed), 5 N nearly across at 7 s, and 4 N nearly
across the other way at 15 s, after the hold. The block stays report 006's: the tip
held from 4.5 s to 5.5 s.

- **0.5 s, not 1 s:** when a 0.5 s push ends, the arm is still moving; after 1 s,
  it has nearly settled where the push holds it, which would make the push a block
  away from the path.
- **The forces across, report 006's:** a push carries the replay farthest at its
  first overshoot, about 0.16 s in, so the replay is carried as far as by report
  006's 0.2 s pushes.
- **The push forward, 3 N:** it sends the ESN ahead on a route of its own (65 mm
  off the path, against the replay's 9 mm). With 4 N, report 006's chosen ESN is
  still 10 mm off the path when the push at 7 s comes, and its hold fails if the
  pushes last 0.55 s; with 3 N, it is 5 mm off, and it holds either way.
- **The times:** the push at 7 s ends 2.1 s before the take arrives, at 9.6 s, so
  that the replay has stopped ringing by then; at 8 s, the replay still rings, and
  the ESN is pushed into the goal early, which spoils the hold. The push at 3 s
  leaves 3.5 s to settle before it.

To watch them with report 006's chosen ESN and the replay, both with joint PD:

```bash
uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_ringing/pushes_robot_lr0.03_sr1.35_is1.5_seed8_filtered.toml
uv run python third_party/skelarm/tools/player.py <run directory>/pd_w20_z0.1/replay_00.sklog.npz
```
