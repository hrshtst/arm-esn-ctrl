# Manual-demonstration ESN dynamics, v2

Reports 011 to 014 measured how report 011's two ESNs drive the robot. This study
asks what they learned, as dynamical systems: is the take's target an equilibrium
that attracts the ESN from anywhere, is the take's path an attractor that the ESN
returns to when pushed off it, or does the ESN time its reach from the reset, like a
clock?

- **The ESNs:** report 011's ESN #1 (leak rate 0.03, spectral radius 1.2, input
  scaling 1.5, seed 8) and ESN #2 (input scaling 1, seed 2), trained on the take.
- **Step 1, from rest anywhere** ([`global_starts.py`](../global_starts.py)): each
  ESN on its own, fed its own output, from start postures all over both joints'
  range, every 10° (1,296), each held for the ESN's 1 s warm-up first, as in
  training, for 20 s. Each start posture is also held for 4 s, to see whether the
  state the ESN starts from depends on the start posture alone.
- **Step 2, pushed along the reach** ([`push_pulses.py`](../push_pulses.py)): on the
  robot, each ESN, the replay of the take by time, and its replay by the arm's
  progress (report 014's), pushed at the tip by one short pulse a run: 3 N or 6 N
  for 0.2 s, every second from 1 s to 9 s and at 12 s, at the target, in 8
  directions (160 pulses, and one undisturbed run, per arm), with joint PD at
  ω = 20 rad/s and ζ = 0.4, given the reference's velocity, its torque computed
  every 10 ms.

A posture kick to an ESN running on its own does not show whether the path attracts
it: the ESN reads its output from a reservoir that follows its input over about
0.33 s (leak rate 0.03), so a kick of 15° for one step moves its output by 0.6°.
A push on the robot displaces the arm, which the ESN then sees for as long as it
stays displaced.

## Step 1: from rest anywhere

```bash
uv run python experiments/global_starts.py experiments/manual_demonstration_v2_esn_dynamics/global_starts.toml
```

| Configuration | What it runs |
| --- | --- |
| `global_starts.toml` | ESN #1 and ESN #2, each from 1,296 start postures every 10°, 20 s |

## Step 1b: the warm-up

Step 1 found that when the 1 s warm-up ends, the state is still moving as fast as the
take's states move during the reach, and it takes another second or two to settle.
[`warmup_esn.py`](../warmup_esn.py) runs each ESN with other warm-ups than the 1 s
it was trained with, from report 011's 49 offset starts and from three starts on the
take's path, at rest: a warm-up longer by some time makes an ESN that times its reach
from the reset arrive earlier by that time; one that times it from the end of the
warm-up arrives as before.

```bash
uv run python experiments/warmup_esn.py experiments/manual_demonstration_v2_esn_dynamics/warmup_lr0.03_sr1.2_is1.5_seed8.toml
```

| Configurations | What they run |
| --- | --- |
| `warmup_{lr0.03_sr1.2_is1.5_seed8,lr0.03_sr1.2_is1_seed2}.toml` | one ESN with warm-ups of 0.25, 0.5, 1, 2, and 4 s, from 52 starts, 20 s |

## Step 2: pushed along the reach

```bash
uv run python experiments/push_pulses.py experiments/manual_demonstration_v2_esn_dynamics/push_pulses.toml
```

| Configuration | What it runs |
| --- | --- |
| `push_pulses.toml` | ESN #1, ESN #2, the replay, and the progress replay, each undisturbed and under 160 pulses: 644 runs |
