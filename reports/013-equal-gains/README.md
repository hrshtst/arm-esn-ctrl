# 013 Equal gains for both joints

**Summary.** Reports 011 and 012 ran joint PD with each joint's gains scaled by its
inertia. With the same gains for both joints instead, kp = 80 N m/rad and
kd = 3 N m s/rad, joint 2's torque chatters at 50 Hz from the start of every run, up
to thousands of N m, while the torque is computed every 10 ms and held between. Here
the cause is worked out, the equal gains that keep the held torque stable are
estimated, and two of them are run in every scenario of reports 011 and 012:
undisturbed, the block, report 011's three pushes, the 49 offset starts, and report
012's ten pushes by hand, for ESN #1, ESN #2, and the replay of the take, given a zero
reference velocity or the reference's. They are compared with reports 011's and 012's
runs at ω = 20 rad/s and ζ = 0.4.

- **The bound:** with the same gains for both joints, the tracking error splits along
  the eigenvectors of the arm's mass matrix, each a single inertia μ under PD. With
  the torque held for T = 10 ms, as the simulation steps the arm, a mode grows unless
  **kd < 2 μ / T − kp T / 10**. The smaller inertia is least at the start of the
  reach, 0.00809 kg m² with joint 2 at 30°, so that kd must be under 1.54 N m s/rad
  at kp = 80: kd = 3 grows wherever joint 2 is under 71.5°.
- **The bound holds.** Undisturbed, kd = 1.7, just past it, chatters as predicted: the
  flipping torque grows by 1.25 to 1.31 per period against the model's 1.26, and dies
  out once the reference takes joint 2 past 36.1°; kd = 3 grows by 3.06 to 3.50
  against 3.06, until joint 2 passes 71.5°.
- **The two gains inside it never chatter from the demonstrated start:** kp = 80 and
  kd = 1.2, the stiffness of kd = 3 with a lightly damped joint 1 (ζ = 0.15), and
  kp = 16 and kd = 1.4, a softer arm (joint 1 at ω = 9.2 rad/s, ζ = 0.40).
  Undisturbed, both track the taught path within 1.2 mm (RMS) given the reference
  velocity, as report 011's gains do. From the offset starts with joint 2 at 15° and
  20°, below or at kp = 16's bound of 20.5°, its ESNs' torque flips for up to 1.8 s,
  at up to 57 N m.
- **kp = 80 and kd = 1.2 keep the arm on the path under pushes.** Over the 13 pushes,
  in the median, the hand strays 25 to 30 mm from the taught path (the replay's
  15 mm) against 102 to 150 mm (70 mm) with report 011's gains, rings 2.6 to 3.5 mm
  against 5.9 to 7.6 mm, and the ESNs' hands are back within 0.7 to 1.1 s against 2.8
  to 3.5 s, after every push. The ESNs follow the pushes less, 0.54 to 0.64 against
  0.69 to 0.80.
- **Its cost is joint 1's light damping:** under logs 000 and 001, the hand reaches
  the goal while it still rings from the push before and swings out of it for about
  0.1 s, so that five of its 18 runs under the logs fail to hold. It presses on the
  block with 7 to 8 N m against 1 to 2 N m.
- **kp = 16 and kd = 1.4 give way as report 011's gains do,** 97 to 121 mm, and come
  back as slowly, in 3.0 to 4.1 s; under log 001, given a zero reference velocity,
  ESN #2's reference never comes back to the reach, and ends 841 mm from the target.

## 1. Question

1. Why does joint PD at kp = 80 N m/rad and kd = 3 N m s/rad, the same for both
   joints, chatter with the torque held for 10 ms, and which equal gains do not?
2. How do report 011's two ESNs and the replay of the take do with such gains,
   against reports 011 and 012's gains scaled by each joint's inertia, in every
   scenario of the two reports?

## 2. Setup

### The arms, the scenarios, and the trackers

The arms are report 011's ESN #1 (leak rate 0.03, spectral radius 1.2, input scaling
1.5, seed 8), ESN #2 (input scaling 1, seed 2), and the replay of the take. The
scenarios are report 011's and report 012's, each run for 24 s:

- **Undisturbed,** from the demonstrated start.
- **The block:** the tip held from 4.5 s to 5.5 s, in mid-reach, then let go.
- **Report 011's three pushes** of 0.5 s at the tip: 3 N nearly forward at 3 s, 5 N
  nearly across at 7 s, and 4 N nearly across the other way at 15 s, after the hold.
- **The 49 offset starts,** undisturbed: the demonstrated start offset by −15° to 15°
  in each joint, every 5°, so that joint 2 starts at 15° to 45°.
- **Report 012's ten pushes by hand,** recorded in three logs and replayed by time:
  2.3 to 8.0 N in the mean, up to 14.4 N, for 0.4 to 1.3 s (see
  [report 012, Section 2](../012-pushes-by-hand/README.md#2-setup)).

The trackers are joint PD with the same gains for both joints, given in `[tracker]` as
`kp` and `kd` (`robot_esn.py`, `2b7d0f3`), given a zero reference velocity or the
reference's, with the torque computed every 10 ms and held between, while the arm is
simulated every 2 ms, as in reports 011 and 012:

- **kp = 80 N m/rad and kd = 1.2 N m s/rad,** in every scenario;
- **kp = 16 N m/rad and kd = 1.4 N m s/rad,** in every scenario;
- **kp = 80 N m/rad and kd = 1.7 or 3 N m s/rad,** past the bound, undisturbed only.

They are compared with reports 011's and 012's joint PD at ω = 20 rad/s and ζ = 0.4,
whose gains are scaled by each joint's inertia at the take's final posture: kp = 74.96
and 8.96 N m/rad, and kd = 3.00 and 0.358 N m s/rad.

### The bound

The tracking error of the arm at rest is taken period by period: the torque
u = −Kp e − Kd ė is computed at a period's start and held, while the simulation
advances the arm in five semi-implicit Euler steps of 2 ms, the rate first, then the
angle. Over the period T, the error then becomes e + T ė + 0.6 T² M⁻¹u, where an exact
hold would give T²/2, and its rate ė + T M⁻¹u. The eigenvalues z of this map tell
whether the error shrinks, every |z| below 1, or grows, one above; s = ln z / T gives
each mode's ω and ζ, and a negative z flips the error's sign every period.

With the same gains for both joints, Kp = kp I and Kd = kd I, M⁻¹Kp and M⁻¹Kd share the
eigenvectors of the mass matrix M: the error splits into two modes, each a single
inertia μ, an eigenvalue of M, under the same PD. For one mode, with a = kp T² / μ and
b = kd T / μ, the map's characteristic polynomial is z² − (2 − 0.6 a − b) z + 1 − b +
0.4 a, whose roots are inside the unit circle if and only if 0.4 kp T < kd < 2 μ / T −
kp T / 10. The upper bound, a root below −1, is the flip: the damping torque, held for
a period, reverses the mode's velocity by more than its size. The mass matrix depends
on joint 2's angle alone, and its smaller eigenvalue grows with it from 0° to 180°, so
that the bound is tightest where joint 2 is least.

The modes are also given, as in reports 011 and 012, for each joint alone, with the
joint's inertia M_ii, ω = √(kp / M_ii) and ζ = kd / (2 √(kp M_ii)), and for the two
joints together with the torque continuous, M ë + Kd ė + Kp e = 0.

### Measures

As in reports 011 and 012: after each disturbance, the following gain, from its onset
to 0.5 s past its end; the ringing, the RMS of the hand's motion above 1.5 Hz over the
next 2 s; the farthest the hand strays from the taught path; and when it is back within
5 mm of it, if before the next disturbance or the run's end. Also whether the hand
arrives, holds within the goal (20 mm) for 2 s, and ends there; the peak joint torque
over the run, but under the block from 4.5 s to 6 s; and the peak hand speed.

New here, **the flips:** the part of joint 2's held torque that flips sign every
period, |u(k−1) − 2 u(k) + u(k+1)| / 4 over the periods k; joint 2's torque flips while
it is above 1 N m. Undisturbed, it stays under 0.9 N m with every stable tracker.

### Reproducing the results

The runs read the take as in report 011 and the ESNs by their run names (copied in
[report 011's results](../011-following-the-arm/results)), and the logs of pushes by
hand under the storage root (copied in [report 012's data](../012-pushes-by-hand/data)).
The configurations are in
[`manual_demonstration_v2_robot_stable_equal_gains`](../../experiments/manual_demonstration_v2_robot_stable_equal_gains/README.md),
each the configuration of the same scenario in
[`manual_demonstration_v2_robot_equal_gains`](../../experiments/manual_demonstration_v2_robot_equal_gains/README.md)
with other gains; kp = 80 and kd = 3 are that experiment's.

| Runs | Command | Commit |
| --- | --- | --- |
| `nominal_{kp80_kd1.2,kp16_kd1.4,kp80_kd1.7}_robot_<ESN>_{zero,reference}_velocity_filtered` (12) | `uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_stable_equal_gains/<name>.toml` | `c421488` |
| `{block,pushes,offsets,recorded000,recorded001,recorded002}_{kp80_kd1.2,kp16_kd1.4}_robot_<ESN>_{zero,reference}_velocity_filtered` (48) | the same | `dcfddb3` |
| `nominal_robot_<ESN>_{zero,reference}_velocity_filtered` (4), kp = 80, kd = 3 | `uv run python experiments/robot_esn.py experiments/manual_demonstration_v2_robot_equal_gains/<name>.toml` | `a2c9e8c` |
| [`summary`](results/summary) | `uv run python reports/013-equal-gains/make_figures.py --logs` | |

Every run record (`run.toml`) shows no uncommitted changes. The undisturbed runs'
configurations at `c421488` differ from the current ones only in their comments on the
bound. Each run's directory here holds its configuration, its run record, its metrics
(`metrics.csv`), and, under a recorded force, the SHA-256 of its log
(`recorded_forces.toml`). [`make_figures.py`](make_figures.py) computes the bound and
the modes, draws `stability.png`, and prints the tables of this report from the copies
here and in reports 011 and 012. With `--logs`, it reads the runs' logs under the
storage root, which are not kept in Git, to measure the flips and draw the other
figures.

## 3. Results

In the comparison figures, the columns are the trackers: kp = 80 and kd = 1.2, kp = 16
and kd = 1.4, and report 011's at ζ = 0.4, given the reference velocity in one figure
and a zero reference velocity in the next. Over time, the ESNs' lines are drawn over
the replay's, each ESN's reference dashed, and the disturbances shaded; under a
disturbance, a first row draws each arm's tip force. In the paths, the rows are the
arms: each panel draws the arm at its initial posture (faint gray), the taught path
(thick gray), the goal, and the tip's path; at each disturbance's onset, a red arrow
from the tip shows its mean force over the disturbance, 25 mm per newton. In the
tables, each cell gives ESN #1 / ESN #2 / the replay, but for the following gain, whose
replay's is 0; – marks a hand not back before the next disturbance or the run's end.

### 3.1 The bound

The smaller eigenvalue of the mass matrix is 0.00809 kg m² at the start (joint 2 at
30°) and 0.01603 kg m² at the target (74.59°); 0.00666 kg m² at the lowest offset
start (15°). The largest kd under which no mode grows:

| Joint 2 | μ (kg m²) | kd at kp = 80 (N m s/rad) | kd at kp = 16 (N m s/rad) | Exact hold, 2 μ / T |
| --- | ---: | ---: | ---: | ---: |
| 0° | 0.00617 | 1.155 | 1.219 | 1.235 |
| 15° | 0.00666 | 1.253 | 1.317 | 1.333 |
| 30° | 0.00809 | 1.538 | 1.602 | 1.618 |
| 74.59° | 0.01603 | 3.127 | 3.191 | 3.207 |

Against joint 2's angle, the largest |z| over 10 ms, and where each tracker's error
grows:

![Stability against joint 2's angle](results/summary/stability.png)

| Tracker | Grows below joint 2 at | With an exact hold | Largest \|z\| over the take |
| --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints | 10.15 deg | stable at every angle | 0.967 to 0.982 |
| kp 16, kd 1.4 for both joints | 20.45 deg | 19.50 deg | 0.949 to 0.973 |
| ω 20, ζ 0.4 (reports 011, 012) | stable at every angle | stable at every angle | 0.928 to 0.964 |
| kp 80, kd 1.7 for both joints | 36.05 deg | 33.15 deg | 0.947 to 1.257 |
| kp 80, kd 3 for both joints | 71.45 deg | 69.45 deg | 0.895 to 3.059 |

**Each joint alone**, at the take's final posture (113.94°, 74.59°), and over the take:

| Tracker | Joint | kp (N m/rad) | kd (N m s/rad) | Inertia (kg m²) | ω (rad/s) | ζ | Over the take: ω, ζ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| kp 80, kd 1.2 for both joints | 1 | 80.00 | 1.200 | 0.1874 | 20.7 | 0.15 | 18.4 to 24.4, 0.14 to 0.18 |
| kp 80, kd 1.2 for both joints | 2 | 80.00 | 1.200 | 0.0224 | 59.8 | 0.45 | 59.8 to 59.8, 0.45 to 0.45 |
| kp 16, kd 1.4 for both joints | 1 | 16.00 | 1.400 | 0.1874 | 9.2 | 0.40 | 8.2 to 10.9, 0.36 to 0.48 |
| kp 16, kd 1.4 for both joints | 2 | 16.00 | 1.400 | 0.0224 | 26.7 | 1.17 | 26.7 to 26.7, 1.17 to 1.17 |
| ω 20, ζ 0.4 (reports 011, 012) | 1 | 74.96 | 2.999 | 0.1874 | 20.0 | 0.40 | 17.8 to 23.6, 0.36 to 0.47 |
| ω 20, ζ 0.4 (reports 011, 012) | 2 | 8.96 | 0.358 | 0.0224 | 20.0 | 0.40 | 20.0 to 20.0, 0.40 to 0.40 |

**The two joints together**, at the take's final posture. The decay over 10 ms is
|z|; a continuous mode's is e^(Re s × 10 ms):

| Tracker | Torque | Mode | Decay over 10 ms | ω (rad/s) | ζ |
| --- | --- | --- | ---: | ---: | ---: |
| kp 80, kd 1.2 for both joints | continuous | oscillates | 0.970 | 20.3 | 0.15 |
| kp 80, kd 1.2 for both joints | continuous | oscillates | 0.688 | 70.6 | 0.53 |
| kp 80, kd 1.2 for both joints | held 10 ms | oscillates | 0.977 | 20.6 | 0.11 |
| kp 80, kd 1.2 for both joints | held 10 ms | oscillates | 0.672 | 87.8 | 0.45 |
| kp 16, kd 1.4 for both joints | continuous | oscillates | 0.965 | 9.1 | 0.40 |
| kp 16, kd 1.4 for both joints | continuous | decays | 0.874 | 13.5 | 1.00 |
| kp 16, kd 1.4 for both joints | continuous | decays | 0.478 | 73.8 | 1.00 |
| kp 16, kd 1.4 for both joints | held 10 ms | oscillates | 0.965 | 9.3 | 0.39 |
| kp 16, kd 1.4 for both joints | held 10 ms | decays | 0.877 | 13.2 | 1.00 |
| kp 16, kd 1.4 for both joints | held 10 ms | decays | 0.190 | 166.0 | 1.00 |
| ω 20, ζ 0.4 (reports 011, 012) | continuous | oscillates | 0.948 | 16.3 | 0.33 |
| ω 20, ζ 0.4 (reports 011, 012) | continuous | oscillates | 0.849 | 28.6 | 0.57 |
| ω 20, ζ 0.4 (reports 011, 012) | held 10 ms | oscillates | 0.951 | 16.7 | 0.30 |
| ω 20, ζ 0.4 (reports 011, 012) | held 10 ms | oscillates | 0.840 | 31.2 | 0.56 |

**At the demonstrated start** (30°, 30°):

| Tracker | Torque | Mode | Decay over 10 ms | ω (rad/s) | ζ |
| --- | --- | --- | ---: | ---: | ---: |
| kp 80, kd 1.2 for both joints | continuous | oscillates | 0.976 | 17.9 | 0.13 |
| kp 80, kd 1.2 for both joints | continuous | oscillates | 0.476 | 99.5 | 0.75 |
| kp 80, kd 1.2 for both joints | held 10 ms | oscillates | 0.982 | 18.1 | 0.10 |
| kp 80, kd 1.2 for both joints | held 10 ms | flips sign every period | 0.338 | 332.4 | 0.33 |
| kp 80, kd 1.2 for both joints | held 10 ms | decays | 0.261 | 134.5 | 1.00 |
| kp 16, kd 1.4 for both joints | continuous | oscillates | 0.972 | 8.0 | 0.35 |
| kp 16, kd 1.4 for both joints | continuous | decays | 0.884 | 12.3 | 1.00 |
| kp 16, kd 1.4 for both joints | continuous | decays | 0.200 | 160.8 | 1.00 |
| kp 16, kd 1.4 for both joints | held 10 ms | oscillates | 0.973 | 8.1 | 0.34 |
| kp 16, kd 1.4 for both joints | held 10 ms | decays | 0.886 | 12.1 | 1.00 |
| kp 16, kd 1.4 for both joints | held 10 ms | flips sign every period | 0.736 | 315.7 | 0.10 |
| ω 20, ζ 0.4 (reports 011, 012) | continuous | oscillates | 0.961 | 14.1 | 0.28 |
| ω 20, ζ 0.4 (reports 011, 012) | continuous | oscillates | 0.715 | 41.0 | 0.82 |
| ω 20, ζ 0.4 (reports 011, 012) | held 10 ms | oscillates | 0.964 | 14.4 | 0.26 |
| ω 20, ζ 0.4 (reports 011, 012) | held 10 ms | oscillates | 0.629 | 51.3 | 0.90 |

- **kd = 3 is past the bound from the start to joint 2 at 71.5°:** at the start, the
  flipping mode grows three-fold per period. The bound is 2° higher than an exact hold
  of the torque gives (69.5°): the simulation's semi-implicit steps move the arm by
  0.6 T² M⁻¹u over a period rather than 0.5, which takes kp T / 10, 0.08 N m s/rad at
  kp = 80, off the bound on kd.
- **kp = 80 and kd = 1.2 are inside the bound down to joint 2 at 10.2°,** every
  start included. Joint 2's mode is fast and well damped (at the start, |z| = 0.34
  per period), but joint 1's is lightly damped, ζ = 0.15 alone and 0.10 to 0.11 with
  the torque held: kd sets how fast joint 1's error decays, kd / (2 M), whatever kp.
- **kp = 16 and kd = 1.4 are inside the bound over the reach, but not below joint 2 at
  20.5°,** where two rows of the offset starts begin. Joint 1 alone has ω = 9.2 rad/s
  and ζ = 0.40, joint 2 is overdamped (ζ = 1.17).

### 3.2 Past the bound

Undisturbed, kp = 80 with kd = 1.7, just past the bound, and kd = 3, far past it: the
growth of the flipping torque per period at the start, fitted until it reaches 10 N m,
against the model's at the start; when joint 2's torque flips from the start, and its
peak torque until then; when the reference takes joint 2 past the bound; and the
reference's joint 2 at the last flip:

| Tracker | Velocity | Model, per period | Measured, per period | Flips (s) | Peak torque (N m) | Reference past the bound (s) | Reference's joint 2 at the last flip (deg) |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| kp 80, kd 1.7 for both joints | reference | 1.257 | 1.27 / 1.27 / 1.25 | 0.01 to 0.92 / 0.01 to 0.96 / 0.38 to 1.16 | 157 / 157 / 130 | 0.56 / 0.60 / 0.85 | 41.1 / 40.9 / 42.4 |
| kp 80, kd 1.7 for both joints | zero | 1.257 | 1.31 / 1.30 / 1.25 | 0.05 to 0.96 / 0.06 to 1.00 / 0.44 to 1.18 | 170 / 170 / 106 | 0.56 / 0.59 / 0.85 | 41.5 / 41.4 / 43.1 |
| kp 80, kd 3 for both joints | reference | 3.059 | 3.23 / 3.23 / 3.06 | 0.00 to 1.68 / 0.00 to 1.75 / 0.07 to 3.06 | 1,918 / 1,713 / 924 | 1.34 / 1.42 / 2.68 | 79.5 / 79.8 / 78.3 |
| kp 80, kd 3 for both joints | zero | 3.059 | 3.50 / 3.49 / 3.08 | 0.02 to 1.74 / 0.02 to 1.83 / 0.09 to 3.09 | 749 / 1,062 / 1,927 | 1.34 / 1.43 / 2.68 | 80.2 / 81.0 / 78.9 |

![Past the bound](results/summary/bound.png)

- **The flips grow as the model predicts.** The replay's arm stays at the start for
  about 0.4 s, as the take does, and its flips grow by 1.25 per period with kd = 1.7 and
  3.06 to 3.08 with kd = 3, against 1.257 and 3.059. The ESNs' references move from the
  first period, and their flips grow a little faster, 1.27 to 1.31 and 3.23 to 3.50.
- **They die out once the reference is past the bound,** 0.31 to 0.41 s after it, at
  joint 2's 41° to 43° with kd = 1.7 and 78° to 81° with kd = 3: the flips have to
  decay from 106 to 1,900 N m to 1 N m.
- **Just past the bound, the chatter is mild:** 106 to 170 N m, for 0.7 to 1.0 s,
  against 750 to 1,900 N m for 1.7 to 3 s with kd = 3. The hand still moves at up to
  0.6 m/s, twice as fast as with the stable gains, and the ESNs' hands keep 11 to
  14 mm (RMS) from the taught path, against 0.7 to 1.1 mm.

### 3.3 Undisturbed and from the offset starts

Arrival (s), final distance from the target (mm), peak torque (N m), peak hand speed
(m/s), and the RMS distance from the taught path over the run (mm):

| Tracker | Arrival (s) | Final distance (mm) | Peak torque (N m) | Peak hand speed (m/s) | Path (mm) |
| --- | --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 9.54 / 9.53 / 9.58 | 2.9 / 2.7 / 2.8 | 0.7 / 0.6 / 0.4 | 0.23 / 0.23 / 0.28 | 0.7 / 0.8 / 0.5 |
| kp 16, kd 1.4 for both joints, reference velocity | 9.49 / 9.48 / 9.49 | 2.9 / 2.7 / 2.8 | 1.0 / 0.8 / 0.4 | 0.23 / 0.23 / 0.26 | 1.2 / 1.2 / 1.1 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 9.55 / 9.54 / 9.56 | 2.9 / 2.7 / 2.8 | 0.2 / 0.1 / 0.6 | 0.23 / 0.23 / 0.27 | 0.7 / 0.9 / 0.4 |
| kp 80, kd 1.7 for both joints, reference velocity | 9.33 / 9.37 / 9.58 | 2.9 / 2.7 / 2.8 | 157.1 / 156.5 / 129.9 | 0.53 / 0.53 / 0.39 | 10.7 / 12.6 / 6.2 |
| kp 80, kd 3 for both joints, reference velocity | 8.31 / 8.07 / 9.57 | 2.9 / 2.7 / 2.8 | 1,917.8 / 1,713.5 / 923.6 | 36.12 / 31.56 / 23.35 | 245.1 / 163.1 / 75.4 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 9.77 / 9.68 / 9.59 | 2.9 / 2.7 / 2.8 | 0.4 / 0.3 / 0.4 | 0.22 / 0.22 / 0.27 | 1.1 / 1.1 / 0.5 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 10.95 / 10.50 / 9.61 | 2.9 / 2.7 / 2.8 | 0.1 / 0.1 / 0.2 | 0.19 / 0.21 / 0.24 | 5.0 / 5.1 / 1.0 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 10.18 / 9.96 / 9.61 | 2.9 / 2.7 / 2.8 | 0.1 / 0.1 / 0.3 | 0.21 / 0.22 / 0.26 | 2.5 / 2.4 / 0.4 |
| kp 80, kd 1.7 for both joints, zero reference velocity | 9.65 / 9.57 / 9.60 | 2.9 / 2.7 / 2.8 | 170.2 / 169.5 / 105.9 | 0.60 / 0.60 / 0.29 | 12.4 / 13.9 / 4.1 |
| kp 80, kd 3 for both joints, zero reference velocity | 8.79 / 8.28 / 9.61 | 2.9 / 2.7 / 2.8 | 748.9 / 1,061.5 / 1,927.4 | 11.08 / 23.98 / 34.69 | 74.5 / 81.4 / 191.1 |

Undisturbed, joint 2's torque never flips with the stable gains: its flipping part
stays under 0.9 N m, the largest at the task's first periods.

![Joint angles undisturbed, given the reference velocity](results/summary/nominal_reference_joints.png)

![Joint torques undisturbed, given the reference velocity](results/summary/nominal_reference_torques.png)

![The tip's path undisturbed, given the reference velocity](results/summary/nominal_reference_paths.png)

![Joint angles undisturbed, given a zero reference velocity](results/summary/nominal_zero_joints.png)

![Joint torques undisturbed, given a zero reference velocity](results/summary/nominal_zero_torques.png)

![The tip's path undisturbed, given a zero reference velocity](results/summary/nominal_zero_paths.png)

From the 49 offset starts, the successes, and the peak torque, median (largest):

| Tracker | ESN #1 | ESN #2 | replay |
| --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 49/49; 10.8 (20.6) N m | 49/49; 13.4 (35.8) N m | 49/49; 31.4 (63.7) N m |
| kp 16, kd 1.4 for both joints, reference velocity | 49/49; 12.9 (54.6) N m | 49/49; 15.9 (44.1) N m | 49/49; 36.7 (81.1) N m |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 49/49; 25.4 (49.1) N m | 49/49; 27.2 (61.1) N m | 49/49; 52.3 (78.5) N m |
| kp 80, kd 1.2 for both joints, zero reference velocity | 49/49; 7.4 (14.5) N m | 49/49; 9.0 (21.8) N m | 49/49; 20.9 (37.5) N m |
| kp 16, kd 1.4 for both joints, zero reference velocity | 49/49; 1.5 (57.4) N m | 49/49; 1.8 (5.9) N m | 49/49; 4.1 (7.2) N m |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 49/49; 6.3 (12.3) N m | 49/49; 6.8 (15.3) N m | 49/49; 13.1 (19.6) N m |

How long joint 2's torque flips from the 49 starts: from how many, the longest and its
start (joint 1, joint 2), and joint 2's start angles from which it flips for over 0.2 s:

| Tracker | Velocity | ESN #1 | ESN #2 | replay |
| --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints | reference | 35 starts, up to 0.12 s (from 30°, 15°) | 44 starts, up to 0.11 s (from 15°, 15°) | 48 starts, up to 0.08 s (from 15°, 15°) |
| kp 80, kd 1.2 for both joints | zero | 25 starts, up to 0.11 s (from 40°, 15°) | 42 starts, up to 0.10 s (from 15°, 15°) | 48 starts, up to 0.07 s (from 15°, 15°) |
| kp 16, kd 1.4 for both joints | reference | 40 starts, up to 1.60 s (from 45°, 15°); over 0.2 s from joint 2 at 15°, 20°, 25° | 46 starts, up to 0.62 s (from 25°, 15°); over 0.2 s from joint 2 at 15°, 20° | 48 starts, up to 0.19 s (from 15°, 15°) |
| kp 16, kd 1.4 for both joints | zero | 11 starts, up to 1.83 s (from 45°, 15°); over 0.2 s from joint 2 at 15°, 20° | 11 starts, up to 1.08 s (from 30°, 15°); over 0.2 s from joint 2 at 15° | 35 starts, up to 0.20 s (from 15°, 15°) |
| ω 20, ζ 0.4 (reports 011, 012) | reference | 17 starts, up to 0.02 s (from 15°, 15°) | 29 starts, up to 0.03 s (from 45°, 15°) | 46 starts, up to 0.04 s (from 45°, 15°) |
| ω 20, ζ 0.4 (reports 011, 012) | zero | none | none | 8 starts, up to 0.01 s (from 15°, 40°) |

![From the 49 starts: the flips and the peak torque](results/summary/offsets.png)

The offset start −15° in both joints, where joint 2 starts at 15°, the lowest:

![Joint angles from an offset start, given the reference velocity](results/summary/offset_reference_joints.png)

![Joint torques from an offset start, given the reference velocity](results/summary/offset_reference_torques.png)

![The tip's path from an offset start, given the reference velocity](results/summary/offset_reference_paths.png)

![Joint angles from an offset start, given a zero reference velocity](results/summary/offset_zero_joints.png)

![Joint torques from an offset start, given a zero reference velocity](results/summary/offset_zero_torques.png)

![The tip's path from an offset start, given a zero reference velocity](results/summary/offset_zero_paths.png)

- **Every arm arrives and holds, from every start.**
- **Undisturbed, kp = 80 and kd = 1.2 track as report 011's gains do:** the hand keeps
  0.5 to 1.1 mm (RMS) from the taught path, against 0.4 to 2.5 mm, and arrives at 9.5
  to 9.8 s, against 9.5 to 10.2 s. kp = 16 and kd = 1.4, given a zero reference
  velocity, lag: the ESNs keep 5.0 to 5.1 mm off the path and arrive at 10.5 to
  11.0 s.
- **From an offset start, every tracker's torque first flips for a few periods:** at
  the task's first period, the reference jumps from the start to the take's start (the
  replay) or to the ESN's first output, and the step is taken up. With kp = 80 and
  kd = 1.2, the flips last at most 0.12 s, joint 2's flipping mode at 15° shrinking by
  0.88 per period; with report 011's gains, at most 0.04 s.
- **kp = 16 and kd = 1.4 flip for up to 1.8 s from joint 2 at 15° and 20°,** below or
  at its bound of 20.5°, where the flipping mode barely shrinks, and once from 25°. The
  flips last until 0.2 to 0.8 s after the reference takes joint 2 past the bound,
  which, from joint 1 at 30° to 45°, takes the ESNs' references up to 1.3 s; from
  joint 2 at 20° and 25°, ESN #1's first dips to 17° to 19°. The flips peak at
  57 N m. The replay's reference jumps to the take's start, joint 2 at 30°, and its
  flips last at most 0.2 s.

### 3.4 Under the block and report 011's pushes

Under the block, as in Section 3.3, the peak torque under the block, from 4.5 s to 6 s:

| Tracker | Arrival (s) | Final distance (mm) | Peak torque (N m) | Peak hand speed (m/s) | Path (mm) |
| --- | --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 10.03 / 9.93 / 9.58 | 2.9 / 2.7 / 2.8 | 8.0 / 7.2 / 16.7 | 1.51 / 1.38 / 3.40 | 11.7 / 7.9 / 7.1 |
| kp 16, kd 1.4 for both joints, reference velocity | 10.03 / 9.89 / 9.49 | 2.9 / 2.7 / 2.8 | 1.7 / 1.5 / 3.6 | 0.39 / 0.37 / 1.02 | 13.1 / 9.3 / 2.0 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 10.08 / 9.98 / 9.56 | 2.9 / 2.7 / 2.8 | 2.2 / 0.9 / 13.1 | 0.52 / 0.56 / 2.04 | 12.7 / 8.5 / 8.4 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 10.28 / 10.04 / 9.59 | 2.9 / 2.7 / 2.8 | 7.5 / 7.3 / 16.7 | 1.38 / 1.36 / 3.39 | 11.4 / 8.6 / 7.1 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 11.56 / 10.82 / 9.61 | 2.9 / 2.7 / 2.8 | 1.2 / 1.5 / 3.6 | 0.25 / 0.34 / 0.96 | 10.2 / 10.4 / 1.9 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 10.74 / 10.31 / 9.61 | 2.9 / 2.7 / 2.8 | 2.1 / 0.8 / 12.4 | 0.41 / 0.47 / 1.98 | 11.3 / 9.6 / 7.8 |

Under report 011's three pushes:

| Tracker | Arrival (s) | Final distance (mm) | Peak torque (N m) | Peak hand speed (m/s) | Path (mm) |
| --- | --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 9.26 / 9.37 / 9.58 | 2.9 / 2.7 / 2.8 | 2.8 / 2.8 / 3.1 | 0.48 / 0.48 / 0.46 | 9.2 / 9.8 / 4.4 |
| kp 16, kd 1.4 for both joints, reference velocity | 9.81 / 9.86 / 9.49 | 2.8 / 2.7 / 2.8 | 2.1 / 2.1 / 2.2 | 0.68 / 0.67 / 0.57 | 58.1 / 46.8 / 20.9 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 9.70 / 9.75 / 9.56 | 2.8 / 2.7 / 2.8 | 2.8 / 2.5 / 3.2 | 1.28 / 1.29 / 1.21 | 62.5 / 57.2 / 26.4 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 9.50 / 9.53 / 9.59 | 2.9 / 2.7 / 2.8 | 2.8 / 2.8 / 3.1 | 0.47 / 0.47 / 0.47 | 8.1 / 8.8 / 4.4 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 9.76 / 10.57 / 9.61 | 2.7 / 2.6 / 2.8 | 2.2 / 2.1 / 2.2 | 0.60 / 0.60 / 0.57 | 36.9 / 37.3 / 20.7 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 9.19 / 10.07 / 9.61 | 2.7 / 2.6 / 2.8 | 2.6 / 2.6 / 3.2 | 1.21 / 1.21 / 1.20 | 54.5 / 55.1 / 26.2 |

**The following gain**, ESN #1 / ESN #2:

| Tracker | Block | Push 1 | Push 2 | Push 3 |
| --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 0.88 / 0.81 | 0.55 / 0.52 | 0.85 / 0.64 | 0.69 / 0.55 |
| kp 16, kd 1.4 for both joints, reference velocity | 0.88 / 0.81 | 0.62 / 0.57 | 0.71 / 0.63 | 0.74 / 0.61 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 0.89 / 0.82 | 0.80 / 0.76 | 0.63 / 0.63 | 0.83 / 0.73 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 0.90 / 0.83 | 0.52 / 0.49 | 0.72 / 0.63 | 0.69 / 0.56 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 0.94 / 0.87 | 0.53 / 0.51 | 0.65 / 0.63 | 0.74 / 0.61 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 0.92 / 0.86 | 0.77 / 0.73 | 0.64 / 0.62 | 0.84 / 0.74 |

**The ringing over the next 2 s** (mm):

| Tracker | Block | Push 1 | Push 2 | Push 3 |
| --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 3.5 / 3.6 / 22.0 | 2.2 / 2.2 / 3.4 | 2.0 / 1.8 / 2.8 | 3.0 / 3.3 / 5.0 |
| kp 16, kd 1.4 for both joints, reference velocity | 2.2 / 2.1 / 9.5 | 4.2 / 4.3 / 4.9 | 4.3 / 4.3 / 4.3 | 3.4 / 3.8 / 5.4 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 3.3 / 3.6 / 17.5 | 4.9 / 5.2 / 6.8 | 8.7 / 8.1 / 10.0 | 5.5 / 6.0 / 8.5 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 3.2 / 3.1 / 21.7 | 2.1 / 2.0 / 3.4 | 1.7 / 1.5 / 2.8 | 2.9 / 3.1 / 5.0 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 1.2 / 1.5 / 8.9 | 3.7 / 3.8 / 4.8 | 2.9 / 2.6 / 4.2 | 3.2 / 3.6 / 5.4 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 2.2 / 2.8 / 17.0 | 4.7 / 4.8 / 6.9 | 7.5 / 6.9 / 10.1 | 5.4 / 5.8 / 8.5 |

**Back within 5 mm of the taught path** (s after the end):

| Tracker | Block | Push 1 | Push 2 | Push 3 |
| --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 2.16 / 2.93 / 0.31 | 0.94 / 1.91 / 0.17 | 0.23 / 2.17 / 0.03 | 0.94 / 0.44 / 0.49 |
| kp 16, kd 1.4 for both joints, reference velocity | 4.84 / 4.06 / 0.34 | 2.96 / – / 0.46 | 2.73 / 3.33 / 0.19 | 4.87 / 3.98 / 0.53 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 2.22 / 3.08 / 0.42 | 2.98 / – / 0.04 | 2.52 / 3.33 / 0.38 | 4.98 / 4.16 / 0.46 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 2.33 / 3.11 / 0.31 | 1.12 / 1.23 / 0.17 | 0.24 / 2.33 / 0.03 | 0.98 / 0.45 / 0.49 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 5.53 / 5.09 / 0.31 | – / – / 0.47 | 2.84 / 4.34 / 0.19 | 5.41 / 4.06 / 0.53 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 2.61 / 4.29 / 0.42 | 3.44 / – / 0.03 | 2.88 / 3.83 / 0.38 | 5.27 / 4.43 / 0.45 |

**The farthest the hand strays from the taught path** (mm):

| Tracker | Block | Push 1 | Push 2 | Push 3 |
| --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 37 / 22 / 30 | 18 / 17 / 12 | 27 / 34 / 12 | 46 / 40 / 27 |
| kp 16, kd 1.4 for both joints, reference velocity | 40 / 24 / 5 | 77 / 74 / 34 | 150 / 144 / 61 | 179 / 162 / 95 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 38 / 23 / 28 | 66 / 67 / 7 | 175 / 206 / 111 | 202 / 166 / 99 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 37 / 23 / 30 | 15 / 14 / 12 | 25 / 29 / 12 | 44 / 39 / 27 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 37 / 29 / 6 | 53 / 55 / 37 | 83 / 95 / 60 | 147 / 137 / 96 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 37 / 25 / 26 | 43 / 56 / 6 | 166 / 197 / 110 | 180 / 154 / 99 |

![Joint angles under the block, given the reference velocity](results/summary/block_reference_joints.png)

![Joint torques under the block, given the reference velocity](results/summary/block_reference_torques.png)

![The tip's path under the block, given the reference velocity](results/summary/block_reference_paths.png)

![Joint angles under the block, given a zero reference velocity](results/summary/block_zero_joints.png)

![Joint torques under the block, given a zero reference velocity](results/summary/block_zero_torques.png)

![The tip's path under the block, given a zero reference velocity](results/summary/block_zero_paths.png)

![Joint angles under the pushes, given the reference velocity](results/summary/pushes_reference_joints.png)

![Joint torques under the pushes, given the reference velocity](results/summary/pushes_reference_torques.png)

![The tip's path under the pushes, given the reference velocity](results/summary/pushes_reference_paths.png)

![Joint angles under the pushes, given a zero reference velocity](results/summary/pushes_zero_joints.png)

![Joint torques under the pushes, given a zero reference velocity](results/summary/pushes_zero_torques.png)

![The tip's path under the pushes, given a zero reference velocity](results/summary/pushes_zero_paths.png)

- **kp = 80 and kd = 1.2 move the ESNs' arms a fraction as far under the pushes:** 14
  to 34 mm from the path at the first two, against 43 to 206 mm, and 39 to 46 mm at the
  target, against 154 to 202 mm. The replay's goes 12 mm at the first two, against 6
  to 111 mm, and 27 mm at the target, against 99 mm. They ring 1.5 to 5.0 mm, against
  4.7 to 10.1 mm, and the ESNs' hands are back within 0.2 to 2.3 s, against 2.5 to
  5.3 s, after every push. The ESNs follow the first push less, 0.49 to 0.55 against
  0.73 to 0.80.
- **Under the block, they wait as much,** following by 0.81 to 0.90 against 0.82 to
  0.92, but press against it with 7 to 8 N m, against 1 to 2 N m; the replay presses
  with 17 N m, against 12 to 13 N m, and rings 22 mm after the release, against 17 mm.
  The release flips joint 2's torque for 0.01 to 0.03 s, by up to 2.4 N m.
- **kp = 16 and kd = 1.4 give way about as far as report 011's gains,** the ESNs 53 to
  179 mm from the path, ring a little less, 2.6 to 5.4 mm, and are back as slowly, in
  2.7 to 5.4 s; ESN #2 is not back after the first push, nor ESN #1 given a zero
  reference velocity. They press on the block with 1 to 2 N m, and the replay with
  4 N m, which after the release catches up gently, 5 to 6 mm from the path and ringing
  9 to 10 mm, against 26 to 30 mm and 17 to 22 mm.

### 3.5 Under the pushes by hand

Logs 000 and 002 hold four and three short pushes, log 001 three long ones, of which
the second is 8 N in the mean for 1.26 s at mid-reach (report 012, Section 2).

#### Log 000

As in Section 3.3:

| Tracker | Arrival (s) | Final distance (mm) | Peak torque (N m) | Peak hand speed (m/s) | Path (mm) |
| --- | --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 9.79 / 9.54* / 9.47* | 2.9 / 2.7 / 2.8 | 4.1 / 4.1 / 3.9 | 0.69 / 0.72 / 0.69 | 15.3 / 13.1 / 6.6 |
| kp 16, kd 1.4 for both joints, reference velocity | 11.15 / 10.53 / 9.60* | 2.9 / 2.6 / 2.8 | 4.5 / 4.8 / 5.2 | 1.18 / 1.05 / 1.29 | 104.9 / 96.0 / 50.1 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 10.15 / 9.87 / 9.56 | 2.9 / 2.7 / 2.8 | 4.6 / 4.4 / 4.5 | 1.26 / 1.26 / 1.38 | 59.0 / 60.2 / 25.0 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 9.97 / 9.80 / 9.47* | 2.9 / 2.7 / 2.8 | 4.0 / 4.1 / 3.9 | 0.67 / 0.70 / 0.68 | 14.6 / 12.5 / 6.6 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 12.01 / 11.73 / 9.60* | 3.0 / 2.7 / 2.8 | 4.2 / 4.5 / 5.1 | 0.82 / 0.91 / 1.30 | 78.1 / 80.2 / 49.9 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 10.32 / 10.04 / 9.59 | 2.9 / 2.7 / 2.8 | 4.4 / 4.3 / 4.2 | 1.17 / 1.18 / 1.38 | 53.2 / 53.4 / 25.0 |

![Joint angles under log 000, given the reference velocity](results/summary/recorded000_reference_joints.png)

![Joint torques under log 000, given the reference velocity](results/summary/recorded000_reference_torques.png)

![The tip's path under log 000, given the reference velocity](results/summary/recorded000_reference_paths.png)

![Joint angles under log 000, given a zero reference velocity](results/summary/recorded000_zero_joints.png)

![Joint torques under log 000, given a zero reference velocity](results/summary/recorded000_zero_torques.png)

![The tip's path under log 000, given a zero reference velocity](results/summary/recorded000_zero_paths.png)

| Following, ESN #1 / ESN #2 | Push 1 | Push 2 | Push 3 | Push 4 |
| --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 0.53 / 0.55 | 0.82 / 0.76 | 0.55 / 0.51 | 0.66 / 0.53 |
| kp 16, kd 1.4 for both joints, reference velocity | 0.51 / 0.52 | 0.81 / 0.74 | 0.70 / 0.57 | 0.56 / 0.48 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 0.58 / 0.66 | 0.80 / 0.71 | 0.83 / 0.72 | 0.62 / 0.61 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 0.61 / 0.61 | 0.82 / 0.75 | 0.56 / 0.52 | 0.67 / 0.53 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 0.59 / 0.57 | 0.92 / 0.81 | 0.75 / 0.55 | 0.58 / 0.48 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 0.63 / 0.69 | 0.83 / 0.73 | 0.54 / 0.59 | 0.63 / 0.62 |

| Ringing (mm) | Push 1 | Push 2 | Push 3 | Push 4 |
| --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 1.3 / 1.3 / 1.7 | 0.9 / 0.9 / 1.7 | 5.6 / 5.7 / 7.0 | 5.3 / 5.5 / 6.9 |
| kp 16, kd 1.4 for both joints, reference velocity | 4.2 / 4.3 / 5.3 | 3.3 / 3.2 / 3.1 | 11.9 / 13.8 / 16.1 | 11.3 / 11.8 / 13.7 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 4.5 / 4.5 / 6.0 | 8.2 / 7.8 / 8.5 | 8.7 / 7.5 / 6.3 | 11.2 / 11.2 / 13.8 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 1.2 / 1.2 / 1.8 | 0.9 / 0.8 / 1.7 | 4.8 / 5.0 / 7.0 | 5.1 / 5.2 / 6.9 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 3.9 / 3.7 / 5.6 | 2.4 / 2.3 / 3.0 | 8.8 / 9.6 / 16.0 | 10.6 / 10.6 / 13.7 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 3.9 / 3.7 / 6.1 | 6.9 / 6.6 / 8.5 | 6.9 / 5.9 / 6.4 | 10.4 / 10.4 / 13.8 |

| Back within 5 mm (s) | Push 1 | Push 2 | Push 3 | Push 4 |
| --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 0.11 / 1.13 / 0.00 | 0.72 / 0.32 / 0.02 | 0.71 / 0.68 / 0.78 | 1.18 / 0.75 / 0.63 |
| kp 16, kd 1.4 for both joints, reference velocity | 2.20 / 3.15 / 0.50 | – / – / 0.15 | – / – / 0.88 | 1.69 / 3.65 / 0.88 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 2.22 / – / 0.07 | – / – / 0.35 | – / 0.93 / 0.41 | 2.63 / 3.33 / 0.47 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 0.13 / 0.12 / 0.00 | 0.77 / 0.30 / 0.02 | 0.68 / 0.67 / 0.78 | 1.26 / 0.81 / 0.63 |
| kp 16, kd 1.4 for both joints, zero reference velocity | – / 2.76 / 0.51 | 1.26 / 1.92 / 0.15 | 2.38 / – / 0.88 | 2.75 / 4.11 / 0.88 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 2.68 / – / 0.07 | – / – / 0.36 | – / 1.12 / 0.42 | 2.43 / 3.45 / 0.47 |

| Farthest (mm) | Push 1 | Push 2 | Push 3 | Push 4 |
| --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 17 / 17 / 14 | 12 / 17 / 7 | 42 / 41 / 23 | 85 / 73 / 40 |
| kp 16, kd 1.4 for both joints, reference velocity | 72 / 73 / 50 | 59 / 81 / 36 | 201 / 188 / 128 | 361 / 340 / 209 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 32 / 32 / 21 | 121 / 163 / 70 | 134 / 101 / 55 | 219 / 221 / 125 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 17 / 17 / 14 | 11 / 15 / 7 | 38 / 38 / 23 | 82 / 71 / 40 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 67 / 70 / 50 | 43 / 48 / 35 | 74 / 103 / 124 | 309 / 303 / 209 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 31 / 32 / 22 | 99 / 126 / 70 | 102 / 81 / 55 | 208 / 207 / 125 |

#### Log 001

As in Section 3.3:

| Tracker | Arrival (s) | Final distance (mm) | Peak torque (N m) | Peak hand speed (m/s) | Path (mm) |
| --- | --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 10.21 / 10.10 / 9.48* | 2.9 / 2.7 / 2.8 | 7.7 / 7.7 / 6.8 | 1.14 / 1.27 / 1.11 | 19.4 / 24.1 / 9.2 |
| kp 16, kd 1.4 for both joints, reference velocity | 14.22* / 18.71* / 9.46 | 2.8 / 2.7 / 2.8 | 7.3 / 7.5 / 9.3 | 2.55 / 1.74 / 2.35 | 102.0 / 73.1 / 60.9 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 10.35 / 10.11 / 9.56 | 2.9 / 2.7 / 2.8 | 7.2 / 6.1 / 6.6 | 1.07 / 1.01 / 1.18 | 71.2 / 56.4 / 25.7 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 10.38 / 10.47 / 9.50* | 2.9 / 2.7 / 2.8 | 7.6 / 7.8 / 6.7 | 1.08 / 1.25 / 1.11 | 16.6 / 23.4 / 9.1 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 13.98* / – / 9.56 | 2.7 / 840.5 / 2.8 | 6.7 / 7.0 / 9.2 | 1.25 / 1.49 / 2.35 | 71.5 / 110.7 / 59.6 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 13.39* / 14.13* / 9.61 | 2.8 / 2.8 / 2.8 | 10.1 / 9.9 / 6.4 | 2.09 / 2.13 / 1.18 | 97.5 / 110.0 / 25.7 |

![Joint angles under log 001, given the reference velocity](results/summary/recorded001_reference_joints.png)

![Joint torques under log 001, given the reference velocity](results/summary/recorded001_reference_torques.png)

![The tip's path under log 001, given the reference velocity](results/summary/recorded001_reference_paths.png)

![Joint angles under log 001, given a zero reference velocity](results/summary/recorded001_zero_joints.png)

![Joint torques under log 001, given a zero reference velocity](results/summary/recorded001_zero_torques.png)

![The tip's path under log 001, given a zero reference velocity](results/summary/recorded001_zero_paths.png)

| Following, ESN #1 / ESN #2 | Push 1 | Push 2 | Push 3 |
| --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 0.37 / 0.46 | 0.74 / 0.59 | 0.64 / 0.48 |
| kp 16, kd 1.4 for both joints, reference velocity | 0.37 / 0.43 | 0.79 / 0.73 | 0.70 / 0.50 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 0.66 / 0.78 | 0.50 / 0.59 | 0.80 / 0.67 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 0.33 / 0.48 | 0.80 / 0.63 | 0.64 / 0.49 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 0.31 / 0.38 | 0.80 / 0.75 | 0.68 / 0.64 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 0.67 / 0.76 | 0.85 / 0.73 | 0.79 / 0.60 |

| Ringing (mm) | Push 1 | Push 2 | Push 3 |
| --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 2.6 / 2.7 / 3.5 | 8.8 / 9.7 / 10.8 | 4.8 / 5.2 / 5.7 |
| kp 16, kd 1.4 for both joints, reference velocity | 3.7 / 4.6 / 4.5 | 24.7 / 22.3 / 25.6 | 12.3 / 8.8 / 19.1 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 2.0 / 2.0 / 2.6 | 8.5 / 4.8 / 4.5 | 11.4 / 11.5 / 12.7 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 2.6 / 2.6 / 3.5 | 7.7 / 8.8 / 10.7 | 4.6 / 4.9 / 5.7 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 3.9 / 4.1 / 4.7 | 17.8 / 17.9 / 25.9 | 12.2 / 5.9 / 19.1 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 1.6 / 1.6 / 2.5 | 27.0 / 25.2 / 4.5 | 9.5 / 9.5 / 12.7 |

| Back within 5 mm (s) | Push 1 | Push 2 | Push 3 |
| --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 1.54 / 2.46 / 0.19 | 2.15 / 1.97 / 0.76 | 1.10 / 0.69 / 0.50 |
| kp 16, kd 1.4 for both joints, reference velocity | 3.27 / – / 0.48 | 4.52 / – / 1.18 | 6.29 / 6.59 / 0.89 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 3.12 / 3.40 / 0.06 | 2.36 / 1.86 / 0.06 | 5.29 / 4.34 / 0.46 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 2.32 / 2.64 / 0.19 | 1.68 / 2.22 / 0.75 | 1.18 / 0.70 / 0.50 |
| kp 16, kd 1.4 for both joints, zero reference velocity | – / – / 0.49 | 5.31 / – / 1.20 | 5.98 / – / 0.89 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 2.87 / 3.78 / 0.06 | 5.30 / – / 0.05 | 5.34 / 3.22 / 0.46 |

| Farthest (mm) | Push 1 | Push 2 | Push 3 |
| --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 23 / 25 / 15 | 66 / 77 / 26 | 62 / 59 / 40 |
| kp 16, kd 1.4 for both joints, reference velocity | 97 / 107 / 63 | 323 / 201 / 153 | 253 / 151 / 182 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 28 / 37 / 18 | 91 / 100 / 50 | 254 / 222 / 128 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 23 / 25 / 15 | 56 / 71 / 26 | 58 / 56 / 40 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 94 / 102 / 66 | 198 / 136 / 145 | 222 / 185 / 182 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 27 / 34 / 17 | 274 / 286 / 52 | 211 / 159 / 128 |

#### Log 002

As in Section 3.3:

| Tracker | Arrival (s) | Final distance (mm) | Peak torque (N m) | Peak hand speed (m/s) | Path (mm) |
| --- | --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 9.63 / 9.57 / 9.58 | 2.9 / 2.7 / 2.8 | 4.2 / 4.2 / 4.1 | 0.59 / 0.59 / 0.59 | 5.9 / 6.3 / 3.2 |
| kp 16, kd 1.4 for both joints, reference velocity | 9.97 / 9.82 / 9.49 | 3.0 / 2.7 / 2.8 | 4.7 / 4.7 / 4.9 | 0.73 / 0.74 / 0.96 | 32.2 / 33.8 / 16.4 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 9.37 / 9.46 / 9.56 | 2.9 / 2.8 / 2.8 | 3.2 / 3.1 / 3.5 | 0.97 / 0.86 / 0.95 | 50.6 / 45.6 / 20.2 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 9.87 / 9.73 / 9.59 | 2.9 / 2.7 / 2.8 | 4.3 / 4.3 / 4.2 | 0.59 / 0.59 / 0.58 | 5.8 / 5.9 / 3.2 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 11.53 / 10.82 / 9.61 | 2.9 / 2.7 / 2.8 | 5.6 / 5.4 / 5.1 | 0.92 / 0.87 / 1.00 | 37.1 / 30.6 / 16.5 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 10.18 / 9.90 / 9.61 | 2.9 / 2.7 / 2.8 | 3.9 / 3.7 / 3.5 | 0.80 / 0.78 / 0.95 | 41.1 / 34.0 / 20.0 |

![Joint angles under log 002, given the reference velocity](results/summary/recorded002_reference_joints.png)

![Joint torques under log 002, given the reference velocity](results/summary/recorded002_reference_torques.png)

![The tip's path under log 002, given the reference velocity](results/summary/recorded002_reference_paths.png)

![Joint angles under log 002, given a zero reference velocity](results/summary/recorded002_zero_joints.png)

![Joint torques under log 002, given a zero reference velocity](results/summary/recorded002_zero_torques.png)

![The tip's path under log 002, given a zero reference velocity](results/summary/recorded002_zero_paths.png)

| Following, ESN #1 / ESN #2 | Push 1 | Push 2 | Push 3 |
| --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 0.52 / 0.54 | 0.25 / 0.43 | 0.75 / 0.64 |
| kp 16, kd 1.4 for both joints, reference velocity | 0.52 / 0.52 | 0.34 / 0.53 | 0.81 / 0.70 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 0.60 / 0.66 | 0.80 / 0.69 | 0.87 / 0.77 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 0.56 / 0.57 | 0.21 / 0.39 | 0.75 / 0.64 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 0.57 / 0.55 | 0.41 / 0.40 | 0.81 / 0.72 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 0.62 / 0.67 | 0.89 / 0.85 | 0.88 / 0.79 |

| Ringing (mm) | Push 1 | Push 2 | Push 3 |
| --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 2.7 / 2.6 / 3.4 | 3.5 / 3.4 / 4.1 | 1.1 / 1.2 / 1.9 |
| kp 16, kd 1.4 for both joints, reference velocity | 5.5 / 5.6 / 6.9 | 9.9 / 9.4 / 12.1 | 3.0 / 3.3 / 2.7 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 6.9 / 6.9 / 8.5 | 6.9 / 7.2 / 4.8 | 2.8 / 3.9 / 7.6 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 2.5 / 2.5 / 3.4 | 3.5 / 3.3 / 4.1 | 1.1 / 1.1 / 1.9 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 5.1 / 5.1 / 6.9 | 12.6 / 11.7 / 12.4 | 2.3 / 2.5 / 2.7 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 6.3 / 6.1 / 8.5 | 2.4 / 2.3 / 4.5 | 2.7 / 3.5 / 7.6 |

| Back within 5 mm (s) | Push 1 | Push 2 | Push 3 |
| --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 0.90 / 1.73 / 0.17 | 0.04 / 1.23 / 0.00 | 0.88 / 0.47 / 0.01 |
| kp 16, kd 1.4 for both joints, reference velocity | – / – / 0.52 | 2.25 / 3.29 / 0.13 | 4.51 / 3.81 / 0.11 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | – / – / 0.09 | 2.25 / 2.83 / 0.20 | 4.89 / 4.19 / 0.44 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 0.38 / 1.87 / 0.17 | 0.06 / 0.23 / 0.00 | 0.93 / 0.54 / 0.01 |
| kp 16, kd 1.4 for both joints, zero reference velocity | – / 2.75 / 0.52 | 4.75 / 4.24 / 0.13 | 4.73 / 3.42 / 0.11 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 2.77 / – / 0.09 | 2.60 / 1.86 / 0.19 | 5.26 / 4.48 / 0.44 |

| Farthest (mm) | Push 1 | Push 2 | Push 3 |
| --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 21 / 21 / 13 | 11 / 16 / 8 | 35 / 30 / 23 |
| kp 16, kd 1.4 for both joints, reference velocity | 104 / 107 / 57 | 57 / 80 / 33 | 121 / 109 / 77 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 50 / 49 / 14 | 112 / 157 / 78 | 179 / 150 / 102 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 21 / 21 / 13 | 11 / 15 / 8 | 34 / 29 / 23 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 98 / 99 / 59 | 116 / 93 / 33 | 97 / 92 / 77 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 47 / 45 / 15 | 60 / 70 / 76 | 166 / 142 / 102 |

- **Log 001's strong push moves kp = 80 and kd = 1.2's ESNs 56 to 77 mm from the
  path,** against 91 to 286 mm, and they go on with the reach, arriving at 10.1 to
  10.5 s. With kp = 16 and kd = 1.4, it carries their hands 136 to 323 mm from the
  path; they arrive only at 14.0 to 18.7 s and leave the goal in their hold, and ESN
  #2's reference, given a zero reference velocity, never comes back to the reach: it
  wanders in mid-reach, joint 1 between about 27° and 63°, and ends 841 mm from the
  target.
- **Five of kp = 80 and kd = 1.2's 18 runs under the logs fail to hold,** the replay's
  under logs 000 and 001 and ESN #2's under log 000 given the reference velocity. Each
  arrives 0.4 to 1.0 s after a push's end, at 9.5 s, while joint 1 still rings, and its
  hand swings out of the 20 mm goal, by 0.7 to 4.3 mm, for 0.07 to 0.17 s, before it
  settles within it. With report 011's gains, the replay's hand comes as close to the
  edge, 19.8 to 20.0 mm, and stays within it. kp = 16 and kd = 1.4's replay fails to
  hold under log 000 the same way, out by up to 4.5 mm for up to 0.18 s.
- **The ESNs follow the pushes by hand less with the stiffer gains,** 0.21 to 0.82
  against 0.50 to 0.89: log 001's first push, 0.33 to 0.48 against 0.66 to 0.78, and
  log 002's second, 0.21 to 0.43 against 0.69 to 0.89.

### 3.6 Over the 13 pushes, and arriving

The median over report 011's three pushes and report 012's ten, with the least and the
most. A hand not back counts as back at the next push or the run's end; the count of
pushes it is not back after follows the return:

| Tracker | Following, ESN #1 / ESN #2 | Ringing (mm) | Back within 5 mm (s); not back | Farthest (mm) |
| --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 0.64 (0.25 to 0.85) / 0.54 (0.43 to 0.76) | 2.7 (0.9 to 8.8) / 2.7 (0.9 to 9.7) / 3.5 (1.7 to 10.8) | 0.90 (0.04 to 2.15); 0 / 1.13 (0.32 to 2.46); 0 / 0.17 (0.00 to 0.78); 0 | 27 (11 to 85) / 30 (16 to 77) / 15 (7 to 40) |
| kp 16, kd 1.4 for both joints, reference velocity | 0.70 (0.34 to 0.81) / 0.57 (0.43 to 0.74) | 4.3 (3.0 to 24.7) / 4.6 (3.2 to 22.3) / 5.4 (2.7 to 25.6) | 2.96 (1.69 to 6.29); 3 / 3.65 (2.02 to 6.59); 6 / 0.50 (0.11 to 1.18); 0 | 121 (57 to 361) / 109 (73 to 340) / 63 (33 to 209) |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 0.80 (0.50 to 0.87) / 0.69 (0.59 to 0.78) | 6.9 (2.0 to 11.4) / 6.9 (2.0 to 11.5) / 7.6 (2.6 to 13.8) | 2.84 (2.02 to 5.29); 3 / 3.33 (0.93 to 4.34); 4 / 0.35 (0.04 to 0.47); 0 | 121 (28 to 254) / 150 (32 to 222) / 70 (7 to 128) |
| kp 80, kd 1.2 for both joints, zero reference velocity | 0.64 (0.21 to 0.82) / 0.56 (0.39 to 0.75) | 2.6 (0.9 to 7.7) / 2.6 (0.8 to 8.8) / 3.5 (1.7 to 10.7) | 0.93 (0.06 to 2.32); 0 / 0.70 (0.12 to 2.64); 0 / 0.17 (0.00 to 0.78); 0 | 25 (11 to 82) / 29 (14 to 71) / 15 (7 to 40) |
| kp 16, kd 1.4 for both joints, zero reference velocity | 0.65 (0.31 to 0.92) / 0.57 (0.38 to 0.81) | 3.9 (2.3 to 17.8) / 4.1 (2.3 to 17.9) / 5.6 (2.7 to 25.9) | 3.50 (1.26 to 5.98); 4 / 4.06 (1.92 to 9.16); 5 / 0.51 (0.11 to 1.20); 0 | 97 (43 to 309) / 99 (48 to 303) / 66 (33 to 209) |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 0.77 (0.54 to 0.89) / 0.73 (0.59 to 0.85) | 6.3 (1.6 to 27.0) / 5.9 (1.6 to 25.2) / 7.6 (2.5 to 13.8) | 2.88 (2.02 to 5.34); 2 / 3.45 (1.12 to 5.90); 5 / 0.36 (0.03 to 0.47); 0 | 102 (27 to 274) / 126 (32 to 286) / 70 (6 to 128) |

When the hand arrives within the goal (s); * marks a hand that leaves the goal during
its 2 s hold, and – one that never arrives:

| Tracker | Undisturbed | Block | Pushes | Log 000 | Log 001 | Log 002 |
| --- | --- | --- | --- | --- | --- | --- |
| kp 80, kd 1.2 for both joints, reference velocity | 9.54 / 9.53 / 9.58 | 10.03 / 9.93 / 9.58 | 9.26 / 9.37 / 9.58 | 9.79 / 9.54* / 9.47* | 10.21 / 10.10 / 9.48* | 9.63 / 9.57 / 9.58 |
| kp 16, kd 1.4 for both joints, reference velocity | 9.49 / 9.48 / 9.49 | 10.03 / 9.89 / 9.49 | 9.81 / 9.86 / 9.49 | 11.15 / 10.53 / 9.60* | 14.22* / 18.71* / 9.46 | 9.97 / 9.82 / 9.49 |
| ω 20, ζ 0.4 (reports 011, 012), reference velocity | 9.55 / 9.54 / 9.56 | 10.08 / 9.98 / 9.56 | 9.70 / 9.75 / 9.56 | 10.15 / 9.87 / 9.56 | 10.35 / 10.11 / 9.56 | 9.37 / 9.46 / 9.56 |
| kp 80, kd 1.2 for both joints, zero reference velocity | 9.77 / 9.68 / 9.59 | 10.28 / 10.04 / 9.59 | 9.50 / 9.53 / 9.59 | 9.97 / 9.80 / 9.47* | 10.38 / 10.47 / 9.50* | 9.87 / 9.73 / 9.59 |
| kp 16, kd 1.4 for both joints, zero reference velocity | 10.95 / 10.50 / 9.61 | 11.56 / 10.82 / 9.61 | 9.76 / 10.57 / 9.61 | 12.01 / 11.73 / 9.60* | 13.98* / – / 9.56 | 11.53 / 10.82 / 9.61 |
| ω 20, ζ 0.4 (reports 011, 012), zero reference velocity | 10.18 / 9.96 / 9.61 | 10.74 / 10.31 / 9.61 | 9.19 / 10.07 / 9.61 | 10.32 / 10.04 / 9.59 | 13.39* / 14.13* / 9.61 | 10.18 / 9.90 / 9.61 |

Every arm ends within 3.0 mm of the target in every run, but ESN #2 with kp = 16 and
kd = 1.4 under log 001 given a zero reference velocity (841 mm).

- **kp = 80 and kd = 1.2 keep the hand a quarter as far from the path,** 25 to 30 mm
  in the median against 102 to 150 mm for the ESNs, and 15 mm against 70 mm for the
  replay; they ring 2.6 to 3.5 mm against 5.9 to 7.6 mm.
- **The ESNs come back in about a second, not three:** 0.70 to 1.13 s in the median
  against 2.84 to 3.45 s, after every push, against 2 to 5 not back. The replay is
  back within 0.17 s against 0.35 s.
- **They follow less:** 0.54 to 0.64 in the median against 0.69 to 0.80.
- **kp = 16 and kd = 1.4 do no better than report 011's gains,** straying 97 to 121 mm
  and coming back in 3.0 to 4.1 s, 3 to 6 pushes not back; they ring less, 3.9 to
  5.6 mm.

## 4. Observations

- **The chatter of equal gains is a sampling instability, and its bound is exact for
  the arm at rest.** With the same gains for both joints, the error splits along the
  mass matrix's eigenvectors, and the held torque destabilizes the mode of the smaller
  inertia once kd > 2 μ / T − kp T / 10. The bound depends on the posture through μ
  alone: here, joint 2's angle, least at the start. The model predicts where the error
  grows and how fast, to within a few percent, and the simulation's own integration
  steps have to be in it: the exact hold puts kd = 3's bound 2° too low.
- **Stiffness keeps the arm on the path; damping sets the ringing and the hold.**
  kp = 80 with kd = 1.2 moves the hand under the pushes no farther than kp = 80 with
  kd = 3 did after its chatter (33 to 39 mm in the median for the ESNs, 22 mm for the
  replay, in this report's earlier version, `55072ed`; here 25 to 30 mm and 15 mm),
  and a quarter as far as report 011's gains, whose joint 2 is nine times softer. With
  kd = 1.2, joint 1 rings longer (ζ = 0.15), which costs five holds under the logs,
  where the hand arrives right after a push.
- **Equal gains trade joint 1's damping against joint 2's bound.** Joint 1's error
  decays at about kd / (2 M₁₁) whatever kp, and joint 2's smaller inertia caps kd at
  1.54 N m s/rad, 1.25 from the offset starts: with the torque held for 10 ms, no equal
  gains give joint 1 alone more than ζ = 0.2 at kp = 80. Softer gains, kp = 16, damp
  joint 1 well but give way under the pushes as report 011's gains do, and lose an ESN
  under log 001.
- **The stiffer arm drags the ESNs' references less,** following by 0.54 to 0.64
  against 0.69 to 0.80, yet they come back sooner: the reference moves less because the
  arm does, and both have less to undo.

## 5. Next steps

1. **Per-joint damping inside the bound:** kp = 80 for both joints, kd = 3 N m s/rad
   for joint 1 and 1.0 for joint 2. Then the joints are coupled again, but the model of
   Section 3.1, the two together, finds every angle stable (largest |z| 0.95 over
   10 ms), where joint 2's kd = 1.1 grows below 12.5°; alone, joint 1 has ζ = 0.39 and
   joint 2 ζ = 0.37. It should keep kp = 80's small displacement without the ringing
   that fails the holds.
2. **The torque every 2 ms,** where the bound on kd is five times as high (8.0 N m s/rad
   at the start, kp = 80), and kd = 3 is stable at every angle.
