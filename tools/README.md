# Tools

Interactive apps that run a trained ESN or the robot live. Both load the robot
and the task from a demonstration configuration, such as
`experiments/demonstrations/reach_tvs.toml`, and both use skelarm's keys:
`Space` play/pause, `→`/`F` one step while paused, `R` or `Home` reset, `Q` quit.

## Watching a trained ESN live

`tools/esn_reference_app.py` runs a trained ESN interactively. It loads the ESN
from the `esn.toml` that `experiments/autonomous_esn.py` saves in its run
directory:

```bash
uv run python tools/esn_reference_app.py experiments/demonstrations/reach_tvs.toml \
    --model <run directory>/esn.toml
```

Drag the arm tip to choose a start posture, then press Play: the ESN is reset,
driven by the held start posture for its warm-up (consumed at once, or played at
negative times with "Show the warm-up in real time" or `--show-warmup`), and then
runs autonomously, and the arm shows every posture it generates until you pause
it. The side panel shows the reach and hold metrics as the run goes. If the TOML
file also has `[controller]` and `[simulator]` tables, as the demonstration
configurations do, "Compare with the demonstrator" simulates the demonstrator's
reach from the same start posture in the background, draws it under the ESN's
path, and compares the two; with the checkbox off (`--no-demonstrator`), nothing
is simulated. Reset returns the arm to the start posture of the last run, ready to
be posed again.

## Simulating the robot interactively

`tools/robot_app.py` simulates the arm in real time, so you can pose it, watch it
reach, and push it by hand:

```bash
# The ESN generates the reference; the tracker follows it.
uv run python tools/robot_app.py experiments/demonstrations/reach_tvs.toml \
    --law computed_torque --omega 10 \
    --model "$STORAGE/results/multi_demonstration_autonomous_reaching/20261002-213015-autonomous_tvs_all_distances/esn.toml"
# The demonstrator's reach from each run's start posture, replayed by time.
uv run python tools/robot_app.py experiments/demonstrations/reach_tvs.toml --law pd --omega 20
# The demonstrator's reach from a given posture, replayed by time: pose the arm
# elsewhere before Play to emulate an initial offset.
uv run python tools/robot_app.py experiments/demonstrations/reach_tvs.toml --law pd --omega 20 --pose 29.4,88.2
# The demonstrator's own controller, without a reference.
uv run python tools/robot_app.py experiments/demonstrations/reach_tvs.toml --demonstrator
# A recorded take, such as one taught by hand, replayed by time; no demonstrator needed.
uv run python tools/robot_app.py experiments/demonstrations/reach_manual_single.toml --law pd --omega 20 \
    --replay reports/004-manual-demonstration/data/20261006-152823-reach_manual_single/demo_00.sklog.npz
```

`$STORAGE` stands for the storage root, and `--law ct` is short for
`--law computed_torque`. With ω, the tracking error is critically damped unless
`--damping` sets a smaller damping ratio, such as `--damping 0.1`. Instead of ω,
`--kp` and `--kd` set the gains directly (one value, or one per joint, such as
`--kp 30,5 --kd 1,0.2`); a small `--kd` makes the tracking error oscillate.
`--zero-reference-velocity` gives the tracking law a zero reference velocity, so
that its derivative term damps the arm's own velocity rather than the velocity
error. The reference generator and the tracker are fixed at launch, and the
side panel shows them, with the tracking error's natural frequency and damping
ratio for each joint.

Before Play, dragging the tip poses the arm. During a run, running or paused,
dragging pulls the tip toward the cursor with a spring force (the drag stiffness
in the panel), which acts on top of the controller's torque. External forces never
act during the ESN's warm-up, which is consumed at once when a run starts. Reset
returns the arm to the start posture of the last run, ready to be posed again. A
faint gray arm shows the initial posture (the given one, or else the last run's
start), and a faint colored arm the reference posture. The side panel shows the
arrival and the hold, the tracking error and the reference's joint speed, and the
joint torque, its squared integral, and the external force, live.
