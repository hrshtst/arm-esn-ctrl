# Single-demonstration autonomous reaching (Stage 1)

An ESN trained on one demonstration, the last of the eight time-varying-stiffness
demonstrations (demo 7), runs on its own, its output fed back as its next input,
from that demonstration's start posture and from postures offset around it. Each
run is compared with the demonstrator's own reach from the same posture.

This experiment is being taken up again for a deeper analysis; its plan and
configurations will be added here.

| Configuration | Trained on | Runs from |
| --- | --- | --- |
| `autonomous_tvs_demo07.toml` | demo 7 only | its start, and offsets of 3 and 10 deg around it |

```bash
uv run python experiments/autonomous_esn.py \
    experiments/single_demonstration_autonomous_reaching/autonomous_tvs_demo07.toml
```

The runner and its outputs are those of
[multi-demonstration autonomous reaching](../multi_demonstration_autonomous_reaching/README.md).

## Runs

None yet on the current demonstrations: the runs of this configuration predated
them and were removed.
