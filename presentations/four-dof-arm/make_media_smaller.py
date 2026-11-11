# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Videos and plots of the four-link arm under smaller disturbances, for slides.

    uv run python presentations/four-dof-arm/make_media_smaller.py

From step 6's runs of the four-link consistent search (step 5's ESN #1 and ESN #2 against
the replay and the progress replay, under smaller pushes and from nearer offset starts;
``experiments/manual_demonstration_four_dof_consistent_search/README.md``), under the
storage root, it writes into ``presentations/four-dof-arm_smaller-disturbances/`` there a
folder per scene, of the same files as ``make_media.py``'s scenes, and each offset size's
tip paths from all 54 starts. ``presentations/four-dof-arm/`` is left as it is: the take,
the undisturbed runs, and the block are there. Its ``README.md`` has ``demo/README.md``
(how to run the arms live in the simulator) as a section, as ``make_media.py``'s does;
``--readme-only`` rewrites the README alone, leaving the media as they are.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from make_media import (
    ESNS,
    FPS,
    REPLAYS,
    TAKES,
    Run,
    all_starts_plot,
    check_replays_equal,
    commit_state,
    export_scenes,
    live_demo,
    readme_only,
)

from arm_esn_ctrl.storage import storage_root

OUT = storage_root() / "presentations" / "four-dof-arm_smaller-disturbances"
# The scenes: folder, scenario (a configuration's name before "_robot_"), offset start, and title.
SCENES = [
    ("01_pushes_half", "pushes_half", None, "Pushed at half force: 1.5 N at 2.6 s, 2.5 N at 6.7 s, 2 N at 14.9 s"),
    (
        "02_pushes_quarter",
        "pushes_quarter",
        None,
        "Pushed at a quarter force: 0.75 N at 2.6 s, 1.25 N at 6.7 s, 1 N at 14.9 s",
    ),
    ("03_offset_+5_+5_+5_+5", "offsets_5deg", "+5,+5,+5,+5", "From the start offset by +5° in every joint"),
    ("04_offset_+0_+0_-5_-5", "offsets_5deg", "+0,+0,-5,-5", "From the start offset by -5° in joints 3 and 4"),
    ("05_offset_+0_-5_-5_-5", "offsets_5deg", "+0,-5,-5,-5", "From the start offset by -5° in joints 2, 3, and 4"),
    (
        "06_offset_+2.5_+2.5_+2.5_+2.5",
        "offsets_2.5deg",
        "+2.5,+2.5,+2.5,+2.5",
        "From the start offset by +2.5° in every joint",
    ),
    (
        "07_offset_+0_+0_-2.5_-2.5",
        "offsets_2.5deg",
        "+0,+0,-2.5,-2.5",
        "From the start offset by -2.5° in joints 3 and 4",
    ),
    (
        "08_offset_+0_-2.5_-2.5_-2.5",
        "offsets_2.5deg",
        "+0,-2.5,-2.5,-2.5",
        "From the start offset by -2.5° in joints 2, 3, and 4",
    ),
]
ALL_STARTS = {"09_offsets_all_5deg": "offsets_5deg", "10_offsets_all_2.5deg": "offsets_2.5deg"}
STARTS = 54  # from each offset size

README = """\
# The four-link arm under smaller disturbances, for slides

Videos and plots of step 6 of
`experiments/manual_demonstration_four_dof_consistent_search/` in the arm-esn-ctrl
repository (its README.md gives the protocol): step 5's ESN #1 and ESN #2, not trained
again, against the replay and the progress replay, under smaller pushes and from nearer
offset starts. They are not kept in Git; `presentations/four-dof-arm/make_media_smaller.py`
there makes them.

The arm, the take, the ESNs, the tracker, and the time axis are those of the folder
`presentations/four-dof-arm/` beside this one, whose README.md describes them. That folder
also holds the take, the undisturbed runs, the block, and the larger pushes and offset
starts, none of which changed.

- **Pushes:** the same three, at the same times, for the same 0.5 s, and in the same
  directions, with every force at half (1.5, 2.5, and 2 N) or at a quarter (0.75, 1.25,
  and 1 N) of its size.
- **Offset starts:** every combination of -s, 0, and s deg in each joint, within the joint
  limits, for s of 5 and of 2.5 deg: 54 starts each. The earlier set has s of 15 deg.

## The folders

| Folder | Scene |
| --- | --- |
{scenes}
| `09_offsets_all_5deg`, `10_offsets_all_2.5deg` | the tip paths of each arm from all 54 starts of that offset size: `tip_xy_esn1.png`, `tip_xy_esn2.png` |

In each of `01` to `08`:

| File | What it shows |
| --- | --- |
| `esn1.mp4`, `esn2.mp4`, `replay.mp4`, `progress_replay.mp4` | one arm, with the player's side panel (the time, the joint angles, the tip, and the external force); the red arrow is the force at the tip |
| `compare_esn1.mp4`, `compare_esn2.mp4` | ESN #1 (or #2), the replay, and the progress replay side by side, with the time and the disturbance acting |
| `joints_esn1.png`, `joints_esn2.png` | each joint's angle over time: the ESN over both replays, the take thick gray, the ESN's and the progress replay's references dashed, the disturbances shaded |
| `torques_esn1.png`, `torques_esn2.png` | each joint's torque over time, likewise (from an offset start, the axis leaves out the task's first 50 ms, whose peak would crush the rest) |
| `tip_xy_esn1.png`, `tip_xy_esn2.png` | the tip paths in the plane, with the take's, the start posture, and the goal |

The replays' runs do not depend on the ESN: ESN #1's and ESN #2's runs give the same
replays, and their videos are made once. All videos are H.264 at {fps:g} fps, rendered by
skelarm's player; the plots are 1800 x 1012 px (16:9), but for the tip paths from all
starts, 2250 x 690 px. [`media.csv`](media.csv) lists every file with the logs it shows,
relative to the storage root.

## Live in the simulator

From the arm-esn-ctrl repository, as its `presentations/four-dof-arm/demo/README.md` says:

{demo}

## Reproducing them

From the repository, at commit `{commit}`{dirty}:

    uv run python presentations/four-dof-arm/make_media_smaller.py

It reads the runs under the storage root and overwrites the files here, this README
included; with `--readme-only`, it rewrites this README alone. This README was written
on {day}.
"""  # noqa: E501


def write_readme() -> None:
    """The folder's README.md, naming the commit of the code and the day."""
    commit, dirty = commit_state()
    (OUT / "README.md").write_text(
        README.format(
            scenes="\n".join(f"| `{name}` | {title} |" for name, _, _, title in SCENES),
            fps=FPS,
            demo=live_demo(),
            commit=commit,
            dirty=dirty,
            day=date.today().isoformat(),
        )
    )


def main() -> None:
    check_replays_equal({"pushes_half": 1, "pushes_quarter": 1, "offsets_5deg": STARTS, "offsets_2.5deg": STARTS})
    OUT.mkdir(parents=True, exist_ok=True)
    media: list[dict[str, str]] = []
    root = storage_root()

    def add(folder: str, name: str, shows: str, logs: list[Path]) -> None:
        media.append(
            {"folder": folder, "file": name, "shows": shows, "logs": "; ".join(str(p.relative_to(root)) for p in logs)}
        )

    taught = Run(root / TAKES["take_filtered"][0])
    export_scenes(OUT, SCENES, taught, add, [])
    for folder_name, scenario in ALL_STARTS.items():
        folder = OUT / folder_name
        folder.mkdir(exist_ok=True)
        for esn in ESNS:
            fig, logs = all_starts_plot(esn, taught, scenario)
            fig.savefig(folder / f"tip_xy_{esn}.png")
            add(
                folder_name,
                f"tip_xy_{esn}.png",
                f"tip paths from all {STARTS} starts, {ESNS[esn][0]} and both replays",
                logs,
            )

    with (OUT / "media.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(media[0]))
        writer.writeheader()
        writer.writerows(media)
    write_readme()
    print(f"wrote {len(media)} files and their README to {OUT}")


if __name__ == "__main__":
    assert set(REPLAYS) == {"replay", "progress"}
    if readme_only(__doc__):
        write_readme()
        print(f"wrote the README to {OUT}")
    else:
        main()
