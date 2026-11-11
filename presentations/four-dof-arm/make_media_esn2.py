# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Plots of ESN #2 against the replay alone, for slides.

    uv run python presentations/four-dof-arm/make_media_esn2.py

The plots of ``make_media.py`` and ``make_media_smaller.py``, from the same runs, with
ESN #1 and the progress replay left out: step 5's ESN #2 against the replay of the take
by time, undisturbed and under the block (step 5's runs), and under the half and the
quarter pushes and from the 5 and 2.5 deg offset starts (step 6's runs). Under the
storage root, it writes them into ``presentations/four-dof-arm_esn2-and-replay/``, a
folder per scene, with a ``README.md`` that names each scene's videos in the other two
folders, and ``media.csv``. It makes no videos, and leaves the other folders as they are.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np
from make_media import (
    ESNS,
    REPLAYS,
    START_CUT,
    TAKES,
    Run,
    all_starts_plot,
    check_replays_equal,
    commit_state,
    disturbances,
    log_path,
    lowercase,
    series_plot,
    start_of,
    tip_plot,
)
from make_media import SCENES as STUDY_SCENES
from make_media_smaller import ALL_STARTS as SMALLER_ALL_STARTS
from make_media_smaller import SCENES as SMALLER_SCENES
from make_media_smaller import STARTS

from arm_esn_ctrl.storage import storage_root

OUT = storage_root() / "presentations" / "four-dof-arm_esn2-and-replay"
ESN = "esn2"
SHOWN = {"replay": REPLAYS["replay"]}  # the replays drawn: the replay by time alone
STUDY, SMALLER = "four-dof-arm", "four-dof-arm_smaller-disturbances"  # the folders of the videos, under presentations/


def numbered(index: int, name: str) -> str:
    """A scene's folder ``name`` (such as "01_pushes_half") numbered ``index`` instead."""
    return f"{index:02d}_{name.split('_', 1)[1]}"


# The scenes, in the slides' order: folder here, folder of its videos, scenario, offset start, and title.
SOURCES = [(STUDY, scene) for scene in STUDY_SCENES if scene[1] in ("nominal", "block")]
SOURCES += [(SMALLER, scene) for scene in SMALLER_SCENES]
SCENES = [
    (numbered(i, name), f"{source}/{name}", scenario, offset, title)
    for i, (source, (name, scenario, offset, title)) in enumerate(SOURCES, 1)
]
# The tip paths from all starts of an offset size: folder here, folder of the plots with both replays, scenario.
ALL_STARTS = [
    (numbered(len(SCENES) + i, name), f"{SMALLER}/{name}", scenario)
    for i, (name, scenario) in enumerate(SMALLER_ALL_STARTS.items(), 1)
]

README = """\
# ESN #2 against the replay, for slides

Plots of the four-link arm's ESN #2 against the replay of the take by time, with ESN #1
and the progress replay left out. They are the plots of the folders `four-dof-arm/`
(step 5) and `four-dof-arm_smaller-disturbances/` (step 6) beside this one, from the same
runs, whose READMEs describe the arm, the take, the ESNs, the tracker, and the
disturbances. They are not kept in Git; `presentations/four-dof-arm/make_media_esn2.py`
in the arm-esn-ctrl repository makes them.

- **{label}**, blue: {setting}.
  Step 5's rule, which asks the reference to follow the arm, chose it.
- **Replay (by time)**, orange: the take replayed by time, whatever the arm does.

## The folders

Each scene's videos are in the folder named, as `esn2.mp4` and `replay.mp4`, one arm
each; its `compare_esn2.mp4` also shows the progress replay.

| Folder | Scene | Its videos |
| --- | --- | --- |
{scenes}
{all_starts}

In each of `01` to `{last:02d}`:

| File | What it shows |
| --- | --- |
| `joints.png` | each joint's angle over time: ESN #2 over the replay, the take thick gray, the ESN's reference dashed, the disturbances shaded |
| `torques.png` | each joint's torque over time, likewise (from an offset start, the axis leaves out the task's first 50 ms, whose peak would crush the rest) |
| `tip_xy.png` | the tip paths in the plane, with the take's, the start posture, and the goal |

The replay's runs are those of ESN #2's runs, checked to equal those of ESN #1's, from
which the replay's videos were made. The plots are 1800 x 1012 px (16:9), but for the
tip paths from all starts, {all_starts_px}. [`media.csv`](media.csv) lists every file
with the logs it shows, relative to the storage root.

## Reproducing them

From the repository, at commit `{commit}`{dirty}:

    uv run python presentations/four-dof-arm/make_media_esn2.py

It reads the runs under the storage root and overwrites the files here, this README
included. This README was written on {day}.
"""  # noqa: E501


def main() -> None:
    # The replay's videos are of ESN #1's runs; the plots here are of ESN #2's, the same.
    starts = {scenario: 1 for _, _, scenario, offset, _ in SCENES if offset is None}
    check_replays_equal(starts | {scenario: STARTS for _, _, scenario in ALL_STARTS})
    OUT.mkdir(parents=True, exist_ok=True)
    root = storage_root()
    label, _, setting = ESNS[ESN]
    against = f"{label} ({setting}) against the replay"
    taught = Run(root / TAKES["take_filtered"][0])
    media: list[dict[str, str]] = []

    def add(folder: str, name: str, shows: str, videos: str, logs: list[Path]) -> None:
        media.append(
            {"folder": folder, "file": name, "shows": shows, "videos": videos}
            | {"logs": "; ".join(str(p.relative_to(root)) for p in logs)}
        )

    for folder_name, videos, scenario, offset, title in SCENES:
        folder = OUT / folder_name
        folder.mkdir(exist_ok=True)
        start = start_of(scenario, ESN, offset)
        arms = ("esn", *SHOWN)
        logs = [log_path(scenario, ESN, arm, start) for arm in arms]
        runs = {arm: Run(path) for arm, path in zip(arms, logs, strict=True)}
        spans = [span for _, span in disturbances(scenario)]
        cut = START_CUT if offset is not None else -np.inf
        for channel, what in (("joints", "Joint angles"), ("torques", "Joint torques")):
            fig = series_plot(runs, ESN, f"{what}, {lowercase(title)}\n{against}", channel, spans, taught, cut)
            fig.savefig(folder / f"{channel}.png")
            add(folder_name, f"{channel}.png", f"{what.lower()} over time: {lowercase(title)}", videos, logs)
        tip_plot(runs, ESN, f"Tip paths, {lowercase(title)}\n{against}", taught).savefig(folder / "tip_xy.png")
        add(folder_name, "tip_xy.png", f"the tip paths: {lowercase(title)}", videos, logs)
        print(f"plotted {folder_name}")

    sizes = set()
    for folder_name, plots, scenario in ALL_STARTS:
        folder = OUT / folder_name
        folder.mkdir(exist_ok=True)
        fig, logs = all_starts_plot(ESN, taught, scenario, SHOWN)
        fig.savefig(folder / "tip_xy.png")
        width, height = fig.canvas.get_width_height()
        sizes.add(f"{width} x {height} px")
        add(folder_name, "tip_xy.png", f"tip paths from all {STARTS} starts (with both replays: {plots})", "", logs)
        print(f"plotted {folder_name}")

    with (OUT / "media.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(media[0]))
        writer.writeheader()
        writer.writerows(media)
    commit, dirty = commit_state()
    (OUT / "README.md").write_text(
        README.format(
            label=label,
            setting=setting,
            scenes="\n".join(f"| `{name}` | {title} | `{videos}/` |" for name, videos, _, _, title in SCENES),
            all_starts="\n".join(
                f"| `{name}` | From all {STARTS} starts offset by -{size}, 0, or +{size}° in each joint: `tip_xy.png`,"
                f" the tip paths of each arm | none; the same plot with both replays is in `{plots}/` |"
                for name, plots, scenario in ALL_STARTS
                for size in [scenario.removeprefix("offsets_").removesuffix("deg")]
            ),
            last=len(SCENES),
            all_starts_px=" and ".join(sorted(sizes)),
            commit=commit,
            dirty=dirty,
            day=date.today().isoformat(),
        )
    )
    print(f"wrote {len(media)} files and their README to {OUT}")


if __name__ == "__main__":
    main()
