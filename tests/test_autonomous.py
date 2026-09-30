# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Tests for the start postures of autonomous runs."""

import numpy as np
import pytest

from arm_esn_ctrl.autonomous import start_postures


def test_start_postures_are_offsets_around_each_demonstration_then_new_ones():
    demos = {"demo_00": np.radians([[10.0, 20.0], [30.0, 40.0]]), "demo_01": np.radians([[50.0, 60.0]])}
    evaluation = {"start_offsets_deg": [[0.0, 0.0], [3.0, -3.0]], "extra_start_q_deg": [[70.0, 80.0]]}

    starts = start_postures(evaluation, demos)

    assert [s.origin for s in starts] == ["demo_00", "demo_00 +3,-3 deg", "demo_01", "demo_01 +3,-3 deg", "new"]
    assert [s.demonstrated for s in starts] == [True, False, True, False, False]
    assert np.degrees(starts[1].q) == pytest.approx([13.0, 17.0])
    assert np.degrees(starts[4].q) == pytest.approx([70.0, 80.0])


def test_extra_start_postures_are_optional():
    demos = {"demo_07": np.radians([[18.2, 119.9]])}

    starts = start_postures({"start_offsets_deg": [[0.0, 0.0]]}, demos)

    assert [s.origin for s in starts] == ["demo_07"]
