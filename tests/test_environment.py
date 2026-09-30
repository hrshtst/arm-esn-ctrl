# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Check that the project and the pinned libraries are importable."""


def test_project_is_importable():
    import arm_esn_ctrl  # noqa: F401


def test_rclib_is_importable():
    import rclib  # noqa: F401


def test_skelarm_is_importable():
    import skelarm  # noqa: F401
