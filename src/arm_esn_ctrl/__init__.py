# Copyright (C) 2026 Hiroshi Atsuta <atsuta@ieee.org>
# SPDX-License-Identifier: GPL-3.0-only

"""Echo state network reference generators for robot-arm control, learned from demonstration."""

import os

# rclib parallelizes its linear algebra with OpenMP, and the rounding of a parallel
# sum depends on how the work is split among threads, so repeated runs can differ in
# the last digits. One thread makes every run reproduce exactly on the same machine.
# This must be set before rclib is first imported; set OMP_NUM_THREADS to override it.
os.environ.setdefault("OMP_NUM_THREADS", "1")
