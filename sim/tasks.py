"""
tasks.py — Task generation for the multi-robot simulator.

A Task represents one pick-and-place operation: pick an object at some location,
move it to a placement location. For Slice 1 we don't actually simulate motion —
we just model how long the task TAKES to execute, based on the pick→place distance.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np

@dataclass
class Task:
    """One pick-and-place operation.

    Attributes:
        task_id:       Unique integer identifier (matches index in the batch).
        pick_xyz:      (3,) numpy array — world-frame XYZ position to pick from.
        place_xyz:     (3,) numpy array — world-frame XYZ position to place at.
        base_duration: How long this task takes to execute (seconds), derived
                       from the pick→place travel distance plus fixed overhead.
    """
    task_id:       int
    pick_xyz:      np.ndarray
    place_xyz:     np.ndarray
    base_duration: float