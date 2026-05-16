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


def generate_task_batch(
    n_tasks: int,
    seed: int,
    workspace_size: float = 1.0,
    base_overhead_s: float = 2.0,
    per_meter_s: float = 3.0,
    noise_std: float = 0.10,
) -> List[Task]:
    """Generate a reproducible batch of N tasks with random positions.

    Duration model:  base_duration = (base_overhead_s + per_meter_s * distance) * noise
    where `noise` is sampled from Normal(mean=1.0, std=noise_std), clipped to >= 0.5.

    Args:
        n_tasks:         How many tasks to generate.
        seed:            RNG seed — same seed always produces the same batch.
        workspace_size:  Side length of the cubic workspace (metres). Positions
                         are sampled uniformly in [-size/2, +size/2] for X/Y.
        base_overhead_s: Fixed time cost per task (grasp + release) in seconds.
        per_meter_s:     Time cost per metre of pick→place travel, in seconds.
        noise_std:       Std-dev of multiplicative noise on duration.

    Returns:
        A list of N Task objects.
    """
    rng = np.random.default_rng(seed)
    tasks: List[Task] = []

    for i in range(n_tasks):
        # Sample pick and place positions in the XY plane; Z = table height.
        pick = rng.uniform(-workspace_size / 2, workspace_size / 2, size=3)
        place = rng.uniform(-workspace_size / 2, workspace_size / 2, size=3)
        pick[2] = 0.05      # Table height — fixed for Slice 1.
        place[2] = 0.05

        # Geometry-derived duration with multiplicative execution noise.
        distance = float(np.linalg.norm(pick - place))
        noise = float(rng.normal(loc=1.0, scale=noise_std))
        noise = max(noise, 0.5)   # Clip — don't let noise produce < 50% time.
        duration = (base_overhead_s + per_meter_s * distance) * noise

        tasks.append(Task(
            task_id=i,
            pick_xyz=pick,
            place_xyz=place,
            base_duration=duration,
        ))

    return tasks