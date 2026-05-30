"""
robots.py — Robot model for the multi-robot simulator.

A Robot is the simulator's representation of one arm. It tracks just enough
state to answer two questions:
  1. Are you free, or busy?
  2. If busy, when will you be free again?

We don't simulate motion — durations are computed from geometry (see tasks.py).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from sim.tasks import Task

class RobotState(Enum):
    """Possible states for a robot in the simulator."""
    IDLE = "idle"
    BUSY = "busy"


@dataclass
class Robot:
    """One arm in the multi-robot simulator.

    Attributes:
        robot_id:        Unique integer identifier (0, 1, 2, ...).
        base_position:   (3,) numpy array — XYZ of the robot's base in the world.
        state:           Current state (IDLE or BUSY).
        free_at_time:    Simulation time at which the robot becomes idle again.
                         Only meaningful when state == BUSY. For IDLE, set to 0.
        current_task:    Task currently being executed, or None if idle.
        tasks_completed: How many tasks this robot has finished so far.
    """
    robot_id:        int
    base_position:   np.ndarray
    state:           RobotState         = RobotState.IDLE
    free_at_time:    float              = 0.0
    current_task:    Optional[Task]     = None
    tasks_completed: int                = 0


    def estimate_duration(self, task: Task) -> float:
        """Estimate how long this robot would take to execute the given task.

        Combines the task's base_duration (geometry of pick→place) with a
        travel-time term for the robot moving from its base to the pick point.
        Travel uses the same per_meter cost as tasks (3 s/m, hardcoded for Slice 1).
        """
        travel_distance = float(np.linalg.norm(self.base_position - task.pick_xyz))
        travel_time = 3.0 * travel_distance      # seconds (matches per_meter_s in tasks.py)
        return task.base_duration + travel_time

    def start_task(self, task: Task, current_sim_time: float) -> None:
        """Mark this robot as BUSY executing `task`, starting now."""
        if self.state != RobotState.IDLE:
            raise RuntimeError(
                f"Robot {self.robot_id} cannot start a task — currently {self.state.value}"
            )
        duration = self.estimate_duration(task)
        self.state = RobotState.BUSY
        self.current_task = task
        self.free_at_time = current_sim_time + duration

    def finish_task(self) -> Task:
        """Mark current task as complete, return to IDLE, return the finished task."""
        if self.state != RobotState.BUSY or self.current_task is None:
            raise RuntimeError(
                f"Robot {self.robot_id} has no task to finish (state={self.state.value})"
            )
        finished = self.current_task
        self.state = RobotState.IDLE
        self.current_task = None
        self.free_at_time = 0.0
        self.tasks_completed += 1
        return finished