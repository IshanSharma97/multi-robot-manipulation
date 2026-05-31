"""
schedulers.py — Task assignment policies for the multi-robot simulator.

A Scheduler decides: "given pending tasks and robots, who does what next?"
Different schedulers implement different policies. They share a common
interface (the Scheduler ABC) so the simulator can swap them transparently.

This module implements two baseline schedulers:
  • RoundRobinScheduler  — cycles through robots in order, ignoring geometry.
  • NearestRobotScheduler — picks the closest idle robot for each task.

These two serve as baselines for Experiment 2 (later) where we'll compare a
learned (RL) scheduler against them.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from sim.tasks import Task
from sim.robots import Robot, RobotState

class Scheduler(ABC):
    """Abstract base class for all task assignment policies.

    Subclasses MUST implement assign(). The simulator calls assign() each time
    one or more robots become idle, and expects a list of (robot_id, task_idx)
    pairs telling it who should start which task right now.
    """

    @abstractmethod
    def assign(
        self,
        pending_tasks: List[Task],
        robots: List[Robot],
        current_sim_time: float,
    ) -> List[Tuple[int, int]]:
        """Decide which pending tasks should be started on which idle robots.

        Args:
            pending_tasks:    Tasks not yet assigned or completed. Index in this
                              list IS the value returned in (robot_id, task_idx).
            robots:           ALL robots in the fleet (both idle and busy).
                              Use robot.state == RobotState.IDLE to filter.
            current_sim_time: The simulator's current clock time, in seconds.
                              Not used by simple schedulers, but available for
                              policies that need it (e.g. RL).

        Returns:
            A list of (robot_id, task_index) pairs. Each pair means "robot_id
            should start pending_tasks[task_index] right now." The list may be
            empty if no good assignment is available (e.g. no idle robots).

            INVARIANT: Each robot_id appears at most once. Each task_index
            appears at most once. The scheduler MUST NOT double-assign.
        """
        ...

class RoundRobinScheduler(Scheduler):
    """Cycle through robots in order, ignoring task and robot positions.

    Maintains a `_next_robot` cursor that points to the robot whose turn it is.
    On each assign() call, scans forward from the cursor to find an idle robot,
    gives it the next pending task, and advances the cursor.

    Used as a baseline for comparing smarter schedulers against.
    """

    def __init__(self, num_robots: int) -> None:
        self._num_robots = num_robots
        self._next_robot = 0          # which robot's turn is it next

    def assign(
        self,
        pending_tasks: List[Task],
        robots: List[Robot],
        current_sim_time: float,
    ) -> List[Tuple[int, int]]:
        assignments: List[Tuple[int, int]] = []
        task_idx = 0                  # which pending task to assign next

        # We'll try every robot once, in round-robin order, until either
        # all idle robots are assigned or we run out of pending tasks.
        for _ in range(self._num_robots):
            if task_idx >= len(pending_tasks):
                break                 # no more pending tasks

            robot = robots[self._next_robot]
            if robot.state == RobotState.IDLE:
                assignments.append((self._next_robot, task_idx))
                task_idx += 1

            # Advance cursor — wrap around at the end of the fleet
            self._next_robot = (self._next_robot + 1) % self._num_robots

        return assignments


class NearestRobotScheduler(Scheduler):
    """Greedy: assign each pending task to its closest idle robot.

    For each pending task, computes Euclidean distance from every idle robot's
    base position to the task's pick position, and picks the nearest.

    Greedy in the sense that it commits to one (task, robot) pairing before
    considering the next task — this is not globally optimal (Hungarian
    assignment would be), but it's close enough for typical workloads and
    much simpler.
    """

    def assign(
        self,
        pending_tasks: List[Task],
        robots: List[Robot],
        current_sim_time: float,
    ) -> List[Tuple[int, int]]:
        assignments: List[Tuple[int, int]] = []

        # Track which robots we've already assigned this call (can't double-book).
        used_robot_ids: set[int] = set()

        for task_idx, task in enumerate(pending_tasks):
            best_robot_id = -1
            best_distance = float("inf")

            for robot in robots:
                if robot.state != RobotState.IDLE:
                    continue
                if robot.robot_id in used_robot_ids:
                    continue

                distance = float(np.linalg.norm(robot.base_position - task.pick_xyz))
                if distance < best_distance:
                    best_distance = distance
                    best_robot_id = robot.robot_id

            if best_robot_id == -1:
                # No idle robot available — all remaining tasks must wait.
                break

            assignments.append((best_robot_id, task_idx))
            used_robot_ids.add(best_robot_id)

        return assignments