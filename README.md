# Multi-Robot Collaborative Manipulation

A framework for coordinating 2–3 robotic arms on shared manipulation tasks,
using reinforcement learning for task allocation and ROS2 + MoveIt2 + Open3D
for the full execution stack.

**Status:** Work in progress.

## Planned experiments

1. **Multi-arm speedup** — does adding arms reduce batch completion time?
2. **RL vs. baseline allocators** — does PPO beat simpler scheduling heuristics?
3. **Motion planning latency** — does trajectory caching reduce planning time?

Results will be reported here once measured.
