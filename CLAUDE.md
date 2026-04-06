# CLAUDE.md — path_planning_pkg

## What is this?

This is **RSS Team 7's** implementation repo for **MIT RSS Lab 6: Path Planning**.
It is a standalone copy of the skeleton code from [mit-rss/path_planning](https://github.com/mit-rss/path_planning).
The team's GitHub org is [rss2026-7](https://github.com/rss2026-7).

## Project overview

The lab has three parts:

- **Part A — Path Planning** (`path_planning/trajectory_planner.py`): Implement a collision-free path planner (grid-based *or* sampling-based; both for bonus) that takes a start pose, goal pose, and occupancy grid map, and publishes a trajectory as a `PoseArray` to `/trajectory/current`.
- **Part B — Pure Pursuit** (`path_planning/trajectory_follower.py`): Implement a pure pursuit controller that follows trajectories by publishing `AckermannDriveStamped` to the drive topic.
- **Part C — Integration**: Combine Parts A and B with the team's particle-filter localization (from Lab 5) for real-time plan-and-drive on the physical racecar.

## Key files

| File | Purpose |
|---|---|
| `path_planning/trajectory_planner.py` | **TODO** — implement `PathPlan` node (map_cb, pose_cb, goal_cb, plan_path) |
| `path_planning/trajectory_follower.py` | **TODO** — implement `PurePursuit` node (pose_callback, tune lookahead/speed/wheelbase) |
| `path_planning/utils.py` | `LineTrajectory` utility class (provided, read-only unless extending) |
| `path_planning/trajectory_builder.py` | RViz click-to-build trajectory tool (provided) |
| `path_planning/trajectory_loader.py` | Load and publish saved `.traj` files (provided) |
| `config/sim/sim_config.yaml` | Sim parameters (odom_topic, map_topic, etc.) |
| `config/real/config.yaml` | Real-robot parameters |
| `maps/stata_basement.{png,yaml}` | Occupancy grid map of the Stata basement |
| `example_trajectories/` | Staff-provided reference trajectories for comparison |

## ROS 2 package info

- Package name: `path_planning` (ament_python)
- Build: `cd ~/racecar_ws && colcon build --packages-select path_planning && source install/setup.bash`

### Launch files

| Launch | What it does |
|---|---|
| `launch/sim/sim_plan.launch.xml` | Planner only (sim, ground-truth odom) |
| `launch/sim/sim_plan_follow.launch.xml` | Planner + follower (sim, ground-truth odom) |
| `launch/sim/pf_sim_plan_follow.launch.xml` | Planner + follower + particle filter (sim) |
| `launch/real/real.launch.xml` | Real robot launch |
| `launch/debug/build_trajectory.launch.xml` | Click in RViz to build a trajectory |
| `launch/debug/load_trajectory.launch.xml` | Load a saved `.traj` file |
| `launch/debug/follow_trajectory.launch.xml` | Follow a loaded trajectory |

### Key ROS topics

| Topic | Type | Description |
|---|---|---|
| `/trajectory/current` | `PoseArray` | Planned trajectory (planner publishes, follower subscribes) |
| `/goal_pose` | `PoseStamped` | Goal from RViz "2D Nav Goal" |
| `/map` | `OccupancyGrid` | Occupancy grid (publishes once) |
| `/odom` or `/pf/pose/odom` | `Odometry` | Car pose (ground-truth or particle filter) |

## Important constraints

- Path planner must compute a path in **< 30 seconds** between two arbitrary points. Staff reference: ~7 seconds for opposite ends of the map.
- Safety controller is **required** on the real car.
- Particle filter (Lab 5, in the [localization](https://github.com/rss2026-7/localization) repo) must be running for real-robot and PF-sim tests.

## Related team repos

- [rss2026-7/localization](https://github.com/rss2026-7/localization) — Lab 5 particle filter
- [rss2026-7/wall_follower_pkg](https://github.com/rss2026-7/wall_follower_pkg) — Lab 2 wall follower
- [rss2026-7/visual_servoing_pkg](https://github.com/rss2026-7/visual_servoing_pkg) — Lab 4 visual servoing
- [rss2026-7/safety_controller_pkg](https://github.com/rss2026-7/safety_controller_pkg) — Safety controller
- [rss2026-7/website](https://github.com/rss2026-7/website) — Team website
