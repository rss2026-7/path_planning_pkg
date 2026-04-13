import cv2
import heapq
import math
import numpy as np
import rclpy
import time

from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from path_planning.utils import LineTrajectory
from rclpy.node import Node


class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('dilation_radius', 10)
        self.declare_parameter('planner', 'astar')       # 'astar' or 'rrt'
        self.declare_parameter('rrt_max_iter', 5000)
        self.declare_parameter('rrt_step_size', 10)      # grid pixels per step
        self.declare_parameter('rrt_goal_bias', 0.1)     # probability [0-1] of sampling goal

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.dilation_radius = self.get_parameter('dilation_radius').get_parameter_value().integer_value
        self.planner = self.get_parameter('planner').get_parameter_value().string_value
        self.rrt_max_iter = self.get_parameter('rrt_max_iter').get_parameter_value().integer_value
        self.rrt_step_size = self.get_parameter('rrt_step_size').get_parameter_value().integer_value
        self.rrt_goal_bias = self.get_parameter('rrt_goal_bias').get_parameter_value().double_value

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory", color=(0.0, 0.5, 1.0))
        self.raw_trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory_raw", color=(1.0, 1.0, 1.0))

        self.map_info = None
        self.occupancy_grid = None
        self.current_pose = None

    def map_cb(self, msg):
        self.map_info = msg.info

        grid = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width))

        # 1 = occupied/unknown, 0 = free
        occupied = ((grid > 50) | (grid < 0)).astype(np.uint8)

        d = self.dilation_radius
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
        self.occupancy_grid = cv2.dilate(occupied, kernel, iterations=1)

        q = msg.info.origin.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.cos_yaw = math.cos(yaw)
        self.sin_yaw = math.sin(yaw)

        self.get_logger().info(
            f"Map received: {msg.info.width}x{msg.info.height}, "
            f"resolution={msg.info.resolution}, yaw={yaw:.3f}")

    def pose_cb(self, msg):
        self.current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )

    def goal_cb(self, msg):
        if self.occupancy_grid is None:
            self.get_logger().warn("No map received yet, cannot plan.")
            return
        if self.current_pose is None:
            self.get_logger().warn("No pose received yet, cannot plan.")
            return

        goal = (msg.pose.position.x, msg.pose.position.y)
        self.plan_path(self.current_pose, goal, self.occupancy_grid)

    def world_to_grid(self, x, y):
        """Convert world coords (x, y) to grid pixel coords (u, v)."""
        origin = self.map_info.origin.position
        dx = x - origin.x
        dy = y - origin.y
        mx = dx * self.cos_yaw + dy * self.sin_yaw
        my = -dx * self.sin_yaw + dy * self.cos_yaw
        u = int(mx / self.map_info.resolution)
        v = int(my / self.map_info.resolution)
        return u, v

    def grid_to_world(self, u, v):
        """Convert grid pixel coords (u, v) to world coords (x, y)."""
        origin = self.map_info.origin.position
        mx = u * self.map_info.resolution
        my = v * self.map_info.resolution
        x = origin.x + mx * self.cos_yaw - my * self.sin_yaw
        y = origin.y + mx * self.sin_yaw + my * self.cos_yaw
        return x, y

    def is_free(self, u, v):
        if 0 <= u < self.map_info.width and 0 <= v < self.map_info.height:
            return self.occupancy_grid[v, u] == 0
        return False

    def get_neighbors(self, u, v):
        """Return free 8-connected neighbors of (u, v)."""
        neighbors = []
        for du, dv in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nu, nv = u + du, v + dv
            if self.is_free(nu, nv):
                neighbors.append((nu, nv))
        return neighbors

    def plan_path(self, start_point, end_point, occupancy_grid):
        self.trajectory.clear()
        self.raw_trajectory.clear()

        start_uv = self.world_to_grid(start_point[0], start_point[1])
        end_uv = self.world_to_grid(end_point[0], end_point[1])

        if not self.is_free(*start_uv):
            self.get_logger().error("Start position is inside an obstacle!")
            return
        if not self.is_free(*end_uv):
            self.get_logger().error("Goal position is inside an obstacle!")
            return

        t0 = time.monotonic()
        if self.planner == 'rrt':
            path_uv = self.rrt(start_uv, end_uv)
        else:
            path_uv = self.a_star(start_uv, end_uv)
        elapsed = time.monotonic() - t0

        if not path_uv:
            self.get_logger().error(f"{self.planner} found no path! ({elapsed:.2f}s)")
            return

        self.get_logger().info(f"{self.planner} found path with {len(path_uv)} waypoints in {elapsed:.2f}s")

        # Publish raw (unsmoothed) path for debugging/presentation
        for u, v in path_uv:
            x, y = self.grid_to_world(u, v)
            self.raw_trajectory.addPoint((x, y))
        self.raw_trajectory.publish_viz()

        path_uv = self.smooth_path(path_uv)
        self.get_logger().info(f"After smoothing: {len(path_uv)} waypoints")

        for u, v in path_uv:
            x, y = self.grid_to_world(u, v)
            self.trajectory.addPoint((x, y))

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    def line_of_sight(self, u0, v0, u1, v1):
        """Check if the straight line between two grid cells is fully free."""
        n = max(abs(u1 - u0), abs(v1 - v0)) + 1
        us = np.round(np.linspace(u0, u1, n)).astype(int)
        vs = np.round(np.linspace(v0, v1, n)).astype(int)
        return bool(np.all(self.occupancy_grid[vs, us] == 0))

    def smooth_path(self, path_uv):
        """Greedily remove intermediate waypoints while maintaining line of sight."""
        if len(path_uv) <= 2:
            return path_uv
        smoothed = [path_uv[0]]
        i = 0
        while i < len(path_uv) - 1:
            j = len(path_uv) - 1
            while j > i + 1:
                u0, v0 = path_uv[i]
                u1, v1 = path_uv[j]
                if self.line_of_sight(u0, v0, u1, v1):
                    break
                j -= 1
            smoothed.append(path_uv[j])
            i = j
        return smoothed

    def a_star(self, start_uv, end_uv):
        """A* search on the dilated occupancy grid.

        Args:
            start_uv: (u, v) grid coords of the start
            end_uv:   (u, v) grid coords of the goal
        Returns:
            List of (u, v) from start to goal, or empty list.
        """
        eu, ev = end_uv
        su, sv = start_uv

        def heuristic(u, v):
            du, dv = abs(eu - u), abs(ev - v)
            return max(du, dv) + (math.sqrt(2) - 1) * min(du, dv)

        # heap entries: (f, g, u, v)
        open_heap = [(heuristic(su, sv), 0.0, su, sv)]
        g_score = {(su, sv): 0.0}
        came_from = {}

        while open_heap:
            f, g, u, v = heapq.heappop(open_heap)

            if u == eu and v == ev:
                path = []
                state = (u, v)
                while state in came_from:
                    path.append(state)
                    state = came_from[state]
                path.append((su, sv))
                return list(reversed(path))

            cur_state = (u, v)
            if g > g_score.get(cur_state, float('inf')):
                continue

            for nu, nv in self.get_neighbors(u, v):
                du, dv = nu - u, nv - v
                move_cost = math.sqrt(2) if (du != 0 and dv != 0) else 1.0
                new_g = g + move_cost
                new_state = (nu, nv)
                if new_g < g_score.get(new_state, float('inf')):
                    g_score[new_state] = new_g
                    came_from[new_state] = cur_state
                    heapq.heappush(open_heap,
                                   (new_g + heuristic(nu, nv), new_g, nu, nv))

        return []

    def rrt(self, start_uv, end_uv):
        """RRT on the dilated occupancy grid.

        Args:
            start_uv: (u, v) grid coords of the start
            end_uv:   (u, v) grid coords of the goal
        Returns:
            List of (u, v) from start to goal, or empty list.
        """
        import random

        tree = {start_uv: None}   # child -> parent
        nodes = [start_uv]
        eu, ev = end_uv
        step = self.rrt_step_size

        for _ in range(self.rrt_max_iter):
            # Goal-biased random sampling
            if random.random() < self.rrt_goal_bias:
                q_rand = end_uv
            else:
                q_rand = (
                    random.randint(0, self.map_info.width  - 1),
                    random.randint(0, self.map_info.height - 1),
                )

            # Nearest node in tree (squared distance is fine for comparison)
            q_near = min(nodes, key=lambda n: (n[0]-q_rand[0])**2 + (n[1]-q_rand[1])**2)

            # Steer toward q_rand by at most `step` pixels
            du = q_rand[0] - q_near[0]
            dv = q_rand[1] - q_near[1]
            dist = math.sqrt(du*du + dv*dv)
            if dist == 0:
                continue
            if dist > step:
                du = int(round(du / dist * step))
                dv = int(round(dv / dist * step))
            q_new = (q_near[0] + du, q_near[1] + dv)

            if not self.is_free(*q_new):
                continue
            if not self.line_of_sight(*q_near, *q_new):
                continue

            tree[q_new] = q_near
            nodes.append(q_new)

            # Check if we can reach the goal from q_new
            dg = math.sqrt((eu - q_new[0])**2 + (ev - q_new[1])**2)
            if dg <= step and self.line_of_sight(*q_new, eu, ev):
                tree[end_uv] = q_new
                break
        else:
            return []   # max iterations reached without connecting

        # Reconstruct path from goal back to start
        path = []
        node = end_uv
        while node is not None:
            path.append(node)
            node = tree[node]
        return list(reversed(path))


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
