import cv2
import heapq
import math
import numpy as np
import rclpy

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
        self.declare_parameter('lambda', 1.0)

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.lambda_turn = self.get_parameter('lambda').get_parameter_value().double_value

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

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.map_info = None
        self.occupancy_grid = None
        self.current_pose = None

    def map_cb(self, msg):
        self.map_info = msg.info

        grid = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width))

        # 1 = occupied/unknown, 0 = free
        occupied = ((grid > 50) | (grid < 0)).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
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
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        theta_idx = round(yaw / (math.pi / 4)) % 8
        self.current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            theta_idx,
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

        start_uv = self.world_to_grid(start_point[0], start_point[1])
        end_uv = self.world_to_grid(end_point[0], end_point[1])
        start_theta = start_point[2] if len(start_point) > 2 else 0

        if not self.is_free(*start_uv):
            self.get_logger().error("Start position is inside an obstacle!")
            return
        if not self.is_free(*end_uv):
            self.get_logger().error("Goal position is inside an obstacle!")
            return

        path_uv = self.a_star((start_uv[0], start_uv[1], start_theta), end_uv)

        if not path_uv:
            self.get_logger().error("A* found no path!")
            return

        for u, v in path_uv:
            x, y = self.grid_to_world(u, v)
            self.trajectory.addPoint((x, y))

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    def a_star(self, start_state, end_uv):
        """A* search on the dilated occupancy grid.

        State: (u, v, theta_idx) where theta_idx in {0..7} = multiples of pi/4.
        Heuristic: octile distance to goal (theta-free, remains admissible).
        Turning penalty: self.lambda_turn * circular_delta(old_theta, new_theta).

        Args:
            start_state: (u, v, theta_idx)
            end_uv:      (u, v) grid coords of the goal
        Returns:
            List of (u, v) from start to goal, or empty list.
        """
        # Maps movement direction (du, dv) to the corresponding heading index.
        DIR_TO_THETA = {
            (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
            (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7,
        }

        eu, ev = end_uv
        su, sv, s_theta = start_state

        def heuristic(u, v):
            du, dv = abs(eu - u), abs(ev - v)
            return max(du, dv) + (math.sqrt(2) - 1) * min(du, dv)

        # heap entries: (f, g, u, v, theta_idx)
        open_heap = [(heuristic(su, sv), 0.0, su, sv, s_theta)]
        g_score = {(su, sv, s_theta): 0.0}
        came_from = {}

        while open_heap:
            f, g, u, v, theta = heapq.heappop(open_heap)

            if u == eu and v == ev:
                path = []
                state = (u, v, theta)
                while state in came_from:
                    path.append((state[0], state[1]))
                    state = came_from[state]
                path.append((su, sv))
                return list(reversed(path))

            cur_state = (u, v, theta)
            if g > g_score.get(cur_state, float('inf')):
                continue

            for nu, nv in self.get_neighbors(u, v):
                du, dv = nu - u, nv - v
                new_theta = DIR_TO_THETA[(du, dv)]
                move_cost = math.sqrt(2) if (du != 0 and dv != 0) else 1.0
                turn_delta = min(abs(new_theta - theta), 8 - abs(new_theta - theta))
                new_g = g + move_cost + self.lambda_turn * turn_delta
                new_state = (nu, nv, new_theta)
                if new_g < g_score.get(new_state, float('inf')):
                    g_score[new_state] = new_g
                    came_from[new_state] = cur_state
                    heapq.heappush(open_heap,
                                   (new_g + heuristic(nu, nv), new_g, nu, nv, new_theta))

        return []



def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
