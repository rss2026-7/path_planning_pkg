import cv2
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

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value

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

        start_uv = self.world_to_grid(*start_point)
        end_uv = self.world_to_grid(*end_point)

        if not self.is_free(*start_uv):
            self.get_logger().error("Start position is inside an obstacle!")
            return
        if not self.is_free(*end_uv):
            self.get_logger().error("Goal position is inside an obstacle!")
            return

        path_uv = self.a_star(start_uv, end_uv)

        if not path_uv:
            self.get_logger().error("A* found no path!")
            return

        for u, v in path_uv:
            x, y = self.grid_to_world(u, v)
            self.trajectory.addPoint((x, y))

        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()

    def a_star(self, start_uv, end_uv):
        """A* search on the dilated occupancy grid.

        Args:
            start_uv: (u, v) grid coords of the start
            end_uv:   (u, v) grid coords of the goal
        Helpers:
            self.get_neighbors(u, v), self.is_free(u, v)
        Returns:
            List of (u, v) from start to goal, or empty list.
        """
        raise NotImplementedError  # TODO: Weiming


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
