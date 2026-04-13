#!/usr/bin/env python3

import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray


class TrajectoryMetrics(Node):
    def __init__(self):
        super().__init__("trajectory_metrics")

        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("trajectory_topic", "/trajectory/current")
        self.declare_parameter("goal_tolerance", 0.3)
        self.declare_parameter("print_every", 10)

        odom_topic = self.get_parameter("odom_topic").value
        trajectory_topic = self.get_parameter("trajectory_topic").value
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.print_every = int(self.get_parameter("print_every").value)

        self.trajectory: List[Tuple[float, float]] = []
        self.errors: List[float] = []
        self.latest_position: Tuple[float, float] | None = None
        self.goal_position: Tuple[float, float] | None = None
        self.sample_count = 0
        self.completed = False

        self.create_subscription(Odometry, odom_topic, self.odom_callback, 10)
        self.create_subscription(PoseArray, trajectory_topic, self.trajectory_callback, 10)

        self.get_logger().info("Trajectory metrics node started.")
        self.get_logger().info(f"Subscribed to odom: {odom_topic}")
        self.get_logger().info(f"Subscribed to trajectory: {trajectory_topic}")

    def trajectory_callback(self, msg: PoseArray):
        if not msg.poses:
            self.get_logger().warn("Received empty trajectory.")
            return

        self.trajectory = [(pose.position.x, pose.position.y) for pose in msg.poses]
        self.goal_position = self.trajectory[-1]

        self.errors.clear()
        self.sample_count = 0
        self.completed = False

        self.get_logger().info(f"Received trajectory with {len(self.trajectory)} points.")
        self.get_logger().info(
            f"Goal = ({self.goal_position[0]:.3f}, {self.goal_position[1]:.3f})"
        )

    def odom_callback(self, msg: Odometry):
        if not self.trajectory:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.latest_position = (x, y)

        cte = self.compute_cross_track_error(x, y, self.trajectory)
        self.errors.append(cte)
        self.sample_count += 1

        avg_cte = sum(self.errors) / len(self.errors)
        max_cte = max(self.errors)

        if self.sample_count % self.print_every == 0:
            self.get_logger().info(
                f"Sample {self.sample_count}: "
                f"current CTE = {cte:.3f} m, "
                f"avg CTE = {avg_cte:.3f} m, "
                f"max CTE = {max_cte:.3f} m"
            )

        if self.goal_position is not None:
            goal_dist = math.hypot(x - self.goal_position[0], y - self.goal_position[1])

            if (not self.completed) and goal_dist <= self.goal_tolerance:
                self.completed = True
                self.get_logger().info("=== TRAJECTORY COMPLETED ===")
                self.get_logger().info(f"Final distance to goal: {goal_dist:.3f} m")
                self.get_logger().info(f"Average CTE: {avg_cte:.3f} m")
                self.get_logger().info(f"Max CTE: {max_cte:.3f} m")

    @staticmethod
    def compute_cross_track_error(
        x: float,
        y: float,
        trajectory: List[Tuple[float, float]],
    ) -> float:
        min_dist = float("inf")
        for px, py in trajectory:
            dist = math.hypot(x - px, y - py)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def destroy_node(self):
        if self.errors:
            avg_cte = sum(self.errors) / len(self.errors)
            max_cte = max(self.errors)

            self.get_logger().info("=== FINAL METRICS SUMMARY ===")
            self.get_logger().info(f"Samples collected: {len(self.errors)}")
            self.get_logger().info(f"Average CTE: {avg_cte:.3f} m")
            self.get_logger().info(f"Max CTE: {max_cte:.3f} m")

            if self.latest_position is not None and self.goal_position is not None:
                final_goal_dist = math.hypot(
                    self.latest_position[0] - self.goal_position[0],
                    self.latest_position[1] - self.goal_position[1],
                )
                success = final_goal_dist <= self.goal_tolerance
                self.get_logger().info(f"Final distance to goal: {final_goal_dist:.3f} m")
                self.get_logger().info(f"Completion: {'SUCCESS' if success else 'FAIL'}")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryMetrics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
