import csv
import os
import time

import rclpy
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = .75        # meters; tune based on speed and path curvature
        self.speed = 1.0            # m/s
        self.wheelbase_length = 0.21  # meters; MIT RACECAR wheelbase

        self.initialized_traj = False
        self.trajectory = LineTrajectory(self, "/followed_trajectory")

        self.pose_sub = self.create_subscription(Odometry,
                                                 self.odom_topic,
                                                 self.pose_callback,
                                                 1)
        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)

        self.lookahead_pub = self.create_publisher(Marker, "/followed_trajectory/lookahead_point", 1)

        # --- Ground truth & PF pose subscriptions (for logging / visualization) ---
        self.gt_pose = None
        self.pf_pose = None

        self.gt_sub = self.create_subscription(Odometry, "/odom", self.gt_pose_callback, 1)
        self.pf_sub = self.create_subscription(Odometry, "/pf/pose/odom", self.pf_pose_callback, 1)

        # Trail markers
        self.gt_trail_pub = self.create_publisher(Marker, "/trajectory_follower/gt_trail", 1)
        self.pf_trail_pub = self.create_publisher(Marker, "/trajectory_follower/pf_trail", 1)
        self.gt_trail_points = []
        self.pf_trail_points = []

        # CSV buffers: list of (timestamp, x, y, cross_track_error)
        self.cte_gt_buffer = []
        self.cte_pf_buffer = []
        self.csv_dir = os.getcwd()

        self.get_logger().info(f"PurePursuit initialized")
        self.get_logger().info(f"  Subscribing to pose on: {self.odom_topic}")
        self.get_logger().info(f"  Subscribing to trajectory on: /trajectory/current")
        self.get_logger().info(f"  Publishing drive commands to: {self.drive_topic}")
        self.get_logger().info(f"  lookahead={self.lookahead}m  speed={self.speed}m/s  wheelbase={self.wheelbase_length}m")

    def _stop(self):
        """Publish a zero drive command and clear the active trajectory."""
        stop_cmd = AckermannDriveStamped()
        stop_cmd.drive.speed = 0.0
        stop_cmd.drive.steering_angle = 0.0
        self.drive_pub.publish(stop_cmd)

    def pose_callback(self, odometry_msg):
        if not self.initialized_traj:
            self.get_logger().info("Pose received but no trajectory yet — waiting.",
                                   throttle_duration_sec=5.0)
            self._stop()
            return

        pose = odometry_msg.pose.pose
        car_x = pose.position.x
        car_y = pose.position.y

        # Extract yaw from quaternion
        q = pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        car_pos = np.array([car_x, car_y])
        pts = np.array(self.trajectory.points)

        if len(pts) < 2:
            return

        # Step 1: Find the closest trajectory segment to the car (vectorized)
        A = pts[:-1]   # segment start points
        B = pts[1:]    # segment end points
        AB = B - A
        AP = car_pos - A

        AB_sq = np.sum(AB * AB, axis=1)
        AB_sq = np.where(AB_sq < 1e-10, 1e-10, AB_sq)   # avoid division by zero
        t = np.clip(np.sum(AP * AB, axis=1) / AB_sq, 0.0, 1.0)

        closest = A + t[:, np.newaxis] * AB
        dists = np.linalg.norm(car_pos - closest, axis=1)
        min_seg_idx = int(np.argmin(dists))

        # Stop when within a small radius of the final goal point
        dist_to_end = np.linalg.norm(car_pos - pts[-1])
        if dist_to_end <= 0.25:
            self.get_logger().info("Reached end of trajectory — stopping.")
            self.initialized_traj = False
            self._stop()
            return

        # Step 2: Find the lookahead point by searching forward from the closest segment.
        # Only accept intersections in front of the car (local_x >= 0) so segments
        # behind the car or perpendicular hits don't produce a backward lookahead point.
        lookahead_point = None
        for i in range(min_seg_idx, len(pts) - 1):
            candidates = self._circle_segment_intersection(car_pos, self.lookahead,
                                                           pts[i], pts[i + 1])
            for pt in candidates:  # t2 (farther) first, then t1
                dx_pt = pt[0] - car_x
                dy_pt = pt[1] - car_y
                local_x = np.cos(yaw) * dx_pt + np.sin(yaw) * dy_pt
                if local_x >= 0:
                    lookahead_point = pt
                    break
            if lookahead_point is not None:
                break

        # Fallback: steer toward the nearest point on the path so the car naturally
        # arcs back onto the trajectory (handles facing-away and overshoot cases).
        if lookahead_point is None:
            lookahead_point = closest[min_seg_idx]

        self._publish_lookahead_marker(lookahead_point)

        # Step 3: Compute pure pursuit steering angle
        dx = lookahead_point[0] - car_x
        dy = lookahead_point[1] - car_y
        L = np.hypot(dx, dy)

        if L < 1e-6:
            steering_angle = 0.0
            drive_speed = self.speed
        else:
            local_x = np.cos(yaw) * dx + np.sin(yaw) * dy

            far_from_path = dists[min_seg_idx] > self.lookahead * 0.5

            if local_x >= 0 or not far_from_path:
                # Normal forward pursuit
                alpha = np.arctan2(
                    np.sin(-yaw) * dx + np.cos(-yaw) * dy,
                    np.cos(-yaw) * dx - np.sin(-yaw) * dy
                )
                steering_angle = np.arctan2(2.0 * self.wheelbase_length * np.sin(alpha), L)
                drive_speed = self.speed
            else:
                # Target is behind AND car is far from path — reverse while steering toward it.
                # Flip effective heading (yaw + pi) so steering geometry is correct in reverse.
                rev_yaw = yaw + np.pi
                alpha = np.arctan2(
                    np.sin(-rev_yaw) * dx + np.cos(-rev_yaw) * dy,
                    np.cos(-rev_yaw) * dx - np.sin(-rev_yaw) * dy
                )
                steering_angle = np.arctan2(2.0 * self.wheelbase_length * np.sin(alpha), L)
                drive_speed = -self.speed
                self.get_logger().info("Recovery: reversing toward path.",
                                       throttle_duration_sec=1.0)

        steering_angle = float(np.clip(steering_angle, -0.34, 0.34))

        # Publish drive command
        drive_cmd = AckermannDriveStamped()
        drive_cmd.drive.speed = float(drive_speed)
        drive_cmd.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_cmd)

        self.get_logger().info(
            f"car=({car_x:.2f},{car_y:.2f}) yaw={np.degrees(yaw):.1f}deg "
            f"lookahead=({lookahead_point[0]:.2f},{lookahead_point[1]:.2f}) "
            f"steer={np.degrees(steering_angle):.1f}deg",
            throttle_duration_sec=0.5
        )

    def _circle_segment_intersection(self, center, radius, A, B):
        """
        Find intersections of a circle (center, radius) with segment AB.
        Returns a list of valid intersection points sorted by t descending
        (farther along segment first), so callers can apply their own forward filter
        and fall back to the closer hit if the farther one is behind the car.

        Algorithm based on: https://codereview.stackexchange.com/a/86428
        """
        d = B - A
        f = A - center

        a = np.dot(d, d)
        b = 2.0 * np.dot(f, d)
        c = np.dot(f, f) - radius * radius

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return []

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        pts = []
        if 0.0 <= t2 <= 1.0:
            pts.append(A + t2 * d)
        if 0.0 <= t1 <= 1.0:
            pts.append(A + t1 * d)
        return pts

    def _publish_lookahead_marker(self, point):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pure_pursuit"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.lookahead_pub.publish(marker)

    # --- Ground truth / PF pose callbacks (logging & trail visualization) ---

    def gt_pose_callback(self, msg):
        pos = msg.pose.pose.position
        self.gt_pose = np.array([pos.x, pos.y])

        if not self.initialized_traj:
            return

        # Trail
        self.gt_trail_points.append((pos.x, pos.y))
        self._publish_trail(self.gt_trail_pub, self.gt_trail_points,
                            r=0.0, g=1.0, b=0.0, marker_id=0)

        # CTE
        cte = self._compute_cross_track_error(self.gt_pose)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.cte_gt_buffer.append((t, pos.x, pos.y, cte))

    def pf_pose_callback(self, msg):
        pos = msg.pose.pose.position
        self.pf_pose = np.array([pos.x, pos.y])

        if not self.initialized_traj:
            return

        # Trail
        self.pf_trail_points.append((pos.x, pos.y))
        self._publish_trail(self.pf_trail_pub, self.pf_trail_points,
                            r=1.0, g=0.0, b=0.0, marker_id=1)

        # CTE
        cte = self._compute_cross_track_error(self.pf_pose)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.cte_pf_buffer.append((t, pos.x, pos.y, cte))

    def _compute_cross_track_error(self, position):
        """Minimum perpendicular distance from position to the planned trajectory."""
        pts = np.array(self.trajectory.points)
        if len(pts) < 2:
            return 0.0

        A = pts[:-1]
        B = pts[1:]
        AB = B - A
        AP = position - A

        AB_sq = np.sum(AB * AB, axis=1)
        AB_sq = np.where(AB_sq < 1e-10, 1e-10, AB_sq)
        t = np.clip(np.sum(AP * AB, axis=1) / AB_sq, 0.0, 1.0)

        closest = A + t[:, np.newaxis] * AB
        dists = np.linalg.norm(position - closest, axis=1)
        return float(np.min(dists))

    def _publish_trail(self, publisher, points, r, g, b, marker_id):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory_trail"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # line width
        marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
        marker.pose.orientation.w = 1.0

        for (px, py) in points:
            p = Point()
            p.x = float(px)
            p.y = float(py)
            p.z = 0.0
            marker.points.append(p)

        publisher.publish(marker)

    def _flush_csv(self):
        """Write current CTE buffers to CSV files."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")

        if self.cte_gt_buffer:
            path = os.path.join(self.csv_dir, f"cte_ground_truth_{timestamp_str}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "x", "y", "cross_track_error"])
                writer.writerows(self.cte_gt_buffer)
            self.get_logger().info(f"Wrote GT CTE CSV: {path}")

        if self.cte_pf_buffer:
            path = os.path.join(self.csv_dir, f"cte_particle_filter_{timestamp_str}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "x", "y", "cross_track_error"])
                writer.writerows(self.cte_pf_buffer)
            self.get_logger().info(f"Wrote PF CTE CSV: {path}")

    def _clear_trail_marker(self, publisher, marker_id):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory_trail"
        marker.id = marker_id
        marker.action = Marker.DELETE
        publisher.publish(marker)

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        # Flush previous run's CTE data to CSV before resetting
        self._flush_csv()

        # Reset trail markers in RViz
        self._clear_trail_marker(self.gt_trail_pub, 0)
        self._clear_trail_marker(self.pf_trail_pub, 1)
        self.gt_trail_points = []
        self.pf_trail_points = []

        # Reset CTE buffers
        self.cte_gt_buffer = []
        self.cte_pf_buffer = []

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    try:
        rclpy.spin(follower)
    except KeyboardInterrupt:
        pass
    finally:
        follower._flush_csv()
        follower.destroy_node()
    rclpy.shutdown()
