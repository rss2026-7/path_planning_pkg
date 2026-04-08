import rclpy
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
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

        self.lookahead = 1.5        # meters; tune based on speed and path curvature
        self.speed = 1.0            # m/s
        self.wheelbase_length = 0.325  # meters; MIT RACECAR wheelbase

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

    def pose_callback(self, odometry_msg):
        if not self.initialized_traj:
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

        # Step 2: Find the lookahead point by searching forward from the closest segment
        # Prefer intersections farther along the path (pick t2 over t1 within a segment)
        lookahead_point = None
        for i in range(min_seg_idx, len(pts) - 1):
            pt = self._circle_segment_intersection(car_pos, self.lookahead,
                                                   pts[i], pts[i + 1])
            if pt is not None:
                lookahead_point = pt
                break

        # If no intersection found (car is near the end or overshot), use the last point
        if lookahead_point is None:
            lookahead_point = pts[-1]

        # Step 3: Compute pure pursuit steering angle
        # Transform lookahead point into the car's local frame
        dx = lookahead_point[0] - car_x
        dy = lookahead_point[1] - car_y
        L = np.hypot(dx, dy)   # actual distance to lookahead point

        if L < 1e-6:
            steering_angle = 0.0
        else:
            # alpha = angle to lookahead point relative to car heading
            alpha = np.arctan2(
                np.sin(-yaw) * dx + np.cos(-yaw) * dy,   # local y
                np.cos(-yaw) * dx - np.sin(-yaw) * dy    # local x (unused, but for clarity)
            )
            # Pure pursuit: steering = arctan(2 * L_wheelbase * sin(alpha) / lookahead)
            steering_angle = np.arctan2(
                2.0 * self.wheelbase_length * np.sin(alpha), L)

        steering_angle = float(np.clip(steering_angle, -0.34, 0.34))

        # Publish drive command
        drive_cmd = AckermannDriveStamped()
        drive_cmd.drive.speed = float(self.speed)
        drive_cmd.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_cmd)

        self.get_logger().debug(
            f"car=({car_x:.2f},{car_y:.2f}) yaw={np.degrees(yaw):.1f}deg "
            f"lookahead=({lookahead_point[0]:.2f},{lookahead_point[1]:.2f}) "
            f"steer={np.degrees(steering_angle):.1f}deg"
        )

    def _circle_segment_intersection(self, center, radius, A, B):
        """
        Find the intersection of a circle (center, radius) with segment AB.
        Returns the intersection point farthest along the segment (largest valid t),
        or None if there is no intersection within the segment.

        Algorithm based on: https://codereview.stackexchange.com/a/86428
        """
        d = B - A
        f = A - center

        a = np.dot(d, d)
        b = 2.0 * np.dot(f, d)
        c = np.dot(f, f) - radius * radius

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # Prefer the farther intersection (t2) as it is "ahead" along the segment
        if 0.0 <= t2 <= 1.0:
            return A + t2 * d
        if 0.0 <= t1 <= 1.0:
            return A + t1 * d
        return None

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
