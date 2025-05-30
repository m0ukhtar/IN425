import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D, Twist, PoseStamped
from nav_msgs.msg import Path
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException
from tf_transformations import euler_from_quaternion
import math

class Motion(Node):
    def __init__(self):
        super().__init__("motion_node")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.robot_pose = Pose2D()
        self.goal_received = False
        self.reached = False

        self.create_subscription(Path, "/path", self.plannerCb, 1)
        self.robot_path_pub = self.create_publisher(Path, "/robot_path", qos_profile=1)
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.create_timer(0.1, self.run)

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform("map", "base_footprint", rclpy.time.Time())
            self.robot_pose.x = trans.transform.translation.x
            self.robot_pose.y = trans.transform.translation.y
            quat = (
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            )
            self.robot_pose.theta = euler_from_quaternion(quat)[2]
        except TransformException as e:
            self.get_logger().info(f"Could not transform base_footprint to map: {e}")

    def plannerCb(self, msg):
        self.reached = False
        self.goal_received = True
        self.path = msg.poses[1:]  # skip robot pose
        self.inc = 0

        self.real_path_msg = Path()
        self.real_path_msg.header.frame_id = "map"
        self.real_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.real_path = []

    def run(self):
        if self.goal_received and not self.reached:
            self.get_robot_pose()

            if self.inc >= len(self.path):
                self.reached = True
                self.linear = 0.0
                self.angular = 0.0
                self.send_velocities()
                return

            current_goal = self.path[self.inc].pose.position
            dx = current_goal.x - self.robot_pose.x
            dy = current_goal.y - self.robot_pose.y

            rho = math.hypot(dx, dy)
            angle_to_goal = math.atan2(dy, dx)
            alpha = self.normalize_angle(angle_to_goal - self.robot_pose.theta)

            k_rho = 1.5
            k_alpha = 2.0

            if abs(alpha) > 0.4:
                self.linear = 0.0
                self.angular = 2.5 * alpha
            else:
                self.linear = min(1.2, k_rho * rho)
                self.angular = k_alpha * alpha

            if rho < 0.2:
                self.inc += 1

            self.send_velocities()
            self.publish_path()

    def send_velocities(self):
        self.linear = self.constrain(self.linear, -2.0, 2.0)
        self.angular = self.constrain(self.angular, -3.0, 3.0)

        cmd_vel = Twist()
        cmd_vel.linear.x = self.linear
        cmd_vel.angular.z = self.angular
        self.vel_pub.publish(cmd_vel)

    def constrain(self, val, min_val, max_val):
        return max(min_val, min(val, max_val))

    def publish_path(self):
        pose = PoseStamped()
        pose.pose.position.x = self.robot_pose.x
        pose.pose.position.y = self.robot_pose.y
        self.real_path.append(pose)
        self.real_path_msg.poses = self.real_path
        self.robot_path_pub.publish(self.real_path_msg)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

def main():
    rclpy.init()
    node = Motion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
