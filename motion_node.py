import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D, Twist, PoseStamped
from nav_msgs.msg import Path
from tf2_ros import Buffer, TransformListener
from tf2_ros.transform_exception import TransformException
from tf_transformations import euler_from_quaternion
import math

class MotionNode(Node):
    def __init__(self):
        super().__init__('motion_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.robot_pose = Pose2D()
        self.goal_received = False
        self.reached = False

        self.path = []
        self.index = 0

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_sub = self.create_subscription(Path, '/path', self.plannerCb, 10)
        self.timer = self.create_timer(0.1, self.run)

    def plannerCb(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.goal_received = True
        self.reached = False
        self.index = 0
        self.get_logger().info(f"Received path with {len(self.path)} points")

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            return x, y, yaw
        except TransformException as e:
            self.get_logger().warn(f"Transform error: {e}")
            return None, None, None

    def run(self):
        if not self.goal_received or self.reached:
            return

        x, y, theta = self.get_robot_pose()
        if x is None:
            return

        if self.index >= len(self.path):
            self.get_logger().info("Goal reached!")
            self.cmd_pub.publish(Twist())
            self.reached = True
            return

        goal_x, goal_y = self.path[self.index]
        dx = goal_x - x
        dy = goal_y - y

        rho = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        alpha = self.normalize_angle(angle_to_goal - theta)

        if rho < 0.2:
            self.index += 1
            return

        cmd = Twist()
        cmd.linear.x = self.clamp(0.6 * rho, 0.0, 0.8)
        cmd.angular.z = self.clamp(2.0 * alpha, -2.0, 2.0)
        self.cmd_pub.publish(cmd)

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(value, max_value))


def main():
    rclpy.init()
    node = MotionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
