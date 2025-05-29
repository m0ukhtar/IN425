import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D, Twist, PoseStamped
from nav_msgs.msg import Path

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import euler_from_quaternion

import math

class Motion(Node):
    def __init__(self):
        Node.__init__(self, "motion_node")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.robot_pose = Pose2D()
        self.goal_received, self.reached = False, False

        self.create_subscription(Path, "/path", self.plannerCb, 1)
        self.robot_path_pub = self.create_publisher(Path, "/robot_path", qos_profile=1)
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.create_timer(0.1, self.run)

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                "map",
                "base_footprint",
                rclpy.time.Time()
            )
            self.robot_pose.x = trans.transform.translation.x
            self.robot_pose.y = trans.transform.translation.y
            quat = (trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w)
            self.robot_pose.theta = euler_from_quaternion(quat)[2]
        except TransformException as e:
            self.get_logger().info(f"Could not transform base_footprint to map: {e}")

    def plannerCb(self, msg):
        self.reached, self.goal_received = False, True
        self.path = msg.poses[1:]
        self.inc = 0

        self.real_path_msg = Path()
        self.real_path_msg.header.frame_id = "map"
        self.real_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.real_path = []

    def run(self):
        if not self.reached and self.goal_received:
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
            rho = math.sqrt(dx**2 + dy**2)

            angle_to_goal = math.atan2(dy, dx)
            alpha = self.normalize_angle(angle_to_goal - self.robot_pose.theta)

            if rho < 0.2:
                self.inc += 1
                return

            self.linear = 0.8 * rho
            self.angular = 2.5 * alpha

            self.send_velocities()
            self.publish_path()

    def send_velocities(self):
        self.linear = self.constrain(self.linear, min=-2.0, max=2.0)
        self.angular = self.constrain(self.angular, min=-3.0, max=3.0)

        cmd_vel = Twist()
        cmd_vel.linear.x = self.linear
        cmd_vel.angular.z = self.angular
        self.vel_pub.publish(cmd_vel)

    def constrain(self, val, min=-2.0, max=2.0):
        if val < min:
            return min
        elif val > max:
            return max
        return val

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def publish_path(self):
        pose = PoseStamped()
        pose.pose.position.x = self.robot_pose.x
        pose.pose.position.y = self.robot_pose.y
        self.real_path.append(pose)
        self.real_path_msg.poses = self.real_path
        self.robot_path_pub.publish(self.real_path_msg)

def main():
    rclpy.init()
    node = Motion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
