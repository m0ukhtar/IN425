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
        Node.__init__(self, "motion_node")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.robot_pose = Pose2D()
        self.goal_received, self.reached = False, False
        self.path = []
        self.inc = 0
        self.linear = 0.0
        self.angular = 0.0

        self.create_subscription(Path, "/path", self.plannerCb, 1)
        self.robot_path_pub = self.create_publisher(Path, "/robot_path", qos_profile=1)
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.create_timer(0.1, self.run)

        # Paramètres de contrôle améliorés
        self.max_linear_velocity = 0.5  # Vitesse plus élevée pour tester
        self.max_angular_velocity = 1.5  # Vitesse angulaire plus élevée
        self.goal_tolerance = 0.15  # Tolérance plus large
        self.angular_tolerance = 0.3  # Tolérance angulaire plus large
        
        self.get_logger().info("Motion node initialized")

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                "map",
                "base_footprint",
                rclpy.time.Time()
            )
            self.robot_pose.x = trans.transform.translation.x
            self.robot_pose.y = trans.transform.translation.y
            quat = (trans.transform.rotation.x, trans.transform.rotation.y, 
                   trans.transform.rotation.z, trans.transform.rotation.w)
            self.robot_pose.theta = euler_from_quaternion(quat)[2]
            return True
        except TransformException as e:
            self.get_logger().warn(f"Could not transform base_footprint to map: {e}")
            return False

    def plannerCb(self, msg):
        if len(msg.poses) == 0:
            self.get_logger().warn("Received empty path!")
            return
            
        self.reached, self.goal_received = False, True
        # Ne pas ignorer le premier point du chemin
        self.path = msg.poses
        self.inc = 0

        self.real_path_msg = Path()
        self.real_path_msg.header.frame_id = "map"
        self.real_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.real_path = []
        
        self.get_logger().info(f"New path received with {len(self.path)} points")

    def run(self):
        # Toujours publier des vitesses nulles si pas de goal
        if not self.goal_received:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.vel_pub.publish(cmd_vel)
            return
            
        if self.reached:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.vel_pub.publish(cmd_vel)
            return
            
        if not self.get_robot_pose():
            self.get_logger().warn("Cannot get robot pose")
            return
            
        if self.inc >= len(self.path):
            self.reached = True
            self.linear = 0.0
            self.angular = 0.0
            self.send_velocities()
            self.get_logger().info("Goal reached!")
            return

        current_goal = self.path[self.inc].pose.position

        dx = current_goal.x - self.robot_pose.x
        dy = current_goal.y - self.robot_pose.y

        rho = math.sqrt(dx ** 2 + dy ** 2)
        angle_to_goal = math.atan2(dy, dx)
        alpha = self.normalize_angle(angle_to_goal - self.robot_pose.theta)

        # Log pour debug
        if self.inc == 0 or self.inc % 10 == 0:  # Log occasionnel
            self.get_logger().info(f"Waypoint {self.inc}/{len(self.path)}: rho={rho:.3f}, alpha={alpha:.3f}")

        # Contrôle simplifié pour assurer le mouvement
        if rho < self.goal_tolerance:
            # Point atteint, passer au suivant
            self.inc += 1
            self.get_logger().info(f"Waypoint {self.inc-1} reached, moving to next")
            return

        if abs(alpha) > self.angular_tolerance:
            # Grande différence d'angle : tourner sur place
            self.linear = 0.0
            self.angular = 1.0 if alpha > 0 else -1.0
        else:
            # Angle OK : avancer avec correction légère
            self.linear = min(self.max_linear_velocity, max(0.1, 0.5 * rho))
            self.angular = 0.8 * alpha

        self.send_velocities()
        self.publish_path()

    def send_velocities(self):
        # Contraintes de sécurité
        self.linear = self.constrain(self.linear, min=-self.max_linear_velocity, 
                                   max=self.max_linear_velocity)
        self.angular = self.constrain(self.angular, min=-self.max_angular_velocity, 
                                    max=self.max_angular_velocity)

        cmd_vel = Twist()
        cmd_vel.linear.x = float(self.linear)
        cmd_vel.angular.z = float(self.angular)
        
        # Log pour debug
        if abs(self.linear) > 0.01 or abs(self.angular) > 0.01:
            self.get_logger().info(f"Sending velocities: linear={self.linear:.3f}, angular={self.angular:.3f}")
        
        self.vel_pub.publish(cmd_vel)

    def constrain(self, val, min=-2.0, max=2.0):
        if val < min:
            return min
        elif val > max:
            return max
        return val

    def publish_path(self):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.robot_pose.x
        pose.pose.position.y = self.robot_pose.y
        pose.pose.orientation.w = 1.0  # Orientation par défaut
        
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
