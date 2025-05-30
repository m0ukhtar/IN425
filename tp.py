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

        self.create_subscription(Path, "/path", self.plannerCb, 1)
        self.robot_path_pub = self.create_publisher(Path, "/robot_path", qos_profile=1)
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.create_timer(0.1, self.run)

        # Paramètres de contrôle améliorés
        self.max_linear_velocity = 0.3  # Réduction de la vitesse max
        self.max_angular_velocity = 1.0  # Réduction de la vitesse angulaire max
        self.goal_tolerance = 0.10  # Tolérance pour atteindre un point
        self.angular_tolerance = 0.15  # Tolérance angulaire réduite

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
        if not self.reached and self.goal_received:
            if not self.get_robot_pose():
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

            # Contrôle amélioré avec priorité à l'orientation
            if abs(alpha) > self.angular_tolerance:
                # Si l'angle est trop grand, on se contente de tourner
                self.linear = 0.0
                self.angular = self.constrain(1.5 * alpha, 
                                            min=-self.max_angular_velocity, 
                                            max=self.max_angular_velocity)
            else:
                # L'orientation est correcte, on peut avancer
                # Vitesse linéaire proportionnelle à la distance
                self.linear = self.constrain(min(self.max_linear_velocity, 0.8 * rho),
                                           min=0.0, max=self.max_linear_velocity)
                # Correction d'orientation fine
                self.angular = self.constrain(0.8 * alpha,
                                            min=-self.max_angular_velocity,
                                            max=self.max_angular_velocity)

            # Vérifier si le point actuel est atteint
            if rho < self.goal_tolerance:
                self.inc += 1
                self.get_logger().info(f"Waypoint {self.inc}/{len(self.path)} reached")

            self.send_velocities()
            self.publish_path()

    def send_velocities(self):
        # Contraintes de sécurité plus strictes
        self.linear = self.constrain(self.linear, min=-self.max_linear_velocity, 
                                   max=self.max_linear_velocity)
        self.angular = self.constrain(self.angular, min=-self.max_angular_velocity, 
                                    max=self.max_angular_velocity)

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
