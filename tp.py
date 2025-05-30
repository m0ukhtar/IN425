#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import Path
from tf2_ros import Buffer, TransformListener, TransformException
from tf_transformations import euler_from_quaternion


class Motion(Node):
    def __init__(self):
        super().__init__("motion_node")

        # TF buffer & listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # état du robot
        self.robot_pose = Pose2D()
        self.goal_received = False
        self.reached = False

        # souscription à la trajectoire planifiée
        self.create_subscription(Path, "/path", self.planner_cb, 1)

        # publication de la trajectoire réellement suivie
        self.robot_path_pub = self.create_publisher(Path, "/robot_path", 1)
        # publication des commandes de vitesse
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)

        # boucle de contrôle à 20 Hz
        self.create_timer(0.05, self.run)

    def get_robot_pose(self):
        """
        Met à jour self.robot_pose à partir de TF.
        Si l'appel échoue, on logue l'erreur mais on garde la dernière pose connue.
        """
        try:
            # ATTENTION : remplacer 'base_link' par la frame correcte de votre robot si besoin
            t = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time())
            self.robot_pose.x = t.transform.translation.x
            self.robot_pose.y = t.transform.translation.y
            q = (
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            )
            self.robot_pose.theta = euler_from_quaternion(q)[2]
        except TransformException as e:
            self.get_logger().warn(f"Could not get transform: {e}")

    def planner_cb(self, msg: Path):
        """
        Callback : on reçoit une Path planifiée.
        On saute le premier point (position actuelle) et on initialise l'index.
        """
        self.get_logger().info("New path received, starting execution.")
        self.path = msg.poses[1:]  # on ignore la pose de départ
        self.inc = 0
        self.goal_received = True
        self.reached = False

        # initialisation du Path « réel »
        self.real_path_msg = Path()
        self.real_path_msg.header.frame_id = "map"
        self.real_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.real_path = []

    def run(self):
        """
        Boucle exécutée à 20Hz : mise à jour de la pose, calcul de commande,
        publication des vitesses et traçage de la trajectoire suivie.
        """
        if not (self.goal_received and not self.reached):
            return

        # mise à jour de la pose (on ne quitte pas si ça échoue)
        self.get_robot_pose()

        # si on a atteint tous les waypoints, on s'arrête
        if self.inc >= len(self.path):
            self.linear = 0.0
            self.angular = 0.0
            self.send_velocities()
            self.reached = True
            self.get_logger().info("Goal reached.")
            return

        # coordonnées du waypoint courant
        target = self.path[self.inc].pose.position
        dx = target.x - self.robot_pose.x
        dy = target.y - self.robot_pose.y

        # distance et angle relatif
        rho = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        alpha = self.normalize_angle(angle_to_goal - self.robot_pose.theta)

        # gains
        k_rho = 2.0
        k_alpha = 1.5

        # si l'angle est grand → rotation sur place
        if abs(alpha) > 0.4:
            self.linear = 0.0
            self.angular = k_alpha * alpha
        else:
            # avance proportionnellement à la distance
            self.linear = min(1.2, k_rho * rho)
            self.angular = k_alpha * alpha

        # on ne passe au point suivant que si proche ET bien orienté
        if rho < 0.25 and abs(alpha) < 0.3:
            self.inc += 1

        # publication
        self.send_velocities()
        self.publish_path()

    def send_velocities(self):
        """Construit et publie un Twist à partir de self.linear, self.angular."""
        # bornes de sécurité
        lin = max(-1.2, min(self.linear, 1.2))
        ang = max(-2.5, min(self.angular, 2.5))

        cmd = Twist()
        cmd.linear.x = lin
        cmd.angular.z = ang
        self.vel_pub.publish(cmd)

    def publish_path(self):
        """Ajoute la position courante à la Path réelle et la publie."""
        ps = Pose2D()
        ps.x = self.robot_pose.x
        ps.y = self.robot_pose.y

        pose_stamped = Path().poses.__class__()  # PoseStamped
        pose_stamped.pose.position.x = ps.x
        pose_stamped.pose.position.y = ps.y

        self.real_path.append(pose_stamped)
        self.real_path_msg.poses = self.real_path
        self.robot_path_pub.publish(self.real_path_msg)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Ramène l’angle dans [-π, +π]."""
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


if __name__ == "__main__":
    main()
