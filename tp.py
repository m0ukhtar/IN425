#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose2D, Twist, PoseStamped
from nav_msgs.msg import Path
from tf2_ros import Buffer, TransformListener, TransformException
from tf_transformations import euler_from_quaternion


class Motion(Node):
    def __init__(self):
        super().__init__("motion_node")

        # TF buffer & listener
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # état du robot
        self.robot_pose   = Pose2D()
        self.goal_received = False
        self.reached       = False

        # souscription à la trajectoire planifiée
        self.create_subscription(Path, "/path", self.planner_cb, 1)

        # publication de la trajectoire réellement suivie
        self.robot_path_pub = self.create_publisher(Path, "/robot_path", 1)
        # publication des commandes de vitesse
        self.vel_pub        = self.create_publisher(Twist, "/cmd_vel", 1)

        # boucle de contrôle à 20 Hz
        self.create_timer(0.05, self.run)

    def get_robot_pose(self):
        """
        Met à jour self.robot_pose à partir de TF.
        Si l'appel échoue, on logue l'erreur mais on garde la dernière pose connue.
        """
        try:
            # Remplacez 'base_link' par la frame publiée par votre robot si besoin
            t = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time())
            self.robot_pose.x     = t.transform.translation.x
            self.robot_pose.y     = t.transform.translation.y
            q = (
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            )
            self.robot_pose.theta = euler_from_quaternion(q)[2]
        except TransformException as e:
            self.get_logger().warn(f"TF failed, keeping last pose: {e}")

    def planner_cb(self, msg: Path):
        """
        Callback : on reçoit une Path planifiée.
        On saute le premier point (position actuelle) et on initialise l'index.
        """
        self.get_logger().info("New path received.")
        # on ignore la pose de départ (robot), on garde tous les autres waypoints
        self.path = msg.poses[1:]
        self.inc  = 0
        self.goal_received = True
        self.reached       = False

        # initialisation du Path « réel »
        self.real_path_msg = Path()
        self.real_path_msg.header.frame_id = "map"
        self.real_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.real_path = []

    def run(self):
        """
        Exécuté à 20 Hz : mise à jour de la pose, calcul de commande,
        publication des vitesses et du tracé réellement suivi.
        """
        if not (self.goal_received and not self.reached):
            return

        # mise à jour de la pose (on continue même en cas d'erreur TF)
        self.get_robot_pose()

        # si tous les waypoints sont faits, on s'arrête
        if self.inc >= len(self.path):
            self._stop_robot()
            self.reached = True
            self.get_logger().info("Goal reached.")
            return

        # vecteur vers le waypoint courant
        target = self.path[self.inc].pose.position
        dx = target.x - self.robot_pose.x
        dy = target.y - self.robot_pose.y

        rho           = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        alpha         = self._normalize(angle_to_goal - self.robot_pose.theta)

        # gains
        k_rho   = 1.5
        k_alpha = 2.0

        # choix du comportement :
        #   - si écart angulaire > 1 rad → rotation sur place
        #   - sinon → avance ET tourne
        if abs(alpha) > 1.0:
            linear  = 0.0
            angular = k_alpha * alpha
        else:
            linear  = k_rho * rho
            angular = k_alpha * alpha

        # limites de sécurité
        self.linear  = max(-1.0, min(linear,  1.0))
        self.angular = max(-2.0, min(angular, 2.0))

        # si proche ET bien orienté → on passe au point suivant
        if rho < 0.2 and abs(alpha) < 0.5:
            self.inc += 1

        self._publish_cmd()
        self._publish_real_path()

    def _stop_robot(self):
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = 0.0
        self.vel_pub.publish(cmd)

    def _publish_cmd(self):
        cmd = Twist()
        cmd.linear.x  = self.linear
        cmd.angular.z = self.angular
        self.vel_pub.publish(cmd)

    def _publish_real_path(self):
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.pose.position.x = self.robot_pose.x
        ps.pose.position.y = self.robot_pose.y
        self.real_path.append(ps)
        self.real_path_msg.poses = self.real_path
        self.robot_path_pub.publish(self.real_path_msg)

    @staticmethod
    def _normalize(angle: float) -> float:
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
