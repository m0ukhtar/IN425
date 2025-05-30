import rclpy
from rclpy.node import Node

from nav_msgs.srv import GetMap
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose2D

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import cv2
import numpy as np
import random
import math

class BiRRT(Node):
    def __init__(self, K=5000, dq=8):
        super().__init__("rrt_node")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.K = K
        self.dq = dq
        self.map = None
        self.map_image = None
        self.goal = None
        self.path = []
        self.robot_pose = Pose2D()
        
        # Rayon de sécurité augmenté pour éviter les murs (en pixels)
        self.safety_radius = 10  
        
        # Rayon de sécurité pour les segments (vérification plus dense)
        self.segment_check_step = 2  

        self.create_subscription(PoseStamped, "/goal_pose", self.goalCb, 1)
        self.path_pub = self.create_publisher(Path, "/path", 1)

        self.getMap()
        self.create_map_image()

    def __del__(self):
        cv2.destroyAllWindows()

    def getMap(self):
        map_cli = self.create_client(GetMap, "map_server/map")
        while not map_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for map service...")
        req = GetMap.Request()
        future = map_cli.call_async(req)
        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                self.map = future.result().map
                self.get_logger().info("Map loaded !")
                return

    def create_map_image(self):
        height, width = self.map.info.height, self.map.info.width
        data = np.array(self.map.data, dtype=np.int8).reshape(height, width)
        self.map_image = np.zeros((height, width), dtype=np.uint8)
        
        # Traitement de la carte : 0 = libre, 100 = occupé, -1 = inconnu
        self.map_image[data == 0] = 255    # Espace libre = blanc
        self.map_image[data == 100] = 0    # Obstacle = noir
        self.map_image[data == -1] = 0     # Inconnu traité comme obstacle
        
        # Flip pour correspondre au système de coordonnées ROS
        self.map_image = np.flipud(self.map_image)
        
        # Dilatation pour augmenter la taille des obstacles (sécurité)
        kernel = np.ones((5, 5), np.uint8)
        self.map_image = cv2.erode(self.map_image, kernel, iterations=1)
        
        cv2.imwrite("/tmp/map_image.png", self.map_image)
        self.get_logger().info(f"Map image created: {width}x{height}")

    def is_valid_point(self, point):
        x, y = int(point[0]), int(point[1])
        h, w = self.map_image.shape
        
        # Vérifier les limites de base
        if x < self.safety_radius or x >= w - self.safety_radius or \
           y < self.safety_radius or y >= h - self.safety_radius:
            return False
        
        # Vérifier la zone de sécurité autour du point
        for dx in range(-self.safety_radius, self.safety_radius + 1):
            for dy in range(-self.safety_radius, self.safety_radius + 1):
                # Vérification circulaire plutôt que carrée
                if dx*dx + dy*dy <= self.safety_radius*self.safety_radius:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if self.map_image[ny][nx] != 255:  # Pas libre
                            return False
        return True

    def is_valid_segment(self, p1, p2):
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
        
        # Distance entre les points
        dist = max(abs(x2 - x1), abs(y2 - y1))
        
        if dist == 0:
            return self.is_valid_point(p1)
        
        # Vérification plus dense le long du segment
        num_checks = max(dist // self.segment_check_step, 5)
        
        for i in range(num_checks + 1):
            t = i / num_checks
            x = int(x1 * (1 - t) + x2 * t)
            y = int(y1 * (1 - t) + y2 * t)
            if not self.is_valid_point((x, y)):
                return False
        return True

    def goalCb(self, msg):
        try:
            res = self.map.info.resolution
            origin = self.map.info.origin.position
            height = self.map.info.height

            # Conversion goal
            x_goal = int((msg.pose.position.x - origin.x) / res)
            y_goal = height - int((msg.pose.position.y - origin.y) / res)

            # Obtenir la position du robot
            trans = self.tf_buffer.lookup_transform("map", "base_footprint", rclpy.time.Time())
            x_start = int((trans.transform.translation.x - origin.x) / res)
            y_start = height - int((trans.transform.translation.y - origin.y) / res)

            start = (x_start, y_start)
            goal = (x_goal, y_goal)

            self.get_logger().info(f"Planning path from {start} to {goal}")

            # Vérifier la validité des points de départ et d'arrivée
            if not self.is_valid_point(start):
                self.get_logger().error("Start point is in obstacle or too close to wall!")
                return
            if not self.is_valid_point(goal):
                self.get_logger().error("Goal point is in obstacle or too close to wall!")
                return

            # Planification du chemin
            path = self.birrt(start, goal)
            if path:
                path = self.reduce_path(path)
                path = self.smooth_path(path)
                self.path = path
                self.publishPath()
                self.get_logger().info(f"Path found with {len(path)} waypoints")
            else:
                self.get_logger().error("No valid path found!")
                
        except TransformException as e:
            self.get_logger().error(f"Transform error: {e}")
        except Exception as e:
            self.get_logger().error(f"Error in goalCb: {e}")

    def birrt(self, start, goal):
        tree_a = {start: None}
        tree_b = {goal: None}
        
        for iteration in range(self.K):
            # 30% de chance de viser directement le goal
            if random.random() < 0.3:
                rand = goal
            else:
                rand = (
                    random.randint(self.safety_radius, self.map_image.shape[1] - self.safety_radius - 1),
                    random.randint(self.safety_radius, self.map_image.shape[0] - self.safety_radius - 1)
                )

            # Étendre l'arbre A
            nearest = min(tree_a.keys(), key=lambda p: (p[0] - rand[0]) ** 2 + (p[1] - rand[1]) ** 2)
            direction = np.array(rand, dtype=float) - np.array(nearest, dtype=float)
            length = np.linalg.norm(direction)
            
            if length < 1e-6:  # Éviter la division par zéro
                continue
                
            # Normaliser et appliquer la distance step
            direction = (direction / length * self.dq)
            new_point = (int(nearest[0] + direction[0]), int(nearest[1] + direction[1]))

            if not self.is_valid_point(new_point) or not self.is_valid_segment(nearest, new_point):
                continue

            tree_a[new_point] = nearest

            # Essayer de connecter avec l'arbre B
            nearest_b = min(tree_b.keys(), key=lambda p: (p[0] - new_point[0]) ** 2 + (p[1] - new_point[1]) ** 2)
            connection_distance = math.sqrt((new_point[0] - nearest_b[0])**2 + (new_point[1] - nearest_b[1])**2)
            
            # Si les arbres sont assez proches et que le segment est valide
            if connection_distance <= self.dq * 2 and self.is_valid_segment(new_point, nearest_b):
                path_a = self.build_path(tree_a, new_point)
                path_b = self.build_path(tree_b, nearest_b)
                final_path = path_a + path_b[::-1]
                self.get_logger().info(f"Path found after {iteration} iterations")
                return final_path

            # Échanger les arbres pour la bidirectionnalité
            tree_a, tree_b = tree_b, tree_a

        self.get_logger().warn("No path found after maximum iterations!")
        return None

    def build_path(self, tree, node):
        path = [node]
        current = node
        while tree[current] is not None:
            current = tree[current]
            path.append(current)
        return path[::-1]

    def reduce_path(self, path):
        if len(path) <= 2:
            return path
        
        reduced = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # Essayer de connecter directement au point le plus loin possible
            j = len(path) - 1
            while j > i + 1:
                if self.is_valid_segment(path[i], path[j]):
                    break
                j -= 1
            reduced.append(path[j])
            i = j
            
        return reduced

    def smooth_path(self, path):
        if len(path) < 3:
            return path
            
        smoothed = [path[0]]
        
        for i in range(1, len(path) - 1):
            p0, p1, p2 = path[i-1], path[i], path[i+1]
            
            # Lissage avec vérification de validité
            candidates = [
                # Point original
                p1,
                # Moyenne pondérée
                (int((p0[0] + 4*p1[0] + p2[0]) / 6), int((p0[1] + 4*p1[1] + p2[1]) / 6)),
                # Moyenne simple
                (int((p0[0] + p1[0] + p2[0]) / 3), int((p0[1] + p1[1] + p2[1]) / 3))
            ]
            
            # Choisir le premier point valide
            for candidate in candidates:
                if self.is_valid_point(candidate):
                    smoothed.append(candidate)
                    break
            else:
                smoothed.append(p1)  # Fallback au point original
                
        smoothed.append(path[-1])
        return smoothed

    def publishPath(self):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        res = self.map.info.resolution
        origin = self.map.info.origin.position
        height = self.map.info.height

        for x, y in self.path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()
            
            # Conversion des coordonnées pixel vers monde
            pose.pose.position.x = x * res + origin.x
            pose.pose.position.y = (height - y) * res + origin.y
            pose.pose.position.z = 0.0
            
            # Orientation par défaut
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            
            msg.poses.append(pose)
            
        self.path_pub.publish(msg)
        self.get_logger().info(f"Published path with {len(msg.poses)} poses")

def main():
    rclpy.init()
    node = BiRRT()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
