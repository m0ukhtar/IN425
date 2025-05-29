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
import math

class BiRRT(Node):
    def __init__(self, K=1000, dq=10):
        Node.__init__(self, "rrt_node")

        self.robot_pose = Pose2D()
        self.path = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map = None
        self.map_image = None
        self.goal = None
        self.K = K
        self.dq = dq

        self.create_subscription(PoseStamped, "/goal_pose", self.goalCb, 1)
        self.path_pub = self.create_publisher(Path, "/path", 1)

        self.getMap()
        self.create_map_image()

    def __del__(self):
        cv2.destroyAllWindows()

    def getMap(self):
        map_cli = self.create_client(GetMap, "map_server/map")
        while not map_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for map service to be available...")
        map_req = GetMap.Request()
        future = map_cli.call_async(map_req)

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                try:
                    self.map = future.result().map
                    self.get_logger().info("Map loaded !")
                except Exception as e:
                    self.get_logger().info(f"Service call failed {e}")
                return

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                "map",
                "base_footprint",
                rclpy.time.Time()
            )
            self.robot_pose.x = trans.transform.translation.x
            self.robot_pose.y = trans.transform.translation.y
        except TransformException as e:
            self.get_logger().info(f"Could not transform base_footprint to map: {e}")

    def create_map_image(self):
        if self.map is None:
            self.get_logger().error("Map not loaded! Cannot create map image.")
            return

        self.get_logger().info(f"Map dimensions: width={self.map.info.width}, height={self.map.info.height}")
        self.get_logger().info(f"Map resolution: {self.map.info.resolution}")
        self.get_logger().info(f"Map origin: x={self.map.info.origin.position.x}, y={self.map.info.origin.position.y}")

        try:
            map_data = np.array(self.map.data, dtype=np.int8).reshape(self.map.info.height, self.map.info.width)
            self.get_logger().info(f"Map data shape: {map_data.shape}, unique values: {np.unique(map_data)}")
        except Exception as e:
            self.get_logger().error(f"Failed to reshape map data: {e}")
            return

        self.map_image = np.zeros((self.map.info.height, self.map.info.width), dtype=np.uint8)
        self.map_image[map_data == 0] = 255
        self.map_image[map_data == 100] = 0
        self.map_image[map_data == -1] = 128

        unique, counts = np.unique(self.map_image, return_counts=True)
        value_counts = dict(zip(unique, counts))
        self.get_logger().info(f"Map image value counts: {value_counts}")

        try:
            cv2.imwrite("/tmp/map_image.png", self.map_image)
            self.get_logger().info("Map image saved to /tmp/map_image.png")
        except Exception as e:
            self.get_logger().error(f"Failed to save map image: {e}")

        self.map_image = np.flipud(self.map_image)
        cv2.imshow("Map", self.map_image)
        cv2.waitKey(100)

    def goalCb(self, msg):
        x_goal_map = msg.pose.position.x
        y_goal_map = msg.pose.position.y

        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        resolution = self.map.info.resolution
        height = self.map.info.height

        x_img = int((x_goal_map - origin_x) / resolution)
        y_img = height - int((y_goal_map - origin_y) / resolution)

        self.goal = (x_img, y_img)

        self.get_logger().info(f"[Goal Callback] Goal en map frame: ({x_goal_map:.2f}, {y_goal_map:.2f})")
        self.get_logger().info(f"[Goal Callback] Goal en image: {self.goal}")
        self.run()

    def run(self):
        if self.goal is None or self.map is None:
            return

        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            x_robot = trans.transform.translation.x
            y_robot = trans.transform.translation.y

            origin_x = self.map.info.origin.position.x
            origin_y = self.map.info.origin.position.y
            resolution = self.map.info.resolution
            height = self.map.info.height

            x_img = int((x_robot - origin_x) / resolution)
            y_img = height - int((y_robot - origin_y) / resolution)
            start = (x_img, y_img)

            self.get_logger().info(f"[run] Start en map: ({x_robot:.2f}, {y_robot:.2f})")
            self.get_logger().info(f"[run] Start en image: ({x_img}, {y_img})")

            if self.map_image[start[1], start[0]] != 255:
                self.get_logger().error("Start position is in an obstacle!")
                return
            if self.map_image[self.goal[1], self.goal[0]] != 255:
                self.get_logger().error("Goal position is in an obstacle!")
                return

            path = self.plan_birrt(start, self.goal)
            if path is None:
                self.get_logger().warn("Aucun chemin trouvé par BiRRT.")
                return

            self.path = path
            self.publishPath()

        except Exception as e:
            self.get_logger().error(f"Erreur dans run(): {str(e)}")

    def publishPath(self):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()

        resolution = self.map.info.resolution
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        height = self.map.info.height

        for x_img, y_img in self.path:
            x_map = x_img * resolution + origin_x
            y_map = (height - y_img) * resolution + origin_y

            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x_map
            pose.pose.position.y = y_map
            pose.pose.orientation.w = 1.0

            msg.poses.append(pose)

        self.path_pub.publish(msg)

    def plan_birrt(self, start, goal):
        def is_free(p):
            x, y = p
            return 0 <= x < self.map_image.shape[1] and 0 <= y < self.map_image.shape[0] and self.map_image[y, x] == 255

        def steer(p1, p2):
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                return p1
            scale = min(self.dq / dist, 1.0)
            return (int(p1[0] + dx * scale), int(p1[1] + dy * scale))

        def get_nearest(tree, point):
            return min(tree, key=lambda p: (p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2)

        def line_is_free(p1, p2):
            line = list(zip(np.linspace(p1[0], p2[0], num=10).astype(int), np.linspace(p1[1], p2[1], num=10).astype(int)))
            return all(is_free(p) for p in line)

        tree_a, tree_b = {start: None}, {goal: None}

        for _ in range(self.K):
            rand = (np.random.randint(0, self.map_image.shape[1]), np.random.randint(0, self.map_image.shape[0]))
            nearest_a = get_nearest(tree_a, rand)
            new_a = steer(nearest_a, rand)
            if not is_free(new_a) or not line_is_free(nearest_a, new_a):
                continue
            tree_a[new_a] = nearest_a

            nearest_b = get_nearest(tree_b, new_a)
            new_b = steer(nearest_b, new_a)
            if is_free(new_b) and line_is_free(nearest_b, new_b):
                tree_b[new_b] = nearest_b
                if line_is_free(new_a, new_b):
                    path = [new_a]
                    while new_a in tree_a:
                        new_a = tree_a[new_a]
                        if new_a: path.insert(0, new_a)
                    while new_b in tree_b:
                        new_b = tree_b[new_b]
                        if new_b: path.append(new_b)
                    return path
            tree_a, tree_b = tree_b, tree_a
        return None

def main():
    rclpy.init()
    node = BiRRT()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
