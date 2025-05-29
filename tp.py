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
    def __init__(self, K=3000, dq=5):
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
        self.safety_radius = 6

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
        self.map_image[data == 0] = 255
        self.map_image[data == 100] = 0
        self.map_image[data == -1] = 128
        self.map_image = np.flipud(self.map_image)
        cv2.imwrite("/tmp/map_image.png", self.map_image)

    def is_valid_point(self, point):
        x, y = point
        h, w = self.map_image.shape
        for dx in range(-self.safety_radius, self.safety_radius + 1):
            for dy in range(-self.safety_radius, self.safety_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if self.map_image[ny][nx] != 255:
                        return False
        return True

    def is_valid_segment(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dist = int(math.hypot(x2 - x1, y2 - y1))
        for i in range(dist):
            t = i / dist
            x = int(x1 * (1 - t) + x2 * t)
            y = int(y1 * (1 - t) + y2 * t)
            if not self.is_valid_point((x, y)):
                return False
        return True

    def goalCb(self, msg):
        res = self.map.info.resolution
        origin = self.map.info.origin.position
        height = self.map.info.height

        x_goal = int((msg.pose.position.x - origin.x) / res)
        y_goal = height - int((msg.pose.position.y - origin.y) / res)

        trans = self.tf_buffer.lookup_transform("map", "base_footprint", rclpy.time.Time())
        x_start = int((trans.transform.translation.x - origin.x) / res)
        y_start = height - int((trans.transform.translation.y - origin.y) / res)

        start = (x_start, y_start)
        goal = (x_goal, y_goal)

        if not self.is_valid_point(start) or not self.is_valid_point(goal):
            self.get_logger().error("Start or goal in obstacle!")
            return

        path = self.birrt(start, goal)
        if path:
            path = self.reduce_path(path)
            path = self.smooth_path(path)
            self.path = path
            self.publishPath()

    def birrt(self, start, goal):
        tree_a = {start: None}
        tree_b = {goal: None}
        for _ in range(self.K):
            rand = goal if random.random() < 0.2 else (
                random.randint(0, self.map_image.shape[1] - 1),
                random.randint(0, self.map_image.shape[0] - 1))

            nearest = min(tree_a.keys(), key=lambda p: (p[0] - rand[0]) ** 2 + (p[1] - rand[1]) ** 2)
            direction = np.array(rand) - np.array(nearest)
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            direction = (direction / length * self.dq).astype(int)
            new_point = (nearest[0] + direction[0], nearest[1] + direction[1])

            if not self.is_valid_point(new_point) or not self.is_valid_segment(nearest, new_point):
                continue

            tree_a[new_point] = nearest

            nearest_b = min(tree_b.keys(), key=lambda p: (p[0] - new_point[0]) ** 2 + (p[1] - new_point[1]) ** 2)
            if self.is_valid_segment(new_point, nearest_b):
                path_a = self.build_path(tree_a, new_point)
                path_b = self.build_path(tree_b, nearest_b)
                return path_a + path_b[::-1]

            tree_a, tree_b = tree_b, tree_a

        self.get_logger().warn("No path found!")
        return None

    def build_path(self, tree, node):
        path = [node]
        while tree[node] is not None:
            node = tree[node]
            path.append(node)
        return path[::-1]

    def reduce_path(self, path):
        if len(path) <= 2:
            return path
        reduced = [path[0]]
        i = 0
        while i < len(path) - 1:
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
            cx = int((p0[0] + 4*p1[0] + p2[0]) / 6)
            cy = int((p0[1] + 4*p1[1] + p2[1]) / 6)
            smoothed.append((cx, cy))
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
            pose.pose.position.x = x * res + origin.x
            pose.pose.position.y = (height - y) * res + origin.y
            msg.poses.append(pose)
        self.path_pub.publish(msg)

def main():
    rclpy.init()
    node = BiRRT()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
