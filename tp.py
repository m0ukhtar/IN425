import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import TransformListener, Buffer
import tf_transformations
import numpy as np
import cv2
import os
from sensor_msgs.msg import LaserScan

class BiRRT:
    def __init__(self, image, logger, max_iter=5000, step_size=10):
        self.image = image
        self.height, self.width = image.shape
        self.logger = logger
        self.max_iter = max_iter
        self.step_size = step_size

    def is_free(self, point):
        x, y = point
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.image[y, x] == 255
        return False

    def line_is_free(self, p1, p2):
        x0, y0 = p1
        x1, y1 = p2
        points = list(zip(*cv2.line(np.zeros_like(self.image), p1, p2)))[0]
        for x, y in points:
            if not self.is_free((x, y)):
                return False
        return True

    def get_random_point(self):
        return (np.random.randint(0, self.width), np.random.randint(0, self.height))

    def get_nearest(self, tree, point):
        return min(tree, key=lambda p: np.linalg.norm(np.array(p) - np.array(point)))

    def steer(self, p1, p2):
        direction = np.array(p2) - np.array(p1)
        length = np.linalg.norm(direction)
        if length == 0:
            return p1
        direction = (direction / length) * min(self.step_size, length)
        new_point = np.array(p1) + direction
        return tuple(map(int, new_point))

    def build_path(self, tree1, tree2, connect_node1, connect_node2):
        path = [connect_node1]
        node = connect_node1
        while node in tree1:
            node = tree1[node]
            if node is None:
                break
            path.insert(0, node)
        node = connect_node2
        while node in tree2:
            node = tree2[node]
            if node is None:
                break
            path.append(node)
        return path

    def run(self, start, goal):
        if not self.is_free(start) or not self.is_free(goal):
            return None

        tree_start = {start: None}
        tree_goal = {goal: None}

        for i in range(self.max_iter):
            rand = self.get_random_point()
            nearest_start = self.get_nearest(tree_start, rand)
            new_start = self.steer(nearest_start, rand)

            if self.is_free(new_start) and self.line_is_free(nearest_start, new_start):
                tree_start[new_start] = nearest_start

                nearest_goal = self.get_nearest(tree_goal, new_start)
                new_goal = self.steer(nearest_goal, new_start)

                if self.is_free(new_goal) and self.line_is_free(nearest_goal, new_goal):
                    tree_goal[new_goal] = nearest_goal
                    if self.line_is_free(new_start, new_goal):
                        return self.build_path(tree_start, tree_goal, new_start, new_goal)

            tree_start, tree_goal = tree_goal, tree_start  # Switch trees

        return None

    def reduce_path(self, path):
        if not path:
            return path
        reduced = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i:
                if self.line_is_free(path[i], path[j]):
                    reduced.append(path[j])
                    i = j
                    break
                j -= 1
        return reduced

class RRTNode(Node):
    def __init__(self):
        super().__init__('rrt_node')

        self.map = None
        self.image = None
        self.goal = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.path_pub = self.create_publisher(Path, '/plan', 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.mapCb, 10)
        self.timer = self.create_timer(1.0, self.run)

    def mapCb(self, msg):
        self.map = msg
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = msg.info.origin.position

        self.get_logger().info(f"Map loaded !")
        self.get_logger().info(f"Map dimensions: width={width}, height={height}")
        self.get_logger().info(f"Map resolution: {resolution}")
        self.get_logger().info(f"Map origin: x={origin.x}, y={origin.y}")

        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        self.image = np.zeros_like(data, dtype=np.uint8)
        self.image[data == 0] = 255
        self.image[data == 100] = 0
        self.image[data == -1] = 128

        self.get_logger().info(f"Map data shape: {data.shape}, unique values: {np.unique(data)}")
        self.get_logger().info(f"Map image shape: {self.image.shape}, unique values: {np.unique(self.image)}")
        self.get_logger().info(f"Map image value counts: {dict(zip(*np.unique(self.image, return_counts=True)))}")

        cv2.imwrite('/tmp/map_image.png', self.image)
        self.get_logger().info("Map image saved to /tmp/map_image.png")

        self.birrt = BiRRT(self.image, self.get_logger())

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

    def run(self):
        if self.goal is None or self.map is None:
            return

        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform('map', 'base_footprint', now)

            x_robot = transform.transform.translation.x
            y_robot = transform.transform.translation.y

            origin_x = self.map.info.origin.position.x
            origin_y = self.map.info.origin.position.y
            resolution = self.map.info.resolution
            height = self.map.info.height

            x_img = int((x_robot - origin_x) / resolution)
            y_img = height - int((y_robot - origin_y) / resolution)
            start = (x_img, y_img)

            self.get_logger().info(f"[run] Start en map: ({x_robot:.2f}, {y_robot:.2f})")
            self.get_logger().info(f"[run] Start en image: ({x_img}, {y_img})")

            if self.image[start[1], start[0]] != 255:
                self.get_logger().error("Start position is in an obstacle!")
                return
            if self.image[self.goal[1], self.goal[0]] != 255:
                self.get_logger().error("Goal position is in an obstacle!")
                return

            path = self.birrt.run(start=start, goal=self.goal)
            if path is None:
                self.get_logger().warn("Aucun chemin trouvé par BiRRT.")
                return

            path = self.birrt.reduce_path(path)
            self.publishPath(path)

        except Exception as e:
            self.get_logger().error(f"Erreur dans run(): {str(e)}")

    def publishPath(self, path):
        if not path:
            return

        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        resolution = self.map.info.resolution
        height = self.map.info.height

        for (x_img, y_img) in path:
            x_map = x_img * resolution + origin_x
            y_map = (height - y_img) * resolution + origin_y
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = x_map
            pose.pose.position.y = y_map
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RRTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
