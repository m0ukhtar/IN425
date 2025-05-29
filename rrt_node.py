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

class BiRRT(Node):
    def __init__(self, K=0, dq=0):
        Node.__init__(self, "rrt_node")

        """ Attributes """
        self.robot_pose = Pose2D()  # Current pose of the robot: self.robot.x, self.robot.y, robot.theta(last one is useless)
        self.path = []  # Path containing the waypoints computed by the Bi-RRT in the image reference frame
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)  # used to get the position of the robot
        # TODO: add your attributes here....
        self.map = None  # OccupancyGrid
        self.map_image = None  # Image representation of the map
        self.goal = None  # Goal position in image frame
        self.K = 1000  # Max iterations for BiRRT
        self.dq = 10  # Extension distance in pixels
        
        """ Publisher and Subscriber """
        self.create_subscription(PoseStamped, "/goal_pose", self.goalCb, 1)
        self.path_pub = self.create_publisher(Path, "/path", 1)

        """ Load the map and create the related image """
        self.getMap()
        # TODO: create the related image
        self.create_map_image()

    def __del__(self):
        """ Called when the object is destroyed """
        cv2.destroyAllWindows() # destroy all the OpenCV windows you displayed

    # **********************************    
    def getMap(self):
        """ Method for getting the map """
        # DO NOT TOUCH
        map_cli = self.create_client(GetMap, "map_server/map")
        while not map_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for map service to be available...")
        map_req = GetMap.Request()
        future = map_cli.call_async(map_req)

        while rclpy.ok():
            rclpy.spin_once(self)
            if future.done():
                try:
                    self.map = future.result().map  # OccupancyGrid instance!
                    self.get_logger().info("Map loaded !")
                except Exception as e:
                    self.get_logger().info(f"Service call failed {e}")
                return

    # **********************************
    def get_robot_pose(self):
        """ Get the current position of the robot """
        # DO NOT TOUCH
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

    # **********************************
    def create_map_image(self):
        """ Create an image from the OccupancyGrid """
        if self.map is None:
            self.get_logger().error("Map not loaded! Cannot create map image.")
            return
        # Log map information for debugging
        self.get_logger().info(f"Map dimensions: width={self.map.info.width}, height={self.map.info.height}")
        self.get_logger().info(f"Map resolution: {self.map.info.resolution}")
        self.get_logger().info(f"Map origin: x={self.map.info.origin.position.x}, y={self.map.info.origin.position.y}")
        # Reshape map data to 2D array
        try:
            map_data = np.array(self.map.data, dtype=np.int8).reshape(self.map.info.height, self.map.info.width)
            self.get_logger().info(f"Map data shape: {map_data.shape}, unique values: {np.unique(map_data)}")
        except Exception as e:
            self.get_logger().error(f"Failed to reshape map data: {e}")
            return
        # Create image: free (0) -> 255 (white), occupied (100) -> 0 (black), unknown (-1) -> 128 (gray)
        self.map_image = np.zeros((self.map.info.height, self.map.info.width), dtype=np.uint8)
        self.map_image[map_data == 0] = 255  # Free
        self.map_image[map_data == 100] = 0  # Occupied
        self.map_image[map_data == -1] = 128  # Unknown
        # Log image statistics
        self.get_logger().info(f"Map image shape: {self.map_image.shape}, unique values: {np.unique(self.map_image)}")
        # Log the count of each value in the image for better debugging
        unique, counts = np.unique(self.map_image, return_counts=True)
        value_counts = dict(zip(unique, counts))
        self.get_logger().info(f"Map image value counts: {value_counts}")
        # Save image for debugging
        try:
            cv2.imwrite("/tmp/map_image.png", self.map_image)
            self.get_logger().info("Map image saved to /tmp/map_image.png")
        except Exception as e:
            self.get_logger().error(f"Failed to save map image: {e}")
        # Flip the image vertically for correct display
        self.map_image = np.flipud(self.map_image)
        # Display the image with a longer wait time
        cv2.imshow("Map", self.map_image)
        cv2.waitKey(100)  # Increased to 100ms to ensure display

    # **********************************
    def goalCb(self, msg):
        """ TODO - Get the goal pose """
        # get the goal pose here
        # ....
        self.run()

    # **********************************
    def run(self):
        """ TODO - Implement the Bi-RRT algorithm """
        pass
        
    # **********************************
    def publishPath(self):
        """ Send the computed path so that RVIZ displays it """
        """ TODO - Transform the waypoints from pixels coordinates to meters in the map frame """
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        path_rviz = []
        for pose_img in self.path:
            pose = PoseStamped()
            # pose.pose.position.x = ...
            # pose.pose.position.y = ...
            path_rviz.append(pose)
        msg.poses = path_rviz
        self.path_pub.publish(msg)

def main():
    # DO NOT TOUCH
    rclpy.init()

    node = BiRRT()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()