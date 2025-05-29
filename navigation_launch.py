from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package = "in425_nav",
            executable = "rrt_node.py",
            output = "screen",
        ),
        Node(
            package = "in425_nav",
            executable = "motion_node.py",
            output = "screen"
        )
    ])