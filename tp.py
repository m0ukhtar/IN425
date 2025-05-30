#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class TestMotion(Node):
    def __init__(self):
        super().__init__('test_motion')
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 1)
        self.get_logger().info("Test motion node started")
        
        # Test simple : faire tourner le robot
        self.test_rotation()
        
    def test_rotation(self):
        """Test simple pour faire tourner le robot"""
        cmd_vel = Twist()
        
        self.get_logger().info("Testing robot rotation...")
        
        # Tourner pendant 3 secondes
        for i in range(30):  # 3 secondes à 10Hz
            cmd_vel.angular.z = 0.5  # Tourner lentement
            cmd_vel.linear.x = 0.0
            self.vel_pub.publish(cmd_vel)
            self.get_logger().info(f"Publishing rotation command {i+1}/30")
            time.sleep(0.1)
        
        # Arrêter
        cmd_vel.angular.z = 0.0
        cmd_vel.linear.x = 0.0
        self.vel_pub.publish(cmd_vel)
        self.get_logger().info("Rotation test completed")
        
        # Test avancer
        self.get_logger().info("Testing robot forward motion...")
        
        for i in range(20):  # 2 secondes
            cmd_vel.linear.x = 0.2  # Avancer lentement
            cmd_vel.angular.z = 0.0
            self.vel_pub.publish(cmd_vel)
            self.get_logger().info(f"Publishing forward command {i+1}/20")
            time.sleep(0.1)
        
        # Arrêter définitivement
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.vel_pub.publish(cmd_vel)
        self.get_logger().info("Forward test completed - robot should stop")

def main():
    rclpy.init()
    node = TestMotion()
    
    try:
        # Faire tourner le node pendant quelques secondes pour le test
        rclpy.spin_once(node, timeout_sec=10.0)
    except KeyboardInterrupt:
        pass
    
    # Arrêter le robot avant de quitter
    cmd_vel = Twist()
    cmd_vel.linear.x = 0.0
    cmd_vel.angular.z = 0.0
    node.vel_pub.publish(cmd_vel)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
