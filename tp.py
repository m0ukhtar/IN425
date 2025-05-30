class Motion(Node):
    def __init__(self):
        super().__init__("motion_node")
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.robot_pose   = Pose2D()
        self.goal_received = False
        self.reached       = False

        self.create_subscription(Path, "/path", self.plannerCb, 1)
        self.robot_path_pub = self.create_publisher(Path, "/robot_path", qos_profile=1)
        self.vel_pub        = self.create_publisher(Twist, "/cmd_vel", 1)
        self.create_timer(0.05, self.run)

    def get_robot_pose(self) -> bool:
        try:
            trans = self.tf_buffer.lookup_transform(
                "map", "base_footprint", rclpy.time.Time())
            self.robot_pose.x     = trans.transform.translation.x
            self.robot_pose.y     = trans.transform.translation.y
            quat = (
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w,
            )
            self.robot_pose.theta = euler_from_quaternion(quat)[2]
            return True
        except TransformException as e:
            self.get_logger().warn(f"TF failed: {e}")
            return False

    def plannerCb(self, msg):
        self.reached       = False
        self.goal_received = True
        # on prend TOUTES les poses, y compris la première
        self.path = msg.poses[:]  
        self.inc  = 0

        self.real_path_msg = Path()
        self.real_path_msg.header.frame_id = "map"
        self.real_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.real_path = []

    def run(self):
        if not (self.goal_received and not self.reached):
            return

        # si on n'a pas de pose valide, on ne fait rien
        if not self.get_robot_pose():
            return

        if self.inc >= len(self.path):
            # plus de point → on s’arrête
            self.linear = 0.0
            self.angular = 0.0
            self.send_velocities()
            self.reached = True
            return

        # calcul du vecteur vers le prochain waypoint
        target = self.path[self.inc].pose.position
        dx = target.x - self.robot_pose.x
        dy = target.y - self.robot_pose.y
        rho           = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        alpha         = self.normalize_angle(angle_to_goal - self.robot_pose.theta)

        # gains
        k_rho   = 2.0
        k_alpha = 1.5

        # rotation sur place si l’angle est trop grand
        if abs(alpha) > 0.4:
            self.linear  = 0.0
            self.angular = k_alpha * alpha
        else:
            self.linear  = min(1.2, k_rho * rho)
            self.angular = k_alpha * alpha

        # on ne passe au point suivant que si on est proche ET bien orienté
        if rho < 0.25 and abs(alpha) < 0.3:
            self.inc += 1

        self.send_velocities()
        self.publish_path()

    # ... send_velocities, constrain, publish_path, normalize_angle identiques ...
