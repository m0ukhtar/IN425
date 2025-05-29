def goalCb(self, msg):
    """
    Callback appelé quand l'utilisateur clique sur 2D Nav Goal dans RVIZ.
    Récupère les coordonnées de la cible (goal) en mètres, puis convertit en pixels.
    """
    # 1. Récupération du goal en mètres (dans le repère de la carte)
    x_goal_map = msg.pose.position.x
    y_goal_map = msg.pose.position.y

    # 2. Transformation en coordonnées dans le repère de l’origine de la map
    origin_x = self.map.info.origin.position.x
    origin_y = self.map.info.origin.position.y
    resolution = self.map.info.resolution

    x_goal_origin = x_goal_map - origin_x
    y_goal_origin = y_goal_map - origin_y

    # 3. Transformation en indices de cellules (pixels) dans l'image
    x_img = int(x_goal_origin / resolution)
    y_img = int(y_goal_origin / resolution)

    # 4. Inversion de l'axe Y (coordonnées image)
    height = self.map.info.height
    y_img = height - y_img  # inversion verticale pour correspondre à l’image OpenCV

    # 5. Stockage du goal sous forme (x, y) en pixels
    self.goal = (x_img, y_img)

    # 6. Affichage de debug (optionnel)
    self.get_logger().info(f"[Goal Callback] Goal en map frame: ({x_goal_map:.2f}, {y_goal_map:.2f})")
    self.get_logger().info(f"[Goal Callback] Goal en image: {self.goal}")
    
def run(self):
    """
    Fonction principale appelée périodiquement. Elle :
    - Récupère la position actuelle du robot.
    - La convertit en coordonnées image.
    - Lance l'algorithme BiRRT.
    - Publie le chemin trouvé.
    """
    if self.goal is None:
        self.get_logger().warn("Aucun goal défini. Cliquez sur 2D Nav Goal dans RVIZ.")
        return

    try:
        # 1. Obtenir la position actuelle du robot en map frame
        now = rclpy.time.Time()
        transform = self.tf_buffer.lookup_transform(
            'map', 'base_footprint', now
        )

        x_robot = transform.transform.translation.x
        y_robot = transform.transform.translation.y

        # 2. Conversion en coordonnées image (pixels)
        origin_x = self.map.info.origin.position.x
        origin_y = self.map.info.origin.position.y
        resolution = self.map.info.resolution
        height = self.map.info.height

        x_robot_origin = x_robot - origin_x
        y_robot_origin = y_robot - origin_y

        x_img = int(x_robot_origin / resolution)
        y_img = height - int(y_robot_origin / resolution)  # Inversion Y

        self.get_logger().info(f"[run] Start en map: ({x_robot:.2f}, {y_robot:.2f})")
        self.get_logger().info(f"[run] Start en image: ({x_img}, {y_img})")

        start = (x_img, y_img)

        # 3. Vérifie si le start ou goal est dans un obstacle
        if self.image[start[1], start[0]] != 255:
            self.get_logger().error("Start position is in an obstacle!")
            return
        if self.image[self.goal[1], self.goal[0]] != 255:
            self.get_logger().error("Goal position is in an obstacle!")
            return

        # 4. Planification avec BiRRT
        self.get_logger().info("Lancement de BiRRT...")
        path = self.birrt.run(start=start, goal=self.goal)

        if path is None:
            self.get_logger().warn("Aucun chemin trouvé par BiRRT.")
            return

        # 5. Réduction du chemin (facultatif si tu l'as codé)
        if hasattr(self.birrt, 'reduce_path'):
            path = self.birrt.reduce_path(path)

        # 6. Publier le chemin
        self.publishPath(path)

    except Exception as e:
        self.get_logger().error(f"Erreur dans run(): {str(e)}")
