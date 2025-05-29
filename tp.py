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
