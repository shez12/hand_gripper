# Standard library imports
import numpy as np

# Third-party imports
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mediapipe_hand.read_hand import _4points_to_3d


CAMERA_PARAMS = {
    'fx': 599.5029,
    'fy': 598.7244,
    'cx': 323.4041,
    'cy': 254.9281
}
fx = CAMERA_PARAMS['fx']
fy = CAMERA_PARAMS['fy']
cx = CAMERA_PARAMS['cx']
cy = CAMERA_PARAMS['cy']



def get_gripper_scale(distance: float) -> float:
    """
    Calculate gripper scale based on distance between points.
    
    Args:
        distance (float): Distance between gripper points in meters
        
    Returns:
        float: Calculated gripper scale rounded to nearest 200
    """
    gripper_scale = 1000 * distance / 0.1
    gripper_scale = max(0, min(gripper_scale, 1000))  # Clamp between 0-1000
    return (gripper_scale // 200) * 200  # Round to nearest 200






class GripperOverlay:
    """A class to handle 3D gripper visualization overlay."""
    
    def __init__(self, ax: Axes3D):
        """
        Initialize the gripper overlay.
        
        Args:
            ax (Axes3D): Matplotlib 3D axes object for rendering
        """
        self.ax = ax
        self.gripper_lines = []
        
        # Constants for gripper dimensions (in meters)
        self.grip_scale = 0
        self.max_spread = 0.1
        self.gripper_width = 0.02
        self.gripper_length = 0.1
        self.gripper_height = 0.02
        self.base_x = 0.5
        self.base_y = 0.5
        self.base_z = 0.5
        self.column_depth = 0.02

    def _create_cuboid_vertices(self, start, end, height, depth):
        """
        args:
            start: tuple, start point
            end: tuple, end point
            height: float, height of the cuboid
            depth: float, depth of the cuboid
        """
        direction = np.array(end) - np.array(start)
        length = np.linalg.norm(direction)
        if length == 0:
            raise ValueError("Start and end points cannot be the same.")

        direction = direction / length  # Normalize the direction

        # Calculate the perpendicular vectors for width and height
        perp_vector = np.cross(direction, [0, 0, 1])
        if np.linalg.norm(perp_vector) == 0:
            perp_vector = np.cross(direction, [0, 1, 0])
        perp_vector = perp_vector / np.linalg.norm(perp_vector)

        height_vector = np.cross(direction, perp_vector)
        height_vector = height_vector / np.linalg.norm(height_vector)

        # Calculate the vertices
        return [
            start,
            start + direction * length,
            start + direction * length + height_vector * height,
            start + height_vector * height,
            start + perp_vector * depth,
            start + direction * length + perp_vector * depth,
            start + direction * length + height_vector * height + perp_vector * depth,
            start + height_vector * height + perp_vector * depth
        ]

    def _draw_lines(self, vertices, color='blue'):
        """
        Draw edges of a cuboid based on given vertices.
        
        Args:
            vertices (list): List of 8 vertices (each a tuple (x, y, z))
            color (str): Color of the lines, default is 'blue'
        """
        # Define the 12 edges of the cuboid
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 垂直边
        ]

        for edge in edges:
            start, end = edge
            line, = self.ax.plot(
                [vertices[start][0], vertices[end][0]],
                [vertices[start][1], vertices[end][1]],
                [vertices[start][2], vertices[end][2]],
                color=color
            )
            self.gripper_lines.append(line)

    def _clear_gripper(self):
        """
        Clear previously drawn gripper and column lines.
        """
        for line in self.gripper_lines:
            line.remove()
        self.gripper_lines = []

    def draw_gripper_from_points(self, top_left, top_right, bottom_left, bottom_right):
        """Draw gripper based on four input points."""
        self._clear_gripper()

        # Calculate base vectors
        base_mid = np.mean([top_left, top_right], axis=0)
        base_vector = np.array(bottom_left) - np.array(bottom_right)
        base_vector = base_vector / np.linalg.norm(base_vector)
        
        plane_vector = np.cross(base_vector, np.array(top_left) - np.array(bottom_right))
        plane_vector = plane_vector / np.linalg.norm(plane_vector)

        # Calculate gripper vectors
        left_vector = right_vector = np.cross(base_vector, plane_vector)
        left_vector = right_vector = left_vector / np.linalg.norm(left_vector)

        # Calculate positions
        distance = np.linalg.norm(np.array(top_left) - np.array(top_right))
        top_left_start = base_mid + base_vector * distance/2
        top_right_start = base_mid - base_vector * distance/2
        top_left_end = top_left_start + left_vector * 0.05
        top_right_end = top_right_start + right_vector * 0.05

        # Draw gripper rods
        for start, end in [(top_left_start, top_left_end), (top_right_start, top_right_end)]:
            rod = self._create_cuboid_vertices(start, end, self.gripper_height, self.column_depth)
            self._draw_lines(rod, color='blue')

        # Draw points and legend
        points = [(top_left, 'red', 'Top Left'), 
                 (top_right, 'green', 'Top Right'),
                 (bottom_left, 'orange', 'Bottom Left'),
                 (bottom_right, 'purple', 'Bottom Right')]
        
        for point, color, label in points:
            self.ax.scatter(*point, color=color, s=50, label=label)
        self.ax.legend()

def draw_gripper_from_points_cv(
        top_left: tuple, 
        top_right: tuple, 
        bottom_left: tuple, 
        bottom_right: tuple, 
        image: np.ndarray,
    ) -> None:
        """
        Draw gripper overlay on OpenCV image based on four input points.
        
        Args:
            top_left: Left column top coordinates (x, y, z)
            top_right: Right column top coordinates (x, y, z)
            bottom_left: Left linkage point coordinates (x, y, z)
            bottom_right: Right linkage point coordinates (x, y, z)
            image: OpenCV image to draw on
            fx, fy: Camera focal lengths
            cx, cy: Camera principal point coordinates
        """
        def project_point(point):
            """Project 3D point to 2D image coordinates."""
            x, y, z = point
            x_2d = int((x * fx / z) + cx)
            y_2d = int((y * fy / z) + cy)
            return (x_2d, y_2d)

        top_left_start,top_left_end,top_right_start,top_right_end = _4points_to_3d([top_left, top_right, bottom_left, bottom_right])

        # Project and draw lines
        top_left_2d = project_point(top_left_start)
        top_left_end_2d = project_point(top_left_end)
        cv2.line(image, top_left_2d, top_left_end_2d, (255, 0, 0), 2)  # Blue line

        top_right_2d = project_point(top_right_start)
        top_right_end_2d = project_point(top_right_end)
        cv2.line(image, top_right_2d, top_right_end_2d, (255, 0, 0), 2)  # Blue line

        # Draw the input points
        bottom_left_2d = project_point(bottom_left)
        bottom_right_2d = project_point(bottom_right)


        cv2.line(image, top_left_end_2d, top_right_end_2d, (0, 0, 255), 2)  # Red line
        cv2.circle(image, top_left_2d, 5, (0, 0, 255), -1)  # Red point
        cv2.circle(image, top_right_2d, 5, (0, 255, 0), -1)  # Green point
        cv2.circle(image, bottom_left_2d, 5, (0, 165, 255), -1)  # Orange point
        cv2.circle(image, bottom_right_2d, 5, (128, 0, 128), -1)  # Purple point
        return top_left_start,top_left_end,top_right_end




