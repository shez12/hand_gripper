# Standard library imports
import sys
import datetime

# Third-party imports
import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import rospy

# Local imports
from mediapipe_hand.gesture import detect_all_finger_state, number_gesture, calculate_distance
from mediapipe_hand.gripper_overlay import GripperOverlay,draw_gripper_from_points_cv

# ROS setup
sys.path.append('/home/hanglok/work/ur_slam')
import ros_utils.myGripper
# rospy.init_node('gripper_control', anonymous=True)

# Initialize 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)

# Initialize gripper and color settings
gripper = GripperOverlay(ax)
colorclass = plt.cm.ScalarMappable(cmap='jet')
colors = colorclass.to_rgba(np.linspace(0, 1, 21))
colormap = colors[:, 0:3]

# Constants should be at the top level of module and in UPPER_CASE
CAMERA_PARAMS = {
    'fx': 599.5029,
    'fy': 598.7244,
    'cx': 323.4041,
    'cy': 254.9281
}
def draw3d(plt, ax, world_landmarks, connection):
    """Draw 3D landmarks and connections.
    
    Args:
        plt: matplotlib.pyplot instance
        ax: 3D axis object
        world_landmarks: MediaPipe landmarks
        connection: List of landmark connections
    """
    ax.clear()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    landmarks = [[landmark.x, landmark.z, landmark.y * (-1)] 
                for landmark in world_landmarks.landmark]
    landmarks = np.array(landmarks)

    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], 
              c=np.array(colormap), s=10)
    
    for connection_pair in connection:
        ax.plot([landmarks[connection_pair[0], 0], landmarks[connection_pair[1], 0]],
                [landmarks[connection_pair[0], 1], landmarks[connection_pair[1], 1]],
                [landmarks[connection_pair[0], 2], landmarks[connection_pair[1], 2]], 
                'k')

    plt.pause(0.001)

def draw_hand(ax, data_point_cloud):
    """Draw hand landmarks and connections in 3D."""
    ax.clear()
    ax.scatter(data_point_cloud[0::3], data_point_cloud[1::3], data_point_cloud[2::3], s=10)
    
    # Draw connections for each finger
    finger_ranges = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
    for start, end in finger_ranges:
        for i in range(start, end):
            ax.plot(
                [data_point_cloud[i*3], data_point_cloud[(i+1)*3]],
                [data_point_cloud[i*3+1], data_point_cloud[(i+1)*3+1]],
                [data_point_cloud[i*3+2], data_point_cloud[(i+1)*3+2]],
                'k'
            )

def process_hand_landmarks(hand_landmarks, rgb_image, depth_image):
    """Process hand landmarks and return 3D points.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        rgb_image: RGB image array
        depth_image: Depth image array
    
    Returns:
        list: 3D point cloud data
    """
    data_pixel, data_point_cloud = [], []
    refer_depth = None
    
    for idx, landmark in enumerate(hand_landmarks.landmark):
        x_pix = int(landmark.x * rgb_image.shape[1])
        y_pix = int(landmark.y * rgb_image.shape[0]) 
        data_pixel.append([x_pix, y_pix])
        
        if idx == 0:
            if depth_image[y_pix, x_pix] == 0:
                return []
            refer_depth = depth_image[y_pix, x_pix] / 1000
            depth_value = refer_depth
        else:
            if refer_depth is None:
                return []
            depth_value = refer_depth + landmark.z
            
        x_3d = (x_pix - CAMERA_PARAMS['cx']) * depth_value / CAMERA_PARAMS['fx']
        y_3d = (y_pix - CAMERA_PARAMS['cy']) * depth_value / CAMERA_PARAMS['fy']
        z_3d = depth_value
        
        data_point_cloud.extend([x_3d, y_3d, z_3d])
    
    return data_point_cloud

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
pipeline.start(config)
align = rs.align(rs.stream.color)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
data_point_cloud_save = []
start_video = 0
end_video = 0
last_gesture_time = 0  # Record the time of the last detected gesture "5"
last_hand_position = None  # Record the position of the last hand
start_record_move = 0
end_record_move = 0

try:
    while True:
        frames_ = pipeline.wait_for_frames()
        frames = align.process(frames_)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for handedness, hand_landmarks in zip(results.multi_handedness, 
                                               results.multi_hand_landmarks):
                which_hand = ('Right' if handedness.classification[0].label == 'Right' 
                            else 'Left')
                mp_drawing.draw_landmarks(color_image, hand_landmarks, 
                                       mp_hands.HAND_CONNECTIONS)
                
                data_point_cloud = process_hand_landmarks(hand_landmarks, rgb_image, 
                                                        depth_image)
                if not data_point_cloud:
                    continue
                
                # Extract landmark points
                point2 = data_point_cloud[6:9]    # thumb base
                point4 = data_point_cloud[12:15]  # thumb tip
                point5 = data_point_cloud[15:18]  # thumb middle
                point8 = data_point_cloud[36:39]  # index finger tip
                point9 = data_point_cloud[27:30]  # index finger base

                draw_hand(ax, data_point_cloud)
                gripper.draw_gripper_from_points(point2, point4, point5, point8)
                draw_gripper_from_points_cv(point2, point4, point5, point8, color_image)
                plt.draw()
                plt.pause(0.001)

                # input('Press Enter to continue...')

        cv2.imshow('MediaPipe Hands', color_image)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.imshow('Depth Image', depth_colormap)
        
        cv2.waitKey(1)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()