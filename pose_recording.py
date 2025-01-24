# Standard library imports
import time

# Third-party imports
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import rospy
from spatialmath import SE3
# Local imports
from util import get_timestamp
from mySensor import load_intrinsics, MyImageSaver, depth_to_colormap
from ros_utils.imu_data import QuaternionListener
from ros_utils.aruco_util import get_marker_pose
from new_camera_ori import IMUSubscriber, filter_gravity, compute_rotation_matrix

# Global constants
ENABLE_IMU = True
FRAME_RATE = 60



num_id = 123


if __name__ == "__main__":
    # Get current timestamp for file naming
    timestamp = get_timestamp()
    csv_filepath = f'data/hand_pose_{timestamp}.csv'
    raw_video_filepath = f'data/hand_pose_raw_{timestamp}.mp4'
    processed_video_filepath = f'data/hand_pose_processed_{timestamp}.mp4'
    quaternion_filepath = f'data/hand_pose_quaternion_{timestamp}.csv'

    # Load camera intrinsics
    intrinsics = load_intrinsics("config/camera_intrinsics.json")
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    # Initialize ROS node and sensors
    rospy.init_node('hand_pose_recording', anonymous=True)
    camera = MyImageSaver(cameraNS='camera1')
    
    if ENABLE_IMU:
        quaternion_listener = QuaternionListener()
    imu_subscriber = IMUSubscriber()
    time.sleep(2)

    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize drawing utils and data storage
    mp_drawing = mp.solutions.drawing_utils
    csv_data = []
    recording = False
    quaternion_list = []  # Move this outside the hand landmarks check

    while True:
        color_image, depth_image = camera.get_frames()
        original_image = color_image.copy()

        if color_image is None or depth_image is None:
            continue  # Skip if no frames are available

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        if results.multi_hand_landmarks:
            # only consider one hand
            if len(results.multi_hand_landmarks) > 1:
                print("more than one hand detected")
                continue
            for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):

                # which_hand = 'Right' if handedness.classification[0].label == 'Right' else 'Left'
                # print("label:", handedness.classification[0].label)
                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                data_pixel,data_point_cloud=[],[]

                for id, landmark in enumerate(hand_landmarks.landmark):
                    x_norm = landmark.x
                    y_norm = landmark.y
                    z_norm = landmark.z
                    
                    x_pix = int(x_norm * rgb_image.shape[1])
                    y_pix = int(y_norm * rgb_image.shape[0])
                    data_pixel.append([x_pix,y_pix])

                    if (id == 0 ):
                        if depth_image[y_pix, x_pix] == 0:
                            continue
                        refer_depth = depth_image[y_pix, x_pix]/1000
                        depth_value = refer_depth 

                    else:
                        # 其余手指根据手腕（id=0）
                        depth_value = refer_depth+z_norm

                    x_3d = (x_pix - cx) * depth_value / fx
                    y_3d = (y_pix - cy) * depth_value / fy
                    z_3d = depth_value 
                    # 获取marker的位姿
                    fixed_marker_pose, corner = get_marker_pose(color_image, num_id)
                    if fixed_marker_pose is not None:
                        camera_pose = fixed_marker_pose.inv()
                        # 计算gripper的位姿
                        temp_data = camera_pose * SE3.Trans(x_3d, y_3d, z_3d)
                        res_data = temp_data.t
                        if id == 0:
                            print(res_data)
                    else:
                        res_data = np.array([x_3d, y_3d, z_3d])

                    data_point_cloud.append(res_data[0])  # x coordinate
                    data_point_cloud.append(res_data[1])  # y coordinate
                    data_point_cloud.append(res_data[2])  # z coordinate
            if recording and ENABLE_IMU:  # Add this to collect quaternions during recording
                quaternion_list.append(quaternion_listener.get_current_quaternion())
                
        # Check for key press events
        key = cv2.waitKey(1000//FRAME_RATE) & 0xFF
        

            
        # Start recording video with blank key
        if key == ord(' '):
            if not recording:
                recording = True
                print("Recording started (via keyboard input)")
                # Initialize two VideoWriters
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                raw_video_writer = cv2.VideoWriter(raw_video_filepath, fourcc, FRAME_RATE/2, (640, 480))
                processed_video_writer = cv2.VideoWriter(processed_video_filepath, fourcc, FRAME_RATE/2, (640, 480))
                
            else: 
                recording = False
                print("Recording ended (via keyboard input)")
                # Save hand pose data
                df = pd.DataFrame(csv_data, columns=[f'{i}_{c}' for i in range(21) for c in ['x_norm', 'y_norm', 'z_norm']])
                df.to_csv(csv_filepath, index=False)
                
                # Save quaternion data
                quaternion_df = pd.DataFrame([
                    {'x': q['x'], 'y': q['y'], 'z': q['z'], 'w': q['w']} 
                    for q in quaternion_list
                ])
                quaternion_df.to_csv(quaternion_filepath, index=False)
                
                # Release both video writers
                raw_video_writer.release()
                processed_video_writer.release()
                cv2.destroyAllWindows()
                break
        if key == ord('q'):
            break
        if recording:
            #保存正则化的点云数据
            csv_data.append(np.array(data_point_cloud).flatten())
            raw_video_writer.write(original_image)
            processed_video_writer.write(color_image)
            # draw a red circle to show recording   
            cv2.circle(color_image, (30, 30), 15, (0, 0, 255), -1)

        depth_colormap = depth_to_colormap(depth_image)
        cv2.imshow("view", cv2.hconcat([color_image, depth_colormap]))

