import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import time
import datetime
from mediapipe_hand.gesture import detect_all_finger_state, number_gesture,calculate_distance
import matplotlib.pyplot as plt
import pandas as pd
import os
from gripper_overlay import get_gripper_scale
import sys
import rospy

sys.path.append('/home/hanglok/work/ur_slam')
import ros_utils.myGripper

rospy.init_node('gripper_control', anonymous=True)

gripper = ros_utils.myGripper.MyGripper()
colorclass = plt.cm.ScalarMappable(cmap='jet')
colors = colorclass.to_rgba(np.linspace(0, 1, int(21)))
colormap = (colors[:, 0:3])

def draw3d(plt, ax, world_landmarks, connnection):
    ax.clear()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    landmarks = []
    for index, landmark in enumerate(world_landmarks.landmark):
        landmarks.append([landmark.x, landmark.z, landmark.y*(-1)])
    landmarks = np.array(landmarks)

    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c=np.array(colormap), s=10)
    for _c in connnection:
        ax.plot([landmarks[_c[0], 0], landmarks[_c[1], 0]],
                [landmarks[_c[0], 1], landmarks[_c[1], 1]],
                [landmarks[_c[0], 2], landmarks[_c[1], 2]], 'k')

    plt.pause(0.001)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")



# 获取当前时间
now = datetime.datetime.now()
# 格式化时间字符串，例如：2023-11-05_15-30-00
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
video_rgb_save_path='data_save/rgb/'+timestamp+'.mp4'
video_depth_save_path='data_save/depth/'+timestamp+'.mp4'
norm_point_csv_save_path='data_save/norm_point_cloud/'+timestamp+'.csv'
# 初始化RealSense管道
pipeline = rs.pipeline()
config = rs.config()

# 配置流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动管道
pipeline.start(config)

# 创建对齐对象与color流对齐
align_to = rs.stream.color
align = rs.align(align_to)

# 初始化MediaPipe手部模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# 初始化绘制工具
mp_drawing = mp.solutions.drawing_utils

# 相机内参
fx = 599.5029
fy = 598.7244
cx = 323.4041
cy = 254.9281

# 初始化视频写入对象
fourcc_rgb = cv2.VideoWriter_fourcc(*'mp4v')
fourcc_depth = cv2.VideoWriter_fourcc(*'mp4v')
out_rgb = cv2.VideoWriter(video_rgb_save_path, fourcc_rgb, 30.0, (640, 480))
out_depth = cv2.VideoWriter(video_depth_save_path, fourcc_depth, 30.0, (640, 480))
data_point_cloud_save=[]

start_video=0
end_video=0

last_gesture_time = 0  # 记录上一次检测到手势“5”的时间
last_hand_position = None  # 记录上一次手部的位置
wait_time=2  # 手部不动2秒开始记录
wait_distance=1   # 手部不动的阈值

start_record_move=0
end_record_move=0

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
            for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):

                which_hand = 'Right' if handedness.classification[0].label == 'Right' else 'Left'
                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                draw3d(plt, ax, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                data_pixel,data_point_cloud=[],[]
                refer_depth=None
                for id, landmark in enumerate(hand_landmarks.landmark):
                    x_norm = landmark.x
                    y_norm = landmark.y
                    z_norm=landmark.z
                    # data_point_cloud.append(x_norm)
                    # data_point_cloud.append(y_norm)
                    # data_point_cloud.append(z_norm)

                    x_pix = int(x_norm * rgb_image.shape[1])
                    y_pix = int(y_norm * rgb_image.shape[0])
                    data_pixel.append([x_pix,y_pix])
                    # cv2.circle(depth_image, (x_pix, y_pix), 5, (255, 0, 0), -1)
                    if (id == 0 ):
                    
                        if depth_image[y_pix, x_pix] == 0:
                                continue
                        refer_depth = depth_image[y_pix, x_pix]/1000
                        depth_value = refer_depth 

                    else:
                        # 其余手指根据手腕（id=0）
                        if refer_depth is None:
                            continue
                        depth_value = refer_depth+z_norm
                    x_3d = (x_pix - cx) * depth_value / fx
                    y_3d = (y_pix - cy) * depth_value / fy
                    z_3d = depth_value 
                    data_point_cloud.append(x_3d)
                    data_point_cloud.append(y_3d)
                    data_point_cloud.append(z_3d)
                    
                    # print(f"{which_hand} Hand - Landmark {id}: ({x_3d}, {y_3d}, {z_3d})")
                point1 = data_point_cloud[(4-1)*3:4*3]#number4
                point2 = data_point_cloud[(8-1)*3:8*3]#number8
                distance=np.linalg.norm(np.array(point1)-np.array(point2))
                gripper_scale=get_gripper_scale(distance)
                print(gripper_scale)
                gripper.set_gripper(gripper_scale,5)


        cv2.imshow('MediaPipe Hands', color_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # cv2.imshow('Depth Image', depth_colormap)
        
        cv2.waitKey(1)

finally:

    pipeline.stop()
    cv2.destroyAllWindows()
    out_rgb.release()
    out_depth.release()