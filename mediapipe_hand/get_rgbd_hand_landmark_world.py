import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import time
import datetime

# 获取当前时间
now = datetime.datetime.now()
# 格式化时间字符串，例如：2023-11-05_15-30-00
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
video_rgb_save_path='video/rgb/'+timestamp+'.mp4'
video_depth_save_path='video/depth/'+timestamp+'.mp4'
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
out_rgb = cv2.VideoWriter(video_rgb_save_path, fourcc_rgb, 10.0, (640, 480))
out_depth = cv2.VideoWriter(video_depth_save_path, fourcc_depth, 10.0, (640, 480))

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

                for id, landmark in enumerate(hand_landmarks.landmark):
                    x_norm = landmark.x
                    y_norm = landmark.y
                    x_pix = int(x_norm * rgb_image.shape[1])
                    y_pix = int(y_norm * rgb_image.shape[0])
                    depth_value = depth_image[y_pix, x_pix]
                    x_3d = (x_pix - cx) * depth_value / fx
                    y_3d = (y_pix - cy) * depth_value / fy
                    z_3d = depth_value * 0.001
                    cv2.circle(depth_image, (x_pix, y_pix), 5, (255, 0, 0), -1)
                    print(f"{which_hand} Hand - Landmark {id}: ({x_3d}, {y_3d}, {z_3d})")

        cv2.imshow('MediaPipe Hands', color_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Depth Image', depth_colormap)

        # 写入视频文件
        out_rgb.write(color_image)
        out_depth.write(depth_colormap)

        time.sleep(0.1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    out_rgb.release()
    out_depth.release()