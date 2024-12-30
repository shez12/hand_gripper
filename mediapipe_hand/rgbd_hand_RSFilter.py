import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import datetime

# 获取当前时间
now = datetime.datetime.now()
# 格式化时间字符串，例如：2023-11-05_15-30-00
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
video_rgb_save_path='video/rgb/'+timestamp+'.avi'
video_depth_save_path='video/depth/'+timestamp+'.avi'

# 初始化RealSense管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# 创建对齐对象与color流对齐
align_to = rs.stream.color
align = rs.align(align_to)

# 创建滤波器
colorizer = rs.colorizer()        
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
# decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
# 通过增加smooth_alpha和smooth_delta选项来强调滤镜的效果：
spatial.set_option(rs.option.filter_magnitude, 5)
spatial.set_option(rs.option.filter_smooth_alpha, 1)
spatial.set_option(rs.option.filter_smooth_delta, 50)
# The filter also offers some basic spatial hole filling capabilities:
spatial.set_option(rs.option.holes_fill, 3)

temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()



# 初始化MediaPipe手部模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# 初始化绘制工具
mp_drawing = mp.solutions.drawing_utils

# 相机内参（这里需要替换为您的相机内参）
fx = 912.2659912109375  # 焦距fx
fy = 911.6720581054688  # 焦距fy
cx = 637.773193359375  # 主点x坐标
cy = 375.817138671875  # 主点y坐标

# 初始化视频写入对象
fourcc_rgb = cv2.VideoWriter_fourcc(*'XVID')
fourcc_depth = cv2.VideoWriter_fourcc(*'XVID')
out_rgb = cv2.VideoWriter(video_rgb_save_path, fourcc_rgb, 30.0, (640, 480))
out_depth = cv2.VideoWriter(video_depth_save_path, fourcc_depth, 30.0, (640, 480))

frames_list = []
i=0
iter=5
try:
    while True:
        # 等待帧
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 如果帧不可用，继续循环
        if not depth_frame or not color_frame:
            continue

        
        frames_list.append(depth_frame)
        i += 1
        if i == iter:
            i = 0
            for x in range(iter):
                frame = frames_list[x]
                # frame = decimation.process(frame)
                frame = depth_to_disparity.process(frame)
                frame = spatial.process(frame)
                frame = temporal.process(frame)
                frame = disparity_to_depth.process(frame)
                frame = hole_filling.process(frame)
            frames_list = []

            depth_frame=frame

            # 将深度帧转换为 numpy 数组
            depth_image = np.asanyarray(depth_frame.get_data())

            # 将颜色帧转换为 numpy 数组
            color_image = np.asanyarray(color_frame.get_data())

            # 将BGR颜色帧转换为RGB
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # 使用MediaPipe进行手部关键点检测
            results = hands.process(rgb_image)
            # 检查是否有手部检测
            if results.multi_hand_landmarks:
                for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                    which_hand = 'Right' if handedness.classification[0].label == 'Right' else 'Left'
                    mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # x_3d_list,y_3d_list,z_3d_list=[]
                    # 遍历每个手部关键点
                    for id, landmark in enumerate(hand_landmarks.landmark):
                        # 获取归一化坐标
                        x_norm = landmark.x
                        y_norm = landmark.y

                        # 将归一化坐标转换为像素坐标
                        x_pix = int(x_norm * rgb_image.shape[1])
                        y_pix = int(y_norm * rgb_image.shape[0])

                        # 从深度图像获取对应的深度值
                        depth_value = depth_image[y_pix, x_pix]* 0.001

                        # 计算三维坐标
                        x_3d = (x_pix - cx) * depth_value / fx
                        y_3d = (y_pix - cy) * depth_value / fy
                        z_3d = depth_value   
                        # 在深度图像上绘制关键点
                        cv2.circle(depth_image, (x_pix, y_pix), 5, (255, 0, 0), -1)

                        # 打印三维坐标
                        print(f"{which_hand} Hand - Landmark {id}: ({x_3d}, {y_3d}, {z_3d})")

            # 显示RGB图像
            cv2.imshow('MediaPipe Hands', color_image)

            # 显示原始深度图像
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow('Depth Image', depth_colormap)


            # 写入视频文件
            out_rgb.write(color_image)
            out_depth.write(depth_colormap)


            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # 停止管道并关闭所有窗口
    pipeline.stop()
    cv2.destroyAllWindows()