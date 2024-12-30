import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs
import time
import datetime
from gesture import detect_all_finger_state, number_gesture,character_gesture
import matplotlib.pyplot as plt


class NumPyStack:
    def __init__(self, iter):
        # 初始化一个大小为5的NumPy数组，用于存储堆栈的元素
        self.stack = np.empty(iter, dtype=object)
        self.top = -1  # 堆栈顶部索引，-1表示堆栈为空

    def is_empty(self):
        # 检查堆栈是否为空
        return self.top == -1

    def push(self, item):
        # 入栈操作
        if self.top < len(self.stack) - 1:
            self.top += 1
            self.stack[self.top] = item
        else:
            raise OverflowError("Stack is full")

    def pop(self):
        # 出栈操作
        if not self.is_empty():
            item = self.stack[self.top]
            self.top -= 1
            return item
        else:
            raise IndexError("Stack is empty")

    def peek(self):
        # 查看堆栈顶部元素
        if not self.is_empty():
            return self.stack[self.top]
        else:
            raise IndexError("Stack is empty")

    def size(self):
        # 返回堆栈中元素的数量
        return self.top + 1
    

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

    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c=np.array(colormap), s=50)
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
out_rgb = cv2.VideoWriter(video_rgb_save_path, fourcc_rgb, 30.0, (640, 480))
out_depth = cv2.VideoWriter(video_depth_save_path, fourcc_depth, 30.0, (640, 480))

start_video=0
end_video=0
stcak_frame_num=32
stack=NumPyStack(stcak_frame_num)

try:
    while True:
        frames_ = pipeline.wait_for_frames()
        frames = align.process(frames_)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
        
        if stack.size<stcak_frame_num:
            stack.push(color_frame)
            continue
        else:
            
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        """
        1.先把rgb保存到栈内
        2.调用depthanyvideo，重新获取深度图。
        """



        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for handedness, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):

                which_hand = 'Right' if handedness.classification[0].label == 'Right' else 'Left'
                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                draw3d(plt, ax, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                data=[]

                for id, landmark in enumerate(hand_landmarks.landmark):
                    x_norm = landmark.x
                    y_norm = landmark.y
                    x_pix = int(x_norm * rgb_image.shape[1])
                    y_pix = int(y_norm * rgb_image.shape[0])
                    data.append([x_pix,y_pix])
                    cv2.circle(depth_image, (x_pix, y_pix), 5, (255, 0, 0), -1)

                    # depth_value = depth_image[y_pix, x_pix]
                    # x_3d = (x_pix - cx) * depth_value / fx
                    # y_3d = (y_pix - cy) * depth_value / fy
                    # z_3d = depth_value * 0.001
                    
                    # print(f"{which_hand} Hand - Landmark {id}: ({x_3d}, {y_3d}, {z_3d})")

                bend_states, straighten_states=detect_all_finger_state(data)

                if start_video!=2 and number_gesture(bend_states, straighten_states)=='5':
                    start_video=1
                if start_video==1 and number_gesture(bend_states, straighten_states)=='0':
                    start_video=2
                    print('strat record video')
                


                if end_video!=2 and number_gesture(bend_states, straighten_states)=='2':
                    end_video=1
                if end_video==1 and number_gesture(bend_states, straighten_states)=='0':
                    end_video=2
                    print('end record video')

            
        cv2.imshow('MediaPipe Hands', color_image)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Depth Image', depth_colormap)

        if start_video==2:
            # 写入视频文件
            out_rgb.write(color_image)
            out_depth.write(depth_colormap)

        if end_video==2:
            break

        cv2.waitKey(1)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    out_rgb.release()
    out_depth.release()