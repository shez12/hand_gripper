import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
import time
from gesture import detect_all_finger_state, judge_data, get_gripper_coordinates
import numpy as np


def read_npy_and_separate(file_path):
    data = np.load(file_path)  # Load the .npy file
    separated_data = []  # Initialize a list to hold separated data
    current_chunk = []  # Temporary list to hold current chunk

    for item in data:
        if item.tolist() == [-1, -1, -1]:  # Check for the delimiter
            if current_chunk:  # If current_chunk is not empty
                separated_data.append(current_chunk)  # Append the chunk
                current_chunk = []  # Reset for the next chunk
        else:
            current_chunk.append(item.tolist())  # Add item to the current chunk

    if current_chunk:  # Append any remaining data
        separated_data.append(current_chunk)

    return separated_data  # Return the separated data



file_path=r'landmarks1730096345.8044925.npy'
# 使用numpy的load函数读取.npy文件
data = read_npy_and_separate(file_path)
# print(data[0])

h,w=900,1600

print(len(data[0]))

for i in range(len(data)):
    data_new=[]
    for j in data[i]:
        data_new.append((int(j[0]*w),int(j[1]*h)))


    # 调用函数，判断每根手指的弯曲或伸直状态
    bend_states, straighten_states = detect_all_finger_state(data_new,data)



    # 调用函数，检测当前手势
    current_state = judge_data(bend_states, straighten_states)

    if current_state==True:
        gripper_coordinates=get_gripper_coordinates(data_new) # return tips_first, tips_first, arc_midpoint 

        print(current_state)
        print(gripper_coordinates)
    else: 
        print(False)
