import rospy
import numpy as np
import sys
from mediapipe_hand.read_hand import hand_pose,get_world_pose,_4points_to_3d
from spatialmath import SE3,SO3
import pandas as pd
from spatialmath.base import trnorm

from ros_utils.pose_util import quat_to_R

sys.path.append('/home/hanglok/work/ur_slam')
from ik_step import init_robot 
import ros_utils.myGripper



def find_relative_se3(SE3_poses_list):
    '''
    args:
        SE3_poses_list: 相对于世界坐标系的SE3位置

    检查相对位姿的旋转角度是否超过20度
    如果超过,则将相对位姿的旋转角度限制在20度以内
    保证position确定的情况下,rotation相对平滑
    '''
    MAX_ANGLE = 10000
    relative_se3_list = []
    
    # 转换输入列表为SE3对象列表
    se3_objects = [SE3(pose) for pose in SE3_poses_list]
    
    for i in range(1, len(se3_objects)):
        relative_se3 = se3_objects[i-1].inv() * se3_objects[i]
        rpy_deg = SO3(trnorm(relative_se3.R)).rpy(order='zyx', unit='deg')
        limited_rpy = [max(min(angle, MAX_ANGLE), -MAX_ANGLE) for angle in rpy_deg]
        new_R = se3_objects[i-1].R * SO3.RPY(limited_rpy, order='zyx', unit='deg')
        
        # 创建新的SE3对象，保持位置不变，更新旋转矩阵
        new_se3 = SE3(se3_objects[i])
        new_se3.R = trnorm(new_R)
        se3_objects[i] = new_se3
        
        # 计算新的相对变换
        relative_se3 = se3_objects[i-1].inv() * se3_objects[i]
        relative_se3_list.append(relative_se3)

    # need reach original final target


    return relative_se3_list


if __name__ == "__main__":
    # object move
    rospy.init_node('dino_bot')
    # gripper = ros_utils.myGripper.MyGripper()
    robot1_ = init_robot("robot1")
    pd_data = hand_pose("data/hand_pose_2025-01-21_14-48-24.csv")
    joint = [-0.4944956938373011, -0.6829765478717249, 1.5807905197143555, 4.269962310791016, 2.18689227104187, -2.3767758051501673]

    se3 = np.zeros((4,4))
    se3[:3,:3] = np.array([[0,0,-1],[0,-1,0],[-1,0,0]])  # Rotation part
    # Translation part can be added to se3[0:3,3] if needed



    data = pd_data.get_hand_pose(2, 34)
    new_keypoints = pd_data.get_keypoints(data,list=[2,4,5,8])

    quaternion_df = pd.read_csv("data/hand_pose_quaternion_2025-01-21_14-48-24.csv")
    quaternion_data = quaternion_df.iloc[2:34]

    SE3_poses = []

    init_R = quat_to_R(quaternion_data.iloc[0])
    x_axis = init_R[:, 0]
    y_axis = init_R[:, 1]
    z_axis = init_R[:, 2]



    top_left_start,top_left_end,top_right_end = _4points_to_3d(x_axis,y_axis,z_axis,new_keypoints[0][0],new_keypoints[0][1],new_keypoints[0][2],new_keypoints[0][3])

    R = np.column_stack(get_world_pose(top_left_start,top_left_end,top_right_end))


    print(R)

    action = SE3.Rt(R,[0,0,0])
    # action = SE3(trnorm(se3))



    trajectory = np.array([
        np.mean(new_keypoints[i], axis=0) 
        for i in range(len(new_keypoints)-1)
    ])
    world_T = []
    
    robot1_.move_to_orientation(action,wait=True)


    for i in range(1,len(new_keypoints)-1):
        R = quat_to_R(quaternion_data.iloc[i])
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        top_left_start,top_left_end,top_right_start = _4points_to_3d(x_axis,y_axis,z_axis,new_keypoints[i][0],new_keypoints[i][1],new_keypoints[i][2],new_keypoints[i][3])
        
        
        world_T.append(SE3.Rt(np.column_stack(get_world_pose(top_left_start,top_left_end,top_right_start)),trajectory[i]))

    for i in world_T:
        i.printline()

    
    new_list = find_relative_se3(world_T)

    for i in new_list:
        i.printline()
    robot1_.step_in_ee(new_list,wait=True)
    
