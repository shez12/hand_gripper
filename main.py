import rospy
import numpy as np
import sys
from mediapipe_hand.read_hand import hand_pose,smooth_trajectory,find_transformation_vectors,_4points_to_3d
from spatialmath import SE3


sys.path.append('/home/hanglok/work/ur_slam')
from ik_step import init_robot 
import ros_utils.myGripper




if __name__ == "__main__":
    # object move
    # rospy.init_node('dino_bot')
    # gripper = ros_utils.myGripper.MyGripper()
    robot1_ = init_robot("robot1")
    pd_data = hand_pose("/home/hanglok/work/hand_gripper/mediapipe_hand/data_save/norm_point_cloud/hand_pose_2024-12-31_16-05-16.csv")
    data = pd_data.get_hand_pose(2, 50)
    keypoints = pd_data.get_keypoints(data,list=[0,5,9])
    new_keypoints = pd_data.get_keypoints(data,list=[4,8,2,9])
    # gripper_scale = pd_data.get_simulated_gripper_size(data)
    # print(gripper_scale)
    SE3_poses = []
    for i in range(len(keypoints)-1):
        vector_list1, point1  = _4points_to_3d(new_keypoints[0],find_transformation=True)
        vector_list2, point2  = _4points_to_3d(new_keypoints[i+1],find_transformation=True)

        T_ori = find_transformation_vectors(vector_list1,vector_list2,point1,point2)
        SE3_poses.append(T_ori)
    smooth_SE3 = smooth_trajectory(SE3_poses)
    # robot1_.step_in_ee(smooth_SE3,wait =False)