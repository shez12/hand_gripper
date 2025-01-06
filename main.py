import rospy
import numpy as np
import sys
from mediapipe_hand.read_hand import hand_pose,smooth_trajectory,find_transformation



sys.path.append('/home/hanglok/work/ur_slam')
from ik_step import init_robot 
import ros_utils.myGripper




if __name__ == "__main__":
    # object move
    rospy.init_node('dino_bot')
    # gripper = ros_utils.myGripper.MyGripper()
    robot1_ = init_robot("robot1")
    pd_data = hand_pose("/home/hanglok/work/hand_gripper/mediapipe_hand/data_save/norm_point_cloud/hand_pose_2024-12-31_16-05-16.csv")
    data = pd_data.get_hand_pose(2, 50)
    keypoints = pd_data.get_keypoints(data)
    gripper_scale = pd_data.get_simulated_gripper_size(data)
    print(gripper_scale)
    SE3_poses = []
    for i in range(len(keypoints)-1):
        T = find_transformation(keypoints[i], keypoints[i+1],translation_only=True)
        SE3_poses.append(T)
    smooth_SE3 = smooth_trajectory(SE3_poses)
    robot1_.step_in_ee(smooth_SE3,wait =False)