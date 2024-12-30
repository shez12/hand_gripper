import rospy
import numpy as np

from mediapipe_hand.read_hand import hand_pose,smooth_trajectory,find_transformation

if __name__ == "__main__":
    # object move
    rospy.init_node('dino_bot')
    # gripper = ros_utils.myGripper.MyGripper()
    # robot1_ = init_robot("robot1")
    pd_data = hand_pose("/home/hanglok/work/hand_pose/mediapipe_hand/data_save/norm_point_cloud/2024-11-28_16-12-26.csv")
    data = pd_data.get_hand_pose(33, 80)
    keypoints = pd_data.get_keypoints(data)
    gripper_scale = pd_data.get_simulated_gripper_size(data)
    print(gripper_scale)
    SE3_poses = []
    for i in range(len(keypoints)-1):
        T = find_transformation(keypoints[i], keypoints[i+1])
        SE3_poses.append(T)
    smooth_SE3 = smooth_trajectory(SE3_poses)
    # robot1_.step_in_ee(smooth_SE3,wait =False)