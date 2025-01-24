import rospy
import numpy as np
import sys
import matplotlib.pyplot as plt
from mediapipe_hand.read_hand import hand_pose, get_world_pose
from spatialmath import SE3
from mediapipe_hand.gripper_overlay import GripperOverlay
import pandas as pd
from ros_utils.pose_util import quat_to_R



def plot_coordinate_frame(ax, rotation_matrix, origin, scale=0.1):
    """
    Helper function to draw coordinate axes.

    Args:
        ax: Matplotlib axis object
        rotation_matrix: 3x3 rotation matrix
        origin: Origin point of the coordinate frame
        scale: Scale factor for axis visualization
    """
    colors = ['r', 'g', 'b']  # x=red, y=green, z=blue
    for i in range(3):
        direction = rotation_matrix[:, i] * scale
        ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            color=colors[i], arrow_length_ratio=0.2
        )


def draw_gripper_axis(ax, top_left_start, top_left_end, top_right_end, trajectory):
    """
    Draw gripper axis and return transformation matrix.

    Args:
        ax: Matplotlib axis object
        top_left_start: Start point of top left line
        top_left_end: End point of top left line
        top_right_end: End point of top right line
        trajectory: Current trajectory point
    
    Returns:
        SE3 transformation matrix
    """
    rotation_matrix = np.column_stack(
        get_world_pose(top_left_start, top_left_end, top_right_end)
    )
    plot_coordinate_frame(ax, rotation_matrix, trajectory)
    return SE3.Rt(rotation_matrix, trajectory)

def read_quaternion(path):
    # read the quaternion from the csv file
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    # Initialize 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Initialize gripper
    gripper = GripperOverlay(ax)

    # Load and process hand pose data
    pd_data = hand_pose("data/hand_pose_2025-01-24_14-34-57.csv")
    data = pd_data.get_hand_pose(2, 34)
    # keypoints = pd_data.get_keypoints(data, list=[0, 5, 9])
    new_keypoints = pd_data.get_keypoints(data, list=[2,4,5,8])
    quaternion_df = read_quaternion("data/hand_pose_quaternion_2025-01-24_14-34-57.csv")
    quaternion_data = quaternion_df.iloc[2:34]

    # Calculate trajectory
    trajectory = np.array([
        np.mean(new_keypoints[i], axis=0) 
        for i in range(len(new_keypoints)-1)
    ])

    T_all = []

    # Main animation loop
    for i in range(len(new_keypoints)-1):
        # Clear previous frame
        ax.cla()
        
        # Reset axis limits
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)

        # # Draw gripper
        # top_left_start, top_left_end, top_right_end = gripper.draw_gripper_from_points(
        #     new_keypoints[i][0],
        #     new_keypoints[i][1],
        #     new_keypoints[i][2],
        #     new_keypoints[i][3]
        # )

        R = quat_to_R(quaternion_data.iloc[i])
        # find x-axis,y-axis,z-axis
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]
        
        top_left_start,top_left_end,top_right_end = gripper.draw_gripper_from_quaternion(x_axis,y_axis,z_axis,new_keypoints[i][0],new_keypoints[i][1],new_keypoints[i][2],new_keypoints[i][3])
        # Calculate and store transformation
        T = draw_gripper_axis(
            ax, top_left_start, top_left_end, top_right_end, trajectory[i]
        )
        T_all.append(T)
        T.printline()

        # Plot trajectory
        ax.plot(
            trajectory[:i, 0], 
            trajectory[:i, 1], 
            trajectory[:i, 2],
            c='r', 
            label="Trajectory"
        )
        
        plt.pause(0.5)
    np.save('T_all.npy', T_all)

    

    
    
