import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from spatialmath import SE3
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from spatialmath.base import trnorm

def get_world_pose(top_left_start,top_left_end,top_right_end):
    # z axis    
    x_vector = (top_left_start - top_left_end)/np.linalg.norm(top_left_end - top_left_start)
    # y axis
    y_vector = (-top_right_end+top_left_end)/np.linalg.norm(-top_right_end+top_left_end)
    # x axis

    z_vector = np.cross(x_vector,y_vector)
    z_vector = z_vector/np.linalg.norm(z_vector)
    # print(x_vector,y_vector,z_vector)
    return [x_vector,y_vector,z_vector]



def _4points_to_3d(x_axis,y_axis,z_axis,point_2,point_4,point_5,point_8):

        point_2,point_4,point_5,point_8 = np.array(point_2),np.array(point_4),np.array(point_5),np.array(point_8)

        distance = np.linalg.norm(point_4-point_8)
        
        top_left_end = point_2
        top_left_start = top_left_end-z_axis*0.05
        top_right_end = point_2+x_axis*distance
        top_right_start = top_right_end-z_axis*0.05
        return top_left_start,top_left_end,top_right_end




def get_gripper_scale(distance):
    #each set is 3 numbers get the distance between the two points(4 and 8)

    gripper_scale=1000*(distance)/0.08
    gripper_scale = max(0, min(gripper_scale, 1000))  # Clamp gripper_scale between 0 and 1000
    gripper_scale = (gripper_scale // 200) * 200     # Round down to the nearest multiple of 200
    return gripper_scale


"""
    "fx": 599.5029,
    "fy": 598.7244,
    "s": 0.0,
    "cx": 323.4041,
    "cy": 254.9281
"""
class hand_pose:
    def __init__(self,path) -> None:
        self.df = pd.read_csv(path)

    def read_csv(self,file_path):
        df = pd.read_csv(file_path)
        return df

    def get_hand_pose(self,start, end):
        return self.df.iloc[start:end]

    def get_keypoints(self, data,list =[0,5,9]):
        '''
        0,1,5,9,13,17
        '''
        landmarks_list = []
        for index, row in data.iterrows():
            landmarks = []
            for i in list:
                points = SE3.Tx(row[f"{i}_x_norm"]) @ SE3.Ty(row[f"{i}_y_norm"]) @ SE3.Tz(row[f"{i}_z_norm"])
                transformations = SE3.Rz(-90,unit='deg') 
                new_points =  transformations * points 
                landmarks.append([new_points.t[0], new_points.t[1], new_points.t[2]])
            landmarks_list.append(landmarks.copy())

        return np.array(landmarks_list)
    

    
    def get_simulated_gripper_size(self, data):
        scale = []
        for index, row in data.iterrows():
            thumb_tip = np.array([row["4_x_norm"], row["4_y_norm"], row["4_z_norm"]])
            index_tip = np.array([row["8_x_norm"], row["8_y_norm"], row["8_z_norm"]])
            distance = np.linalg.norm(thumb_tip - index_tip)
            scale.append(get_gripper_scale(distance))
        return scale
    

    def draw_carton(self, data, delay=0.5):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        for i in range(len(data)):
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_zlabel("Z-axis")
            ax.scatter(data[i][:, 0], data[i][:, 1], data[i][:, 2])  # 绘制当前帧的数据
            plt.draw()  # 更新图形
            plt.pause(delay)  # 暂停以展示当前帧
            if i < len(data) - 1:
                ax.cla()  # 清除当前图像内容，为下一帧准备
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
        
        plt.show()  # 展示最后一帧
        plt.close()  # 关闭图形


def find_transformation_vectors(vectorlist1, vectorlist2,point1,point2):
    """
    Find transformation matrix between two sets of orthogonal vectors
    Args:
        vectorlist1: List of 3 orthogonal unit vectors [end_vector, left_vector, cross_vector]
        vectorlist2: List of 3 orthogonal unit vectors [end_vector, left_vector, cross_vector]
    Returns:
        roll pitch yaw
    """
    R1 = np.column_stack(vectorlist1)
    R2 = np.column_stack(vectorlist2)
    T1 = np.eye(4)
    T1[:3, :3] = R1
    T1[:3, 3] = point1

    T2 = np.eye(4)
    T2[:3, :3] = R2
    T2[:3, 3] = point2

    #from T1 to T2
    T = np.linalg.inv(T1) @ T2  
    return T


def find_transformation(X, Y,translation_only=False):

    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
    if translation_only:
        T = np.eye(4)
        T[:3, 3] = cY - cX
        return T
    # Subtract centroids to obtain centered sets of points
    Xc = X - cX
    Yc = Y - cY
    # Calculate covariance matrix
    C = np.dot(Xc.T, Yc)
    # Compute SVD
    U, S, Vt = np.linalg.svd(C)
    # Determine rotation matrix
    R = np.dot(Vt.T, U.T)
    # Determine translation vector
    t = cY - np.dot(R, cX)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T




def draw_movingframe(T_ori_trans, keypoints, is_smoothed=False):
    """
    动态绘制点云和坐标系的移动过程
    :param point_clouds: 初始点云数据 (N x 3 的 numpy 数组)
    :param T: SE3 变换矩阵列表[T_ori,T_trans]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置固定的坐标轴范围
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    trajectory = []  # List to store trajectory points
    points =[]  
    points.append(keypoints[0])
    for i in range(len(T_ori_trans)):
        pointcloud2 = []
        # t_trans,T_ori = T_ori_trans[i]
        for j in range(len(keypoints[i])):
            if is_smoothed:
                point_temp = (SE3(T_ori_trans[i])*SE3.Trans(points[-1][j])).t
            else:
                point_temp = (SE3(T_ori_trans[i])*SE3.Trans(keypoints[i][j])).t
            pointcloud2.append(point_temp)
        
        print("key points",keypoints[i+1])
        print("overall ",pointcloud2)
        print("============")
        
        points.append(np.array(pointcloud2))  # Convert pointcloud to numpy array

        centroid  = np.mean(pointcloud2,axis=0)

        trajectory.append(centroid)

    # Apply Gaussian smoothing to the trajectory
    smoothed_trajectory = np.array(trajectory)
        
    def plot_coordinate_frame(ax, T, origin, scale=0.1):
        """Helper function to draw coordinate axes"""
        R = T[:3, :3]
        colors = ['r', 'g', 'b']  # x=red, y=green, z=blue
        for i in range(3):
            direction = R[:, i] * scale
            ax.quiver(origin[0], origin[1], origin[2],
                     direction[0], direction[1], direction[2],
                     color=colors[i], arrow_length_ratio=0.2)

    for n in range(20):
        for i in range(1, len(smoothed_trajectory)):
            ax.cla()
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_zlabel("Z-axis")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)

            # Plot trajectory and point cloud
            ax.plot(smoothed_trajectory[:i, 0], smoothed_trajectory[:i, 1], smoothed_trajectory[:i, 2], 
                   c='r', label="Trajectory")
            ax.scatter(smoothed_trajectory[i, 0], smoothed_trajectory[i, 1], smoothed_trajectory[i, 2], 
                      c='b', s=5, label="Point Cloud")
            
            # Draw coordinate frame at current position
            current_centroid = smoothed_trajectory[i]
            plot_coordinate_frame(ax, T_ori_trans[i-1], current_centroid)

            ax.legend()
            plt.pause(0.5)

    plt.show()
    plt.close()

if __name__ == "__main__":
    pd_data = hand_pose("/home/hanglok/work/hand_gripper/mediapipe_hand/data_save/norm_point_cloud/hand_pose_2024-12-31_16-05-16.csv")
    data = pd_data.get_hand_pose(2,50)
    keypoints = pd_data.get_keypoints(data,list=[0,5,9])
    new_keypoints = pd_data.get_keypoints(data,list=[4,8,2,9])
    full_hand = pd_data.get_keypoints(data,list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    pd_data.draw_carton(full_hand)


    





    