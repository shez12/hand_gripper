import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from spatialmath import SE3
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

def get_gripper_scale(distance):
    #each set is 3 numbers get the distance between the two points(4 and 8)

    gripper_scale=1000*(distance)/0.08
    gripper_scale = max(0, min(gripper_scale, 1000))  # Clamp gripper_scale between 0 and 1000
    gripper_scale = (gripper_scale // 200) * 200     # Round down to the nearest multiple of 200
    return gripper_scale


def decompose_transformations(transformations):
    translations = []
    rotations = []
    for T in transformations:
        translations.append(T[:3, 3])
        rotations.append(R.from_matrix(T[:3, :3]))

    return np.array(translations), rotations

def smooth_transformations(translations, rotations):
    # Smooth translations
    smoothed_translations = gaussian_filter1d(translations, sigma=1, axis=0)
    
    # Convert rotations to rotation vectors for smoothing
    rotation_vectors = np.array([r.as_rotvec() for r in rotations])
    smoothed_rotation_vectors = gaussian_filter1d(rotation_vectors, sigma=1, axis=0)                                                                                                                                                                                        
    
    # Convert back to rotation objects
    smoothed_rotations = [R.from_rotvec(rv) for rv in smoothed_rotation_vectors]
    
    return smoothed_translations, smoothed_rotations

def recompose_transformations(smoothed_translations, smoothed_rotations):
    smoothed_transformations = []
    for t, r in zip(smoothed_translations, smoothed_rotations):
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()
        T[:3, 3] = t
        smoothed_transformations.append(T)
    return smoothed_transformations


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

    def get_keypoints(self, data):
        '''
        0,1,5,9,13,17
        '''
        landmarks_list = []
        for index, row in data.iterrows():
            landmarks = []
            for i in [0,1,5,13,17]:
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
    


def draw3d(plt, ax, world_landmarks, connnection):
    colorclass = plt.cm.ScalarMappable(cmap='jet')
    colors = colorclass.to_rgba(np.linspace(0, 1, int(21)))
    colormap = (colors[:, 0:3])
    ax.clear()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    landmarks = []
    for index, landmark in enumerate(world_landmarks.landmark):
        landmarks.append([landmark.x, landmark.z, landmark.y*(-1)])
    landmarks = np.array(landmarks)

    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c=np.array(colormap), s=10)
    for _c in connnection:
        ax.plot([landmarks[_c[0], 0], landmarks[_c[1], 0]],
                [landmarks[_c[0], 1], landmarks[_c[1], 1]],
                [landmarks[_c[0], 2], landmarks[_c[1], 2]], 'k')

    plt.pause(0.001)


def find_transformation(X, Y):

    # Calculate centroids
    cX = np.mean(X, axis=0)
    cY = np.mean(Y, axis=0)
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



def smooth_trajectory(T_):

    T_list = []
    T_overall =np.eye(4)
    for i in T_:
        T_overall = T_overall @ i
        T_list.append(T_overall)
    

    translations, rotations = decompose_transformations(T_list)

    smoothed_translations, smoothed_rotations = smooth_transformations(translations, rotations)
    smooth_transformations_ = recompose_transformations(smoothed_translations, smoothed_rotations)


    # transfer from T_overall to T
    T_out = []
    T_overall  = np.eye(4)
    T_out.append(SE3(smooth_transformations_[0]))
    for i in range(0,len(smooth_transformations_)-2,2):
        T_out.append(SE3(np.linalg.inv(smooth_transformations_[i]) @ smooth_transformations_[i+1]))
    return T_out









def draw_movingframe(T):
    """
    动态绘制点云和坐标系的移动过程
    :param point_clouds: 初始点云数据 (N x 3 的 numpy 数组)
    :param T: SE3 变换矩阵列表
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
    T_overall = SE3(np.eye(4))
    T_list = []
    for i in range(len(T)):
        ax.cla()  # 清除当前图形内容

        # 更新坐标轴标签
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 2)
        ax.set_zlim(-1, 1)


        # 变换后的点云
        T_overall = T_overall *T[i]
        T_list.append(T_overall)
        
        transformed_points = T_overall.t
        centroid  = transformed_points

        trajectory.append(centroid)

    # Apply Gaussian smoothing to the trajectory



    smoothed_trajectory = np.array(trajectory)
     


    for i in range(1, len(smoothed_trajectory)):
        ax.cla()  # 清除当前图形内容
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 2)
        ax.set_zlim(-1, 1)

        ax.plot(smoothed_trajectory[:i, 0], smoothed_trajectory[:i, 1], smoothed_trajectory[:i, 2], c='r', label="Trajectory")
        ax.scatter(smoothed_trajectory[i, 0], smoothed_trajectory[i, 1], smoothed_trajectory[i, 2], c='b', s=5, label="Point Cloud")
        
        ax.legend()
        plt.pause(.1)  # 每帧暂停 1 秒
        
    plt.show()
    plt.close()






    