import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3

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


def get_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp

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

def draw_movingframe(T,keypoints):
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
    for i in range(20):
        for i in range(1, len(smoothed_trajectory)):
            ax.cla()  # 清除当前图形内容
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_zlabel("Z-axis")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)

            ax.plot(smoothed_trajectory[:i, 0], smoothed_trajectory[:i, 1], smoothed_trajectory[:i, 2], c='r', label="Trajectory")
            ax.scatter(smoothed_trajectory[i, 0], smoothed_trajectory[i, 1], smoothed_trajectory[i, 2], c='b', s=5, label="Point Cloud")
            ax.scatter(keypoints[i][:, 0], keypoints[i][:, 1], keypoints[i][:, 2])
            ax.legend()
            plt.pause(.1)  # 每帧暂停 1 秒
        
    plt.show()
    plt.close()
    