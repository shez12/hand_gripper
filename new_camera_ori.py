#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Imu
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
gravity = None
class IMUSubscriber:
    def __init__(self):
        # 存储最新的加速度数据
        self.current_accel = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
        # 创建订阅者
        self.sub = rospy.Subscriber('camera1/accel/sample', Imu, self.imu_callback)
        
        rospy.loginfo("IMU Subscriber initialized")

    def imu_callback(self, msg):
        """IMU消息回调函数"""
        # 更新当前加速度数据
        self.current_accel['x'] = msg.linear_acceleration.x
        self.current_accel['y'] = msg.linear_acceleration.y
        self.current_accel['z'] = msg.linear_acceleration.z
        
        # 打印加速度数据
        # print(f"Linear Acceleration - x: {self.current_accel['x']:.4f}, "
        #       f"y: {self.current_accel['y']:.4f}, "
        #       f"z: {self.current_accel['z']:.4f}")

    def get_current_acceleration(self):
        """获取当前加速度数据"""
        return self.current_accel



def filter_gravity(accel_data, alpha=0.9):
    """低通滤波器提取重力方向"""
    global gravity
    accel_array = np.array([accel_data['x'], accel_data['y'], accel_data['z']])
    if gravity is None:
        gravity = accel_array
    else:
        gravity = alpha * gravity + (1 - alpha) * accel_array
    return gravity / np.linalg.norm(gravity)
  
def compute_rotation_matrix(gravity_vector):
    """计算旋转矩阵，将相机坐标系对齐到重力坐标系"""
    # 方向
    z_axis = gravity_vector
    x_axis = np.cross([0, 0, 1], z_axis)
    if np.linalg.norm(x_axis) < 1e-6:  # 处理特殊情况
        x_axis = [1, 0, 0]
    else:
        x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.vstack([x_axis, y_axis, z_axis]).T

def plot_coordinate_system(ax, origin, rotation_matrix, label, color):
    """绘制3D坐标系"""
    axes_length = 0.2  # 坐标轴长度
    x_axis = rotation_matrix[:, 0] * axes_length
    y_axis = rotation_matrix[:, 1] * axes_length
    z_axis = rotation_matrix[:, 2] * axes_length

    ax.quiver(*origin, *x_axis, color=color[0], label=f"{label} X-axis", arrow_length_ratio=0.1)
    ax.quiver(*origin, *y_axis, color=color[1], label=f"{label} Y-axis", arrow_length_ratio=0.1)
    ax.quiver(*origin, *z_axis, color=color[2], label=f"{label} Z-axis", arrow_length_ratio=0.1)







if __name__ == '__main__':
    rospy.init_node('imu_subscriber', anonymous=True)
    imu_subscriber = IMUSubscriber()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    while True:

                # 提取加速度计数据
            accel_data = imu_subscriber.get_current_acceleration()
            gravity_vector = filter_gravity(accel_data)

            # 计算重力坐标系旋转矩阵
            rotation_matrix = compute_rotation_matrix(gravity_vector)

            # 清除旧的绘图
            ax.cla()
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # 绘制重力坐标系
            plot_coordinate_system(ax, [0, 0, 0], np.eye(3), "Gravity", ['r', 'g', 'b'])
            
            # 绘制相机坐标系
            plot_coordinate_system(ax, [0.5, 0.5, 0.5], rotation_matrix, "Camera", ['r', 'g', 'b'])

            # 显示图例
            ax.legend()

            # 更新绘图
            plt.draw()
            plt.pause(0.1)
