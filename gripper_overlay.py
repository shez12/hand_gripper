import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np




def get_gripper_scale(distance):
    #each set is 3 numbers get the distance between the two points(4 and 8)

    gripper_scale=1000*(distance)/0.1
    gripper_scale = max(0, min(gripper_scale, 1000))  # Clamp gripper_scale between 0 and 1000
    gripper_scale = (gripper_scale // 200) * 200     # Round down to the nearest multiple of 200
    return gripper_scale


class GripperOverlay:
    def __init__(self, ax):
        self.ax = ax
        self.gripper_lines = []
        self.grip_scale = 0  # 初始化为完全并拢
        self.max_spread = 0.1  # 最大分离距离 10 cm
        self.gripper_width = 0.02  # 2 cm
        self.gripper_length = 0.1  # 6 cm
        self.gripper_height = 0.02  # 2 cm
        self.base_x = 0.5  # 50 cm
        self.base_y = 0.5  # 50 cm
        self.base_z = 0.5  # 50 cm
        self.column_depth = 0.02  # 2 cm

        self._draw_gripper_parts()

    def set_grip_scale(self, scale):
        """
        设置夹爪的张开程度。
        :param scale: int, 取值范围为 0 - 1000
        """
        if scale < 0:
            scale = 0
        elif scale > 1000:
            scale = 1000
        self.grip_scale = scale
        self._draw_gripper_parts()

    def _create_cuboid_vertices(self, x, y, z, width, height, depth):
        """
        创建长方体的8个顶点。
        """
        return [
            [x, y, z],
            [x + width, y, z],
            [x + width, y + height, z],
            [x, y + height, z],
            [x, y, z + depth],
            [x + width, y, z + depth],
            [x + width, y + height, z + depth],
            [x, y + height, z + depth]
        ]

    def _draw_lines(self, vertices, color='blue'):
        """
        根据顶点绘制长方体的边缘线。
        """
        # 定义长方体的12条边
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
            (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
            (0, 4), (1, 5), (2, 6), (3, 7)   # 垂直边
        ]

        for edge in edges:
            start, end = edge
            line, = self.ax.plot(
                [vertices[start][0], vertices[end][0]],
                [vertices[start][1], vertices[end][1]],
                [vertices[start][2], vertices[end][2]],
                color=color
            )
            self.gripper_lines.append(line)

    def _clear_gripper(self):
        """
        清除之前绘制的夹爪和柱子的线条。
        """
        for line in self.gripper_lines:
            line.remove()
        self.gripper_lines = []

    def _draw_gripper_parts(self):
        # 清除之前的夹爪和柱子
        self._clear_gripper()

        # 计算分离偏移量
        spread_ratio = self.grip_scale / 1000.0  # 0.0 到 1.0
        spread_offset = spread_ratio * self.max_spread  # 最大分离距离 0.1 m

        print(f"Grip Scale: {self.grip_scale}, Spread Offset: {spread_offset}")

        # 左右夹爪沿 x 轴偏移
        left_gripper_x = self.base_x - spread_offset / 2
        right_gripper_x = self.base_x + self.gripper_width + spread_offset / 2

        print(f"Left Gripper X: {left_gripper_x}, Right Gripper X: {right_gripper_x}")

        # 创建夹爪顶点
        left_gripper = self._create_cuboid_vertices(
            left_gripper_x,
            self.base_y,
            self.base_z,
            self.gripper_width,
            self.gripper_height,
            self.gripper_length
        )
        right_gripper = self._create_cuboid_vertices(
            right_gripper_x,
            self.base_y,
            self.base_z,
            self.gripper_width,
            self.gripper_height,
            self.gripper_length
        )

        # 绘制夹爪边缘
        self._draw_lines(left_gripper, color='blue')
        self._draw_lines(right_gripper, color='blue')

        # 计算水平柱子的长度
        column_length = self.gripper_width * 2 + spread_offset  # 根据分离距离调整

        print(f"Column Length: {column_length}")

        # 计算柱子的位置（位于夹爪的底部中间，水平连接两端）
        column_x = self.base_x - spread_offset / 2  # 起始 x 位置，考虑当前分离
        column_y = self.base_y  # y 位置，与夹爪底面对齐
        column_z = self.base_z - (self.column_depth / 2)  # z 位置，确保柱子居中

        print(f"Column X: {column_x}, Column Y: {column_y}, Column Z: {column_z}")

        # 创建柱子顶点
        horizontal_column = self._create_cuboid_vertices(
            column_x,
            column_y,
            column_z,
            column_length,
            self.gripper_height,
            self.column_depth
        )

        # 绘制柱子边缘
        self._draw_lines(horizontal_column, color='blue')

    def update_plot_limits(self):
        """
        更新坐标轴的显示范围，以适应夹爪和柱子的动态变化。
        """
        self.ax.auto_scale_xyz(
            [self.base_x - self.max_spread, self.base_x + self.gripper_width + self.max_spread],
            [self.base_y - 0.05, self.base_y + self.gripper_height + 0.05],
            [self.base_z - self.gripper_length - 0.05, self.base_z + 0.05]
        )

if __name__ == "__main__":
    # 创建 Matplotlib 图形和 3D 坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置初始坐标轴范围 (单位：米)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    # 实例化 GripperOverlay 并传入 ax
    gripper = GripperOverlay(ax)

    # 更新坐标轴范围以适应初始位置
    gripper.update_plot_limits()

    # 示例：动态调整 grip_scale
    import time

    try:
        for scale in range(0, 1001, 100):
            gripper.set_grip_scale(scale)
            gripper.update_plot_limits()
            plt.draw()
            plt.pause(0.5)  # 暂停0.5秒观察变化
    except KeyboardInterrupt:
        pass  # 允许用户通过 Ctrl+C 中断循环

    plt.show()
