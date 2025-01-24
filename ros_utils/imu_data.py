#!/usr/bin/env python
import rospy
import time
from geometry_msgs.msg import Quaternion
import pandas as pd

class QuaternionListener:
    def __init__(self):
        # 初始化四元数数据存储
        self.current_quaternion = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'w': 0.0
        }
        
        # 创建订阅者
        self.subscriber = rospy.Subscriber('imu_quaternion', Quaternion, self.quaternion_callback)

    def quaternion_callback(self, msg):
        # 更新当前四元数数据
        self.current_quaternion['x'] = msg.x
        self.current_quaternion['y'] = msg.y
        self.current_quaternion['z'] = msg.z
        self.current_quaternion['w'] = msg.w
        

    def get_current_quaternion(self):
        """获取最新的四元数数据"""
        return self.current_quaternion













if __name__ == '__main__':

    listener = QuaternionListener()
    while True:
        print(listener.get_current_quaternion())