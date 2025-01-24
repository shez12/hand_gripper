#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
import numpy as np

class RTABMapPoseListener:
    def __init__(self, verbose=False):
        self.position = None
        self.orientation = None
        self.verbose = verbose
        rospy.Subscriber('/rtabmap/odom', Odometry, self._odom_callback)
        rospy.sleep(0.1)
        
    def _odom_callback(self, msg):
        # Store the pose estimation
        self.position = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        self.orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, 
                            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        if self.verbose:
            print("Pose Estimation:")
            print("Position (x, y, z):", self.position)
            print("Orientation (x, y, z, w):", self.orientation)

    def get_pose(self):
        return np.array(self.position + self.orientation)


if __name__ == '__main__':
    try:
        rospy.init_node('rtabmap_pose_listener', anonymous=True)

        # Non-verbose mode
        rtabmap_pose_listener = RTABMapPoseListener(verbose=False)
        print('Testing RTABMapPoseListener')
        for i in range(10):
            print(rtabmap_pose_listener.get_pose())
            
    except rospy.ROSInterruptException:
        pass
