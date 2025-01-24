#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from omni_msgs.msg import OmniButtonEvent
from geometry_msgs.msg import PoseStamped

class MyOmni:
    def __init__(self):
        topic = '/phantom/phantom/joint_states'  # Define the topic for Omni joint states
        self.gray_button = 0
        self.white_button = 0
        self.gray_button_flag = False 
        self.white_button_flag = False

        # Create subscribers for the button and joint states
        rospy.Subscriber(topic, JointState, self.subscriber_callback)
        rospy.Subscriber('/phantom/phantom/button', OmniButtonEvent, self.button_callback)
        rospy.Subscriber('/phantom/phantom/pose', PoseStamped, self.pose_callback)
        rospy.sleep(2)

    def button_callback(self, data):
        # Process the button event
        self.gray_button = data.grey_button
        self.white_button = data.white_button

        if data.grey_button > 0:
            rospy.loginfo("Gray Button pressed!")
            self.gray_button_flag = not self.gray_button_flag  # Toggle the gray button flag
        
        if data.white_button > 0:
            rospy.loginfo("White Button pressed!")
            self.white_button_flag = not self.white_button_flag

    def subscriber_callback(self, data):
        self.joints = np.array(data.position)

    def pose_callback(self, data):
        pos = data.pose.position
        ori = data.pose.orientation
        self.pose = np.array([pos.x, pos.y, pos.z,  ori.x, ori.y, ori.z, ori.w,])

if __name__ == '__main__':
    try:
        rospy.init_node('phantom_omni_joint_echo')
        my_omni = MyOmni()  # Initialize the myOmni object
        rospy.sleep(1)
        while not rospy.is_shutdown():
            # Print joint states while recording is active
            omni_joints = my_omni.joints
            omni_pose = my_omni.pose
            print("Omni Joints:", omni_joints)
            print('Omni Pose', omni_pose)
            rospy.sleep(0.1)  # Control the loop rate

    except rospy.ROSInterruptException:
        pass
