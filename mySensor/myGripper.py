#!/usr/bin/env python
import rospy
from dh_gripper_msgs.msg import GripperCtrl, GripperState

class MyGripper:
    def __init__(self, topic_name='/gripper/ctrl', state_topic_name='/gripper/states'):
        # Create a publisher on the specified topic with a queue size of 10
        self.pub = rospy.Publisher(topic_name, GripperCtrl, queue_size=10)

        # Create a subscriber on the gripper state topic
        self.sub = rospy.Subscriber(state_topic_name, GripperState, self.gripper_state_callback)

    def gripper_state_callback(self, msg):
        # This callback function is called every time a new message is received on the gripper state topic
        # rospy.loginfo(f"Gripper position: {msg.position}")
        self.current_position = msg.position

    def set_gripper(self, position, force, speed=50.0, wait=False):
        # Create a GripperCtrl message
        grip_msg = GripperCtrl()
        grip_msg.initialize = False
        grip_msg.position = position
        grip_msg.force = force
        grip_msg.speed = speed
        # Publish the message
        self.pub.publish(grip_msg)


if __name__ == '__main__':
    rospy.init_node('my_gripper_node', anonymous=True)
    # Create an instance of MyGripper
    gripper = MyGripper()

    # Run the gripper control loop
    while not rospy.is_shutdown():
        gripper.set_gripper(1000.0, 20.0)  # Example values

        for pos in range(100, 1000, 200):
            gripper.set_gripper(position=pos, force=2)
            rospy.sleep(1)
            print('current pos', gripper.current_position)
            