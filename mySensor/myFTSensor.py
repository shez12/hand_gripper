import rospy
from geometry_msgs.msg import WrenchStamped
import numpy as np

def array2force(array):
    # Convert a numpy array to a force message
    force_msg = WrenchStamped()
    force_msg.wrench.force.x = array[0]
    force_msg.wrench.force.y = array[1]
    force_msg.wrench.force.z = array[2]
    return force_msg

def force2array(force_msg):
    # Convert a force message to a numpy array
    force_array = np.array([
        force_msg.x,
        force_msg.y,
        force_msg.z
    ])
    return force_array

class MyFTSensor:
    def __init__(self, topic='/sunrise/force', omni_flag=False):
        self.force_subscriber = rospy.Subscriber(topic, WrenchStamped, self.subscriber_callback)
        self.omni_flag = omni_flag

        if omni_flag:
            self.force_publisher = rospy.Publisher('/phantom/phantom/force_feedback', WrenchStamped, queue_size=10)

    def subscriber_callback(self, data):
        self.force = force2array(data.wrench.force)
        self.torque = force2array(data.wrench.torque)
        if self.omni_flag:
            # Apply a factor (0.1 in this case)
            modified_force = self.force  # Apply the factor
            modified_force[2] = -modified_force[2]
            # Publish the modified force message
            self.force_publisher.publish(array2force(modified_force))

    def get_FT(self, sensor, take_torque):
        if take_torque:
            return np.concatenate((self.force, self.torque))
        else:
            return self.force
        
    
if __name__ == "__main__":
    rospy.init_node('my_ft_sensor_test_node')
    FTSensor = MyFTSensor(omni_flag=False)
    rospy.sleep(1)  # Adjust the time as needed

    while not rospy.is_shutdown():
        print(f"force{FTSensor.force}, torque{FTSensor.torque}", )
        rospy.sleep(0.5)