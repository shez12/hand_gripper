import rospy
from sensor_msgs.msg import CameraInfo
import numpy as np
import json

config_path = "../config/camera_intrinsics.json"

def camera_info_callback(msg, json_filename=config_path):
    # Extract camera matrix K from the message
    K_flat = msg.K
    # Reshape K to a 3x3 matrix
    K = np.reshape(K_flat, (3, 3))
    
    # Extract intrinsic parameters from K
    f_x = K[0, 0]  # Focal length in x direction
    f_y = K[1, 1]  # Focal length in y direction
    s = K[0, 1]    # Skew factor
    c_x = K[0, 2]  # Principal point in x direction
    c_y = K[1, 2]  # Principal point in y direction
    
    # Print the intrinsic parameters
    print("Intrinsic Parameters:")
    print("Focal Length (f_x):", f_x)
    print("Focal Length (f_y):", f_y)
    print("Skew Factor (s):", s)
    print("Principal Point (c_x, c_y):", c_x, c_y)
    
    # Create a dictionary to store the intrinsic parameters
    intrinsic_params = {
        "fx": f_x,
        "fy": f_y,
        "s": s,
        "cx": c_x,
        "cy": c_y
    }
    
    # Save the intrinsic parameters to a JSON file
    with open(json_filename, "w") as json_file:
        json.dump(intrinsic_params, json_file, indent=4)
    
    # Shutdown the node after receiving the camera matrix
    rospy.signal_shutdown("Camera matrix received")

def compute_intrinsic_parameters():
    rospy.init_node('camera_info_subscriber')
    # Subscribe to the camera info topic
    rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, camera_info_callback)
    # Spin ROS
    rospy.spin()


def load_intrinsics(json_filename=config_path):
    with open(json_filename, "r") as file:
        intrinsic_params = json.load(file)
    return intrinsic_params

if __name__ == '__main__':
    try:
        compute_intrinsic_parameters()
    except rospy.ROSInterruptException:
        pass
