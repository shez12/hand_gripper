import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import os

class MyImageSaver:
    def __init__(self, cameraNS="camera"):
        self.bridge = CvBridge()
        self.cameraNS = cameraNS
        self.rgb_image = None
        self.depth_image = None
        self.rgb_sub = rospy.Subscriber(f'/{cameraNS}/color/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber(f'/{cameraNS}/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        self.count = 0
        self.folder_path = "data/images"+time.strftime("-%Y%m%d-%H%M%S")
        #wait to receive first image
        while self.rgb_image is None:
            rospy.sleep(0.1)
            print('waiting for first image')
        print(f'init MyImageSaver at {self.folder_path}')

    def rgb_callback(self, data):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("Error saving RGB image: %s", str(e))

    def depth_callback(self, data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr("Error saving depth image: %s", str(e))

    def generate_timestamp(self):
        return time.strftime("%Y%m%d-%H%M%S")

    def save_image(self, image, prefix):
        os.makedirs(self.folder_path, exist_ok=True)
        prefix = self.cameraNS+'_'+prefix
        image_filename = os.path.join(self.folder_path,f"{prefix}_{self.count}.png")
        cv2.imwrite(image_filename, image)
        print(f"write to {image_filename}")
    

    def record(self):
        self.save_image(self.rgb_image, "rgb")
        self.save_image(self.depth_image, 'depth')
        self.count += 1

    def spin(self):
        rospy.spin()

    def get_frames(self):
        return self.rgb_image, self.depth_image

if __name__ == '__main__':
    rospy.init_node('image_saver')
    image_saver = MyImageSaver(cameraNS='camera')
    framedelay = 1000//20

    # Example usage: Save RGB and depth images
    while not rospy.is_shutdown():
        frame = image_saver.rgb_image
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(framedelay) & 0xFF 
        if key==ord('s'):
            image_saver.record()  # Save images
        elif key==ord('q'):
            break
