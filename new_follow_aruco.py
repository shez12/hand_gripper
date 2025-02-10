import cv2
import numpy as np
import json
from spatialmath import SE3
from spatialmath.base import trnorm
import numpy as np

from my_utils.robot_utils import robot_fk
from record_episode import RealEnv
from my_utils.aruco_util import get_marker_pose,set_aruco_dict
from my_kalmen_filter import KalmenFilter


# func 分析frame 
'''
    1. 两个对象
        1.1 一个对象是marker
        1.2 一个对象是camera
'''

class Camera2World:
    def __init__(self,camera_name):
        """
        Initialize Camera2World transformer.
        
        Args:
            camera_name (str): Name identifier for the camera
        """
        self.camera_name = camera_name
        self.camera_pose = None

    def set_camera_pose(self,camera_pose):
        self.camera_pose = camera_pose
    
    def convert_camera_pose_to_world_pose(self,marker_pose):
        if self.camera_pose is None:
            raise ValueError("Camera pose must be set before converting coordinates")
        return self.camera_pose.inv() * marker_pose


# 记录更新位置
'''
    Map: 
        key: marker id
        value: marker pose
'''


def same_position(pose1, pose2, angle_threshold=0.1, translation_threshold=0.01):
    """
    Compare two poses to determine if they are effectively the same position.
    
    Args:
        pose1, pose2: Marker poses to compare
        angle_threshold: Maximum angle difference in radians (default: 0.1)
        translation_threshold: Maximum translation difference in meters (default: 0.01)
    
    Returns:
        bool: True if poses are considered same, False otherwise
    """
    if pose1 is None or pose2 is None:
        return False
        
    R_diff = np.dot(pose1.R.T, pose2.R)
    angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    translation_norm = np.linalg.norm(pose1.t - pose2.t)
    return angle < angle_threshold and translation_norm < translation_threshold

class PositionMap:
    def __init__(self, id_list,camera_num):
        """
        Initialize position tracking for multiple markers.
        
        Args:
            id_list: List of marker IDs to track
        """
        self.position_map = {}
        self.overall_map = []
        
        self.camera_num = camera_num
        self.filter_list = {str(id): KalmenFilter() for id in id_list}
        self.temp_map = {str(id): [] for id in id_list}

    def reset_temp_map(self,marker_id):
        self.temp_map[marker_id] = []

    def filter_pose(self, marker_id, marker_pose):
        """Apply Kalman filter to marker pose."""
        marker_id = str(marker_id)
        if marker_id not in self.filter_list:
            self.filter_list[marker_id] = KalmenFilter()
            return marker_pose
            
        self.filter_list[marker_id].new_markerpose(marker_pose)
        self.filter_list[marker_id].Kalman_Filter()
        return self.filter_list[marker_id].get_pose()

    def update_position(self, marker_id, marker_pose):
        """Update marker position if significantly different from previous position."""
        marker_id = str(marker_id)
        
        if marker_pose is None:
            # 如果marker_pose为None，则将该marker_id的位置设置为None
            # self.position_map[marker_id] = None
            self.temp_map[marker_id].append(None)
            return

        filtered_pose = self.filter_pose(marker_id, marker_pose)
        
        if (marker_id in self.position_map and 
            self.position_map[marker_id] is not None and
            same_position(self.position_map[marker_id], filtered_pose)):
            # 如果marker_pose与之前的位置相同，则不更新位置
            self.temp_map[marker_id].append(self.position_map[marker_id])
            return

        self.temp_map[marker_id].append(filtered_pose)
        print(f"Updated position for marker {marker_id}:")
        print(filtered_pose)

    def combine_temp_map(self,marker_id):
        if len(self.temp_map[marker_id]) != self.camera_num:
            return
        # 去除None
        self.temp_map[marker_id] = [pose for pose in self.temp_map[marker_id] if pose is not None]
        if len(self.temp_map[marker_id]) == 0:
            self.position_map[marker_id] = None
        else:
            self.position_map[marker_id] = self.temp_map[marker_id][0]

        self.reset_temp_map(marker_id)




    def get_position(self, marker_id):
        marker_id  = str(marker_id)
        return self.position_map[marker_id]
    
    def add2overall_map(self):
        self.overall_map.append(self.position_map.copy())





def main():
    camera_name = ["camera1","camera3"]
    with open('marker_info.json', 'r') as f:
        marker_info = json.load(f)
    env = RealEnv(robot_names=['robot1'], camera_names=camera_name)
    id_list = []
    for marker_id in marker_info:
        id_list.append(marker_id)
    position_map = PositionMap(id_list,len(camera_name))
    print("calibrating .............. ")
    res_se3 = auto_regist_camera(0)
    
    try:
        while True:
            obs = env.get_observation()            
            for cam_name, img in obs['images'].items():
                img_copy = img.copy()

                for marker_id in id_list:
                    marker_pose, corner = get_marker_pose(
                                            img_copy, 
                                            marker_size=marker_info[marker_id]["marker_size"],
                                            id=marker_id, 
                                            aruco_dict=set_aruco_dict(marker_info[marker_id]["aruco_dict"])
                                            )
                    
                    # convert camera pose to world pose
                    # ToDo

                    if cam_name == "camera3" and marker_pose is not None:
                        # res_se3 = SE3(trnorm(np.array([
                        #     [-0.7457,    0.2934,   -0.5982,    0.3759],
                        #     [ 0.6658,    0.2919,   -0.6867,    0.7309],
                        #     [-0.02685,  -0.9104,   -0.413,    -0.1444],
                        #     [ 0,         0,         0,         1      ]
                        # ]))) # 事先校准的机械臂底座到相机3的变换
                        marker_pose = res_se3 * marker_pose

                    if cam_name == "camera1" and marker_pose is not None:
                        marker_pose = robot_fk(env.robots['robot1'],marker_pose)

                    # 更新位置
                    position_map.update_position(marker_id, marker_pose)
                    position_map.combine_temp_map(marker_id)
                    # Display the image
                    cv2.imshow(cam_name, cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
                position_map.add2overall_map()
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt
                

                
    except KeyboardInterrupt:
        print("\nExiting program...")
        cv2.destroyAllWindows()

    print(position_map.position_map)





def auto_regist_camera(marker_id_input):
    '''
    camera1:眼在手上（已标定） 
    camera3:固定机位
        env: 环境
            camera1:眼在手上（已标定） 
            robot1:机械臂1
    args:
        maker_id:在公共视野里的aruco marker id
    return 
        camera3:机械臂底座到相机3的变换
    '''
    marker_id_input = str(marker_id_input)
    transformations = SE3([
        [0.03391, -0.01653, 0.9993, 0.03622],
        [0.9994, -0.0091, -0.03407, -0.02485],
        [0.009689, 0.9998, 0.01621, -0.1034],
        [0, 0, 0, 1]
    ])# 相机1手眼标定

    #创建两个env
    env1 = RealEnv(robot_names=['robot1'], camera_names=["camera1"])
    env2 = RealEnv(robot_names=['robot1'], camera_names=["camera3"])

    # 获取marker_info
    with open('marker_info.json', 'r') as f:
        marker_info = json.load(f)

    id_list = [marker_id_input]
    position_map1 = PositionMap(id_list,1)
    position_map2 = PositionMap(id_list,1)
    n = 0
    while n<1000:
        #滤波
        obs1 = env1.get_observation()
        obs2 = env2.get_observation()
        image1_copy = obs1['images']['camera1'].copy()
        image2_copy = obs2['images']['camera3'].copy()
        marker_pose1, corner1 = get_marker_pose(
                                            image1_copy, 
                                            marker_size=marker_info[marker_id_input]["marker_size"],
                                            id=marker_id_input, 
                                            aruco_dict=set_aruco_dict(marker_info[marker_id_input]["aruco_dict"])
                                            )
        marker_pose2, corner2 = get_marker_pose(
                                            image2_copy, 
                                            marker_size=marker_info[marker_id_input]["marker_size"],
                                            id=marker_id_input, 
                                            aruco_dict=set_aruco_dict(marker_info[marker_id_input]["aruco_dict"])
                                            )
        position_map1.update_position(marker_id_input, marker_pose1)
        position_map2.update_position(marker_id_input, marker_pose2)
        position_map1.combine_temp_map(marker_id_input)
        position_map2.combine_temp_map(marker_id_input)
        n += 1
        # cv2.imshow("camera1", cv2.cvtColor(image1_copy, cv2.COLOR_RGB2BGR))
        # cv2.imshow("camera3", cv2.cvtColor(image2_copy, cv2.COLOR_RGB2BGR))

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


    # 计算机械臂底座到相机3的变换
    camera1_pose = position_map1.position_map[marker_id_input]
    camera3_pose = position_map2.position_map[marker_id_input]


    res = camera1_pose * camera3_pose.inv() #相机之间的俄转换
    robot_pose = env2.robots['robot1'].get_pose_se3()
    res = robot_pose * transformations * res

    print("camera1 to camera3",res)
    return res







if __name__ == "__main__":
    main()
    # res = auto_regist_camera(0)
    # print(res)
