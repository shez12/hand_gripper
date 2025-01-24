from spatialmath import SE3,SO3


def limit_rotation(SE3_poses_list):
    '''
    args:
    SE3_poses_list: list of SE3 objects,each SE3(i) means transformation from 0 to i
    '''

    #if rotation angle is to big, then just rotation 3/4
    pre_se3 = SE3.Tx(0)
    #how to define the rotation angle is too big?--> convert to rpy
    for pose_se3 in SE3_poses_list:
        #convert to rpy
        new_se3 = pre_se3.inv()*pose_se3
        rpy_deg = SO3(pose_se3.R).rpy(order='zyx', unit='deg')
        
        # Limit roll angle
        if rpy_deg[0] > 50:
            rpy_deg[0] = 50
        elif rpy_deg[0] < -50:
            rpy_deg[0] = -50
            
        # Limit pitch angle
        if rpy_deg[1] > 50:
            rpy_deg[1] = 50
        elif rpy_deg[1] < -50:
            rpy_deg[1] = -50
            
        # Limit yaw angle
        if rpy_deg[2] > 50:
            rpy_deg[2] = 50
        elif rpy_deg[2] < -50:
            rpy_deg[2] = -50
            
        # Convert back to SE3
        new_R = SO3.RPY(rpy_deg, order='zyx', unit='deg')
        pose_se3.R = new_R.R
        pre_se3 = pose_se3
    return SE3_poses_list















