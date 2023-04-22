import numpy as np
import time
import rospy
from geometry_msgs.msg import Pose
from autolab_core import RigidTransform
from frankapy import FrankaArm

def main(fa):
    # fa.reset_joints()
    # fa.open_gripper()
    # time.sleep(2)
    # fa.close_gripper()
    # fa.run_guide_mode(20, block=False)
    end_effector_pose = fa.get_pose()

    offset_pose = RigidTransform(translation=np.array([0.0425, 0, -0.01]),
                                 rotation=np.array([[0,0,1],[-1,0,0],[0,-1,0]]),
                                 from_frame="aruco_pose", to_frame="franka_tool")
    
    aruco_pose = rospy.wait_for_message('/aruco_simple/pose', Pose, timeout=5)
    aruco_rt_pose = RigidTransform.from_pose_msg(aruco_pose, from_frame="aruco_pose")

    camera_global = end_effector_pose*offset_pose*aruco_rt_pose.inverse()

    aruco_global = camera_global*aruco_rt_pose

    # camera pose in global (robot_base frame)
    camera_global.save("kinect_transform.tf")

if __name__ == "__main__" :
    fa = FrankaArm()
    # fa.reset_joints()
    fa.open_gripper()
    # main(fa)