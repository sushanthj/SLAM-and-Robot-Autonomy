import socket
import time

import numpy as np
import rospy
from autolab_core import RigidTransform
from frankapy import FrankaArm
from geometry_msgs.msg import Pose
from move_arm import relative_translate_end_effector, rotate_end_effector
from utils import average_rigid_transforms, get_calib_file_path

CONFIG = {
    "calib_block_pre_inserted": True,
    "num_calib_snapshots": 10,
}


def main(fa):
    # Reset joints
    fa.reset_joints()

    if not CONFIG["calib_block_pre_inserted"]:
        # Open gripper to insert calibration block with aruco marker.
        fa.open_gripperrobot_base()
        # You have 3 seconds to insert the calibration block.
        time.sleep(2)
        # Close gripper to hold calibration block.
        fa.close_gripper()

    # Move gripper to a pose with good visibility in the camera's FOV.
    rot = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    relative_translate_end_effector(fa, x_offset=0.25, duration=3.0)
    rotate_end_effector(fa, rot, duration=3.0)

    # Get end effector pose from franka arm.
    end_effector_pose = fa.get_pose()
    # Offset from end effector pose to the center of the aruco marker on the calibration block.
    offset_pose = RigidTransform(
        translation=np.array([0.0425, 0, -0.01]),
        rotation=np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
        from_frame="aruco_pose",
        to_frame="franka_tool",
    )

    # Capture marker pose at multiple timestamps and average these to reduce effect of outliers.
    camera_global_snapshots = []
    for i in range(CONFIG["num_calib_snapshots"]):
        aruco_pose = rospy.wait_for_message(
            "/aruco_simple/pose", Pose, timeout=5
        )
        aruco_rt_pose = RigidTransform.from_pose_msg(
            aruco_pose, from_frame="aruco_pose"
        )
        camera_global = (
            end_effector_pose * offset_pose * aruco_rt_pose.inverse()
        )
        print("camera_global: {}".format(camera_global))
        camera_global_snapshots.append(camera_global)

    avg_camera_global = average_rigid_transforms(camera_global_snapshots)
    print("avg_camera_global: {}".format(avg_camera_global))

    # Calibration usage:
    # aruco_global = camera_global*aruco_rt_pose

    # camera pose in global (robot_base frame)
    avg_camera_global.save(get_calib_file_path())


if __name__ == "__main__":
    fa = FrankaArm()
    main(fa)
