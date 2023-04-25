import time
import copy

import numpy as np
import rospy
from autolab_core import RigidTransform
from frankapy import FrankaArm
from geometry_msgs.msg import Pose
from IPython import embed
from move_arm import relative_translate_end_effector, rotate_end_effector
from utils import average_rigid_transforms, get_calib_file_path

CONFIG = {
    "pallet_marker_pose_topic": "/aruco_simple/pose",
    "camera_calib_file": get_calib_file_path(),
    "num_marker_snapshots": 1,
}


def get_pregrasp_offset_in_pallet_frame():
    print("Getting pregrasp offset in pallet frame.")
    # Offset to aim for the grasping positin in the 3d printed grip.
    grasp_translational_offset_in_pallet_frame = RigidTransform(
        translation=np.array([0, -0.146, 0.00]),
        # rotation=np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
        rotation=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        from_frame="grasp_trans_offset",
        # from_frame="grasp",
        to_frame="pallet",
    )
    # Then rotate to align the end effector frame with the aruco marker frame convention.
    grasp_rotational_offset_in_pallet_frame = RigidTransform(
        translation=np.array([0, 0, 0]),
        rotation=np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
        from_frame="grasp",
        to_frame="grasp_trans_offset",
    )
    grasp_offset_in_pallet_frame = (
        grasp_translational_offset_in_pallet_frame
        * grasp_rotational_offset_in_pallet_frame
    )

    # grasp_offset_in_pallet_frame = grasp_translational_offset_in_pallet_frame

    return grasp_offset_in_pallet_frame


def get_grasp_offset_in_pallet_frame():
    print("Getting grasp offset in pallet frame.")
    # Offset to aim for the grasping positin in the 3d printed grip.
    grasp_translational_offset_in_pallet_frame = RigidTransform(
        translation=np.array([0, -0.106, -0.019]),
        # rotation=np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
        rotation=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        from_frame="grasp_trans_offset",
        # from_frame="grasp",
        to_frame="pallet",
    )
    # Then rotate to align the end effector frame with the aruco marker frame convention.
    grasp_rotational_offset_in_pallet_frame = RigidTransform(
        translation=np.array([0, 0, 0]),
        rotation=np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
        from_frame="grasp",
        to_frame="grasp_trans_offset",
    )
    grasp_offset_in_pallet_frame = (
        grasp_translational_offset_in_pallet_frame
        * grasp_rotational_offset_in_pallet_frame
    )

    # grasp_offset_in_pallet_frame = grasp_translational_offset_in_pallet_frame

    return grasp_offset_in_pallet_frame


def detect_pallet_pose_in_camera_frame():
    print("Detecting pallet pose in camera frame.")
    pallet_pose_in_camera_frame_snapshots = []
    for i in range(CONFIG["num_marker_snapshots"]):
        aruco_pose_msg = rospy.wait_for_message(
            CONFIG["pallet_marker_pose_topic"], Pose, timeout=5
        )
        pallet_pose_in_camera_frame = RigidTransform.from_pose_msg(
            aruco_pose_msg, from_frame="pallet", to_frame="camera"
        )
        pallet_pose_in_camera_frame_snapshots.append(
            pallet_pose_in_camera_frame
        )
    return average_rigid_transforms(pallet_pose_in_camera_frame_snapshots)


def get_pallet_pose_in_robot_base_frame():
    print("Getting pallet pose in robot base frame.")
    camera_pose_in_robot_base_frame = RigidTransform.load(
        CONFIG["camera_calib_file"]
    )
    print(f"camera_pose_in_robot_base_frame: {camera_pose_in_robot_base_frame}")
    print(
        f"camera_pose_in_robot_base_frame.euler_angles: {camera_pose_in_robot_base_frame.euler_angles}"
    )
    pallet_pose_in_camera_frame = detect_pallet_pose_in_camera_frame()
    print(f"pallet_pose_in_camera_frame: {pallet_pose_in_camera_frame}")
    print(
        f"pallet_pose_in_camera_frame.euler_angles: {pallet_pose_in_camera_frame.euler_angles}"
    )
    pallet_pose_in_robot_base_frame = (
        camera_pose_in_robot_base_frame * pallet_pose_in_camera_frame
    )
    print(f"pallet_pose_in_robot_base_frame: {pallet_pose_in_robot_base_frame}")
    print(
        f"pallet_pose_in_robot_base_frame.euler_angles: {pallet_pose_in_robot_base_frame.euler_angles}"
    )
    return pallet_pose_in_robot_base_frame


def overwrite_frame_names_for_franka_api(tx):
    tx_copy = copy.deepcopy(tx)
    tx_copy.from_frame = "franka_tool"
    tx_copy.to_frame = "world"

    new_translation = tx.translation
    new_translation[2] = 0.025
    tx_copy.translation = new_translation
    return tx_copy


def main(fa):

    fa.reset_joints()
    from_frame = "franka_tool"
    to_frame = "world"
    temp = RigidTransform(
        translation=np.array([0.45, 0, 0.005]),
        rotation=np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        from_frame=from_frame,
        to_frame=to_frame,
    )
    fa.goto_pose(temp, duration=10, use_impedance=True)

    # # Reset joints
    # fa.reset_joints()

    # # Open gripper to prepare for pickup.
    # fa.open_gripper()

    # pallet_pose_in_robot_base_frame = get_pallet_pose_in_robot_base_frame()
    # pregrasp_offset_in_pallet_frame = get_pregrasp_offset_in_pallet_frame()
    # pregrasp_pose_in_robot_base_frame = (
    #     pallet_pose_in_robot_base_frame * pregrasp_offset_in_pallet_frame
    # )
    # grasp_offset_in_pallet_frame = get_grasp_offset_in_pallet_frame()
    # grasp_pose_in_robot_base_frame = (
    #     pallet_pose_in_robot_base_frame * grasp_offset_in_pallet_frame
    # )
    # fa.goto_pose(
    #     overwrite_frame_names_for_franka_api(pregrasp_pose_in_robot_base_frame),
    #     duration=10,
    #     use_impedance=True,
    # )
    # fa.goto_pose(
    #     overwrite_frame_names_for_franka_api(grasp_pose_in_robot_base_frame),
    #     duration=3,
    #     use_impedance=True,
    # )

    # # Open gripper to prepare for pickup.
    # fa.goto_gripper(0.052)

    # # Reset joints
    # fa.reset_joints()

    # time.sleep(5)
    # fa.open_gripper()


if __name__ == "__main__":
    fa = FrankaArm()
    main(fa)
