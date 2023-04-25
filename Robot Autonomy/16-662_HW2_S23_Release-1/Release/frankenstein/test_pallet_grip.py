import numpy as np
from autolab_core import RigidTransform
# from pyrsistent import T
from frankapy import FrankaArm
import copy

OPEN_GRIPPER_WIDTH_M = 0.048
CLOSE_GRIPPER_WIDTH_M = 0.0

def lock_pallet():
    fa.goto_gripper(OPEN_GRIPPER_WIDTH_M)

def unlock_pallet():
    fa.goto_gripper(CLOSE_GRIPPER_WIDTH_M)

def print_state(fa):
    T_ee_world = fa.get_pose()
    print('Translation: {}'.format(T_ee_world.translation))
    print('Rotation: {}'.format(T_ee_world.quaternion))

    joints = fa.get_joints()
    print('Joints: {}'.format(joints))

    gripper_width = fa.get_gripper_width()
    print('Gripper width: {}'.format(gripper_width))

    gripper_width = fa.get_gripper_width()
    print('Gripper width: {}'.format(gripper_width))

    force_torque = fa.get_ee_force_torque()
    print('Forces and Torques: {}'.format(force_torque))

def get_current_translation(fa):
    return fa.get_pose().translation

def get_current_rotation(fa):
    return fa.get_pose().rotation

# def move_joints(fa):
#     fa.goto_joints([0.0, -0.7, 0.0, -2.15, 0.0, 1.57, 0.7])
#     fa.close_gripper()
#     fa.open_gripper()
#     fa.goto_gripper(0.03)

def move_end_effector(fa, rot, trans, duration=None, use_impedance=False, cartesian_impedances=None, verbose=True):
    
    if verbose:
        print("\nState before movement:")
        print_state(fa)

    des_pose= RigidTransform(rotation=rot,
                             translation=trans,
                             from_frame='franka_tool', to_frame='world')
    fa.goto_pose(des_pose, duration=duration, use_impedance=use_impedance, cartesian_impedances=cartesian_impedances)

    if verbose:
        print("\nState after movement:")
        print_state(fa)

def relative_translate_end_effector(fa, x_offset=0.0, y_offset=0.0, z_offset=0.0, duration=None, use_impedance=False, cartesian_impedances=None, verbose=False):
    new_trans = copy.deepcopy(get_current_translation(fa))
    new_trans[0] += x_offset
    new_trans[1] += y_offset
    new_trans[2] += z_offset

    move_end_effector(fa,
                      rot=get_current_rotation(fa),
                      trans=new_trans,
                      duration=duration,
                      use_impedance=use_impedance,
                      cartesian_impedances=cartesian_impedances,
                      verbose=verbose)

def main():
    # First reset joints
    fa.reset_joints()


    # Rotation for pickup from y to -y direction
    rot = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0]])

    # Move end effector down to the same z as the pickup slot in pallet
    new_trans = get_current_translation(fa)
    new_trans[0] +=0.05
    new_trans[2] = 0.1
    move_end_effector(fa,
                      rot=rot,
                      trans=new_trans,
                      duration=5,
                      use_impedance=True,
                      verbose=False)
    
    # Insert end effector into pallet handle
    relative_translate_end_effector(fa,
                      y_offset=-0.06,
                      duration=2,
                      use_impedance=True,
                      verbose=False)
    relative_translate_end_effector(fa,
                      z_offset=0.02,
                      duration=1,
                      use_impedance=True,
                      verbose=False)
    
    lock_pallet()

    # Lift up pallet
    relative_translate_end_effector(fa,
                      z_offset=0.2,
                      duration=3,
                      use_impedance=True,
                      verbose=False)

    # Lower pallet
    relative_translate_end_effector(fa,
                      z_offset=-0.2,
                      duration=3,
                      use_impedance=True,
                      verbose=False)

    unlock_pallet()

    # Remove end effector from pallet handle
    relative_translate_end_effector(fa,
                      z_offset=-0.02,
                      duration=1,
                      use_impedance=True,
                      verbose=False)
    relative_translate_end_effector(fa,
                      y_offset=0.06,
                      duration=2,
                      use_impedance=True,
                      verbose=False)
    
    # First reset joints
    fa.reset_joints()

    lock_pallet()




if __name__ == "__main__" :
    fa = FrankaArm()
    main()
