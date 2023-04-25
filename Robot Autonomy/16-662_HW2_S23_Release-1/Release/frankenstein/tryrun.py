import numpy as np
from autolab_core import RigidTransform
# from pyrsistent import T
from frankapy import FrankaArm


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

def move_joints(fa):
    fa.goto_joints([0.0, -0.7, 0.0, -2.15, 0.0, 1.57, 0.7])
    fa.close_gripper()
    fa.open_gripper()
    fa.goto_gripper(0.03)

def move_end_effector(fa, rot, trans, duration=None, use_impedance=False, cartesian_impedances=None):
    des_pose= RigidTransform(rotation=rot,
                             translation=np.array(trans),
                             from_frame='franka_tool', to_frame='world')
    fa.goto_pose(des_pose, duration=duration, use_impedance=use_impedance, cartesian_impedances=cartesian_impedances)
    # fa.goto_pose(des_pose, use_impedance=False)

def main():
    # Config
    rot = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]])
    pickup_trans = np.array([0.3, -0.2, 0.00])
    end_trans = np.array([0.3, -0.2, 0.4])
    duration=10.0
    use_impedance = False
    cartesian_impedances = [3000, 3000, 200, 300, 300, 300]

    # Execute
    print_state(fa)
    # move_joints(fa)

    fa.open_gripper()
    move_end_effector(fa,
                      rot=rot,    fa.reset_joints()
ance=use_impedance,
                      cartesian_impedances=cartesian_impedances)
    
    fa.goto_gripper(0.02)
    move_end_effector(fa,
                      rot=rot,
                      trans=end_trans,
                      duration=duration,
                      use_impedance=use_impedance,
                      cartesian_impedances=cartesian_impedances)



if __name__ == "__main__" :
    fa = FrankaArm()
    # fa.open_gripper()
    fa.reset_joints()
    main()