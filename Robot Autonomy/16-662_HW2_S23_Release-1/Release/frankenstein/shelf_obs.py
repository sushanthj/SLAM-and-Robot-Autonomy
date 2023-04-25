from random import sample, seed
from re import A
import time
import pickle
import numpy as np

# import vrep_interface as vpi
import RobotUtil as rt
import Franka
import time

# Mujoco Imports
# import mujoco as mj
# from mujoco import viewer


# Seed the random object
seed(10)

# Open the simulator model from the MJCF file
xml_filepath = "../franka_emika_panda/panda_with_hand_torque.xml"

np.random.seed(0)
deg_to_rad = np.pi / 180.0

# Initialize robot object
mybot = Franka.FrankArm()


def shelf_obs(x_offset, y_offset, z_offset):
    # Add obstacle descriptions into pointsObs and axesObs
    # (bounding boxes for all the obstacles present in the environment (pointObs and envaxes))
    pointsObs = []
    axesObs = []

    # LAB CONFIGURATION FOR FRANKA BELOW-------------------------------------------------------------
    # DO NOT RESET BELOW SETUP
    top_box_lwh = [[0.5, 0.32, 0.13], [0, 0, 0.895]]
    bottom_box_lwh = [[0.5, 0.32, 0.10], [0, 0, 0]]
    left_box_lwh = [[0.03, 0.32, 0.96], [0, 0, 0]]
    right_box_lwh = [[0.03, 0.32, 0.96], [0.485, 0, 0]]
    insert_box_lwh = [[0.5, 0.32, 0.03], [0, 0, 0.21]]
    lwh_list = []
    obstacle_parameters_list = []
    lwh_list.append(top_box_lwh)
    lwh_list.append(bottom_box_lwh)
    lwh_list.append(left_box_lwh)
    lwh_list.append(right_box_lwh)
    lwh_list.append(insert_box_lwh)
    # ctr_list.append(top_box_ctr)
    # ctr_list.append(bottom_box_ctr)
    # ctr_list.append(left_box_ctr)
    # ctr_list.append(right_box_ctr)
    # ctr_list.append(insert_box_ctr)
    # left_reach = [0.40829459, -0.16118822, 0.07832987]
    # right_reach = [0.42463734, 0.15759294, 0.07898102]
    # cuboid box obstacle center coordinate
    ctr_list = []
    obstacle_parameters_list = []
    for lwh, ctr in lwh_list:
        x_ctr = lwh[0] / 2 + x_offset + ctr[0]
        y_ctr = lwh[1] / 2 + y_offset + ctr[1]
        z_ctr = lwh[2] / 2 + z_offset + ctr[2]
        ctrs = [x_ctr] + [y_ctr] + [z_ctr]
        obstacle_parameters = [
            x_ctr,
            y_ctr,
            z_ctr,
            0,
            0,
            0,
            lwh[0],
            lwh[1],
            lwh[2],
        ]
        obstacle_parameters_list.append(obstacle_parameters)
        ctr_list.append(ctrs)
    obstacle_parameters_list.append([0.2, 0, -0.05, 0, 0, 0, 1.2, 1, 0.01])

    # for each entry(sublist) in this list, position(len=3) + orientation(len=3) + cuboid_dim(len3)
    # obstacle_parameters = [[x_ctr, y_ctr, z_ctr, 0, 0, 0, length_box, width_box, height_box],
    #                        [0.15, 0.46, 0.5, 0, 0, 0, 1.2, 0.01, 1.1],
    #                        [0.15, -0.46, 0.5, 0, 0, 0, 1.2, 0.01, 1.1],
    #                        [-0.41, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1],
    #                        [0.75, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1],
    #                        [0.2, 0, 1, 0, 0, 0, 1.2, 1, 0.01],
    #                        [0.2, 0, -0.05, 0, 0, 0, 1.2, 1, 0.01]]

    for para in obstacle_parameters_list:
        xyz = para[:3]
        rpy = para[3:6]
        lwh = para[6:9]
        envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H(rpy, xyz), lwh)
        pointsObs.append(envpoints), axesObs.append(envaxes)
        # mybot.PlotCollisionBlockPoints(q_trial_2, pointsObs)
    return pointsObs, axesObs


if __name__ == "__main__":
    # mybot.reset_joints()

    q0 = [
        -2.70234157e-04,
        -7.84911342e-01,
        -9.66534154e-04,
        -2.35588574e00,
        5.17835122e-04,
        1.57050601e00,
        7.85670914e-01,
    ]
    q1 = [
        -0.12479884,
        0.9208731,
        -0.68638693,
        -1.89045153,
        1.19001496,
        1.20857264,
        0.93175768,
    ]
    q2 = [
        0.19469057,
        1.00462594,
        0.61927976,
        -1.77413309,
        -1.15911437,
        0.9704871,
        0.5900407,
    ]

    q_trial_1 = [
        -0.25357045,
        0.56281512,
        -0.42689346,
        -2.17925254,
        0.00389436,
        2.65625358,
        1.94666666,
    ]
    q_trial_2 = [
        0.34,
        5.18718091e-01,
        3.01030268e-01,
        -2.28148579e00,
        2.38826352e-04,
        2.76413768e00,
        -2.25361371e-01,
    ]

    x_offset = 0.22
    y_offset = 0.33
    z_offset = 0

    pointsObs, axesObs = shelf_obs(x_offset, y_offset, z_offset)
    mybot.PlotCollisionBlockPoints(q_trial_2, pointsObs)

# Load the xml file here
# model = mj.MjModel.from_xml_path(xml_filepath)
# data = mj.MjData(model)

# # Set the simulation scene to the home configuration
# mj.mj_resetDataKeyframe(model, data, 0)

# # Set the position controller callback
# mj.set_mjcb_control(position_control)

# # Launch the simulate viewer
# viewer.launch(model, data)
