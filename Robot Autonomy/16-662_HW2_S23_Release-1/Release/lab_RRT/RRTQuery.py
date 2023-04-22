from random import sample, seed
from re import A
import time
import pickle
import numpy as np

# import vrep_interface as vpi
import RobotUtil as rt
import Franka
import time
import frankapy


# Seed the random object
seed(10)

# Open the simulator model from the MJCF file
xml_filepath = "../franka_emika_panda/panda_with_hand_torque.xml"

np.random.seed(0)
deg_to_rad = np.pi / 180.

# Initialize robot object
mybot = Franka.FrankArm()

# Initialize robot object
realbot = frankapy.FrankaArm()

# Initialize some variables related to the simulation
joint_counter = 0

# Initializing planner variables as global for access between planner and simulator
plan = []
interpolated_plan = []
plan_length = len(plan)
inc = 1

# Add obstacle descriptions into pointsObs and axesObs
# (bounding boxes for all the obstacles present in the environment (pointObs and envaxes))
pointsObs = []
axesObs = []

# LAB CONFIGURATION FOR FRANKA BELOW-------------------------------------------------------------
# DO NOT RESET BELOW SETUP
length_box = 0.29
width_box = 0.29
height_box = 0.15
left_reach = [0.40829459, -0.16118822, 0.07832987]
right_reach = [0.42463734, 0.15759294, 0.07898102]
# cuboid box obstacle center coordinate
x_ctr = (left_reach[0] + right_reach[0]) / 2
y_ctr = (left_reach[1] + right_reach[1]) / 2
z_ctr = height_box / 2

# for each entry(sublist) in this list, position(len=3) + orientation(len=3) + cuboid_dim(len3)
obstacle_parameters = [[x_ctr, y_ctr, z_ctr, 0, 0, 0, length_box, width_box, height_box], 
                       [0.15, 0.46, 0.5, 0, 0, 0, 1.2, 0.01, 1.1], 
                       [0.15, -0.46, 0.5, 0, 0, 0, 1.2, 0.01, 1.1], 
                       [-0.41, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1], 
                       [0.75, 0, 0.5, 0, 0, 0, 0.01, 1, 1.1], 
                       [0.2, 0, 1, 0, 0, 0, 1.2, 1, 0.01], 
                       [0.2, 0, -0.05, 0, 0, 0, 1.2, 1, 0.01]]

for para in obstacle_parameters:
    xyz = para[:3]
    rpy = para[3:6]
    lwh = para[6:9]
    envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H(rpy, xyz), lwh)
    pointsObs.append(envpoints), axesObs.append(envaxes)

# DO NOT RESET ABOVE SETUP
# LAB CONFIGURATION FOR FRANKA ABOVE-------------------------------------------------------------

# define start and goal
deg_to_rad = np.pi / 180.
    # Reset obstacle descriptions into pointsObs and axesObs


thresh = 0.1
FoundSolution = False
SolutionInterpolated = False


# Utility function to find the index of the nearset neighbor in an array of neighbors in prevPoints
def FindNearest(prevPoints, newPoint):
    D = np.array([np.linalg.norm(np.array(point) - np.array(newPoint)) for point in prevPoints])
    return D.argmin()


# Utility function for smooth linear interpolation of RRT plan, used by the controller
def naive_interpolation(plan):  # take a minimal path, populate qs between the minimal nodes to make it smooth
    angle_resolution = 0.01

    global interpolated_plan
    global SolutionInterpolated
    interpolated_plan = np.empty((1, 7))
    np_plan = np.array(plan)
    interpolated_plan[0] = np_plan[0]

    for i in range(np_plan.shape[0] - 1):
        max_joint_val = np.max(np_plan[i + 1] - np_plan[i])   # scaler maximal change
        number_of_steps = int(np.ceil(max_joint_val / angle_resolution))
        inc = (np_plan[i + 1] - np_plan[i]) / number_of_steps

        for j in range(1, number_of_steps + 1):
            step = np_plan[i] + j * inc
            interpolated_plan = np.append(interpolated_plan, step.reshape(1, 7), axis=0)

    SolutionInterpolated = True
    print("Plan has been interpolated successfully!")


# TODO: - Create RRT to find path to a goal configuration by completing the function
# below. Use the global rrtVertices, rrtEdges, plan and FoundSolution variables in 
# your algorithm

def resetParameters():
    global FoundSolution  # boolean flag
    global plan  # list
    global rrtVertices  # list
    global rrtEdges  # list
    global joint_counter
    global interpolated_plan
    global plan_length
    global inc
    global pointsObs
    global axesObs
    global SolutionInterpolated
    # Initialize some variables related to the simulation
    joint_counter = 0
    # Initializing planner variables as global for access between planner and simulator
    plan = []
    interpolated_plan = []
    plan_length = len(plan)
    inc = 1

    # Reset graphs
    rrtVertices = []
    rrtEdges = []
    # Reset flags
    FoundSolution = False
    SolutionInterpolated = False

def RRTQuery(qInit, qGoal):
    global FoundSolution  # boolean flag
    global plan  # list
    global rrtVertices  # list
    global rrtEdges  # list

    resetParameters()
    rrtVertices.append(qInit)
    rrtEdges.append(0)

    while len(rrtVertices) < 30000 and not FoundSolution:
        # Fill in the algorithm here
        print("Graph size = ", len(rrtVertices))

        # Goal biasing with 20% chance
        goal_biasing = np.random.uniform(0, 1) < 0.37  # boolean flag
        if goal_biasing:
            qRand = qGoal
        else:
            qRand = mybot.SampleRobotConfig()
            qRand[5] = np.pi - np.pi / 6
            qRand[6] = 0

        idNear = FindNearest(rrtVertices, qRand)
        qNear = rrtVertices[idNear]

        while np.linalg.norm(np.array(qRand) - np.array(qNear)) > thresh:  # proceed toward qRand/qGoal as much as possible
            # Compute stepped q toward qRand qualified by step size
            qConnect = np.array(qNear) + thresh * (np.array(qRand) - np.array(qNear)) / np.linalg.norm(np.array(qRand) - np.array(qNear))

            if not mybot.DetectCollisionEdge(qConnect, qNear, pointsObs, axesObs):
                rrtVertices.append(qConnect)
                rrtEdges.append(idNear)
                qNear = qConnect
            else:
                break

        qConnect = qRand
        if not mybot.DetectCollisionEdge(qConnect, qNear, pointsObs, axesObs):
            rrtVertices.append(qConnect)
            rrtEdges.append(idNear)
            if goal_biasing:
                FoundSolution = True
                break

        idNear = FindNearest(rrtVertices, qGoal)
        if np.linalg.norm(np.array(qGoal) - np.array(rrtVertices[idNear])) < 0.025:
            rrtVertices.append(qGoal)
            rrtEdges.append(idNear)
            FoundSolution = True
            break

    print("Here")
    ### if a solution was found
    if FoundSolution:
        # Extract path
        c = -1  # Assume last added vertex is at goal
        plan.insert(0, rrtVertices[c])

        while True:
            c = rrtEdges[c]
            plan.insert(0, rrtVertices[c])
            if c == 0:
                break

        # TODO - Path shortening
        for i in range(150):
            anchorA = np.random.randint(0, len(plan) - 2)
            anchorB = np.random.randint(anchorA + 1, len(plan) - 1)

            shiftA = np.random.uniform(0, 1)
            shiftB = np.random.uniform(0, 1)

            candidateA = (1 - shiftA) * np.array(plan[anchorA]) + shiftA * np.array(plan[anchorA + 1])
            candidateB = (1 - shiftB) * np.array(plan[anchorB]) + shiftB * np.array(plan[anchorB + 1])

            if not mybot.DetectCollisionEdge(candidateA, candidateB, pointsObs, axesObs):
                while anchorB > anchorA:
                    plan.pop(anchorB)
                    anchorB -= 1
                plan.insert(anchorA + 1, candidateB)
                plan.insert(anchorA + 1, candidateA)

        for (i, q) in enumerate(plan):
            print("Plan step: ", i, "and joint: ", q)
        plan_length = len(plan)

        naive_interpolation(plan)

        return plan

    else:
        print("No solution found")


################################# YOU DO NOT NEED TO EDIT ANYTHING BELOW THIS ##############################

def position_control(model, data):
    global joint_counter
    global inc
    global plan
    global plan_length
    global interpolated_plan

    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Check if plan is available, if not go to the home position
    if (FoundSolution == False or SolutionInterpolated == False):
        desired_joint_positions = np.array(qInit)

    else:

        # If a plan is available, cycle through poses
        plan_length = interpolated_plan.shape[0]

        if np.linalg.norm(interpolated_plan[joint_counter] - data.qpos[:7]) < 0.01 and joint_counter < plan_length:
            joint_counter += inc

        desired_joint_positions = interpolated_plan[joint_counter]

        # goes back and forth between qinit and qgoal
        if joint_counter == plan_length - 1:
            inc = -1 * abs(inc)
            joint_counter -= 1
        if joint_counter == 0:
            inc = 1 * abs(inc)

    # Set the desired joint velocities
    desired_joint_velocities = np.array([0, 0, 0, 0, 0, 0, 0])

    # Desired gain on position error (K_p)
    Kp = np.eye(7, 7) * 300

    # Desired gain on velocity error (K_d)
    Kd = 50

    # Set the actuator control torques
    data.ctrl[:7] = data.qfrc_bias[:7] + Kp @ (desired_joint_positions - data.qpos[:7]) + Kd * (
            desired_joint_velocities - data.qvel[:7])

if __name__ == "__main__":
    realbot.reset_joints()
    q0 = [-2.70234157e-04, -7.84911342e-01, -9.66534154e-04, -2.35588574e+00, 5.17835122e-04, 1.57050601e+00, 7.85670914e-01]
    q1 = [-0.12479884, 0.9208731, -0.68638693, -1.89045153, 1.19001496, 1.20857264, 0.93175768]
    q2 = [0.19469057, 1.00462594, 0.61927976, -1.77413309, -1.15911437, 0.9704871, 0.5900407]

    q_trial_1 = [-0.25357045, 0.56281512, -0.42689346, -2.17925254, 0.00389436, 2.65625358, 1.94666666]
    q_trial_2 = [0.34, 5.18718091e-01, 3.01030268e-01, -2.28148579e+00, 2.38826352e-04, 2.76413768e+00, -2.25361371e-01]
    # Debug visualizations for virtual walls and blocks
    # mybot.PlotCollisionBlockPoints(qInit, pointsObs)
    # mybot.PlotCollisionBlockPoints(q1, pointsObs)
    # mybot.PlotCollisionBlockPoints(q2, pointsObs)
    # LOOKS GOOD!
    mybot.PlotCollisionBlockPoints(q_trial_1, pointsObs)
    mybot.PlotCollisionBlockPoints(q_trial_2, pointsObs)

	# Compute the RRT solution
    path1 = RRTQuery(q0, q_trial_1)
    path2 = RRTQuery(q_trial_1, q_trial_2)
    if type(path1) is not None and type(path2) is not None:

        total_path = path1 + path2
        print("Path length = ", len(total_path))
        print("Path traversing ...")
        for i, pos in enumerate(total_path):
            print("Node = ", i)
            realbot.goto_joints(pos)
        print("Goal reached!")
        
        realbot.reset_joints()
        print("Finshed and resetting...")
