from random import sample, seed
from re import A
import time
import pickle
import numpy as np

# import vrep_interface as vpi
import RobotUtil as rt
import Franka
import time
import ipdb

# Mujoco Imports
import mujoco as mj
from mujoco import viewer
from scipy.spatial.transform import Rotation

# Seed the random object
seed(10)

# Open the simulator model from the MJCF file
xml_filepath = "../franka_emika_panda/panda_with_hand_torque.xml"
# xml_filepath = "../franka_emika_panda/panda_nohand_torque_fixed_board_2.xml"
# xml_filepath = "../franka_emika_panda/panda_with_hand_torque_2.xml"

np.random.seed(0)
deg_to_rad = np.pi/180.

#Initialize robot object
mybot = Franka.FrankArm()

# Initialize some variables related to the simulation
joint_counter = 0

# Initializing planner variables as global for access between planner and simulator
plan=[]
interpolated_plan = []
plan_length = len(plan)
inc = 1

# Add obstacle descriptions into pointsObs and axesObs
# List containing corners and axes of obstacles
pointsObs=[]
axesObs=[]

# # NOTE: We hard code the obstacle pose and dimentions
# # envpts, envaxes  = rt.BlockDesc2Points(pose, dimension) # object 1
envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1,0,1.0]),[1.3,1.4,0.1])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1,-0.65,0.475]),[1.3,0.1,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1, 0.65,0.475]),[1.3,0.1,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[-0.5, 0, 0.475]),[0.1,1.2,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.45, 0, 0.25]),[0.5,0.4,0.5])
pointsObs.append(envpoints), axesObs.append(envaxes)

# define start and goal
deg_to_rad = np.pi/180.

# set the initial and goal joint configurations
# NOTE: In the lab, this can be found by manually placing the robot end effector
# to the desired position and then record the joint angles
# qInit = [-np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, 0, np.pi - np.pi/6, 0]
# qGoal = [np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, 0, np.pi - np.pi/6, 0]
qInit = [-1.53, -1.49, 1.51, -1.53, 1.48, 1.65, 0]
qGoal = [1.56, -1.56, -1.64, -1.56, -1.59, 1.59, -2.01]

# Initialize some data containers for the RRT planner
rrtVertices=[]
rrtEdges=[]

rrtVertices.append(qInit)
rrtEdges.append(0)

### GLOBAL TUNING VARIABLES ############################################
# threshold which determines how close we are to goal
thresh=0.1
cost_thresh = 0.2
starting_roll = 3.129 #1.57
starting_pitch = -0.14
rejection_threshold = 1
angle_resolution = 0.2
joint_to_constrain = 7
learning_rate = 0.2
path_shortening = True
#########################################################################
# NOTE: for tight perf. reqs. use cost_thres = 0.1, rejection_thres = 0.3

FoundSolution=False
SolutionInterpolated = False

# Utility function to find the index of the nearset neighbor in an array of neighbors in prevPoints
def FindNearest(prevPoints,newPoint):
	D=np.array([np.linalg.norm(np.array(point)-np.array(newPoint)) for point in prevPoints])
	return D.argmin()

# Utility function for smooth linear interpolation of RRT plan, used by the controller
def naive_interpolation(plan):

	global interpolated_plan
	global SolutionInterpolated
	interpolated_plan = np.empty((1,7))
	np_plan = np.array(plan)
	interpolated_plan[0] = np_plan[0]

	for i in range(np_plan.shape[0]-1):
		max_joint_val = np.max(np_plan[i+1] - np_plan[i])
		number_of_steps = int(np.ceil(max_joint_val/angle_resolution))
		inc = (np_plan[i+1] - np_plan[i])/number_of_steps

		for j in range(1,number_of_steps+1):
			step = np_plan[i] + j*inc
			interpolated_plan = np.append(interpolated_plan, step.reshape(1,7), axis=0)


	SolutionInterpolated = True
	print("Plan has been interpolated successfully!")


def project_to_constrain(qRand):
	"""
	Project to make roll and pitch zero where possible. We do this by gradient descent

	Our cost function is C = (3.14 - roll)**2 + pitch**2 (we want to minize roll and pitch)
	NOTE: (3.14 - roll) since we have init roll of 3.14
	"""

	# do forward kinematics and get the roll, pitch at qRand
	roll, pitch, yaw, J = get_roll_pitch_of_rand_pt(qRand)
	# print(f"init roll={roll} and pitch={pitch} and yaw={yaw}")

	if (abs(starting_roll-abs(roll))) > rejection_threshold or \
		(abs(starting_pitch - abs(pitch)) > rejection_threshold):
		return qRand, True

	count = 0

	# while(((starting_roll-abs(roll))**2 + pitch**2 + (starting_yaw - abs(yaw)) > cost_thresh) and count < 1000):
	while(((starting_roll-abs(roll))**2 + (starting_pitch-abs(pitch))**2 > cost_thresh)
       		and count < 1000):
		grad_cost_wrt_xyzrpy = np.expand_dims(np.array([0,0,0, 2*roll, 2*pitch, 0]), axis=1)
		gradient = J.T @ grad_cost_wrt_xyzrpy

		qRand = np.expand_dims(np.array(qRand), axis=1) - learning_rate * gradient
		qRand = np.squeeze(qRand).tolist()
		roll, pitch, yaw, J = get_roll_pitch_of_rand_pt(qRand)
		count += 1

	# print(f"final roll={roll} and pitch={pitch} and yaw={yaw}")

	return qRand, False


def get_roll_pitch_of_rand_pt(qRand):
	# do forward kinematics and get the Tcurr, J at qRand
	Tcurr, J = mybot.ForwardKin_for_check(qRand)
	last_link_rotation = np.asarray(Tcurr[joint_to_constrain])[0:3,0:3]
	r = Rotation.from_matrix(last_link_rotation)
	roll, pitch, yaw = r.as_euler('xyz')

	return roll, pitch, yaw, J

#TODO: - Create RRT to find path to a goal configuration by completing the function
# below. Use the global rrtVertices, rrtEdges, plan and FoundSolution variables in
# your algorithm

def RRTQuery():

	global FoundSolution
	global plan
	global rrtVertices
	global rrtEdges

	roll, pitch, yaw, J = get_roll_pitch_of_rand_pt(qInit)
	print("starting roll, pitch, and yaw", roll, pitch, yaw)

	# making the assumption that we should find solution within 3000 iterations
	while len(rrtVertices)<10000 and not FoundSolution:

		# TODO : Fill in the algorithm here
		# create a random node (x,y as a 2,1 array)
		qRand = mybot.SampleRobotConfig()

		# introduce the goal bias. (set the random node as goal with a certain prob)
		if np.random.uniform(0,1) < thresh:
			qRand = qGoal

		"""Constrained RRT step"""
		# NOTE: now that we have a qRand, if we want this qRand to be such that the
		# end effector has roll and pitch as zero
		qRand, flag = project_to_constrain(qRand)
		for i in range(len(qRand)):
			if (qRand[i] > mybot.qmax[i] or qRand[i] < mybot.qmin[i]):
				flag = True

		if flag:
			continue

		idNear = FindNearest(rrtVertices, qRand)
		qNear = rrtVertices[idNear]

		qNear, qRand = np.asarray(qNear), np.asarray(qRand)

		# if it's above threshold, move in the direction of the new node, but only upto the
		# threshold (which limits max distance between two nodes)
		while np.linalg.norm(qRand - qNear) > thresh:
			# qConnect = qNear + thres * unit_vector_pointing_towards_qRand
			qConnect = qNear + thresh * ((qRand-qNear) / np.linalg.norm(qRand-qNear))

			if not mybot.DetectCollisionEdge(qConnect, qNear, pointsObs, axesObs):
				rrtVertices.append(qConnect)
				rrtEdges.append(idNear)
				qNear = qConnect

			else:
				break

		# check for collisions
		qConnect = qRand
		if not mybot.DetectCollisionEdge(qConnect, qNear, pointsObs, axesObs):
			# if no collision in new joint angles (qConnect), then add as a valid node and edge
			rrtVertices.append(qConnect)
			rrtEdges.append(idNear)

		# check if the qGoal is close to some node
		idNear = FindNearest(rrtVertices, qGoal)
		# if the qGoal is really close (< 0.025) then we've pretty much reached goal!
		if np.linalg.norm(np.asarray(qGoal) - np.asarray(rrtVertices[idNear])) < 0.025:
			# add the goal node as our final node
			rrtVertices.append(qGoal)
			rrtEdges.append(idNear)
			print("SOLUTION FOUND")
			FoundSolution = True

		print(len(rrtVertices))

	### if a solution was found
	if FoundSolution:
		# Extract path
		c=-1 # Assume last added vertex is at goal
		plan.insert(0, rrtVertices[c])

		print("rrtEdges are \n", rrtEdges)
		print("***************")

		while True:
			# print("c before", c)
			c=rrtEdges[c]
			# print("c after", c)
			plan.insert(0, rrtVertices[c])
			if c==0:
				break

		if path_shortening:
			for i in range(150):
				anchorA = np.random.randint(0, len(plan) - 2)
				anchorB = np.random.randint(anchorA + 1, len(plan) - 1)

				shiftA = np.random.uniform(0, 1)
				shiftB = np.random.uniform(0, 1)

				candidateA = (1 - shiftA) * np.array(plan[anchorA]) + shiftA * np.array(plan[anchorA + 1])
				try:
					candidateB = (1 - shiftB) * np.array(plan[anchorB]) + shiftB * np.array(plan[anchorB + 1])
				except:
					ipdb.set_trace()

				if not mybot.DetectCollisionEdge(candidateA, candidateB, pointsObs, axesObs):
					while anchorB > anchorA:
						plan.pop(anchorB)
						anchorB = anchorB - 1

					plan.insert(anchorA + 1, candidateB)
					plan.insert(anchorA + 1, candidateA)

		for (i, q) in enumerate(plan):
			print("Plan step: ", i, "and joint: ", q)
		plan_length = len(plan)

		naive_interpolation(plan)

		return

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
	if (FoundSolution==False or SolutionInterpolated==False):
		desired_joint_positions = np.array(qInit)

	else:

		# If a plan is available, cycle through poses
		plan_length = interpolated_plan.shape[0]

		if np.linalg.norm(interpolated_plan[joint_counter] - data.qpos[:7]) < 0.01 and joint_counter < plan_length:
			joint_counter+=inc

		desired_joint_positions = interpolated_plan[joint_counter]

		if joint_counter==plan_length-1:
			inc = -1*abs(inc)
			joint_counter-=1
		if joint_counter==0:
			inc = 1*abs(inc)


	# Set the desired joint velocities
	desired_joint_velocities = np.array([0,0,0,0,0,0,0])

	# Desired gain on position error (K_p)
	Kp = np.eye(7,7)*300

	# Desired gain on velocity error (K_d)
	Kd = 50

	# Set the actuator control torques
	data.ctrl[:7] = data.qfrc_bias[:7] + Kp@(desired_joint_positions-data.qpos[:7]) + Kd*(desired_joint_velocities-data.qvel[:7])


if __name__ == "__main__":

	# Load the xml file here
	model = mj.MjModel.from_xml_path(xml_filepath)
	data = mj.MjData(model)

	# Set the simulation scene to the home configuration
	mj.mj_resetDataKeyframe(model, data, 0)

	# Set the position controller callback
	mj.set_mjcb_control(position_control)

	# Compute the RRT solution
	RRTQuery()

	# Launch the simulate viewer
	viewer.launch(model, data)