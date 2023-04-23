import numpy as np
import math

def rpyxyz2H(rpy,xyz):
	Ht=[[1,0,0,xyz[0]],
	    [0,1,0,xyz[1]],
            [0,0,1,xyz[2]],
            [0,0,0,1]]

	Hx=[[1,0,0,0],
	    [0,math.cos(rpy[0]),-math.sin(rpy[0]),0],
            [0,math.sin(rpy[0]),math.cos(rpy[0]),0],
            [0,0,0,1]]

	Hy=[[math.cos(rpy[1]),0,math.sin(rpy[1]),0],
            [0,1,0,0],
            [-math.sin(rpy[1]),0,math.cos(rpy[1]),0],
            [0,0,0,1]]

	Hz=[[math.cos(rpy[2]),-math.sin(rpy[2]),0,0],
            [math.sin(rpy[2]),math.cos(rpy[2]),0,0],
            [0,0,1,0],
            [0,0,0,1]]

	H=np.matmul(np.matmul(np.matmul(Ht,Hz),Hy),Hx)

	return H

def R2axisang(R):
	ang = math.acos(( R[0,0] + R[1,1] + R[2,2] - 1)/2)
	Z = np.linalg.norm([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
	if Z==0:
		return[1,0,0], 0.
	x = (R[2,1] - R[1,2])/Z
	y = (R[0,2] - R[2,0])/Z
	z = (R[1,0] - R[0,1])/Z

	return[x, y, z], ang


def BlockDesc2Points(H, Dim):
	"""
	Args:
		  H (numpy array) : Homogenous Transformation Matrix
			   Dim (list) : Dimensions in length, width and height

	Returns:
		  (corners, axes) : corners and axes of bounding box of object
	"""
	center = H[0:3,3]
	axes=[ H[0:3,0],H[0:3,1],H[0:3,2]]

	corners=[
		center,
		center+(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
		center+(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.),
		center+(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
		center+(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.),
		center-(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
		center-(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.),
		center-(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
		center-(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.)
		]
	# returns corners of BB and axes
	return corners, axes



def CheckPointOverlap(pointsA,pointsB,axis):
	# TODO: check if sets of points projected on axis are overlapping

	# print("Shape of A:", pointsA.shape)
	# print("Shape of B:", pointsB.shape)
	# print("Shape of axis:", axis.shape)

	status = False

	pointsA = np.asarray(pointsA)
	pointsB = np.asarray(pointsB)

	# print("Shape of A:", pointsA.shape)
	# print("Shape of B:", pointsB.shape)

	# print("Points A:", pointsA)
	# print("Points B:", pointsB)
	# print("Axis:", axis)

	# print(np.linalg.norm(axis))

	norm_axis = axis/np.linalg.norm(axis)
	projectionA_scals = np.dot(axis, pointsA.T)
	project_axis = np.column_stack((norm_axis,norm_axis,norm_axis,norm_axis,norm_axis,norm_axis,norm_axis,norm_axis,norm_axis))

	projectionA = project_axis*projectionA_scals

	projectionB_scals = np.dot(axis, pointsB.T)

	projectionB = project_axis*projectionB_scals



	# print("Projected Shape of A:", projectionA.shape)

	if (np.min(projectionA[0,:])>np.max(projectionB[0,:]) or
     	np.max(projectionA[0,:])<np.min(projectionB[0,:])) or \
		(np.min(projectionA[1,:])>np.max(projectionB[1,:]) or
   		np.max(projectionA[1,:])<np.min(projectionB[1,:])) or \
		(np.min(projectionA[2,:])>np.max(projectionB[2,:]) or
   		np.max(projectionA[2,:])<np.min(projectionB[2,:])):
		# print("There is a separation")
		# print("Points A:", pointsA)
		# print("Points B:", pointsB)

		# print("Points projected on axis", axis)
		# print("Projected Points A:", projectionA)
		# print("Projected Points B:", projectionB)

		return False


	return True



def CheckBoxBoxCollision(pointsA,axesA,pointsB,axesB):
	"""

	Args:
		pointsA : corner points of cuboid 1
		axesA : axes of cuboid 1 (x,y,z unit vectors)
		pointsB : corner points of cuboid 2
		axesB : axes of cuboid 2 (x,y,z unit vectors)

	Returns:
		_type_: Bool
	"""
	# Sphere check
	if np.linalg.norm(pointsA[0]-pointsB[0])> (np.linalg.norm(pointsA[0]-pointsA[1])+np.linalg.norm(pointsB[0]-pointsB[1])):
		return False

	# Checking for collision along surface normals
	for i in range(3):
		if not CheckPointOverlap(pointsA,pointsB,axesA[i]):
			return False

	for j in range(3):
		if not CheckPointOverlap(pointsA,pointsB,axesB[j]):
			return False


	# Checking for edge-edge collisions
	for i in range(3):
		for j in range(3):
			if not CheckPointOverlap(pointsA,pointsB, np.cross(axesA[i],axesB[j])):
				return False


	return True

def so3(axis):

	so3_axis = np.asarray([
		[0,-axis[2], axis[1]],
		[axis[2], 0, -axis[0]],
		[-axis[1], axis[0], 0]
		])
	return so3_axis

def MatrixExp(axis,theta):

	so3_axis = so3(axis)

	R = np.eye(3) + np.sin(theta)*so3_axis + (1 - np.cos(theta))*np.matmul(so3_axis, so3_axis)

	last = np.zeros((1,4))
	last[0,3] = 1

	H_r = np.vstack((np.hstack((R,np.zeros((3,1)))),last))

	return H_r