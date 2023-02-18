import numpy as np
import RobotUtil as rt
import math
import sys
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


class FrankArm:

    def __init__(self):
		# Robot descriptor taken from URDF file (rpy xyz for each rigid link transform) - NOTE: don't change
        self.Rdesc=[
			[0, 0, 0, 0., 0, 0.333], # From robot base to joint1
			[-np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0, -0.316, 0],
			[np.pi/2, 0, 0, 0.0825, 0, 0],
			[-np.pi/2, 0, 0, -0.0825, 0.384, 0],
			[np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0.088, 0, 0],
            [0, 0, 0, 0, 0, 0.107] # From joint5 to end-effector center
			]

		#Define the axis of rotation for each joint
        self.axis=[
				[0, 0, 1],
				[0, 0, 1],
				[0, 0, 1],
				[0, 0, 1],
				[0, 0, 1],
				[0, 0, 1],
				[0, 0, 1],
				[0, 0, 1]
        		]

		#Set base coordinate frame as identity - NOTE: don't change
        self.Tbase= [[1,0,0,0],
			[0,1,0,0],
			[0,0,1,0],
			[0,0,0,1]]

		#Initialize matrices - NOTE: don't change this part
        self.Tlink=[] # Transforms for each link (const)
        self.Tjoint=[] # Transforms for each joint (init eye)
        self.Tcurr=[] # Coordinate frame of current (init eye)
        for i in range(len(self.Rdesc)):
            self.Tlink.append(rt.rpyxyz2H(self.Rdesc[i][0:3],self.Rdesc[i][3:6]))
            self.Tcurr.append([[1,0,0,0],[0,1,0,0],[0,0,1,0.],[0,0,0,1]])
            self.Tjoint.append([[1,0,0,0],[0,1,0,0],[0,0,1,0.],[0,0,0,1]])

        self.Tlinkzero=rt.rpyxyz2H(self.Rdesc[0][0:3],self.Rdesc[0][3:6])

        self.Tlink[0]=np.matmul(self.Tbase,self.Tlink[0])

		# initialize Jacobian matrix
        self.J=np.zeros((6,7))

        self.q=[0.,0.,0.,0.,0.,0.,0.]
        self.ForwardKin([0.,0.,0.,0.,0.,0.,0.])

    def ForwardKin(self,ang):
        '''
		inputs: target joint angles
		outputs: joint transforms for each joint, Jacobian matrix
		'''

        # input angles
        self.q[0:-1] = ang

		# Compute current joint and end effector coordinate frames (self.Tjoint). Remember that not all joints rotate about the z axis!
        # we need to use the input angles and transform each joint to finally get the end effector pose
        for i in range(len(self.q)):
            # initially self.Tjoint would be at the home position of each joint defined above
            self.Tjoint[i] = [
                                [math.cos(self.q[i]), -math.sin(self.q[i]), 0, 0],
                                [math.sin(self.q[i]), math.cos(self.q[i]), 0, 0],
                                [0,0,1,0],
                                [0,0,0,1],
                             ]

            if i == 0:
                self.Tcurr[i] = np.matmul(self.Tlink[i], self.Tjoint[i])
            else:
                # compute the i'th joint's frame by chaining previous frame && link_transform && join_transform
                self.Tcurr[i] = np.matmul(np.matmul(self.Tcurr[i-1], self.Tlink[i]), self.Tjoint[i])

        # compute the jacobian
        # loop over all joints expect the last one. Tcurr is the transformation matrix of each joint
        for i in range(len(self.Tcurr)-1):
            #p = compare translation of last joint with ith joint
            p = self.Tcurr[-1][0:3,3] - self.Tcurr[i][0:3,3]
            a = self.Tcurr[i][0:3,2] # [0:3,2] = z-axis of i'th joint
            # a = angular vel of the revolute joint and p = position vector from joint i -> end effector
            # cross product of the two gives that column of the jacobian
            self.J[0:3,i] = np.cross(a,p)
            self.J[3:7,i] = a

        return self.Tcurr, self.J


    def IterInvKin(self,ang,TGoal,x_eps=1e-3, r_eps=1e-3):
        '''
		inputs: starting joint angles (ang), target end effector pose (TGoal)

		outputs: computed joint angles to achieve desired end effector pose,
		Error in your IK solution compared to the desired target
		'''

        W = np.eye(7)
        W[-1,0] = 1.0
        W[2,2] = 100.0
        W[3,3] = 100.0
        W[-1,-1] = 100.0

        C = np.eye(6)
        C[0,0] = 1000000.0
        C[1,1] = 1000000.0
        C[2,2] = 1000000.0
        C[3,3] = 1000.0
        C[4,4] = 1000.0
        C[5,5] = 1000.0

        self.ForwardKin(ang)

        Err = np.ones(6, dtype=np.float32)

        while(np.linalg.norm(Err[0:3]) > x_eps and np.linalg.norm(Err[3:6] > r_eps)):
            # compute rotation error
            rErrR = np.matmul(TGoal[0:3, 0:3], np.transpose(self.Tcurr[-1][0:3,0:3]))
            # convert rotation error to axis angle form
            rErrAxis, rErrAng = rt.R2axisang(rErrR)

            # limit rotation angle
            if rErrAng > 0.1: rErrAng = 0.1

            if rErrAng < -0.1: rErrAng = -0.1

            # final rotation error
            rErr = [rErrAxis[0]*rErrAng, rErrAxis[1]*rErrAng, rErrAxis[2]*rErrAng]

            # compute position error
            xErr = TGoal[0:3,3] - self.Tcurr[-1][0:3,3]
            if np.linalg.norm(xErr) > 0.01:
                xErr = (xErr*0.01)/np.linalg.norm(xErr)

            # update angles with pseudo inverse
            Err[0:3] = xErr
            Err[3:6] = rErr

            """
            compute the new angles to command
            """

            # Damped Least Squares approach to calculate Inverse Kinematics
            # self.q[0:7] += J# * delta_x (where delta_x = Err)

            # finding the J#
            J = self.J
            J_hash = np.linalg.inv(W) @ J.T @ np.linalg.inv(J @ np.linalg.inv(W) @ J.T + np.linalg.inv(C))

            # update self.q by increments
            self.q[0:7] += J_hash @ Err

            # do forward kinematics with new angles
            self.ForwardKin(self.q[0:7])

        return self.q[0:-1], Err