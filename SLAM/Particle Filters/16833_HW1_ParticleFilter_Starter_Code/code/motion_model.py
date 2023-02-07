'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.1
        self._alpha2 = 0.1
        self._alpha3 = 0.1
        self._alpha4 = 0.1


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """

        delta_rot_1 = math.atan2((u_t1[1] - u_t0[1]), (u_t1[0] - u_t0[0])) - u_t0[2]
        delta_trans = math.sqrt((u_t0[0] - u_t1[0])**2 + (u_t0[1] - u_t1[1])**2)
        delta_rot_2 = u_t1[2] - u_t0[2] - delta_rot_1

        # prediction step (we need to sample...) but how?
        delta_rot1_pred = delta_rot_1 - np.random.normal(0, (self._alpha1*delta_rot_1**2 + self._alpha2*delta_trans**2), size=1)[0]
        delta_trans_pred = delta_trans - np.random.normal(0, (self._alpha3*delta_trans**2 + self._alpha4*delta_rot_1**2 + self._alpha4*delta_rot_2**2), size=1)[0]
        delta_rot2_pred = delta_rot_2 - np.random.normal(0,(self._alpha1*delta_rot_2**2 + self._alpha2*delta_trans**2),size=1)[0]

        x_t1 = x_t0[0] + delta_trans_pred * math.cos(x_t0[2] + delta_rot1_pred)
        y_t1 = x_t0[1] + delta_trans_pred * math.sin(x_t0[2] + delta_rot1_pred)
        theta_t1 = x_t0[2] + delta_rot1_pred + delta_rot2_pred

        return np.array([x_t1, y_t1, theta_t1])
