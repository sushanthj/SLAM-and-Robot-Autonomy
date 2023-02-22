'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''
import copy
import sys
import numpy as np
import math


def limit_angle(angle_):
    """
    Truncate angles outside [-pi,pi]
    Correct wrap around/spill over to avoid potential divergence of PF
    """
    angle = copy.deepcopy(angle_)

    while angle > math.pi:
        angle -= 2 * math.pi

    while angle < -math.pi:
        angle += 2 * math.pi

    # if angle!=angle_:
    #     #some correction was made
    #     print("Limiting angle between -pi and pi")
    #     print("{} radians corrected to {} radians".format(angle_, angle))

    return angle


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

        # # DO NOT TOUCH.
        # self._alpha1 = 0.17 / 300  # associated with heading angle
        # self._alpha2 = 0.17 / 300
        # self._alpha3 = 1 / 300 # associated with the wheel odom.
        # self._alpha4 = 1 / 300

        self._alpha1 = 0.00001
        self._alpha2 = 0.00001
        self._alpha3 = 0.0001
        self._alpha4 = 0.0001

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code hereasd
        """
        y_bar_1 = u_t1[1]
        y_bar_0 = u_t0[1]

        x_bar_1 = u_t1[0]
        x_bar_0 = u_t0[0]

        yaw_bar_1 = u_t1[2]
        yaw_bar_0 = u_t0[2]

        x_0 = x_t0[0]
        y_0 = x_t0[1]
        yaw_0 = x_t0[2]

        # sample_motion_model_odometry chapter 5.
        delta_rot_1 = np.arctan2(y_bar_1 - y_bar_0, x_bar_1 - x_bar_0) - yaw_bar_0
        delta_trans = np.linalg.norm(np.array([x_bar_1 - x_bar_0, y_bar_1 - y_bar_0]), ord=2)
        delta_rot_2 = yaw_bar_1 - yaw_bar_0 - delta_rot_1

        delta_rot_1 = limit_angle(delta_rot_1)
        delta_rot_2 = limit_angle(delta_rot_2)

        # delta_rot_1_hat = delta_rot_1 - self.sample(self._alpha1 * delta_rot_1 ** 2 + self._alpha2 * delta_trans ** 2)
        # delta_trans_hat = delta_trans - self.sample(self._alpha3 * delta_trans ** 2 + self._alpha4 * (delta_rot_1 ** 2 + delta_rot_2 ** 2))
        # delta_rot_2_hat = delta_rot_2 - self.sample(self._alpha1 * delta_rot_2 ** 2 + self._alpha2 * delta_trans ** 2)
        #
        delta_rot_1_hat = delta_rot_1 - np.random.normal(0, self._alpha1 * delta_rot_1 ** 2 + self._alpha2 * delta_trans ** 2)
        delta_trans_hat = delta_trans - np.random.normal(0, self._alpha3 * delta_trans ** 2 + self._alpha4 * (delta_rot_1 ** 2 + delta_rot_2 ** 2))
        delta_rot_2_hat = delta_rot_2 - np.random.normal(0, self._alpha1 * delta_rot_2 ** 2 + self._alpha2 * delta_trans ** 2)

        delta_rot_1_hat = limit_angle(delta_rot_1_hat)
        delta_rot_2_hat = limit_angle(delta_rot_2_hat)

        x_1 = x_0 + delta_trans_hat * np.cos(yaw_0 + delta_rot_1_hat)
        y_1 = y_0 + delta_trans_hat * np.sin(yaw_0 + delta_rot_1_hat)
        yaw_1 = yaw_0 + delta_rot_1_hat + delta_rot_2_hat

        yaw_1 = limit_angle(yaw_1)

        return np.array([x_1, y_1, yaw_1])

    def sample(self, var):
        return var * np.sum(np.random.uniform(-1, 1, (12, 1))) / 12
