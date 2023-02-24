'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time

import torch
from matplotlib import pyplot as plt
from scipy.stats import norm, expon, uniform

from map_reader import MapReader


DEBUG = True

def norm_pdf(z, z_star, sigma):
    return np.exp(-(z - z_star) ** 2/ (2.0 * sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2))


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 2
        self._z_short = 0.12
        self._z_max = 0.05
        self._z_rand = 300

        # Contextualize why you chose this.
        # self._z_hit = 0.02
        # self._z_short = 0.005 # Might be too high
        # self._z_max = 0.01
        # self._z_rand = 300 # Might want to increase it

        self._sigma_hit = 100
        self._lambda_short = 0.00070 # Small values smoother curve?

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 8183

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2
        self._occupancy_map = occupancy_map
        self._occupancy_map_confidence_threshold = 0.45
        self._occupancy_map_resolution_centimeters_per_pixel = 10

        self._short_distribution = expon(scale=1 / self._lambda_short)
        self._rand_distribution = uniform(loc=0, scale=self._max_range * 2)

        self._max_width = 100
        self._max_distribution = uniform(loc=self._max_range, scale=self._max_width)
        self._discretization = 1

        if DEBUG:
            self.visualize_probability_distribution()
            self._pdf(500, 1000)

        self._vectorized_pdf = np.vectorize(self._pdf)

    def _pdf(self, z, z_star):
        hit_distribution = norm(loc=z_star, scale=self._sigma_hit)
        pdf_hit = self._z_hit * hit_distribution.pdf(z)
        pdf_short = self._z_short * self._short_distribution.pdf(z)
        pdf_max = self._z_max * self._max_distribution.pdf(z)
        pdf_rand = self._z_rand * self._rand_distribution.pdf(z)
        return pdf_hit + pdf_short + pdf_max + pdf_rand

    def visualize_probability_distribution(self):
        zs = np.linspace(0, self._max_range + self._max_width + 1000, 1000)
        z_star = 1000
        pdfs = [self._pdf(z, z_star) for z in zs]
        plt.plot(zs, pdfs)

        plt.xlim(0, self._max_range + self._max_width + 1000)
        plt.ylim(0, 0.1)

        plt.show()

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        """
        TODO : Add your code here
        """
        prob_zt1 = 0.05
        z_t1_arr = self.ray_casting(x_t1)
        # grid = np.array([self._z_hit, self._z_short, self._z_max, self._z_rand])
        # z = np.random.uniform()
        # np.random.uniform()
        # num_beams = len(z_t1_arr)
        # assert num_beams == 180
        # origin = np.array([0, 0], dtype=np.float32)
        # for i in range(len(z_t1_arr)):
        #     origin + z_t1_arr[i]
        return prob_zt1


    def ray_casting_vectorized_centimeters(self, X_t1):
        num_particles = len(X_t1)
        num_beams = 360 // self._discretization
        X_body, Y_body, Yaw = X_t1[:, 0], X_t1[:, 1], X_t1[:, 2]
        X_laser = X_body + 25 * np.cos(Yaw)
        Y_laser = Y_body + 25 * np.sin(Yaw)

        X_laser = np.repeat(X_laser.reshape(-1, 1), num_beams, axis=1) # m, 180
        Y_laser = np.repeat(Y_laser.reshape(-1, 1), num_beams, axis=1) # m, 180

        angles = np.array(list(range(0, 360, self._discretization)))
        assert len(angles) == num_beams

        beam_hit_length = np.ones_like(X_laser) * self._max_range
        beam_step = self._occupancy_map_resolution_centimeters_per_pixel // 2.1
        np.arange(0, self._max_range, beam_step)
        for ray_length in np.arange(0, self._max_range, beam_step):
            # The beams start from the RHS of the robot, the yaw angle is measured from the heading of the robot.
            # Hence the minus 90 degrees.
            X_beams = X_laser + np.cos(np.radians(angles) + np.repeat(Yaw.reshape(-1, 1), 360 // self._discretization, axis=1)) * ray_length
            Y_beams = Y_laser + np.sin(np.radians(angles) + np.repeat(Yaw.reshape(-1, 1), 360 // self._discretization, axis=1)) * ray_length

            X_beams_pixels = np.round(X_beams / 10).astype(int)
            Y_beams_pixels = np.round(Y_beams / 10).astype(int)

            X_beams_pixels = np.clip(X_beams_pixels, 0, 799)
            Y_beams_pixels = np.clip(Y_beams_pixels, 0, 799)

            occupancy_vals = self._occupancy_map[Y_beams_pixels, X_beams_pixels]
            # occupancy_vals = self._occupancy_map[X, Y]

            beam_hit_length = np.minimum(
                beam_hit_length,
                np.where(occupancy_vals > self._occupancy_map_confidence_threshold, ray_length, self._max_range)
            )

        Z_star_t_arr = beam_hit_length
        return Z_star_t_arr
