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

    def ray_casting(self, x_t1) -> np.ndarray:

        x_body, y_body, yaw = x_t1[0], x_t1[1], x_t1[2]

        # instruct.txt said that the lidar is 25 centimeters ahead of the body
        x_laser = x_body + 25 * np.cos(yaw)
        y_laser = y_body + 25 * np.sin(yaw)

        num_laser_scans = 180

        z_star_t1_arr = np.zeros(180)
        for i in range(num_laser_scans):
            # Convert cartesian coordinate to pixel coordinate.
            # center_image_pixel = np.array([map_height_pixel - round(y_laser / 10), round(x_laser / 10)])
            center_cartesian_pixel = np.array([x_laser / 10, y_laser / 10])
            ray_angle_degrees = i * 2
            ray_angle_radians = math.radians(ray_angle_degrees)

            hit = False
            reached_maximum_range = False
            step_pixel = 1
            beam_length_pixel = 0
            while not hit and not reached_maximum_range:
                beam_length_pixel += step_pixel
                pixel_idx = center_cartesian_pixel + np.array(
                    [beam_length_pixel * np.cos(ray_angle_radians), beam_length_pixel * np.sin(ray_angle_radians)])
                pixel_idx = np.round(pixel_idx).astype(int)

                if pixel_idx[0] >= self._occupancy_map.shape[0] or pixel_idx[1] >= self._occupancy_map.shape[1]:
                    break

                value = self._occupancy_map[pixel_idx[0], pixel_idx[1]]
                # X, Y -> Y, X
                hit = value > 0.4
                reached_maximum_range = (beam_length_pixel * 10 >= self._max_range)
            z_star_t1_arr[i] = beam_length_pixel

        return z_star_t1_arr

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

    def ray_casting_vectorized(self, X_t1):
        num_particles = len(X_t1)
        num_beams = 360 // self._discretization
        X_body, Y_body, Yaw = X_t1[:, 0], X_t1[:, 1], X_t1[:, 2]
        X_laser = X_body + 25 * np.cos(Yaw)
        Y_laser = Y_body + 25 * np.sin(Yaw)
        X_laser_pixels = X_laser / 10
        Y_laser_pixels = Y_laser / 10

        X_laser = np.repeat(X_laser_pixels.reshape(-1, 1), num_beams, axis=1) # m, 180
        Y_laser = np.repeat(Y_laser_pixels.reshape(-1, 1), num_beams, axis=1) # m, 180

        angles = np.array(list(range(0, 360, self._discretization)))
        assert len(angles) == num_beams

        max_range_pixels = self._max_range / self._occupancy_map_resolution_centimeters_per_pixel
        beam_hit_length_pixels = np.ones_like(X_laser) * max_range_pixels
        for ray_length in range(0, int(round(max_range_pixels) + 1)):

            # The beams start from the RHS of the robot, the yaw angle is measured from the heading of the robot.
            # Hence the minus 90 degrees.
            X_beams_pixels = X_laser + \
                             np.cos(np.radians(angles) + np.repeat(Yaw.reshape(-1, 1), 360 // self._discretization, axis=1)) * ray_length
            Y_beams_pixels = Y_laser + \
                             np.sin(np.radians(angles) + np.repeat(Yaw.reshape(-1, 1), 360 // self._discretization, axis=1)) * ray_length

            X_beams_pixels = np.round(X_beams_pixels).astype(int)
            Y_beams_pixels = np.round(Y_beams_pixels).astype(int)

            X_beams_pixels = np.clip(X_beams_pixels, 0, 799)
            Y_beams_pixels = np.clip(Y_beams_pixels, 0, 799)

            occupancy_vals = self._occupancy_map[Y_beams_pixels, X_beams_pixels]

            beam_hit_length_pixels = np.minimum(beam_hit_length_pixels,
                                                np.where(occupancy_vals > self._occupancy_map_confidence_threshold, ray_length, max_range_pixels))

        Z_star_t_arr = beam_hit_length_pixels
        return Z_star_t_arr

    def beam_range_finder_model_vectorized(self, z_t1_arr, X_t1):
        Z_star_t_arr_pixels = self.ray_casting_vectorized(X_t1)

        Z_star_t_arr_cm = Z_star_t_arr_pixels * self._occupancy_map_resolution_centimeters_per_pixel

        # z_t1_arr = np.where(z_t1_arr > self._max_range, self._max_range + self._max_width, z_t1_arr)
        pdf_hit = self._z_hit * norm_pdf(z=z_t1_arr[::self._discretization], z_star=Z_star_t_arr_cm, sigma=self._sigma_hit)
        pdf_short = self._z_short * self._short_distribution.pdf(z_t1_arr[::self._discretization])
        pdf_max = self._z_max * self._max_distribution.pdf(z_t1_arr[::self._discretization])
        pdf_rand = self._z_rand * self._rand_distribution.pdf(z_t1_arr[::self._discretization])

        X = np.clip(np.round(X_t1[:, 0] / 10).astype(int), 0, 799)
        Y = np.clip(np.round(X_t1[:, 1] / 10).astype(int), 0, 799)

        belief = pdf_hit + pdf_short + pdf_max + pdf_rand
        belief = np.sum(belief, axis=1)

        # A particle cannot fall on an occupied area or an unknown area, assign zero likelihood on that.
        pruned_belief = np.where(np.logical_or(self._occupancy_map[Y, X] == -1,
                                               self._occupancy_map[Y, X] >= 0.2), 0, belief)

        return pruned_belief
