'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time

import torch
# from chamferdist import ChamferDistance
from matplotlib import pyplot as plt
from scipy.stats import norm, expon, uniform

from map_reader import MapReader
from lookup_table import LookupTable

DEBUG = True

# chamfer_distance = ChamferDistance()


def norm_pdf(z, z_star, sigma):
    return np.exp(-(z - z_star) ** 2/ (2.0 * sigma ** 2) / np.sqrt(2 * np.pi * sigma ** 2))


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map, lookup=True):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """

        # 1 0.2 0.05
        # 1 0.2 0.1
        # 1 0.3 0.05
        # 1 0.3 0.1
        # 2 0.2 0.05
        # 2 0.2 0.1
        # 2 0.3 0.05
        # 2 0.3 0.1

        self._z_hit = 100 # 1, 2
        self._z_short = 0.0001 # 0.2, 0.3
        self._z_max = 1 # 0.05, 0.1
        self._z_rand = 1

        # Contextualize why you chose this.
        # self._z_hit = 0.02
        # self._z_short = 0.005 # Might be too high
        # self._z_max = 0.01
        # self._z_rand = 300 # Might want to increase it

        self._sigma_hit = 10
        self._lambda_short = 0.12 # Small values smoother curve?

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 8138

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2
        self._occupancy_map = occupancy_map
        self._occupancy_map_confidence_threshold = 0.35
        self._occupancy_map_resolution_centimeters_per_pixel = 10

        self._short_distribution = expon(scale=1 / self._lambda_short)
        self._rand_distribution = uniform(loc=0, scale=self._max_range)

        self._max_width = 100
        self._max_distribution = uniform(loc=self._max_range, scale=self._max_width)
        self._discretization = 1

        if DEBUG:
            self.visualize_probability_distribution()
            self._pdf(500, 1000)

        self._lookup = lookup
        self._lookup_table = LookupTable(discretization=self._discretization)

        self._vectorized_pdf = np.vectorize(self._pdf)

    def _pdf(self, z, z_star):
        hit_distribution = norm(loc=z_star, scale=self._sigma_hit)
        pdf_hit = self._z_hit * hit_distribution.pdf(z)
        pdf_short = self._z_short * self._short_distribution.pdf(z)
        pdf_max = self._z_max * self._max_distribution.pdf(z)
        pdf_rand = self._z_rand * self._rand_distribution.pdf(z) * self._max_range
        return pdf_hit + pdf_short + pdf_max + pdf_rand

    def visualize_probability_distribution(self):
        zs = np.linspace(0, self._max_range + self._max_width, 1000)
        z_star = 1000
        pdfs = [self._pdf(z, z_star) for z in zs]
        plt.plot(zs, pdfs)

        plt.xlim(0, self._max_range + self._max_width + 1000)
        plt.ylim(0, 1.3)

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

    def get_max_dist(self, z_t):
        pdf = np.zeros_like(z_t)
        pdf[z_t == self._max_range] = 1
        return pdf

    def get_p_rand(self, z_t):
        p_rand = np.zeros_like(z_t)
        p_rand[np.where((z_t >= 0) & (z_t < self._max_range))] = 1
        return p_rand

    def ray_casting_vectorized_centimeters(self, X_t1):
        num_particles = len(X_t1)
        num_beams = 180 // self._discretization
        X_body, Y_body, Yaw = X_t1[:, 0], X_t1[:, 1], X_t1[:, 2]
        X_laser = X_body + 25 * np.cos(Yaw)
        Y_laser = Y_body + 25 * np.sin(Yaw)

        X_laser = np.repeat(X_laser.reshape(-1, 1), num_beams, axis=1) # m, 180
        Y_laser = np.repeat(Y_laser.reshape(-1, 1), num_beams, axis=1) # m, 180

        angles = np.array(list(range(-90, 90, self._discretization)))
        assert len(angles) == num_beams

        beam_hit_length = np.ones_like(X_laser) * self._max_range
        beam_step = self._occupancy_map_resolution_centimeters_per_pixel // 2.1
        np.arange(0, self._max_range, beam_step)
        for ray_length in np.arange(0, self._max_range, beam_step):
            # The beams start from the RHS of the robot, the yaw angle is measured from the heading of the robot.
            # Hence the minus 90 degrees.
            X_beams = X_laser + np.cos(np.radians(angles) + np.repeat(Yaw.reshape(-1, 1), 180 // self._discretization, axis=1)) * ray_length
            Y_beams = Y_laser + np.sin(np.radians(angles) + np.repeat(Yaw.reshape(-1, 1), 180 // self._discretization, axis=1)) * ray_length

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

    def ray_casting_vectorized(self, X_t1):
        num_particles = len(X_t1)
        num_beams = 180 // self._discretization
        X_body, Y_body, Yaw = X_t1[:, 0], X_t1[:, 1], X_t1[:, 2]
        X_laser = X_body + 25 * np.cos(Yaw)
        Y_laser = Y_body + 25 * np.sin(Yaw)
        X_laser_pixels = X_laser / 10 # m, 1
        Y_laser_pixels = Y_laser / 10 # m, 1

        X_laser = np.repeat(X_laser_pixels.reshape(-1, 1), num_beams, axis=1) # m, 180
        Y_laser = np.repeat(Y_laser_pixels.reshape(-1, 1), num_beams, axis=1) # m, 180

        angles = np.array(list(range(-90, 90, self._discretization)))
        assert len(angles) == num_beams

        max_range_pixels = self._max_range / self._occupancy_map_resolution_centimeters_per_pixel
        beam_hit_length_pixels = np.ones_like(X_laser) * max_range_pixels
        for ray_length in range(0, int(round(max_range_pixels) + 1)):

            # The beams start from the RHS of the robot, the yaw angle is measured from the heading of the robot.
            # Hence the minus 90 degrees.
            X_beams_pixels = X_laser + \
                             np.cos(np.radians(angles) + np.repeat(Yaw.reshape(-1, 1), 180 // self._discretization, axis=1)) * ray_length
            Y_beams_pixels = Y_laser + \
                             np.sin(np.radians(angles) + np.repeat(Yaw.reshape(-1, 1), 180 // self._discretization, axis=1)) * ray_length

            X_beams_pixels = np.round(X_beams_pixels).astype(int)
            Y_beams_pixels = np.round(Y_beams_pixels).astype(int)

            X_beams_pixels = np.clip(X_beams_pixels, 0, 799)
            Y_beams_pixels = np.clip(Y_beams_pixels, 0, 799)
            #
            occupancy_vals = self._occupancy_map[Y_beams_pixels, X_beams_pixels]
            # occupancy_vals = self._occupancy_map[X, Y]

            beam_hit_length_pixels = np.minimum(beam_hit_length_pixels,
                                                np.where(occupancy_vals > self._occupancy_map_confidence_threshold, ray_length + 1, max_range_pixels))

        Z_star_t_arr = beam_hit_length_pixels
        return Z_star_t_arr

    def ray_casting_beam_alive(self, X_t1):
        num_particles = len(X_t1)
        num_beams = 180 // self._discretization
        X_body, Y_body, Yaw = X_t1[:, 0], X_t1[:, 1], X_t1[:, 2]
        X_laser = X_body + 25 * np.cos(Yaw)
        Y_laser = Y_body + 25 * np.sin(Yaw)

        X_laser = np.repeat(X_laser[:, None], num_beams, axis=1)
        Y_laser = np.repeat(Y_laser[:, None], num_beams, axis=1)
        angles = np.radians(np.array(list(range(-90, 90, self._discretization))))
        Yaw_laser = np.repeat(angles[None, :], num_particles, axis=0) + Yaw[:, None]

        ray_particles = np.zeros((num_particles, 3, num_beams))
        ray_particles[:, 0, :] = X_laser
        ray_particles[:, 1, :] = Y_laser
        ray_particles[:, 2, :] = Yaw_laser
        ray_particles = ray_particles.reshape(-1, 3)

        live_beams = np.ones(num_particles * num_beams, dtype=bool)
        ray_length = np.zeros_like(live_beams, dtype=float)
        beam_step = self._occupancy_map_resolution_centimeters_per_pixel / 2

        while np.sum(live_beams):
            ray_particles[live_beams, 0] += beam_step * np.cos(ray_particles[live_beams, 2])
            ray_particles[live_beams, 1] += beam_step * np.sin(ray_particles[live_beams, 2])
            ray_length[live_beams] += beam_step

            # Inside the playing field.
            live_beams_indices = np.where(live_beams)[0]
            xs = np.round(ray_particles[live_beams, 0] / 10).astype(int)
            ys = np.round(ray_particles[live_beams, 0] / 10).astype(int)
            inside_x = np.logical_and(xs >= 0, xs < self._occupancy_map.shape[1])
            inside_y = np.logical_and(ys >= 0, ys < self._occupancy_map.shape[0])
            inside = np.logical_and(inside_x, inside_y)
            live_beams[live_beams_indices[np.where(np.logical_not(inside))]] = False

            # Occupancy
            live_beams_indices = live_beams_indices[inside]
            inside_indices = np.where(inside)[0]
            x_inside, y_inside = xs[inside_indices], ys[inside_indices]
            occupied = [[self._occupancy_map[y_inside, x_inside] > self._occupancy_map_confidence_threshold],
                        [self._occupancy_map[y_inside, x_inside] == -1]]
            live_beams[live_beams_indices[np.where(occupied)[0]]] = False

            # Max range
            too_far = np.where(ray_length > self._max_range)[0]
            live_beams[too_far] = False
            ray_length[too_far] = self._max_range

        zt_star = ray_length.reshape(num_particles, num_beams)
        return zt_star

    def lookup(self, X_t1):
        xs = X_t1[:, 0]
        ys = X_t1[:, 1]
        yaws = X_t1[:, 2]
        Z_star_t_arr_cm = self._lookup_table.lookup(xs, ys, yaws)

        return Z_star_t_arr_cm

    def ray_casting_vectorized_write(self, X_t1):
        num_particles = len(X_t1)
        num_beams = 360
        X_body, Y_body, Yaw = X_t1[:, 0], X_t1[:, 1], X_t1[:, 2]
        X_laser = X_body + 25 * np.cos(Yaw)
        Y_laser = Y_body + 25 * np.sin(Yaw)
        X_laser_pixels = X_laser / 10 # m, 1
        Y_laser_pixels = Y_laser / 10 # m, 1

        X_laser = np.repeat(X_laser_pixels.reshape(-1, 1), num_beams, axis=1) # m, 180
        Y_laser = np.repeat(Y_laser_pixels.reshape(-1, 1), num_beams, axis=1) # m, 180

        angles = np.array(list(range(0, 360, 1)))
        assert len(angles) == num_beams

        max_range_pixels = self._max_range / self._occupancy_map_resolution_centimeters_per_pixel
        beam_hit_length_pixels = np.ones_like(X_laser) * max_range_pixels
        for ray_length in range(0, int(round(max_range_pixels) + 1)):
            # The beams start from the RHS of the robot, the yaw angle is measured from the heading of the robot.
            # Hence the minus 90 degrees.
            X_beams_pixels = X_laser + \
                             np.cos(np.radians(angles) + np.repeat(Yaw.reshape(-1, 1), 360, axis=1)) * ray_length
            Y_beams_pixels = Y_laser + \
                             np.sin(np.radians(angles) + np.repeat(Yaw.reshape(-1, 1), 360, axis=1)) * ray_length

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
        # Z_star_t_arr_pixels = self.ray_casting_vectorized(X_t1)
        # Z_star_t_arr_cm = Z_star_t_arr_pixels * self._occupancy_map_resolution_centimeters_per_pixel

        if self._lookup:
            xs = X_t1[:, 0]
            ys = X_t1[:, 1]
            yaws = X_t1[:, 2]
            Z_star_t_arr_cm = self._lookup_table.lookup(xs, ys, yaws)
        else:
            Z_star_t_arr_cm = self.ray_casting_vectorized_centimeters(X_t1)
        # Z_star_t_arr_cm = self.ray_casting_beam_alive(X_t1)

        pdf_hit = self._z_hit * norm_pdf(z=z_t1_arr[::self._discretization], z_star=Z_star_t_arr_cm, sigma=self._sigma_hit)
        pdf_short = self._z_short * self._short_distribution.pdf(z_t1_arr[::self._discretization])
        # pdf_max = self._z_max * self._max_distribution.pdf(z_t1_arr[::self._discretization])
        pdf_max = self._z_max * self.get_max_dist(z_t1_arr[::self._discretization])
        pdf_rand = self._z_rand * self.get_p_rand(z_t1_arr[::self._discretization])
        # pdf_rand = self._z_rand * self._rand_distribution.pdf(z_t1_arr[::self._discretization])

        X = np.clip(np.round(X_t1[:, 0] / 10).astype(int), 0, 799)
        Y = np.clip(np.round(X_t1[:, 1] / 10).astype(int), 0, 799)

        belief = pdf_hit + pdf_short + pdf_max + pdf_rand
        print("belief shape before", belief.shape)

        # belief = np.sum(np.log(belief), axis=1)
        # belief = np.exp(belief)

        belief = np.sum(belief, axis=1)
        print("belief shape after", belief.shape)

        # A particle cannot fall on an occupied area or an unknown area, assign zero likelihood on that.
        pruned_belief = np.where(np.logical_or(self._occupancy_map[Y, X] == -1,
                                               self._occupancy_map[Y, X] >= 0.45), 0, belief)

        return pruned_belief
