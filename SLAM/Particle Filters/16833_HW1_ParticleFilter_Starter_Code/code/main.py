'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
from multiprocessing import Pool, cpu_count
from random import random, randrange
from typing import Optional

import numpy as np
import sys, os

from particle_processor import ParticleProcessor
from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time


def visualize_map(occupancy_map):
    fig = plt.figure(figsize=(15, 15))
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, z_t, output_path, sensor_model: Optional[SensorModel] = None):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0

    if z_t is not None:
        x_est = x_locs.mean()
        y_est = y_locs.mean()
        yaws = X_bar[:, 2]
        yaw_est = yaws.mean()

        # plt.arrow(x_est, y_est, 10 * np.cos(yaw_est), 10 * np.sin(yaw_est))
        if sensor_model is not None:

            # angles = np.radians(np.array(list(range(-90, 90, 1)))) + np.repeat(X_bar[:50, 2].reshape(-1, 1), 180, axis=1)
            # z_star_pixels = sensor_model.ray_casting_vectorized(X_bar[:50])
            # x_beams = np.repeat(X_bar[:50, 0].reshape(-1, 1), 180, axis=1) + z_star_pixels * 10 * np.cos(angles)
            # y_beams = np.repeat(X_bar[:50, 1].reshape(-1, 1), 180, axis=1) + z_star_pixels * 10 * np.sin(angles)

            # Test case.
            yaw = X_bar[:, 2].mean()
            X_t1_test = np.array([[X_bar[:, 0].mean(), X_bar[:, 1].mean(), yaw]])
            angles = np.radians(np.array(list(range(-90, 90, sensor_model._discretization)))) + X_bar[:, 2].mean()  # Might be this?
            Z_star_t_arr = sensor_model.lookup(X_t1_test)
            x_beams = X_t1_test[0][0] + Z_star_t_arr[0] * np.cos(angles) * 10
            y_beams = X_t1_test[0][1] + Z_star_t_arr[0] * np.sin(angles) * 10

            # yaw = X_bar[:, 2].mean()
            # yaw = limit_angle(yaw)
            # X_t1_test = np.array([[6500, 1500, 90]])
            # angles = np.radians(np.array(list(range(-90, 90, sensor_model._discretization)))) + X_t1_test[:, 2]  # Might be this?
            # Z_star_t_arr = sensor_model.ray_casting_vectorized(X_t1_test)
            # x_beams = X_t1_test[0][0] + Z_star_t_arr[0] * np.cos(angles) * 10
            # y_beams = X_t1_test[0][1] + Z_star_t_arr[0] * np.sin(angles) * 10

            # Convert to pixel coordinates
            x_beams /= 10
            y_beams /= 10

        print(yaw_est)
        angles = np.radians(np.array(list(range(-90, 90, 1)))) + yaw_est
        x_est = np.repeat(x_est.reshape(-1, 1), 180, axis=1)
        y_est = np.repeat(y_est.reshape(-1, 1), 180, axis=1)

        x_beams_est = x_est + z_t * np.cos(angles) / 10 # z_t in cm, x_est in pixels
        y_beams_est = y_est + z_t * np.sin(angles) / 10

        # Beams
        scat_3 = plt.scatter(x_beams, y_beams, c='g', marker='x', alpha=0.2)
        scat_4 = plt.scatter(x_beams_est, y_beams_est, c='r', marker='x', alpha=0.2)

    angles = np.radians(np.array(list(range(0, 360, 2))))
    x_est + z_t * np.cos(angles)
    y_est + z_t * np.sin(angles)
    # Particles
    scat = plt.scatter(x_locs, y_locs, c='b', marker='o', alpha=0.2)
    scat_2 = plt.scatter([x_locs.mean()], [y_locs.mean()], c='r', marker='o', alpha=0.2)

    d = 2
    arrow_plot = plt.quiver(x_locs, y_locs, d * np.cos(yaws), d * np.sin(yaws))
    # Belief

    plt.pause(0.0000001)
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))

    arrow_plot.remove()
    scat.remove()
    scat_2.remove()

    if z_t is not None:
        scat_3.remove()
        scat_4.remove()


def init_particles(num_particles, occupancy_map):
    seed = 12388 # 1.28 AM
    seed = 103 #2.42 PM,
    # seed = 1134099 #2.42 PM,
    # seed = 1134099 #4.49 PM, 20000 WITH ANNEALING
    # seed = 7 #2.42 PM,

    empty = np.argwhere(np.logical_and(occupancy_map < 0.15, occupancy_map != -1))
    # empty = empty[np.logical_and(empty[:, 0] < 500, empty[:, 0] > 320)]
    # empty = empty[np.logical_and(empty[:, 1] < 525, empty[:, 0] > 350)]
    indices = np.random.choice(list(range(len(empty))), num_particles)

    # resolution.
    xs = empty[indices, 1].reshape(-1, 1) * 10
    ys = empty[indices, 0].reshape(-1, 1) * 10

    angles = np.ones((num_particles, 1)) * np.radians(np.random.uniform(170, 190) - 5)
    which_axis = np.random.randint(0, 4, num_particles)
    #
    angles[which_axis == 0] = np.radians(np.random.uniform(-5, 5) - 5)
    angles[which_axis == 1] = np.radians(np.random.uniform(85, 95) - 5)
    angles[which_axis == 2] = np.radians(np.random.uniform(175, 185) - 5)
    angles[which_axis == 3] = np.radians(np.random.uniform(265, 275) - 5)

    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    # change this to theta vals
    X_bar_init = np.hstack((xs, ys, angles, w0_vals))
    return X_bar_init


def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform(0, 7000, (num_particles, 1))
    x0_vals = np.random.uniform(3000, 7000, (num_particles, 1))

    theta0_vals = np.random.uniform(-3.14, 3.14, (num_particles, 1))

    # initialize weights for all particles
    w0_vals = np.ones((num_particles, 1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w0_vals))

    return X_bar_init


def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    """
    TODO : Add your code here
    This version converges faster than init_particles_random
    """
    X_bar_init = np.zeros((num_particles, 4))

    return X_bar_init


if __name__ == '__main__':
    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """
    """
    Initialize Parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=5000, type=int)
    parser.add_argument('--decrease_factor', default=0.95, type=float)
    parser.add_argument('--num_particles_min', default=1000, type=int)
    parser.add_argument('--results_dir_suffix', default='./results', type=str)
    parser.add_argument('--seed', default=20098, type=str)
    parser.add_argument('--visualize', default=True, action='store_true')
    args = parser.parse_args()

    if args.seed is None:
        seed = np.random.randint(0, 100000, 1)
    else:
        seed = [args.seed]

    np.random.seed(seed)

    results_dir = f'{args.results_dir_suffix}_{seed[0]}'
    dirname = os.path.join(results_dir, 'code')
    os.makedirs(dirname)
    os.system(
        f"cp *.py {dirname}"
    )

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)

    particle_processor = ParticleProcessor(motion_model=motion_model, sensor_model=sensor_model)
    resampler = Resampling()

    num_particles = args.num_particles
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles(num_particles, occupancy_map)

    # X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True
    ranges = None
    moving = True

    for time_idx, line in enumerate(logfile):

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):
            # [x, y, theta] coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6]
            # 180 range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.ones((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.

        # X_t1 = motion_model.update_vectorized(u_t0, u_t1, X_bar[:, :3])
        # if meas_type != 'L':
        #     X_bar_new[:, :3] = X_t1
        #     X_bar_new[:, 3] = X_bar[:, 3]
        # X_t1 = []
        for m in range(0, num_particles):
            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)
            # X_t1.append(x_t1)
            X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        # X_t1 = np.array(X_t1)

        """
        VECTORIZED IMPLEMENTATION.
        """
        print("Moving by some", np.linalg.norm(u_t1[:2] - u_t0[:2]))
        threshold = 0.1
        if np.linalg.norm(u_t1[:2] - u_t0[:2]) > threshold:
            print("MOVING")
            moving = True

        if meas_type == "L":
            # Update the belief
            z_t = ranges
            # X_bar_new[:, :3] = X_t1

            if moving:
                W_t = sensor_model.beam_range_finder_model_vectorized(z_t, X_bar_new[:, :3])
                X_bar_new[:, 3] = W_t

        # """
        # SENSOR MODEL
        # """
        # if (meas_type == "L"):
        #     z_t = ranges
        #     # print(f"Processing {m} beam range finder.")
        #     w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
        #     X_bar_new[m, :] = np.hstack((x_t1, w_t))
        # else:
        #     X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        if moving:
            X_bar = resampler.low_variance_sampler(X_bar)

        if meas_type == 'L' and moving:
            top_k = int(args.decrease_factor * len(X_bar))
            num_particles = top_k

            if num_particles < args.num_particles_min:
                num_particles = args.num_particles_min

            X_bar = np.array(sorted(X_bar, key=lambda x: x[3])[::-1])
            X_bar = X_bar[:num_particles, :]

        if args.visualize:
            if ranges is not None:
                z_t = ranges
                visualize_timestep(X_bar, time_idx, z_t, results_dir, sensor_model=sensor_model)
            else:
                visualize_timestep(X_bar, time_idx, None, results_dir, sensor_model=sensor_model)
