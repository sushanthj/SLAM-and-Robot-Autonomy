'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
import numpy as np
import sys, os

from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time
import ipdb


def visualize_map(occupancy_map):
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    plt.ion()
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    # plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    # plt.xlim([-50, 50])
    # plt.ylim([-50, 50])
    plt.pause(0.00001)
    scat.remove()


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
    parser.add_argument('--num_particles', default=500, type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    # reads the wean.dat which is map of Wean hall of size 8000x8000 and stored as an
    # occupancy grid. The grid cells are spaced 10cm apart, therefore there are 800x800 cells
    map_obj = MapReader(src_path_map)
    # returns occupancy map in the correct structure we need (a numpy array of size 800x800)
    # each element of this matrix represents the probability of that points being occupied.
    # if you play around, you'll see there are many points with the probability ~0.1 to ~0.8,
    # but there are no grid cells with probability == 1.
    # probabilty == -1 tells us we have no clue about whats in that cell
    occupancy_map = map_obj.get_map()
    ipdb.set_trace()

    # contains information on robot sensor readings (laser and odometry) at each timestep
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = args.num_particles
    X_bar = init_particles_random(num_particles, occupancy_map)
    # X_bar = init_particles_freespace(num_particles, occupancy_map)
    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if args.visualize:
        visualize_map(occupancy_map)

    first_time_idx = True
    for time_idx, line in enumerate(logfile):

        ipdb.set_trace()

        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        # L : laser scan measurement, O : odometry measurement
        meas_type = line[0]

        # convert measurement values from string to double
        """
        - meas_vals is a numpy array of size 187. The first 3 elements are the odometry reading
        - the last element of meas_vals is the time_stamp
        - The 3:6 th elements of meas_vals are the x,y,thetha of laser in odometry frame
        - You can ignore this 3:6 th elements as you can assume a fixed offset of the laser
        """
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')

        # odometry reading [x, y, theta] in odometry frame
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1] # size=1

        # ignore pure odometry measurements for (faster debugging)
        # if ((time_stamp <= 0.0) | (meas_type == "O")):
        #     continue

        if (meas_type == "L"):

            #! You can ignore this odometry laser but it basically is = [x, y, theta]
            #! Instead use the information that laser is 25 cm offset forward (forward = x-axis)
            # NOTE: odometry_laser is coordinates of laser in odometry frame
            odometry_laser = meas_vals[3:6] # ignore

            # NOTE: array of size=180, range measurement values from single laser scan
            ranges = meas_vals[6:-1]

        print("Processing time step {} at time {}s".format(
            time_idx, time_stamp))

        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        # init random new particles of size num_particles
        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        # this is the odometry update (from robot odometry sensors).
        # This update is applied to each particle (i.e. it is applied at each iteration of the for loop)
        u_t1 = odometry_robot # size=3

        # Note: this formulation is intuitive but not vectorized; looping in python is SLOW.
        # Vectorized version will receive a bonus. i.e., the functions take all particles as the input and process them in a vector.
        for m in range(0, num_particles):
            """
            MOTION MODEL (predict how the particle would move + add some noise)
            """
            # original particle position
            x_t0 = X_bar[m, 0:3]
            # update particle position according to motion model
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            """
            SENSOR MODEL (Need to do some ray-casting to determing p(z_t | [x_t, Map])
            """
            if (meas_type == "L"):
                # ranges is size=180, represents the robot's sensor reading
                z_t = ranges
                # weigh each particle according to p(z_t | [x_t, Map])
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1) # Line 5
                # append these weights to each particle (we'll resample below)
                X_bar_new[m, :] = np.hstack((x_t1, w_t)) # Line 6
            else:
                X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING (update which particles we want to finally use, we are refining our predictions)

        We use weights w_t associated with each particle to resample
        """
        X_bar = resampler.low_variance_sampler(X_bar)

        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)
            pass
