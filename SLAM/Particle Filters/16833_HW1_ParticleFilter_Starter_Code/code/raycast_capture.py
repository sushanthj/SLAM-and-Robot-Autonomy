'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import argparse
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np
import sys, os

from particle_processor import ParticleProcessor
from map_reader import MapReader
from motion_model import MotionModel, limit_angle
from sensor_model import SensorModel
from resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time
import ipdb
import multiprocessing

VISUALIZE = True

def visualize_map(occupancy_map):
    fig = plt.figure(figsize=(15, 15))
    mng = plt.get_current_fig_manager()
    plt.ion()
    x_locs = [800]
    y_locs = [600]

    x_locs_2 = [400]
    y_locs_2 = [300]

    scat = plt.scatter(x_locs_2, y_locs_2, c='b', marker='o', alpha=0.2)
    d = 2
    arrow_plot = plt.quiver(x_locs_2, y_locs_2, d * np.cos(0), d * np.sin(0))

    a = np.load('./raycast_lookup/complete.npz')
    lookup = a['arr']
    look = lookup[x_locs[0], y_locs[0], :]


    xs, ys = [], []
    for angle, ray in enumerate(look):
        rad = np.radians(angle)

        print(ray)
        x = x_locs[0] + np.cos(rad - np.pi / 2) * (ray/10)
        y = y_locs[0] + np.sin(rad - np.pi / 2) * (ray/10)
        xs.append(x)
        ys.append(y)

    plt.scatter(xs, ys)
    plt.imshow(occupancy_map, cmap='Greys')
    plt.axis([0, 800, 0, 800])
    plt.waitforbuttonpress()

def get_ray_cast_per_square(start_pos_x, grid_discretize, map_shape_axis_1):
    print(f"computing grid cell {start_pos_x}")
    raycast_lookup = np.zeros((grid_discretize, map_shape_axis_1, 360), dtype=np.float16)

    for xpos in range(raycast_lookup.shape[0]):
        for ypos in range(raycast_lookup.shape[1]):
            # print("finished yaw angle")
            # use ray casting to find the range_finders simulated readings at this robot pose

            # to make use of vectorized ray casting, init some dummy particles
            dummy_particles = np.ones((1,3))
            dummy_particles[:,0] = (xpos + start_pos_x) * 10
            dummy_particles[:,1] = (ypos) * 10
            dummy_particles[:,2] = 0

            raycast_vals = sensor_model.ray_casting_vectorized_centimeters(dummy_particles)
            raycast_lookup[xpos][ypos] = raycast_vals[0,:]

    name = os.path.join('./raycast_lookup', str(start_pos_x))
    np.savez(name, arr=raycast_lookup)


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
    parser.add_argument('--num_particles', default=4000, type=int)
    parser.add_argument('--visualize', default=True, action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    if VISUALIZE:
        visualize_map(occupancy_map)

    else:
        motion_model = MotionModel()
        sensor_model = SensorModel(occupancy_map)

        particle_processor = ParticleProcessor(motion_model=motion_model, sensor_model=sensor_model)
        resampler = Resampling()

        grid_discretize = 5

        # multiprocessing.set_start_method('forkserver', force=True)
        # TODO: Check which number of processes is fastest, not just max
        pool = multiprocessing.Pool(processes=12)

        lookup_resolution = [occupancy_map.shape[0]*2, occupancy_map.shape[1]*2]
        print(f"running raycasting on map of resolution {lookup_resolution}")

        items = [(xpos, grid_discretize, lookup_resolution[1]) for xpos in range(0, lookup_resolution[0], grid_discretize)]
        pool.starmap(get_ray_cast_per_square, items)
        print("done")
        pool.join()

