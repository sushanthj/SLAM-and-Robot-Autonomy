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

def get_ray_cast_per_square(start_pos_x, start_pos_y, grid_discretize):
    # print(f"computing grid cell {start_pos_x},{start_pos_y}")
    raycast_lookup = np.zeros((grid_discretize, grid_discretize, 360), dtype=np.int8)

    for xpos in range(raycast_lookup.shape[0]):
        for ypos in range(raycast_lookup.shape[1]):
            # print("finished yaw angle")
            # use ray casting to find the range_finders simulated readings at this robot pose

            # to make use of vectorized ray casting, init some dummy particles
            dummy_particles = np.ones((1,3))
            dummy_particles[:,0] = xpos + start_pos_x
            dummy_particles[:,1] = ypos + start_pos_y
            dummy_particles[:,2] = 0

            raycast_vals = sensor_model.ray_casting_vectorized(dummy_particles)
            raycast_lookup[xpos][ypos] = raycast_vals[0,:]

    name = os.path.join('./raycast_lookup', (str(start_pos_x)+str(start_pos_y)))
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

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)

    particle_processor = ParticleProcessor(motion_model=motion_model, sensor_model=sensor_model)
    resampler = Resampling()

    # raycast_lookup = np.zeros((occupancy_map.shape[0], occupancy_map.shape[1], 360, 180), dtype=np.float16)
    grid_discretize = 10

    # multiprocessing.set_start_method('forkserver', force=True)
    # TODO: Check which number of processes is fastest, not just max
    pool = multiprocessing.Pool(processes=8)


    for xpos in range(0, occupancy_map.shape[0], grid_discretize):
        items = [(xpos, ypos, grid_discretize) for ypos in range(0, occupancy_map.shape[1], grid_discretize)]
        ipdb.set_trace()
        pool.starmap(get_ray_cast_per_square, items)
        print("done with", xpos)
            # pool.starmap(get_ray_cast_per_square, args=(xpos, ypos, grid_discretize))
            # get_ray_cast_per_square(xpos, ypos, grid_discretize)
            # worker.get()