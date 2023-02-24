import argparse

from map_reader import MapReader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=2000, type=int)
    parser.add_argument('--decrease_factor', default=0.75, type=float)
    parser.add_argument('--num_particles_min', default=1000, type=int)
    parser.add_argument('--results_dir_suffix', default='./results', type=str)
    parser.add_argument('--seed', default=None, type=str)
    parser.add_argument('--visualize', default=True, action='store_true')

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)