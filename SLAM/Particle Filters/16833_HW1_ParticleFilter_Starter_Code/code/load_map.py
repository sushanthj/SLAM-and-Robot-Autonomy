from typing import Optional

import numpy as np


class RayCastLookup:
    def __init__(self,
                 data_str: str = '../data/complete.npz',
                 occupancy_map: Optional[np.ndarray] = None):
        self._data_str = data_str
        self._lookup_table = np.load(self._data_str)['arr']

    def get_data(self, x_arr, y_arr, yaw_arr_rad):
        """
        :param x_arr: in cartesian coordinates
        :param y_arr: in cartesian coordinates
        :param yaw_arr_rad:
        :return:
        """

        yaw_arr_deg = np.round(np.degrees(yaw_arr_rad)).int()

        self._lookup_table[x_arr]



if __name__ == "__main__":
    a = np.load('../data/complete.npz')
    lookup_table = a['arr']
