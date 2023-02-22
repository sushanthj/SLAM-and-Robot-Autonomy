import multiprocessing
from typing import Optional, Any
import motion_model

import numpy as np


class ParticleProcessor:
    def __init__(self, motion_model, sensor_model):
        self._motion_model = motion_model
        self._sensor_model = sensor_model

        cpu_count = multiprocessing.cpu_count()
        self._pool = multiprocessing.Pool(cpu_count)

        self._u_t0: Optional[Any] = None
        self._u_t1: Optional[Any] = None
        self._meas_type: Optional[Any] = None
        self._ranges: Optional[Any] = None
        self._X_bar: Optional[Any] = None

    def single_process(self, x_bar_m, m):
        assert self._u_t0 is not None
        assert self._u_t1 is not None
        assert self._meas_type is not None
        assert self._ranges is not None

        x_t1 = self._motion_model.update(self._u_t0, self._u_t1, x_bar_m[:3])

        if self._meas_type == 'L':
            z_t = self._ranges
            print(f"Processing {m} beam range finder.")
            w_t = self._sensor_model.beam_range_finder_model(z_t, x_t1)
            x_bar_new = np.hstack((x_t1, w_t))
        else:
            x_bar_new = np.hstack((x_t1, x_bar_m[3]))
        return x_bar_new, m

    def pool_process(self, X_bar, u_t0, u_t1, meas_type, ranges):
        self._u_t1 = u_t1
        self._u_t0 = u_t0
        self._meas_type = meas_type
        self._ranges = ranges

        results = self._pool.map(self.single_process, (list(X_bar), list(range(len(X_bar)))))

        return results



# def particle_processor(m, X_bar, meas_type, ranges, ):
#     x_t0 = X_bar[m, 0:3]
#     x_t1 = motion_model.update(u_t0, u_t1, x_t0)

#     """
#     SENSOR MODEL
#     """
#     if (meas_type == "L"):
#         z_t = ranges
#         print(f"Processing {m} beam range finder.")
#         w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
#         X_bar_new[m, :] = np.hstack((x_t1, w_t))
#     else:
#         X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))