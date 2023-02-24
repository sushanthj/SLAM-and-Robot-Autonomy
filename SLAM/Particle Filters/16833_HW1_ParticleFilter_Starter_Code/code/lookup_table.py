import numpy as np
from matplotlib import pyplot as plt


class LookupTable:
    def __init__(self,
                 filename: str = "./raycast_lookup/complete.npz",
                 discretization: int = 1):
        self._filename = filename
        self._discretization = discretization

        binary = np.load(self._filename)
        self._resolution = 5
        self._arr = binary['arr']

    def lookup(self, xs, ys, yaws):
        num_beams = 180 // self._discretization
        num_particles = len(xs)

        yaws_degrees = np.round(np.degrees(yaws)).astype(int)

        xs_pixels = np.round(xs / self._resolution).astype(int)
        ys_pixels = np.round(ys / self._resolution).astype(int)

        degrees_range = np.repeat(np.array(list(range(-90, 90, 1))).reshape(-1, num_beams), num_particles, axis=0)
        degrees_to_query = yaws_degrees[:, None] + degrees_range

        degrees_to_query %= 360
        rays = self._arr[xs_pixels, ys_pixels]
        rays = rays.reshape(-1)

        indices_offset = np.array(range(num_particles)) * 360
        indices_to_query = indices_offset[:, None] + degrees_to_query
        indices_to_query = indices_to_query.reshape(-1)

        queried_rays = rays[indices_to_query]
        queried_rays = queried_rays.reshape(num_particles, num_beams)

        angles = np.radians(np.array(list(range(0, 180, 1))))

        debug = False
        if debug:
            rays_1 = queried_rays[0]
            rays_2 = queried_rays[1]

            xs = np.cos(angles) * rays_1
            ys = np.sin(angles) * rays_1

            plt.scatter(xs, ys)
            plt.show()

            xs = np.cos(angles) * rays_2
            ys = np.sin(angles) * rays_2

            plt.scatter(xs, ys)
            plt.show()

        return queried_rays
