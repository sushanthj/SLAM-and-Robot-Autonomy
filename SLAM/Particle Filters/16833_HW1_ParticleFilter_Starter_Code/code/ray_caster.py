import torch
from torch import nn


class RayCaster(nn.Module):
    def __init__(self, occupancy_map, discretization, max_range, resolution, occupancy_map_confidence_threshold):
        super().__init__()
        self._occupancy_map = occupancy_map
        self._discretization = discretization
        self._max_range = max_range
        self._occupancy_map_resolution_centimeters_per_pixel = resolution
        self._occupancy_map_confidence_threshold = occupancy_map_confidence_threshold

    def forward(self, X_t1):
        num_particles = len(X_t1)
        num_beams = 180 // self._discretization
        X_body, Y_body, Yaw = X_t1[:, 0], X_t1[:, 1], X_t1[:, 2]
        X_laser = X_body + 25 * torch.cos(Yaw)
        Y_laser = Y_body + 25 * torch.sin(Yaw)
        X_laser_pixels = X_laser / 10  # m, 1
        Y_laser_pixels = Y_laser / 10  # m, 1

        X_laser = torch.repeat_interleave(X_laser_pixels.reshape(-1, 1), num_beams, dim=1)  # m, 180
        Y_laser = torch.repeat_interleave(Y_laser_pixels.reshape(-1, 1), num_beams, dim=1)  # m, 180

        angles = torch.Tensor(list(range(-90, 90, self._discretization))).cuda()
        assert len(angles) == num_beams

        max_range_pixels = self._max_range / self._occupancy_map_resolution_centimeters_per_pixel
        beam_hit_length_pixels = torch.ones_like(X_laser).cuda() * max_range_pixels
        for ray_length in range(0, int(round(max_range_pixels) + 1)):
            # The beams start from the RHS of the robot, the yaw angle is measured from the heading of the robot.
            # Hence the minus 90 degrees.
            X_beams_pixels = X_laser + \
                             torch.cos(torch.deg2rad(angles) + torch.repeat_interleave(Yaw.reshape(-1, 1), 180 // self._discretization,
                                                                   dim=1)) * ray_length
            Y_beams_pixels = Y_laser + \
                             torch.sin(torch.deg2rad(angles) + torch.repeat_interleave(Yaw.reshape(-1, 1), 180 // self._discretization,
                                                                   dim=1)) * ray_length

            X_beams_pixels = torch.round(X_beams_pixels).int()
            Y_beams_pixels = torch.round(Y_beams_pixels).int()

            X_beams_pixels = torch.clip(X_beams_pixels, 0, 799)
            Y_beams_pixels = torch.clip(Y_beams_pixels, 0, 799)

            occupancy_vals = torch.Tensor(self._occupancy_map[Y_beams_pixels.cpu(), X_beams_pixels.cpu()]).cuda()

            beam_hit_length_pixels = torch.minimum(beam_hit_length_pixels,
                torch.where(occupancy_vals > self._occupancy_map_confidence_threshold, ray_length, max_range_pixels))

        Z_star_t_arr = beam_hit_length_pixels
        return Z_star_t_arr
