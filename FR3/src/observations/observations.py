"""
TODO: add description
"""

from FR3.src.observations.channel import Channel
from FR3.src.observations.band import Band
from FR3.utils.constants import *
from FR3.utils.global_functions import power_from_dbm

import torch



class Observations:
    def __init__(self, data):
        self.data = data
        self.nf = NOISE_FIGURE
        self.n0 = N_0
        self.ns = NS

    # TODO: need to think of a way to make it faster
    def init_observations(self, channel: Channel, band: Band):
        """
        Initialize the observations for the given channel and band.
        Args:
            channel:
            band:

        Returns:

        """
        bw_loss = 10 * np.log10(band.get_bw())
        noise_amp = power_from_dbm(self.nf + bw_loss + self.n0)
        channel_response = torch.zeros((band.n, band.k, self.ns), dtype=torch.complex128)
        for l in range(channel.num_paths):
            # Assume random phase beamforming
            F = torch.exp(1j * torch.from_numpy(np.random.rand(1, 1, self.ns)) * 2 * torch.pi)
            # Phase delay for the K sub-carriers
            try:
                time_base = band.compute_time_steering(channel.toas[l])
            except IndexError:
                pass
            # Different phase in each antenna element
            angle_base = band.compute_angle_steering(channel.doas[l])

            angle_base = torch.from_numpy(angle_base).to(DEVICE).to(torch.complex128)
            time_base = torch.from_numpy(time_base).to(DEVICE).to(torch.complex128)

            steering_mat = torch.matmul(angle_base, time_base)
            steering_mat = F * steering_mat.unsqueeze(-1)
            channel_response += power_from_dbm(channel.powers[l]) / noise_amp * steering_mat
        # Add white Gaussian noise, use numpy for the random numbers
        normal_gaussian_noise = (1 / np.sqrt(2) *
                                 (torch.from_numpy(np.random.randn(band.n, band.k, self.ns))
                                  + 1j * torch.from_numpy(np.random.randn(band.n, band.k, self.ns))))
        self.data = channel_response + normal_gaussian_noise

    def set_observations(self, data):
        self.data = data

    def add_observation(self, observation):
        self.data.append(observation)

    def get_observations(self):
        return self.data

    def get_observation(self, index):
        return self.data[index]

    def get_observation_count(self):
        return len(self.data)

    def clear_observations(self):
        self.data = []

    def print_observations(self):
        for observation in self.data:
            print(observation)

