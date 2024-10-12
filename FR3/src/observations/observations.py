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
        # Phase delay for the K sub-carriers for all paths
        time_base = band.compute_time_steering(channel.toas).to(DEVICE).to(torch.complex128)
        # Different phase in each antenna element
        angle_base = band.compute_angle_steering(channel.doas).to(DEVICE).to(torch.complex128).transpose(0, 1)
        steering_mat = torch.einsum("ln, lt -> lnt", angle_base, time_base)
        # Assume random phase beamforming
        F = torch.exp(1j * torch.from_numpy(np.random.rand(channel.num_paths, 1, 1, self.ns)) * 2 * torch.pi).to(
            DEVICE).to(torch.complex128)
        steering_mat = torch.einsum("lntj, lnt -> lntj", F, steering_mat)
        channel_response = torch.sum(
            (power_from_dbm(channel.powers) / noise_amp).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(
                DEVICE) * steering_mat, dim=0)
        # Add white Gaussian noise, use numpy for the random numbers
        normal_gaussian_noise = (1 / np.sqrt(2) *
                                 (torch.from_numpy(np.random.randn(band.n, band.k, self.ns))
                                  + 1j * torch.from_numpy(np.random.randn(band.n, band.k, self.ns)))).to(device=DEVICE).to(torch.complex128)
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

