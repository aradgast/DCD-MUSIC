"""
TODO: Add description
"""
import torch
import numpy as np
from typing import Tuple
import os
import pandas as pd

from FR3.src.observations.band import Band, Bands
from FR3.utils.constants import *
from FR3.utils.global_functions import create_bs_locs, create_scatter_points, calc_power, compute_path_loss

class Channel:
    def __init__(self, synthetic: bool):
        self.toas = None
        self.doas = None
        self.zoas = None
        self.bs_loc = None
        self.powers = None
        self.num_paths = None
        self.medium_speed = None
        self.is_synthetic = synthetic

    def init_channel(self, bs_ind: int, ue_pos, band, scatterers=None):
        ue_pos = np.array(ue_pos)
        if self.is_synthetic:
            self.bs_loc = create_bs_locs(bs_ind)
            scatterers = create_scatter_points(SYNTHETIC_L_MAX)
            self.__generate_synthetic_params(self.bs_loc, ue_pos, scatterers, band)
        else:
            self.__load_ny_scenario(bs_ind, ue_pos, band)

    def get_bs_loc(self):
        return self.bs_loc

    def get_medium_speed(self):
        return self.medium_speed

    def get_toas(self):
        return self.toas

    def get_doas(self):
        return self.doas

    def get_powers(self):
        return self.powers

    def __load_ny_scenario(self, bs_ind, ue_pos, band: Band):
        """
        Generate the parameters for each of the L_MAX paths in the current bs-ue link.
        Each path includes the toa, aoa and power.
        Args:
            bs_ind:
            ue_pos:
            band:

        Returns:

        """
        csv_path = os.path.join(DATA_DIR, band.get_fc_file_name(), f"bs_{str(bs_ind)}.csv")
        csv_loaded = pd.read_csv(csv_path)
        row_ind = csv_loaded.index[(csv_loaded[['rx_x', 'rx_y']] == ue_pos).all(axis=1)].item()
        row = csv_loaded.iloc[row_ind]
        if row['link state'] != 1:
            raise ValueError("NLOS location! currently supporting only LOS")
        bs_loc = np.array(row[['tx_x', 'tx_y']]).astype(float)
        n_paths = row['n_path'].astype(int)
        powers, toas, doas = [], [], []
        medium_speed = C
        for path in range(1, n_paths + 1):
            initial_power = INIT_POWER
            loss_db = row[f'path_loss_{path}']
            received_power = initial_power - loss_db
            toa = row[f'delay_{path}']
            if path == 1:
                medium_speed = np.linalg.norm(ue_pos - bs_loc) / toa
            # If path time is above the maximal limit
            if toa > band.k / band.bw:
                continue
            doa = np.radians(row[f'aod_{path}'])
            normalized_doa = doa - BS_ORIENTATION
            # The base station can see 90 degrees to each side of its orientation
            if -np.pi / 2 < normalized_doa < np.pi / 2:
                powers.append(received_power), toas.append(toa), doas.append(normalized_doa)


        assert all([toas[l] < band.get_k() / (band.get_bw() / 1e6) for l in range(len(toas))])

        self.doas = doas
        self.toas = toas
        self.powers = powers
        self.medium_speed = medium_speed
        self.num_paths = n_paths
        self.bs_loc = bs_loc





    def __generate_synthetic_params(self, bs_loc, ue_pos, scatterers, band):
        toas = [0 for _ in range(SYNTHETIC_L_MAX)]
        doas = [0 for _ in range(SYNTHETIC_L_MAX)]
        powers = [0 for _ in range(SYNTHETIC_L_MAX)]
        ue_pos = torch.from_numpy(ue_pos).to(DEVICE).to(torch.float64)
        bs_loc = torch.from_numpy(bs_loc).to(DEVICE).to(torch.float64)
        scatterers = torch.from_numpy(scatterers).to(DEVICE).to(torch.float64)
        toas[0] = (torch.linalg.norm(ue_pos - bs_loc) / C)
        doas[0] = (torch.arctan2(ue_pos[1] - bs_loc[1], ue_pos[0] - bs_loc[0]))
        initial_power = np.sqrt(0.5) * (1 + 1j)
        powers[0] = calc_power(initial_power, bs_loc, ue_pos, band.fc) / compute_path_loss(toas[0], band.fc)
        for l in range(1, SYNTHETIC_L_MAX):
            doas[l] = torch.arctan2(scatterers[l - 1, 1] - bs_loc[1], scatterers[l - 1, 0] - bs_loc[0])
            toas[l] = ((torch.linalg.norm(bs_loc - scatterers[l - 1]) + torch.linalg.norm(ue_pos - scatterers[l - 1])) / C)
            initial_power = np.sqrt(1 / 2) * (1 + 1j)
            powers[l] = calc_power(calc_power(initial_power, bs_loc, scatterers[l - 1], band.fc), scatterers[l - 1],
                                   ue_pos, band.fc) / compute_path_loss(toas[l], band.fc)
        assert all([toas[l] < band.get_k() / (band.get_bw() / 1e6) for l in range(len(toas))])

        self.toas = toas
        self.doas = doas
        self.powers = powers
        self.num_paths = SYNTHETIC_L_MAX
        self.bs_loc = bs_loc


class Channels:
    def __init__(self, synthetic: bool):
        self.channels = []
        self.synthetic = synthetic

    def __iter__(self):
        return iter(self.channels)

    def init_channels(self, bs_inds, ue_pos, bands: Bands):
        for idx, bs_ind in enumerate(bs_inds):
            channel = Channel(self.synthetic)
            channel.init_channel(bs_ind, ue_pos, bands.get_band(idx))
            self.channels.append(channel)

    def get_channels(self):
        return self.channels

    def get_channel(self, index):
        return self.channels[index]

    def get_num_channels(self):
        return len(self.channels)

    def get_bs_locs(self):
        return [channel.get_bs_loc() for channel in self.channels]

    def get_medium_speeds(self):
        return [channel.get_medium_speed() for channel in self.channels]

    def get_toas(self):
        return [channel.get_toas() for channel in self.channels]

    def get_doas(self):
        return [channel.get_doas() for channel in self.channels]

    def get_powers(self):
        return [channel.get_powers() for channel in self.channels]