"""
NY Dataset - This file contains the class for the NY Dataset.
The data contains observations from the NY scenario.
"""
import os
import pandas as pd
from pandas.errors import DataError
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torch

from FR3.src.observations.band import Bands
from FR3.src.observations.observations import Observations
from FR3.src.observations.channel import Channel
from FR3.utils.constants import C

class NYDataset(Dataset):
    def __init__(self, data_dir, bands: Bands):
        self.data_dir = data_dir
        self.bands = bands
        self.data = None

    def create_data(self, bs_ind: int=1):
        """
        Load the data from the given directory and create the observations.
        Each sample from the dataset is a dictionary a key suitable for each band, then the data dimension is as follows:
        [number of sensors, number of sub-carriers, number of samples]
        the number of samples is the only dimension that is fixed.
        In addition, there is a key for the ground truth of the angle, time delay and power.

        Returns:
            data: pandas DataFrame

        """
        name_of_bands = [band.get_fc_file_name() for band in self.bands]
        # check if all the bands are available
        for name in name_of_bands:
            if not os.path.exists(os.path.join(self.data_dir, name)):
                raise FileNotFoundError(f"The band {name} is missing.")

        data = []
        # hold the csv files for the bs index for each band
        csv_files = [os.path.join(self.data_dir, name, f"bs_{str(bs_ind)}.csv") for name in name_of_bands]
        csv_loaded = [pd.read_csv(file) for file in csv_files]
        # start with extract all possible locations of the ue from the csv files named rx_x and rx_y
        csv_tmp = csv_loaded[0]
        ue_locations = csv_tmp[csv_tmp["link state"] == 1][['rx_x', 'rx_y']].to_numpy()

        # now we can use Channel and Observations to create the data by iterating over the ue locations
        channel = Channel(synthetic=False)
        for ue_pos in tqdm(ue_locations):
            link_flag = False
            observations = []
            gt = {'angle': [], 'time_delay': [], 'power': [], 'ue_pos': ue_pos, "bs_loc": None, "medium_speed": C}
            for idx, band in enumerate(self.bands):
                if link_flag:
                    break
                try:
                    channel.init_channel(bs_ind, ue_pos, band, csv_loaded=csv_loaded[idx])
                    gt['bs_loc'] = channel.get_bs_loc()
                # Except value error in case the link state is not 1. in that case, skip this ue position.
                except ValueError:
                    link_flag = True
                    continue
                # the angle and time_delay should be the same for all paths, for the LOS signal.
                gt['angle'].append(channel.get_LOS_doa())
                gt['time_delay'].append(channel.get_LOS_toa())
                gt['power'].append(channel.get_powers())
                gt["medium_speed"] = channel.get_medium_speed()
                obser = Observations([])
                obser.init_observations(channel, band)
                observations.append(obser.get_observations())
            # create the sample
            if len(observations) == 0:
                continue
            # Check if the angle and time delay are the same for all bands.
            if len(set(gt['angle'])) == 1 and len(set(gt['time_delay'])) == 1:
                gt['angle'] = gt['angle'][0]
                gt['time_delay'] = gt['time_delay'][0]
            else:
                # check if the variance is close enough to zero.
                var_time_delay = np.round(np.var(gt["time_delay"]), 9)
                var_angle = np.round(np.var(gt["angle"]), 3)
                if var_time_delay > 0 or var_angle > 0:
                    raise DataError("NYDataset.__load_data: The angle or time delay is not the same for all bands.")
                else:
                    gt['angle'] = gt['angle'][0]
                    gt['time_delay'] = gt['time_delay'][0]
            sample = {'data': observations, 'gt': gt}
            data.append(sample)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):


        sample = self.data[idx]
        data = sample['data']
        gt = sample['gt']
        gt['power'] = 0 # remove the power from the ground truth
        return data, gt

    def save(self, path):
        """
        Save the dataset to a given path.
        Args:
            path(str): The path to save the dataset.

        Returns:
            None
        """
        torch.save(self.data, path)

    def load(self, path):
        """
        Load the dataset from a given path.
        Args:
            path(str): The path to load the dataset.

        Returns:
            None
        """
        self.data = torch.load(path)

if __name__ == "__main__":
    # test the NYDataset
    data_dir = r"C:\git_repos\DCD-MUSIC\FR3\data\FR3"
    fcs = [6, 12, 18, 24]
    bws = [6, 12, 24, 48]
    ns = [4, 8, 16, 24]
    ks = [50, 100, 75, 100]
    bands = Bands(fcs, ns, ks, bws)
    dataset = NYDataset(data_dir, bands)
    dataset.create_data()
    dataset.save("test.npy")
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # for data, gt in dataloader:
    #     print(data, gt)
    #     break

