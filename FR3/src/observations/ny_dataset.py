"""
NY Dataset - This file contains the class for the NY Dataset.
The data contains observations from the NY scenario.
"""
import os
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from FR3.src.observations.band import Bands
from FR3.src.observations.observations import Observations
from FR3.src.observations.channel import Channel

class NYDataset(Dataset):
    def __init__(self, data_dir, fcs: list, bws: list, ns: list, ks: list):
        self.data_dir = data_dir
        self.bands = Bands(fcs, ns, ks, bws)
        self.data = self.__load_data()

    def __load_data(self, bs_ind: int=1):
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
        # start with extract all possible locations of the ue from the csv files named rx_x and rx_y
        tmp_data = pd.read_csv(os.path.join(self.data_dir, name_of_bands[0], f"bs_{str(bs_ind)}.csv"))
        ue_locations = tmp_data[tmp_data["link state"] == 1][['rx_x', 'rx_y']].to_numpy()

        # now we can use Channel and Observations to create the data by iterating over the ue locations
        channel = Channel(synthetic=False)
        for ue_pos in tqdm(ue_locations):
            link_flag = False
            observations = []
            gt = {'angle': [], 'time_delay': [], 'power': [], 'ue_pos': ue_pos}
            for band in self.bands:
                if link_flag:
                    break
                try:
                    channel.init_channel(bs_ind, ue_pos, band)
                except ValueError:
                    link_flag = True
                    continue
                gt['angle'].append(channel.get_doas())
                gt['time_delay'].append(channel.get_toas())
                gt['power'].append(channel.get_powers())
                obser = Observations([])
                obser.init_observations(channel, band)
                observations.append(obser.get_observations())
            # create the sample
            if len(observations) == 0:
                continue
            sample = {'data': observations, 'gt': gt}
            data.append(sample)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):


        sample = self.data[idx]
        data = sample['data']
        gt = sample['gt']

        return data, gt

if __name__ == "__main__":
    # test the NYDataset
    data_dir = r"C:\git_repos\DCD-MUSIC\FR3\data\FR3"
    fcs = [6, 12, 18, 24]
    bws = [6, 12, 24, 48]
    ns = [4, 8, 16, 24]
    ks = [50, 100, 75, 100]

    dataset = NYDataset(data_dir, fcs, bws, ns, ks)

