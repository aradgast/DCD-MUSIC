"""
This module contains the classes for the Band and Bands objects. The Band object represents a single band of the system
and contains the carrier frequency, number of sensors, number of sub-carriers, and bandwidth.
The Bands object is a collection of Band objects.

"""
import numpy as np
import torch

from FR3.utils.constants import DEVICE

def get_steering_vector(params):
    return torch.exp(-2j * np.pi * params)

class Band:
    def __init__(self, fc, n, k, bw):
        self.fc = fc * 10 ** 9
        self.n = n
        self.k = k
        self.bw = bw * 10 ** 6

    def __str__(self):
        return (f'Carrier Freq.: {self.fc / 10 ** 9} GHz, Number of Sensors: {self.n},'
                f' Number of sub-carriers: {self.k}, Bandwidth: {self.bw / 10 ** 6} MHz')

    def get_fc_file_name(self):
        return f'{self.fc // 10 ** 9}GHz'

    def get_fc(self):
        return self.fc

    def get_n(self):
        return self.n

    def get_k(self):
        return self.k

    def get_bw(self):
        return self.bw

    def compute_time_steering(self, toas):
        if isinstance(toas, list):
            toas = torch.from_numpy(np.array(toas)).to(DEVICE)
        if isinstance(toas, np.ndarray):
            toas = torch.from_numpy(toas).to(DEVICE)
        toas = torch.atleast_2d(toas).reshape(-1, 1)
        # create the K frequency bins
        time_basis_vector = torch.linspace(self.fc - self.bw / 2, self.fc + self.bw / 2, self.k)
        time_basis_vector = time_basis_vector.reshape(1, -1).to(DEVICE).to(torch.float64)
        # simulate the phase at each frequency bins
        combination = torch.matmul(toas, time_basis_vector)
        array_response_combination = get_steering_vector(combination)
        # might have duplicates depending on the frequency, BW and number of subcarriers
        # so remove the recurring time basis vectors - assume only the first one is valid
        # and the ones after can only cause recovery errors
        # TODO: check if this is the correct way to handle the duplicates
        # if array_response_combination.shape[0] > 1:
        #     first_row_duplicates = torch.all(torch.isclose(array_response_combination, array_response_combination[0]),
        #                                      dim=1)
        #     if torch.sum(first_row_duplicates).item() > 1:
        #         dup_row = torch.nonzero(first_row_duplicates)[1].item()
        #         array_response_combination = array_response_combination[:dup_row]
        return array_response_combination

    def compute_angle_steering(self, angles):
        if isinstance(angles, list):
            angles = torch.from_numpy(np.array(angles)).to(DEVICE)
        if isinstance(angles, np.ndarray):
            angles = torch.from_numpy(angles).to(DEVICE)
        # create the steering based on the doas
        angles = torch.atleast_2d(angles)
        loc_shifts = torch.arange(self.n, device=DEVICE, dtype=torch.float64).reshape(-1, 1)
        return get_steering_vector(torch.matmul(loc_shifts, torch.sin(angles).reshape(1, -1)) / 2)


class Bands:
    def __init__(self, fcs, ns, ks, bws):
        self.bands = []
        for fc, n, k, bw in zip(fcs, ns, ks, bws):
            self.bands.append(Band(fc, n, k, bw))

    def __iter__(self):
        return iter(self.bands)

    def get_bands(self):
        return self.bands

    def get_band(self, index) -> Band:
        return self.bands[index]

    def get_band_count(self):
        return len(self.bands)

    def clear_bands(self):
        self.bands = []

    def print_bands(self):
        for band in self.bands:
            print(band)

    def get_band_by_fc(self, fc):
        for band in self.bands:
            if band.fc == fc:
                return band
        return None

    def get_all_fcs(self):
        return [band.fc for band in self.bands]

    def get_bands_filenames(self):
        return [band.get_fc_file_name() for band in self.bands]

    def get_band_index_by_fc(self, fc):
        for i in range(len(self.bands)):
            if self.bands[i].fc == fc:
                return i
        return -1

