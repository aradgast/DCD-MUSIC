"""
TODO: add description and implementation
"""
import os
import sys
import scipy as sc
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.methods_pack.music import SubspaceMethod
from src.system_model import SystemModel
from FR3.src.observations.band import Bands
from FR3.utils.constants import *


class FR3_MUSIC(SubspaceMethod):
    def __init__(self, bands: Bands):
        super().__init__()
        self.angels_dict = None
        self.time_dict = None
        self.search_grid = None
        self.normalized_music_spectrum = None
        self.bands = bands
        self.__define_grid_params()
        self.__init_search_grid()

    # TODO: need to understand how to treat the sub-carriers bins in the OFDM signal.
    def forward(self, x):
        """

        Args:
            x:

        Returns:

        """
        peak, chosen_f, k_val = None, None, -np.inf
        self.normalized_music_spectrum = {}
        # calculate it for each sub-band
        for band in self.bands:
            band_idx = self.bands.get_band_index_by_fc(band.get_fc())
            sub_band_x = x[band_idx].to(DEVICE)
            # _, noise, _, _ = self.subspace_separation(self.pre_processing(sub_band_x, mode="subcarrier"), number_of_sources=1)
            noise = self.subspace_separation(self.pre_processing(sub_band_x.reshape(-1, sub_band_x.shape[-1])),
                                     number_of_sources=1)[1]
            # iterate over the batch
            for batch in range(noise.shape[0]):
                v1 = torch.matmul(self.search_grid[band.get_fc()], noise[batch].conj())
                inverse_spectrum = torch.norm(v1, dim=-1) # norm to get the spectrum
                music_spectrum = (1 / inverse_spectrum).reshape(len(self.angels_dict), -1)
                normalized_music_spectrum = torch.log10(10 * music_spectrum / music_spectrum.min().item()).cpu().numpy()
                k_peak_tmp = sc.signal.find_peaks(normalized_music_spectrum.flatten(), height=ALG_THRESHOLD * np.mean(normalized_music_spectrum))[0]
                if len(k_peak_tmp) == 0:
                    max_peak_k = [np.argmax(normalized_music_spectrum.flatten())]
                else:
                    max_peak_k = k_peak_tmp[np.argsort(normalized_music_spectrum.flatten()[k_peak_tmp])[::-1]][0]
                maximum_ind_k = np.array(np.unravel_index(max_peak_k, normalized_music_spectrum.shape))
                if normalized_music_spectrum[*maximum_ind_k] > k_val:
                    peak = maximum_ind_k
                    k_val = normalized_music_spectrum[*maximum_ind_k]
                    chosen_f = band.get_fc()
                    self.normalized_music_spectrum[band.get_fc()] = normalized_music_spectrum
        if chosen_f is None:
            peak = maximum_ind_k
            chosen_f = band.get_fc()
            self.normalized_music_spectrum[band.get_fc()] = normalized_music_spectrum


        # need to return the DOAs and TOAs, and the related power
        angle = self.angels_dict[peak[0]]
        time_delay = self.time_dict[chosen_f][peak[1]]
        power = self.normalized_music_spectrum[chosen_f][peak[0], peak[1]]
        return angle, time_delay, power


    def __init_search_grid(self):
        """
        After the dicts are defined, we need to create the grid,
        assuming different number of elements in the sensors array for each band.

        Returns:

        """

        angle_options = [band.compute_angle_steering(self.angels_dict) for band in self.bands]
        time_options = [band.compute_time_steering(self.time_dict[band.get_fc()]) for band in self.bands]
        self.search_grid = {fc:None for fc in self.bands.get_all_fcs()}

        for band in self.bands:
            band_idx = self.bands.get_band_index_by_fc(band.get_fc())
            multiplication = np.kron(angle_options[band_idx].T, time_options[band_idx])
            self.search_grid[band.get_fc()] = torch.from_numpy(multiplication).to(DEVICE)
            # angle_option = torch.from_numpy(angle_options[band_idx]).to(DEVICE)
            # time_option = torch.from_numpy(time_options[band_idx]).to(DEVICE)
            # self.search_grid[band.get_fc()] = torch.einsum("na, tk -> natk", angle_option, time_option) # TODO


    def __define_grid_params(self):
        """
        Simple init for the possible values for angle and time.
        Returns:

        """
        self.angels_dict = np.arange(-np.pi / 2, np.pi / 2, np.deg2rad(DOA_RES))
        self.time_dict = {band.get_fc():np.arange(0, band.get_k() / band.get_bw(), TIME_RES) for band in self.bands}
