"""
TODO: add description and implementation
"""
import os
import sys
import scipy as sc
import torch

from train_dcd import batch_size

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

    def forward(self, x):
        """

        Args:
            x:

        Returns:

        """
        batch_size = x[0].shape[0]
        peaks = torch.zeros(batch_size, 2, dtype=torch.int64, device=DEVICE)
        k_vals = torch.zeros(batch_size, dtype=torch.float32, device=DEVICE)
        # init k_vals with -inf
        k_vals.fill_(-np.inf)
        chosen_fs = torch.zeros(batch_size, dtype=torch.int64, device=DEVICE)
        self.normalized_music_spectrum = {}
        # calculate it for each sub-band
        for band in self.bands:
            band_idx = self.bands.get_band_index_by_fc(band.get_fc())
            sub_band_x = x[band_idx].to(DEVICE)
            batch_size = sub_band_x.shape[0]
            # _, noise, _, _ = self.subspace_separation(self.pre_processing(sub_band_x, mode="subcarrier"), number_of_sources=1)
            noise = self.subspace_separation(self.pre_processing(sub_band_x.reshape(batch_size, -1, sub_band_x.shape[-1])),
                                     number_of_sources=1)[1]
            v1 = torch.einsum("an, bnm -> bam", self.search_grid[band.get_fc()], noise.conj())
            inverse_spectrum = torch.norm(v1, dim=-1)  # norm to get the spectrum
            music_spectrum = (1 / inverse_spectrum).view(batch_size, len(self.angels_dict), -1)
            min_per_row = torch.min(music_spectrum, dim=-1)[0]
            min_per_matrix = torch.min(min_per_row, dim=-1, keepdim=True)[0][:, :, None]
            normalized_music_spectrum = torch.log10(10 * music_spectrum / min_per_matrix).cpu().numpy()
            # iterate over the batch
            for batch in range(batch_size):
                # v1 = torch.matmul(self.search_grid[band.get_fc()], noise[batch].conj())
                # inverse_spectrum = torch.norm(v1, dim=-1) # norm to get the spectrum
                # music_spectrum = (1 / inverse_spectrum).reshape(len(self.angels_dict), -1)
                # normalized_music_spectrum = torch.log10(10 * music_spectrum / music_spectrum.min().item()).cpu().numpy()
                k_peak_tmp = sc.signal.find_peaks(normalized_music_spectrum[batch].flatten(), height=ALG_THRESHOLD * np.mean(normalized_music_spectrum[batch]))[0]
                if len(k_peak_tmp) == 0:
                    max_peak_k = [np.argmax(normalized_music_spectrum[batch].flatten())]
                else:
                    max_peak_k = k_peak_tmp[np.argsort(normalized_music_spectrum[batch].flatten()[k_peak_tmp])[::-1]][0]
                maximum_ind_k = torch.from_numpy(np.array(np.unravel_index(max_peak_k, normalized_music_spectrum[batch].shape)))
                if normalized_music_spectrum[batch][*maximum_ind_k] > k_vals[batch]:
                    peaks[batch][None, :] = maximum_ind_k.squeeze()
                    k_vals[batch] = normalized_music_spectrum[batch][*maximum_ind_k]
                    chosen_fs[batch] = band.get_fc()
                    # self.normalized_music_spectrum[band.get_fc()] = normalized_music_spectrum

            # instead of the batch loop, we can use the following code:
            # v1 = torch.einsum("an, bnm -> bam", self.search_grid[band.get_fc()], noise.conj())
            # inverse_spectrum = torch.norm(v1, dim=-1) # norm to get the spectrum
            # music_spectrum = (1 / inverse_spectrum).reshape(batch_size, len(self.angels_dict), -1)
            # min_per_row = torch.min(music_spectrum, dim=-1)[0]
            # min_per_matrix = torch.min(min_per_row, dim=-1)[0]
            # normalized_music_spectrum = torch.log10(10 * music_spectrum / min_per_matrix).cpu().numpy()
            # peaks = torch.zeros(batch_size, 2, dtype=torch.int64, device=device)
            # for batch in range(batch_size):
            #     k_peak_tmp = sc.signal.find_peaks(normalized_music_spectrum[batch].flatten(), height=ALG_THRESHOLD * np.mean(normalized_music_spectrum[batch]))[0]
            #     if len(k_peak_tmp) == 0:
            #         max_peak_k = [np.argmax(normalized_music_spectrum[batch].flatten())]
            #     else:
            #         max_peak_k = k_peak_tmp[np.argsort(normalized_music_spectrum[batch].flatten()[k_peak_tmp])[::-1]][0]
            #     maximum_ind_k = np.array(np.unravel_index(max_peak_k, normalized_music_spectrum[batch].shape))
            #     peaks[batch] = maximum_ind_k




        # # check if there is an element in the chosen_fs that is zero
        # for i in range(batch_size):
        #     if chosen_fs[i] == 0:
        #         # if there is no peak, set the peak for the last band
        #         peak = maximum_ind_k
        #     chosen_f = band.get_fc()
        #     self.normalized_music_spectrum[band.get_fc()] = normalized_music_spectrum


        # need to return the DOAs and TOAs, and the related power
        angles = self.angels_dict[peaks.cpu().numpy()[:, 0]]
        time_delays = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            time_delays[i] = self.time_dict[chosen_fs[i].item()][peaks.cpu().numpy()[i, 1]]
        # power = self.normalized_music_spectrum[chosen_f][peak[0], peak[1]]
        return angles, time_delays, None


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
            self.search_grid[band.get_fc()] = torch.kron(angle_options[band_idx].transpose(0, 1).contiguous(), time_options[band_idx].contiguous())
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
