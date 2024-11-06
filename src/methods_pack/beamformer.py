"""
This file contain an implementation for a Beamformer to use for Far field(DOA) and Near field(DOA and Range) scenarios.
"""
import torch
from torch.nn import Module
from scipy.signal import find_peaks

from src.criterions import CartesianLoss
from src.system_model import SystemModel
from src.utils import *
from src.criterions import CartesianLoss, RMSPELoss

class Beamformer(Module):
    def __init__(self, system_model: SystemModel):
        super(Beamformer, self).__init__()
        self.system_model = system_model
        self.angles_dict = None
        self.ranges_dict = None
        self.steering_dict = None
        self.__init_grid_params()
        self.__init_steering_dict()
        self.__init_criteria()
        self.eval() # This a torch based model without any trainable parameters.

    def forward(self, x):
        samp_cov = self.pre_processing(x)
        spectrum = self.get_spectrum(samp_cov)
        peaks = self.find_peaks(spectrum)
        labels = self.get_labels(peaks)
        return labels

    def get_spectrum(self, cov):
        if self.system_model.params.field_type == "Far":
            # in this case, the steering search space is 2D -> NxA, whereas A is the size of the search grid.
            v1 = torch.einsum("an, bnm -> bam", self.steering_dict.conj().transpose(0, 1), cov)
            spectrum = torch.einsum("ban, na -> ba", v1, self.steering_dict)
        else:
            # in this case, the steering search space is 3D -> NxAxR, whereas R is the size of the ranges search dictionary.
            v1 = torch.einsum("arn, bnm -> barm", self.steering_dict.conj().transpose(0, 2).transpose(0, 1), cov)
            spectrum = torch.einsum("barn, nar -> bar", v1, self.steering_dict)
        return spectrum

    def find_peaks(self, spectrum):
        if self.system_model.params.field_type == "Far":
            return self.__find_peaks_far_field(spectrum)
        else:
            return self.__find_peaks_near_field(spectrum)

    def get_labels(self, peaks):
        if self.ranges_dict is None:
            # in this case, peaks is of size BxM, whereas M is the number of sources.
            labels = self.angles_dict[peaks]
        else:
            # in this case, peaks is of size Bx2M.
            peaks_angle_axis, peaks_range_axis = torch.split(peaks, peaks.shape[1] //2, dim=1)
            angles = self.angles_dict[peaks_angle_axis]
            ranges = self.ranges_dict[peaks_range_axis]
            labels = angles, ranges
        return labels

    @staticmethod
    def pre_processing(x):
        """
        The pre-processing stage of the beamformer is the calculation of the sample covariance.

        Args:
            x: The input signal with dim BxNxT, for B the batch size, N the number of sensors,
                and T the number of snapshots.

        Returns:
            tensor: a Tensor of size BxNxN represent the sample covariance for each element in the batch.
        """
        return sample_covariance(x)

    def test_step(self, batch, batch_idx):
        x, sources_num, label, masks = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        test_length = x.shape[0]
        x = x.to(device)
        if max(sources_num) * 2 == label.shape[1]:
            angles, ranges = torch.split(label, max(sources_num), dim=1)
            angles = angles.to(device)
            ranges = ranges.to(device)
            masks, _ = torch.split(masks, max(sources_num), dim=1)  # TODO
        else:
            angles = label.to(device)  # only angles
        # Check if the sources number is the same for all samples in the batch
        if (sources_num != sources_num[0]).any():
            # in this case, the sources number is not the same for all samples in the batch
            raise Exception(f"train_model:"
                            f" The sources number is not the same for all samples in the batch.")
        else:
            sources_num = sources_num[0]

        predictions = self(x)
        if isinstance(predictions, tuple):
            angles_prediction, ranges_prediction = predictions
            rmspe = self.criterion(angles_prediction, angles, ranges_prediction, ranges).item()
            _, rmspe_angle, rmspe_range = self.separated_criterion(angles_prediction, angles, ranges_prediction, ranges)
            rmspe = (rmspe, rmspe_angle.item(), rmspe_range.item())
        else:
            rmspe = self.criterion(predictions, angles).item()

        return rmspe, 0, test_length

    def __find_peaks_far_field(self, spectrum):
        source_number = self.system_model.params.M
        batch_size = spectrum.shape[0]

        peaks = torch.zeros(batch_size, source_number, dtype=torch.int64, device=device)
        for batch in range(batch_size):
            elem_spectrum = spectrum[batch].cpu().detach().numpy().squeeze()
            # Find spectrum peaks
            peaks_tmp = find_peaks(elem_spectrum, threshold=0.0)[0]
            if len(peaks_tmp) < source_number:
                warnings.warn(f"Beamformer.__find_peaks_far_field: No peaks were found! taking max values instead.")
                # random_peaks = np.random.randint(0, search_space.shape[0], (source_number - peaks_tmp.shape[0],))
                random_peaks = torch.topk(self.angles_dict, source_number - peaks_tmp.shape[0],
                                          largest=True).indices.cpu().detach().numpy()
                peaks_tmp = np.concatenate((peaks_tmp, random_peaks))
            # Sort the peak by their amplitude
            sorted_peaks = peaks_tmp[np.argsort(elem_spectrum[peaks_tmp])[::-1]]
            peaks[batch] = torch.from_numpy(sorted_peaks[0:source_number]).to(device)
        # if the model is not in training mode, return the peaks
        return peaks

    def __find_peaks_near_field(self, spectrum):
        source_number = self.system_model.params.M
        batch_size = spectrum.shape[0]

        max_row = torch.zeros((batch_size, source_number)
                              , dtype=torch.int64, device=device)
        max_col = torch.zeros((batch_size, source_number)
                              , dtype=torch.int64, device=device)
        for batch in range(batch_size):
            elem_spectrum = spectrum[batch].detach().cpu().numpy().squeeze()
            # Flatten the spectrum
            spectrum_flatten = elem_spectrum.flatten()
            # Find spectrum peaks
            peaks = find_peaks(spectrum_flatten)[0]
            # Sort the peak by their amplitude
            sorted_peaks = peaks[np.argsort(spectrum_flatten[peaks])[::-1]]
            # convert the peaks to 2d indices
            original_idx = torch.from_numpy(np.column_stack(np.unravel_index(sorted_peaks, elem_spectrum.shape))).T
            if source_number > 1:
                # pass
                original_idx = keep_far_enough_points(original_idx, source_number, 10)
            max_row[batch] = original_idx[0][0: source_number]
            max_col[batch] = original_idx[1][0: source_number]
        # if the model is not in training mode, return the peaks.
        peaks = torch.cat((max_row, max_col), dim=1)
        return peaks

    def __init_grid_params(self):
        """
        Set values for the angle and range dict, depends on the scenario.
        Returns:
            None.

        """
        angle_range = np.deg2rad(self.system_model.params.doa_range)
        angle_resolution = np.deg2rad(self.system_model.params.doa_resolution / 2)
        # if it's the Far field case, need to init angles range.
        self.angles_dict = torch.arange(-angle_range, angle_range, angle_resolution, device=device,
                                   dtype=torch.float64).requires_grad_(False).to(torch.float64)
        if self.system_model.params.field_type.startswith("Near"):
            # if it's the Near field, there are 3 possabilities.
            fresnel = self.system_model.fresnel
            fraunhofer = self.system_model.fraunhofer
            fraunhofer_ratio = self.system_model.params.max_range_ratio_to_limit
            distance_resolution = self.system_model.params.range_resolution / 2
            self.ranges_dict = torch.arange(np.floor(fresnel),
                                          fraunhofer * fraunhofer_ratio,
                                          distance_resolution,
                                          device=device, dtype=torch.float64).requires_grad_(False)
        else:
            raise ValueError(f"Beamformer.__init_grid_params: Unrecognized field type for Beamformer class init stage,"
                             f" got {self.system_model.params.field_type} but only Far and Near are allowed.")

    def __init_steering_dict(self):
        """
        By using the dictionaries for angles and ranges, init the steering matrix to use for the beamformer.

        Returns:
            None.
        """
        if self.ranges_dict is None:
            self.steering_dict = self.system_model.steering_vec(self.angles_dict, nominal=True)
        else:
            self.steering_dict = self.system_model.steering_vec(self.angles_dict, self.ranges_dict,
                                                           nominal=True, generate_search_grid=True)
    def __init_criteria(self):
        self.criterion = CartesianLoss()
        self.separated_criterion = RMSPELoss(0)
