import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy as sc

from src.system_model import SystemModel
from src.methods_pack.subspace_method import SubspaceMethod
from src.utils import *


class MUSIC(SubspaceMethod):
    """
    This is implementation of the MUSIC method for localization in Far and Near field environments.
    For Far field - only "angle" can be estimated
    For Near field - "angle", "range" and "angle, range" are the possible options.
    """

    def __init__(self, system_model: SystemModel, estimation_parameter: str):
        """

        Args:
            system_model:
            estimation_parameter:
        """
        super().__init__(system_model)
        self.estimation_params = estimation_parameter
        self.angels = None
        self.distances = None
        self.search_grid = None
        self.music_spectrum = None
        self.__define_grid_params()
        if self.estimation_params == "range":
            self.cell_size = int(self.distances.shape[0] * 0.2)
        elif self.estimation_params == "angle, range":
            self.cell_size_angle = int(self.angels.shape[0] * 0.2)
            self.cell_size_distance = int(self.distances.shape[0] * 0.2)

        self.search_grid = None
        # if this is the music 2D case, the search grid is constant and can be calculated once.
        if self.system_model.params.field_type.startswith("Near"):
            if self.angels is not None and self.distances is not None:
                self.set_search_grid()
        else:
            self.set_search_grid()
        self.noise_subspace = None
        # self.filter = Filter(int(self.distances.shape[0] * 0.05), int(self.distances.shape[0] * 0.2), number_of_filter=5)

    def forward(self, cov: torch.Tensor, number_of_sources:int, known_angles=None, known_distances=None, is_soft: bool = True):
        """

        Parameters
        ----------
        cov - the covariance tensor to preform the MUSIC one. size: BatchSizeX#SensorsX#Sensors
        known_angles - in case of "range" estimation for the Near-field, use the known angles to create the search grid.
        known_distances - same as "known_angles", but for "angle" estimation.
        is_soft - decide if using hard-decision(peak finder) or soft-decision(the approximated peak finder which is differentiable)

        Returns
        -------
        the returned value depends on the self.estimation_params:
            (self.estimation_params == "angle") - torch.Tensor with the predicted angles
            (self.estimation_params == "range") - torch.Tensor with the predicted ranges
            (self.estimation_params == "angle, range") - tuple, each one of the elements is torch.Tensor for the predicted param.
        """
        if self.system_model.params.M is not None:
            M = self.system_model.params.M
        else:
            M = number_of_sources
        # single param estimation: the search grid should be updated for each batch, else, it's the same search grid.
        if self.system_model.params.field_type.startswith("Near"):
            if self.estimation_params in ["angle", "range"]:
                if known_angles.shape[-1] == 1:
                    self.set_search_grid(known_angles=known_angles, known_distances=known_distances)
                else:
                    params = torch.zeros((cov.shape[0], M), dtype=torch.float64, device=device)
                    for source in range(M):
                        params_source, _, _ = self.forward(cov, number_of_sources=M,known_angles=known_angles[:, source][:, None],
                                                     is_soft=is_soft)
                        params[:, source] = params_source.squeeze().requires_grad_(True)
                    return params
        _, Un, source_estimation, eigen_regularization = self.subspace_separation(cov.to(torch.complex128), M)
        # self.noise_subspace = Un.cpu().detach().numpy()
        inverse_spectrum = self.get_inverse_spectrum(Un.to(device)).to(device)
        self.music_spectrum = 1 / inverse_spectrum
        #####
        # self.music_spectrum = torch.div(self.music_spectrum, torch.max(torch.max(self.music_spectrum, dim=2).values, dim=1).values.unsqueeze(1).unsqueeze(2))
        # self.music_spectrum = torch.pow(self.music_spectrum, 5)
        #####
        params = self.peak_finder(is_soft, M)

        return params, source_estimation, eigen_regularization

    def get_music_spectrum_from_noise_subspace(self, Un):
        inverse_spectrum = self.get_inverse_spectrum(Un.to(torch.complex128))
        self.music_spectrum = 1 / inverse_spectrum
        return self.music_spectrum

    def adjust_cell_size(self):
        if self.estimation_params == "range":
            # if self.cell_size > int(self.distances.shape[0] * 0.02):
            if self.cell_size > 1 or self.cell_size > int(self.distances.shape[0] * 0.02):
                # self.cell_size -= int(np.ceil(self.distances.shape[0] * 0.01))
                self.cell_size = int(0.95 * self.cell_size)
                if self.cell_size % 2 == 0:
                    self.cell_size -= 1
        elif self.estimation_params == "angle, range":
            if self.cell_size_angle > 3:
                self.cell_size_angle = int(0.95 * self.cell_size_angle)
                if self.cell_size_angle % 2 == 0:
                    self.cell_size_angle -= 1
            if self.cell_size_distance > 3:
                self.cell_size_distance = int(0.95 * self.cell_size_distance)
                if self.cell_size_distance % 2 == 0:
                    self.cell_size_distance -= 1

    def get_inverse_spectrum(self, noise_subspace: torch.Tensor):
        """

        Parameters
        ----------
        noise_subspace - the noise related subspace vectors of size BatchSizex#SENSORSx(#SENSORS-#SOURCES)

        Returns
        -------
        in all cases it will return the inverse spectrum,
        in case of single param estimation it will be 1D inverse spectrum: BatchSizex(length_search_grid)
        in case of dual param estimation it will be 2D inverse spectrum:
                                                    BatchSizex(length_search_grid_angle)x(length_search_grid_distance)
        """
        if self.system_model.params.field_type.startswith("Far"):
            var1 = torch.einsum("an, bnm -> bam", self.search_grid.conj().transpose(0, 1)[:, :noise_subspace.shape[1]],
                                noise_subspace)
            inverse_spectrum = torch.norm(var1, dim=2)
        else:
            if self.estimation_params.startswith("angle, range"):
                var1 = torch.einsum("adk, bkl -> badl",
                                    torch.transpose(self.search_grid.conj(), 0, 2).transpose(0, 1)[:, :,
                                    :noise_subspace.shape[1]],
                                    noise_subspace)
                # get the norm value for each element in the batch.
                inverse_spectrum = torch.norm(var1, dim=-1) ** 2
            elif self.estimation_params.endswith("angle"):
                var1 = torch.einsum("ban, nam -> abm", self.search_grid.conj().transpose(0, 2),
                                    noise_subspace.transpose(0, 1))
                inverse_spectrum = torch.norm(var1, dim=2).T
            elif self.estimation_params.startswith("range"):
                var1 = torch.einsum("dbn, nbm -> bdm", self.search_grid.conj().transpose(0, 2),
                                    noise_subspace.transpose(0, 1))
                inverse_spectrum = torch.norm(var1, dim=2)
            else:
                raise ValueError("unknown estimation param")
        return inverse_spectrum

    def peak_finder(self, is_soft: bool, source_number: int = None):
        """

        Parameters
        ----------
        is_soft: this boolean paramter will determine wether to use derivative approxamtion of the peak_finder for
         the training stage.

        Returns
        -------
        the predicted param(torch.Tensor) or params(tuple)
        """
        if is_soft:
            return self._maskpeak(source_number)
        else:
            return self._peak_finder(source_number)

    def set_search_grid(self, known_angles: torch.Tensor = None, known_distances: torch.Tensor = None):
        if self.system_model.params.field_type.startswith("Far"):
            self.__set_search_grid_far_field()
        elif self.system_model.params.field_type.startswith("Near"):
            self.__set_search_grid_near_field(known_angles=known_angles, known_distances=known_distances)
        else:
            raise ValueError(f"set_search_grid: Unrecognized field type: {self.system_model.params.field_type}")

    def __set_search_grid_far_field(self):
        array = torch.Tensor(self.system_model.array[:, None]).to(torch.float64).to(device)
        theta = self.angels[:, None]
        time_delay = torch.einsum("nm, na -> na",
                                  array,
                                  torch.sin(theta).repeat(1, self.system_model.params.N).T
                                  * self.system_model.dist_array_elems["NarrowBand"])
        self.search_grid = torch.exp(-2 * 1j * torch.pi * time_delay)

    def __set_search_grid_near_field(self, known_angles: torch.Tensor = None, known_distances: torch.Tensor = None):
        """

        Returns:

        """
        dist_array_elems = self.system_model.dist_array_elems["NarrowBand"]
        if known_angles is None:
            theta = self.angels[:, None]
        else:
            theta = known_angles.float()
            if len(theta.shape) == 1:
                theta = torch.atleast_1d(theta)[:, None].to(torch.float64)

        if known_distances is None:
            distances = self.distances[:, None].to(torch.float64)
        else:
            distances = known_distances.float()
            if len(distances.shape) == 1:
                distances = torch.atleast_1d(distances)[:, None]
        array = torch.Tensor(self.system_model.array[:, None]).to(torch.float64).to(device)
        array_square = torch.pow(array, 2).to(torch.float64)

        first_order = torch.einsum("nm, na -> na",
                                   array,
                                   torch.sin(theta).repeat(1, self.system_model.params.N).T * dist_array_elems)

        second_order = -0.5 * torch.div(torch.pow(torch.cos(theta) * dist_array_elems, 2), distances.T)
        second_order = second_order[:, :, None].repeat(1, 1, self.system_model.params.N)
        second_order = torch.einsum("nm, nda -> nda",
                                    array_square,
                                    torch.transpose(second_order, 2, 0)).transpose(1, 2)

        first_order = first_order[:, :, None].repeat(1, 1, second_order.shape[-1])

        time_delay = first_order + second_order

        self.search_grid = torch.exp(2 * -1j * torch.pi * time_delay)

    def plot_spectrum(self, highlight_corrdinates=None, batch: int = 0, method: str = "heatmap"):
        if self.estimation_params == "angle, range":
            self._plot_3d_spectrum(highlight_corrdinates, batch, method)
        else:
            self._plot_1d_spectrum(highlight_corrdinates, batch)

    def _peak_finder(self, source_number: int = None):
        if self.system_model.params.field_type.startswith("Far"):
            return self._peak_finder_1D(self.angels, source_number)
        else:
            if self.estimation_params.startswith("angle, range"):
                return self._peak_finder_2D(source_number)
            elif self.estimation_params.endswith("angle"):
                return self._peak_finder_1D(self.angels)
            elif self.estimation_params.startswith("range"):
                return self._peak_finder_1D(self.distances)

    def _maskpeak(self, source_number: int = None):
        if self.system_model.params.field_type.startswith("Far"):
            return self._maskpeak_1D(self.angels, source_number)
        else:
            if self.estimation_params.startswith("angle, range"):
                return self._maskpeak_2D(source_number)
            elif self.estimation_params.endswith("angle"):
                return self._maskpeak_1D(self.angels, source_number)
            elif self.estimation_params.startswith("range"):
                return self._maskpeak_1D(self.distances, source_number)

    def _maskpeak_1D(self, search_space, source_number: int = None):
        flag = True
        if flag:

            # top_indxs = torch.topk(self.music_spectrum, 1, dim=1)[1]
            peaks = torch.zeros(self.music_spectrum.shape[0], 1).to(torch.int64)
            for batch in range(peaks.shape[0]):
                music_spectrum = self.music_spectrum[batch].cpu().detach().numpy().squeeze()
                # Find spectrum peaks
                peaks_tmp = list(sc.signal.find_peaks(music_spectrum)[0])
                # Sort the peak by their amplitude
                peaks_tmp.sort(key=lambda x: music_spectrum[x], reverse=True)
                if len(peaks_tmp) == 0:
                    peaks_tmp = torch.randint(self.distances.shape[0], (1,))
                else:
                    peaks_tmp = peaks_tmp[0]
                peaks[batch] = peaks_tmp
            top_indxs = peaks.to(device)
            cell_idx = (top_indxs - self.cell_size + torch.arange(2 * self.cell_size + 1, dtype=torch.long,
                                                                  device=device))
            cell_idx %= self.music_spectrum.shape[1]
            # if False:
            #     for batch in range(self.music_spectrum.shape[0]):
            #         cell_idx[batch][cell_idx[batch] < 0] = top_indxs[batch]
            cell_idx = cell_idx.unsqueeze(-1)
            metrix_thr = torch.gather(self.music_spectrum.unsqueeze(-1).expand(-1, -1, cell_idx.size(-1)), 1, cell_idx).requires_grad_(True)
            soft_max = torch.softmax(metrix_thr, dim=1)
            soft_decision = torch.einsum("bkm, bkm -> bm", search_space[cell_idx], soft_max).to(device)
        else:
            soft_decision = self.filter(self.music_spectrum, search_space)

        return soft_decision

    def _maskpeak_2D(self, source_number: int = None):
        if source_number is None:
            source_number = self.system_model.params.M
        batch_size = self.music_spectrum.shape[0]
        soft_row = torch.zeros((batch_size, source_number), device=device)
        soft_col = torch.zeros((batch_size, source_number), device=device)
        max_row = torch.zeros((self.music_spectrum.shape[0], source_number)
                              , dtype=torch.int64, device=device)
        max_col = torch.zeros((self.music_spectrum.shape[0], source_number)
                              , dtype=torch.int64, device=device)
        for batch in range(self.music_spectrum.shape[0]):
            music_spectrum = self.music_spectrum[batch].detach().cpu().numpy().squeeze()
            # Flatten the spectrum
            spectrum_flatten = music_spectrum.flatten()
            # Find spectrum peaks
            peaks = sc.signal.find_peaks(spectrum_flatten)[0]
            # Sort the peak by their amplitude
            sorted_peaks = peaks[np.argsort(spectrum_flatten[peaks])[::-1]]
            # convert the peaks to 2d indices
            original_idx = torch.from_numpy(np.column_stack(np.unravel_index(sorted_peaks, music_spectrum.shape))).T
            if source_number > 1:
                original_idx = keep_far_enough_points(original_idx, source_number, 30)
            max_row[batch] = original_idx[0][0: source_number]
            max_col[batch] = original_idx[1][0: source_number]
        for source in range(source_number):
            max_row_cell_idx = (max_row[:, source][:, None]
                                - self.cell_size_angle
                                + torch.arange(2 * self.cell_size_angle + 1, dtype=torch.int32, device=device))
            max_row_cell_idx %= self.music_spectrum.shape[1]
            max_row_cell_idx = max_row_cell_idx.reshape(batch_size, -1, 1)

            max_col_cell_idx = (max_col[:, source][:, None]
                                - self.cell_size_distance
                                + torch.arange(2 * self.cell_size_distance + 1, dtype=torch.int32, device=device))
            max_col_cell_idx %= self.music_spectrum.shape[2]
            max_col_cell_idx = max_col_cell_idx.reshape(batch_size, 1, -1)

            metrix_thr = self.music_spectrum.gather(1, max_row_cell_idx.expand(-1, -1, self.music_spectrum.shape[2]))
            metrix_thr = metrix_thr.gather(2, max_col_cell_idx.repeat(1, max_row_cell_idx.shape[-2], 1))
            soft_max = torch.softmax(metrix_thr.view(batch_size, -1), dim=1).reshape(metrix_thr.shape)
            soft_row[:, source][:, None] = torch.einsum("bla, bad -> bl",
                                                        self.angels[max_row_cell_idx].transpose(1, 2),
                                                        torch.sum(soft_max, dim=2).unsqueeze(-1))
            soft_col[:, source][:, None] = torch.einsum("bmc, bcm -> bm",
                                                        self.distances[max_col_cell_idx],
                                                        torch.sum(soft_max, dim=1).unsqueeze(-1))

        return soft_row, soft_col

    def _peak_finder_1D(self, search_space, source_num: int = None):
        if source_num is None:
            source_num = self.system_model.params.M
        predict_param = torch.zeros(self.music_spectrum.shape[0], 1, device=device)
        for batch in range(self.music_spectrum.shape[0]):
            music_spectrum = self.music_spectrum[batch].cpu().detach().numpy().squeeze()
            # Find spectrum peaks
            peaks = list(sc.signal.find_peaks(music_spectrum)[0])
            # Sort the peak by their amplitude
            peaks.sort(key=lambda x: music_spectrum[x], reverse=True)
            tmp = search_space[peaks[0:1]]
            if tmp.nelement() == 0:
                rand_idx = torch.randint(self.distances.shape[0], (1,))
                tmp = self.distances[rand_idx]
                # tmp = self._maskpeak_1D(search_space)
                # print("_peak_finder_1D: No peaks were found!")
            else:
                pass
            predict_param[batch] = tmp

        return torch.Tensor(predict_param)

    def _peak_finder_2D(self, source_num: int = None):
        if source_num is None:
            source_num = self.system_model.params.M
        predict_theta = np.zeros((self.music_spectrum.shape[0], source_num)
                                 , dtype=np.float64)
        predict_dist = np.zeros((self.music_spectrum.shape[0], source_num)
                                , dtype=np.float64)
        angels = self.angels.cpu().detach().numpy()
        distances = self.distances.cpu().detach().numpy()
        for batch in range(self.music_spectrum.shape[0]):
            music_spectrum = self.music_spectrum[batch].detach().cpu().numpy().squeeze()
            # Flatten the spectrum
            spectrum_flatten = music_spectrum.flatten()
            # Find spectrum peaks
            peaks = list(sc.signal.find_peaks(spectrum_flatten)[0])
            # Sort the peak by their amplitude
            peaks.sort(key=lambda x: spectrum_flatten[x], reverse=True)
            # convert the peaks to 2d indices
            original_idx = np.array(np.unravel_index(peaks, music_spectrum.shape))
            if self.system_model.params.M > 1:
                original_idx = keep_far_enough_points(original_idx, source_num, 30)
            original_idx = original_idx[:, 0:source_num]
            predict_theta[batch] = angels[original_idx[0]]
            predict_dist[batch] = distances[original_idx[1]]

        return torch.from_numpy(predict_theta).to(device), torch.from_numpy(predict_dist).to(device)

    def _init_spectrum(self, batch_size):
        if self.system_model.params.field_type == "Far":
            self.music_spectrum = torch.zeros(batch_size, len(self.angels))
        else:
            if self.estimation_params.startswith("angle, range"):
                self.music_spectrum = torch.zeros(batch_size, len(self.angels), len(self.distances))
            elif self.estimation_params.endswith("angle"):
                self.music_spectrum = torch.zeros(batch_size, len(self.angels))
            elif self.estimation_params.startswith("range"):
                self.music_spectrum = torch.zeros(batch_size, len(self.distances))

    def __define_grid_params(self):
        if self.system_model.params.field_type.startswith("Far"):
            # if it's the Far field case, need to init angles range.
            self.angels = torch.arange(-1 * torch.pi / 3, torch.pi / 3, torch.pi / 720, device=device,
                                       dtype=torch.float64).requires_grad_(True).to(torch.float64)
        elif self.system_model.params.field_type.startswith("Near"):
            # if it's the Near field, there are 3 possabilities.
            fresnel = self.system_model.fresnel
            fraunhofer = self.system_model.fraunhofer
            if self.estimation_params.startswith("angle"):
                self.angels = torch.arange(-1 * torch.pi / 3, torch.pi / 3, torch.pi / 360,
                                           device=device).requires_grad_(True).to(torch.float64)
                # self.angels = torch.from_numpy(np.arange(-np.pi / 2, np.pi / 2, np.pi / 90)).requires_grad_(True)
            if self.estimation_params.endswith("range"):
                self.distances = torch.arange(np.floor(fresnel), fraunhofer * 0.5, .5, device=device,
                                              dtype=torch.float64).requires_grad_(True)
            else:
                raise ValueError(f"estimation_parameter allowed values are [(angle), (range), (angle, range)],"
                                 f" got {self.estimation_params}")
        else:
            raise ValueError(f"Unrecognized field type for MUSIC class init stage,"
                             f" got {self.system_model.params.field_type} but only Far and Near are allowed.")

    def _plot_1d_spectrum(self, highlight_corrdinates, batch):
        if self.estimation_params == "angle":
            x = np.rad2deg(self.angels.detach().cpu().numpy())
            x_label = "angle [deg]"
        elif self.estimation_params == "range":
            x = self.distances.detach().cpu().numpy()
            x_label = "distance [m]"
        else:
            raise ValueError(f"_plot_1d_spectrum: No such option for param estimation.")
        y = self.music_spectrum[batch].detach().cpu().numpy()
        plt.figure()
        plt.plot(x, y.T, label="Music Spectrum")
        if highlight_corrdinates is not None:
            for idx, dot in enumerate(highlight_corrdinates):
                plt.scatter(dot, label=f"dot {idx}")
        plt.title("MUSIC SPECTRUM")
        plt.grid()
        plt.ylabel("Spectrum power")
        plt.xlabel(x_label)
        plt.legend()
        plt.show()

    def _plot_3d_spectrum(self, highlight_coordinates, batch, method):
        """
        Plot the MUSIC 2D spectrum.

        """
        if method == "3D":
            # Creating figure
            distances = self.distances.detach().cpu().numpy()
            angles = self.angels.detach().cpu().numpy()
            spectrum = self.music_spectrum[batch].detach().cpu().numpy()
            x, y = np.meshgrid(distances, np.rad2deg(angles))
            # Plotting the 3D surface
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(x, y, 10 * np.log10(spectrum), cmap='viridis')

            if highlight_coordinates:
                highlight_coordinates = np.array(highlight_coordinates)
                ax.scatter(
                    highlight_coordinates[:, 0],
                    np.rad2deg(highlight_coordinates[:, 1]),
                    np.log1p(highlight_coordinates[:, 2]),
                    color='red',
                    s=50,
                    label='Highlight Points'
                )
            ax.set_title('MUSIC spectrum')
            ax.set_xlim(distances[0], distances[-1])
            ax.set_ylim(np.rad2deg(angles[0]), np.rad2deg(angles[-1]))
            # Adding labels
            ax.set_ylabel('Theta [deg]')
            ax.set_xlabel('Radius [m]')
            ax.set_zlabel('Power [dB]')
            plt.colorbar(ax.plot_surface(x, y, 10 * np.log10(spectrum), cmap='viridis'), shrink=0.5, aspect=5)

            if highlight_coordinates:
                ax.legend()  # Adding a legend

            # Display the plot
            plt.show()
        elif method == "heatmap":
            xmin, xmax = np.min(self.distances.cpu().detach().numpy()), np.max(self.distances.cpu().detach().numpy())
            ymin, ymax = np.min(self.angels.cpu().detach().numpy()), np.max(self.angels.cpu().detach().numpy())
            spectrum = self.music_spectrum[batch].cpu().detach().numpy()
            plt.imshow(spectrum, cmap="hot",
                       extent=[xmin, xmax, np.rad2deg(ymin), np.rad2deg(ymax)], origin='lower', aspect="auto")
            if highlight_coordinates is not None:
                for idx, dot in enumerate(highlight_coordinates):
                    x = self.distances.cpu().detach().numpy()[dot[1]]
                    y = np.rad2deg(self.angels.cpu().detach().numpy()[dot[0]])
                    plt.scatter(x, y, label=f"dot {idx}: ({np.round(x, decimals=3), np.round(y, decimals=3)})")
                plt.legend()
            plt.colorbar()
            plt.title("MUSIC Spectrum heatmap")
            plt.xlabel("Distances [m]")
            plt.ylabel("Angles [deg]")

            plt.figaspect(2)
            plt.show()

    def __str__(self):
        return f"music_{self.estimation_params}"


def keep_far_enough_points(tensor, M, D):
    # # Calculate pairwise distances between columns
    # distances = cdist(tensor.T, tensor.T, metric="euclidean")
    #
    # # Keep the first M columns as far enough points
    # selected_cols = []
    # for i in range(tensor.shape[1]):
    #     if len(selected_cols) >= M:
    #         break
    #     if all(distances[i, col] >= D for col in selected_cols):
    #         selected_cols.append(i)
    #
    # # Remove columns that are less than distance D from each other
    # filtered_tensor = tensor[:, selected_cols]
    # retrun filtered_tensor
    ##############################################
    # Extract x_coords (first dimension)
    x_coords = tensor[0, :]

    # Keep the first M columns that are far enough apart in x_coords
    selected_cols = []
    for i in range(tensor.shape[1]):
        if len(selected_cols) >= M:
            break
        if i == 0:
            selected_cols.append(i)
            continue
        if all(abs(x_coords[i] - x_coords[col]) >= D for col in selected_cols):
            selected_cols.append(i)

    # Select the columns that meet the distance criterion
    filtered_tensor = tensor[:, selected_cols]

    return filtered_tensor


class Filter(nn.Module):
    def __init__(self, min_cell_size, max_cell_size, number_of_filter=10):
        super(Filter, self).__init__()
        self.number_of_filters = number_of_filter
        self.cell_sizes = torch.linspace(min_cell_size, max_cell_size, number_of_filter).to(torch.int32).to(device)
        self.cell_bank = {}
        for cell_size in enumerate(self.cell_sizes.data):
            cell_size = cell_size[1]
            self.cell_bank[cell_size] = torch.arange(-cell_size, cell_size, 1, dtype=torch.long, device=device)
        self.fc = nn.Linear(self.number_of_filters, 1)
        self.fc.weight.data = torch.randn(1, number_of_filter) / 100 + (1 / number_of_filter)
        self.fc.weight.data = self.fc.weight.data.to(torch.float64)
        self.fc.bias.data = torch.Tensor([0])
        self.fc.bias.data = self.fc.bias.data.to(torch.float64)
        self.fc.bias.requires_grad_(False)
        self.relu = nn.ReLU()

    def forward(self, input, search_space):
        peaks = torch.zeros(input.shape[0], 1).to(torch.int64)
        for batch in range(peaks.shape[0]):
            music_spectrum = input[batch].cpu().detach().numpy().squeeze()
            # Find spectrum peaks
            peaks_tmp = list(sc.signal.find_peaks(music_spectrum)[0])
            # Sort the peak by their amplitude
            peaks_tmp.sort(key=lambda x: music_spectrum[x], reverse=True)
            if len(peaks_tmp) == 0:
                peaks_tmp = torch.randint(search_space.shape[0], (1,))
            else:
                peaks_tmp = peaks_tmp[0]
            peaks[batch] = peaks_tmp
        top_1 = peaks
        output = torch.zeros(input.shape[0], self.number_of_filters).to(device).to(torch.float64)
        for idx, cell in enumerate(self.cell_bank.values()):
            tmp_cell = top_1 + cell
            tmp_cell %= input.shape[1]
            tmp_cell = tmp_cell.unsqueeze(-1)
            metrix_thr = torch.gather(input.unsqueeze(-1).expand(-1, -1, tmp_cell.size(-1)), 1, tmp_cell)
            soft_max = torch.softmax(metrix_thr, dim=1)
            output[:, idx] = torch.einsum("bkm, bkm -> bm", search_space[tmp_cell], soft_max).squeeze()
        output = self.fc(output)
        output = self.relu(output)
        self.clip_weights_values()
        return output

    def clip_weights_values(self):
        self.fc.weight.data = torch.clip(self.fc.weight.data, 0.1, 1)
        self.fc.weight.data /= torch.sum(self.fc.weight.data)
