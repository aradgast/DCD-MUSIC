import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.utils import *
from src.system_model import SystemModel


class SubspaceMethod(nn.Module):
    """

    """

    def __init__(self, system_model: SystemModel=None):
        super(SubspaceMethod, self).__init__()
        self.eigen_threshold = nn.Parameter(torch.tensor(.5, requires_grad=False))
        self.normalized_eigenvals = None

    def subspace_separation(self,
                            covariance: torch.Tensor,
                            number_of_sources: torch.tensor = None) \
            -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.tensor):
        """

        Args:
            covariance:
            number_of_sources:

        Returns:
            the signal ana noise subspaces, both as torch.Tensor().
        """
        eigenvalues, eigenvectors = torch.linalg.eig(covariance)
        sorted_idx = torch.argsort(torch.real(eigenvalues), descending=True)
        sorted_eigvectors = torch.gather(eigenvectors, 2,
                                         sorted_idx.unsqueeze(-1).expand(-1, -1, covariance.shape[-1]).transpose(1, 2))
        # number of sources estimation
        real_sorted_eigenvals = torch.gather(torch.real(eigenvalues), 1, sorted_idx)
        self.normalized_eigen = real_sorted_eigenvals / real_sorted_eigenvals[:, 0][:, None]
        source_estimation = torch.linalg.norm(
            nn.functional.relu(
                self.normalized_eigen - self.__get_eigen_threshold() * torch.ones_like(self.normalized_eigen)),
            dim=1, ord=0).to(torch.int)
        if number_of_sources is None:
            warnings.warn("Number of sources is not defined, using the number of sources estimation.")
        # if source_estimation == sorted_eigvectors.shape[2]:
        #     source_estimation -= 1
            signal_subspace = sorted_eigvectors[:, :, :source_estimation]
            noise_subspace = sorted_eigvectors[:, :, source_estimation:]
        else:
            signal_subspace = sorted_eigvectors[:, :, :number_of_sources]
            noise_subspace = sorted_eigvectors[:, :, number_of_sources:]

        if self.training:
            l_eig = self.eigen_regularization(number_of_sources)
        else:
            l_eig = None

        return signal_subspace.to(device), noise_subspace.to(device), source_estimation, l_eig

    def eigen_regularization(self, number_of_sources: int):
        """

        Args:
            normalized_eigenvalues:
            number_of_sources:

        Returns:

        """
        l_eig = (self.normalized_eigen[:, number_of_sources - 1] - self.__get_eigen_threshold(level="high")) * \
                (self.normalized_eigen[:, number_of_sources] - self.__get_eigen_threshold(level="low"))
        # l_eig = -(self.normalized_eigen[:, number_of_sources - 1] - self.__get_eigen_threshold(level="high")) + \
                # (self.normalized_eigen[:, number_of_sources] - self.__get_eigen_threshold(level="low"))
        l_eig = torch.sum(l_eig)
        # eigen_regularization = nn.functional.elu(eigen_regularization, alpha=1.0)
        return l_eig

    def __get_eigen_threshold(self, level: str = None):
        if self.training:
            if level is None:
                return self.eigen_threshold
            elif level == "high":
                return self.eigen_threshold + 0.0
            elif level == "low":
                return self.eigen_threshold - 0.0
        else:
            return self.eigen_threshold + 0.1

    def pre_processing(self, x: torch.Tensor, mode: str = "sample"):
        if mode == "sample":
            Rx = self.__sample_covariance(x)
        elif mode == "sps":
            Rx = self.__spatial_smoothing_covariance(x)
        elif mode == "subcarrier":
            # calculate the covariance matrix for each subcarrier
            batch_size, sensor_number, subcarriers_number, samples_number = x.shape
            Rx = torch.zeros(batch_size, subcarriers_number, sensor_number, sensor_number, dtype=torch.complex128, device=device)
            for k in range(subcarriers_number):
                Rx[:, k] = self.pre_processing(x[:, :, k, :], mode="sample")
            # # focusing method - just sum the covariance matrices in the subcarriers dimension
            # Rx = torch.sum(Rx, dim=0)
            # # make sure the first dimension is the batch size
            # if Rx.dim() == 2:
            #     Rx = Rx[None, :, :]
        else:
            raise ValueError(
                f"SubspaceMethod.pre_processing: method {mode} is not recognized for covariance calculation.")

        return Rx

    def __sample_covariance(self, x: torch.Tensor):
        """
        Calculates the sample covariance matrix.

        Args:
        -----
            X (np.ndarray): Input samples matrix.

        Returns:
        --------
            covariance_mat (np.ndarray): Covariance matrix.
        """
        if x.dim() == 2:
            x = x[None, :, :]
        batch_size, sensor_number, samples_number = x.shape
        Rx = torch.einsum("bmt, btl -> bml", x, torch.conj(x).transpose(1, 2)) / samples_number
        return Rx

    def __spatial_smoothing_covariance(self, x: torch.Tensor):
        """
        Calculates the covariance matrix using spatial smoothing technique.

        Args:
        -----
            X (np.ndarray): Input samples matrix.

        Returns:
        --------
            covariance_mat (np.ndarray): Covariance matrix.
        """

        if x.dim() == 2:
            x = x[None, :, :]
        batch_size, sensor_number, samples_number = x.shape
        # Define the sub-arrays size
        sub_array_size = sensor_number // 2 + 1
        # Define the number of sub-arrays
        number_of_sub_arrays = sensor_number - sub_array_size + 1
        # Initialize covariance matrix
        Rx_smoothed = torch.zeros(batch_size, sub_array_size, sub_array_size, dtype=torch.complex128, device=device)

        for j in range(number_of_sub_arrays):
            # Run over all sub-arrays
            x_sub = x[:, j:j + sub_array_size, :]
            # Calculate sample covariance matrix for each sub-array
            sub_covariance = torch.einsum("bmt, btl -> bml", x_sub, torch.conj(x_sub).transpose(1, 2)) / (samples_number-1)
            # Aggregate sub-arrays covariances
            Rx_smoothed += sub_covariance.to(device) / number_of_sub_arrays
        # Divide overall matrix by the number of sources
        return Rx_smoothed

    def plot_eigen_spectrum(self, batch_idx: int=0):
        """
        Plot the eigenvalues spectrum.

        Args:
        -----
            batch_idx (int): Index of the batch to plot.
        """
        plt.figure()
        plt.stem(self.normalized_eigen[batch_idx].cpu().detach().numpy(), label="Normalized Eigenvalues")
        # ADD threshold line
        plt.axhline(y=self.__get_eigen_threshold(), color='r', linestyle='--', label="Threshold")
        plt.title("Eigenvalues Spectrum")
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Eigenvalue")
        plt.legend()
        plt.grid()
        plt.show()
