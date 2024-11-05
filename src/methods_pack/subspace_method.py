import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.utils import *
from src.system_model import SystemModel


class SubspaceMethod(nn.Module):
    """

    """

    def __init__(self, system_model: SystemModel):
        super(SubspaceMethod, self).__init__()
        self.system_model = system_model
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
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
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

    def get_noise_subspace(self, covariance: torch.Tensor, number_of_sources: int):
        """

        Args:
            covariance:
            number_of_sources:

        Returns:

        """
        _, noise_subspace, _, _ = self.subspace_separation(covariance, number_of_sources)
        return noise_subspace

    def get_signal_subspace(self, covariance: torch.Tensor, number_of_sources: int):
        """

        Args:
            covariance:
            number_of_sources:

        Returns:

        """
        signal_subspace, _, _, _ = self.subspace_separation(covariance, number_of_sources)
        return signal_subspace

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

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def __init_criteria(self):
        raise NotImplementedError

    def __get_eigen_threshold(self, level: str = None):
        if self.training:
            if level is None:
                return self.eigen_threshold
            elif level == "high":
                return self.eigen_threshold + 0.0
            elif level == "low":
                return self.eigen_threshold - 0.0
        else:
            if self.system_model.params.M is not None:
                return self.eigen_threshold - self.system_model.params.M / self.system_model.params.N
            else:
                return self.eigen_threshold - 0.1

    def pre_processing(self, x: torch.Tensor, mode: str = "sample"):
        if mode == "sample":
            Rx = sample_covariance(x)
        elif mode == "sps":
            Rx = spatial_smoothing_covariance(x)
        else:
            raise ValueError(
                f"SubspaceMethod.pre_processing: method {mode} is not recognized for covariance calculation.")

        return Rx


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
        plt.axhline(y=self.__get_eigen_threshold().cpu().detach().numpy(), color='r', linestyle='--', label="Threshold")
        plt.title("Eigenvalues Spectrum")
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Eigenvalue")
        plt.legend()
        plt.grid()
        plt.show()

    def source_estimation_accuracy(self, sources_num, source_estimation):
        return torch.sum(source_estimation == sources_num * torch.ones_like(source_estimation).float()).item()