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
        sorted_idx = torch.argsort(torch.abs(eigenvalues), descending=True)
        sorted_eigvectors = torch.gather(eigenvectors, 2,
                                         sorted_idx.unsqueeze(-1).expand(-1, -1, covariance.shape[-1]).transpose(1, 2))
        # number of sources estimation
        source_estimation, l_eig = self.estimate_number_of_sources(eigenvalues, method="threshold",
                                                                   number_of_sources=number_of_sources)
        if number_of_sources is None:
            warnings.warn("Number of sources is not defined, using the number of sources estimation.")
        # if source_estimation == sorted_eigvectors.shape[2]:
        #     source_estimation -= 1
            signal_subspace = sorted_eigvectors[:, :, :source_estimation]
            noise_subspace = sorted_eigvectors[:, :, source_estimation:]
        else:
            signal_subspace = sorted_eigvectors[:, :, :number_of_sources]
            noise_subspace = sorted_eigvectors[:, :, number_of_sources:]

        return signal_subspace.to(device), noise_subspace.to(device), source_estimation, l_eig

    def estimate_number_of_sources(self, eigenvalues, method:str="threshold", number_of_sources: int = None):
        """

        Args:
            eigenvalues:

        Returns:

        """
        sorted_eigenvals = torch.sort(torch.real(eigenvalues), descending=True, dim=1).values
        l_eig = None
        if method == "threshold":
            self.normalized_eigenvals = sorted_eigenvals / sorted_eigenvals[:, 0][:, None]
            source_estimation = torch.linalg.norm(
                nn.functional.relu(
                    self.normalized_eigenvals - self.__get_eigen_threshold() * torch.ones_like(self.normalized_eigenvals)),
                dim=1, ord=0).to(torch.int)
            # return regularization term if training
            if self.training:
                l_eig = self.eigen_regularization(number_of_sources)
        elif method in ["mdl", "aic"]:
            # mdl -> calculate the value of the mdl test for each number of sources
            # and choose the number of sources that minimizes the mdl test
            optimal_test = torch.ones(eigenvalues.shape[0], device=device) * float("inf")
            optimal_m = torch.zeros(eigenvalues.shape[0], device=device)
            for m in range(1, eigenvalues.shape[1]):
                m = torch.tensor(m, device=device)
                # calculate the test
                test = self.hypothesis_testing(sorted_eigenvals, m, method=method)
                # update the optimal number of sources by masking the current number of sources
                optimal_m = torch.where(test < optimal_test, m, optimal_m)
                # update the optimal mdl value
                optimal_test = torch.where(test < optimal_test, test, optimal_test)
                if self.training and m == number_of_sources:
                    l_eig = torch.sum(test)

            source_estimation = optimal_m

        else:
            raise ValueError(f"SubspaceMethod.estimate_number_of_sources: method {method} is not recognized.")
        return source_estimation, l_eig

    def hypothesis_testing(self, eigenvalues, number_of_sources, method="mdl"):
        # extract the number of snapshots and the number of antennas
        T = self.system_model.params.T
        N = self.system_model.params.N
        M = number_of_sources
        # calculate the number of degrees of freedom
        dof = 2 * N * M - M ** 2 + 1
        if method.lower().startswith("mdl"):
            penalty = dof * np.log(T)
        elif method.lower().startswith("aic"):
            penalty = 2 * dof
        else:
            raise ValueError(f"SubspaceMethod.hypothesis_testing: method {method} is not recognized.")
        if method.lower().endswith("snr"):
            snr = self.snr_estimation(eigenvalues, M)
            # snr = self.system_model.params.snr
            penalty = penalty * (1 + 0.1 * torch.exp(-snr))
            # penalty = penalty*(1 + 2*np.exp(-snr))
        # calculate the ll
        ll = self.get_ll(eigenvalues, M)
        mdl = ll + penalty
        return mdl

    def snr_estimation(self, eigenvalues, M):
        snr = 10 * torch.log10(torch.mean(eigenvalues[:, :M], dim=1) / torch.mean(eigenvalues[:, M:], dim=1))
        return snr

    def get_ll(self, eigenvalues, M):
        T = self.system_model.params.T
        N = self.system_model.params.N
        ll = -T * torch.sum(torch.log(eigenvalues[:, M:]), dim=1) + T * (N - M) * torch.log(torch.mean(eigenvalues[:, M:], dim=1))
        return ll

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
        l_eig = (self.normalized_eigenvals[:, number_of_sources - 1] - self.__get_eigen_threshold(level="high")) * \
                (self.normalized_eigenvals[:, number_of_sources] - self.__get_eigen_threshold(level="low"))
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
        # if self.training:
        #     if level is None:
        #         return self.eigen_threshold
        #     elif level == "high":
        #         return self.eigen_threshold + 0.0
        #     elif level == "low":
        #         return self.eigen_threshold - 0.0
        # else:
        #     if self.system_model.params.M is not None:
        #         return self.eigen_threshold - self.system_model.params.M / self.system_model.params.N
        #     else:
        #         return self.eigen_threshold - 0.1
        return self.eigen_threshold

    def pre_processing(self, x: torch.Tensor, mode: str = "sample"):
        if mode == "sample":
            Rx = sample_covariance(x)
        elif mode == "sps":
            Rx = spatial_smoothing_covariance(x)
            try:
                self.update_number_of_sensors(Rx.shape[1])
            except AttributeError:
                pass
        else:
            raise ValueError(
                f"SubspaceMethod.pre_processing: method {mode} is not recognized for covariance calculation.")

        return Rx


    def plot_eigen_spectrum(self, batch_idx: int=0, normelized_eign: torch.Tensor = None):
        """
        Plot the eigenvalues spectrum.

        Args:
        -----
            batch_idx (int): Index of the batch to plot.
        """
        if normelized_eign is None:
            normelized_eign = self.normalized_eigenvals
        plt.figure()
        plt.stem(normelized_eign[batch_idx].cpu().detach().numpy(), label="Normalized Eigenvalues")
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