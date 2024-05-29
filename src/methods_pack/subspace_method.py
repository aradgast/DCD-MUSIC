import torch
import torch.nn as nn

from src.utils import *
from src.system_model import SystemModel

class SubspaceMethod(nn.Module):
    """

    """

    def __init__(self, system_model: SystemModel):
        super(SubspaceMethod, self).__init__()
        self.system_model = system_model

    def subspace_separation(self, covariance: torch.Tensor, number_of_sources: int = None):
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
        signal_subspace = sorted_eigvectors[:, :, :number_of_sources]
        noise_subspace = sorted_eigvectors[:, :, number_of_sources:]
        return signal_subspace.to(device), noise_subspace.to(device)