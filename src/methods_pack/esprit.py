import torch
import torch.nn as nn

from src.methods_pack.subspace_method import SubspaceMethod
from src.system_model import SystemModel


class ESPRIT(SubspaceMethod):
    def __init__(self, system_model: SystemModel):
        super().__init__(system_model)

    def forward(self, cov: torch.Tensor):
        # get the signal subspace
        signal_subspace, _ = self.subspace_separation(cov, number_of_sources=self.system_model.params.M)
        # create 2 overlapping matrices
        upper = signal_subspace[:, :-1]
        lower = signal_subspace[:, 1:]
        phi = torch.linalg.lstsq(upper, lower)[0]  # identical to pinv(A) @ B but faster and stable.
        eigvalues, _ = torch.linalg.eig(phi)
        eigvals_phase = torch.angle(eigvalues)
        prediction = -1 * torch.arcsin((1 / torch.pi) * eigvals_phase)

        return prediction

    def __str__(self):
        return "esprit"


def esprit(Rz: torch.Tensor, M: int, batch_size: int):
    """Implementation of the model-based Esprit algorithm, support Pytorch, intended for
        MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
        as it accepts the surrogate covariance matrix.
        it is equivalent to src.methods: RootMUSIC.narrowband() method.

    Args:
    -----
        Rz (torch.Tensor): Focused covariance matrix
        M (int): Number of sources
        batch_size: the number of batches

    Returns:
    --------
        doa_batches (torch.Tensor): The predicted doa, over all batches.
    """

    doa_batches = []

    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        # Extract eigenvalues and eigenvectors using EVD
        eigenvalues, eigenvectors = torch.linalg.eig(R)

        # Get signal subspace
        Us = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, :M]
        # Separate the signal subspace into 2 overlapping subspaces
        Us_upper, Us_lower = (
            Us[0: R.shape[0] - 1],
            Us[1: R.shape[0]],
        )
        # Generate Phi matrix
        phi = torch.linalg.pinv(Us_upper) @ Us_lower
        # Find eigenvalues and eigenvectors (EVD) of Phi
        phi_eigenvalues, _ = torch.linalg.eig(phi)
        # Calculate the phase component of the roots
        eigenvalues_angels = torch.angle(phi_eigenvalues)
        # Calculate the DoA out of the phase component
        doa_predictions = -1 * torch.arcsin((1 / torch.pi) * eigenvalues_angels)
        doa_batches.append(doa_predictions)

    return torch.stack(doa_batches, dim=0)
