import torch
import torch.nn as nn

from src.methods_pack.subspace_method import SubspaceMethod
from src.system_model import SystemModel
from src.utils import *


class RootMusic(SubspaceMethod):
    def __init__(self, system_model: SystemModel):
        super(RootMusic, self).__init__(system_model)

    def forward(self, cov: torch.Tensor):
        batch_size = cov.shape[0]
        _, noise_subspace = self.subspace_separation(cov, number_of_sources=self.system_model.params.M)
        poly_generator = torch.einsum("bnk, bkn -> bnn", noise_subspace, noise_subspace.conj().transpose(1, 2))
        diag_sum = self.sum_of_diag(poly_generator)
        roots = self.find_roots(diag_sum)
        # Calculate doa
        # angles_prediction_all = self.get_doa_from_roots(roots)

        # the actual prediction is for roots that are closest to the unit circle.
        roots = self.extract_roots_closest_unit_circle(roots, k=self.system_model.params.M)
        angles_prediction = self.get_doa_from_roots(roots)
        return angles_prediction

    def get_doa_from_roots(self, roots):
        roots_phase = torch.angle(roots)
        angle_predicted = torch.arcsin(
            (1 / (2 * np.pi * self.system_model.dist_array_elems["Narrowband"])) * roots_phase)
        return angle_predicted

    def extract_roots_closest_unit_circle(self, roots, k: int):
        distances = torch.abs(torch.abs(roots) - 1)
        sorted_indcies = torch.argsort(distances, dim=1)
        k_closest = sorted_indcies[:, :k]
        closest_roots = roots.gather(1, k_closest)
        return closest_roots

    def sum_of_diag(self, tensor: torch.Tensor):
        tensor = self.__check_diag_sums_dim(tensor)
        N = torch.shape[-1]
        diag_indcies = torch.linspace(-N + 1, N - 1, 2 * N - 1)
        sum_of_diags = torch.zeros(tensor.shape[0], 2 * N - 1)
        for idx, diag_idx in enumerate(diag_indcies):
            sum_of_diags[:, idx] = torch.sum(torch.diagonal(tensor, dim1=-2, dim2=-1, offset=diag_idx), dim=-1)
        return sum_of_diags

    def find_roots(self, coeffs: torch.Tensor):
        A = torch.diag(torch.ones(coeffs.shape[-1] - 2, dtype=coeffs.dtype), -1)
        A = A.repeat(coeffs.shape[0], 1, 1)  # repeat for all elements in the batch
        A[:, 0, :] = -coeffs[:, 1:] / coeffs[:, 0]
        roots = torch.linalg.eigvals(A)
        return roots

    def __check_diag_sums_dim(self, tensor):
        if len(tensor.shape) != 3:
            if len(tensor.shape) == 2:
                tensor = tensor.unsqueeze(0)
            else:
                raise ValueError("sum_of_diag: Input tensor shape should be 2 or 3 dim")
        else:
            if tensor.shape[-1] != tensor.shape[-2]:
                raise ValueError("sum_of_diag: input tensor should be square matrices as a batch.")
        return tensor


def root_music(Rz: torch.Tensor, M: int, batch_size: int):
    """Implementation of the model-based Root-MUSIC algorithm, support Pytorch, intended for
        MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
        as it accepts the surrogate covariance matrix.
        it is equivalent tosrc.methods: RootMUSIC.narrowband() method.

    Args:
    -----
        Rz (torch.Tensor): Focused covariance matrix
        M (int): Number of sources
        batch_size: the number of batches

    Returns:
    --------
        doa_batches (torch.Tensor): The predicted doa, over all batches.
        doa_all_batches (torch.Tensor): All doa predicted, given all roots, over all batches.
        roots_to_return (torch.Tensor): The unsorted roots.
    """

    dist = 0.5
    f = 1
    doa_batches = []
    doa_all_batches = []
    Bs_Rz = Rz
    for iter in range(batch_size):
        R = Bs_Rz[iter]
        # Extract eigenvalues and eigenvectors using EVD
        eigenvalues, eigenvectors = torch.linalg.eig(R)
        # Assign noise subspace as the eigenvectors associated with M greatest eigenvalues
        Un = eigenvectors[:, torch.argsort(torch.abs(eigenvalues)).flip(0)][:, M:]
        # Generate hermitian noise subspace matrix
        F = torch.matmul(Un, torch.t(torch.conj(Un)))
        # Calculates the sum of F matrix diagonals
        diag_sum = sum_of_diags_torch(F)
        # Calculates the roots of the polynomial defined by F matrix diagonals
        roots = find_roots_torch(diag_sum)
        # Calculate the phase component of the roots
        roots_angels_all = torch.angle(roots)
        # Calculate doa
        doa_pred_all = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels_all)
        doa_all_batches.append(doa_pred_all)
        roots_to_return = roots
        # Take only roots which inside the unit circle
        roots = roots[
            sorted(range(roots.shape[0]), key=lambda k: abs(abs(roots[k]) - 1))
        ]
        mask = (torch.abs(roots) - 1) < 0
        roots = roots[mask][:M]
        # Calculate the phase component of the roots
        roots_angels = torch.angle(roots)
        # Calculate doa
        doa_pred = torch.arcsin((1 / (2 * np.pi * dist * f)) * roots_angels)
        doa_batches.append(doa_pred)

    return (
        torch.stack(doa_batches, dim=0),
        torch.stack(doa_all_batches, dim=0),
        roots_to_return,
    )
