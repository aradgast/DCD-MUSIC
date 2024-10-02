"""
TODO: Add description and implementation
"""
from typing import List, Tuple

from FR3.utils.constants import *

class Beamformer:
    """
    The classical Beamformer - single sub-band version
    """

    def run(self, y: np.ndarray, basis_vectors: List[np.ndarray], second_dim: bool = False, use_gpu=False) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        y is channel observations
        basis vectors are the set/sets of beamforming vectors in the dictionary
        if second_dim is False then do a 1 dimensional peak search
        if second_dim is True do a 2 dimensions peak search
        returns a tuple of the [max_ind, spectrum]
        """
        # find the peak in 2D spectrum
        if second_dim:
            # compute the spectrum values for each basis vector
            norm_values = self._compute_beamforming_spectrum(basis_vectors, use_gpu, y)
            # unravel the peak's maximum index by the spectrum size
            maximum_ind = np.array(np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape))
        # find the peak in 1D spectrum
        else:
            # compute the beamforming spectrum
            matmul_res = np.einsum('ij,jmk->imk', basis_vectors.conj(), y)
            norm_values = np.linalg.norm(np.linalg.norm(matmul_res, axis=1), axis=1)
            # return peak
            maximum_ind = np.argmax(norm_values)
        return np.array([maximum_ind]), norm_values

    def _compute_beamforming_spectrum(self, basis_vectors: List[np.ndarray], use_gpu: bool,
                                      y: np.ndarray) -> np.ndarray:
        # compute with cpu
        if not use_gpu:
            aoas = basis_vectors[0]
            toas = basis_vectors[1]
            left_matmul = np.einsum('ij,jmk->imk', aoas.conj(), y)
            right_matmul = np.einsum('ijk,jm->imk', left_matmul, toas.conj().T)
            norm_values = np.linalg.norm(right_matmul, axis=2)
        # do calculations on GPU - much faster for big matrices
        else:
            y = torch.tensor(y).to(DEVICE)
            aoas = basis_vectors[0]
            toas = basis_vectors[1]
            left_matmul = torch.einsum('ij,jmk->imk', aoas.conj(), y)
            right_matmul = torch.einsum('ijk,jm->imk', left_matmul, toas.conj().T)
            norm_values = torch.linalg.norm(right_matmul, axis=2)
            norm_values = norm_values.cpu().numpy()
        return norm_values



class MultiBandBeamformer(Beamformer):
    """
    The Proposed Multi-Band Beamformer.
    """

    def __init__(self):
        super(MultiBandBeamformer, self).__init__()

    def run(self, y: List[np.ndarray], basis_vectors: List[np.ndarray], second_dim: bool = False,
            use_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        K = len(y)
        peak, chosen_k = None, None
        norm_values_list = []
        for k in range(K):
            # compute the spectrum values for sub-band
            norm_values = self._compute_beamforming_spectrum(basis_vectors[k], use_gpu, y[k])
            maximum_ind = np.array(np.unravel_index(np.argmax(norm_values, axis=None), norm_values.shape))
            norm_values_list.append(norm_values)
            # only if peak is above noise level
            if norm_values[maximum_ind[0], maximum_ind[1]] > ALG_THRESHOLD * np.mean(norm_values):
                peak = maximum_ind
                chosen_k = k
        # if all are below the noise level - choose the last sub-band
        if chosen_k is None:
            peak = maximum_ind
            chosen_k = K - 1
        self.k = chosen_k
        return np.array([peak]), norm_values_list[self.k]