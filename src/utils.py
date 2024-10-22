"""Subspace-Net 
Details
----------
Name: utils.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose:
--------
This script defines some helpful functions:
    * sum_of_diag: returns the some of each diagonal in a given matrix.
    * sum_of_diag_torch: returns the some of each diagonal in a given matrix, Pytorch oriented.
    * find_roots: solves polynomial equation defines by polynomial coefficients. 
    * find_roots_torch: solves polynomial equation defines by polynomial coefficients, Pytorch oriented.. 
    * set_unified_seed: Sets unified seed for all random attributed in the simulation.
    * get_k_angles: Retrieves the top-k angles from a prediction tensor.
    * get_k_peaks: Retrieves the top-k peaks (angles) from a prediction tensor using peak finding.
    * gram_diagonal_overload(self, Kx: torch.Tensor, eps: float): generates Hermitian and PSD (Positive Semi-Definite) matrix,
        using gram operation and diagonal loading.
"""

# Imports
import numpy as np
import torch

torch.cuda.empty_cache()
import random
import scipy
import warnings

# Constants
R2D = 180 / np.pi
D2R = 1 / R2D
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plot_styles = {
    'CCRB': {'color': 'r', 'linestyle': '-', 'marker': 'o', "markersize": 10},
    'DCDMUSIC': {'color': 'g', 'linestyle': '--', 'marker': 's', "markersize": 10},
    'TransMUSIC': {'color': 'b', 'linestyle': '-.', 'marker': 'd', "markersize": 10},
    '2D-MUSIC': {'color': 'c', 'linestyle': ':', 'marker': '^', "markersize": 10},
    '2D-MUSIC(SPS)': {'color': 'c', 'linestyle': ':', 'marker': '^', "markersize": 10},
    'SubspaceNet': {'color': 'y', 'linestyle': '--', 'marker': 'p', "markersize": 10},
    'esprit': {'color': 'm', 'linestyle': '-', 'marker': 'v', "markersize": 10},
    'esprit(SPS)': {'color': 'm', 'linestyle': '-', 'marker': 'v', "markersize": 10},
    'music': {'color': 'g', 'linestyle': '--', 'marker': 's', "markersize": 10},
    'music(SPS)': {'color': 'g', 'linestyle': '--', 'marker': 's', "markersize": 10},
}
# device = "cpu"
print("Running on device: ", device)


# Functions
# def sum_of_diag(matrix: np.ndarray) -> list:
def sum_of_diag(matrix: np.ndarray):
    """Calculates the sum of diagonals in a square matrix.

    Args:
        matrix (np.ndarray): Square matrix for which diagonals need to be summed.

    Returns:
        list: A list containing the sums of all diagonals in the matrix, from left to right.

    Raises:
        None

    Examples:
        >>> matrix = np.array([[1, 2, 3],
                               [4, 5, 6],
                               [7, 8, 9]])
        >>> sum_of_diag(matrix)
        [7, 12, 15, 8, 3]

    """
    diag_sum = []
    diag_index = np.linspace(
        -matrix.shape[0] + 1,
        matrix.shape[0] + 1,
        2 * matrix.shape[0] - 1,
        endpoint=False,
        dtype=int,
    )
    for idx in diag_index:
        diag_sum.append(np.sum(matrix.diagonal(idx)))
    return diag_sum


def sum_of_diags_torch(matrix: torch.Tensor):
    """Calculates the sum of diagonals in a square matrix.
    equivalent sum_of_diag, but support Pytorch.

    Args:
        matrix (torch.Tensor): Square matrix for which diagonals need to be summed.

    Returns:
        torch.Tensor: A list containing the sums of all diagonals in the matrix, from left to right.

    Raises:
        None

    Examples:
        >>> matrix = torch.tensor([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9]])
        >>> sum_of_diag(matrix)
            torch.tensor([7, 12, 15, 8, 3])
    """
    diag_sum = []
    diag_index = torch.linspace(
        -matrix.shape[0] + 1, matrix.shape[0] - 1, 2 * matrix.shape[0] - 1, dtype=int
    )
    for idx in diag_index:
        diag_sum.append(torch.sum(torch.diagonal(matrix, idx)))
    return torch.stack(diag_sum, dim=0)


# def find_roots(coefficients: list) -> np.ndarray:
def find_roots(coefficients: list):
    """Finds the roots of a polynomial defined by its coefficients.

    Args:
        coefficients (list): List of polynomial coefficients in descending order of powers.

    Returns:
        np.ndarray: An array containing the roots of the polynomial.

    Raises:
        None

    Examples:
        >>> coefficients = [1, -5, 6]  # x^2 - 5x + 6
        >>> find_roots(coefficients)
        array([3., 2.])

    """
    coefficients = np.array(coefficients)
    A = np.diag(np.ones((len(coefficients) - 2,), coefficients.dtype), -1)
    if np.abs(coefficients[0]) == 0:
        A[0, :] = -coefficients[1:] / (coefficients[0] + 1e-9)
    else:
        A[0, :] = -coefficients[1:] / coefficients[0]
    roots = np.array(np.linalg.eigvals(A))
    return roots


def find_roots_torch(coefficients: torch.Tensor):
    """Finds the roots of a polynomial defined by its coefficients.
    equivalent to src.utils.find_roots, but support Pytorch.

    Args:
        coefficients (torch.Tensor): List of polynomial coefficients in descending order of powers.

    Returns:
        torch.Tensor: An array containing the roots of the polynomial.

    Raises:
        None

    Examples:
        >>> coefficients = torch.tensor([1, -5, 6])  # x^2 - 5x + 6
        >>> find_roots(coefficients)
        tensor([3., 2.])

    """
    A = torch.diag(torch.ones(len(coefficients) - 2, dtype=coefficients.dtype), -1)
    A[0, :] = -coefficients[1:] / coefficients[0]
    roots = torch.linalg.eigvals(A)
    return roots


def set_unified_seed(seed: int = 42):
    """
    Sets the seed value for random number generators in Python libraries.

    Args:
        seed (int): The seed value to set for the random number generators. Defaults to 42.

    Returns:
        None

    Raises:
        None

    Examples:
        >>> set_unified_seed(42)

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)


# def get_k_angles(grid_size: float, k: int, prediction: torch.Tensor) -> torch.Tensor:
def get_k_angles(grid_size: float, k: int, prediction: torch.Tensor):
    """
    Retrieves the top-k angles from a prediction tensor.

    Args:
        grid_size (float): The size of the angle grid (range) in degrees.
        k (int): The number of top angles to retrieve.
        prediction (torch.Tensor): The prediction tensor containing angle probabilities, sizeof equal to grid_size .

    Returns:
        torch.Tensor: A tensor containing the top-k angles in degrees.

    Raises:
        None

    Examples:
        >>> grid_size = 6
        >>> k = 3
        >>> prediction = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])
        >>> get_k_angles(grid_size, k, prediction)
        tensor([ 90., -18.,   54.])

    """
    angles_grid = torch.linspace(-90, 90, grid_size)
    doa_prediction = angles_grid[torch.topk(prediction.flatten(), k).indices]
    return doa_prediction


# def get_k_peaks(grid_size, k: int, prediction) -> torch.Tensor:
def get_k_peaks(grid_size: int, k: int, prediction: torch.Tensor):
    """
    Retrieves the top-k peaks (angles) from a prediction tensor using peak finding.

    Args:
        grid_size (int): The size of the angle grid (range) in degrees.
        k (int): The number of top peaks (angles) to retrieve.
        prediction (torch.Tensor): The prediction tensor containing the peak values.

    Returns:
        torch.Tensor: A tensor containing the top-k angles in degrees.

    Raises:
        None

    Examples:
        >>> grid_size = 6
        >>> k = 3
        >>> prediction = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])
        >>> get_k_angles(grid_size, k, prediction)
        tensor([ 90., -18.,   54.])

    """
    angels_grid = torch.linspace(-90, 90, grid_size)
    peaks, peaks_data = scipy.signal.find_peaks(
        prediction.detach().numpy().flatten(), prominence=0.05, height=0.01
    )
    peaks = peaks[np.argsort(peaks_data["peak_heights"])[::-1]]
    doa_prediction = angels_grid[peaks]
    while doa_prediction.shape[0] < k:
        doa_prediction = torch.cat(
            (
                doa_prediction,
                torch.Tensor(np.round(np.random.rand(1) * 180, decimals=2) - 90.00),
            ),
            0,
        )

    return doa_prediction[:k]


# def gram_diagonal_overload(Kx: torch.Tensor, eps: float) -> torch.Tensor:
def gram_diagonal_overload(Kx: torch.Tensor, eps: float, batch_size: int):
    """Multiply a matrix Kx with its Hermitian conjecture (gram matrix),
        and adds eps to the diagonal values of the matrix,
        ensuring a Hermitian and PSD (Positive Semi-Definite) matrix.

    Args:
    -----
        Kx (torch.Tensor): Complex matrix with shape [BS, N, N],
            where BS is the batch size and N is the matrix size.
        eps (float): Constant multiplier added to each diagonal element.
        batch_size(int): The number of batches

    Returns:
    --------
        torch.Tensor: Hermitian and PSD matrix with shape [BS, N, N].

    """
    # Insuring Tensor input
    if not isinstance(Kx, torch.Tensor):
        Kx = torch.tensor(Kx)

    Kx_garm = torch.matmul(torch.transpose(Kx.conj(), 1, 2).to("cpu"), Kx.to("cpu")).to(device)
    eps_addition = (eps * torch.diag(torch.ones(Kx_garm.shape[-1]))).to(device)
    Kx_Out = Kx_garm + eps_addition
    return Kx_Out


def calculate_covariance_tensor(sampels: torch.Tensor, method: str = "simple"):
    if method in ["simple", "sample"]:
        if sampels.dim() == 2:
            Rx = torch.cov(sampels)[None, :, :]
        elif sampels.dim() == 3:
            Rx = torch.stack([torch.cov(sampels[i, :, :]) for i in range(sampels.shape[0])])

    elif method == "sps":
        Rx = _spatial_smoothing_covariance(sampels)[None, :, :]
    else:
        raise ValueError(f"calculate_covariance_tensor: method {method} is not recognized for covariance calculation.")

    return Rx


def _spatial_smoothing_covariance(sampels: torch.Tensor):
    """
    Calculates the covariance matrix using spatial smoothing technique.

    Args:
    -----
        X (np.ndarray): Input samples matrix.

    Returns:
    --------
        covariance_mat (np.ndarray): Covariance matrix.
    """

    X = sampels.squeeze()
    N = X.shape[0]
    # Define the sub-arrays size
    sub_array_size = int(N / 2) + 1
    # Define the number of sub-arrays
    number_of_sub_arrays = N - sub_array_size + 1
    # Initialize covariance matrix
    covariance_mat = torch.zeros((sub_array_size, sub_array_size), dtype=torch.complex128)

    for j in range(number_of_sub_arrays):
        # Run over all sub-arrays
        x_sub = X[j: j + sub_array_size, :]
        # Calculate sample covariance matrix for each sub-array
        sub_covariance = torch.cov(x_sub)
        # Aggregate sub-arrays covariances
        covariance_mat += sub_covariance / number_of_sub_arrays
    # Divide overall matrix by the number of sources
    return covariance_mat


def parse_loss_results_for_plotting(loss_results: dict):
    plt_res = {}
    plt_acc = False
    for test, results in loss_results.items():
        for method, loss_ in results.items():
            if plt_res.get(method) is None:
                plt_res[method] = {"Overall": []}
            plt_res[method]["Overall"].append(loss_["Overall"])
            if loss_.get("Accuracy") is not None:
                if "Accuracy" not in plt_res[method].keys():
                    plt_res[method]["Accuracy"] = []
                    plt_acc = True
                plt_res[method]["Accuracy"].append(loss_["Accuracy"])
    return plt_res, plt_acc


def print_loss_results_from_simulation(loss_results: dict):
    """
    Print the loss results from the simulation.
    """
    for test, value_dict in loss_results.items():
        print("#" * 10 + f"{test} TEST RESULTS" + "#" * 10)
        for test_value, results in value_dict.items():
            if test == "SNR":
                print(f"{test} = {test_value} [dB]: ")
            else:
                print(f"{test} = {test_value}: ")
            for method, loss in results.items():
                txt = f"\t{method.upper(): <30}: "
                for key, value in loss.items():
                    if value is not None:
                        if key == "Accuracy":
                            txt += f"{key}: {value * 100:.2f} %|"
                        else:
                            txt += f"{key}: {value:.6e} |"
                print(txt)
            print("\n")
        print("\n")


if __name__ == "__main__":
    # sum_of_diag example
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sum_of_diag(matrix)

    matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sum_of_diags_torch(matrix)

    # find_roots example
    coefficients = [1, -5, 6]
    find_roots(coefficients)

    # get_k_angles example
    grid_size = 6
    k = 3
    prediction = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])
    get_k_angles(grid_size, k, prediction)

    # get_k_peaks example
    grid_size = 6
    k = 3
    prediction = torch.tensor([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])
    get_k_peaks(grid_size, k, prediction)
