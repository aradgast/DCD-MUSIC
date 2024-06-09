"""Subspace-Net 
Details
----------
    Name: data_handler.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 03/06/23

Purpose:
--------
    This scripts handle the creation and processing of synthetic datasets
    based on specified parameters and model types.
    It includes functions for generating datasets, reading data from files,
    computing autocorrelation matrices, and creating covariance tensors.

Attributes:
-----------
    Samples (from src.signal_creation): A class for creating samples used in dataset generation.

    The script defines the following functions:
    * create_dataset: Generates a synthetic dataset based on the specified parameters and model type.
    * read_data: Reads data from a file specified by the given path.
    * autocorrelation_matrix: Computes the autocorrelation matrix for a given lag of the input samples.
    * create_autocorrelation_tensor: Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.
    * create_cov_tensor: Creates a 3D tensor containing the real part,
        imaginary part, and phase component of the covariance matrix.
    * set_dataset_filename: Returns the generic suffix of the datasets filename.

"""

# Imports
import itertools

import torch
from tqdm import tqdm
from src.signal_creation import Samples
from pathlib import Path
from src.system_model import SystemModelParams
from src.utils import *
from torch.utils.data import Dataset, Sampler


def create_dataset(
        system_model_params: SystemModelParams,
        samples_size: float,
        model_type: str,
        save_datasets: bool = False,
        datasets_path: Path = None,
        true_doa: list = None,
        true_range: list = None,
        phase: str = None,
):
    """
    Generates a synthetic dataset based on the specified parameters and model type.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams
        samples_size (float): The size of the dataset.
        model_type (str): The type of the model.
        save_datasets (bool, optional): Specifies whether to save the dataset. Defaults to False.
        datasets_path (Path, optional): The path for saving the dataset. Defaults to None.
        true_doa (list, optional): Predefined angles. Defaults to None.
        true_range (list, optional): Predefined ranges. Defaults to None.
        phase (str, optional): The phase of the dataset (test or training phase for CNN model). Defaults to None.

    Returns:
    --------
        tuple: A tuple containing the desired dataset comprised of (X-samples, Y-labels).

    """
    time_series = []
    labels = []
    sources_num = []
    samples_model = Samples(system_model_params)
    # Generate permutations for CNN model training dataset
    if model_type.startswith("DeepCNN") and phase.startswith("train"):
        doa_permutations = []
        angles_grid = np.linspace(start=-90, stop=90, num=361)
        for comb in itertools.combinations(angles_grid, system_model_params.M):
            doa_permutations.append(list(comb))

    if model_type.startswith("DeepCNN") and phase.startswith("train"):
        for i, doa in tqdm(enumerate(doa_permutations)):
            # Samples model creation
            samples_model.set_doa(doa)

            # Observations matrix creation
            X = torch.tensor(
                samples_model.samples_creation(
                    noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
                )[0],
                dtype=torch.complex64,
            )
            # Ground-truth creation
            Y = torch.zeros_like(torch.tensor(angles_grid))
            for angle in doa:
                Y[list(angles_grid).index(angle)] = 1
            time_series.append(X)
            labels.append(Y)
    else:
        for i in tqdm(range(samples_size)):
            if system_model_params.M is None:
                M = np.random.randint(4, np.min((5, system_model_params.N-1)))
            else:
                M = system_model_params.M
            # Samples model creation
            samples_model.set_doa(true_doa, M)
            if system_model_params.field_type.lower().endswith("near"):
                samples_model.set_range(true_range, M)
            # Observations matrix creation
            X = torch.tensor(
                samples_model.samples_creation(
                    noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1, source_number=M
                )[0],
                dtype=torch.complex128,
            )
            # Ground-truth creation
            Y = torch.tensor(samples_model.doa, dtype=torch.float32)
            if system_model_params.field_type.endswith("Near"):
                Y1 = torch.tensor(samples_model.distances, dtype=torch.float32)
                Y = torch.cat((Y, Y1), dim=0)
            time_series.append(X)
            labels.append(Y)
            sources_num.append(M)

    generic_dataset = TimeSeriesDataset(time_series, labels, sources_num)
    if save_datasets:
        generic_dataset_filename = f"Generic_DataSet" + set_dataset_filename(
            system_model_params, samples_size
        )
        samples_model_filename = f"samples_model" + set_dataset_filename(
            system_model_params, samples_size
        )

        torch.save(obj=generic_dataset, f=datasets_path / phase / generic_dataset_filename)
        if phase.startswith("test"):
            torch.save(obj=samples_model, f=datasets_path / phase / samples_model_filename)

    return generic_dataset, samples_model


# def read_data(Data_path: str) -> torch.Tensor:
def read_data(path: str):
    """
    Reads data from a file specified by the given path.

    Args:
    -----
        path (str): The path to the data file.

    Returns:
    --------
        torch.Tensor: The loaded data.

    Raises:
    -------
        None

    Examples:
    ---------
        >>> path = "data.pt"
        >>> read_data(path)

    """
    assert isinstance(path, (str, Path))
    data = torch.load(path)
    return data


# def autocorrelation_matrix(X: torch.Tensor, lag: int) -> torch.Tensor:
def autocorrelation_matrix(X: torch.Tensor, lag: int):
    """
    Computes the autocorrelation matrix for a given lag of the input samples.

    Args:
    -----
        X (torch.Tensor): Samples matrix input with shape [N, T].
        lag (int): The requested delay of the autocorrelation calculation.

    Returns:
    --------
        torch.Tensor: The autocorrelation matrix for the given lag.

    """
    meu = torch.mean(X, dim=1).reshape(-1, 1).to("cpu")
    center_x = X.to("cpu") - meu
    x1 = center_x[:, :center_x.shape[1] - lag].to("cpu").to(torch.complex128)
    x2 = torch.conj(center_x[:, lag:]).T.to("cpu").to(torch.complex128)
    Rx_lag = torch.matmul(x1, x2) / (center_x.shape[-1] - lag - 1)
    Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), 0).to("cpu")
    return Rx_lag


# def create_autocorrelation_tensor(X: torch.Tensor, tau: int) -> torch.Tensor:
def create_autocorrelation_tensor(X: torch.Tensor, tau: int):
    """
    Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (BS, N, T).
        tau (int): Maximal time difference for the autocorrelation tensor.

    Returns:
    --------
        torch.Tensor: Tensor containing all the autocorrelation matrices,
                    with size (Batch size, tau, 2N, N).

    Raises:
    -------
        None

    """
    Rx_tau = []
    for i in range(tau):
        Rx_tau.append(autocorrelation_matrix(X, lag=i))
    Rx_autocorr = torch.stack(Rx_tau, dim=0)
    return Rx_autocorr


# def create_cov_tensor(X: torch.Tensor) -> torch.Tensor:
def create_cov_tensor(X: torch.Tensor):
    """
    Creates a 3D tensor of size (NxNx3) containing the real part, imaginary part, and phase component of the covariance matrix.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (N, T).

    Returns:
    --------
        Rx_tensor (torch.Tensor): Tensor containing the auto-correlation matrices, with size (Batch size, N, N, 3).

    Raises:
    -------
        None

    """
    Rx = torch.cov(X)
    Rx_tensor = torch.stack((torch.real(Rx), torch.imag(Rx), torch.angle(Rx)), 2)
    return Rx_tensor


def load_datasets(
        system_model_params: SystemModelParams,
        model_type: str,
        samples_size: float,
        datasets_path: Path,
        train_test_ratio: float,
        is_training: bool = False,
):
    """
    Load different datasets based on the specified parameters and phase.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        model_type (str): The type of the model.
        samples_size (float): The size of the overall dataset.
        datasets_path (Path): The path to the datasets.
        train_test_ratio (float): The ration between train and test datasets.
        is_training (bool): Specifies whether to load the training dataset.

    Returns:
    --------
        List: A list containing the loaded datasets.

    """
    datasets = []
    # Define test set size
    test_samples_size = int(train_test_ratio * samples_size)
    # Generate datasets filenames
    # model_dataset_filename = f"{model_type}_DataSet" + set_dataset_filename(
    #     system_model_params, test_samples_size
    # )
    generic_dataset_filename = f"Generic_DataSet" + set_dataset_filename(
        system_model_params, test_samples_size
    )
    samples_model_filename = f"samples_model" + set_dataset_filename(
        system_model_params, test_samples_size
    )

    # Whether to load the training dataset
    if is_training:
        # Load training dataset
        try:
            model_trainingset_filename = f"Generic_DataSet" + set_dataset_filename(
                system_model_params, samples_size
            )
            train_dataset = read_data(
                datasets_path / "train" / model_trainingset_filename
            )
            datasets.append(train_dataset)
        except:
            raise Exception("load_datasets: Training dataset doesn't exist")
    # Load test dataset
    # try:
    #     test_dataset = read_data(datasets_path / "test" / model_dataset_filename)
    #     datasets.append(test_dataset)
    # except:
    #     raise Exception("load_datasets: Test dataset doesn't exist")
    # Load generic test dataset
    try:
        generic_test_dataset = read_data(
            datasets_path / "test" / generic_dataset_filename
        )
        datasets.append(generic_test_dataset)
    except:
        raise Exception("load_datasets: Generic test dataset doesn't exist")
    # Load samples models
    try:
        samples_model = read_data(datasets_path / "test" / samples_model_filename)
        datasets.append(samples_model)
    except:
        raise Exception("load_datasets: Samples model dataset doesn't exist")
    return datasets


def set_dataset_filename(system_model_params: SystemModelParams, samples_size: float):
    """Returns the generic suffix of the datasets filename.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        samples_size (float): The size of the overall dataset.

    Returns:
    --------
        str: Suffix dataset filename
    """
    if system_model_params.M is None:
        M = "rand"
    else:
        M = system_model_params.M
    suffix_filename = (
            f"_{system_model_params.field_type}_field_"
            f"{system_model_params.signal_type}_"
            + f"{system_model_params.signal_nature}_{samples_size}_M={M}_"
            + f"N={system_model_params.N}_T={system_model_params.T}_SNR={system_model_params.snr}_"
            + f"eta={system_model_params.eta}_sv_noise_var{system_model_params.sv_noise_var}_"
            + f"bias={system_model_params.bias}_"
            + ".h5"
    )
    return suffix_filename


class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y, M):
        self.X = X
        self.Y = Y
        self.M = M


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.Y[idx]


def collate_fn(batch):
    time_series, source_num, labels = zip(*batch)

    # Find the maximum length in this batch
    max_length = max([lb.size(0) for lb in labels])

    # Pad labels and create masks
    padded_labels = torch.zeros(len(batch), max_length, dtype=torch.float32)
    masks = torch.zeros(len(batch), max_length, dtype=torch.float32)

    for i, lb in enumerate(labels):
        length = lb.size(0)
        if source_num[i] != length:
            # this is a near field dataset
            angles, distances = torch.split(lb, source_num[i], dim=0)
            lb = torch.cat((angles, torch.zeros(max_length // 2 - source_num[i], dtype=torch.float32)))
            lb = torch.cat((lb, distances, torch.zeros(max_length // 2 - source_num[i], dtype=torch.float32)))
            mask = torch.zeros(max_length, dtype=torch.float32)
            mask[: length // 2] = 1
            mask[max_length // 2: max_length // 2 + length // 2] = 1
        else:
            lb = torch.cat((lb, torch.zeros(max_length - length, dtype=torch.long)))
            mask = torch.zeros(max_length, dtype=torch.float32)
            mask[:length] = 1
        padded_labels[i] = lb
        masks[i] = mask

    # Stack labels
    time_series = torch.stack(time_series).squeeze()
    sources_num = torch.tensor(source_num)


    return time_series, sources_num, padded_labels, masks


class SameLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def _create_batches(self):
        length_to_indices = {}
        for idx, (_, source_num, _) in enumerate(self.data_source):
            if source_num not in length_to_indices:
                length_to_indices[source_num] = []
            length_to_indices[source_num].append(idx)

        batches = []
        for indices in length_to_indices.values():
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i + self.batch_size])

        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
