"""Subspace-Net 
Details
----------
Name: signal_creation.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the Samples class, which inherits from SystemModel class.
This class is used for defining the samples model.
"""

# Imports
from random import sample

import torch

from src.system_model import SystemModel, SystemModelParams
from src.utils import *

class Samples(SystemModel):
    """
    Class used for defining and creating signals and observations.
    Inherits from SystemModel class.

    ...

    Attributes:
    -----------
        doa (np.ndarray): Array of angels (directions) of arrival.

    Methods:
    --------
        set_doa(doa): Sets the direction of arrival (DOA) for the signals.
        samples_creation(noise_mean: float = 0, noise_variance: float = 1, signal_mean: float = 0,
            signal_variance: float = 1): Creates samples based on the specified mode and parameters.
        noise_creation(noise_mean, noise_variance): Creates noise based on the specified mean and variance.
        signal_creation(signal_mean=0, signal_variance=1, SNR=10): Creates signals based on the specified mode and parameters.
    """

    def __init__(self, system_model_params: SystemModelParams):
        """Initializes a Samples object.

        Args:
        -----
        system_model_params (SystemModelParams): an instance of SystemModelParams,
            containing all relevant system model parameters.

        """
        super().__init__(system_model_params)
        self.distances = None

    def set_doa(self, doa, M):
        """
        Sets the direction of arrival (DOA) for the signals.

        Args:
        -----
            doa (np.ndarray): Array containing the DOA values.

        """

        def create_doa_with_gap(gap: float, M):
            """Create angles with a value gap.

            Args:
            -----
                gap (float): Minimal gap value.

            Returns:
            --------
                np.ndarray: DOA array.

            """
            # LEGACY CODE
            # while True:
            #     # DOA = np.round(np.random.rand(M) * 180, decimals=2) - 90
            #     DOA = np.random.randint(-55, 55, M)
            #     DOA.sort()
            #     diff_angles = np.array(
            #         [np.abs(DOA[i + 1] - DOA[i]) for i in range(M - 1)]
            #     )
            #     if (np.sum(diff_angles > gap) == M - 1) and (
            #         np.sum(diff_angles < (180 - gap)) == M - 1
            #     ):
            #         break

            # based on https://stackoverflow.com/questions/51918580/python-random-list-of-numbers-in-a-range-keeping-with-a-minimum-distance
            doa_range = self.params.doa_range
            doa_resolution = self.params.doa_resolution
            if doa_resolution <= 0:
                raise ValueError("DOA resolution must be positive.")
            if M <= 0:
                raise ValueError("M (number of elements) must be positive.")
            if gap <= 0:
                raise ValueError("Gap must be positive.")

            # Compute the range of possible DOA values
            # Ensure the sampled DOAs do not exceed [-doa_range, +doa_range]
            max_offset = (gap - 1) * (M - 1)
            effective_range = 2 * doa_range - max_offset
            if effective_range <= 0:
                raise ValueError(f"Invalid effective range: {effective_range}. Check your parameters.")

            # Define the valid range for sampling
            if doa_resolution >= 1:
                valid_range = range(0, effective_range, doa_resolution)
                sampled_values = sorted(sample(valid_range, M))
            else:
                step_count = int(effective_range // doa_resolution)
                valid_range = range(step_count)
                sampled_values = sorted(sample(valid_range, M))
                sampled_values = [x * doa_resolution for x in sampled_values]

            # Compute DOAs
            DOA = [(gap - 1) * i + x - doa_range for i, x in enumerate(sampled_values)]

            # Ensure all DOAs fall naturally within the valid range
            if any(d < -doa_range or d > doa_range for d in DOA):
                raise ValueError("Computed DOAs exceed the valid range. Check your logic.")

            # Round results to 3 decimal places
            DOA = np.round(DOA, 3)

            return DOA

        if doa == None:
            # Generate angels with gap greater than 0.2 rad (nominal case)
            self.doa = np.array(create_doa_with_gap(gap=15, M=M)) * D2R
        else:
            # Generate
            self.doa = np.deg2rad(doa)

    def set_range(self, distance: list | np.ndarray, M) -> np.ndarray:
        """

        Args:
            distance:

        Returns:

        """

        def choose_distances(M, min_val: float, max_val: int, distance_resolution: float = 1.0) -> np.ndarray:
            """
            Choose distances for the sources.

            Args:
                M (int): Number of sources.
                min_val (float): Minimal value of the distances.
                max_val (int): Maximal value of the distances.
                distance_resolution (float, optional): Resolution of the distances. Defaults to 1.0.

            """
            distances_options = np.arange(min_val, max_val, distance_resolution)
            distances = np.random.choice(distances_options, M, replace=True)
            return np.round(distances, 3)

        if distance is None:
            self.distances = choose_distances(M, min_val=np.ceil(self.fresnel),
                                              max_val=np.floor(self.fraunhofer * self.params.max_range_ratio_to_limit),
                                              distance_resolution=self.params.range_resolution)
        else:
            self.distances = np.array(distance)

    def samples_creation(
        self,
        noise_mean: float = 0,
        noise_variance: float = 1,
        signal_mean: float = 0,
        signal_variance: float = 1,
        source_number: int = None,
    ):
        """Creates samples based on the specified mode and parameters.

        Args:
        -----
            noise_mean (float, optional): Mean of the noise. Defaults to 0.
            noise_variance (float, optional): Variance of the noise. Defaults to 1.
            signal_mean (float, optional): Mean of the signal. Defaults to 0.
            signal_variance (float, optional): Variance of the signal. Defaults to 1.

        Returns:
        --------
            tuple: Tuple containing the created samples, signal, steering vectors, and noise.

        Raises:
        -------
            Exception: If the signal_type is not defined.

        """
        # Generate signal matrix
        signal = self.signal_creation(signal_mean, signal_variance, source_number=source_number)
        signal = torch.from_numpy(signal)
        # Generate noise matrix
        noise = self.noise_creation(noise_mean, noise_variance)
        noise = torch.from_numpy(noise)
        if self.params.field_type.startswith("far"):
            A = self.steering_vec(self.doa, f_c=self.f_rng[self.params.signal_type])
            if self.params.signal_type.startswith("broadband"):
                samples = torch.einsum("nmk, mk -> nk", A, signal)
            else:
                samples = (A @ signal) + noise
        elif self.params.field_type.startswith("near"):
            A = self.steering_vec(angles=self.doa, ranges=self.distances, nominal=False, generate_search_grid=False,
                                  f_c=self.f_rng[self.params.signal_type])
            if self.params.signal_type.startswith("broadband"):
                samples = torch.einsum("nmk, mk -> nk", A, signal) + noise
            else:
                samples = (A @ signal) + noise
        else:
            raise Exception(f"Samples.params.field_type: Field type {self.params.field_type} is not defined")
        if self.params.signal_type.startswith("broadband"):
            # transform the signal to the time domain in broadband settings
            samples = torch.fft.ifft(samples, n=self.params.T, dim=1) + noise
        return samples, signal, A, noise

    def noise_creation(self, noise_mean, noise_variance):
        """Creates noise based on the specified mean and variance.

        Args:
        -----
            noise_mean (float): Mean of the noise.
            noise_variance (float): Variance of the noise.

        Returns:
        --------
            np.ndarray: Generated noise.

        """
        # for NarrowBand signal_type Noise represented in the time domain
        noise =  (
            np.sqrt(noise_variance)
            * (np.sqrt(2) / 2)
            * (
                np.random.randn(self.params.N, self.params.T)
                + 1j * np.random.randn(self.params.N, self.params.T)
            )
            + noise_mean
        )
        # for Broadband signal_type Noise represented in the frequency domain
        # if self.params.signal_type.startswith("broadband"):
            # noise  = np.fft.fft(noise,axis=1)
            # noise  = np.fft.fft(noise, n=self.params.number_subcarriers,axis=1)
        return noise

    def signal_creation(self, signal_mean: float = 0, signal_variance: float = 1, source_number: int = None):
        """
        Creates signals based on the specified signal nature and parameters.

        Args:
        -----
            signal_mean (float, optional): Mean of the signal. Defaults to 0.
            signal_variance (float, optional): Variance of the signal. Defaults to 1.

        Returns:
        --------
            np.ndarray: Created signals.

        Raises:
        -------
            Exception: If the signal type is not defined.
            Exception: If the signal nature is not defined.
        """
        M = source_number
        if self.params.snr is None:
            snr = np.random.uniform(-10, 10)
        else:
            snr = self.params.snr
        amplitude = 10 ** (snr / 10)
        # NarrowBand signal creation
        if self.params.signal_type == "narrowband":
            if self.params.signal_nature == "non-coherent":
                # create M non-coherent signals
                return (
                    amplitude
                    * (np.sqrt(2) / 2)
                    * np.sqrt(signal_variance)
                    * (
                        np.random.randn(M, self.params.T)
                        + 1j * np.random.randn(M, self.params.T)
                    )
                    + signal_mean
                )

            elif self.params.signal_nature == "coherent":
                # Coherent signals: same amplitude and phase for all signals
                sig = (
                    amplitude
                    * (np.sqrt(2) / 2)
                    * np.sqrt(signal_variance)
                    * (
                        np.random.randn(1, self.params.T)
                        + 1j * np.random.randn(1, self.params.T)
                    )
                    + signal_mean
                )
                return np.repeat(sig, M, axis=0)

        # OFDM Broadband signal creation
        elif self.params.signal_type.startswith("broadband"):
            if self.params.signal_nature == "non-coherent":
                # create M non-coherent broadband signals
                # each source, has K subcarriers which are generated as a complex number.
                # the symbols are of size M x K
                symbols = (amplitude
                           * (
                                   np.random.randn(M, self.params.number_subcarriers)
                                   + 1j * np.random.randn(M, self.params.number_subcarriers)))
            elif self.params.signal_nature == "coherent":
                # Coherent signals: same amplitude and phase for all signals
                symbols = (amplitude
                            * (
                                      np.random.randn(1, self.params.number_subcarriers)
                                      + 1j * np.random.randn(1, self.params.number_subcarriers)))
                symbols = np.repeat(symbols, M, axis=0)
            # each symbol is multiplied by the phase shift of the subcarrier
            # the phase shift is a function of the subcarrier frequency and the time
            # the phase shift is of size K x T
            phase_shift = np.exp(
                1j * 2 * np.pi
                * (np.arange(self.params.number_subcarriers)[:, None] - self.params.number_subcarriers // 2) @ self.time_axis["broadband"][:, None].T
                * self.params.signal_bandwidth
                / (self.params.number_subcarriers)
            )
            # the signal is the product of the symbols and the phase shift normalized by the number of subcarriers
            # the signal is of size M x T
            signal = symbols @ phase_shift / self.params.number_subcarriers
            # the creation of this signal is in the time domain, if we want to multiply it by the steering vector
            # we need to transform it to the frequency domain
            signal_frq = np.fft.fft(signal, n=self.params.number_subcarriers, axis=1)
            # signal_frq = np.fft.fft(signal, axis=1)
            return signal_frq

        else:
            raise Exception(f"signal type {self.params.signal_type} is not defined")
