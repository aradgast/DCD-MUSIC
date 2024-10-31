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
import numpy as np
from random import sample
from src.system_model import SystemModel, SystemModelParams
from src.utils import D2R

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
            range_size = 2 * self.params.doa_range - (gap - 1) * (M - 1)
            # assert(range_size<0) , Warning(range_size<10)
            doa_resolution = self.params.doa_resolution
            if self.params.doa_resolution >= 1:
                DOA = [(gap - 1) * i + x - self.params.doa_range for i, x in
                       enumerate(sorted(sample(range(0, range_size, doa_resolution), M)))]
            else:
                DOA = [((gap - 1) * i + x - self.params.doa_range)*doa_resolution for i, x in enumerate(sorted(sample(range(int(range_size//doa_resolution)), M)))]
                DOA = np.round(DOA, 3)
            return DOA

        if doa == None:
            # Generate angels with gap greater than 0.2 rad (nominal case)
            self.doa = np.array(create_doa_with_gap(gap=15, M=M)) * D2R
        else:
            # Generate
            self.doa = np.array(doa) * D2R

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
            self.distances = distance

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
        # Generate noise matrix
        noise = self.noise_creation(noise_mean, noise_variance)
        # Generate Narrowband samples
        if self.params.signal_type.startswith("NarrowBand"):
            if self.params.field_type.startswith("Far"):
                A = np.array([self.steering_vec(theta) for theta in self.doa]).T
                samples = (A @ signal) + noise
            elif self.params.field_type.startswith("Near"):
                A = self.steering_vec(theta=self.doa, distance=self.distances, nominal=False, generate_search_grid=False)
                samples = (A @ signal) + noise
            else:
                raise Exception(f"Samples.params.field_type: Field type {self.params.field_type} is not defined")
            return samples, signal, A, noise
        # Generate Broadband samples
        elif self.params.signal_type.startswith("Broadband"):
            samples = []
            SV = []

            for idx in range(self.f_sampling["Broadband"]):
                # mapping from index i to frequency f
                if idx > int(self.f_sampling["Broadband"]) // 2:
                    f = -int(self.f_sampling["Broadband"]) + idx
                else:
                    f = idx
                A = np.array([self.steering_vec(theta, f) for theta in self.doa]).T
                samples.append((A @ signal[:, idx]) + noise[:, idx])
                SV.append(A)
            samples = np.array(samples)
            SV = np.array(SV)
            samples_time_domain = np.fft.ifft(samples.T, axis=1)[:, : self.params.T]
            return samples_time_domain, signal, SV, noise
        else:
            raise Exception(
                f"Samples.samples_creation: signal type {self.params.signal_type} is not defined"
            )

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
        if self.params.signal_type.startswith("NarrowBand"):
            return (
                np.sqrt(noise_variance)
                * (np.sqrt(2) / 2)
                * (
                    np.random.randn(self.params.N, self.params.T)
                    + 1j * np.random.randn(self.params.N, self.params.T)
                )
                + noise_mean
            )
        # for Broadband signal_type Noise represented in the frequency domain
        elif self.params.signal_type.startswith("Broadband"):
            noise = (
                np.sqrt(noise_variance)
                * (np.sqrt(2) / 2)
                * (
                    np.random.randn(self.params.N, len(self.time_axis["Broadband"]))
                    + 1j
                    * np.random.randn(self.params.N, len(self.time_axis["Broadband"]))
                )
                + noise_mean
            )
            return np.fft.fft(noise)
        else:
            raise Exception(
                f"Samples.noise_creation: signal type {self.params.signal_type} is not defined"
            )

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
        amplitude = 10 ** (self.params.snr / 10)
        # NarrowBand signal creation
        if self.params.signal_type == "NarrowBand":
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
        elif self.params.signal_type.startswith("Broadband"):
            num_sub_carriers = self.max_freq[
                "Broadband"
            ]  # number of subcarriers per signal
            if self.params.signal_nature == "non-coherent":
                # create M non-coherent signals
                signal = np.zeros(
                    (self.params.M, len(self.time_axis["Broadband"]))
                ) + 1j * np.zeros((self.params.M, len(self.time_axis["Broadband"])))
                for i in range(self.params.M):
                    for j in range(num_sub_carriers):
                        sig_amp = (
                            amplitude
                            * (np.sqrt(2) / 2)
                            * (np.random.randn(1) + 1j * np.random.randn(1))
                        )
                        signal[i] += sig_amp * np.exp(
                            1j
                            * 2
                            * np.pi
                            * j
                            * len(self.f_rng["Broadband"])
                            * self.time_axis["Broadband"]
                            / num_sub_carriers
                        )
                    signal[i] *= 1 / num_sub_carriers
                return np.fft.fft(signal)
            # Coherent signals: same amplitude and phase for all signals
            elif self.params.signal_nature == "coherent":
                signal = np.zeros(
                    (1, len(self.time_axis["Broadband"]))
                ) + 1j * np.zeros((1, len(self.time_axis["Broadband"])))
                for j in range(num_sub_carriers):
                    sig_amp = (
                        amplitude
                        * (np.sqrt(2) / 2)
                        * (np.random.randn(1) + 1j * np.random.randn(1))
                    )
                    signal += sig_amp * np.exp(
                        1j
                        * 2
                        * np.pi
                        * j
                        * len(self.f_rng["Broadband"])
                        * self.time_axis["Broadband"]
                        / num_sub_carriers
                    )
                signal *= 1 / num_sub_carriers
                return np.tile(np.fft.fft(signal), (self.params.M, 1))
            else:
                raise Exception(
                    f"signal nature {self.params.signal_nature} is not defined"
                )

        else:
            raise Exception(f"signal type {self.params.signal_type} is not defined")
