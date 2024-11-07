"""Subspace-Net 
Details
----------
Name: system_model.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the SystemModel class for defining the settings of the DoA estimation system model.
"""

# Imports
import numpy as np
from dataclasses import dataclass

import torch
from torch.cuda import device
from src.utils import *


@dataclass
class SystemModelParams:
    """Class for setting parameters of a system model.
    Initialize the SystemModelParams object.

    Parameters:
        None

    Attributes:
        M (int): Number of sources.
        N (int): Number of sensors.
        T (int): Number of observations.
        signal_type (str): Signal type ("NarrowBand" or "Broadband").
        field_type (str): field type ("Far" or "Near")
        freq_values (list): Frequency values for Broadband signal.
        signal_nature (str): Signal nature ("non-coherent" or "coherent").
        snr (float): Signal-to-noise ratio.
        eta (float): Level of deviation from sensor location.
        bias (float): Sensors locations bias deviation.
        sv_noise_var (float): Steering vector added noise variance.

    Returns:
        None
    """

    M = None
    N = None
    T = None
    field_type = "Far"
    signal_type = "NarrowBand"
    freq_values = [0, 500]
    signal_nature = "non-coherent"
    snr = 10
    eta = 0
    bias = 0
    sv_noise_var = 0
    doa_range = 55
    doa_resolution = 1
    max_range_ratio_to_limit = 0.4
    range_resolution = 1

    def set_parameter(self, name: str, value):
        """
        Set the value of the desired system model parameter.

        Args:
            name(str): the name of the SystemModelParams attribute.
            value (int, float, optional): the desired value to assign.

        Returns:
            SystemModelParams: The SystemModelParams object.
        """
        self.__setattr__(name, value)
        return self


class SystemModel(object):
    def __init__(self, system_model_params: SystemModelParams):
        """Class used for defining the settings of the system model.

        Attributes:
        -----------
            field_type (str): Field environment approximation type. Options: "Far", "Near".
            signal_type (str): Signals type. Options: "NarrowBand", "Broadband".
            N (int): Number of sensors.
            M (int): Number of sources.
            freq_values (list, optional): Frequency range for broadband signals. Defaults to None.
            min_freq (dict): Minimal frequency value for different scenarios.
            max_freq (dict): Maximal frequency value for different scenarios.
            f_rng (dict): Frequency range of interest for different scenarios.
            f_sampling (dict): Sampling rate for different scenarios.
            time_axis (dict): Time axis for different scenarios.
            dist (dict): Distance between array elements for different scenarios.
            array (np.ndarray): Array of sensor locations.

        Methods:
        --------
            define_scenario_params(freq_values: list): Defines the signal_type parameters.
            create_array(): Creates the array of sensor locations.
            steering_vec(theta: np.ndarray, f: float = 1, array_form: str = "ULA",
                eta: float = 0, geo_noise_var: float = 0) -> np.ndarray: Computes the steering vector.

        """
        self.array = None
        self.dist_array_elems = None
        self.time_axis = None
        self.f_sampling = None
        self.max_freq = None
        self.min_freq = None
        self.f_rng = None
        self.params = system_model_params
        # Assign signal type parameters
        self.define_scenario_params()
        # Define array indices
        self.create_array()
        # Calculation for the Fraunhofer and Fresnel
        self.fraunhofer, self.fresnel = self.calc_fresnel_fraunhofer_distance()

    def define_scenario_params(self):
        """Defines the signal type parameters based on the specified frequency values."""
        freq_values = self.params.freq_values
        # Define minimal frequency value
        self.min_freq = {"NarrowBand": None, "Broadband": freq_values[0]}
        # Define maximal frequency value
        self.max_freq = {"NarrowBand": None, "Broadband": freq_values[1]}
        # Frequency range of interest
        self.f_rng = {
            "NarrowBand": None,
            "Broadband": np.linspace(
                start=self.min_freq["Broadband"],
                stop=self.max_freq["Broadband"],
                num=self.max_freq["Broadband"] - self.min_freq["Broadband"],
                endpoint=False,
            ),
        }
        # Define sampling rate as twice the maximal frequency
        self.f_sampling = {
            "NarrowBand": None,
            "Broadband": 2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"]),
        }
        # Define time axis
        self.time_axis = {
            "NarrowBand": None,
            "Broadband": np.linspace(
                0, 1, self.f_sampling["Broadband"], endpoint=False
            ),
        }
        # distance between array elements
        self.dist_array_elems = {
            "NarrowBand": 1 / 2,
            "Broadband": 1
                         / (2 * (self.max_freq["Broadband"] - self.min_freq["Broadband"])),
        }

    def create_array(self):
        """create an array of sensors locations, around to origin."""
        self.array = np.linspace(0, self.params.N, self.params.N, endpoint=False)

    def calc_fresnel_fraunhofer_distance(self) -> tuple:
        """
        In the Far and Near field scenrios, those distances are relevant for the distance grid creation.
        wavelength = 1
        spacing = wavelength / 2
        diemeter = (N-1) * spacing
        Fraunhofer  = 2 * diemeter ** 2 / wavelength
        Fresnel = 0.62 * (diemeter ** 3 / wavelength) ** 0.5
        Returns:
            tuple: fraunhofer(float), fresnel(float)
        """
        wavelength = 1
        spacing = wavelength / 2
        diemeter = (self.params.N - 1) * spacing
        fraunhofer = 2 * diemeter ** 2 / wavelength
        fresnel = 0.62 * (diemeter ** 3 / wavelength) ** 0.5
        # fresnel = ((diemeter ** 4) / (8 * wavelength)) ** (1 / 3)

        return fraunhofer, fresnel

    def steering_vec(
            self, angles: np.ndarray, ranges: np.ndarray = None, f_c: float = 1, array_form: str = "ULA",
            nominal: bool = False, generate_search_grid: bool = False) -> torch.Tensor:
        """
        Computes the steering vector based on the specified parameters.
        Args:
            angles: the angles of the sources from origin.
            ranges: the ranges of the sources from origin. In case of Far field, the value is None.
            f_c: the carrier frequency, in case of narrowband, the value is always 1.
            nominal: a flag that suggest if there is any kind of calibration errors.
            array_form: The type of array used.
            generate_search_grid (bool): wether to generate a grid to search on,
             create all combination of angles and ranges, or just create the steering matrix of sources.

        Returns:

        """
        if array_form.startswith("ULA"):
            if self.params.field_type.startswith("Far"):
                return self.steering_vec_far_field(angles, f_c=f_c, nominal=nominal)
            elif self.params.field_type.startswith("Near"):
                return self.steering_vec_near_field(angles, ranges=ranges, f_c=f_c,
                                                    nominal=nominal, generate_search_grid=generate_search_grid)
            else:
                raise Exception(f"SystemModel.field_type:"
                                f" field type of approximation {self.params.field_type} is not defined")
        else:
            raise Exception(f"SystemModel.steering_vec: array form {array_form} is not defined")

    def steering_vec_far_field(
            self, angles: [np.ndarray, torch.Tensor], f_c: float = 1, nominal: bool = False
    ):
        """
        Computes the steering vector based on the specified parameters.

        Args:
        -----
            angles (np.ndarray): Array of angles.
            f (float, optional): Frequency. Defaults to 1.
            nominal (bool): flag for creating sv without array mismatches.

        Returns:
        --------
            np.ndarray: Computed steering vector.

        """
        array = torch.Tensor(self.array[:, None]).to(torch.float64).to(device)
        if isinstance(angles, np.ndarray):
            angles = torch.from_numpy(angles)
        theta = (angles[:, None]).to(torch.float64).to(device)
        dist_array_elems = self.dist_array_elems["NarrowBand"]

        if not nominal:
            dist_array_elems += torch.from_numpy(
                np.random.uniform(low=-1 * self.params.eta, high=self.params.eta, size=self.params.N))
            dist_array_elems = dist_array_elems.unsqueeze(-1).to(device)

        time_delay = torch.einsum("nm, na -> na",
                                  array,
                                  torch.sin(theta).repeat(1, self.params.N).T
                                  * dist_array_elems)

        if not nominal:
            # Calculate additional steering vector noise
            mis_geometry_noise = ((np.sqrt(2) / 2) * np.sqrt(self.params.sv_noise_var)
                                  * (np.random.randn(*time_delay.shape) + 1j * np.random.randn(*time_delay.shape)))
            mis_geometry_noise = torch.from_numpy(mis_geometry_noise).to(device)
        else:
            mis_geometry_noise = 0.0

        steering_matrix = torch.exp(-2 * 1j * torch.pi * time_delay) + mis_geometry_noise

        return steering_matrix

    def steering_vec_near_field(self, angles: [np.ndarray, torch.Tensor], ranges: [np.ndarray, torch.Tensor],
                                f_c: float = 1,
                                nominal: bool = True, generate_search_grid: bool = False) -> torch.Tensor:
        """

        Args:
            angles: the angles of the sources from origin.
            ranges: the ranges of the sources from origin.
            f_c: the carrier frequency, in case of narrowband, the value is always 1.
            nominal: a flag that suggest if there is any kind of calibration errors.
            generate_search_grid (bool): weather to generate a grid to search on,
             create all combination of angles and ranges, or just create the steering matrix of sources.

        Returns:
            np.ndarray: the steering matrix.
        """

        dist_array_elems = self.dist_array_elems["NarrowBand"]
        if not nominal:
            dist_array_elems += torch.from_numpy(
                np.random.uniform(low=-1 * self.params.eta, high=self.params.eta, size=self.params.N)).to(device)
            dist_array_elems = dist_array_elems.unsqueeze(-1)
        if isinstance(dist_array_elems, float):
            dist_array_elems = dist_array_elems * torch.ones(self.params.N, 1, device=device, dtype=torch.float64)

        if isinstance(angles, np.ndarray):
            angles = torch.from_numpy(angles[:, None])
        else:
            if angles.dim() == 1:
                angles = angles[:, None]
        theta = angles.to(torch.float64).to(device)
        if isinstance(ranges, np.ndarray):
            ranges = torch.from_numpy(ranges[:, None])
        else:
            if ranges.dim() == 1:
                ranges = ranges[:, None]
        distances = ranges.to(torch.float64).to(device)

        if theta.shape[0] != distances.shape[0]:
            distances = distances.unsqueeze(0)
        else:
            distances = distances.unsqueeze(-1)

        array = torch.from_numpy(self.array[:, None]).to(torch.float64).to(device)
        array_square = torch.pow(array, 2)

        first_order = torch.einsum("nm, na -> na",
                                   array,
                                   torch.sin(theta).repeat(1, self.params.N).T * dist_array_elems)
        if generate_search_grid:
            first_order = torch.tile(first_order[:, :, None], (1, 1, distances.shape[0]))

        dist_array_elems = dist_array_elems.squeeze(-1)
        if theta.dim() == 2:
            theta = theta.squeeze(-1)
        second_order = -0.5 * torch.div(
            torch.pow(torch.outer(torch.cos(theta), dist_array_elems), 2).unsqueeze(1), distances)
        second_order = torch.einsum("nm, nar -> nar",
                                    array_square,
                                    torch.transpose(second_order, 2, 0)).transpose(1, 2)

        if not generate_search_grid:
            time_delay = first_order + second_order.squeeze(-1)
        else:
            time_delay = first_order + second_order

        if not nominal:
            # Calculate additional steering vector noise
            mis_geometry_noise = ((np.sqrt(2) / 2) * np.sqrt(self.params.sv_noise_var)
                                  * (np.random.randn(*time_delay.shape) + 1j * np.random.randn(*time_delay.shape)))
            mis_geometry_noise = torch.from_numpy(mis_geometry_noise).to(device)
        else:
            mis_geometry_noise = 0.0

        steering_matrix = torch.exp(2 * -1j * torch.pi * time_delay) + mis_geometry_noise
        return steering_matrix
