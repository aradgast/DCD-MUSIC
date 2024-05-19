"""Subspace-Net 
Details
----------
Name: models.py
Authors: Dor Haim Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the tested NN-models and the model-based DL models, which used for simulation.
The implemented models:
    * DeepRootMUSIC: model-based deep learning algorithm as described in:
        [1] D. H. Shmuel, J. P. Merkofer, G. Revach, R. J. G. van Sloun and N. Shlezinger,
        "Deep Root Music Algorithm for Data-Driven Doa Estimation," ICASSP 2023 - 
        2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
        Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10096504.
        
    * SubspaceNet: model-based deep learning algorithm as described in:
        [2] "SubspaceNet: Deep Learning-Aided Subspace methods for DoA Estimation".
    
    * DA-MUSIC: Deep Augmented MUSIC model-based deep learning algorithm as described in
        [3] J. P. Merkofer, G. Revach, N. Shlezinger, and R. J. van Sloun, “Deep
        augmented MUSIC algorithm for data-driven DoA estimation,” in IEEE
        International Conference on Acoustics, Speech and Signal Processing
        (ICASSP), 2022, pp. 3598-3602."
        
    * DeepCNN: Deep learning algorithm as described in:
        [4] G. K. Papageorgiou, M. Sellathurai, and Y. C. Eldar, “Deep networks
        for direction-of-arrival estimation in low SNR,” IEEE Trans. Signal
        Process., vol. 69, pp. 3714-3729, 2021.

Functions:
----------
This script also includes the implementation of Root-MUSIC algorithm, as it is written using Pytorch library,
for the usage of src.models: SubspaceNet implementation.
"""
import os

# Imports
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import scipy as sc
from scipy.spatial.distance import cdist
import warnings
from src.utils import gram_diagonal_overload, device
from src.utils import sum_of_diags_torch, find_roots_torch
import time
import matplotlib.pyplot as plt
from torch.autograd import Variable
from src.system_model import SystemModel, SystemModelParams

warnings.simplefilter("ignore")


# Constants

class ModelGenerator(object):
    """
    Generates an instance of the desired model, according to model configuration parameters.
    Attributes:
            model_type(str): The network type
            field_type(str) : The field type region ("Far" or "Near")
            diff_method(str): The differentiable method to use("root_music", "esprit", "music")
            tau(int): The number of input features to the network(time shift for auto-correlation)
            model : An instance of the model
            system_model(SystemModel): An instance of SystemModel
    """

    def __init__(self, system_model: SystemModel = None):
        """
        Initialize ModelParams object.
        """
        self.model_type: str = str(None)
        self.field_type: str = str(None)
        self.diff_method: str = str(None)
        self.tau: int = None
        self.model = None
        self.system_model = system_model

    def set_field_type(self, field_type: str = None):
        """

        Args:
            field_type:

        Returns:

        """
        if field_type.lower() == "near":
            self.field_type = "Near"
        elif field_type.lower() == "far":
            self.field_type = "Far"
        else:
            raise Exception(f"ModelParams.set_field_type:"
                            f" unrecognized {field_type} as field type, available options are Near or Far.")
        return self

    def set_tau(self, tau: int = None):
        """
        Set the value of tau parameter for SubspaceNet model.

        Parameters:
            tau (int): The number of lags.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If tau parameter is not provided for SubspaceNet model.
        """
        if self.model_type.endswith("SubspaceNet"):
            if not isinstance(tau, int):
                raise ValueError(
                    "ModelParams.set_tau: tau parameter must be provided for SubspaceNet model"
                )
            self.tau = tau
        return self

    def set_diff_method(self, diff_method: str = "root_music"):
        """
        Set the differentiation method for SubspaceNet model.

        Parameters:
            diff_method (str): The differantiable subspace method ("esprit" or "root_music").

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If the diff_method is not defined for SubspaceNet model.
        """
        if self.model_type.startswith("SubspaceNet"):
            if self.field_type.startswith("Far"):
                if diff_method not in ["esprit", "root_music"]:
                    raise ValueError(
                        f"ModelParams.set_diff_method:"
                        f" {diff_method} is not defined for SubspaceNet model on {self.field_type} scenario")
                self.diff_method = diff_method
            elif self.field_type.startswith("Near"):
                if diff_method not in ["music_2D"]:
                    raise ValueError(f"ModelParams.set_diff_method: "
                                     f"{diff_method} is not defined for SubspaceNet model on {self.field_type} scenario")
                self.diff_method = diff_method
        elif self.model_type.startswith("CascadedSubspaceNet"):
            if diff_method not in ["music_1D"]:
                raise ValueError(f"ModelParams.set_diff_method: "
                                 f"{diff_method} is not defined for CascadedSubspaceNet model")
            self.diff_method = diff_method
        return self

    def set_model_type(self, model_type: str):
        """
        Set the model type.

        Parameters:
            model_type (str): The model type.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If model type is not provided.
        """
        if not isinstance(model_type, str):
            raise ValueError(
                "ModelParams.set_model_type: model type has not been provided"
            )
        self.model_type = model_type
        return self

    def get_model_filename(self, system_model_params: SystemModelParams):
        """

        Parameters
        ----------
        system_model_params

        Returns
        -------
        file name to the wieghts of a network.
        different from get_simulation_filename by not considering parameters that are not relevant to the network itself.
        """
        return (
                f"SubspaceNet_"
                + f"N={system_model_params.N}_"
                + f"tau={self.tau}_"
                + f"{system_model_params.signal_type}_"
                + f"diff_method=esprit_"
                + f"{system_model_params.field_type}_field_"
                + f"{system_model_params.signal_nature}"
        )

        # return (
        #     f"SubspaceNet_M={system_model_params.M}_"
        #     + f"T={system_model_params.T}_SNR_{system_model_params.snr}_"
        #     + f"tau={self.tau}_{system_model_params.signal_type}_"
        #     + f"diff_method=esprit_"
        #     + f"{system_model_params.field_type}_field_"
        #     + f"{system_model_params.signal_nature}_eta={system_model_params.eta}_"
        #     + f"bias={system_model_params.bias}_"
        #     + f"sv_noise={system_model_params.sv_noise_var}"
        # )

    def set_model(self, system_model_params):
        """
        Set the model based on the model type and system model parameters.

        Parameters:
            system_model_params (SystemModelParams): The system model parameters.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            Exception: If the model type is not defined.
        """
        if self.model_type.startswith("DA-MUSIC"):
            self.model = DeepAugmentedMUSIC(
                N=system_model_params.N,
                T=system_model_params.T,
                M=system_model_params.M,
            )
        elif self.model_type.startswith("DeepCNN"):
            self.model = DeepCNN(N=system_model_params.N, grid_size=361)
        elif self.model_type.startswith("SubspaceNet"):
            self.model = SubspaceNet(tau=self.tau,
                                     N=system_model_params.N,
                                     diff_method=self.diff_method,
                                     system_model=self.system_model,
                                     field_type=self.field_type)
        elif self.model_type.startswith("CascadedSubspaceNet"):
            self.model = CascadedSubspaceNet(tau=self.tau,
                                             N=system_model_params.N,
                                             system_model=self.system_model,
                                             state_path=self.get_model_filename(system_model_params))
        else:
            raise Exception(f"ModelGenerator.set_model: Model type {self.model_type} is not defined")

        return self


class DeepRootMUSIC(nn.Module):
    """DeepRootMUSIC is model-based deep learning model for DOA estimation problem.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        conv1 (nn.Conv2d): Convolution layer 1.
        conv2 (nn.Conv2d): Convolution layer 2.
        conv3 (nn.Conv2d): Convolution layer 3.
        deconv1 (nn.ConvTranspose2d): De-convolution layer 1.
        deconv2 (nn.ConvTranspose2d): De-convolution layer 2.
        deconv3 (nn.ConvTranspose2d): De-convolution layer 3.
        DropOut (nn.Dropout): Dropout layer.
        LeakyReLU (nn.LeakyReLU): Leaky reLu activation function, with activation_value.

    Methods:
    --------
        anti_rectifier(X): Applies the anti-rectifier operation to the input tensor.
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.
        gram_diagonal_overload(Kx, eps): Applies Gram operation and diagonal loading to a complex matrix.

    """

    def __init__(self, tau: int, activation_value: float):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            activation_value (float): Value for the activation function.

        """
        super(DeepRootMUSIC, self).__init__()
        self.tau = tau
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=2)
        self.LeakyReLU = nn.LeakyReLU(activation_value)
        self.DropOut = nn.Dropout(0.2)

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of the DeepRootMUSIC.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.LeakyReLU(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.LeakyReLU(x)
        # DCNN block #1
        x = self.deconv1(x)
        x = self.LeakyReLU(x)
        # DCNN block #2
        x = self.deconv2(x)
        x = self.LeakyReLU(x)
        # DCNN block #3
        x = self.DropOut(x)
        Rx = self.deconv3(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N:, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(Kx_tag, eps=1)  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to Root-MUSIC algorithm
        doa_prediction, doa_all_predictions, roots = root_music(
            Rz, self.M, self.batch_size
        )
        return doa_prediction, doa_all_predictions, roots, Rz


# TODO: inherit SubspaceNet from DeepRootMUSIC
class SubspaceNet(nn.Module):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        N (int):
        diff_method (str):
        field_type (str):
        conv1 (nn.Conv2d): Convolution layer 1.
        conv2 (nn.Conv2d): Convolution layer 2.
        conv3 (nn.Conv2d): Convolution layer 3.
        deconv1 (nn.ConvTranspose2d): De-convolution layer 1.
        deconv2 (nn.ConvTranspose2d): De-convolution layer 2.
        deconv3 (nn.ConvTranspose2d): De-convolution layer 3.
        DropOut (nn.Dropout): Dropout layer.
        ReLU (nn.ReLU): ReLU activation function.

    Methods:
    --------
        anti_rectifier(X): Applies the anti-rectifier operation to the input tensor.
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.
        gram_diagonal_overload(Kx, eps): Applies Gram operation and diagonal loading to a complex matrix.

    """

    def __init__(self, tau: int, N: int, diff_method: str = "root_music",
                 system_model: SystemModel = None, field_type: str = "Far"):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            M (int): Number of sources.

        """
        super(SubspaceNet, self).__init__()
        self.tau = tau
        self.N = N
        self.p = 0.1
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.DropOut = nn.Dropout(self.p)
        self.ReLU = nn.ReLU()
        self.diff_method = None
        self.field_type = field_type
        self.system_model = system_model
        # Set the subspace method for training
        self.set_diff_method(diff_method, system_model)

    def forward(self, Rx_tau: torch.Tensor, is_soft=True, known_angles=None):
        """
        Performs the forward pass of the SubspaceNet.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].
            is_soft (bool): Determines if the model is using the softmask solution to find peaks,
                            or the regular non deferentiable solution.
            known_angles (torch.Tensor): The known angles for the near-field scenario.

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        ############################
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.anti_rectifier(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.anti_rectifier(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.anti_rectifier(x)

        x = self.deconv2(x)
        x = self.anti_rectifier(x)
        # DCNN block #3
        x = self.deconv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #4
        x = self.DropOut(x)
        Rx = self.deconv4(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, :self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N:, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(
            Kx=Kx_tag, eps=1, batch_size=self.batch_size
        )  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to the differentiable subspace algorithm

        if self.field_type == "Far":
            method_output = self.diff_method(Rz)
            if isinstance(method_output, tuple):
                # Root MUSIC output
                doa_prediction, doa_all_predictions, roots = method_output
            else:
                # Esprit output
                doa_prediction = method_output
                doa_all_predictions, roots = None, None
            return doa_prediction, doa_all_predictions, roots, Rz

        elif self.field_type == "Near":
            if known_angles is None:
                doa_prediction, distance_prediction = self.diff_method(Rz, is_soft=is_soft)
                return doa_prediction, distance_prediction, Rz
            else:  # the angles are known
                distance_prediction = self.diff_method(cov=Rz, known_angles=known_angles, is_soft=is_soft)
                return known_angles, distance_prediction, Rz

    def set_diff_method(self, diff_method: str, system_model):
        """Sets the differentiable subspace method for training subspaceNet.
            Options: "root_music", "esprit"

        Args:
        -----
            diff_method (str): differentiable subspace method.

        Raises:
        -------
            Exception: Method diff_method is not defined for SubspaceNet
        """
        if self.field_type == "Far":
            if diff_method.startswith("root_music"):
                self.diff_method = root_music
            elif diff_method.startswith("esprit"):
                self.diff_method = Esprit_torch(system_model=system_model)
            else:
                raise Exception(f"SubspaceNet.set_diff_method:"
                                f" Method {diff_method} is not defined for SubspaceNet in "
                                f"{self.field_type} scenario")
        elif self.field_type == "Near":
            if diff_method.startswith("music_2D"):
                self.diff_method = MUSIC(system_model=system_model, estimation_parameter="angle, range")
            elif diff_method.startswith("music_1D"):
                self.diff_method = MUSIC(system_model=system_model, estimation_parameter="range")
            else:
                raise Exception(f"SubspaceNet.set_diff_method:"
                                f" Method {diff_method} is not defined for SubspaceNet in "
                                f"{self.field_type} Field scenario")

    def anti_rectifier(self, X):
        """Applies the anti-rectifier operation to the input tensor.

        Args:
        -----
            X (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: Output tensor after applying the anti-rectifier operation.

        """
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)

    def adjust_diff_method_temperature(self, epoch):
        if isinstance(self.diff_method, MUSIC):
            if epoch % 20 == 0 and epoch != 0:
                self.diff_method.adjust_cell_size()
                print(f"Model temepartue updated --> {self.get_diff_method_temperature()}")

    def get_diff_method_temperature(self):
        if isinstance(self.diff_method, MUSIC):
            if self.diff_method.estimation_params == "range":
                return self.diff_method.cell_size
            elif self.diff_method.estimation_params == "angle, range":
                return {"angle_cell_size": self.diff_method.cell_size_angle,
                        "distance_cell_size": self.diff_method.cell_size_distance}




class CascadedSubspaceNet(SubspaceNet):
    """
    The CascadedSubspaceNet is a suggested solution for localization in near-field.
    It uses 2 SubspaceNet:
    The first, is SubspaceNet+Esprit/RootMusic, to get the angle from the input tensor.
    The second, uses the first to extract the angles, and then uses the angles to get the distance.
    """

    def __init__(self, tau: int, N: int, system_model: SystemModel = None, state_path: str=""):
        super(CascadedSubspaceNet, self).__init__(tau, N, "music_1D", system_model, "Near")
        self.angle_extractor = None
        self.state_path = state_path
        self.__init_angle_extractor(path=self.state_path)

    def forward(self, Rx_tau: torch.Tensor, is_soft: bool=True, train_angle_extractor: bool=False):
        """
        Performs the forward pass of the CascadedSubspaceNet. Using the subspaceNet forward but,
        calling the angle extractor method first.

        Args
            Rx_tau: The input tensor.
            is_soft: Determines if the model is in using the sofmask solution to find peaks,
             or the regular non deferentiable solution.
            train_angle_extractor: Determines if the angle extractor model is in inference mode(=True),
                                    or training mode(=False).

        Returns
            The distance prediction.
         """
        angles = self.extract_angles(Rx_tau, train_angle_extractor=train_angle_extractor)
        angles, distances, Rx = super().forward(Rx_tau, is_soft=is_soft, known_angles=angles)
        return angles, distances

    def __init_angle_extractor(self, path: str = None):
        self.angle_extractor = SubspaceNet(tau=self.tau,
                                           N=self.N,
                                           diff_method="esprit",
                                           system_model=self.system_model,
                                           field_type="Far")

        self._load_state_for_angle_extractor(path)

    def _load_state_for_angle_extractor(self, path: str = None):
        cwd = os.getcwd()
        if path is None:
            path = self.state_path
        ref_path = os.path.join(cwd, "data", "weights", "final_models", path)
        self.angle_extractor.load_state_dict(torch.load(ref_path, map_location=device))

    def extract_angles(self, Rx_tau: torch.Tensor, train_angle_extractor: bool = False):
        """

        Args:
            Rx_tau: The input tensor.
            is_inference: Determines if the model is in inference mode(=True), or training mode(=False).
            Default is True.

        Returns:
                The angles from the first SubspaceNet model.
        """
        if not train_angle_extractor:
            with torch.no_grad():
                self.angle_extractor.eval()
                angles = self.angle_extractor(Rx_tau)[0]
                self.angle_extractor.train()
            # angles = torch.sort(angles, dim=1)[0]
        else:
            angles = self.angle_extractor(Rx_tau)[0]
            # angles = torch.sort(angles, dim=1)[0]
        return angles


class DeepAugmentedMUSIC(nn.Module):
    """DeepAugmentedMUSIC is a model-based deep learning model for Direction of Arrival (DOA) estimation.

    Attributes:
        N (int): Number of sensors.
        T (int): Number of observations.
        M (int): Number of sources.
        angels (torch.Tensor): Tensor containing angles from -pi/2 to pi/2.
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden layer.
        rnn (nn.GRU): Recurrent neural network module.
        fc (nn.Linear): Fully connected layer.
        fc1 (nn.Linear): Fully connected layer.
        fc2 (nn.Linear): Fully connected layer.
        fc3 (nn.Linear): Fully connected layer.
        ReLU (nn.ReLU): Rectified Linear Unit activation function.
        DropOut (nn.Dropout): Dropout layer.
        BatchNorm (nn.BatchNorm1d): Batch normalization layer.
        sv (torch.Tensor): Steering vector.

    Methods:
        steering_vec(): Computes the steering vector based on the specified parameters.
        spectrum_calculation(Un: torch.Tensor): Calculates the MUSIC spectrum.
        pre_MUSIC(Rz: torch.Tensor): Applies the MUSIC operation for generating the spectrum.
        forward(X: torch.Tensor): Performs the forward pass of the DeepAugmentedMUSIC model.
    """

    def __init__(self, N: int, T: int, M: int):
        """Initializes the DeepAugmentedMUSIC model.

        Args:
        -----
            N (int): Number of sensors.
            M (int): Number of sources.
            T (int): Number of observations.
        """
        super(DeepAugmentedMUSIC, self).__init__()
        self.N, self.T, self.M = N, T, M
        self.angels = torch.linspace(-1 * np.pi / 2, np.pi / 2, 361)
        self.input_size = 2 * self.N
        self.hidden_size = 2 * self.N
        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size * self.N)
        self.fc1 = nn.Linear(self.angels.shape[0], self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.M)
        self.ReLU = nn.ReLU()
        self.DropOut = nn.Dropout(0.25)
        self.BatchNorm = nn.BatchNorm1d(self.T)
        self.sv = self.steering_vec()
        # Weights initialization
        nn.init.xavier_uniform(self.fc.weight)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)

    def steering_vec(self):
        """Computes the ideal steering vector based on the specified parameters.
            equivalent to src.system_model.steering_vec method, but support pyTorch.

        Returns:
        --------
            tensor.Torch: the steering vector
        """
        sv = []
        for angle in self.angels:
            a = torch.exp(
                -1 * 1j * np.pi * torch.linspace(0, self.N - 1, self.N) * np.sin(angle)
            )
            sv.append(a)
        return torch.stack(sv, dim=0)

    def spectrum_calculation(self, Un: torch.Tensor):
        spectrum_equation = []
        for i in range(self.angels.shape[0]):
            spectrum_equation.append(
                torch.real(
                    torch.conj(self.sv[i]).T @ Un @ torch.conj(Un).T @ self.sv[i]
                )
            )
        spectrum_equation = torch.stack(spectrum_equation, dim=0)
        spectrum = 1 / spectrum_equation

        return spectrum, spectrum_equation

    def pre_MUSIC(self, Rz: torch.Tensor):
        """Applies the MUSIC operration for generating spectrum

        Args:
            Rz (torch.Tensor): Generated covariance matrix

        Returns:
            torch.Tensor: The generated MUSIC spectrum
        """
        spectrum = []
        bs_Rz = Rz
        for iter in range(self.batch_size):
            R = bs_Rz[iter]
            # Extract eigenvalues and eigenvectors using EVD
            _, eigenvectors = torch.linalg.eig(R)
            # Noise subspace as the eigenvectors which associated with the M first eigenvalues
            Un = eigenvectors[:, self.M:]
            # Calculate MUSIC spectrum
            spectrum.append(self.spectrum_calculation(Un)[0])
        return torch.stack(spectrum, dim=0)

    def forward(self, X: torch.Tensor):
        """
        Performs the forward pass of the DeepAugmentedMUSIC model.

        Args:
        -----
            X (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: The estimated DOA.
        """
        # X shape == [Batch size, N, T]
        self.BATCH_SIZE = X.shape[0]
        ## Architecture flow ##
        # decompose X and concatenate real and imaginary part
        X = torch.cat(
            (torch.real(X), torch.imag(X)), 1
        )  # Shape ==  [Batch size, 2N, T]
        # Reshape Output shape: [Batch size, T, 2N]
        X = X.view(X.size(0), X.size(2), X.size(1))
        # Apply batch normalization
        X = self.BatchNorm(X)
        # GRU Clock
        gru_out, hn = self.rnn(X)
        Rx = gru_out[:, -1]
        # Reshape Output shape: [Batch size, 1, 2N]
        Rx = Rx.view(Rx.size(0), 1, Rx.size(1))
        # FC Block
        Rx = self.fc(Rx)  # Shape: [Batch size, 1, 2N^2])
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_view = Rx.view(self.BATCH_SIZE, 2 * self.N, self.N)
        Rx_real = Rx_view[:, : self.N, :]  # Shape [Batch size, N, N])
        Rx_imag = Rx_view[:, self.N:, :]  # Shape [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape [Batch size, N, N])
        # Build MUSIC spectrum
        spectrum = self.pre_MUSIC(Kx_tag)  # Shape [Batch size, 361(grid_size)])
        # Apply peak detection using FC block #2
        y = self.ReLU(self.fc1(spectrum))  # Shape [Batch size, 361(grid_size)])
        y = self.ReLU(self.fc2(y))  # Shape [Batch size, 2N])
        y = self.ReLU(self.fc2(y))  # Shape [Batch size, 2N)
        # Find doa
        DOA = self.fc3(y)  # Shape [Batch size, M)
        return DOA


class DeepCNN(nn.Module):
    """DeepCNN is a convolutional neural network model for DoA  estimation.

    Args:
        N (int): Input dimension size.
        grid_size (int): Size of the output grid.

    Attributes:
        N (int): Input dimension size.
        grid_size (int): Size of the output grid.
        conv1 (nn.Conv2d): Convolutional layer 1.
        conv2 (nn.Conv2d): Convolutional layer 2.
        fc1 (nn.Linear): Fully connected layer 1.
        BatchNorm (nn.BatchNorm2d): Batch normalization layer.
        fc2 (nn.Linear): Fully connected layer 2.
        fc3 (nn.Linear): Fully connected layer 3.
        fc4 (nn.Linear): Fully connected layer 4.
        DropOut (nn.Dropout): Dropout layer.
        Sigmoid (nn.Sigmoid): Sigmoid activation function.
        ReLU (nn.ReLU): Rectified Linear Unit activation function.

    Methods:
        forward(X: torch.Tensor): Performs the forward pass of the DeepCNN model.
    """

    def __init__(self, N, grid_size):
        ## input dim (N, T)
        super(DeepCNN, self).__init__()
        self.N = N
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2)
        self.fc1 = nn.Linear(256 * (self.N - 5) * (self.N - 5), 4096)
        self.BatchNorm = nn.BatchNorm2d(256)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, self.grid_size)
        self.DropOut = nn.Dropout(0.3)
        self.Sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, X):
        # X shape == [Batch size, N, N, 3]
        X = X.view(X.size(0), X.size(3), X.size(2), X.size(1))  # [Batch size, 3, N, N]
        ## Architecture flow ##
        # CNN block #1: 3xNxN-->256x(N-2)x(N-2)
        X = self.conv1(X)
        X = self.ReLU(X)
        # CNN block #2: 256x(N-2)x(N-2)-->256x(N-3)x(N-3)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #3: 256x(N-3)x(N-3)-->256x(N-4)x(N-4)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #4: 256x(N-4)x(N-4)-->256x(N-5)x(N-5)
        X = self.conv2(X)
        X = self.ReLU(X)
        # FC BLOCK
        # Reshape Output shape: [Batch size, 256 * (self.N - 5) * (self.N - 5)]
        X = X.view(X.size(0), -1)
        X = self.DropOut(self.ReLU(self.fc1(X)))  # [Batch size, 4096]
        X = self.DropOut(self.ReLU(self.fc2(X)))  # [Batch size, 2048]
        X = self.DropOut(self.ReLU(self.fc3(X)))  # [Batch size, 1024]
        X = self.fc4(X)  # [Batch size, grid_size]
        X = self.Sigmoid(X)
        return X


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


class MUSIC(SubspaceMethod):
    """
    This is implementation of the MUSIC method for localization in Far and Near field environments.
    For Far field - only "angle" can be estimated
    For Near field - "angle", "range" and "angle, range" are the possible options.
    """

    def __init__(self, system_model: SystemModel, estimation_parameter: str):
        """

        Args:
            system_model:
            estimation_parameter:
        """
        super().__init__(system_model)
        self.estimation_params = estimation_parameter
        self.angels = None
        self.distances = None
        self.search_grid = None
        self.music_spectrum = None
        self.__define_grid_params()
        if self.estimation_params == "range":
            self.cell_size = int(self.distances.shape[0] * 0.3)
        elif self.estimation_params == "angle, range":
            self.cell_size_angle = int(self.angels.shape[0] * 0.2)
            self.cell_size_distance = int(self.distances.shape[0] * 0.2)

        self.search_grid = None
        # if this is the music 2D case, the search grid is constant and can be calculated once.
        if self.system_model.params.field_type.startswith("Near"):
            if self.angels is not None and self.distances is not None:
                self.set_search_grid()
        else:
            self.set_search_grid()
        self.noise_subspace = None
        # self.filter = Filter(int(self.distances.shape[0] * 0.05), int(self.distances.shape[0] * 0.2), number_of_filter=5)

    def forward(self, cov, known_angles=None, known_distances=None, is_soft: bool = True):
        """

        Parameters
        ----------
        cov - the covariance tensor to preform the MUSIC one. size: BatchSizeX#SensorsX#Sensors
        known_angles - in case of "range" estimation for the Near-field, use the known angles to create the search grid.
        known_distances - same as "known_angles", but for "angle" estimation.
        is_soft - decide if using hard-decision(peak finder) or soft-decision(the approximated peak finder which is differentiable)

        Returns
        -------
        the returned value depends on the self.estimation_params:
            (self.estimation_params == "angle") - torch.Tensor with the predicted angles
            (self.estimation_params == "range") - torch.Tensor with the predicted ranges
            (self.estimation_params == "angle, range") - tuple, each one of the elements is torch.Tensor for the predicted param.
        """
        # single param estimation: the search grid should be updated for each batch, else, it's the same search grid.
        if self.system_model.params.field_type.startswith("Near"):
            if self.estimation_params in ["angle", "range"]:
                if known_angles.shape[-1] == 1:
                    self.set_search_grid(known_angles=known_angles, known_distances=known_distances)
                else:
                    params = torch.zeros((cov.shape[0], self.system_model.params.M), dtype=torch.float64)
                    for source in range(self.system_model.params.M):
                        params_source = self.forward(cov, known_angles=known_angles[:, source][:, None],
                                                     is_soft=is_soft)
                        params[:, source] = params_source.squeeze()
                    return params
        _, Un = self.subspace_separation(cov.to(torch.complex128), self.system_model.params.M)
        # self.noise_subspace = Un.cpu().detach().numpy()
        inverse_spectrum = self.get_inverse_spectrum(Un)
        self.music_spectrum = 1 / inverse_spectrum
        #####
        # tensor = self.music_spectrum.view(self.music_spectrum.shape[0], 1, self.music_spectrum.shape[1])
        # smoothed_tensor = nn.functional.conv1d(tensor, self.gaussian_kernel.view(1, 1, self.kernel_size), padding=(self.kernel_size - 1) // 2)
        # self.music_spectrum = smoothed_tensor.view(self.music_spectrum.shape[0], self.music_spectrum.shape[1])
        #####
        params = self.peak_finder(is_soft)
        return params

    def adjust_cell_size(self):
        if self.estimation_params == "range":
            # if self.cell_size > int(self.distances.shape[0] * 0.02):
            if self.cell_size > 3 or self.cell_size > int(self.distances.shape[0] * 0.02):
                # self.cell_size -= int(np.ceil(self.distances.shape[0] * 0.01))
                self.cell_size = int(0.95 * self.cell_size)
                if self.cell_size % 2 == 0:
                    self.cell_size -= 1
        elif self.estimation_params == "angle, range":
            if self.cell_size_angle > 3:
                self.cell_size_angle = int(0.95 * self.cell_size_angle)
                if self.cell_size_angle % 2 == 0:
                    self.cell_size_angle -= 1
            if self.cell_size_distance > 3:
                self.cell_size_distance = int(0.95 * self.cell_size_distance)
                if self.cell_size_distance % 2 == 0:
                    self.cell_size_distance -= 1

    def get_inverse_spectrum(self, noise_subspace: torch.Tensor):
        """

        Parameters
        ----------
        noise_subspace - the noise related subspace vectors of size BatchSizex#SENSORSx(#SENSORS-#SOURCES)

        Returns
        -------
        in all cases it will return the inverse spectrum,
        in case of single param estimation it will be 1D inverse spectrum: BatchSizex(length_search_grid)
        in case of dual param estimation it will be 2D inverse spectrum:
                                                    BatchSizex(length_search_grid_angle)x(length_search_grid_distance)
        """
        if self.system_model.params.field_type.startswith("Far"):
            var1 = torch.einsum("an, bnm -> bam", self.search_grid.conj().transpose(0, 1), noise_subspace)
            inverse_spectrum = torch.norm(var1, dim=2)
        else:
            if self.estimation_params.startswith("angle, range"):
                var1 = torch.einsum("adk, bkl -> badl",
                                    torch.transpose(self.search_grid.conj(), 0, 2).transpose(0, 1),
                                    noise_subspace)
                # get the norm value for each element in the batch.
                inverse_spectrum = torch.norm(var1, dim=-1) ** 2
            elif self.estimation_params.endswith("angle"):
                var1 = torch.einsum("ban, nam -> abm", self.search_grid.conj().transpose(0, 2),
                                    noise_subspace.transpose(0, 1))
                inverse_spectrum = torch.norm(var1, dim=2).T
            elif self.estimation_params.startswith("range"):
                var1 = torch.einsum("dbn, nbm -> bdm", self.search_grid.conj().transpose(0, 2),
                                    noise_subspace.transpose(0, 1))
                inverse_spectrum = torch.norm(var1, dim=2)
            else:
                raise ValueError("unknown estimation param")
        return inverse_spectrum

    def peak_finder(self, is_soft: bool):
        """

        Parameters
        ----------
        is_soft: this boolean paramter will determine wether to use derivative approxamtion of the peak_finder for
         the training stage.

        Returns
        -------
        the predicted param(torch.Tensor) or params(tuple)
        """
        if is_soft:
            return self._maskpeak()
        else:
            return self._peak_finder()

    def set_search_grid(self, known_angles: torch.Tensor = None, known_distances: torch.Tensor = None):
        if self.system_model.params.field_type.startswith("Far"):
            self.__set_search_grid_far_field()
        elif self.system_model.params.field_type.startswith("Near"):
            self.__set_search_grid_near_field(known_angles=known_angles, known_distances=known_distances)
        else:
            raise ValueError(f"set_search_grid: Unrecognized field type: {self.system_model.params.field_type}")

    def __set_search_grid_far_field(self):
        array = torch.Tensor(self.system_model.array[:, None]).to(torch.float64).to(device)
        theta = self.angels[:, None]
        time_delay = torch.einsum("nm, na -> na",
                                  array,
                                  torch.sin(theta).repeat(1, self.system_model.params.N).T
                                  * self.system_model.dist_array_elems["NarrowBand"])
        self.search_grid = torch.exp(-2 * 1j * torch.pi * time_delay)

    def __set_search_grid_near_field(self, known_angles: torch.Tensor = None, known_distances: torch.Tensor = None):
        """

        Returns:

        """
        dist_array_elems = self.system_model.dist_array_elems["NarrowBand"]
        if known_angles is None:
            theta = self.angels[:, None]
        else:
            theta = known_angles.float()
            if len(theta.shape) == 1:
                theta = torch.atleast_1d(theta)[:, None].to(torch.float64)

        if known_distances is None:
            distances = self.distances[:, None].to(torch.float64)
        else:
            distances = known_distances.float()
            if len(distances.shape) == 1:
                distances = torch.atleast_1d(distances)[:, None]
        array = torch.Tensor(self.system_model.array[:, None]).to(torch.float64).to(device)
        array_square = torch.pow(array, 2).to(torch.float64)

        first_order = torch.einsum("nm, na -> na",
                                   array,
                                   torch.sin(theta).repeat(1, self.system_model.params.N).T * dist_array_elems)

        second_order = -0.5 * torch.div(torch.pow(torch.cos(theta) * dist_array_elems, 2), distances.T)
        second_order = second_order[:, :, None].repeat(1, 1, self.system_model.params.N)
        second_order = torch.einsum("nm, nda -> nda",
                                    array_square,
                                    torch.transpose(second_order, 2, 0)).transpose(1, 2)

        first_order = first_order[:, :, None].repeat(1, 1, second_order.shape[-1])

        time_delay = first_order + second_order

        self.search_grid = torch.exp(2 * -1j * torch.pi * time_delay)

    def plot_spectrum(self, highlight_corrdinates=None, batch: int = 0, method: str = "heatmap"):
        if self.estimation_params == "angle, range":
            self._plot_3d_spectrum(highlight_corrdinates, batch, method)
        else:
            self._plot_1d_spectrum(highlight_corrdinates, batch)

    def _peak_finder(self):
        if self.system_model.params.field_type.startswith("Far"):
            return self._peak_finder_1D(self.angels, self.system_model.params.M)
        else:
            if self.estimation_params.startswith("angle, range"):
                return self._peak_finder_2D()
            elif self.estimation_params.endswith("angle"):
                return self._peak_finder_1D(self.angels)
            elif self.estimation_params.startswith("range"):
                return self._peak_finder_1D(self.distances)

    def _maskpeak(self):
        if self.system_model.params.field_type.startswith("Far"):
            return self._maskpeak_1D(self.angels)
        else:
            if self.estimation_params.startswith("angle, range"):
                return self._maskpeak_2D()
            elif self.estimation_params.endswith("angle"):
                return self._maskpeak_1D(self.angels)
            elif self.estimation_params.startswith("range"):
                return self._maskpeak_1D(self.distances)

    def _maskpeak_1D(self, search_space):
        flag = True
        if flag:

            # top_indxs = torch.topk(self.music_spectrum, 1, dim=1)[1]
            peaks = torch.zeros(self.music_spectrum.shape[0], 1).to(torch.int64)
            for batch in range(peaks.shape[0]):
                music_spectrum = self.music_spectrum[batch].cpu().detach().numpy().squeeze()
                # Find spectrum peaks
                peaks_tmp = list(sc.signal.find_peaks(music_spectrum)[0])
                # Sort the peak by their amplitude
                peaks_tmp.sort(key=lambda x: music_spectrum[x], reverse=True)
                if len(peaks_tmp) == 0:
                    peaks_tmp = torch.randint(self.distances.shape[0], (1,))
                else:
                    peaks_tmp = peaks_tmp[0]
                peaks[batch] = peaks_tmp
            top_indxs = peaks
            cell_idx = (top_indxs - self.cell_size + torch.arange(2 * self.cell_size + 1, dtype=torch.long,
                                                                  device=device))
            cell_idx %= self.music_spectrum.shape[1]
            cell_idx = cell_idx.unsqueeze(-1)
            metrix_thr = torch.gather(self.music_spectrum.unsqueeze(-1).expand(-1, -1, cell_idx.size(-1)), 1, cell_idx)
            soft_max = torch.softmax(metrix_thr, dim=1)
            soft_decision = torch.einsum("bkm, bkm -> bm", search_space[cell_idx], soft_max)
        else:
            soft_decision = self.filter(self.music_spectrum, search_space)

        return soft_decision

    def _maskpeak_2D(self):

        batch_size = self.music_spectrum.shape[0]
        soft_row = torch.zeros((batch_size, self.system_model.params.M))
        soft_col = torch.zeros((batch_size, self.system_model.params.M))
        max_row = torch.zeros((self.music_spectrum.shape[0], self.system_model.params.M), dtype=torch.int64)
        max_col = torch.zeros((self.music_spectrum.shape[0], self.system_model.params.M), dtype=torch.int64)
        for batch in range(self.music_spectrum.shape[0]):
            music_spectrum = self.music_spectrum[batch].detach().cpu().numpy().squeeze()
            # Flatten the spectrum
            spectrum_flatten = music_spectrum.flatten()
            # Find spectrum peaks
            peaks = sc.signal.find_peaks(spectrum_flatten)[0]
            # Sort the peak by their amplitude
            sorted_peaks = peaks[np.argsort(spectrum_flatten[peaks])[::-1]]
            # convert the peaks to 2d indices
            original_idx = torch.from_numpy(np.column_stack(np.unravel_index(sorted_peaks, music_spectrum.shape))).T
            if self.system_model.params.M > 1:
                original_idx = keep_far_enough_points(original_idx, self.system_model.params.M, 85)
            max_row[batch] = original_idx[0][0 : self.system_model.params.M]
            max_col[batch] = original_idx[1][0 : self.system_model.params.M]
        for source in range(self.system_model.params.M):
            max_row_cell_idx = (max_row[:, source][:, None]
                                - self.cell_size_angle
                                + torch.arange(2 * self.cell_size_angle + 1, dtype=torch.int32, device=device))
            max_row_cell_idx %= self.music_spectrum.shape[1]
            max_row_cell_idx = max_row_cell_idx.reshape(batch_size, -1, 1)

            max_col_cell_idx = (max_col[:, source][:, None]
                                - self.cell_size_distance
                                + torch.arange(2 * self.cell_size_distance + 1, dtype=torch.int32, device=device))
            max_col_cell_idx %= self.music_spectrum.shape[2]
            max_col_cell_idx = max_col_cell_idx.reshape(batch_size, 1, -1)

            metrix_thr = self.music_spectrum.gather(1, max_row_cell_idx.expand(-1, -1, self.music_spectrum.shape[2]))
            metrix_thr = metrix_thr.gather(2, max_col_cell_idx.repeat(1, max_row_cell_idx.shape[-2], 1))
            soft_max = torch.softmax(metrix_thr.view(batch_size, -1), dim=1).reshape(metrix_thr.shape)
            soft_row[:, source][:, None] = torch.einsum("bla, bad -> bl",
                                                        self.angels[max_row_cell_idx].transpose(1, 2),
                                                        torch.sum(soft_max, dim=2).unsqueeze(-1))
            soft_col[:, source][:, None] = torch.einsum("bmc, bcm -> bm",
                                                        self.distances[max_col_cell_idx],
                                                        torch.sum(soft_max, dim=1).unsqueeze(-1))

        return soft_row, soft_col

    def _peak_finder_1D(self, search_space, num_peaks: int = 1):
        predict_param = torch.zeros(self.music_spectrum.shape[0], num_peaks)
        for batch in range(self.music_spectrum.shape[0]):
            music_spectrum = self.music_spectrum[batch].cpu().detach().numpy().squeeze()
            # Find spectrum peaks
            peaks = list(sc.signal.find_peaks(music_spectrum)[0])
            # Sort the peak by their amplitude
            peaks.sort(key=lambda x: music_spectrum[x], reverse=True)
            tmp = search_space[peaks[0:num_peaks]]
            if tmp.nelement() == 0:
                rand_idx = torch.randint(self.distances.shape[0], (1,))
                tmp = self.distances[rand_idx]
                # tmp = self._maskpeak_1D(search_space)
                # print("_peak_finder_1D: No peaks were found!")
            else:
                pass
            predict_param[batch] = tmp

        return torch.Tensor(predict_param)

    def _peak_finder_2D(self):
        predict_theta = np.zeros((self.music_spectrum.shape[0], self.system_model.params.M), dtype=np.float64)
        predict_dist = np.zeros((self.music_spectrum.shape[0], self.system_model.params.M), dtype=np.float64)
        angels = self.angels.cpu().detach().numpy()
        distances = self.distances.cpu().detach().numpy()
        for batch in range(self.music_spectrum.shape[0]):
            music_spectrum = self.music_spectrum[batch].detach().cpu().numpy().squeeze()
            # Flatten the spectrum
            spectrum_flatten = music_spectrum.flatten()
            # Find spectrum peaks
            peaks = list(sc.signal.find_peaks(spectrum_flatten)[0])
            # Sort the peak by their amplitude
            peaks.sort(key=lambda x: spectrum_flatten[x], reverse=True)
            # convert the peaks to 2d indices
            original_idx = np.array(np.unravel_index(peaks, music_spectrum.shape))
            original_idx = original_idx[:, 0:self.system_model.params.M]
            predict_theta[batch] = angels[original_idx[0]]
            predict_dist[batch] = distances[original_idx[1]]

        return torch.from_numpy(predict_theta), torch.from_numpy(predict_dist)

    def _init_spectrum(self, batch_size):
        if self.system_model.params.field_type == "Far":
            self.music_spectrum = torch.zeros(batch_size, len(self.angels))
        else:
            if self.estimation_params.startswith("angle, range"):
                self.music_spectrum = torch.zeros(batch_size, len(self.angels), len(self.distances))
            elif self.estimation_params.endswith("angle"):
                self.music_spectrum = torch.zeros(batch_size, len(self.angels))
            elif self.estimation_params.startswith("range"):
                self.music_spectrum = torch.zeros(batch_size, len(self.distances))

    def __define_grid_params(self):
        if self.system_model.params.field_type.startswith("Far"):
            # if it's the Far field case, need to init angles range.
            self.angels = torch.arange(-1 * torch.pi / 3, torch.pi / 3, torch.pi / 360, device=device,
                                       dtype=torch.float64).requires_grad_(True).to(torch.float64)
        elif self.system_model.params.field_type.startswith("Near"):
            # if it's the Near field, there are 3 possabilities.
            fresnel = self.system_model.fresnel
            fraunhofer = self.system_model.fraunhofer
            if self.estimation_params.startswith("angle"):
                self.angels = torch.arange(-1 * torch.pi / 3, torch.pi / 3, torch.pi / 360,
                                           device=device).requires_grad_(True).to(torch.float64)
                # self.angels = torch.from_numpy(np.arange(-np.pi / 2, np.pi / 2, np.pi / 90)).requires_grad_(True)
            if self.estimation_params.endswith("range"):
                self.distances = torch.arange(np.floor(fresnel), fraunhofer + 0.5, .5, device=device,
                                              dtype=torch.float64).requires_grad_(True)
            else:
                raise ValueError(f"estimation_parameter allowed values are [(angle), (range), (angle, range)],"
                                 f" got {self.estimation_params}")
        else:
            raise ValueError(f"Unrecognized field type for MUSIC class init stage,"
                             f" got {self.system_model.params.field_type} but only Far and Near are allowed.")

    def _plot_1d_spectrum(self, highlight_corrdinates, batch):
        if self.estimation_params == "angle":
            x = np.rad2deg(self.angels.detach().cpu().numpy())
            x_label = "angle [deg]"
        elif self.estimation_params == "range":
            x = self.distances.detach().cpu().numpy()
            x_label = "distance [m]"
        else:
            raise ValueError(f"_plot_1d_spectrum: No such option for param estimation.")
        y = self.music_spectrum[batch].detach().cpu().numpy()
        plt.figure()
        plt.plot(x, y.T, label="Music Spectrum")
        if highlight_corrdinates is not None:
            for idx, dot in enumerate(highlight_corrdinates):
                plt.scatter(dot, label=f"dot {idx}")
        plt.title("MUSIC SPECTRUM")
        plt.grid()
        plt.ylabel("Spectrum power")
        plt.xlabel(x_label)
        plt.legend()
        plt.show()

    def _plot_3d_spectrum(self, highlight_coordinates, batch, method):
        """
        Plot the MUSIC 2D spectrum.

        """
        if method == "3D":
            # Creating figure
            distances = self.distances.detach().cpu().numpy()
            angles = self.angels.detach().cpu().numpy()
            spectrum = self.music_spectrum[batch].detach().cpu().numpy()
            x, y = np.meshgrid(distances, np.rad2deg(angles))
            # Plotting the 3D surface
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(x, y, 10 * np.log10(spectrum), cmap='viridis')

            if highlight_coordinates:
                highlight_coordinates = np.array(highlight_coordinates)
                ax.scatter(
                    highlight_coordinates[:, 0],
                    np.rad2deg(highlight_coordinates[:, 1]),
                    np.log1p(highlight_coordinates[:, 2]),
                    color='red',
                    s=50,
                    label='Highlight Points'
                )
            ax.set_title('MUSIC spectrum')
            ax.set_xlim(distances[0], distances[-1])
            ax.set_ylim(np.rad2deg(angles[0]), np.rad2deg(angles[-1]))
            # Adding labels
            ax.set_ylabel('Theta [deg]')
            ax.set_xlabel('Radius [m]')
            ax.set_zlabel('Power [dB]')
            plt.colorbar(ax.plot_surface(x, y, 10 * np.log10(spectrum), cmap='viridis'), shrink=0.5, aspect=5)

            if highlight_coordinates:
                ax.legend()  # Adding a legend

            # Display the plot
            plt.show()
        elif method == "heatmap":
            xmin, xmax = np.min(self.distances.cpu().detach().numpy()), np.max(self.distances.cpu().detach().numpy())
            ymin, ymax = np.min(self.angels.cpu().detach().numpy()), np.max(self.angels.cpu().detach().numpy())
            spectrum = self.music_spectrum[batch].cpu().detach().numpy()
            plt.imshow(spectrum, cmap="hot",
                       extent=[xmin, xmax, np.rad2deg(ymin), np.rad2deg(ymax)], origin='lower', aspect="auto")
            if highlight_coordinates is not None:
                for idx, dot in enumerate(highlight_coordinates):
                    x = self.distances.cpu().detach().numpy()[dot[1]]
                    y = np.rad2deg(self.angels.cpu().detach().numpy()[dot[0]])
                    plt.scatter(x, y, label=f"dot {idx}: ({np.round(x, decimals=3), np.round(y, decimals=3)})")
                plt.legend()
            plt.colorbar()
            plt.title("MUSIC Spectrum heatmap")
            plt.xlabel("Distances [m]")
            plt.ylabel("Angles [deg]")

            plt.figaspect(2)
            plt.show()


class Esprit_torch(SubspaceMethod):
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
        doa_predictions = -1 * torch.arcsin((1 / np.pi) * eigenvalues_angels)
        doa_batches.append(doa_predictions)

    return torch.stack(doa_batches, dim=0)

class Filter(nn.Module):
    def __init__(self, min_cell_size, max_cell_size, number_of_filter=10):
        super(Filter, self).__init__()
        self.number_of_filters = number_of_filter
        self.cell_sizes = torch.linspace(min_cell_size, max_cell_size, number_of_filter).to(torch.int32).to(device)
        self.cell_bank = {}
        for cell_size in enumerate(self.cell_sizes.data):
            cell_size = cell_size[1]
            self.cell_bank[cell_size] = torch.arange(-cell_size, cell_size, 1, dtype=torch.long, device=device)
        self.fc = nn.Linear(self.number_of_filters, 1)
        self.fc.weight.data = torch.randn(1, number_of_filter) / 100 + (1 / number_of_filter)
        self.fc.weight.data = self.fc.weight.data.to(torch.float64)
        self.fc.bias.data = torch.Tensor([0])
        self.fc.bias.data = self.fc.bias.data.to(torch.float64)
        self.fc.bias.requires_grad_(False)
        self.relu = nn.ReLU()

    def forward(self, input, search_space):
        peaks = torch.zeros(input.shape[0], 1).to(torch.int64)
        for batch in range(peaks.shape[0]):
            music_spectrum = input[batch].cpu().detach().numpy().squeeze()
            # Find spectrum peaks
            peaks_tmp = list(sc.signal.find_peaks(music_spectrum)[0])
            # Sort the peak by their amplitude
            peaks_tmp.sort(key=lambda x: music_spectrum[x], reverse=True)
            if len(peaks_tmp) == 0:
                peaks_tmp = torch.randint(search_space.shape[0], (1,))
            else:
                peaks_tmp = peaks_tmp[0]
            peaks[batch] = peaks_tmp
        top_1 = peaks
        output = torch.zeros(input.shape[0], self.number_of_filters).to(device).to(torch.float64)
        for idx, cell in enumerate(self.cell_bank.values()):
            tmp_cell = top_1 + cell
            tmp_cell %= input.shape[1]
            tmp_cell = tmp_cell.unsqueeze(-1)
            metrix_thr = torch.gather(input.unsqueeze(-1).expand(-1, -1, tmp_cell.size(-1)), 1, tmp_cell)
            soft_max = torch.softmax(metrix_thr, dim=1)
            output[:, idx] = torch.einsum("bkm, bkm -> bm", search_space[tmp_cell], soft_max).squeeze()
        output = self.fc(output)
        output = self.relu(output)
        self.clip_weights_values()
        return output

    def clip_weights_values(self):
        self.fc.weight.data = torch.clip(self.fc.weight.data, 0.1, 1)
        self.fc.weight.data /= torch.sum(self.fc.weight.data)

class SubspaceNetEsprit(SubspaceNet):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.
        SubspaceNetEsprit is based on the ability to perform back-propagation using ESPRIT algorithm,
        instead of RootMUSIC.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.

    Methods:
    --------
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.

    """

    def __init__(self, tau: int, M: int):
        super().__init__(tau, M)

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of the SubspaceNet.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.anti_rectifier(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.anti_rectifier(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #1
        x = self.deconv2(x)
        x = self.anti_rectifier(x)
        # DCNN block #2
        x = self.deconv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #3
        x = self.DropOut(x)
        Rx = self.deconv4(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N:, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(
            Kx=Kx_tag, eps=1, batch_size=self.batch_size
        )  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to Esprit algorithm
        doa_prediction = esprit(Rz, self.M, self.batch_size)
        return doa_prediction, Rz


def keep_far_enough_points(tensor, M, D):
    # Calculate pairwise distances between columns
    distances = cdist(tensor.T, tensor.T, metric="euclidean")

    # Keep the first M columns as far enough points
    selected_cols = []
    for i in range(tensor.shape[1]):
        if len(selected_cols) >= M:
            break
        if all(distances[i, col] >= D for col in selected_cols):
            selected_cols.append(i)

    # Remove columns that are less than distance D from each other
    filtered_tensor = tensor[:, selected_cols]

    return filtered_tensor