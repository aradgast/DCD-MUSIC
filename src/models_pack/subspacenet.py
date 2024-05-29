"""
SubspaceNet: model-based deep learning algorithm as described in:
        [2] "SubspaceNet: Deep Learning-Aided Subspace methods for DoA Estimation".
"""

import torch
import torch.nn as nn
from src.system_model import SystemModel
from src.utils import *

from src.methods_pack.music import MUSIC
from src.methods_pack.esprit import ESPRIT, esprit
from src.methods_pack.root_music import RootMusic, root_music


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

    def forward(self, x: torch.Tensor, is_soft=True, known_angles=None):
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
        x = self.pre_processing(x)
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = x.shape[-1]
        self.batch_size = x.shape[0]
        ############################
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(x)
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

    def pre_processing(self, x):
        """
        The input data is a complex signal of size [batch, N, T] and the input to the model supposed to be complex
         tensors of size [batch, tau, 2N, N].
        """
        batch_size = x.shape[0]
        Rx_tau = torch.zeros(batch_size, self.tau, 2 * self.N, self.N, device=device)
        meu = torch.mean(x, dim=-1, keepdim=True).to(device)
        center_x = x - meu

        for i in range(self.tau):
            x1 = center_x[:, :, :center_x.shape[-1] - i].to(torch.complex128)
            x2 = torch.conj(center_x[:, :, i:]).transpose(1, 2).to(torch.complex128)
            Rx_lag = torch.einsum("BNT, BTM -> BNM", x1, x2) / (center_x.shape[-1] - i - 1)
            Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), dim=1)
            Rx_tau[:, i, :, :] = Rx_lag

        return Rx_tau


    def get_model_file_name(self):
        return f"SubspaceNet_" + \
                f"N={self.N}_" + \
                f"tau={self.tau}_" + \
                f"M={self.system_model.params.M}_" + \
                f"{self.system_model.params.signal_type}_" + \
                f"SNR={self.system_model.params.snr}_" + \
                f"diff_method=esprit_" + \
                f"{self.system_model.params.field_type}_field_" +  \
                f"{self.system_model.params.signal_nature}"


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
                self.diff_method = ESPRIT(system_model=system_model)
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

    def get_model_name(self):
        return "SubspaceNet"

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