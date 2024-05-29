import torch
import torch.nn as nn
import os

from src.models_pack.subspacenet import SubspaceNet
from src.system_model import SystemModel
from src.utils import *

class CascadedSubspaceNet(SubspaceNet):
    """
    The CascadedSubspaceNet is a suggested solution for localization in near-field.
    It uses 2 SubspaceNet:
    The first, is SubspaceNet+Esprit/RootMusic, to get the angle from the input tensor.
    The second, uses the first to extract the angles, and then uses the angles to get the distance.
    """

    def __init__(self, tau: int, N: int, system_model: SystemModel = None,state_path: str=""):
        super(CascadedSubspaceNet, self).__init__(tau, N, "music_1D", system_model, "Near")
        self.angle_extractor = None
        self.state_path = state_path
        self.__init_angle_extractor(path=self.state_path)


    def forward(self, Rx_tau: torch.Tensor, is_soft: bool = True, train_angle_extractor: bool = False):
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
        if path is None or path == "":
            path = self.angle_extractor.get_model_file_name()
        ref_path = os.path.join(cwd, "data", "weights", "final_models", path)
        try:
            self.angle_extractor.load_state_dict(torch.load(ref_path, map_location=device))
        except FileNotFoundError:
            print(f"Model state not found in {ref_path}")

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

    def get_model_file_name(self):
        return f"CascadedSubspaceNet_" + \
                f"N={self.N}_" + \
                f"tau={self.tau}_" + \
                f"M={self.system_model.params.M}_" + \
                f"{self.system_model.params.signal_type}_" + \
                f"SNR={self.system_model.params.snr}_" + \
                f"{self.system_model.params.signal_nature}"

    def get_model_name(self):
        return "CascadedSubspaceNet"