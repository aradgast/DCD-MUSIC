import os
from pathlib import Path

from src.models_pack.subspacenet import SubspaceNet
from src.system_model import SystemModel
from src.utils import *


class DCDMUSIC(SubspaceNet):
    """
    The Deep-Cascadede-defferntiable MUSIC is a suggested solution for localization in near-field.
    It uses 2 SubspaceNet:
    The first, is SubspaceNet+Esprit/RootMusic/MUSIC(with Maskpeak), to get the angle from the input tensor.
    The second, uses the first to extract the angles, and then uses the angles to get the distance.
    """

    def __init__(self, tau: int, system_model: SystemModel = None, state_path: str = None):
        super(DCDMUSIC, self).__init__(tau, "music_1D", system_model, "Near")
        self.angle_extractor = None
        self.state_path = state_path
        self.__init_angle_extractor(path=self.state_path)

    def forward(self, x: torch.Tensor, number_of_sources: int, train_angle_extractor: bool = False):
        """
        Performs the forward pass of the DCD-MUSIC. Using the subspaceNet forward but,
        calling the angle extractor method first.

        Args
            x: The input signal.
            number_of_sources: The number of sources in the signal.
            train_angle_extractor: Determines if the model is training the angle branch.

        Returns
            The distance prediction.
         """
        eigen_regularization = None
        angles, sources_estimation, eigen_regularization = self.extract_angles(
            x, number_of_sources, train_angle_extractor=train_angle_extractor)
        _, distances, _ = super().forward(x, sources_num=number_of_sources, known_angles=angles)
        return angles, distances, sources_estimation, eigen_regularization

    def __init_angle_extractor(self, path: str = None):
        self.angle_extractor = SubspaceNet(tau=self.tau,
                                           diff_method="esprit",
                                           system_model=self.system_model,
                                           field_type="Far")
        if path is None:
            path = self.angle_extractor.get_model_file_name()
        self._load_state_for_angle_extractor(path)

    def _load_state_for_angle_extractor(self, path: str = None):
        cwd = Path(__file__).parent.parent.parent
        if path is None or path == "":
            path = self.angle_extractor.get_model_file_name()
        ref_path = os.path.join(cwd, "data", "weights", "final_models", path)
        try:
            self.angle_extractor.load_state_dict(torch.load(ref_path, map_location=device))
        except FileNotFoundError as e:
            print(f"DCDMUSIC._load_state_for_angle_extractor: Model state not found in {ref_path}")

    def extract_angles(self, Rx_tau: torch.Tensor, number_of_sources: int, train_angle_extractor: bool = False):
        """

        Args:
            Rx_tau: The input tensor.
            number_of_sources: The number of sources in the signal.
            train_angle_extractor: Determines if the model is training the angle branch.

        Returns:
                The angles from the first SubspaceNet model.
        """
        eigen_regularization = None
        if not train_angle_extractor:
            with torch.no_grad():
                self.angle_extractor.eval()
                angles, sources_estimation, _ = self.angle_extractor(Rx_tau, number_of_sources)
                self.angle_extractor.train()
            # angles = torch.sort(angles, dim=1)[0]
        else:
            angles, sources_estimation, eigen_regularization = self.angle_extractor(Rx_tau, number_of_sources)
            # angles = torch.sort(angles, dim=1)[0]
        return angles, sources_estimation, eigen_regularization

    def print_model_params(self):
        return f"tau={self.tau}"

    def get_model_params(self):
        return {"tau": self.tau}
