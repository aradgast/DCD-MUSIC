import os
from pathlib import Path

from sympy.physics.vector.printing import params

from src.criterions import CartesianLoss, RMSPELoss, NoiseOrthogonalLoss
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

    def __init__(self, tau: int, system_model: SystemModel,diff_method: tuple = ("esprit", "music_1d"), state_path: str = None):
        super(DCDMUSIC, self).__init__(tau, diff_method[1], system_model, "Near")
        self.angle_extractor = None
        self.angle_extractor_diff_method = diff_method[0]
        self.state_path = state_path
        self.__init_angle_extractor(path=self.state_path)
        self.train_angle_extractor = False
        self.__set_criterion()
        self.set_eigenregularization_schedular()

    def forward(self, x: torch.Tensor, number_of_sources: int = None):
        """
        Performs the forward pass of the DCD-MUSIC. Using the subspaceNet forward but,
        calling the angle extractor method first.

        Args
            x: The input signal.
            number_of_sources: The number of sources in the signal.

        Returns
            The distance prediction.
         """
        if self.loss_type == "orthogonality" and self.training:
            Rz = self.get_surrogate_covariance(x)
            _, noise_subspace, source_estimation, eigen_regularization = self.diff_method.subspace_separation(Rz,
                                                                                                              number_of_sources)
            return noise_subspace, source_estimation, eigen_regularization

        if not self.train_angle_extractor:
            angles, sources_estimation, eigen_regularization = self.extract_angles(x, number_of_sources)
        else: # in this case, when using orthogonality loss, the angle extractor doesn't extract angles. TODO
            raise NotImplementedError

        _, distances, _ = super().forward(x, sources_num=number_of_sources, known_angles=angles)
        return angles, distances, sources_estimation, eigen_regularization

    def __init_angle_extractor(self, path: str = None):
        self.angle_extractor = SubspaceNet(tau=self.tau,
                                           diff_method=self.angle_extractor_diff_method,
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

    def extract_angles(self, Rx_tau: torch.Tensor, number_of_sources: int):
        """

        Args:
            Rx_tau: The input tensor.
            number_of_sources: The number of sources in the signal.
            train_angle_extractor: Determines if the model is training the angle branch.

        Returns:
                The angles from the first SubspaceNet model.
        """
        eigen_regularization = None
        if not self.train_angle_extractor:
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
        params = self.get_model_params()
        return f"tau={params.get('tau')}_diff_methods={params.get('diff_methods')[0]}_{params.get('diff_methods')[1]}"

    def get_model_params(self):
        if str(self.angle_extractor.diff_method).startswith("music"):
            angle_extractor_diff_method = str(self.angle_extractor.diff_method) + "_" + self.angle_extractor.loss_type
        else:
            angle_extractor_diff_method = str(self.angle_extractor.diff_method)
        if str(self.diff_method).startswith("music"):
            diff_method = str(self.diff_method) + "_" + self.loss_type
        else:
            diff_method = str(self.diff_method)
        return {"tau": self.tau, "diff_methods": (angle_extractor_diff_method ,diff_method)}

    def training_step(self, batch, batch_idx):
        x, sources_num, labels, masks = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if (sources_num != sources_num[0]).any():
            raise ValueError(f"SubspaceNet.__training_step_near_field: "
                             f"Number of sources in the batch is not equal for all samples.")
        sources_num = sources_num[0]
        angles, ranges = torch.split(labels, sources_num, dim=1)
        masks, _ = torch.split(masks, sources_num, dim=1)
        x = x.requires_grad_(True).to(device)
        angles = angles.requires_grad_(True).to(device)
        ranges = ranges.requires_grad_(True).to(device)
        if self.loss_type == "orthogonality":
            noise_subspace, sources_estimation, eigen_regularization = self(x, sources_num)
            loss = self.train_loss(noise_subspace=noise_subspace, angles=angles, ranges=ranges)
        else:
            angles_pred, distances_pred, sources_estimation, eigen_regularization = self(x, sources_num)
            loss = self.train_loss(angles, angles_pred, ranges, distances_pred)
        if isinstance(loss, tuple):
            loss = loss[0]
        acc = self.source_estimation_accuracy(sources_num, sources_estimation)
        if eigen_regularization is None:
            return loss, acc
        else:
            return loss + eigen_regularization * self.eigenregularization_weight, acc


    def validation_step(self, batch, batch_idx):
        x, sources_num, labels, masks = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if (sources_num != sources_num[0]).any():
            raise ValueError(f"SubspaceNet.__training_step_near_field: "
                             f"Number of sources in the batch is not equal for all samples.")
        sources_num = sources_num[0]
        angles, ranges = torch.split(labels, sources_num, dim=1)
        masks, _ = torch.split(masks, sources_num, dim=1)
        x = x.to(device)
        angles = angles.to(device)
        ranges = ranges.to(device)

        angles_pred, ranges_pred, sources_estimation, eigen_regularization = self(x, sources_num)
        loss = self.validation_loss(angles_pred, angles, ranges_pred, ranges)
        acc = self.source_estimation_accuracy(sources_num, sources_estimation)
        return loss, acc

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        x = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        angles, distances, sources_estimation, eigen_regularization = self.forward(x, self.system_model.params.M)
        return angles, distances, sources_estimation

    def __set_criterion(self):
        if self.loss_type == "orthogonality":
            self.train_loss = NoiseOrthogonalLoss(
                array=torch.Tensor(self.system_model.array[:, None]).to(torch.float64).to(device),
                sensors_distance=self.system_model.dist_array_elems["NarrowBand"])
        else:
            if self.train_angle_extractor:
                self.train_loss = CartesianLoss()
            else:
                self.train_loss = RMSPELoss(balance_factor=0.0)
        self.validation_loss = CartesianLoss()
        self.test_loss = CartesianLoss()

    def update_criterion(self):
        self.__set_criterion()


