import os
from pathlib import Path

from sympy.physics.vector.printing import params

from src.criterions import CartesianLoss, RMSPELoss, MusicSpectrumLoss
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

    def __init__(self, tau: int, system_model: SystemModel,diff_method: tuple = ("esprit", "music_1d"),
                 train_loss_type: str=("rmspe", "rmspe"),
                 state_path: str = None, angle_extractor: SubspaceNet = None, train_angle_extractor: bool = False):
        super(DCDMUSIC, self).__init__(tau, diff_method[1], train_loss_type[1],system_model, "near")
        self.angle_extractor = None
        self.angle_extractor_diff_method = diff_method[0]
        self.angle_extractor_train_loss_type = train_loss_type[0]
        self.state_path = state_path
        self.__init_angle_extractor(path=self.state_path, model=angle_extractor)
        self.train_angle_extractor = train_angle_extractor
        self.__set_criterion()
        self.set_eigenregularization_schedular(init_value=0.0, step_size=10000, gamma=1.0)
        self.schedular_min_weight = 0.0

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
        if self.train_loss_type == "music_spectrum" and self.training:
            angles, sources_estimation, eigen_regularization = self.extract_angles(x, number_of_sources)
            Rz = self.get_surrogate_covariance(x)
            _, noise_subspace, source_estimation, eigen_regularization = self.diff_method.subspace_separation(Rz,
                                                                                                              number_of_sources)
            return angles, noise_subspace, source_estimation, eigen_regularization

        else:
            angles, sources_estimation, eigen_regularization = self.extract_angles(x, number_of_sources)

            _, distances, _ = super().forward(x, sources_num=number_of_sources, known_angles=angles)
            return angles, distances, sources_estimation, eigen_regularization

    def __init_angle_extractor(self, path: str = None, model: SubspaceNet = None):
        if model is not None:
            self.angle_extractor = model
        else:
            self.angle_extractor = SubspaceNet(tau=self.tau,
                                               diff_method=self.angle_extractor_diff_method,
                                               train_loss_type=self.angle_extractor_train_loss_type,
                                               system_model=self.system_model,
                                               field_type="far")
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
            # raise FileNotFoundError(f"DCDMUSIC._load_state_for_angle_extractor: Model state not found in {ref_path}")
            print(f"DCDMUSIC._load_state_for_angle_extractor: Model state not found in {ref_path}")
        print(f"DCDMUSIC._load_state_for_angle_extractor: Model state loaded from {ref_path}")

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
            if self.angle_extractor.training:
                self.angle_extractor.eval()
            with torch.no_grad():
                angles, sources_estimation, _ = self.angle_extractor(Rx_tau, number_of_sources)
        else:
            if not self.angle_extractor.training:
                self.angle_extractor.train()
            angles, sources_estimation, eigen_regularization = self.angle_extractor(Rx_tau, number_of_sources)
        return angles, sources_estimation, eigen_regularization

    def print_model_params(self):
        params = self.get_model_params()
        return f"tau={params.get('tau')}_diff_methods={params.get('diff_methods')[0]}_{params.get('diff_methods')[1]}"

    def get_model_params(self):
        if str(self.angle_extractor.diff_method).startswith("music"):
            angle_extractor_diff_method = str(self.angle_extractor.diff_method) + "_" + self.angle_extractor.train_loss_type
        else:
            angle_extractor_diff_method = str(self.angle_extractor_diff_method)
        if str(self.diff_method).startswith("music"):
            diff_method = str(self.diff_method) + "_" + self.train_loss_type
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
        if self.train_loss_type == "music_spectrum":
            if self.diff_method.estimation_params == "angle, range":
                noise_subspace, sources_estimation, eigen_regularization = self(x, sources_num)
                loss = self.train_loss(noise_subspace=noise_subspace, angles=angles, ranges=ranges)
            else:
                angles_pred, noise_subspace, sources_estimation, eigen_regularization = self(x, sources_num)
                loss = self.train_loss(noise_subspace=noise_subspace, angles=angles_pred, ranges=ranges)
        else:
            angles_pred, distances_pred, sources_estimation, eigen_regularization = self(x, sources_num)
            loss = self.train_loss(angles, angles_pred, ranges, distances_pred)
        if isinstance(loss, tuple):
            loss = loss[0]
        acc = self.source_estimation_accuracy(sources_num, sources_estimation)
        return loss, acc
        # if eigen_regularization is None:
        #     return loss, acc
        # else:
        #     if self.diff_method.estimation_params == "range":
        #         eigen_regularization = 0
        #     return loss + eigen_regularization * self.eigenregularization_weight, acc


    def validation_step(self, batch, batch_idx, is_test: bool=False):
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

        if is_test:
            _, loss_angle, loss_range = self.test_loss_separated(angles_pred, angles, ranges_pred, ranges)
            return (loss, loss_angle, loss_range), acc

        return loss, acc

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, is_test=True)

    def predict_step(self, batch, batch_idx):
        x = batch
        if x.dim() == 2:
            x = x.unsqueeze(0)
        angles, distances, sources_estimation, eigen_regularization = self.forward(x, self.system_model.params.M)
        return angles, distances, sources_estimation

    def __set_criterion(self):
        if self.train_loss_type == "music_spectrum":
            self.train_loss = MusicSpectrumLoss(
                array=torch.Tensor(self.system_model.array[:, None]).to(torch.float64).to(device),
                sensors_distance=self.system_model.dist_array_elems["narrowband"])
        else:
            if self.train_angle_extractor:
                self.train_loss = CartesianLoss()
            else:
                self.train_loss = RMSPELoss(balance_factor=0.0)
        self.validation_loss = CartesianLoss()
        self.test_loss = CartesianLoss()
        self.test_loss_separated = RMSPELoss(1.0)

    def update_criterion(self):
        self.__set_criterion()

    def update_angle_extractor_training(self, train_angle_extractor: bool):
        self.train_angle_extractor = train_angle_extractor
        self.__set_angle_extractor_requires_grad(train_angle_extractor)

    def __set_angle_extractor_requires_grad(self, requires_grad: bool):
        for param in self.angle_extractor.parameters():
            param.requires_grad = requires_grad
