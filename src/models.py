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
# Imports
import warnings


from src.system_model import SystemModel, SystemModelParams
from src.models_pack.trans_music import TransMUSIC
from src.models_pack.subspacenet import SubspaceNet
from src.models_pack.cascaded_subspacenet import CascadedSubspaceNet
from src.models_pack.deep_augmented_music import DeepAugmentedMUSIC
from src.models_pack.deep_cnn import DeepCNN


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
                f"{self.model_type}"
                + f"N={system_model_params.N}_"
                + f"tau={self.tau}_"
                + f"M={system_model_params.M}_"
                + f"{system_model_params.signal_type}_"
                + f"SNR={system_model_params.snr}_"
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
        elif self.model_type.startswith("TransMUSIC"):
            self.model = TransMUSIC(system_model=self.system_model)
        else:
            raise Exception(f"ModelGenerator.set_model: Model type {self.model_type} is not defined")

        return self

