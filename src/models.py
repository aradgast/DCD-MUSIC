"""Subspace-Net 
Details
----------
Name: models.py
Authors: Dor Haim Shmuel, Arad Gast
Created: 01/10/21
Edited: 29/05/24
"""

# Imports
import warnings
# Internal Imports
from src.system_model import SystemModel, SystemModelParams
from src.models_pack.trans_music import TransMUSIC
from src.models_pack.subspacenet import SubspaceNet
from src.models_pack.dcd_music import DCDMUSIC
from src.models_pack.deep_augmented_music import DeepAugmentedMUSIC
from src.models_pack.deep_cnn import DeepCNN
from src.models_pack.deep_root_music import DeepRootMUSIC

# warnings.simplefilter("ignore")


class ModelGenerator(object):
    """
    Generates an instance of the desired model, according to model configuration parameters.
    Attributes:
            model_type(str): The network type
            system_model_params(SystemModelParams): the system model parameters.
            model_params(dict): The parameters for the model.
            model(nn.Module) : An instance of the model
    """

    def __init__(self):
        """
        Initialize ModelParams object.
        """
        self.model_type: str = str(None)
        self.system_model: SystemModel = None
        self.model_params: dict = None

    def set_model_type(self, model_type: str):
        """
        Set the model type.

        Parameters:
            model_type (str): The model type.

        Returns:
            ModelGenerator: The updated ModelGenerator object.

        Raises:
            ValueError: If model type is not provided.
        """
        if not isinstance(model_type, str):
            raise ValueError(
                "ModelGenerator.set_model_type: model type has not been provided"
            )
        self.model_type = model_type
        return self

    def set_system_model(self, system_model: SystemModel):
        """
        Set the system model.

        Parameters:
            system_model (SystemModel): The system model.

        Returns:
            ModelGenerator: The updated ModelParams object.

        Raises:
            ValueError: If system_model is not provided.
        """
        if not isinstance(system_model, SystemModel):
            raise ValueError(
                "ModelGenerator.set_system_model: system model params has not been provided"
            )
        self.system_model = system_model
        return self

    def set_model_params(self, model_params: dict):
        """
        Set the model params.

        Parameters:
            model_params (dict): The model's parameters.

        Returns:
            ModelGenerator: The updated ModelParams object.

        Raises:
            ValueError: If model type is not provided.
        """
        if not isinstance(model_params, dict):
            raise ValueError(
                "ModelGenerator.set_model_params: model params has not been provided"
            )
        # verify params for model.
        try:
            self.__verify_model_params(model_params)
        except Exception as e:
            print(e)
        self.model_params = model_params
        return self

    def set_model(self):
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
            self.__set_da_music()
        elif self.model_type.startswith("DR_MUSIC"):
            self.__set_dr_music()
        elif self.model_type.startswith("DeepCNN"):
            self.__set_deepcnn()
        elif self.model_type.startswith("SubspaceNet"):
            self.__set_subspacenet()
        elif self.model_type.startswith("DCDMUSIC"):
            self.__set_dcd_music()
        elif self.model_type.startswith("TransMUSIC"):
            self.__set_transmusic()
        else:
            raise Exception(f"ModelGenerator.set_model: Model type {self.model_type} is not defined")

        return self

    def __set_subspacenet(self):
        """

        """
        tau = self.model_params.get("tau")
        diff_method = self.model_params.get("diff_method")
        field_type = self.model_params.get("field_type")
        self.model = SubspaceNet(tau=tau,
                                 diff_method=diff_method,
                                 system_model=self.system_model,
                                 field_type=field_type)

    def __set_dcd_music(self):
        """

        """
        tau = self.model_params.get("tau")
        self.model = DCDMUSIC(tau=tau,
                              system_model=self.system_model)

    def __set_transmusic(self):
        """

        """
        self.model = TransMUSIC(system_model=self.system_model)

    def __set_deepcnn(self):
        """

        """
        N = self.system_model.params.N
        grid_size = self.model_params.get("grid_size")
        self.model = DeepCNN(N=N, grid_size=grid_size)

    def __set_da_music(self):
        """

        """
        N = self.system_model.params.N
        T = self.system_model.params.T
        M = self.system_model.params.M
        self.model = DeepAugmentedMUSIC(N=N, T=T, M=M)

    def __set_dr_music(self):
        """

        """
        tau = self.model_params.get("tau")
        activation_val = self.model_params.get("activation_value")
        self.model = DeepRootMUSIC(tau=tau, activation_value=activation_val)

    def __verify_model_params(self, model_params):
        """
        There are different models, and each one has different set of parameters to set.
        This function just verify the correctness of the params depending on the model.
        """

        if self.model_type.lower() == "subspacenet":
            self.__verify_subspacenet_params(model_params)
        elif self.model_type.lower() == "dcdmusic":
            self.__verify_dcdmuisc_params(model_params)
        elif self.model_type.lower() == "transmusic":
            self.__verify_transmusic_params(model_params)
        else:
            raise ValueError(f"ModelGenerator.__verify_model_params:"
                             f" currently there is no verification support for {self.model_type}")

    def __verify_transmusic_params(self, model_params):
        """

        """
        pass

    def __verify_subspacenet_params(self, model_params):
        """
        tau: int, diff_method: str = "root_music", field_type: str = "Far"
        """
        tau = model_params.get("tau")
        if not isinstance(tau, int) or not (tau < self.system_model.params.T):
            raise ValueError(f"ModelGenerator.__verify_subspacenet_params:"
                             f" Tau has to be an int and smaller than T")

        field_type = model_params.get("field_type")
        if not isinstance(field_type, str) or not (field_type.lower() in ["far", "near"]):
            raise ValueError(f"ModelGenerator.__verify_subspacenet_params:"
                             f"field_type has to be a str and the possible values are far or near.")

        diff_method = model_params.get("diff_method")
        if isinstance(diff_method, str):
            if field_type.lower() == "far":
                if not (diff_method.lower() in ["root_music", "esprit", "music_1d"]):
                    raise ValueError(f"ModelGenerator.__verify_subspacenet_params:"
                                     f"for Far field possible diff methods are: root_music, esprit or music_1d")
            else:  # field_type.lower() == "near":
                if not (diff_method.lower() in ["music_1d", "music_2d"]):
                    raise ValueError(f"ModelGenerator.__verify_subspacenet_params:"
                                     f"for Near field possible diff methods are: music_2d or music_1d")
        else:
            raise ValueError(f"ModelGenerator.__verify_subspacenet_params:"
                             f"field type was not given as a model param.")

    def __verify_dcdmuisc_params(self, model_params):
        """
        tau: int
        """
        tau = model_params.get("tau")
        if not isinstance(tau, int) or not (tau < self.system_model.params.T):
            raise ValueError(f"ModelGenerator.__verify_dcdmuisc_params:"
                             f" Tau has to be an int and smaller than T")

    def __str__(self):
        return f"{self.model.get_model_name()}"