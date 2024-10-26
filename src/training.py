"""
Subspace-Net

Details
----------
Name: training.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This code provides functions for training and simulating the Subspace-Net model.

Classes:
----------
- TrainingParams: A class that encapsulates the training parameters for the model.

Methods:
----------
- train: Function for training the model.
- train_model: Function for performing the training process.
- plot_learning_curve: Function for plotting the learning curve.
- simulation_summary: Function for printing a summary of the simulation parameters.

Attributes:
----------
None
"""
import warnings

from src.methods_pack.esprit import ESPRIT
from src.methods_pack.subspace_method import SubspaceMethod

EIGEN_REGULARIZATION_WEIGHT = 100
# Imports
import matplotlib.pyplot as plt
import copy
from pathlib import Path
import torch.optim as optim
from datetime import datetime
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
# internal imports
from src.utils import *
from src.criterions import *
from src.system_model import SystemModel, SystemModelParams
from src.models import (SubspaceNet, DeepCNN, DeepAugmentedMUSIC,
                        ModelGenerator, DCDMUSIC, TransMUSIC, DeepRootMUSIC)
from src.methods_pack.music import MUSIC
from src.evaluation import evaluate_dnn_model
from src.data_handler import TimeSeriesDataset, collate_fn, SameLengthBatchSampler


class TrainingParams(object):
    """
    A class that encapsulates the training parameters for the model.

    Methods
    -------
    - __init__: Initializes the TrainingParams object.
    - set_batch_size: Sets the batch size for training.
    - set_epochs: Sets the number of epochs for training.
    - set_model: Sets the model for training.
    - load_model: Loads a pre-trained model.
    - set_optimizer: Sets the optimizer for training.
    - set_schedular: Sets the scheduler for learning rate decay.
    - set_criterion: Sets the loss criterion for training.
    - set_training_dataset: Sets the training dataset for training.

    Raises
    ------
    Exception: If the model type is not defined.
    Exception: If the optimizer type is not defined.
    """

    def __init__(self):
        """
        Initializes the TrainingParams object.
        """
        self.criterion = None
        self.model = None
        self.diff_method = None
        self.tau = None
        self.model_type = None
        self.epochs = None
        self.batch_size = None
        self.training_objective = None

    def set_training_objective(self, training_objective: str):
        """

        Args:
            training_objective:

        Returns:

        """
        if training_objective.lower() == "angle":
            self.training_objective = "angle"
        elif training_objective.lower() == "range":
            self.training_objective = "range"
        elif training_objective.lower() == "angle, range":
            self.training_objective = "angle, range"
        elif training_objective.lower() == "source_estimation":
            self.training_objective = "source_estimation"
        else:
            raise Exception(f"TrainingParams.set_training_objective:"
                            f" Unrecognized training objective : {training_objective}.")
        return self

    def set_batch_size(self, batch_size: int):
        """
        Sets the batch size for training.

        Args
        ----
        - batch_size (int): The batch size.

        Returns
        -------
        self
        """
        self.batch_size = batch_size
        return self

    def set_epochs(self, epochs: int):
        """
        Sets the number of epochs for training.

        Args
        ----
        - epochs (int): The number of epochs.

        Returns
        -------
        self
        """
        self.epochs = epochs
        return self

    # TODO: add option to get a Model instance also
    def set_model(self, model_gen: ModelGenerator = None):
        """
        Sets the model for training.

        Args
        ----
        - model_gen (ModelGenerator): The system model object.

        Returns
        -------
        self

        Raises
        ------
        Exception: If the model type is not defined.
        """
        # assign model to device
        self.model = model_gen.model.to(device)
        return self

    def load_model(self, loading_path: Path):
        """
        Loads a pre-trained model.

        Args
        ----
        - loading_path (Path): The path to the pre-trained model.

        Returns
        -------
        self
        """
        # Load model from given path
        try:
            self.model.load_state_dict(torch.load(loading_path, map_location=device), strict=False)
        except FileNotFoundError as e:
            print(e)
            print(f"\nTrainingParams.load_model: Model not found in {loading_path}")
        return self

    def set_optimizer(self, optimizer: str, learning_rate: float, weight_decay: float):
        """
        Sets the optimizer for training.

        Args
        ----
        - optimizer (str): The optimizer type.
        - learning_rate (float): The learning rate.
        - weight_decay (float): The weight decay value (L2 regularization).

        Returns
        -------
        self

        Raises
        ------
        Exception: If the optimizer type is not defined.
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # Assign optimizer for training
        if optimizer.startswith("Adam"):
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer.startswith("SGD"):
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer == "SGD Momentum":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=learning_rate, momentum=0.9
            )
        else:
            raise Exception(
                f"TrainingParams.set_optimizer: Optimizer {optimizer} is not defined"
            )
        return self

    def set_schedular(self, step_size: float, gamma: float):
        """
        Sets the scheduler for learning rate decay.

        Args:
        ----------
        - step_size (float): Number of steps for learning rate decay iteration.
        - gamma (float): Learning rate decay value.

        Returns:
        ----------
        self
        """
        # Number of steps for learning rate decay iteration
        self.step_size = step_size
        # learning rate decay value
        self.gamma = gamma
        # Assign schedular for learning rate decay
        self.schedular = lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )
        return self

    def set_criterion(self, criterion: str, balance_factor: float = None):
        """
        Sets the loss criterion for training.

        Returns
        -------
        self
        """
        criterion = criterion.lower()
        # Define loss criterion
        if criterion.startswith("bce"):
            self.criterion = nn.BCELoss()
        elif criterion.startswith("mse"):
            self.criterion = nn.MSELoss()
        elif criterion.startswith("mspe"):
            self.criterion = MSPELoss()
        elif criterion.startswith("rmspe"):
            self.criterion = RMSPELoss(balance_factor=balance_factor)
        elif criterion.startswith("cartesian") and self.training_objective == "angle, range":
            self.criterion = CartesianLoss()
        elif criterion.startswith("ce") and self.training_objective == "source_estimation":
            self.criterion = nn.CrossEntropyLoss(reduction="sum")
        else:
            raise Exception(
                f"TrainingParams.set_criterion: criterion {criterion} is not defined"
            )
        return self

    def set_training_dataset(self, train_dataset: list):
        """
        Sets the training dataset for training.

        Args
        ----
        - train_dataset (list): The training dataset.

        Returns
        -------
        self
        """
        # Divide into training and validation datasets
        train_dataset, valid_dataset = train_test_split(
            train_dataset, test_size=0.1, shuffle=True
        )
        print("Training DataSet size", len(train_dataset))
        print("Validation DataSet size", len(valid_dataset))

        # init sampler
        batch_sampler_train = SameLengthBatchSampler(train_dataset, batch_size=self.batch_size)
        batch_sampler_valid = SameLengthBatchSampler(valid_dataset, batch_size=32, shuffle=False)
        # Transform datasets into DataLoader objects
        self.train_dataset = torch.utils.data.DataLoader(
            train_dataset,collate_fn=collate_fn, batch_sampler=batch_sampler_train
        )
        self.valid_dataset = torch.utils.data.DataLoader(
            valid_dataset,collate_fn=collate_fn, batch_sampler=batch_sampler_valid
        )
        # self.train_dataset = torch.utils.data.DataLoader(
        #     train_dataset, shuffle=True, batch_size=self.batch_size, drop_last=False
        # )
        # self.valid_dataset = torch.utils.data.DataLoader(
        #     valid_dataset, shuffle=False, batch_size=32, drop_last=True
        # )
        return self


def train(
        training_parameters: TrainingParams,
        plot_curves: bool = True,
        saving_path: Path = None,
        save_figures: bool = False,
):
    """
    Wrapper function for training the model.

    Args:
    ----------
    - training_params (TrainingParams): An instance of TrainingParams containing the training parameters.
    - model_name (str): The name of the model.
    - plot_curves (bool): Flag to indicate whether to plot learning and validation loss curves. Defaults to True.
    - saving_path (Path): The directory to save the trained model.

    Returns:
    ----------
    model: The trained model.
    loss_train_list: List of training loss values.
    loss_valid_list: List of validation loss values.

    Raises:
    ----------
    Exception: If the model type is not defined.
    Exception: If the optimizer type is not defined.
    """
    # Set the seed for all available random operations
    # set_unified_seed()
    # Current date and time
    print("\n----------------------\n")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    print("date and time =", dt_string)
    # Train the model
    train_res = train_model(training_parameters,
                            checkpoint_path=saving_path)
    model = train_res.get("model")
    loss_train_list = train_res.get("loss_train_list")
    loss_valid_list = train_res.get("loss_valid_list")
    loss_train_list_angles = train_res.get("loss_train_list_angles")
    loss_train_list_ranges = train_res.get("loss_train_list_ranges")
    loss_valid_list_angles = train_res.get("loss_valid_list_angles")
    loss_valid_list_ranges = train_res.get("loss_valid_list_ranges")
    acc_train_list = train_res.get("acc_train_list")
    acc_valid_list = train_res.get("acc_valid_list")
    figures_saving_path = Path(saving_path).parent / "simulations" / "results" / "plots"
    if plot_curves:
        if acc_train_list is not None and acc_valid_list is not None:
            fig_acc = plot_accuracy_curve(
                list(range(1, training_parameters.epochs + 1)), acc_train_list, acc_valid_list,
                model_name=model._get_name()
            )
            if save_figures:
                fig_acc.savefig(figures_saving_path / f"Accuracy_{model.get_model_name()}_{dt_string_for_save}.png")
            fig_acc.show()
        fig_loss = plot_learning_curve(
            list(range(1, training_parameters.epochs + 1)), loss_train_list, loss_valid_list,
            model_name=model._get_name(),
            angle_train_loss=loss_train_list_angles,
            angle_valid_loss=loss_valid_list_angles,
            range_train_loss=loss_train_list_ranges,
            range_valid_loss=loss_valid_list_ranges
        )
        if save_figures:
            fig_loss.savefig(figures_saving_path / f"Loss_{model.get_model_name()}_{dt_string_for_save}.png")
        fig_loss.show()

    # Save models best weights
    torch.save(model.state_dict(), saving_path / model.get_model_file_name())
    # Plot learning and validation loss curves
    return model, loss_train_list, loss_valid_list


def train_model(training_params: TrainingParams, checkpoint_path=None) -> dict:
    """
    Function for training the model.

    Args:
    -----
        training_params (TrainingParams): An instance of TrainingParams containing the training parameters.
        model_name (str): The name of the model.
        checkpoint_path (str): The path to save the checkpoint.

    Returns:
    --------
        model: The trained model.
        loss_train_list (list): List of training losses per epoch.
        loss_valid_list (list): List of validation losses per epoch.
    """
    # Initialize model and optimizer
    model = training_params.model
    model = model.to(device)
    optimizer = training_params.optimizer
    # Initialize losses
    loss_train_list = []
    loss_valid_list = []
    loss_train_list_angles = []
    loss_train_list_ranges = []
    loss_valid_list_angles = []
    loss_valid_list_ranges = []
    acc_train_list = []
    acc_valid_list = []
    min_valid_loss = np.inf
    # train_length = len(training_params.train_dataset)
    if isinstance(model, TransMUSIC):
        if training_params.training_objective == "source_estimation":
            transmusic_mode = "num_source_train"
        else:
            transmusic_mode = "subspace_train"
        model.update_train_mode(transmusic_mode)
    if isinstance(model, DCDMUSIC):
        if training_params.training_objective == "angle, range":
            model.training_angle_extractor = True
        elif training_params.training_objective == "range":
            model.training_angle_extractor = False
        model.update_criterion()

    # Set initial time for start training
    since = time.time()
    print("\n---Start Training Stage ---\n")
    # Run over all epochs
    for epoch in range(training_params.epochs):

        epoch_train_loss = 0.0
        epoch_train_loss_angle = 0.0
        epoch_train_loss_distance = 0.0
        epoch_train_acc = 0.0
        # init tmp loss values
        train_loss, train_loss_angle, train_loss_distance = None, None, None
        # Set model to train mode
        model.train()
        train_length = 0

        for idx, data in enumerate(tqdm((training_params.train_dataset))):
            if isinstance(model, (SubspaceNet, TransMUSIC)):
                train_loss, acc = model.training_step(data, idx)
                if isinstance(train_loss, tuple):
                    train_loss, train_loss_angle, train_loss_distance = train_loss
                    epoch_train_loss_angle += train_loss_angle.item()
                    epoch_train_loss_distance += train_loss_distance.item()
                epoch_train_loss += train_loss.item()

                train_length += data[0].shape[0]
                epoch_train_acc += acc
            else:
                raise NotImplementedError(f"train_model: Training for model {model.get_model_name()} is not implemented")

            ############################################################################################################
            # Back-propagation stage
            try:
                train_loss.backward(retain_graph=True)
            except RuntimeError as r:
                raise Exception(f"linalg error: \n{r}")

            # optimizer update
            optimizer.step()
            # reset gradients
            model.zero_grad()

        ################################################################################################################
        epoch_train_loss /= train_length
        epoch_train_acc /= train_length

        # End of epoch. Calculate the average loss
        loss_train_list.append(epoch_train_loss)
        if epoch_train_loss_angle != 0.0 and epoch_train_loss_distance != 0.0:
            loss_train_list_angles.append(epoch_train_loss_angle / train_length)
            loss_train_list_ranges.append(epoch_train_loss_distance / train_length)
        # Update schedular
        training_params.schedular.step()
        try:
            model.update_eigenregularization_weight()
        except AttributeError:
            pass

        # Calculate evaluation loss
        valid_loss = evaluate_dnn_model(
            model,
            training_params.valid_dataset,
            training_params.criterion,
        )
        loss_valid_list.append(valid_loss.get("Overall"))

        # Report results
        result_txt = (f"[Epoch : {epoch + 1}/{training_params.epochs}]"
                      f" Train loss = {epoch_train_loss:.6f}, Validation loss = {valid_loss.get('Overall'):.6f}")

        if valid_loss.get("Angle") is not None and valid_loss.get("Distance") is not None:
            loss_valid_list_angles.append(valid_loss.get("Angle"))
            loss_valid_list_ranges.append(valid_loss.get("Distance"))
            result_txt += f"\nAngle loss = {valid_loss.get('Angle'):.6f}, Range loss = {valid_loss.get('Distance'):.6f}"
        acc_train_list.append(epoch_train_acc * 100)
        acc_valid_list.append(valid_loss.get('Accuracy') * 100)
        result_txt += (f"\nAccuracy for sources estimation: Train = {100 * epoch_train_acc:.2f}%, "
                       f"Validation = {valid_loss.get('Accuracy') * 100:.2f}%")
        result_txt += f"\nlr {training_params.schedular.get_last_lr()[0]}"

        print(result_txt)
        # Save best model weights
        if min_valid_loss > valid_loss.get("Overall"):
            print(
                f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss.get('Overall'):.6f}) \t Saving The Model"
            )
            min_valid_loss = valid_loss.get("Overall")
            best_epoch = epoch
            # Saving State Dict
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), checkpoint_path / model.get_model_file_name())
        if isinstance(model, SubspaceNet):
            model.adjust_diff_method_temperature(epoch)
    # Training complete
    time_elapsed = time.time() - since
    print("\n--- Training summary ---")
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Minimal Validation loss: {min_valid_loss:4f} at epoch {best_epoch}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), checkpoint_path / model.get_model_file_name())
    res = {"model": model, "loss_train_list": loss_train_list, "loss_valid_list": loss_valid_list}
    if len(loss_train_list_angles) > 0 and len(loss_train_list_ranges) > 0:
        res["loss_train_list_angles"] = loss_train_list_angles
        res["loss_train_list_ranges"] = loss_train_list_ranges
        res["loss_valid_list_angles"] = loss_valid_list_angles
        res["loss_valid_list_ranges"] = loss_valid_list_ranges
    if len(acc_train_list) > 0 and len(acc_valid_list) > 0:
        res["acc_train_list"] = acc_train_list
        res["acc_valid_list"] = acc_valid_list
    return res


def plot_accuracy_curve(epoch_list, train_acc: list, validation_acc: list, model_name: str = None):
    """
    Plot the learning curve.

    Args:
    -----
        epoch_list (list): List of epochs.
        train_loss (list): List of training losses per epoch.
        validation_loss (list): List of validation losses per epoch.
    """
    figure = plt.figure(figsize=(10, 6))
    title = "Learning Curve: Accuracy per Epoch"
    if model_name is not None:
        title += f" {model_name}"
    plt.title(title)
    plt.plot(epoch_list, train_acc, label="Train")
    plt.plot(epoch_list, validation_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    return figure


def plot_learning_curve(epoch_list, train_loss: list, validation_loss: list, model_name: str = None,
                        angle_train_loss=None, angle_valid_loss=None, range_train_loss=None, range_valid_loss=None):
    """
    Plot the learning curve.

    Args:
    -----
        epoch_list (list): List of epochs.
        train_loss (list): List of training losses per epoch.
        validation_loss (list): List of validation losses per epoch.
    """
    title = "Learning Curve: Loss per Epoch"
    if model_name is not None:
        title += f" {model_name}"
    if angle_train_loss is not None and range_train_loss is not None:

        # create 3 subplots, the main one will spread over 2 cols, and the other 2 will be under it.
        fig = plt.figure(figsize=(10, 6))
        ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax_angle = plt.subplot2grid((2, 2), (1, 0), colspan=1)
        ax_range = plt.subplot2grid((2, 2), (1, 1), colspan=1)

        ax.set_title(title)
        ax.plot(epoch_list, train_loss, label="Train")
        ax.plot(epoch_list, validation_loss, label="Validation")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend(loc="best")
        ax_angle.plot(epoch_list, angle_train_loss, label="Train")
        ax_angle.plot(epoch_list, angle_valid_loss, label="Validation")
        ax_angle.set_xlabel("Epochs")
        ax_angle.set_ylabel("Angle Loss [rad]")
        ax_angle.legend(loc="best")
        ax_range.plot(epoch_list, range_train_loss, label="Train")
        ax_range.plot(epoch_list, range_valid_loss, label="Validation")
        ax_range.set_xlabel("Epochs")
        ax_range.set_ylabel("Range Loss [m]")
        ax_range.legend(loc="best")
        # tight layout
        plt.tight_layout()
    else:
        fig = plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.plot(epoch_list, train_loss, label="Train")
        plt.plot(epoch_list, validation_loss, label="Validation")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="best")
    return fig


def simulation_summary(
        system_model_params: SystemModelParams,
        model_type: str,
        parameters: TrainingParams = None,
        phase="training",
):
    """
    Prints a summary of the simulation parameters.

    Parameters
    ----------
    system_model_params
    model_type
    parameters
    phase

    """
    if system_model_params.M is None:
        M = "Random"
    else:
        M = system_model_params.M
    print("\n--- New Simulation ---\n")
    print(f"Description: Simulation of {model_type}, {phase} stage")
    print("System model parameters:")
    print(f"Number of sources = {M} {system_model_params.signal_nature}")
    print(f"Number of sensors = {system_model_params.N}")
    print(f"field_type = {system_model_params.field_type}")
    print(f"signal_type = {system_model_params.signal_type}")
    print(f"Observations = {system_model_params.T}")
    print(f"SNR = {system_model_params.snr}")
    print(f"Spacing deviation (eta) = {system_model_params.eta}")
    print(f"Bias spacing deviation (eta) = {system_model_params.bias}")
    print(f"Geometry noise variance = {system_model_params.sv_noise_var}")
    print("Simulation parameters:")
    print(f"Model: {model_type}")
    print(f"Model parameters:", *[f" {key} = {val} |" for key,val in parameters.model.get_model_params().items() if val is not None])
    if phase.startswith("training"):
        print("Training parameters:")
        print(f"Epochs = {parameters.epochs}")
        print(f"Batch Size = {parameters.batch_size}")
        print(f"Learning Rate = {parameters.learning_rate}")
        print(f"Weight decay = {parameters.weight_decay}")
        print(f"Gamma Value = {parameters.gamma}")
        print(f"Step Value = {parameters.step_size}")


def get_simulation_filename(
        system_model_params: SystemModelParams, model_config: ModelGenerator
):
    """

    Parameters
    ----------
    system_model_params
    model_config

    Returns
    -------
    File name to a simulation ran.
    """
    return (
        f"{model_config.model.get_model_name()}_"
        f"N={system_model_params.N}_"
        f"M={system_model_params.M}_"
        f"T={system_model_params.T}_"
        f"SNR_{system_model_params.snr}_"
        f"{system_model_params.signal_type}_"
        f"{system_model_params.field_type}_field_"
        f"{system_model_params.signal_nature}"
        f"_eta={system_model_params.eta}_"
        f"bias={system_model_params.bias}_"
        f"sv_noise={system_model_params.sv_noise_var}"
    )


def get_model_filename(system_model_params: SystemModelParams, model_name: str):
    """

    Parameters
    ----------
    system_model_params
    model_config

    Returns
    -------
    file name to the wieghts of a network.
    different from get_simulation_filename by not considering parameters that are not relevant to the network itself.
    """
    if model_name.lower() == "DCDMUSIC":
        return (
                f"{model_name}_"
                + f"N={system_model_params.N}_"
                + f"tau=8_"
                + f"M={system_model_params.M}_"
                + f"{system_model_params.signal_type}_"
                + f"SNR={system_model_params.snr}_"
                + f"diff_method=music_1D_"
                + f"{system_model_params.field_type}_field_"
                + f"{system_model_params.signal_nature}"
        )
    else:
        return (
                f"{model_name}_"
                + f"N={system_model_params.N}_"
                + f"tau={model_config.tau}_"
                + f"M={system_model_params.M}_"
                + f"{system_model_params.signal_type}_"
                + f"SNR={system_model_params.snr}_"
                + f"diff_method={model_config.diff_method}_"
                + f"{system_model_params.field_type}_field_"
                + f"{system_model_params.signal_nature}"
        )
