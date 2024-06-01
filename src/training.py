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

# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import copy
from pathlib import Path
import torch.optim as optim
from datetime import datetime
from torch.autograd import Variable
from tqdm import tqdm
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from src.utils import *
from src.criterions import *
from src.system_model import SystemModel, SystemModelParams
from src.models import SubspaceNet, DeepCNN, DeepAugmentedMUSIC, ModelGenerator, CascadedSubspaceNet, TransMUSIC
from src.evaluation import evaluate_dnn_model


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
            print("#" * 40 + "Nothing will be loaded" + "#" * 40)
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

    def set_criterion(self, criterion:str, balance_factor: float = None):
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
        # Transform datasets into DataLoader objects
        self.train_dataset = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        self.valid_dataset = torch.utils.data.DataLoader(
            valid_dataset, batch_size=32, shuffle=False, drop_last=True
        )
        return self


def train(
        training_parameters: TrainingParams,
        model_name: str,
        plot_curves: bool = True,
        saving_path: Path = None,
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
    set_unified_seed()
    # Current date and time
    print("\n----------------------\n")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    print("date and time =", dt_string)
    # Train the model
    model, loss_train_list, loss_valid_list = train_model(
        training_parameters, model_name=model_name, checkpoint_path=saving_path
    )
    # Save models best weights
    torch.save(model.state_dict(), saving_path / model.get_model_file_name())
    # Plot learning and validation loss curves
    if plot_curves:
        plot_learning_curve(
            list(range(training_parameters.epochs)), loss_train_list, loss_valid_list, model_name=model.get_model_name())
    return model, loss_train_list, loss_valid_list


def train_model(training_params: TrainingParams, model_name: str, checkpoint_path=None):
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
    optimizer = training_params.optimizer
    # Initialize losses
    loss_train_list = []
    loss_valid_list = []
    min_valid_loss = np.inf
    # Set initial time for start training
    since = time.time()
    training_angle_extractor = False
    if training_params.training_objective.endswith("angle, range"):
        training_angle_extractor = True
    print("\n---Start Training Stage ---\n")
    # Run over all epochs
    for epoch in range(training_params.epochs):
        # if isinstance(model, CascadedSubspaceNet):
        #     if epoch == int(training_params.epochs * 0.9):
        #         training_angle_extractor = not training_angle_extractor
        #         if training_angle_extractor:
        #             print("Switching to training angle extractor")
        #             if isinstance(training_params.criterion, RMSPELoss):
        #                 training_params.criterion.adjust_balance_factor()
        #         else:
        #             print("turn off training angle extractor")
        train_length = 0
        overall_train_loss = 0.0
        overall_train_loss_angle = 0.0
        overall_train_loss_distance = 0.0
        # Set model to train mode
        model.train()
        model = model.to(device)
        for data in tqdm(training_params.train_dataset):
            x, true_label = data
            if training_params.training_objective.lower() == "angle, range" or training_params.training_objective.lower() == "range":
                DOA, RANGE = torch.split(true_label, true_label.size(1) // 2, dim=1)
                RANGE = Variable(RANGE, requires_grad=True).to(device)
            elif training_params.training_objective.lower() == "angle":
                if model.system_model.params.field_type.lower() == "near":
                    # if the data_model and the model are not synced.
                    DOA, _ = torch.split(true_label, true_label.size(1) // 2, dim=1)
                else:
                    DOA = true_label

            train_length += DOA.shape[0]
            # Cast observations and DoA to Variables
            x = Variable(x, requires_grad=True).to(device)
            DOA = Variable(DOA, requires_grad=True).to(device)
            # Get model output
            # This if condition is mainly to spearte the case which we want to train the angle extractor.
            if isinstance(model, CascadedSubspaceNet):
                model_output = model(x, train_angle_extractor=training_angle_extractor)
            else:
                model_output = model(x)
            if isinstance(model, SubspaceNet):
                # in this case there are 2 labels - angles and distances.
                if isinstance(model, CascadedSubspaceNet) or training_params.training_objective == "angle, range":
                    DOA_predictions = model_output[0]
                    RANGE_predictions = model_output[1]
                elif training_params.training_objective.endswith("angle"):
                    DOA_predictions = model_output[0]
            elif isinstance(model, TransMUSIC):
                if training_params.training_objective == "angle":
                    DOA_predictions = model_output[0]
                    prob_source_number = model_output[1]
                elif training_params.training_objective == "angle, range":
                    DOA_predictions, RANGE_predictions = model_output[0][:, 0], model_output[0][:, 1]
                    prob_source_number = model_output[1]
                estimated_source_number = torch.argmax(prob_source_number, dim=1)

            else:
                # Deep Augmented MUSIC or DeepCNN
                DOA_predictions = model_output

            # Compute training loss
            if isinstance(model, DeepCNN):
                train_loss = training_params.criterion(DOA_predictions.float(), DOA.float())
            elif isinstance(model, TransMUSIC):
                DOA_predictions = DOA_predictions[:, :DOA.shape[1]]
                if training_params.training_objective == "angle":
                    train_loss = training_params.criterion(DOA_predictions, DOA)
                elif training_params.training_objective == "angle, range":
                    RANGE_predictions = RANGE_predictions[:, :RANGE.shape[1]]
                    train_loss = training_params.criterion(DOA_predictions, DOA, RANGE_predictions, RANGE)
            elif isinstance(model, SubspaceNet):
                if training_params.training_objective == "angle":
                    train_loss = training_params.criterion(DOA_predictions, DOA)
                elif training_params.training_objective == "range":
                    train_loss = training_params.criterion(DOA_predictions, DOA, RANGE_predictions, RANGE)
                elif training_params.training_objective == "angle, range":
                    if isinstance(training_params.criterion, RMSPELoss):
                        train_loss, train_loss_angle, train_loss_distance = training_params.criterion(DOA_predictions,
                                                                                                      DOA,
                                                                                                      RANGE_predictions,
                                                                                                      RANGE,
                                                                                                      is_separted=True)
                    elif isinstance(training_params.criterion, CartesianLoss):
                        train_loss = training_params.criterion(DOA_predictions,
                                                               DOA,
                                                               RANGE_predictions,
                                                               RANGE)
            else:
                raise Exception(f"Model type {training_params.model_type} is not defined")
            # Back-propagation stage
            try:
                train_loss.backward(retain_graph=True)
            except RuntimeError as r:
                raise Exception(f"linalg error: \n{r}")

            # optimizer update
            optimizer.step()
            # reset gradients
            model.zero_grad()
            # add batch loss to overall epoch loss
            if isinstance(training_params.criterion, nn.BCELoss):
                # BCE is averaged
                overall_train_loss += train_loss.item() * len(data[0])
            elif isinstance(training_params.criterion, RMSPELoss) or isinstance(training_params.criterion, CartesianLoss):
                # RMSPE is averaged over the dataset size
                overall_train_loss += train_loss.item() / len(training_params.train_dataset)
                # overall_train_loss_angle += train_loss_angle.item() / len(training_params.train_dataset)
                # overall_train_loss_distance += train_loss_distance.item() / len(training_params.train_dataset)
            else:
                raise Exception(f"Criterion type {training_params.criterion} is not defined")
        # add epoch loss to the list
        loss_train_list.append(overall_train_loss)
        # Update schedular
        training_params.schedular.step()


        # Calculate evaluation loss
        valid_loss = evaluate_dnn_model(
            model,
            training_params.valid_dataset,
            training_params.criterion,
            model_type=training_params.model_type,
        )
        loss_valid_list.append(valid_loss)
        # Report results
        print(
            "epoch : {}/{}, Train loss = {:.6f}, Validation loss = {:.6f}".format(
                epoch + 1, training_params.epochs, overall_train_loss, valid_loss
            )
        )
        print("lr {}".format(training_params.optimizer.param_groups[0]["lr"]))
        # Save best model weights for early stoppings
        if min_valid_loss > valid_loss:
            print(
                f"Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model"
            )
            min_valid_loss = valid_loss
            best_epoch = epoch
            # Saving State Dict
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), checkpoint_path / model.get_model_file_name())
        if isinstance(model, SubspaceNet):
            model.adjust_diff_method_temperature(epoch)
        # if isinstance(training_params.criterion, RMSPELoss):
        #     training_params.criterion.adjust_balance_factor(overall_train_loss)
    # Training complete
    time_elapsed = time.time() - since
    print("\n--- Training summary ---")
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print(
        "Minimal Validation loss: {:4f} at epoch {}".format(min_valid_loss, best_epoch)
    )

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), checkpoint_path / model.get_model_file_name())
    return model, loss_train_list, loss_valid_list


def plot_learning_curve(epoch_list, train_loss: list, validation_loss: list, model_name: str = None):
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
    plt.title(title)
    plt.plot(epoch_list, train_loss, label="Train")
    plt.plot(epoch_list, validation_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    # plt.show()


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
    print("\n--- New Simulation ---\n")
    print(f"Description: Simulation of {model_type}, {phase} stage")
    print("System model parameters:")
    print(f"Number of sources = {system_model_params.M}")
    print(f"Number of sensors = {system_model_params.N}")
    print(f"field_type = {system_model_params.field_type}")
    print(f"signal_type = {system_model_params.signal_type}")
    print(f"Observations = {system_model_params.T}")
    print(
        f"SNR = {system_model_params.snr}, {system_model_params.signal_nature} sources"
    )
    print(f"Spacing deviation (eta) = {system_model_params.eta}")
    print(f"Bias spacing deviation (eta) = {system_model_params.bias}")
    print(f"Geometry noise variance = {system_model_params.sv_noise_var}")
    print("Simulation parameters:")
    print(f"Model: {model_type}")
    print(f"Model parameters: {parameters.model.get_model_params()}")
    if phase.startswith("training"):
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
            f"{model_config.model.get_model_params()}_"
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
    if model_name.lower() == "cascadedsubspacenet":
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
