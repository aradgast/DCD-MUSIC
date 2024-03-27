"""
Subspace-Net

Details
----------
Name: evaluation.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This module provides functions for evaluating the performance of Subspace-Net and others Deep learning benchmarks,
add for conventional subspace methods. 
This scripts also defines function for plotting the methods spectrums.
In addition, 


Functions:
----------
evaluate_dnn_model: Evaluate the DNN model on a given dataset.
evaluate_augmented_model: Evaluate an augmented model that combines a SubspaceNet model.
evaluate_model_based: Evaluate different model-based algorithms on a given dataset.
add_random_predictions: Add random predictions if the number of predictions
    is less than the number of sources.
evaluate: Wrapper function for model and algorithm evaluations.


"""
import numpy as np
import torch
# Imports
import torch.nn as nn
from matplotlib import pyplot as plt
from src.utils import device
from src.criterions import RMSPELoss, MSPELoss, RMSELoss
from src.criterions import RMSPE, MSPE
from src.methods import MUSIC, RootMUSIC, Esprit, MVDR, MUSIC_2D
from src.utils import *
from src.models import SubspaceNet, MUSIC, RootMusic, Esprit
from src.plotting import plot_spectrum
from src.system_model import SystemModel


def get_model_based_method(method_name: str, system_model: SystemModel):
    """

    Parameters
    ----------
    method_name

    Returns
    -------

    """
    if method_name.lower() == "music_1d":
        return MUSIC(system_model=system_model, estimation_parameter="angle")
    if method_name.lower() == "music_2d":
        return MUSIC(system_model=system_model, estimation_parameter="angle, range")
    if method_name.lower() == "root_music":
        return RootMusic(system_model)
    if method_name.lower() == "esprit":
        return Esprit(system_model)
def evaluate_dnn_model(
        model,
        dataset: list,
        criterion: nn.Module,
        plot_spec: bool = False,
        figures: dict = None,
        model_type: str = "SubspaceNet",
        is_separted: bool = False):
    """
    Evaluate the DNN model on a given dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        dataset (list): The evaluation dataset.
        criterion (nn.Module): The loss criterion for evaluation.
        plot_spec (bool, optional): Whether to plot the spectrum for SubspaceNet model. Defaults to False.
        figures (dict, optional): Dictionary containing figure objects for plotting. Defaults to None.
        model_type (str, optional): The type of the model. Defaults to "SubspaceNet".

    Returns:
        float: The overall evaluation loss.

    Raises:
        Exception: If the loss criterion is not defined for the specified model type.
        Exception: If the model type is not defined.
    """

    # Initialize values
    overall_loss = 0.0
    overall_loss_angle = 0.0
    overall_loss_distance = 0.0
    test_length = 0
    # Set model to eval mode
    model.eval()
    # Gradients calculation isn't required for evaluation

    with torch.no_grad():
        for data in dataset:
            X, true_label = data
            if model_type.startswith("SubspaceNet"):
                if model.system_model.params.field_type.endswith("Near"):
                    DOA, RANGE = torch.split(true_label, true_label.size(1) // 2, dim=1)
                    RANGE.to(device)
                else:
                    DOA = true_label
            else:
                DOA = true_label
            test_length += DOA.shape[0]
            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Get model output
            if model_type.startswith("SubspaceNet") and model.system_model.params.field_type.endswith("Near"):
                if model.diff_method.estimation_params == "range":
                    model_output = model(X, is_soft=False, known_angles=DOA)
                else:
                    model_output = model(X, is_soft=False)
            else:
                model_output = model(X)
            if model_type.startswith("DA-MUSIC"):
                # Deep Augmented MUSIC
                DOA_predictions = model_output
            elif model_type.startswith("DeepCNN"):
                # Deep CNN
                if isinstance(criterion, nn.BCELoss):
                    # If evaluation performed over validation set, loss is BCE
                    DOA_predictions = model_output
                    # find peaks in the pseudo spectrum of probabilities
                    DOA_predictions = (
                            get_k_peaks(361, DOA.shape[1], DOA_predictions[0]) * D2R
                    )
                    DOA_predictions = DOA_predictions.view(1, DOA_predictions.shape[0])
                elif isinstance(criterion, [RMSPELoss, MSPELoss]):
                    # If evaluation performed over testset, loss is RMSPE / MSPE
                    DOA_predictions = model_output
                else:
                    raise Exception(
                        f"evaluate_dnn_model: Loss criterion is not defined for {model_type} model"
                    )
            elif model_type.startswith("SubspaceNet"):
                if model.system_model.params.field_type.endswith("Near"):
                    if model.diff_method.estimation_params == "range":
                        RANGE_predictions = model_output[0]
                        DOA_predictions = DOA
                    else:
                        RANGE_predictions = model_output[1]
                        DOA_predictions = model_output[0]
                elif model.system_model.params.field_type.endswith("Far"):
                    # Default - SubSpaceNet
                    DOA_predictions = model_output[0]
            else:
                raise Exception(
                    f"evaluate_dnn_model: Model type {model_type} is not defined"
                )
            # Compute prediction loss
            if model_type.startswith("DeepCNN") and isinstance(criterion, RMSPELoss):
                eval_loss = criterion(DOA_predictions.float(), DOA.float())
            else:
                if model.system_model.params.field_type.endswith("Near"):
                    if model.diff_method.angels is None:
                        eval_loss = criterion(RANGE.to(torch.float64), RANGE_predictions)
                    else:
                        eval_loss = criterion(DOA_predictions, DOA, RANGE_predictions, RANGE, is_separted)
                        if is_separted:
                            eval_loss, eval_loss_angle, eval_loss_distance = eval_loss
                            overall_loss_angle += eval_loss_angle.item() / len(dataset)
                            overall_loss_distance += eval_loss_distance.item() / len(dataset)
                else:
                    eval_loss = criterion(DOA_predictions, DOA)
            # add the batch evaluation loss to epoch loss
            overall_loss += eval_loss.item() / len(dataset)

    # Plot spectrum for SubspaceNet model
    if plot_spec and model_type.startswith("SubspaceNet"):
        DOA_all = model_output[1]
        roots = model_output[2]
        plot_spectrum(
            predictions=DOA_all * R2D,
            true_DOA=DOA[0] * R2D,
            roots=roots,
            algorithm="SubNet+R-MUSIC",
            figures=figures,
        )
    if is_separted:
        overall_loss = {"Overall":  overall_loss,
                        "Angle":    overall_loss_angle,
                        "Distance": overall_loss_distance}
    return overall_loss


def evaluate_augmented_model(
        model: SubspaceNet,
        dataset,
        system_model,
        criterion=RMSPE,
        algorithm: str = "music",
        plot_spec: bool = False,
        figures: dict = None,
):
    """
    Evaluate an augmented model that combines a SubspaceNet model with another subspace method on a given dataset.

    Args:
    -----
        model (nn.Module): The trained SubspaceNet model.
        dataset: The evaluation dataset.
        system_model (SystemModel): The system model for the hybrid algorithm.
        criterion: The loss criterion for evaluation. Defaults to RMSPE.
        algorithm (str): The hybrid algorithm to use (e.g., "music", "mvdr", "esprit"). Defaults to "music".
        plot_spec (bool): Whether to plot the spectrum for the hybrid algorithm. Defaults to False.
        figures (dict): Dictionary containing figure objects for plotting. Defaults to None.

    Returns:
    --------
        float: The average evaluation loss.

    Raises:
    -------
        Exception: If the algorithm is not supported.
        Exception: If the algorithm is not supported
    """
    # Initialize parameters for evaluation
    hybrid_loss = []
    if not isinstance(model, SubspaceNet):
        raise Exception("evaluate_augmented_model: model is not from type SubspaceNet")
    # Set model to eval mode
    model.eval()
    # Initialize instances of subspace methods
    methods = {
        "mvdr": MVDR(system_model),
        "music": MUSIC(system_model, estimation_parameter="angle"),
        "esprit": Esprit(system_model),
        "r-music": RootMUSIC(system_model),
        "music_2D": MUSIC(system_model, estimation_parameter="angle, range")
    }
    # If algorithm is not in methods
    if methods.get(algorithm) is None:
        raise Exception(
            f"evaluate_augmented_model: Algorithm {algorithm} is not supported."
        )
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            X, true_label = data
            if algorithm.endswith("2D"):
                DOA, RANGE = torch.split(true_label, true_label.size(1) // 2, dim=1)
                RANGE.to(device)
            else:
                DOA = true_label

            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Apply method with SubspaceNet augmentation
            method_output = methods[algorithm].narrowband(
                X=X, mode="SubspaceNet", model=model
            )
            # Calculate loss, if algorithm is "music" or "esprit"
            if not algorithm.startswith("mvdr"):
                if algorithm.endswith("2D"):
                    predictions_doa, predictions_distance = method_output[0], method_output[1]
                    loss = criterion(predictions_doa, DOA * R2D, predictions_distance, RANGE)
                else:
                    predictions, M = method_output[0], method_output[-1]
                    # If the amount of predictions is less than the amount of sources
                    predictions = add_random_predictions(M, predictions, algorithm)
                    # Calculate loss criterion
                    loss = criterion(predictions, DOA * R2D)
                hybrid_loss.append(loss)
            else:
                hybrid_loss.append(0)
            # Plot spectrum, if algorithm is "music" or "mvdr"
            if not algorithm.startswith("esprit"):
                if plot_spec and i == len(dataset.dataset) - 1:
                    predictions, spectrum = method_output[0], method_output[1]
                    figures[algorithm]["norm factor"] = np.max(spectrum)
                    plot_spectrum(
                        predictions=predictions,
                        true_DOA=DOA * R2D,
                        system_model=system_model,
                        spectrum=spectrum,
                        algorithm="SubNet+" + algorithm.upper(),
                        figures=figures,
                    )
    return np.mean(hybrid_loss)


def evaluate_model_based(
        dataset: list,
        system_model,
        criterion: RMSPE,
        plot_spec=False,
        algorithm: str = "music",
        figures: dict = None,
        is_separted: bool = False):
    """
    Evaluate different model-based algorithms on a given dataset.

    Args:
        dataset (list): The evaluation dataset.
        system_model (SystemModel): The system model for the algorithms.
        criterion: The loss criterion for evaluation. Defaults to RMSPE.
        plot_spec (bool): Whether to plot the spectrum for the algorithms. Defaults to False.
        algorithm (str): The algorithm to use (e.g., "music", "mvdr", "esprit", "r-music"). Defaults to "music".
        figures (dict): Dictionary containing figure objects for plotting. Defaults to None.

    Returns:
        float: The average evaluation loss.

    Raises:
        Exception: If the algorithm is not supported.
    """
    # Initialize parameters for evaluation
    loss_list = []
    loss_list_angle = []
    loss_list_distance = []
    model_based = get_model_based_method(algorithm, system_model)

    for i, data in enumerate(dataset):
        X, doa = data
        X = X[0]
        # Root-MUSIC algorithms
        if algorithm.endswith("r-music"):
            root_music = RootMUSIC(system_model)
            if algorithm.startswith("sps"):
                # Spatial smoothing
                predictions, roots, predictions_all, _, M = root_music.narrowband(
                    X=X, mode="spatial_smoothing"
                )
            else:
                # Conventional
                predictions, roots, predictions_all, _, M = root_music.narrowband(
                    X=X, mode="sample"
                )
            # If the amount of predictions is less than the amount of sources
            predictions = add_random_predictions(M, predictions, algorithm)
            # Calculate loss criterion
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)
            # Plot spectrum
            if plot_spec and i == len(dataset.dataset) - 1:
                plot_spectrum(
                    predictions=predictions_all,
                    true_DOA=doa[0] * R2D,
                    roots=roots,
                    algorithm=algorithm.upper(),
                    figures=figures,
                )
        # MUSIC algorithms
        elif algorithm.endswith("music_1d"):
            if algorithm.startswith("bb"):
                # Broadband MUSIC
                predictions, spectrum, M = model_based(X=X)
            elif algorithm.startswith("sps"):
                # Spatial smoothing
                predictions, spectrum, M = model_based(
                    X=X, mode="spatial_smoothing"
                )
            elif algorithm.startswith("music"):
                # Conventional
                Rx = torch.from_numpy(np.cov(X.detach().numpy()))[None, :, :]
                predictions = model_based(Rx, is_soft=False)
            # If the amount of predictions is less than the amount of sources
            # predictions = add_random_predictions(M, predictions, algorithm)
            # Calculate loss criterion
            loss = criterion(predictions, doa)
            loss_list.append(loss)

        # ESPRIT algorithms
        elif "esprit" in algorithm:
            esprit = Esprit(system_model)
            if algorithm.startswith("sps"):
                # Spatial smoothing
                predictions, M = esprit.narrowband(X=X, mode="spatial_smoothing")
            else:
                # Conventional
                predictions, M = esprit.narrowband(X=X, mode="sample")
            # If the amount of predictions is less than the amount of sources
            predictions = add_random_predictions(M, predictions, algorithm)
            # Calculate loss criterion
            loss = criterion(predictions, doa * R2D)
            loss_list.append(loss)

        # MVDR algorithm
        elif algorithm.startswith("mvdr"):
            mvdr = MVDR(system_model)
            # Conventional
            _, spectrum = mvdr.narrowband(X=X, mode="sample")
            # Plot spectrum
            if plot_spec and i == len(dataset.dataset) - 1:
                plot_spectrum(
                    predictions=None,
                    true_DOA=doa * R2D,
                    system_model=system_model,
                    spectrum=spectrum,
                    algorithm=algorithm.upper(),
                    figures=figures,
                )
        elif algorithm.endswith("2D"):
            y = doa[0]
            doa, distances = y[:len(y) // 2][None, :], y[len(y) // 2:][None, :]

            Rx = torch.cov(X)[None, :, :]
            doa_prediction, distance_prediction = model_based(Rx, is_soft=False)
            if is_separted:
                rmspe, rmspe_angle, rmspe_distance = criterion(doa_prediction, doa, distance_prediction, distances,
                                                               is_separted)

                loss_list_angle.append(rmspe_angle.item())
                loss_list_distance.append(rmspe_distance.item())
            else:
                rmspe = criterion(doa_prediction, doa, distance_prediction, distances)
            loss_list.append(rmspe.item())

        else:
            raise Exception(
                f"evaluate_augmented_model: Algorithm {algorithm} is not supported."
            )
    if is_separted:
        return {"Overall":  torch.mean(torch.Tensor(loss_list)).item(),
                "Angle":    torch.mean(torch.Tensor(loss_list_angle)).item(),
                "Distance": torch.mean(torch.Tensor(loss_list_distance)).item()}
    else:
        return torch.mean(torch.Tensor(loss_list)).item()


def add_random_predictions(M: int, predictions: np.ndarray, algorithm: str):
    """
    Add random predictions if the number of predictions is less than the number of sources.

    Args:
        M (int): The number of sources.
        predictions (np.ndarray): The predicted DOA values.
        algorithm (str): The algorithm used.

    Returns:
        np.ndarray: The updated predictions with random values.

    """
    # Convert to np.ndarray array
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    while predictions.shape[0] < M:
        # print(f"{algorithm}: cant estimate M sources")
        predictions = np.insert(
            predictions, 0, np.round(np.random.rand(1) * 180, decimals=2) - 90.00
        )
    return predictions


def evaluate(
        model: nn.Module,
        model_type: str,
        model_test_dataset: list,
        generic_test_dataset: list,
        criterion: nn.Module,
        subspace_criterion,
        system_model,
        figures: dict,
        plot_spec: bool = True,
        augmented_methods: list = None,
        subspace_methods: list = None,
):
    """
    Wrapper function for model and algorithm evaluations.

    Parameters:
        model (nn.Module): The DNN model.
        model_type (str): Type of the model.
        model_test_dataset (list): Test dataset for the model.
        generic_test_dataset (list): Test dataset for generic subspace methods.
        criterion (nn.Module): Loss criterion for (DNN) model evaluation.
        subspace_criterion: Loss criterion for subspace method evaluation.
        system_model: instance of SystemModel.
        figures (dict): Dictionary to store figures.
        plot_spec (bool, optional): Whether to plot spectrums. Defaults to True.
        augmented_methods (list, optional): List of augmented methods for evaluation.
            Defaults to None.
        subspace_methods (list, optional): List of subspace methods for evaluation.
            Defaults to None.

    Returns:
        None
    """
    res = {}
    # Set default methods for SubspaceNet augmentation
    if not isinstance(augmented_methods, list) and model_type.startswith("SubspaceNet"):
        augmented_methods = [
            # "mvdr",
            # "r-music",
            # "esprit",
            # "music",
            # "music_2D",
        ]
    # Set default model-based subspace methods
    if not isinstance(subspace_methods, list):
        subspace_methods = [
            # "esprit",
            # "music_1d",
            # "r-music",
            # "mvdr",
            # "sps-r-music",
            # "sps-esprit",
            # "sps-music"
            # "bb-music",
            "music_2D"
        ]
    # Evaluate SubspaceNet + differentiable algorithm performances
    model_test_loss = evaluate_dnn_model(
        model=model,
        dataset=model_test_dataset,
        criterion=criterion,
        plot_spec=plot_spec,
        figures=figures,
        model_type=model_type,
        is_separted=True
    )
    res[model_type] = model_test_loss
    print(f"{model_type} Test loss = {model_test_loss}")
    # Evaluate SubspaceNet augmented methods
    for algorithm in augmented_methods:
        loss = evaluate_augmented_model(
            model=model,
            dataset=model_test_dataset,
            system_model=system_model,
            criterion=subspace_criterion,
            algorithm=algorithm,
            plot_spec=plot_spec,
            figures=figures,
        )
        print("augmented {} test loss = {}".format(algorithm, loss))
    # Evaluate classical subspace methods
    for algorithm in subspace_methods:
        loss = evaluate_model_based(
            generic_test_dataset,
            system_model,
            criterion=subspace_criterion,
            plot_spec=plot_spec,
            algorithm=algorithm,
            figures=figures,
            is_separted=True
        )
        print("{} test loss = {}".format(algorithm.lower(), loss))
        res[algorithm] = loss
    return res
