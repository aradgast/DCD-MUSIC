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
import os
import time
import numpy as np
# Imports
import torch.nn as nn
from src.utils import *

from src.criterions import RMSPELoss, MSPELoss, RMSELoss, CartesianLoss
from src.criterions import RMSPE, MSPE
# import methods
from src.methods import MVDR
from src.methods_pack.music import MUSIC
from src.methods_pack.root_music import RootMusic
from src.methods_pack.esprit import ESPRIT
from src.methods_pack.mle import MLE
# import models
from src.models import (ModelGenerator, SubspaceNet, CascadedSubspaceNet, DeepAugmentedMUSIC,
                        DeepCNN, DeepRootMUSIC, TransMUSIC)

from src.plotting import plot_spectrum
from src.system_model import SystemModel, SystemModelParams


def get_model_based_method(method_name: str, system_model: SystemModel):
    """

    Parameters
    ----------
    method_name(str): the method to use - music_1d, music_2d, root_music, esprit...
    system_model(SystemModel) : the system model to use as an argument to the method class.

    Returns
    -------
    an instance of the method.
    """
    if method_name.lower().endswith("music_1d"):
        return MUSIC(system_model=system_model, estimation_parameter="angle")
    if method_name.lower().endswith("music_2d"):
        return MUSIC(system_model=system_model, estimation_parameter="angle, range")
    if method_name.lower() == "root_music":
        return RootMusic(system_model)
    if method_name.lower().endswith("esprit"):
        return ESPRIT(system_model)


def get_model(model_name: str, params: dict, system_model: SystemModel):
    model_config = (
        ModelGenerator()
        .set_model_type(model_name)
        .set_system_model(system_model)
        .set_model_params(params)
        .set_model()
    )
    model = model_config.model
    path = os.path.join(os.getcwd(), "data", "weights", "final_models", model.get_model_file_name())
    try:
        model.load_state_dict(torch.load(path))
    except FileNotFoundError as e:
        print("####################################")
        print(e)
        print("####################################")
        try:
            print(f"Model {model_name} not found in final_models, trying to load from temp weights.")
            path = os.path.join(os.getcwd(), "data", "weights", model.get_model_file_name())
            model.load_state_dict(torch.load(path))
        except FileNotFoundError as e:
            print("####################################")
            print(e)
            print("####################################")
            print(f"Model {model_name} not found")
    return model.to(device)


def evaluate_dnn_model(
        model: nn.Module,
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
            if model.system_model.params.field_type.endswith("Far"):
                DOA = true_label.to(device)
            elif model.system_model.params.field_type.endswith("Near"):
                DOA, RANGE = torch.split(true_label, true_label.size(1) // 2, dim=1)
                RANGE.to(device)

            test_length += DOA.shape[1]
            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Get model output
            if isinstance(model, SubspaceNet) and model.field_type.endswith("Near"):
                if isinstance(model, CascadedSubspaceNet):
                    model_output = model(X, is_soft=False, train_angle_extractor=False)
                else:  # SubspaceNet with 2D MUSIC
                    model_output = model(X, is_soft=False)
            else:
                model_output = model(X)
            if isinstance(model, DeepAugmentedMUSIC):
                # Deep Augmented MUSIC
                DOA_predictions = model_output.to(device)
            elif isinstance(model, DeepCNN):
                # Deep CNN
                if isinstance(criterion, nn.BCELoss):
                    # If evaluation performed over validation set, loss is BCE
                    DOA_predictions = model_output.to(device)
                    # find peaks in the pseudo spectrum of probabilities
                    DOA_predictions = (
                            get_k_peaks(361, DOA.shape[1], DOA_predictions[0]) * D2R
                    )
                    DOA_predictions = DOA_predictions.view(1, DOA_predictions.shape[0])
                elif isinstance(criterion, [RMSPELoss, MSPELoss]):
                    # If evaluation performed over testset, loss is RMSPE / MSPE
                    DOA_predictions = model_output.to(device)
                else:
                    raise Exception(
                        f"evaluate_dnn_model: Loss criterion is not defined for {model_type} model"
                    )
            elif isinstance(model, SubspaceNet):
                if model.field_type.endswith("Near"):
                    # 2 possible outputs: DOA and RANGE
                    DOA_predictions = model_output[0].to(device)
                    RANGE_predictions = model_output[1].to(device)
                elif model.field_type.endswith("Far"):
                    # Default - SubSpaceNet
                    DOA_predictions = model_output[0]
            elif isinstance(model, TransMUSIC):
                if model.estimation_params == "angle":
                    DOA_predictions = model_output[0]
                    prob_source_number = model_output[1]
                elif model.estimation_params == "angle, range":
                    DOA_predictions, RANGE_predictions = model_output[0][:, 0], model_output[0][:, 1]
                    prob_source_number = model_output[1]
                estimated_source_number = torch.argmax(prob_source_number, dim=1)
            else:
                raise Exception(
                    f"evaluate_dnn_model: Model type {model_type} is not defined"
                )
            # Compute prediction loss
            if isinstance(model, DeepCNN) and isinstance(criterion, RMSPELoss):
                eval_loss = criterion(DOA_predictions.float(), DOA.float())
            elif isinstance(model, TransMUSIC):
                DOA_predictions = DOA_predictions[:, :DOA.shape[1]]
                if model.estimation_params == "angle":
                    eval_loss = criterion(DOA_predictions, DOA)
                elif model.estimation_params == "angle, range":
                    RANGE_predictions = RANGE_predictions[:, :RANGE.shape[1]]
                    if isinstance(criterion, RMSPELoss):
                        eval_loss = criterion(DOA_predictions, DOA, RANGE_predictions, RANGE, is_separted)
                        if is_separted:
                            eval_loss, eval_loss_angle, eval_loss_distance = eval_loss
                            overall_loss_angle += eval_loss_angle.item() / len(dataset)
                            overall_loss_distance += eval_loss_distance.item() / len(dataset)
                    else:
                        eval_loss = criterion(DOA_predictions, DOA, RANGE_predictions, RANGE.to(device))
            elif isinstance(model, SubspaceNet):
                if model.field_type.endswith("Near"):
                    if not (isinstance(criterion, nn.BCELoss) or isinstance(criterion, CartesianLoss)):
                        eval_loss = criterion(DOA_predictions, DOA, RANGE_predictions, RANGE.to(device), is_separted)
                        if is_separted:
                            eval_loss, eval_loss_angle, eval_loss_distance = eval_loss
                            overall_loss_angle += eval_loss_angle.item() / len(dataset)
                            overall_loss_distance += eval_loss_distance.item() / len(dataset)
                    else:
                        eval_loss = criterion(DOA_predictions, DOA.to(device), RANGE_predictions, RANGE.to(device))
                else:
                    eval_loss = criterion(DOA_predictions, DOA)
            else:
                raise Exception(f"evaluate_dnn_model: Model type is not defined: {model_type}")
            # add the batch evaluation loss to epoch loss
            overall_loss += eval_loss.item() / len(dataset)

    # Plot spectrum for SubspaceNet model
    if plot_spec and model_type.endswith("SubspaceNet"):
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
        overall_loss = {"Overall": overall_loss,
                        "Angle": overall_loss_angle,
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
        "esprit": ESPRIT(system_model),
        "r-music": RootMusic(system_model),
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
    if algorithm.lower() == "crb":
        if system_model.params.signal_nature.lower() == "non-coherent":
            crb = evaluate_crb(dataset, system_model.params, mode="cartesian")
            return crb
    model_based = get_model_based_method(algorithm, system_model)
    for i, data in enumerate(dataset):
        X, doa = data
        # X = X[0]
        # Root-MUSIC algorithms
        if algorithm.endswith("r-music"):
            root_music = RootMusic(system_model)
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
                Rx = model_based.pre_processing(X, mode="sps")
            elif algorithm.startswith("music"):
                # Conventional
                Rx = model_based.pre_processing(X, mode="sample")
            predictions = model_based(Rx, is_soft=False)
            # If the amount of predictions is less than the amount of sources
            # predictions = add_random_predictions(M, predictions, algorithm)
            # Calculate loss criterion
            loss = criterion(predictions, doa)
            loss_list.append(loss)

        # ESPRIT algorithms
        elif "esprit" in algorithm:
            # esprit = ESPRIT(system_model)
            if algorithm.startswith("sps"):
                # Spatial smoothing
                Rx = model_based.pre_processing(X, mode="sps")
            else:
                # Conventional
                Rx = model_based.pre_processing(X, mode="sample")
            predictions = model_based(Rx)
            # If the amount of predictions is less than the amount of sources
            # predictions = add_random_predictions(M, predictions, algorithm)
            # Calculate loss criterion
            if doa.shape[1] != predictions.shape[1]:
                y = doa[0]
                doa, distances = y[:len(y) // 2][None, :], y[len(y) // 2:][None, :]
            loss = criterion(predictions, doa)
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
            y = doa
            doa, distances = y[:, :y.shape[-1] // 2], y[:, y.shape[-1] // 2:]
            doa = doa.to(device)
            distances = distances.to(device)
            if algorithm.startswith("sps"):
                Rx = model_based.pre_processing(X, mode="sps")
            else:
                Rx = model_based.pre_processing(X, mode="sample")
            doa_prediction, distance_prediction = model_based(Rx, is_soft=False)
            if isinstance(criterion, RMSPELoss) and is_separted:
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
        return {"Overall": torch.mean(torch.Tensor(loss_list)).item(),
                "Angle": torch.mean(torch.Tensor(loss_list_angle)).item(),
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


def evaluate_crb(dataset: list,
                 params: SystemModelParams,
                 mode: str="separate"):
    u_snr = 10 ** (params.snr / 10)
    if params.field_type.lower() == "far":
        print("CRB calculation is not supported for Far Field yet.")
        return None
    elif params.field_type.lower() == "near":
        if params.signal_nature.lower() == "non-coherent":
            angles = []
            distances = []
            ucrb_cartzien = None
            for i, data in enumerate(dataset):
                _, labels = data
                angles.extend(*labels[:, :labels.shape[1] // 2][None, :].detach().numpy())
                distances.extend(*labels[:, labels.shape[1] // 2:][None, :].detach().numpy())
            angles = np.array(angles)
            distances = np.array(distances)
            snr_coeff = (1 + 1 / (u_snr * params.N))
            ucrb_angle = (3 * 2 ** 2) / (2 * u_snr * params.T * (np.pi * np.cos(angles)) ** 2)
            ucrb_angle *= (8 * params.N - 11) * (2 * params.N - 1)
            ucrb_angle /= params.N * (params.N ** 2 - 1) * (params.N ** 2 - 4)
            ucrb_angle *= snr_coeff

            ucrb_distance = 6 * distances ** 2 * 2 ** 4 / (u_snr * params.T * np.pi ** 2)  # missing /wavelength
            ucrb_distance *= snr_coeff
            ucrb_distance /= params.N ** 2 * (params.N ** 2 - 1) * (params.N ** 2 - 4) * np.cos(angles) ** 4
            num = 15 * distances ** 2
            num += (30 / 2) * distances * (params.N - 1) * np.sin(angles)  # missing *wavelength
            num += (1 / 2) ** 2 * (8 * params.N - 11) * (2 * params.N - 1) * np.sin(angles) ** 2  # missing * wavelength ** 2
            ucrb_distance *= num
            if mode == "cartesian":
                # Need to calculate the cross term as well, and change coordinates.
                ucrb_cross = - snr_coeff * (3 * distances)
                ucrb_cross /= u_snr * params.T * np.pi ** 2 * (1 / 2) ** 3
                ucrb_cross *= 15 * distances*(params.N - 1) + (1 / 2) * (8 * params.N - 11) * (2 * params.N - 1) * np.sin(angles)
                ucrb_cross /= params.N * (params.N ** 2 - 1) * (params.N ** 2 - 4) * np.cos(angles) ** 3

                #change coordinates
                ucrb_cartzien = distances ** 2 * ucrb_angle + ucrb_distance
                ucrb_cartzien -= distances ** 2 * np.sin(2 * angles) * ucrb_angle
                # ucrb_cartzien += np.sin(2 * angles) * ucrb_distance
                # ucrb_cartzien += 2 * distances * np.cos(2 * angles) * ucrb_cross
                ucrb_cartzien = np.mean(ucrb_cartzien)

            return {"Overall": ucrb_cartzien, "Angle": np.mean(ucrb_angle), "Distance": np.mean(ucrb_distance)}
        else:
            print("UCRB calculation for the coherent is not supported yet")
    else:
        print("Unrecognized field type.")
    return


def evaluate_mle(dataset: list, system_model: SystemModel, criterion):
    """
    Evaluate the Maximum Likelihood Estimation (MLE) algorithm on a given dataset.

    Args:
        dataset (list): The evaluation dataset.
        system_model (SystemModel): The system model for the MLE algorithm.

    Returns:
        float: The average evaluation loss.
    """
    # initialize mle instance
    mle = MLE(system_model)
    # Initialize parameters for evaluation
    loss_list = []
    for i, data in enumerate(dataset):
        X, labels = data
        Rx = calculate_covariance_tensor(X, method="simple").to(device)
        angles = labels[:, :labels.shape[-1] // 2].to(device)
        distances = labels[:, labels.shape[-1] // 2:].to(device)
        # Apply MLE algorithm
        pred_angle, pred_distance = mle(Rx)
        # Calculate loss criterion
        loss = criterion(pred_angle.to(device), angles, pred_distance.to(device), distances)
        loss_list.append(loss.item())
    return {"Overall": np.mean(loss_list)}


def evaluate(
        generic_test_dataset: list,
        criterion: nn.Module,
        system_model: SystemModel,
        figures: dict,
        plot_spec: bool = True,
        models: dict = None,
        augmented_methods: list = None,
        subspace_methods: list = None,
):
    """
    Wrapper function for model and algorithm evaluations.

    Parameters:
        generic_test_dataset (list): Test dataset for generic subspace methods.
        criterion (nn.Module): Loss criterion for (DNN) model evaluation.
        system_model: instance of SystemModel.
        figures (dict): Dictionary to store figures.
        plot_spec (bool, optional): Whether to plot spectrums. Defaults to True.
        models (dict): dict that contains the models to evluate and their parameters.
        augmented_methods (list, optional): List of augmented methods for evaluation.
            Defaults to None.
        subspace_methods (list, optional): List of subspace methods for evaluation.
            Defaults to None.

    Returns:
        None
    """
    res = {}
    # Evaluate SubspaceNet + differentiable algorithm performances
    for model_name, params in models.items():
        model = get_model(model_name, params, system_model)
        # num_of_params = sum(p.numel() for p in model.parameters())
        # total_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
        # print(f"Number of parameters in {model_name}: {num_of_params} with total size: {total_size} bytes")
        start = time.time()
        model_test_loss = evaluate_dnn_model(
            model=model,
            dataset=generic_test_dataset,
            criterion=criterion,
            plot_spec=plot_spec,
            figures=figures,
            model_type=model_name,
            is_separted=True)
        print(f"{model_name} evaluation time: {time.time() - start}")
        res[model_name] = model_test_loss
    # Evaluate SubspaceNet augmented methods
    for algorithm in augmented_methods:
        loss = evaluate_augmented_model(
            model=model,
            dataset=generic_test_dataset,
            system_model=system_model,
            criterion=criterion,
            algorithm=algorithm,
            plot_spec=plot_spec,
            figures=figures,
        )
        res["augmented" + algorithm] = loss
    # Evaluate classical subspace methods
    for algorithm in subspace_methods:
        start = time.time()
        loss = evaluate_model_based(
            generic_test_dataset,
            system_model,
            criterion=criterion,
            plot_spec=plot_spec,
            algorithm=algorithm,
            figures=figures,
            is_separted=True
        )
        print(f"{algorithm} evaluation time: {time.time() - start}")
        res[algorithm] = loss
    # MLE
    # mle_loss = evaluate_mle(generic_test_dataset, system_model, criterion)
    # res["MLE"] = mle_loss
    for method, loss_ in res.items():
        print(f"{method.upper() + ' test loss' : <30} = {loss_}")
    return res
