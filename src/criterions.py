"""Subspace-Net 
Details
----------
Name: criterions.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 03/06/23

Purpose:
--------
The purpose of this script is to define and document several loss functions (RMSPELoss and MSPELoss)
and a helper function (permute_prediction) for calculating the Root Mean Square Periodic Error (RMSPE)
and Mean Square Periodic Error (MSPE) between predicted values and target values.
The script also includes a utility function RMSPE and MSPE that calculates the RMSPE and MSPE values
for numpy arrays.

This script includes the following Classes anf functions:

* permute_prediction: A function that generates all possible permutations of a given prediction tensor.
* RMSPELoss (class): A custom PyTorch loss function that calculates the RMSPE loss between predicted values
    and target values. It inherits from the nn.Module class and overrides the forward method to perform
    the loss computation.
* MSPELoss (class): A custom PyTorch loss function that calculates the MSPE loss between predicted values
  and target values. It inherits from the nn.Module class and overrides the forward method to perform the loss computation.
* RMSPE (function): A function that calculates the RMSPE value between the DOA predictions and target DOA values for numpy arrays.
* MSPE (function): A function that calculates the MSPE value between the DOA predictions and target DOA values for numpy arrays.
* set_criterions(function): Set the loss criteria based on the criterion name.

"""

import numpy as np
import torch.nn as nn
import torch
from itertools import permutations

from numpy.ma.core import array

from src.utils import *
import time

from train_dcd import number_sensors, number_sources

BALANCE_FACTOR = 1.0


def add_line_to_file(file_name, line_to_add):
    try:
        with open(file_name, 'r+') as file:
            lines = file.readlines()
            if not lines or lines[-1].strip() != line_to_add:
                file.write('\n' + line_to_add)
                # print(f"Added line '{line_to_add}' to the file.")
            else:
                pass
                # print(f"Line '{line_to_add}' already exists in the file.")
    except FileNotFoundError:
        with open(file_name, 'w') as file:
            file.write(line_to_add)
            # print(f"Created file '{file_name}' with line '{line_to_add}'.")


def permute_prediction(prediction: torch.Tensor):
    """
    Generates all the available permutations of the given prediction tensor.

    Args:
        prediction (torch.Tensor): The input tensor for which permutations are generated.

    Returns:
        torch.Tensor: A tensor containing all the permutations of the input tensor.

    Examples:
        >>> prediction = torch.tensor([1, 2, 3])
        >>>> permute_prediction(prediction)
            torch.tensor([[1, 2, 3],
                          [1, 3, 2],
                          [2, 1, 3],
                          [2, 3, 1],
                          [3, 1, 2],
                          [3, 2, 1]])
        
    """
    torch_perm_list = []
    prediction = torch.atleast_1d(prediction)
    for p in list(permutations(range(prediction.shape[0]), prediction.shape[0])):
        torch_perm_list.append(prediction.index_select(0, torch.tensor(list(p), dtype=torch.int64).to(device)))
    predictions = torch.stack(torch_perm_list, dim=0)
    return predictions


class RMSELoss(nn.MSELoss):
    def __init__(self, *args):
        super(RMSELoss, self).__init__(*args)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = super(RMSELoss, self).forward(input.to(device), target.to(device))
        return torch.sqrt(mse_loss)


class RMSPELoss(nn.Module):
    """Root Mean Square Periodic Error (RMSPE) loss function.
    This loss function calculates the RMSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the RMSPE loss between the predictions and target values.

    Example:
        criterion = RMSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """

    def __init__(self, balance_factor=None):
        super(RMSPELoss, self).__init__()
        if balance_factor is None:
            self.balance_factor = nn.Parameter(torch.Tensor([BALANCE_FACTOR])).to(device).to(torch.float64)
        else:
            self.balance_factor = nn.Parameter(torch.Tensor([balance_factor])).to(device).to(torch.float64)

    def forward(self, angles_pred: torch.Tensor, angles: torch.Tensor,
                ranges_pred: torch.Tensor = None, ranges: torch.Tensor = None):
        """
        Compute the RMSPE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa,
        and possibly distance_predictions and distance.
        The predicted values and target values are expected to be in radians for the DOA values.
        The method iterates over the batch dimension and calculates the RMSPE loss for each sample in the batch.
        It utilizes the permute_prediction function to generate all possible permutations of the predicted values
        to consider all possible alignments. For each permutation, it calculates the error between the prediction
        and target values, applies modulo pi to ensure the error is within the range [-pi/2, pi/2], and then calculates the RMSE.
        The minimum RMSE value among all permutations is selected for each sample,
         including the RMSE for the distance values with the same permutation.
        Finally, the method averged the RMSE values for all samples in the batch and returns the result as the computed loss.

        Args:
            angles_pred (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            angles (torch.Tensor): Target values tensor of shape (batch_size, num_targets).
            ranges_pred (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            The default value is None.
            ranges (torch.Tensor): Target values tensor of shape (batch_size, num_targets).The default value is None.


        Returns:
            torch.Tensor: The computed RMSPE loss.

        Raises:
            None
        """
        # Calculate RMSPE loss for only DOA
        num_sources = angles_pred.shape[1]
        perm = list(permutations(range(num_sources), num_sources))
        num_of_perm = len(perm)

        err_angle = (angles_pred[:, perm] - torch.tile(angles[:, None, :], (1, num_of_perm, 1)).to(torch.float32))
        # Calculate error with modulo pi in the range [-pi/2, pi/2]
        err_angle += torch.pi / 2
        err_angle %= torch.pi
        err_angle -= torch.pi / 2
        rmspe_angle = np.sqrt(1 / num_sources) * torch.linalg.norm(err_angle, dim=-1)
        if ranges is None:
            rmspe, min_idx = torch.min(rmspe_angle, dim=-1)
        else:
            err_distance = (ranges_pred[:, perm].to(device) - torch.tile(ranges[:, None, :], (1, num_of_perm, 1)).to(device))
            rmspe_distance = np.sqrt(1 / num_sources) * torch.linalg.norm(err_distance, dim=-1)
            rmspe_angle, min_idx = torch.min(rmspe_angle, dim=-1)
            # always consider the permutation which yields the minimal RMSPE over the angles.
            rmspe_distance = torch.gather(rmspe_distance, 1, min_idx.unsqueeze(0)).squeeze()
            rmspe = self.balance_factor * rmspe_angle + (1 - self.balance_factor) * rmspe_distance
        result = torch.sum(rmspe)
        if ranges is None:
            return result
        else:
            result_angle = torch.sum(rmspe_angle)
            result_distance = torch.sum(rmspe_distance)
            return result, result_angle, result_distance

            # rmspe = []
            # for iter in range(doa_predictions.shape[0]):
            #     rmspe_list = []
            #     batch_predictions_doa = doa_predictions[iter].to(device)
            #     targets_doa = doa[iter].to(device).to(torch.float32)
            #     prediction_perm_doa = permute_prediction(batch_predictions_doa).to(device)
            #     for prediction_doa in prediction_perm_doa:
            #         # Calculate error with modulo pi
            #         error = (((prediction_doa - targets_doa) + (np.pi / 2)) % np.pi) - np.pi / 2
            #         # Calculate RMSE over all permutations
            #         rmspe_val = (1 / np.sqrt(len(targets_doa))) * torch.linalg.norm(error)
            #         rmspe_list.append(rmspe_val)
            #     rmspe_tensor = torch.stack(rmspe_list, dim=0)
            #     rmspe_min = torch.min(rmspe_tensor)
            #     rmspe.append(rmspe_min)
            # result = torch.mean(torch.stack(rmspe, dim=0))
            # if (result_ != result):
            #     raise ValueError("ERROR in RMSPE loss")
        # Calculate RMSPE loss for both DOA and distance
            # rmspe = []
            # rmspe_angle = []
            # rmspe_distance = []
            # for iter in range(doa_predictions.shape[0]):
            #     rmspe_list = []
            #     rmspe_angle_list = []
            #     rmspe_distance_list = []
            #
            #     batch_predictions_doa = doa_predictions[iter].to(device)
            #     targets_doa = doa[iter].to(device)
            #     prediction_perm_doa = permute_prediction(batch_predictions_doa).to(device)
            #
            #     batch_predictions_distance = distance_predictions[iter].to(device)
            #     targets_distance = distance[iter].to(device)
            #     prediction_perm_distance = permute_prediction(batch_predictions_distance).to(device)
            #
            #     for prediction_doa, prediction_distance in zip(prediction_perm_doa, prediction_perm_distance):
            #         # Calculate error with modulo pi
            #         angle_err = ((((prediction_doa - targets_doa) + (torch.pi / 2)) % torch.pi) - (torch.pi / 2))
            #         # Calculate error for distance
            #         distance_err = (prediction_distance - targets_distance)
            #         # Calculate RMSE over all permutations for each element
            #         rmspe_angle_val = (1 / np.sqrt(len(targets_doa))) * torch.linalg.norm(angle_err)
            #         rmspe_distance_val = (1 / np.sqrt(len(targets_distance))) * torch.linalg.norm(distance_err)
            #         # Sum the rmpse with a balance factor
            #         rmspe_val = self.balance_factor * rmspe_angle_val + (1 - self.balance_factor) * rmspe_distance_val
            #         rmspe_list.append(rmspe_val)
            #         rmspe_angle_list.append(rmspe_angle_val)
            #         rmspe_distance_list.append(rmspe_distance_val)
            #     rmspe_tensor = torch.stack(rmspe_list, dim=0)
            #     rmspe_angle_tensor = torch.stack(rmspe_angle_list, dim=0)
            #     rmspe_distnace_tensor = torch.stack(rmspe_distance_list, dim=0)
            #     # Choose minimal error from all permutations
            #     if rmspe_tensor.shape[0] == 1:
            #         rmspe_min = torch.min(rmspe_tensor)
            #         rmspe_angle.append(rmspe_angle_tensor.item())
            #         rmspe_distance.append(rmspe_distnace_tensor.item())
            #     else:
            #         rmspe_min, min_idx = torch.min(rmspe_tensor, dim=0)
            #         rmspe_angle.append(rmspe_angle_tensor[min_idx])
            #         rmspe_distance.append(rmspe_distnace_tensor[min_idx])
            #     rmspe.append(rmspe_min)
            # result = torch.mean(torch.stack(rmspe, dim=0))
            # if is_separted:
            #     result_angle = torch.mean(torch.Tensor(rmspe_angle), dim=0)
            #     result_distance = torch.mean(torch.Tensor(rmspe_distance), dim=0)
            #     return result, result_angle, result_distance
            # else:
            #     return result

    def adjust_balance_factor(self, loss=None):
        # if self.balance_factor > 0.4:
        #     if loss < 0.1 and self.balance_factor == torch.Tensor([BALANCE_FACTOR]):
        #         self.balance_factor *= 0.95
        #         print(f"Balance factor for RMSPE updated --> {self.balance_factor.item()}")
        #     if loss < 0.01 and self.balance_factor == torch.Tensor([BALANCE_FACTOR]) * 0.95:
        #         self.balance_factor *= 0.9
        #         print(f"Balance factor for RMSPE updated --> {self.balance_factor.item()}")
        #     if loss < 0.001:
        #         self.balance_factor *= 0.85
        #         print(f"Balance factor for RMSPE updated --> {self.balance_factor.item()}")
        self.balance_factor = 0.1
class MSPELoss(nn.Module):
    """Mean Square Periodic Error (MSPE) loss function.
    This loss function calculates the MSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(doa_predictions: torch.Tensor, doa: torch.Tensor) -> torch.Tensor:
            Computes the MSPE loss between the predictions and target values.

    Example:
        criterion = MSPELoss()
        predictions = torch.tensor([0.5, 1.2, 2.0])
        targets = torch.tensor([0.8, 1.5, 1.9])
        loss = criterion(predictions, targets)
    """

    def __init__(self):
        super(MSPELoss, self).__init__()

    def forward(self, doa_predictions: torch.Tensor, doa,
                distance_predictions: torch.Tensor = None, distance: torch.Tensor = None):
        """Compute the RMSPE loss between the predictions and target values.
        The forward method takes two input tensors: doa_predictions and doa.
        The predicted values and target values are expected to be in radians.
        The method iterates over the batch dimension and calculates the RMSPE loss for each sample in the batch.
        It utilizes the permute_prediction function to generate all possible permutations of the predicted values
        to consider all possible alignments. For each permutation, it calculates the error between the prediction
        and target values, applies modulo pi to ensure the error is within the range [-pi/2, pi/2], and then calculates the RMSPE.
        The minimum RMSPE value among all permutations is selected for each sample.
        Finally, the method sums up the RMSPE values for all samples in the batch and returns the result as the computed loss.

        Args:
            doa_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            doa (torch.Tensor): Target values tensor of shape (batch_size, num_targets).
            distance_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            The default value is None.
            distance (torch.Tensor): Target values tensor of shape (batch_size, num_targets).The default value is None.

        Returns:
            torch.Tensor: The computed MSPE loss.

        Raises:
            None
        """
        rmspe = []
        for iter in range(doa_predictions.shape[0]):
            rmspe_list = []
            batch_predictions = doa_predictions[iter].to(device)
            targets = doa[iter].to(device)
            prediction_perm = permute_prediction(batch_predictions).to(device)
            for prediction in prediction_perm:
                # Calculate error with modulo pi
                error = (((prediction - targets) + (np.pi / 2)) % np.pi) - np.pi / 2
                # Calculate MSE over all permutations
                rmspe_val = (1 / len(targets)) * (torch.linalg.norm(error) ** 2)
                rmspe_list.append(rmspe_val)
            rmspe_tensor = torch.stack(rmspe_list, dim=0)
            rmspe_min = torch.min(rmspe_tensor)
            # Choose minimal error from all permutations
            rmspe.append(rmspe_min)
        result = torch.sum(torch.stack(rmspe, dim=0))

        if distance_predictions is not None:
            if distance is None:
                raise Exception("Target distances values are missing!")
            mse_loss = nn.MSELoss()
            distance_loss = mse_loss(distance_predictions, distance)
            result += distance_loss
        return result


def RMSPE(doa_predictions: np.ndarray, doa: np.ndarray,
          distance_predictions: np.ndarray = None, distance: np.ndarray = None, is_separted: bool = False):
    """
    Calculate the Root Mean Square Periodic Error (RMSPE) between the DOA predictions and target DOA values.

    Args:
        doa_predictions (np.ndarray): Array of DOA predictions.
        doa (np.ndarray): Array of target DOA values.
        distance_predictions (np.ndarray):
        distance (distance):

    Returns:
        float: The computed RMSPE value.

    Raises:
        None
    """
    balance_factor = BALANCE_FACTOR
    rmspe_list = []
    rmspe_angle_list = []
    rmspe_distance_list = []

    if distance_predictions is not None:
        for p_doa, p_distance in zip(list(permutations(doa_predictions, len(doa_predictions))),
                                     list(permutations(distance_predictions, len(distance_predictions)))):
            p_doa, p_distance = np.array(p_doa, dtype=np.float32), np.array(p_distance, dtype=np.float32)
            doa, distance = np.array(doa, dtype=np.float64), np.array(distance, dtype=np.float64)
            # Calculate error with modulo pi
            error_angle = (((p_doa - doa) + np.pi / 2) % np.pi) - (np.pi / 2)
            error_distance = (p_distance - distance)
            # Calculate RMSE over all permutations
            rmspe_angle = (1 / np.sqrt(len(p_doa), dtype=np.float64)) * np.linalg.norm(error_angle)
            rmspe_distance = (1 / np.sqrt(len(p_distance), dtype=np.float64)) * np.linalg.norm(error_distance)
            rmspe_val = balance_factor * rmspe_angle + (1 - balance_factor) * rmspe_distance
            rmspe_list.append(rmspe_val)
            rmspe_angle_list.append(rmspe_angle)
            rmspe_distance_list.append(rmspe_distance)
        # Choose minimal error from all permutations
        if is_separted:
            rmspe, min_idx = np.min(rmspe_list), np.argmin(rmspe_list)
            rmspe_angle = rmspe_angle_list[min_idx]
            rmspe_distance = rmspe_distance_list[min_idx]
            res = rmspe, rmspe_angle, rmspe_distance
        else:
            res = np.min(rmspe_list)
    else:
        for p_doa in list(permutations(doa_predictions, len(doa_predictions))):
            p_doa = np.array(p_doa, dtype=np.float32)
            doa = np.array(doa, dtype=np.float64)
            # Calculate error with modulo pi
            error = ((p_doa - doa) + np.pi / 2) % np.pi - (np.pi / 2)
            # Calculate RMSE over all permutations
            rmspe = (1 / np.sqrt(len(p_doa))) * np.linalg.norm(error)
            rmspe_list.append(rmspe)
        # Choose minimal error from all permutations
        res = np.min(rmspe_list)


    return res


def MSPE(doa_predictions: np.ndarray, doa: np.ndarray,
         distance_predictions: np.ndarray = None, distance: np.ndarray = None, is_separted: bool = False):
    """Calculate the Mean Square Percentage Error (RMSPE) between the DOA predictions and target DOA values.

    Args:
        doa_predictions (np.ndarray): Array of DOA predictions.
        doa (np.ndarray): Array of target DOA values.

    Returns:
        float: The computed RMSPE value.

    Raises:
        None
    """
    balance_factor = BALANCE_FACTOR
    mspe_list = []
    mspe_angle_list = []
    mspe_distance_list = []
    for p_doa, p_distance in zip(list(permutations(doa_predictions, len(doa_predictions))),
                                 list(permutations(distance_predictions, len(distance_predictions)))):
        p_doa, p_distance = np.array(p_doa, dtype=np.float32), np.array(p_distance, dtype=np.float32)
        doa, distance = np.array(doa, dtype=np.float64), np.array(distance, dtype=np.float64)
        # Calculate error with modulo pi
        error_angle = ((p_doa - doa) + np.pi / 2) % np.pi - (np.pi / 2)
        error_distance = (p_distance - distance)
        # Calculate MSE over all permutations
        mspe_angle = (1 / len(p_doa)) * (np.linalg.norm(error_angle) ** 2)
        mspe_distance = (1 / len(p_distance)) * (np.linalg.norm(error_distance) ** 2)
        mspe_val = balance_factor * mspe_angle + (1 - balance_factor) * mspe_distance
        mspe_list.append(mspe_val)
        mspe_angle_list.append(mspe_angle)
        mspe_distance_list.append(mspe_distance)
    if is_separted:
        mspe, min_idx = np.min(mspe_list), np.argmin(mspe_list)
        mspe_angle = mspe_angle_list[min_idx]
        mspe_distance = mspe_distance_list[min_idx]
        res = mspe, mspe_angle, mspe_distance
    else:
        res = np.min(mspe_list)
    return res

class CartesianLoss(nn.Module):
    def __init__(self):
        super(CartesianLoss, self).__init__()

    def forward(self, angles_pred: torch.Tensor, angles: torch.Tensor, ranges_pred: torch.Tensor,
                ranges: torch.Tensor):
        """
        the input given is expected to contain angels and distances.
        """
        M = angles.shape[1]
        if angles_pred.shape[1] > angles.shape[1]:
            # in this case, randomly drop some of the predictions
            indices = torch.randperm(angles_pred.shape[1])[:M].to(device)
            angles_pred = torch.gather(angles_pred, 1, indices[None, :])
            ranges_pred = torch.gather(ranges_pred, 1, indices[None, :])

        elif angles_pred.shape[1] < angles.shape[1]:
            # add a random angle to the predictions
            random_angles = torch.distributions.uniform.Uniform(-torch.pi / 3, torch.pi / 3).sample([angles_pred.shape[0], M - angles_pred.shape[1]])
            random_ranges = torch.distributions.uniform.Uniform(torch.min(ranges).item(), torch.max(ranges).item()).sample([angles_pred.shape[0], M - angles_pred.shape[1]])
            angles_pred = torch.cat((angles_pred, random_angles.to(device)), dim=1)
            ranges_pred = torch.cat((ranges_pred, random_ranges.to(device)), dim=1)

        number_of_samples = angles_pred.shape[0]
        true_x = torch.cos(angles) * ranges
        true_y = torch.sin(angles) * ranges
        coords_true = torch.stack((true_x, true_y), dim=2)
        pred_x = torch.cos(angles_pred) * ranges_pred
        pred_y = torch.sin(angles_pred) * ranges_pred
        coords_pred = torch.stack((pred_x, pred_y), dim=2)
        # need to consider all possible permutations for M sources
        perm = list(permutations(range(M), M))
        perm = torch.tensor(perm, dtype=torch.int64).to(device)
        num_of_perm = len(perm)

        error = torch.tile(coords_true[:, None, :, :], (1, num_of_perm, 1, 1)) - coords_pred[:, perm]
        loss = torch.sqrt(torch.sum(error ** 2, dim=-1))
        loss = torch.mean(loss, dim=-1)
        loss = torch.min(loss, dim=-1)
        return torch.sum(loss[0])
        # loss = []
        # for batch in range(number_of_samples):
        #     loss_per_sample = []
        #     for p in perm:
        #         loss_per_sample.append(torch.sqrt(torch.sum((coords_true[batch] - coords_pred[batch, p, :]) ** 2, dim=1)).mean())
        #     loss.append(torch.min(torch.stack(loss_per_sample, dim=0)))
        # if (loss_[0] != torch.stack(loss, dim=0)).all():
        #     raise ValueError("Error in Cartesian Loss")

class NoiseOrthogonalLoss(nn.Module):
    def __init__(self, array, sensors_distance):
        super(NoiseOrthogonalLoss, self).__init__()
        self.array = array
        self.sensors_distance = sensors_distance
        self.number_sensors = array.shape[1]

    def forward(self, **kwargs):
        if "ranges" in kwargs:
            return self.__forward_with_ranges(**kwargs)
        else:
            return self.__forward_without_ranges(**kwargs)

    def __forward_without_ranges(self, **kwargs):
        theta = kwargs["angles"].unsqueeze(-1)
        time_delay = torch.einsum("nm, ban -> ban",
                                  self.array,
                                  torch.sin(theta).repeat(1, 1, self.number_sensors) * self.sensors_distance)
        search_grid = torch.exp(-2 * 1j * torch.pi * time_delay)
        var1 = torch.bmm(search_grid.conj(), kwargs["noise_subspace"].to(torch.complex128))
        inverse_spectrum = torch.norm(var1, dim=-1)
        spectrum = 1 / inverse_spectrum
        loss = -torch.sum(spectrum, dim=1).sum()
        return loss

    def __forward_with_ranges(self, **kwargs):
        theta = kwargs["angles"][:, :, None]
        distances = kwargs["ranges"][:, :, None].to(torch.float64)
        array_square = torch.pow(self.array, 2).to(torch.float64)
        noise_subspace = kwargs["noise_subspace"].to(torch.complex128)

        first_order = torch.einsum("nm, bna -> bna",
                                   self.array,
                                   torch.sin(theta).repeat(1, 1, self.number_sensors).transpose(1, 2) * self.sensors_distance)

        second_order = -0.5 * torch.div(torch.pow(torch.cos(theta) * self.sensors_distance, 2), distances.transpose(1, 2))
        second_order = second_order[:, :, :, None].repeat(1, 1, 1, self.number_sensors)
        second_order = torch.einsum("nm, bnda -> bnda",
                                    array_square,
                                    second_order.transpose(3, 1).transpose(2, 3))

        first_order = first_order[:, :, :, None].repeat(1, 1, 1, second_order.shape[-1])

        time_delay = first_order + second_order

        search_grid = torch.exp(2 * -1j * torch.pi * time_delay)
        var1 = torch.einsum("badk, bkl -> badl",
                            search_grid.conj().transpose(1, 3).transpose(1, 2)[:, :, :, :noise_subspace.shape[1]],
                            noise_subspace)
        # get the norm value for each element in the batch.
        inverse_spectrum = torch.linalg.diagonal(torch.norm(var1, dim=-1)) ** 2
        # spectrum = 1 / inverse_spectrum
        loss = torch.sum(inverse_spectrum, dim=-1).sum()
        return loss


def set_criterions(criterion_name: str, balance_factor: float = 0.0):
    """
    Set the loss criteria based on the criterion name.

    Parameters:
        criterion_name (str): Name of the criterion.

    Returns:
        criterion (nn.Module): Loss criterion for model evaluation.
        subspace_criterion (Callable): Loss criterion for subspace method evaluation.

    Raises:
        Exception: If the criterion name is not defined.
    """
    if criterion_name.startswith("rmspe"):
        criterion = RMSPELoss(balance_factor)
        subspace_criterion = RMSPELoss(balance_factor)
    elif criterion_name.startswith("mspe"):
        criterion = MSPELoss()
        subspace_criterion = MSPE
    elif criterion_name.startswith("mse"):
        criterion = nn.MSELoss()
        subspace_criterion = MSPE
    elif criterion_name.startswith("rmse"):
        criterion = RMSPELoss(balance_factor)
        subspace_criterion = RMSPELoss(balance_factor)
    elif criterion_name.startswith("cartesian"):
        criterion = CartesianLoss()
        subspace_criterion = CartesianLoss()
    else:
        raise Exception(f"criterions.set_criterions: Criterion {criterion_name} is not defined")
    print(f"Loss measure = {criterion_name}")
    return criterion, subspace_criterion


if __name__ == "__main__":
    prediction = torch.tensor([1, 2, 3])
    print(permute_prediction(prediction))
