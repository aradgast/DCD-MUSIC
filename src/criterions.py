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
from src.utils import *
import time
BALANCE_FACTOR = 0.6


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

    def forward(self, doa_predictions: torch.Tensor, doa: torch.Tensor,
                distance_predictions: torch.Tensor = None, distance: torch.Tensor = None, is_separted: bool = False):
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
            doa_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            doa (torch.Tensor): Target values tensor of shape (batch_size, num_targets).
            distance_predictions (torch.Tensor): Predicted values tensor of shape (batch_size, num_predictions).
            The default value is None.
            distance (torch.Tensor): Target values tensor of shape (batch_size, num_targets).The default value is None.


        Returns:
            torch.Tensor: The computed RMSPE loss.

        Raises:
            None
        """
        # Calculate RMSPE loss for only DOA
        num_sources = doa_predictions.shape[1]
        num_samples = doa_predictions.shape[0]
        perm = list(permutations(range(num_sources), num_sources))
        if distance is None:
            rmspe = []
            for iter in range(doa_predictions.shape[0]):
                rmspe_list = []
                batch_predictions_doa = doa_predictions[iter].to(device)
                targets_doa = doa[iter].to(device)
                prediction_perm_doa = permute_prediction(batch_predictions_doa).to(device)
                for prediction_doa in prediction_perm_doa:
                    # Calculate error with modulo pi
                    error = (((prediction_doa - targets_doa) + (np.pi / 2)) % np.pi) - np.pi / 2
                    # Calculate RMSE over all permutations
                    rmspe_val = (1 / np.sqrt(len(targets_doa))) * torch.linalg.norm(error)
                    rmspe_list.append(rmspe_val)
                rmspe_tensor = torch.stack(rmspe_list, dim=0)
                rmspe_min = torch.min(rmspe_tensor)
                rmspe.append(rmspe_min)
            result = torch.mean(torch.stack(rmspe, dim=0))
            error_ = ((doa_predictions[:, perm] - torch.tile(doa[:, :, None], (1, num_sources))) + torch.pi / 2) % torch.pi - torch.pi / 2
            rmspe_ = torch.sqrt((1 / num_sources) * torch.linalg.norm(error_, dim=2))
            res_ = torch.min(rmspe_, dim=1)[0]
            return result
        # Calculate RMSPE loss for both DOA and distance
        else:
            rmspe = []
            rmspe_angle = []
            rmspe_distance = []
            for iter in range(doa_predictions.shape[0]):
                rmspe_list = []
                rmspe_angle_list = []
                rmspe_distance_list = []

                batch_predictions_doa = doa_predictions[iter].to(device)
                targets_doa = doa[iter].to(device)
                prediction_perm_doa = permute_prediction(batch_predictions_doa).to(device)

                batch_predictions_distance = distance_predictions[iter].to(device)
                targets_distance = distance[iter].to(device)
                prediction_perm_distance = permute_prediction(batch_predictions_distance).to(device)

                for prediction_doa, prediction_distance in zip(prediction_perm_doa, prediction_perm_distance):
                    # Calculate error with modulo pi
                    angle_err = ((((prediction_doa - targets_doa) + (torch.pi / 2)) % torch.pi) - (torch.pi / 2))
                    # Calculate error for distance
                    distance_err = (prediction_distance - targets_distance)
                    # Calculate RMSE over all permutations for each element
                    rmspe_angle_val = (1 / np.sqrt(len(targets_doa))) * torch.linalg.norm(angle_err)
                    rmspe_distance_val = (1 / np.sqrt(len(targets_distance))) * torch.linalg.norm(distance_err)
                    # Sum the rmpse with a balance factor
                    rmspe_val = self.balance_factor * rmspe_angle_val + (1 - self.balance_factor) * rmspe_distance_val
                    rmspe_list.append(rmspe_val)
                    rmspe_angle_list.append(rmspe_angle_val)
                    rmspe_distance_list.append(rmspe_distance_val)
                rmspe_tensor = torch.stack(rmspe_list, dim=0)
                rmspe_angle_tensor = torch.stack(rmspe_angle_list, dim=0)
                rmspe_distnace_tensor = torch.stack(rmspe_distance_list, dim=0)
                # Choose minimal error from all permutations
                if rmspe_tensor.shape[0] == 1:
                    rmspe_min = torch.min(rmspe_tensor)
                    rmspe_angle.append(rmspe_angle_tensor.item())
                    rmspe_distance.append(rmspe_distnace_tensor.item())
                else:
                    rmspe_min, min_idx = torch.min(rmspe_tensor, dim=0)
                    rmspe_angle.append(rmspe_angle_tensor[min_idx])
                    rmspe_distance.append(rmspe_distnace_tensor[min_idx])
                rmspe.append(rmspe_min)

            result = torch.mean(torch.stack(rmspe, dim=0))
            if is_separted:
                result_angle = torch.mean(torch.Tensor(rmspe_angle), dim=0)
                result_distance = torch.mean(torch.Tensor(rmspe_distance), dim=0)
                return result, result_angle, result_distance
            else:
                return result

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

    def forward(self, predictions_angle: torch.Tensor, targets_angle: torch.Tensor, predictions_distance: torch.Tensor,
                targets_distance: torch.Tensor):
        """
        the input given is expected to contain angels and distances.
        """
        M = predictions_angle.shape[1]
        number_of_samples = predictions_angle.shape[0]
        true_x = torch.cos(targets_angle) * targets_distance
        true_y = torch.sin(targets_angle) * targets_distance
        coords_true = torch.stack((true_x, true_y), dim=2)
        pred_x = torch.cos(predictions_angle) * predictions_distance
        pred_y = torch.sin(predictions_angle) * predictions_distance
        coords_pred = torch.stack((pred_x, pred_y), dim=2)
        # need to consider all possible permutations for M sources
        perm = list(permutations(range(M), M))
        # loss = []
        # for batch in range(number_of_samples):
        #     loss_per_sample = []
        #     for p in perm:
        #         loss_per_sample.append(torch.sqrt(torch.sum((coords_true[batch] - coords_pred[batch, p, :]) ** 2, dim=1)).mean())
        #     loss.append(torch.min(torch.stack(loss_per_sample, dim=0)))
        loss = torch.min(torch.mean(torch.sqrt(torch.sum((torch.tile(coords_true[:, None, :, :], (1, M, 1, 1)) - coords_pred[:, perm]) ** 2, dim=-1)), dim=-1), dim=-1)
        # if (loss_[0] != torch.stack(loss, dim=0)).all():
        #     raise ValueError("Error in Cartesian Loss")
        return torch.mean(loss[0])

def set_criterions(criterion_name: str, balance_factor: float = 0.6):
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
        global BALANCE_FACTOR
        BALANCE_FACTOR = 0
        criterion = RMSELoss()
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
