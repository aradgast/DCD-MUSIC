"""
Implements the loss functions used for training the model.
RMSPELoss: Root Mean Square Periodic Error loss function.
CartesianLoss: Cartesian loss function.
MusicSpectrumLoss: Music Spectrum loss function.

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
    """
    Root Mean Square Periodic Error (RMSPE) loss function.
    This loss function calculates the RMSPE between the predicted values and the target values.
    The predicted values and target values are expected to be in radians.
    In case of Near field, the arguments could be the predicted ranges and the target ranges in addition to the angles.
    The minimal rmse results is used over the angles and projected to the range.
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
        rmspe_angle_all_permutations = np.sqrt(1 / num_sources) * torch.linalg.norm(err_angle, dim=-1)
        if ranges is None:
            rmspe, min_idx = torch.min(rmspe_angle_all_permutations, dim=-1)
        else:
            rmspe_angle, min_idx = torch.min(rmspe_angle_all_permutations, dim=-1)
            # create the projected permutation using the min_idx
            projected_permutations = torch.tensor(perm, dtype=torch.long, device=device)[min_idx]
            projected_ranges_pred = torch.gather(ranges_pred, 1, projected_permutations)
            projected_err_ranges = projected_ranges_pred - ranges
            projected_rmse_ranges = np.sqrt(1 / num_sources) * torch.linalg.norm(projected_err_ranges, dim=-1)


            rmspe = self.balance_factor * rmspe_angle + (1 - self.balance_factor) * projected_rmse_ranges
        result = torch.sum(rmspe)
        if ranges is None:
            return result
        else:
            result_angle = torch.sum(rmspe_angle)
            result_distance = torch.sum(projected_rmse_ranges)
            return result, result_angle, result_distance

    def adjust_balance_factor(self, loss=None):
        self.balance_factor = 0.1


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
            random_angles = torch.distributions.uniform.Uniform(-torch.pi / 3, torch.pi / 3).sample(
                [angles_pred.shape[0], M - angles_pred.shape[1]])
            random_ranges = torch.distributions.uniform.Uniform(torch.min(ranges).item(),
                                                                torch.max(ranges).item()).sample(
                [angles_pred.shape[0], M - angles_pred.shape[1]])
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
        cartesian_distance_all_permutations = torch.sqrt(torch.sum(error ** 2, dim=-1))
        mean_cartesian_distance_all_permutations = torch.mean(cartesian_distance_all_permutations, dim=-1)
        mean_cartesian_distance = torch.min(mean_cartesian_distance_all_permutations, dim=-1)
        return torch.sum(mean_cartesian_distance[0])


class MusicSpectrumLoss(nn.Module):
    def __init__(self, array: torch.Tensor, sensors_distance: float, mode:str = "inverse_spectrum",
                 aggregate: str = "sum"):
        super(MusicSpectrumLoss, self).__init__()
        self.array = array
        self.sensors_distance = sensors_distance
        self.number_sensors = array.shape[0]
        if mode not in ["spectrum", "inverse_spectrum"]:
            raise Exception(f"MusicSpectrumLoss: mode {mode} is not defined")
        self.mode = mode
        self.aggregate = aggregate


    def forward(self, **kwargs):
        if "ranges" in kwargs:
            return self.__forward_with_ranges(**kwargs)
        else:
            return self.__forward_without_ranges(**kwargs)

    def __forward_without_ranges(self, **kwargs):
        angles = kwargs["angles"].unsqueeze(-1)
        time_delay = torch.einsum("nm, ban -> ban",
                                  self.array,
                                  torch.sin(angles).repeat(1, 1, self.number_sensors) * self.sensors_distance)
        search_grid = torch.exp(-2 * 1j * torch.pi * time_delay)
        var1 = torch.bmm(search_grid.conj(), kwargs["noise_subspace"].to(torch.complex128))
        inverse_spectrum = torch.norm(var1, dim=-1)
        if self.mode == "inverse_spectrum":
            loss = torch.sum(inverse_spectrum, dim=1).sum()
        elif self.mode == "spectrum":
            loss = -torch.sum(1 / inverse_spectrum, dim=1).sum()
        return loss

    def __forward_with_ranges(self, **kwargs):
        angles = kwargs["angles"][:, :, None]
        ranges = kwargs["ranges"][:, :, None].to(torch.float64)
        array_square = torch.pow(self.array, 2).to(torch.float64)
        noise_subspace = kwargs["noise_subspace"].to(torch.complex128)

        first_order = torch.einsum("nm, bna -> bna",
                                   self.array,
                                   torch.sin(angles).repeat(1, 1, self.number_sensors).transpose(1, 2) * self.sensors_distance)

        second_order = -0.5 * torch.div(torch.pow(torch.cos(angles) * self.sensors_distance, 2),
                                        ranges)
        second_order = second_order.repeat(1, 1, self.number_sensors)
        second_order = torch.einsum("nm, bna -> bna", array_square, second_order.transpose(1, 2))

        time_delay = first_order + second_order

        search_grid = torch.exp(2 * -1j * torch.pi * time_delay)
        var1 = torch.einsum("bak, bkl -> bal",
                            search_grid.conj().transpose(1, 2)[:, :, :noise_subspace.shape[1]],
                            noise_subspace)
        # get the norm value for each element in the batch.
        inverse_spectrum = torch.norm(var1, dim=-1) ** 2
        if self.mode == "inverse_spectrum":
            loss = torch.sum(inverse_spectrum, dim=-1)
        elif self.mode == "spectrum":
            loss = -torch.sum(1 / inverse_spectrum, dim=-1)
        else:
            raise Exception(f"MusicSpectrumLoss: mode {self.mode} is not defined")

        if self.aggregate == "sum":
            return torch.sum(loss)
        elif self.aggregate == "mean":
            return torch.mean(loss)
        else:
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
    # elif criterion_name.startswith("mspe"):
    #     criterion = MSPELoss()
    elif criterion_name.startswith("mse"):
        criterion = nn.MSELoss()
    elif criterion_name.startswith("rmse"):
        criterion = RMSELoss()
    elif criterion_name.startswith("cartesian"):
        criterion = CartesianLoss()
    else:
        raise Exception(f"criterions.set_criterions: Criterion {criterion_name} is not defined")
    print(f"set_criterions: Loss measure for evaluation = {criterion._get_name()}")
    return criterion


if __name__ == "__main__":
    prediction = torch.tensor([1, 2, 3])
    print(permute_prediction(prediction))
