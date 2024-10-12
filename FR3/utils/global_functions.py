from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares
from FR3.utils.constants import *
import random
import torch

def set_unified_seed(seed: int = 42):
    """
    Sets the seed value for random number generators in Python libraries.

    Args:
        seed (int): The seed value to set for the random number generators. Defaults to 42.

    Returns:
        None

    Raises:
        None

    Examples:
        >>> set_unified_seed(42)

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)

def optimize_to_estimate_position(bs_locs: np.ndarray, toa_values, aoa_values, medium_speed) -> np.ndarray:
    def cost_func_2d(ue: np.ndarray) -> List[float]:
        costs = []
        # LOS AOA constraints
        for i in range(len(bs_locs)):
            cost = abs(np.arctan2(ue[1] - bs_locs[i][1], ue[0] - bs_locs[i][0]) - aoa_values[i] - BS_ORIENTATION)
            costs.append(cost)
        # LOS TOA constraints
        for i in range(len(bs_locs)):
            cost = abs(np.linalg.norm(ue - bs_locs[i]) / medium_speed - toa_values[i])
            costs.append(cost)
        return costs

    # LOS computation of location in case of angle and time estimations
    # set the ue location as the bs location at first
    if isinstance(bs_locs, torch.Tensor):
        bs_locs = bs_locs.cpu().numpy()
    bs_locs = np.atleast_2d(bs_locs)
    bs_locs = list(bs_locs)
    aoa_values = list(np.atleast_1d(aoa_values))
    toa_values = list(np.atleast_1d(toa_values))
    initial_ue_loc = np.array(bs_locs[0])
    # estimate the UE using least squares
    res = least_squares(cost_func_2d, initial_ue_loc).x
    est_ue_pos, est_scatterers = res[:2], res[2:].reshape(-1, 2)
    return est_ue_pos

def power_from_dbm(dbm):
    if isinstance(dbm, list):
        dbm = np.array(dbm)
    if isinstance(dbm, np.ndarray):
        dbm = torch.from_numpy(dbm)
    return 10 ** ((dbm - 30) / 20)

def dbm_from_power(power):
    return 10 * np.log10(power) + 30


# Functions for the synthetic channel

# Return true if point C is on the left side of line AB
def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

# Return true if line segments AB and CD intersect
def intersect(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float], d: Tuple[float, float]) -> bool:
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

# Calculate the power of the path
def calc_power(P0: float, bs_loc: np.ndarray, ue_pos: np.ndarray, fc: float) -> float:
    # decrease the power if the path propagates through a wall
    for wall1, wall2 in zip(WALLS[:-1], WALLS[1:]):
        if intersect(bs_loc, ue_pos, wall1, wall2):
            P0 /= LOSS_FACTOR[fc]
    return P0

# Calculate the path loss
def compute_path_loss(toa: float, fc: float) -> float:
    # free path loss computation
    loss_db = 20 * np.log10(toa) + 20 * np.log10(fc) + 20 * np.log10(4 * np.pi) # the minus 6 is to match Tomer's code
    return 10 ** (loss_db / 20)

# Create the scatter points
def create_scatter_points(L: int) -> np.ndarray:
    scatterers = np.array([[15, 12], [15, 14], [12, 14]])

    return scatterers[:L - 1]

# Create the base station locations
def create_bs_locs(bs_ind: int) -> np.ndarray:
    bs_locs = np.array([[0, 0], [0, -5], [0, 5]])
    return bs_locs[bs_ind - 1]