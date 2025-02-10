"""
Implements the loss functions used for training the model.
RMSPELoss: Root Mean Square Periodic Error loss function.
CartesianLoss: Cartesian loss function.
MusicSpectrumLoss: Music Spectrum loss function.

"""
import torch.nn as nn
import torch
from itertools import permutations
from src.utils import *

from metrics.rmspe_loss import RMSPELoss
from metrics.rmse_loss import RMSELoss
from metrics.cartesian_loss import CartesianLoss
from metrics.music_spec_loss import MusicSpectrumLoss
from metrics.beamformer_loss import BeamFromingLoss

if __name__ == "__main__":
    prediction = torch.tensor([1, 2, 3])
