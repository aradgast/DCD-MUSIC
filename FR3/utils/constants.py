import torch
import numpy as np
import os

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {DEVICE}")
C = 300 * 1e6 # speed of light meter / micro-second
MU_SEC = 10 ** (-6)  # mu seconds factor

ALG_THRESHOLD = 1.2  # ratio of signal to noise ratio for the algorithms
DOA_RES = 1 # resolution of the DOA in degrees
TIME_RES = 0.01 * MU_SEC # time resolution
BS_ORIENTATION = -np.pi / 2  # orientation of the BS

WALLS = np.array([[8, 11], [8, 15], [11, 15], [11, 11], [8, 11]]) # walls in the room
LOSS_FACTOR = {6e9: 1.1, 24e9: 100} # loss factor for the walls

SYNTHETIC_L_MAX = 4 # maximum number of paths for the synthetic channel

INIT_POWER = 20 # dBm
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "FR3")

NOISE_FIGURE = 7 # dB
N_0 = -174 # dBm
NS = 50 # number of pilot samples


# TODO: change the global variables to be read from the config file