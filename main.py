"""Subspace-Net main script 
    Details
    -------
    Name: main.py
    Authors: D. H. Shmuel, Arad Gast
    Created: 01/10/21
    Edited: 31/05/24

    Purpose
    --------
    This script allows the user to apply the proposed algorithms,
    by wrapping all the required procedures and parameters for the simulation.
    This scripts calls the following functions:
        * create_dataset: For creating training and testing datasets 
        * training: For training DR-MUSIC model
        * evaluate_dnn_model: For evaluating subspace hybrid models

    This script requires that requirements.txt will be installed within the Python
    environment you are running this script in.

"""
# Imports
import os
import warnings
from src.training import *
from run_simulation import run_simulation

# Initialization
# warnings.simplefilter("ignore")
os.system("cls||clear")
plt.close("all")

scenario_dict = {
    "coherent": [1],
    "non-coherent": [],
}

system_model_params = {
    "N": 15,                                    # number of antennas
    "M": None,                                     # number of sources
    "T": 100,                                   # number of snapshots
    "snr": None,                                # if defined, values in scenario_dict will be ignored
    "field_type": "Near",                       # Near, Far
    "signal_nature": None,                      # if defined, values in scenario_dict will be ignored
    "eta": 0,                                   # steering vector error
    "bias": 0,
    "sv_noise_var": 0
}
model_config = {
    "model_type": "SubspaceNet",                # SubspaceNet, CascadedSubspaceNet, DeepCNN, TransMUSIC, DR_MUSIC
    "model_params": {}
}
if model_config.get("model_type") == "SubspaceNet":
    model_config["model_params"]["diff_method"] = "esprit"  # esprit, music_1D, music_2D
    model_config["model_params"]["tau"] = 8
    model_config["model_params"]["field_type"] = "Far"     # Near, Far

elif model_config.get("model_type") == "CascadedSubspaceNet":
    model_config["model_params"]["tau"] = 8

elif model_config.get("model_type") == "DeepCNN":
    model_config["model_params"]["grid_size"] = 361

training_params = {
    "samples_size": 1024 * 100,
    "train_test_ratio": .05,
    "training_objective": "angle",       # angle, range
    "batch_size": 256,
    "epochs": 150,
    "optimizer": "Adam",                        # Adam, SGD
    "learning_rate": 0.0001,
    "weight_decay": 1e-9,
    "step_size": 70,
    "gamma": 0.5,
    "true_doa_train": None,                 # if set, this doa will be set to all samples in the train dataset
    "true_range_train": None,                 # if set, this range will be set to all samples in the train dataset
    "true_doa_test": None,                  # if set, this doa will be set to all samples in the test dataset
    "true_range_test": None,                   # if set, this range will be set to all samples in the train dataset
    "criterion": "rmspe",                   # rmse, rmspe, mse, mspe, bce, cartesian
    "balance_factor": 1.0                # if None, the balance factor will be set to the default value -> 0.6
}
evaluation_params = {
    "criterion": "rmspe",                       # rmse, rmspe, mse, mspe, cartesian
    "balance_factor": training_params["balance_factor"],
    "models": {
                # "CascadedSubspaceNet": {"tau": 8},
                # "SubspaceNet": {"tau": 8,
                #                 "diff_method": "esprit",
                #                 "field_type": "Far"},
                # "TransMUSIC": {},

            },
    "augmented_methods": [
        # "mvdr",
        # "r-music",
        # "esprit",
        # "music",
        # "music_2D",
    ],
    "subspace_methods": [
        # "esprit",
        # "music_1d",
        # "root_music",
        # "mvdr",
        # "sps_root_music",
        # "sps_esprit",
        # "sps_music_1d"
        # "bb-music",
        # "music_2D",
        # "sps_music_2D",
        # "CRB"
    ]
}
simulation_commands = {
    "SAVE_TO_FILE": False,
    "CREATE_DATA": False,
    "LOAD_MODEL": False,
    "TRAIN_MODEL": True,
    "SAVE_MODEL": True,
    "EVALUATE_MODE": True,
    "PLOT_RESULTS": False
}

if __name__ == "__main__":
    # torch.set_printoptions(precision=12)
    start = time.time()
    loss = run_simulation(simulation_commands=simulation_commands,
                          system_model_params=system_model_params,
                          model_config=model_config,
                          training_params=training_params,
                          evaluation_params=evaluation_params,
                          scenario_dict=scenario_dict)
    print("Total time: ", time.time() - start)

