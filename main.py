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
from src.training import *
from run_simulation import run_simulation
import argparse

# Initialization
os.system("cls||clear")
plt.close("all")

scenario_dict = {
    "SNR": [-10, -5, 0, 5, 10],
    "T": [10, 20, 50, 70, 100],
    "eta": [0.0, 0.01, 0.02, 0.03, 0.04],
}

system_model_params = {
    "N": 15,  # number of antennas
    "M": 2,  # number of sources
    "T": 100,  # number of snapshots
    "snr": 0,  # if defined, values in scenario_dict will be ignored
    "field_type": "Near",  # Near, Far
    "signal_nature": "non-coherent",  # if defined, values in scenario_dict will be ignored
    "eta": 0.0,  # steering vector error
    "bias": 0,
    "sv_noise_var": 0.0,
    "doa_range": 55,
    "doa_resolution": 1,
    "max_range_ratio_to_limit": 0.4,
    "range_resolution": 1,
}
model_config = {
    "model_type": "DCDMUSIC",  # SubspaceNet, DCDMUSIC, DeepCNN, TransMUSIC, DR_MUSIC
    "model_params": {}
}
if model_config.get("model_type") == "SubspaceNet":
    model_config["model_params"]["diff_method"] = "music_2D"  # esprit, music_1D, music_2D
    model_config["model_params"]["train_loss_type"] = "music_spectrum"  # music_spectrum, rmspe
    model_config["model_params"]["tau"] = 8
    model_config["model_params"]["field_type"] = "Near"  # Near, Far

elif model_config.get("model_type") == "DCDMUSIC":
    model_config["model_params"]["tau"] = 8
    model_config["model_params"]["diff_method"] = ("music_1D", "music_1D")  # ("esprit", "music_1D")
    model_config["model_params"]["train_loss_type"] = ("music_spectrum", "rmspe")  # ("rmspe", "rmspe"), ("rmspe",
    # "music_spectrum"), ("music_spectrum", "rmspe")

elif model_config.get("model_type") == "DeepCNN":
    model_config["model_params"]["grid_size"] = 361

training_params = {
    "samples_size": 100,
    "train_test_ratio": 1,
    "training_objective": "angle, range",  # angle, range, source_estimation
    "batch_size": 128,
    "epochs": 2,
    "optimizer": "Adam",  # Adam, SGD
    "learning_rate": 0.001,
    "weight_decay": 1e-9,
    "step_size": 50,
    "gamma": 0.5,
    "true_doa_train": None,  # if set, this doa will be set to all samples in the train dataset
    "true_range_train": None,  # if set, this range will be set to all samples in the train dataset
    "true_doa_test": None,  # if set, this doa will be set to all samples in the test dataset
    "true_range_test": None,  # if set, this range will be set to all samples in the train dataset
}
evaluation_params = {
    "criterion": "cartesian",  # rmse, rmspe, mse, mspe, cartesian
    "balance_factor": 1.0,
    "models": {
        "DCDMUSIC": {"tau": 8,
                     "diff_method": ("esprit", "music_1D"),
                     "train_loss_type": ("rmspe", "rmspe")},
        # "DCDMUSIC1Ortho": {"tau": 8,
        #                    "diff_method": ("esprit", "music_1D"),
        #                    "train_loss_type": ("music_spectrum", "rmspe")},
        # "DCDMUSIC2Ortho": {"tau": 8,
        #                    "diff_method": ("music_1D", "music_1D"),
        #                    "train_loss_type": ("rmspe", "music_spectrum")},
        "SubspaceNet": {"tau": 8,
                        "diff_method": "music_2D",
                        "train_loss_type": "music_spectrum",
                        "field_type": "Near"},
        # "TransMUSIC": {},
    },
    "augmented_methods": [
        # "mvdr",
        # "r-music",
        # "esprit",
        # "music",
        # "2D-MUSIC",
    ],
    "subspace_methods": [
        # "ESPRIT",
        # "1D-MUSIC",
        # "Root-MUSIC",
        "Beamformer",
        "2D-MUSIC",
        # "CCRB"
    ]
}
simulation_commands = {
    "SAVE_TO_FILE": False,
    "CREATE_DATA": True,
    "LOAD_MODEL": False,
    "TRAIN_MODEL": False,
    "SAVE_MODEL": False,
    "EVALUATE_MODE": True,
    "PLOT_RESULTS": True,  # if True, the learning curves will be plotted
    "PLOT_LOSS_RESULTS": True,  # if True, the RMSE results of evaluation will be plotted
    "PLOT_ACC_RESULTS": False,  # if True, the accuracy results of evaluation will be plotted
    "SAVE_PLOTS": False,  # if True, the plots will be saved to the results folder
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation with optional parameters.")
    parser.add_argument('--snr', type=str, help='SNR value', default=None)
    parser.add_argument('--N', type=int, help='Number of antennas', default=None)
    parser.add_argument('--M', type=int, help='Number of sources', default=None)
    parser.add_argument('--field_type', type=str, help='Field type', default=None)
    parser.add_argument('--signal_nature', type=str, help='Signal nature', default=None)
    parser.add_argument('--model_type', type=str, help='Model type', default=None)
    parser.add_argument('--train', type=int, help='Train model', default=None)
    parser.add_argument('--train_criteria', type=str, help='Training criteria', default=None)
    parser.add_argument('--eval', type=int, help='Evaluate model', default=None)
    parser.add_argument('--eval_criteria', type=str, help='Evaluation criteria', default=None)
    parser.add_argument('--samples_size', type=int, help='Samples size', default=None)
    parser.add_argument('--train_test_ratio', type=int, help='Train test ratio', default=None)
    return parser.parse_args()


if __name__ == "__main__":
    # torch.set_printoptions(precision=12)

    args = parse_arguments()
    if args.snr is not None:
        system_model_params["snr"] = int(args.snr)
    if args.N is not None:
        system_model_params["N"] = int(args.N)
    if args.M is not None:
        system_model_params["M"] = int(args.M)
    if args.field_type is not None:
        system_model_params["field_type"] = args.field_type
    if args.signal_nature is not None:
        system_model_params["signal_nature"] = args.signal_nature
    if args.model_type is not None:
        model_config["model_type"] = args.model_type
    if args.train is not None:
        simulation_commands["TRAIN_MODEL"] = args.train
    if args.train_criteria is not None:
        training_params["training_objective"] = args.train_criteria
    if args.eval is not None:
        simulation_commands["EVALUATE_MODE"] = args.eval
    if args.eval_criteria is not None:
        evaluation_params["criterion"] = args.eval_criteria
    if args.samples_size is not None:
        training_params["samples_size"] = args.samples_size
    if args.train_test_ratio is not None:
        training_params["train_test_ratio"] = args.train_test_ratio

    start = time.time()
    loss = run_simulation(simulation_commands=simulation_commands,
                          system_model_params=system_model_params,
                          model_config=model_config,
                          training_params=training_params,
                          evaluation_params=evaluation_params,
                          scenario_dict=scenario_dict)
    print("Total time: ", time.time() - start)
