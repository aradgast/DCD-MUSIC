"""
This script is used to run the simulation with the given parameters. The parameters can be set in the script or
by using the command line arguments. The script will run the simulation with the given parameters and save the
results to the results folder. The results will include the learning curves, RMSE results, and the accuracy results
of the evaluation. The results will be saved in the results folder in the project directory.

The script can be run with the following command line arguments:
    --snr: SNR value
    --N: Number of antennas
    --M: Number of sources
    --field_type: Field type
    --signal_nature: Signal nature
    --model_type: Model type
    --train: Train model
    --train_criteria: Training criteria
    --eval: Evaluate model
    --eval_criteria: Evaluation criteria
    --samples_size: Samples size
    --train_test_ratio: Train test ratio

"""
# Imports
import os
from curses.ascii import isalpha

from src.training import *
from run_simulation import run_simulation
import argparse

# Initialization
os.system("cls||clear")
plt.close("all")

scenario_dict = {
    # "SNR": [-10, 0, 10],
    # "T": [10, 20, 50, 70, 100],
    # "eta": [0.0, 0.01, 0.02, 0.03, 0.04],
    # "M": [2, 3, 4, 5, 6, 7],
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
    "SAVE_PLOTS": True,  # if True, the plots will be saved to the results folder
}

system_model_params = {
    "N": 15,  # number of antennas
    "M": 2,  # number of sources
    "T": 2,  # number of snapshots
    "snr": 20,  # if defined, values in scenario_dict will be ignored
    "field_type": "Near",  # Near, Far
    "signal_type": "Narrowband",  # Narrowband, broadband
    "signal_nature": "non-coherent",  # if defined, values in scenario_dict will be ignored
    "eta": 0.0,  # steering vector uniform error variance
    "bias": 0, # steering vector bias error
    "sv_noise_var": 0.0, # steering vector addative gaussian error noise variance
    "doa_range": 55, # The range of the DOA values [-doa_range, doa_range]
    "doa_resolution": 1, # The resolution of the DOA values in degrees
    "max_range_ratio_to_limit": 0.5, # The ratio of the maximum range in respect to the Fraunhofer distance
    "range_resolution": 1, # The resolution of the range values in meters
    "wavelength": 1, # The carrier wavelength of the signal in meters
}
model_config = {
    "model_type": "SubspaceNet",  # SubspaceNet, DCD-MUSIC, DeepCNN, TransMUSIC, DR_MUSIC
    "model_params": {}
}
if model_config.get("model_type") == "SubspaceNet":
    model_config["model_params"]["diff_method"] = "music_2D"  # esprit, music_1D, music_2D
    model_config["model_params"]["train_loss_type"] = "music_spectrum"  # music_spectrum, rmspe
    model_config["model_params"]["tau"] = 8
    model_config["model_params"]["field_type"] = "Near"  # Far, Near

elif model_config.get("model_type") == "DCD-MUSIC":
    model_config["model_params"]["tau"] = 8
    model_config["model_params"]["diff_method"] = ("esprit", "music_1D")  # ("esprit", "music_1D")
    model_config["model_params"]["train_loss_type"] = ("rmspe", "rmspe")  # ("rmspe", "rmspe"), ("rmspe",
    # "music_spectrum"), ("music_spectrum", "rmspe")

elif model_config.get("model_type") == "DeepCNN":
    model_config["model_params"]["grid_size"] = 361

training_params = {
    "samples_size": 100,
    "train_test_ratio": 1,
    "training_objective": "angle, range",  # angle, range, source_estimation
    "batch_size": 1,
    "epochs": 30,
    "optimizer": "Adam",  # Adam, SGD
    "scheduler": "StepLR",  # StepLR, ReduceLROnPlateau
    "learning_rate": 0.001,
    "weight_decay": 1e-9,
    "step_size": 20,
    "gamma": 0.5,
    "true_doa_train": None,  # if set, this doa will be set to all samples in the train dataset
    "true_range_train": None,  # if set, this range will be set to all samples in the train dataset
    "true_doa_test": None,  # if set, this doa will be set to all samples in the test dataset
    "true_range_test": None,  # if set, this range will be set to all samples in the train dataset
    "use_wandb": False,
    "simulation_name": None,
}
evaluation_params = {
    "models": {
        # "DCD-MUSIC(RMSPE, diffMUSIC)": {"tau": 8,
        #              "diff_method": ("esprit", "music_1D"),
        #              "train_loss_type": ("rmspe", "rmspe")},
        # "DCD-MUSIC(MusicSpec, diffMUSIC)": {"tau": 8,
        #                    "diff_method": ("music_1D", "music_1D"),
        #                    "train_loss_type": ("music_spectrum", "rmspe")},
        # "DCD-MUSIC(RMSPE, MusicSpec)": {"tau": 8,
        #                    "diff_method": ("esprit", "music_1D"),
        #                    "train_loss_type": ("rmspe", "music_spectrum")},
        # "SubspaceNet": {"tau": 8,
        #                 "diff_method": "music_2D",
        #                 "train_loss_type": "music_spectrum",
        #                 "field_type": "near"},
        # "TransMUSIC": {},
    },
    "augmented_methods": [
        # ("SubspaceNet", "beamformer", {"tau": 8, "diff_method": "music_2D", "train_loss_type": "music_spectrum", "field_type": "near"}),
        # ("SubspaceNet", "beamformer", {"tau": 8, "diff_method": "esprit", "train_loss_type": "rmspe", "field_type": "far"}),
        # ("SubspaceNet", "esprit", {"tau": 8, "diff_method": "esprit", "train_loss_type": "rmspe", "field_type": "far"}),
    ],
    "subspace_methods": [
        # "ESPRIT",
        # "1D-MUSIC",
        # "Root-MUSIC",
        # "Beamformer",
        # "2D-MUSIC",
        # "TOPS",
        "CS_Estimator",
        # "CCRB"
    ]
}



def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation with optional parameters.")
    parser.add_argument('-s', "--snr" ,type=int, help='SNR value', default=None)
    parser.add_argument('-n', "--number_of_sensors",type=int, help='Number of antennas', default=None)
    parser.add_argument('-m', "--number_of_sources", help='Number of sources, could be int or tuple for random case', default=None)
    parser.add_argument('-ft', '--field_type', type=str, help='Field type, far or near field.', default=None)
    parser.add_argument('-sn','--signal_nature', type=str, help='Signal nature; non-coherent or coherent', default=None)
    parser.add_argument('-mt','--model_type', type=str, help='Model type; SubspaceNet, DCD-MUSIC, DeepCNN, TransMUSIC, DR_MUSIC', default=None)
    parser.add_argument('-t', '--train', type=bool, help='Train model', default=None)
    parser.add_argument('-to','--training_objective', type=str, help='Training objective; angle, range or angle, range.', default=None)
    parser.add_argument('-e','--eval', type=bool, help='Evaluate model', default=None)
    parser.add_argument('-ss','--samples_size', type=int, help='Samples size', default=None)
    parser.add_argument('-ttr', '--train_test_ratio', type=float, help='Train test ratio', default=None)
    parser.add_argument('-w', '--wandb', type=bool, help='Use wandb', default=None)
    return parser.parse_args()


if __name__ == "__main__":
    # torch.set_printoptions(precision=12)

    args = parse_arguments()
    if args.snr is not None:
        system_model_params["snr"] = args.snr
    if args.number_of_sensors is not None:
        system_model_params["N"] = args.number_of_sensors
    if args.number_of_sources is not None:
        # catch a case of random number of sources between two possible values
        str_m = args.number_of_sources
        if isalpha(str_m):
            system_model_params["M"] = int(str_m)
        else:
            system_model_params["M"] = tuple(map(int, str_m.split(',')))
    if args.field_type is not None:
        system_model_params["field_type"] = args.field_type
    if args.signal_nature is not None:
        system_model_params["signal_nature"] = args.signal_nature
    if args.model_type is not None:
        model_config["model_type"] = args.model_type
    if args.train is not None:
        simulation_commands["TRAIN_MODEL"] = args.train
    if args.training_objective is not None:
        training_params["training_objective"] = args.training_objective
    if args.eval is not None:
        simulation_commands["EVALUATE_MODE"] = args.eval
    if args.samples_size is not None:
        training_params["samples_size"] = args.samples_size
    if args.train_test_ratio is not None:
        training_params["train_test_ratio"] = args.train_test_ratio
    if args.wandb is not None:
        training_params["use_wandb"] = args.wandb

    start = time.time()
    loss = run_simulation(simulation_commands=simulation_commands,
                          system_model_params=system_model_params,
                          model_config=model_config,
                          training_params=training_params,
                          evaluation_params=evaluation_params,
                          scenario_dict=scenario_dict)
    print("Total time: ", time.time() - start)
