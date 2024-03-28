"""Subspace-Net main script 
    Details
    -------
    Name: main.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 30/06/23

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
import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings
from src.system_model import SystemModelParams
from src.signal_creation import *
from src.data_handler import *
from src.criterions import set_criterions
from src.training import *
from src.evaluation import evaluate
from src.plotting import initialize_figures
from pathlib import Path
from src.models import ModelGenerator
from run_simulation import run_simulation

# Initialization
warnings.simplefilter("ignore")
os.system("cls||clear")
plt.close("all")



if __name__ == "__main__":
    # torch.set_printoptions(precision=12)
    now = datetime.now()
    plot_path = Path.cwd() / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    # torch.set_printoptions(precision=12)
    scenrio_dict = {"coherent": [5, 10, 15, 20, 25],
                    "non-coherent": [-5, 0, 5, 10, 15]}
    res = {}
    for mode, snr_list in scenrio_dict.items():
        res[mode] = {}
        for snr in snr_list:
            system_model_params = {
                "N": 5,
                "M": 2,
                "T": 100,
                "snr": snr,
                "field_type": "Near",
                "signal_nature": mode,
                "eta": 0,
                "bias": 0,
                "sv_noise_var": 0
            }
            model_config = {
                "model_type": "SubspaceNet",
                "diff_method": "music_1D",
                "tau": 8
            }
            training_params = {
                "samples_size": 50000,
                "train_test_ratio": 0.1,
                "training_objective": "range",
                "batch_size": 256,
                "epochs": 150,
                "optimizer": "Adam",
                "learning_rate": 0.0001,
                "weight_decay": 1e-9,
                "step_size": 70,
                "gamma": 0.5
            }
            evaluation_params = {
                "criterion": "rmspe"
            }
            simulation_commands = {
                "SAVE_TO_FILE": False,
                "CREATE_DATA": True,
                "LOAD_MODEL": False,
                "TRAIN_MODEL": True,
                "SAVE_MODEL": False,
                "EVALUATE_MODE": True
            }

            res[mode][snr] = run_simulation(simulation_commands,
                                            system_model_params,
                                            model_config,
                                            training_params,
                                            evaluation_params)
    for mode, snr_dict in res.items():
        if snr_dict:
            plt.figure()
            snr_values = snr_dict.keys()
            plt_res = {}
            for snr, results in snr_dict.items():
                for method, res in results.items():
                    if method not in plt_res:
                        plt_res[method] = []
                    plt_res[method].append(res["Overall"])

            plt.title(f"{system_model_params["M"]} {mode} sources results")
            for method, res in plt_res.items():
                plt.scatter(snr_values, res, label=method)
            plt.legend()
            plt.grid()
            plt.xlabel("SNR [dB]")
            plt.ylabel("RMSE [m]")
            plt.savefig(os.path.join(plot_path, f"{mode}_sources_results_{dt_string_for_save}.png"))
            plt.show()

