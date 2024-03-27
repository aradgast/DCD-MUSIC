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

# Initialization
warnings.simplefilter("ignore")
os.system("cls||clear")
plt.close("all")



if __name__ == "__main__":
    # torch.set_printoptions(precision=12)
    scenrio_dict = {"coherent": [],
                    "non-coherent" : [-5, -2]}
    res = {}
    for mode, snr_list in scenrio_dict.items():
        res[mode] = {}
        for snr in snr_list:
            # Initialize seed
            set_unified_seed()

            # Initialize paths
            external_data_path = Path.cwd() / "data"
            scenario_data_path = "uniform_bias_spacing"
            datasets_path = external_data_path / "datasets" / scenario_data_path
            simulations_path = external_data_path / "simulations"
            saving_path = external_data_path / "weights"
            # create folders if not exists
            datasets_path.mkdir(parents=True, exist_ok=True)
            (datasets_path / "train").mkdir(parents=True, exist_ok=True)
            (datasets_path / "test").mkdir(parents=True, exist_ok=True)
            datasets_path.mkdir(parents=True, exist_ok=True)
            simulations_path.mkdir(parents=True, exist_ok=True)
            saving_path.mkdir(parents=True, exist_ok=True)
            # Initialize time and date
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
            # Operations commands
            commands = {"SAVE_TO_FILE": False,  # Saving results to file or present them over CMD
                        "CREATE_DATA": True,  # Creating new dataset
                        "LOAD_MODEL": True,  # Load specific model for training
                        "TRAIN_MODEL": True,  # Applying training operation
                        "SAVE_MODEL": True,  # Saving tuned model
                        "EVALUATE_MODE": True,  # Evaluating desired algorithms
                        }
            commands["LOAD_DATA"] = not (commands["CREATE_DATA"])  # Loading data from exist dataset

            # Saving simulation scores to external file
            if commands["SAVE_TO_FILE"]:
                file_path = (
                        simulations_path / "results" / "scores" / Path(dt_string_for_save + f"_{mode}_snr_{snr}" + ".txt")
                )
                sys.stdout = open(file_path, "w")
            # Define system model parameters
            system_model_params = (
                SystemModelParams()
                .set_parameter("N", 5)
                .set_parameter("M", 2)
                .set_parameter("T", 100)
                .set_parameter("snr", snr)
                .set_parameter("field_type", "Near")
                .set_parameter("signal_nature", mode)
            )
            system_model = SystemModel(system_model_params)
            # Generate model configuration
            model_config = (
                ModelGenerator(system_model)
                .set_model_type("SubspaceNet")
                .set_field_type(system_model_params.field_type)
                .set_diff_method("music_1D")
                .set_tau(8)
                .set_model(system_model_params)
            )
            # Define samples size
            samples_size = 16  # Overall dateset size
            train_test_ratio = .1  # training and testing datasets ratio
            # Sets simulation filename
            simulation_filename = get_simulation_filename(
                system_model_params=system_model_params, model_config=model_config
            )
            # Print new simulation intro
            print("------------------------------------")
            print("---------- New Simulation ----------")
            print("------------------------------------")
            print("date and time =", dt_string)

            # Datasets loading
            if commands["LOAD_DATA"]:
                try:
                    (
                        train_dataset,
                        test_dataset,
                        generic_test_dataset,
                        samples_model,
                    ) = load_datasets(
                        system_model_params=system_model_params,
                        model_type=model_config.model_type,
                        samples_size=samples_size,
                        datasets_path=datasets_path,
                        train_test_ratio=train_test_ratio,
                        is_training=True,
                    )
                except Exception as e:
                    print(e)
                    print("Dataset not found")
                    commands["CREATE_DATA"] = True
                    commands["LOAD_DATA"] = False

            # Datasets creation
            if commands["CREATE_DATA"] and not commands["LOAD_DATA"]:
                # Define which datasets to generate
                create_training_data = True  # Flag for creating training data
                create_testing_data = True  # Flag for creating test data
                print("Creating Data...")
                if create_training_data:
                    # Generate training dataset
                    train_dataset, _, _ = create_dataset(
                        system_model_params=system_model_params,
                        samples_size=samples_size,
                        model_type=model_config.model_type,
                        tau=model_config.tau,
                        save_datasets=True,
                        datasets_path=datasets_path,
                        true_doa=None,
                        true_range=None,
                        phase="train",
                    )
                if create_testing_data:
                    # Generate test dataset
                    test_dataset, generic_test_dataset, samples_model = create_dataset(
                        system_model_params=system_model_params,
                        samples_size=int(train_test_ratio * samples_size),
                        model_type=model_config.model_type,
                        tau=model_config.tau,
                        save_datasets=True,
                        datasets_path=datasets_path,
                        true_doa=None,
                        true_range=None,
                        phase="test",
                    )

            # Training stage
            if commands["TRAIN_MODEL"]:
                # Assign the training parameters object
                simulation_parameters = (
                    TrainingParams()
                    .set_training_objective("range")
                    .set_batch_size(256)
                    .set_epochs(2)
                    .set_model(model=model_config)
                    .set_optimizer(optimizer="Adam", learning_rate=0.0001, weight_decay=1e-9)
                    .set_training_dataset(train_dataset)
                    .set_schedular(step_size=100, gamma=0.5)
                    .set_criterion()

                )
                if commands["LOAD_MODEL"]:
                    try:
                        simulation_parameters.load_model(loading_path=saving_path / "final_models" / get_model_filename(system_model_params, model_config))
                    except Exception as e:
                        print(e)
                        print("Model not found.")

                # Print training simulation details
                simulation_summary(
                    system_model_params=system_model_params,
                    model_type=model_config.model_type,
                    parameters=simulation_parameters,
                    phase="training",
                )
                # Perform simulation training and evaluation stages
                model, loss_train_list, loss_valid_list = train(
                    training_parameters=simulation_parameters,
                    model_name=simulation_filename,
                    saving_path=saving_path,
                )
                # Save model weights
                if commands["SAVE_MODEL"]:
                    torch.save(
                        model.state_dict(),
                        saving_path / "final_models" / Path(get_model_filename(model_config=model_config, system_model_params=system_model_params)),
                    )
                # Plots saving
                if commands["SAVE_TO_FILE"]:
                    plt.savefig(
                        simulations_path
                        / "results"
                        / "plots"
                        / Path(dt_string_for_save + r".png")
                    )
                else:
                    plt.show()

            # Evaluation stage
            if commands["EVALUATE_MODE"]:
                # Initialize figures dict for plotting
                figures = initialize_figures()
                # Define loss measure for evaluation
                criterion, subspace_criterion = set_criterions("rmspe")
                # Load datasets for evaluation
                if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
                    test_dataset, generic_test_dataset, samples_model = load_datasets(
                        system_model_params=system_model_params,
                        model_type=model_config.model_type,
                        samples_size=samples_size,
                        datasets_path=datasets_path,
                        train_test_ratio=train_test_ratio,
                    )

                # Generate DataLoader objects
                model_test_dataset = torch.utils.data.DataLoader(
                    test_dataset, batch_size=1, shuffle=False, drop_last=False
                )
                generic_test_dataset = torch.utils.data.DataLoader(
                    generic_test_dataset, batch_size=1, shuffle=False, drop_last=False
                )
                # Load pre-trained model
                if not commands["TRAIN_MODEL"]:
                    # Define an evaluation parameters instance
                    simulation_parameters = (
                        TrainingParams()
                        .set_model(model=model_config)
                        .load_model(
                            loading_path=saving_path
                                         / "final_models"
                                         / simulation_filename
                        )
                    )
                    model = simulation_parameters.model
                # print simulation summary details
                simulation_summary(
                    system_model_params=system_model_params,
                    model_type=model_config.model_type,
                    phase="evaluation",
                    parameters=simulation_parameters,
                )
                # Evaluate DNN models, augmented and subspace methods
                loss = evaluate(
                    model=model,
                    model_type=model_config.model_type,
                    model_test_dataset=model_test_dataset,
                    generic_test_dataset=generic_test_dataset,
                    criterion=criterion,
                    subspace_criterion=subspace_criterion,
                    system_model=samples_model,
                    figures=figures,
                    plot_spec=False,
                )
            plt.show()
            print("end")

            res[mode][snr] = loss
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

            plt.title(f"{system_model_params.M} {mode} sources results")
            for method, res in plt_res.items():
                plt.scatter(snr_values, res, label=method)
            plt.legend()
            plt.grid()
            plt.xlabel("SNR [dB]")
            plt.ylabel("RMSE [m]")
            plt.show()

