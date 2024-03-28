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


def run_simulation(*args):
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
    simulation_commands = args[0]
    save_to_file = simulation_commands["SAVE_TO_FILE"]  # Saving results to file or present them over CMD
    create_data = simulation_commands["CREATE_DATA"]  # Creating new dataset
    load_model = simulation_commands["LOAD_MODEL"]  # Load specific model for training
    train_model = simulation_commands["TRAIN_MODEL"]  # Applying training operation
    save_model = simulation_commands["SAVE_MODEL"]  # Saving tuned model
    evaluate_mode = simulation_commands["EVALUATE_MODE"]  # Evaluating desired algorithms
    load_data = not (create_data)  # Loading data from exist dataset

    # Saving simulation scores to external file
    if save_to_file:
        file_path = (
                simulations_path / "results" / "scores" / Path(dt_string_for_save + ".txt")
        )
        sys.stdout = open(file_path, "w")
    # Define system model parameters
    SYSTEM_MODEL_PARAMS = args[1]
    system_model_params = (
        SystemModelParams()
        .set_parameter("N", SYSTEM_MODEL_PARAMS["N"])
        .set_parameter("M", SYSTEM_MODEL_PARAMS["M"])
        .set_parameter("T", SYSTEM_MODEL_PARAMS["T"])
        .set_parameter("snr", SYSTEM_MODEL_PARAMS["snr"])
        .set_parameter("field_type", SYSTEM_MODEL_PARAMS["field_type"])
        .set_parameter("signal_nature", SYSTEM_MODEL_PARAMS["signal_nature"])
        .set_parameter("eta", SYSTEM_MODEL_PARAMS["eta"])
        .set_parameter("bias", SYSTEM_MODEL_PARAMS["bias"])
        .set_parameter("sv_noise_var", SYSTEM_MODEL_PARAMS["sv_noise_var"])
    )
    system_model = SystemModel(system_model_params)
    # Generate model configuration
    MODEL_CONFIG = args[2]
    model_config = (
        ModelGenerator(system_model)
        .set_model_type(MODEL_CONFIG["model_type"])
        .set_field_type(system_model_params.field_type)
        .set_diff_method(MODEL_CONFIG["diff_method"])
        .set_tau(MODEL_CONFIG["tau"])
        .set_model(system_model_params)
    )
    # Define samples size
    TRAINING_PARAMS = args[3]
    samples_size = TRAINING_PARAMS["samples_size"]  # Overall dateset size
    train_test_ratio = TRAINING_PARAMS["train_test_ratio"]  # training and testing datasets ratio
    # Sets simulation filename
    simulation_filename = get_simulation_filename(system_model_params=system_model_params,
                                                  model_config=model_config)
    # Print new simulation intro
    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")

    if load_data:
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
            create_data = True
            load_data = False
    if create_data and not load_data:
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

    if train_model:
        # Assign the training parameters object
        simulation_parameters = (
            TrainingParams()
            .set_training_objective(TRAINING_PARAMS["training_objective"])
            .set_batch_size(TRAINING_PARAMS["batch_size"])
            .set_epochs(TRAINING_PARAMS["epochs"])
            .set_model(model=model_config)
            .set_optimizer(optimizer=TRAINING_PARAMS["optimizer"], learning_rate=TRAINING_PARAMS["learning_rate"],
                           weight_decay=TRAINING_PARAMS["weight_decay"])
            .set_training_dataset(train_dataset)
            .set_schedular(step_size=TRAINING_PARAMS["step_size"], gamma=TRAINING_PARAMS["gamma"])
            .set_criterion()

        )
        if load_model:
            try:
                simulation_parameters.load_model(
                    loading_path=saving_path / "final_models" / get_model_filename(system_model_params, model_config))
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
        if save_model:
            torch.save(
                model.state_dict(),
                saving_path / "final_models" / Path(
                    get_model_filename(model_config=model_config, system_model_params=system_model_params)),
            )
        # Plots saving
        if save_to_file:
            plt.savefig(
                simulations_path
                / "results"
                / "plots"
                / Path(dt_string_for_save + r".png")
            )
        else:
            plt.show()

        # Evaluation stage
        EVALUTION_PARAMS = args[4]
        if evaluate_mode:
            # Initialize figures dict for plotting
            figures = initialize_figures()
            # Define loss measure for evaluation
            criterion, subspace_criterion = set_criterions(EVALUTION_PARAMS["criterion"])
            # Load datasets for evaluation
            if not (create_data or load_data):
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
            if not train_model:
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
        if loss is not None:
            return loss


if __name__ == "__main__":
    now = datetime.now()
    plot_path = Path.cwd() / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    # torch.set_printoptions(precision=12)
    scenrio_dict = {"coherent": [0, 5, 10, 15],
                    "non-coherent": []}
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
                "signal_nature": mode
            }
            model_config = {
                "model_type": "SubspaceNet",
                "diff_method": "music_1D",
                "tau": 8
            }
            training_params = {
                "samples_size": 1024,
                "train_test_ratio": 0.1,
                "training_objective": "range",
                "batch_size": 32,
                "epochs": 100,
                "optimizer": "Adam",
                "learning_rate": 0.0001,
                "weight_decay": 1e-9,
                "step_size": 80,
                "gamma": 0.1
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
