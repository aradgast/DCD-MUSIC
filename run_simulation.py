# Imports
import sys
import os
from src.data_handler import *
from src.training import *
from src.evaluation import evaluate
from src.plotting import initialize_figures
from pathlib import Path
from src.models import ModelGenerator


def __run_simulation(**kwargs):
    SIMULATION_COMMANDS = kwargs["simulation_commands"]
    SYSTEM_MODEL_PARAMS = kwargs["system_model_params"]
    MODEL_CONFIG = kwargs["model_config"]
    TRAINING_PARAMS = kwargs["training_params"]
    EVALUATION_PARAMS = kwargs["evaluation_params"]
    scenario_dict = kwargs["scenario_dict"]
    save_to_file = SIMULATION_COMMANDS["SAVE_TO_FILE"]  # Saving results to file or present them over CMD
    create_data = SIMULATION_COMMANDS["CREATE_DATA"]  # Creating new dataset
    load_model = SIMULATION_COMMANDS["LOAD_MODEL"]  # Load specific model for training
    train_model = SIMULATION_COMMANDS["TRAIN_MODEL"]  # Applying training operation
    save_model = SIMULATION_COMMANDS["SAVE_MODEL"]  # Saving tuned model
    evaluate_mode = SIMULATION_COMMANDS["EVALUATE_MODE"]  # Evaluating desired algorithms
    load_data = not create_data  # Loading data from exist dataset
    print("Running simulation...")
    if train_model:
        print("Training model - ", MODEL_CONFIG.get('model_type'))
        print("Training objective - ", TRAINING_PARAMS.get('training_objective'))
    print("Scenrios that will be tested: ")
    M = SYSTEM_MODEL_PARAMS.get("M")
    if M is None:
        M = "Different number of"
    for mode, snr_list in scenario_dict.items():
        if snr_list:
            print(f"{M} {mode} sources: {snr_list}")

    now = datetime.now()
    plot_path = Path(__file__).parent / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    # torch.set_printoptions(precision=12)

    res = {}
    for mode, snr_list in scenario_dict.items():
        res[mode] = {}
        for snr in snr_list:
            # Initialize seed
            set_unified_seed()

            # Initialize paths
            external_data_path = Path(__file__).parent / "data"
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
            (saving_path / "final_models").mkdir(parents=True, exist_ok=True)

            # Initialize time and date
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
            # Operations commands



            # Saving simulation scores to external file
            suffix = ""
            if train_model:
                suffix += f"_train_{MODEL_CONFIG.get('model_type')}_{TRAINING_PARAMS.get('training_objective')}"
            suffix += f"_{mode}_SNR_{snr}.txt"

            if save_to_file:
                orig_stdout = sys.stdout
                file_path = (
                        simulations_path / "results" / "scores" / Path(dt_string_for_save + suffix)
                )
                sys.stdout = open(file_path, "w")
            # Define system model parameters
            system_model_params = (
                SystemModelParams()
                .set_parameter("N", SYSTEM_MODEL_PARAMS["N"])
                .set_parameter("M", SYSTEM_MODEL_PARAMS["M"])
                .set_parameter("T", SYSTEM_MODEL_PARAMS["T"])
                .set_parameter("snr", snr)
                .set_parameter("field_type", SYSTEM_MODEL_PARAMS["field_type"])
                .set_parameter("signal_nature", mode)
                .set_parameter("eta", SYSTEM_MODEL_PARAMS["eta"])
                .set_parameter("bias", SYSTEM_MODEL_PARAMS["bias"])
                .set_parameter("sv_noise_var", SYSTEM_MODEL_PARAMS["sv_noise_var"])
            )
            system_model = SystemModel(system_model_params)
            # Generate model configuration
            model_config = (
                ModelGenerator()
                .set_model_type(MODEL_CONFIG.get("model_type"))
                .set_system_model(system_model)
                .set_model_params(MODEL_CONFIG.get("model_params"))
                .set_model()
            )
            # Define samples size
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
                    print("#############################################")
                    print("Error loading datasets")
                    print("#############################################")
                    create_data = True
                    load_data = False
            if create_data and not load_data:
                # Define which datasets to generate
                create_training_data = True  # Flag for creating training data
                create_testing_data = True  # Flag for creating test data
                print("Creating Data...")
                if create_training_data:
                    # Generate training dataset
                    # train_dataset, _, _ = create_dataset(
                    train_dataset, _ = create_dataset(
                        system_model_params=system_model_params,
                        samples_size=samples_size,
                        model_type=model_config.model_type,
                        save_datasets=True,
                        datasets_path=datasets_path,
                        true_doa=TRAINING_PARAMS["true_doa_train"],
                        true_range=TRAINING_PARAMS["true_range_train"],
                        phase="train",
                    )
                if create_testing_data:
                    # Generate test dataset
                    # test_dataset, generic_test_dataset, samples_model = create_dataset(
                    generic_test_dataset, samples_model = create_dataset(
                        system_model_params=system_model_params,
                        samples_size=int(train_test_ratio * samples_size),
                        model_type=model_config.model_type,
                        save_datasets=True,
                        datasets_path=datasets_path,
                        true_doa=TRAINING_PARAMS["true_doa_test"],
                        true_range=TRAINING_PARAMS["true_range_test"],
                        phase="test",
                    )

            if train_model:
                # Assign the training parameters object
                simulation_parameters = (
                    TrainingParams()
                    .set_training_objective(TRAINING_PARAMS["training_objective"])
                    .set_batch_size(TRAINING_PARAMS["batch_size"])
                    .set_epochs(TRAINING_PARAMS["epochs"])
                    .set_model(model_gen=model_config)
                    .set_optimizer(optimizer=TRAINING_PARAMS["optimizer"],
                                   learning_rate=TRAINING_PARAMS["learning_rate"],
                                   weight_decay=TRAINING_PARAMS["weight_decay"])
                    .set_training_dataset(train_dataset)
                    .set_schedular(step_size=TRAINING_PARAMS["step_size"],
                                   gamma=TRAINING_PARAMS["gamma"])
                    .set_criterion(TRAINING_PARAMS["criterion"], TRAINING_PARAMS["balance_factor"])

                )
                if load_model:
                    try:
                        simulation_parameters.load_model(
                            loading_path=saving_path / "final_models" / model_config.model.get_model_file_name())
                        # if isinstance(simulation_parameters.model, CascadedSubspaceNet):
                        #     simulation_parameters.model._load_state_for_angle_extractor()
                    except Exception as e:
                        print("#############################################")
                        print(e)
                        print("simulation_parameters.load_model: Error loading model")
                        print("#############################################")

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
                    save_figures=save_to_file,
                )
                # Save model weights
                if save_model:
                    torch.save(model.state_dict(),
                            saving_path / "final_models" / Path(model.get_model_file_name()))
                # # Plots saving
                # if save_to_file:
                #     plt.savefig(
                #         simulations_path
                #         / "results"
                #         / "plots"
                #         / Path(dt_string_for_save + r".png")
                #     )
                # else:
                #     plt.show()

            # Evaluation stage
            if evaluate_mode:
                if not train_model:
                    model = None
                # Initialize figures dict for plotting
                figures = initialize_figures()
                # Define loss measure for evaluation
                criterion, subspace_criterion = set_criterions(EVALUATION_PARAMS["criterion"],
                                                               EVALUATION_PARAMS["balance_factor"])
                # Load datasets for evaluation
                if not create_data or load_data:
                            generic_test_dataset, samples_model = load_datasets(
                        system_model_params=system_model_params,
                        model_type=model_config.model_type,
                        samples_size=samples_size,
                        datasets_path=datasets_path,
                        train_test_ratio=train_test_ratio,
                    )

                # Generate DataLoader objects
                # model_test_dataset = torch.utils.data.DataLoader(
                #     test_dataset, batch_size=32, shuffle=False, drop_last=True
                # )
                batch_sampler_test = SameLengthBatchSampler(generic_test_dataset, batch_size=8)
                generic_test_dataset = torch.utils.data.DataLoader(generic_test_dataset,
                                                                   collate_fn=collate_fn,
                                                                   batch_sampler=batch_sampler_test,
                                                                   shuffle=False)
                # generic_test_dataset = torch.utils.data.DataLoader(generic_test_dataset,
                #                                                    shuffle=False, drop_last=False,
                #                                                      batch_size=32)
                # Evaluate DNN models, augmented and subspace methods
                loss = evaluate(
                    generic_test_dataset=generic_test_dataset,
                    criterion=criterion,
                    system_model=system_model,
                    figures=figures,
                    plot_spec=False,
                    models=EVALUATION_PARAMS["models"],
                    augmented_methods=EVALUATION_PARAMS["augmented_methods"],
                    subspace_methods=EVALUATION_PARAMS["subspace_methods"],
                    model_tmp=model
                )
                plt.show()
                print("end")
                res[mode][snr] = loss
                if save_to_file:
                    sys.stdout.close()
                    sys.stdout = orig_stdout
    if res is not None:
        if SIMULATION_COMMANDS["PLOT_RESULTS"] == True:
            plt_acc = False
            for signal_nature, snr_dict in res.items():
                if snr_dict:
                    # plt.figure()
                    if isinstance(criterion, RMSPELoss):
                        if "Distance" in snr_dict[list(snr_dict.keys())[0]].keys():
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
                            snr_values = snr_dict.keys()
                            plt_res = {}
                            for snr, results in snr_dict.items():
                                for method, loss_ in results.items():
                                    if plt_res.get(method) is None:
                                        if plt_res[method].get("Accuracy") is not None:
                                            key = method + f": {loss_['Accuracy'] * 100:.2f} %"
                                        else:
                                            key = method
                                        plt_res[key] = {"Angle": [], "Distance": []}
                                    # plt_res[method].append(loss_["Overall"])
                                    plt_res[key]["Angle"].append(loss_["Angle"])
                                    plt_res[key]["Distance"].append(loss_["Distance"])
                                    if loss_.get("Accuracy") is not None:
                                        if "Accuracy" not in plt_res[key].keys():
                                            plt_res[key]["Accuracy"] = []
                                            plt_acc = True
                                        plt_res[key]["Accuracy"].append(loss_["Accuracy"])
                            if SYSTEM_MODEL_PARAMS.get("M") is None:
                                suptitle = "Different number of "
                            else:
                                suptitle = f"{SYSTEM_MODEL_PARAMS['M']} "
                            suptitle += f"{signal_nature} sources results"
                            fig.suptitle(suptitle)
                            idx = 0
                            for method, loss_ in plt_res.items():
                                if method == "CRB":
                                    line_style = "-."
                                else:
                                    line_style = None
                                ax1.plot(snr_values, loss_["Angle"], label=method, linestyle=line_style, marker=MARKER_DICT[idx])
                                ax2.plot(snr_values, loss_["Distance"], label=method, linestyle=line_style, marker=MARKER_DICT[idx])
                                idx += 1
                            ax1.legend()
                            ax2.legend()
                            ax1.grid()
                            ax2.grid()
                            ax1.set_xlabel("SNR [dB]")
                            ax2.set_xlabel("SNR [dB]")
                            ax1.set_ylabel("RMSE [rad]")
                            ax2.set_ylabel("RMSE [m]")
                            ax1.set_title("Angle RMSE")
                            ax2.set_title("Distance RMSE")
                            ax1.set_yscale("log")
                            ax2.set_yscale("log")
                            fig.savefig(os.path.join(simulations_path,
                                                     "results",
                                                     "plots",
                                                     f"summary_{signal_nature}_sources_results_{dt_string_for_save}.png"))
                            fig.show()
                            if plt_acc:
                                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                                idx = 0
                                for method, loss_ in plt_res.items():
                                    if loss_.get("Accuracy") is not None:
                                        ax.plot(snr_values, loss_["Accuracy"], label=method, linestyle=line_style, marker=MARKER_DICT[idx])
                                        idx += 1
                                ax.legend()
                                ax.grid()
                                ax.set_xlabel("SNR [dB]")
                                ax.set_ylabel("Accuracy [%]")
                                ax.set_title("Accuracy")
                                ax.set_yscale("linear")
                                fig.savefig(os.path.join(simulations_path,
                                                         "results",
                                                         "plots",
                                                         f"summary_acc_{signal_nature}_sources_results_{dt_string_for_save}.png"))
                                fig.show()
                        else: #FAR
                            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                            snr_values = snr_dict.keys()
                            plt_res = {}
                            for snr, results in snr_dict.items():
                                for method, loss_ in results.items():
                                    if method not in plt_res:
                                        plt_res[method] = {"Overall": []}
                                    plt_res[method]["Overall"].append(loss_["Overall"])
                                    if loss_.get("Accuracy") is not None:
                                        if "Accuracy" not in plt_res[method].keys():
                                            plt_res[method]["Accuracy"] = []
                                            plt_acc = True
                                        plt_res[method]["Accuracy"].append(loss_["Accuracy"])
                            if SYSTEM_MODEL_PARAMS.get("M") is None:
                                suptitle = "Different number of "
                            else:
                                suptitle = f"{SYSTEM_MODEL_PARAMS['M']} "
                            suptitle += f"{signal_nature} sources results"
                            fig.suptitle(suptitle)
                            idx = 0
                            for method, loss_ in plt_res.items():
                                ax.plot(snr_values, loss_["Overall"], label=method, marker=MARKER_DICT[idx])
                                idx += 1
                            ax.legend()
                            ax.grid()
                            ax.set_xlabel("SNR [dB]")
                            ax.set_ylabel("RMSE [rad]")
                            ax.set_title("Overall RMSPE loss")
                            ax.set_yscale("log")
                            fig.savefig(os.path.join(simulations_path,
                                                     "results",
                                                     "plots",
                                                     f"summary_{signal_nature}_sources_results_{dt_string_for_save}.png"))
                            fig.show()
                            if plt_acc:
                                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                                idx = 0
                                for method, loss_ in plt_res.items():
                                    if loss_.get("Accuracy") is not None:
                                        ax.plot(snr_values, loss_["Accuracy"], label=method, marker=MARKER_DICT[idx])
                                ax.legend()
                                ax.grid()
                                ax.set_xlabel("SNR [dB]")
                                ax.set_ylabel("Accuracy [%]")
                                ax.set_title("Accuracy")
                                ax.set_yscale("linear")
                                fig.savefig(os.path.join(simulations_path,
                                                         "results",
                                                         "plots",
                                                         f"summary_acc_{signal_nature}_sources_results_{dt_string_for_save}.png"))
                                fig.show()

                    elif isinstance(criterion, CartesianLoss):
                        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                        snr_values = snr_dict.keys()
                        plt_res = {}
                        for snr, results in snr_dict.items():
                            for method, loss_ in results.items():
                                if plt_res.get(method) is None:
                                    plt_res[method] = {"Overall": []}
                                plt_res[method]["Overall"].append(loss_["Overall"])
                                if loss_.get("Accuracy") is not None:
                                    if "Accuracy" not in plt_res[method].keys():
                                        plt_res[method]["Accuracy"] = []
                                        plt_acc = True
                                    plt_res[method]["Accuracy"].append(loss_["Accuracy"])
                        if SYSTEM_MODEL_PARAMS.get("M") is None:
                            suptitle = "Different number of "
                        else:
                            suptitle = f"{SYSTEM_MODEL_PARAMS['M']} "
                        suptitle += f"{signal_nature} sources results"
                        fig.suptitle(suptitle)
                        idx = 0
                        for method, loss_ in plt_res.items():
                            if method == "CRB":
                                line_style = "dashdot"
                            else:
                                line_style = "-"
                            if loss_.get("Accuracy") is not None:
                                label = method + f": {np.mean(loss_['Accuracy']) * 100:.2f} %"
                            else:
                                label = method
                            if not np.isnan((loss_.get("Overall"))).any():
                                ax.plot(snr_values, loss_["Overall"], label=label, linestyle=line_style,
                                        marker=MARKER_DICT[idx], markersize=10)
                                idx += 1
                        ax.legend()
                        ax.grid()
                        ax.set_xlabel("SNR [dB]")
                        ax.set_ylabel("RMSE [m]")
                        ax.set_title("Overall RMSE - Cartesian Loss")
                        ax.set_yscale("linear")
                        fig.savefig(os.path.join(simulations_path,
                                                 "results",
                                                 "plots",
                                                 f"summary_{signal_nature}_sources_results_{dt_string_for_save}.png"))
                        fig.show()
                        if plt_acc:
                            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                            idx = 0
                            for method, loss_ in plt_res.items():
                                if loss_.get("Accuracy") is not None:
                                    ax.plot(snr_values, np.array(loss_["Accuracy"]) * 100, label=method, marker=MARKER_DICT[idx])
                                    idx += 1
                            ax.legend()
                            ax.grid()
                            ax.set_xlabel("SNR [dB]")
                            ax.set_ylabel("Accuracy [%]")
                            ax.set_title("Accuracy")
                            ax.set_yscale("linear")
                            fig.savefig(os.path.join(simulations_path,
                                                     "results",
                                                     "plots",
                                                     f"summary_acc_{signal_nature}_sources_results_{dt_string_for_save}.png"))
                            fig.show()
        if save_to_file:
            file_path = (
                    simulations_path / "results" / "scores" / Path(dt_string_for_save +"_summary" + ".txt")
            )
            sys.stdout = open(file_path, "w")
        for signal_nature, snr_dict in res.items():
            if len(snr_dict.keys()) >= 1:
                print("#" * 20, signal_nature.upper(), "#" * 20)
                for snr, results in snr_dict.items():
                    print(f"SNR = {snr} [dB]: ")
                    for method, loss in results.items():
                        txt = f"\t{method.upper(): <30}: "
                        for key, value in loss.items():
                            if value is not None:
                                if key == "Accuracy":
                                    txt += f"{key}: {value * 100:.2f} %|"
                                else:
                                    txt += f"{key}: {value:.6f} |"

                        print(txt)
        if save_to_file:
            sys.stdout.close()
            sys.stdout = orig_stdout
        return res


def run_simulation(**kwargs):
    """
    This function is used to run an end to end simulation, including creating or loading data, training or loading
    a NN model and do an evaluation for the algorithms.
    """
    if kwargs["system_model_params"]["signal_nature"] is not None and kwargs["system_model_params"]["snr"] is not None:
        kwargs["scenario_dict"] = {}
        kwargs["scenario_dict"][kwargs["system_model_params"]["signal_nature"]] = [kwargs["system_model_params"]["snr"]]

    loss = __run_simulation(**kwargs)
    if loss is not None:
        return loss


if __name__ == "__main__":
    now = datetime.now()
