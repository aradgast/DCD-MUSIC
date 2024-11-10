# Imports
import sys
import os
from src.data_handler import *
from src.training import *
from src.plotting import *
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
    plot_mode = SIMULATION_COMMANDS["PLOT_RESULTS"]  # Plotting results
    save_plots = SIMULATION_COMMANDS["SAVE_PLOTS"]  # Saving plots
    load_data = not create_data  # Loading data from exist dataset
    print("Running simulation...")
    if train_model:
        print("Training model - ", MODEL_CONFIG.get('model_type'))
        print("Training objective - ", TRAINING_PARAMS.get('training_objective'))

    now = datetime.now()
    plot_path = Path(__file__).parent / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    # torch.set_printoptions(precision=12)

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

    # Saving simulation scores to external file
    suffix = ""
    if train_model:
        suffix += f"_train_{MODEL_CONFIG.get('model_type')}_{TRAINING_PARAMS.get('training_objective')}"
    suffix += (f"_{SYSTEM_MODEL_PARAMS['signal_nature']}_SNR_{SYSTEM_MODEL_PARAMS['snr']}_T_{SYSTEM_MODEL_PARAMS['T']}"
               f"_eta{SYSTEM_MODEL_PARAMS['eta']}.txt")

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
        .set_parameter("snr", SYSTEM_MODEL_PARAMS["snr"])
        .set_parameter("field_type", SYSTEM_MODEL_PARAMS["field_type"])
        .set_parameter("signal_nature", SYSTEM_MODEL_PARAMS["signal_nature"])
        .set_parameter("eta", SYSTEM_MODEL_PARAMS["eta"])
        .set_parameter("bias", SYSTEM_MODEL_PARAMS["bias"])
        .set_parameter("sv_noise_var", SYSTEM_MODEL_PARAMS["sv_noise_var"])
        .set_parameter("doa_range", SYSTEM_MODEL_PARAMS["doa_range"])
        .set_parameter("doa_resolution", SYSTEM_MODEL_PARAMS["doa_resolution"])
        .set_parameter("max_range_ratio_to_limit", SYSTEM_MODEL_PARAMS["max_range_ratio_to_limit"])
        .set_parameter("range_resolution", SYSTEM_MODEL_PARAMS["range_resolution"])
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
        if train_model:
            try:
                train_dataset = load_datasets(
                    system_model_params=system_model_params,
                    samples_size=samples_size,
                    datasets_path=datasets_path,
                    train_test_ratio=train_test_ratio,
                    is_training=True,
                )
            except Exception as e:
                print(e)
                print("#############################################")
                print("load_datasets: Error loading train dataset")
                print("#############################################")
                create_data = True
                load_data = False
        if evaluate_mode:
            try:
                generic_test_dataset, _ = load_datasets(
                    system_model_params=system_model_params,
                    samples_size=int(train_test_ratio * samples_size),
                    datasets_path=datasets_path,
                    train_test_ratio=train_test_ratio,
                    is_training=False,
                )
            except Exception as e:
                print(e)
                print("#############################################")
                print("load_datasets: Error loading test dataset")
                print("#############################################")
                create_data = True
                load_data = False
    if create_data and not load_data:
        # Define which datasets to generate
        print("Creating Data...")
        if train_model:
            # Generate training dataset
            train_dataset, _ = create_dataset(
                system_model_params=system_model_params,
                samples_size=samples_size,
                save_datasets=True,
                datasets_path=datasets_path,
                true_doa=TRAINING_PARAMS["true_doa_train"],
                true_range=TRAINING_PARAMS["true_range_train"],
                phase="train",
            )
        if evaluate_mode:
            # Generate test dataset
            generic_test_dataset, _ = create_dataset(
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
        )
        if load_model:
            try:
                simulation_parameters.load_model(
                    loading_path=saving_path / "final_models" / model_config.model.get_model_file_name())
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
            saving_path=saving_path,
            save_figures=save_plots,
            plot_curves=plot_mode
        )
        # Save model weights
        if save_model:
            torch.save(model.state_dict(),
                       saving_path / "final_models" / Path(model.get_model_file_name()))

    # Evaluation stage
    if evaluate_mode:
        if not train_model:
            model = None
        # Define loss measure for evaluation
        criterion = set_criterions(EVALUATION_PARAMS["criterion"],
                                                       EVALUATION_PARAMS["balance_factor"])

        batch_sampler_test = SameLengthBatchSampler(generic_test_dataset, batch_size=64)
        generic_test_dataset = torch.utils.data.DataLoader(generic_test_dataset,
                                                           collate_fn=collate_fn,
                                                           batch_sampler=batch_sampler_test,
                                                           shuffle=False)

        # Evaluate DNN models, augmented and subspace methods
        loss = evaluate(
            generic_test_dataset=generic_test_dataset,
            criterion=criterion,
            system_model=system_model,
            models=EVALUATION_PARAMS["models"],
            augmented_methods=EVALUATION_PARAMS["augmented_methods"],
            subspace_methods=EVALUATION_PARAMS["subspace_methods"],
            model_tmp=model
        )
        # plt.show()
        print("END OF EVALUATION")
        if save_to_file:
            sys.stdout.close()
            sys.stdout = orig_stdout
        return loss
    return


def run_simulation(**kwargs):
    """
    This function is used to run an end to end simulation, including creating or loading data, training or loading
    a NN model and do an evaluation for the algorithms.
    The type of the run is based on the scenrio_dict. the avialble scrorios are:
    - SNR: a list of SNR values to be tested
    - T: a list of number of snapshots to be tested
    - eta: a list of steering vector error values to be tested
    """
    if kwargs["scenario_dict"] == {}:
        loss = __run_simulation(**kwargs)
        return loss
    loss_dict = {}
    default_snr = kwargs["system_model_params"]["snr"]
    default_T = kwargs["system_model_params"]["T"]
    default_eta = kwargs["system_model_params"]["eta"]
    for key, value in kwargs["scenario_dict"].items():
        if key == "SNR":
            loss_dict["SNR"] = {snr: None for snr in value}
            print(f"Testing SNR values: {value}")
            for snr in value:
                kwargs["system_model_params"]["snr"] = snr
                loss = __run_simulation(**kwargs)
                loss_dict["SNR"][snr] = loss
                kwargs["system_model_params"]["snr"] = default_snr
        if key == "T":
            loss_dict["T"] = {T: None for T in value}
            print(f"Testing T values: {value}")
            for T in value:
                kwargs["system_model_params"]["T"] = T
                loss = __run_simulation(**kwargs)
                loss_dict["T"][T] = loss
                kwargs["system_model_params"]["T"] = default_T
        if key == "eta":
            loss_dict["eta"] = {eta: None for eta in value}
            print(f"Testing eta values: {value}")
            for eta in value:
                kwargs["system_model_params"]["eta"] = eta
                loss = __run_simulation(**kwargs)
                loss_dict["eta"][eta] = loss
                kwargs["system_model_params"]["eta"] = default_eta
    if None not in list(next(iter(loss_dict.values())).values()):
        print_loss_results_from_simulation(loss_dict)
        if kwargs["simulation_commands"]["PLOT_LOSS_RESULTS"]:
            plot_results(loss_dict,
                         plot_acc=kwargs["simulation_commands"]["PLOT_ACC_RESULTS"],
                         save_to_file=kwargs["simulation_commands"]["SAVE_PLOTS"])

    return loss_dict


if __name__ == "__main__":
    now = datetime.now()
