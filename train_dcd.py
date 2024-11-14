# This is a runner script for training the DCD model.
# It is a simple script that create the data, the model and train it.

import sys
from src.training import *
from src.data_handler import *
import argparse

# default values for the argparse
number_sensors = 8
number_sources = None
number_snapshots = 100
snr = 0
field_type = "Near"
signal_nature = "coherent"
err_loc_sv = 0.0
tau = 8
sample_size = 1024
train_test_ratio = 0.1
batch_size = 128
epochs = 2
optimizer = "Adam"
learning_rate = 0.001
weight_decay = 1e-9
step_size = 50
gamma = 0.5
diff_method = ("esprit", "music_1D")
train_loss_type = ("rmspe", "rmspe")


def train_dcd_music(*args, **kwargs):
    SIMULATION_COMMANDS = kwargs["simulation_commands"]
    SYSTEM_MODEL_PARAMS = kwargs["system_model_params"]
    MODEL_PARAMS = kwargs["model_params"]
    TRAINING_PARAMS = kwargs["training_params"]
    save_to_file = SIMULATION_COMMANDS["SAVE_TO_FILE"]  # Saving results to file or present them over CMD
    create_data = SIMULATION_COMMANDS["CREATE_DATA"]  # Creating new dataset
    load_model = SIMULATION_COMMANDS["LOAD_MODEL"]  # Load specific model for training
    load_data = not create_data  # Loading data from exist dataset
    print("Running simulation...")
    print("Training model - DCD-MUSIC, all training steps.")

    now = datetime.now()
    plot_path = Path(__file__).parent / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")

    # Initialize seed
    set_unified_seed()

    # Initialize paths
    datasets_path, simulations_path, saving_path = initialize_data_paths(Path(__file__).parent / "data")

    # Saving simulation scores to external file
    suffix = ""
    suffix += f"_train_DCD-MUSIC_all_steps"
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
    )
    system_model = SystemModel(system_model_params)

    # Define samples size
    samples_size = TRAINING_PARAMS["samples_size"]  # Overall dateset size
    train_test_ratio = TRAINING_PARAMS["train_test_ratio"]  # training and testing datasets ratio

    # Print new simulation intro
    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")

    if load_data:
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
            print("load_datasets: Error loading datasets")
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
    # Generate model configuration
    model_config = (
        ModelGenerator()
        .set_model_type("SubspaceNet")
        .set_system_model(system_model)
        .set_model_params({"diff_method": diff_method[0], "train_loss_type": train_loss_type[0],
                           "tau": MODEL_PARAMS.get("tau"), "field_type": "Far"})
        .set_model()
    )
    # Assign the training parameters object
    simulation_parameters = (
        TrainingParams()
        .set_training_objective("angle")
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
    )
    # Save model weights
    torch.save(model.state_dict(),
               saving_path / "final_models" / Path(model.get_model_file_name()))
    print("END OF TRAINING - Step 1: angle branch training.")

    # Update model configuration
    model_config.set_model_type("DCD-MUSIC")
    model_config.set_model_params({"tau": MODEL_PARAMS.get("tau"),
                                   "diff_method": diff_method,
                                   "train_loss_type": train_loss_type})
    model_config.set_model()
    # Assign the training parameters object
    simulation_parameters.set_training_objective("range")
    simulation_parameters.set_model(model_gen=model_config)
    simulation_parameters.set_optimizer(optimizer=TRAINING_PARAMS["optimizer"],
                                        learning_rate=TRAINING_PARAMS["learning_rate"],
                                        weight_decay=TRAINING_PARAMS["weight_decay"])
    simulation_parameters.set_schedular(step_size=TRAINING_PARAMS["step_size"],
                       gamma=TRAINING_PARAMS["gamma"])

    if load_model:
        try:
            simulation_parameters.load_model(
                loading_path=saving_path / "final_models" / model_config.model.get_model_file_name())
        except Exception as e:
            print("#############################################")
            print(e)
            print("simulation_parameters.load_model: Error loading model")
            print("#############################################")
    # update eigen regularization to 0

    # Perform simulation training and evaluation stages
    model, loss_train_list, loss_valid_list = train(
        training_parameters=simulation_parameters,
        saving_path=saving_path,
    )
    # Save model weights
    torch.save(model.state_dict(),
               saving_path / "final_models" / Path(model.get_model_file_name()))
    print("END OF TRAINING - Step 2: distance branch training.")

    # Assign the training parameters object
    simulation_parameters.set_training_objective("angle, range")
    simulation_parameters.set_optimizer(optimizer=TRAINING_PARAMS["optimizer"],
                                        learning_rate=TRAINING_PARAMS["learning_rate"] / 2,
                                        weight_decay=TRAINING_PARAMS["weight_decay"])
    simulation_parameters.set_schedular(step_size=TRAINING_PARAMS["step_size"],
                                        gamma=TRAINING_PARAMS["gamma"])
    simulation_parameters.model.init_model_train_params()

    if load_model:
        try:
            simulation_parameters.load_model(
                loading_path=saving_path / "final_models" / model_config.model.get_model_file_name())
        except Exception as e:
            print("#############################################")
            print(e)
            print("simulation_parameters.load_model: Error loading model")
            print("#############################################")

    # Perform simulation training and evaluation stages
    model, loss_train_list, loss_valid_list = train(
        training_parameters=simulation_parameters,
        saving_path=saving_path,
    )
    # Save model weights
    torch.save(model.state_dict(),
               saving_path / "final_models" / Path(model.get_model_file_name()))
    print("END OF TRAINING - Step 3: adaption by position.")
    if save_to_file:
        sys.stdout = orig_stdout


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation with optional parameters.")
    parser.add_argument('-n', '--number_sensors', type=int, help='Number of sensors', default=number_sensors)
    parser.add_argument('-m', '--number_sources', type=int, help='Number of sources', default=number_sources)
    parser.add_argument('-t', '--number_snapshots', type=int, help='Number of snapshots', default=number_snapshots)
    parser.add_argument('-s', '--snr', type=int, help='SNR value', default=snr)
    parser.add_argument('-ft', '--field_type', type=str, help='Field type', default=field_type)
    parser.add_argument('-sn', '--signal_nature', type=str, help='Signal nature', default=signal_nature)
    parser.add_argument('-eta', '--err_loc_sv', type=float, help="Error in sensors' locations", default=err_loc_sv)

    parser.add_argument('-tau', type=int, help="Number of autocorrelation features", default=tau)

    parser.add_argument('-size', '--sample_size', type=int, help='Samples size', default=sample_size)
    parser.add_argument('-ratio', type=float, help='Train test ratio', default=train_test_ratio)
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=batch_size)
    parser.add_argument('-ep', '--epochs', type=int, help='Number of epochs', default=epochs)
    parser.add_argument('-op', "--optimizer", type=str, help='Optimizer type', default=optimizer)
    parser.add_argument('-lr', "--learning_rate", type=float, help='Learning rate', default=learning_rate)
    parser.add_argument('-wd', "--weight_decay", type=float, help='Weight decay for optimizer', default=weight_decay)
    parser.add_argument('-sp', "--step_size", type=int, help='Step size for schedular', default=step_size)
    parser.add_argument('-gm', "--gamma", type=float, help='Gamma value for schedular', default=gamma)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    system_model_params = {
        "N": args.number_sensors,  # number of antennas
        "M": args.number_sources,  # number of sources
        "T": args.number_snapshots,  # number of snapshots
        "snr": args.snr,  # if defined, values in scenario_dict will be ignored
        "field_type": args.field_type,  # Near, Far
        "signal_nature": args.signal_nature,  # if defined, values in scenario_dict will be ignored
        "eta": args.err_loc_sv,  # steering vector error
        "bias": 0,
        "sv_noise_var": 0.0
    }
    model_params = {
        "tau": args.tau
    }
    training_params = {
        "samples_size": args.sample_size,
        "train_test_ratio": args.ratio,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": args.optimizer,  # Adam, SGD
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "true_doa_train": None,  # if set, this doa will be set to all samples in the train dataset
        "true_range_train": None,  # if set, this range will be set to all samples in the train dataset
        "true_doa_test": None,  # if set, this doa will be set to all samples in the test dataset
        "true_range_test": None,  # if set, this range will be set to all samples in the train dataset
    }
    simulation_commands = {
        "SAVE_TO_FILE": False,
        "CREATE_DATA": False,
        "LOAD_MODEL": False,
    }
    train_dcd_music(simulation_commands=simulation_commands,
                    system_model_params=system_model_params,
                    model_params=model_params,
                    training_params=training_params)
