#!/usr/bin/env python3
import sys
from os.path import dirname as up
import os
from pathlib import Path


import coloredlogs
import h5py as h5
import numpy as np
import ruamel.yaml as yaml
import torch
from dantro import logging
from dantro._import_tools import import_module_from_path, get_from_module
from dantro.tools import recursive_update
from pprint import pformat

import multiprocessing as mp
import datetime
import optuna


sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

RC_model = import_module_from_path(mod_path=up(up(__file__)), mod_str="RC_model")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)

np.set_printoptions(precision=5)


# ----------------------------------------------------------------------------------------------------------------------
# Performing the simulation run
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    cfg_file_path = sys.argv[1]

    log.note("   Preparing model run ...")
    log.info(f"   Loading config file:\n         {cfg_file_path}")
    yamlc = yaml.YAML(typ="safe")
    with open(cfg_file_path) as cfg_file:
        cfg = yamlc.load(cfg_file)
    model_name = cfg.get("root_model_name", "RC_model")
    log.note(f"   Model name:  {model_name}")
    model_cfg = cfg[model_name]

    # Select the training device and number of threads to use
    device = model_cfg["Training"].get("device", None)
    if device is None:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    num_threads = model_cfg["Training"].get("num_threads", None)
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    log.info(
        f"   Using '{device}' as training device. Number of threads: {torch.get_num_threads()}"
    )

    # Get the random number generator
    log.note("   Creating global RNG ...")
    rng = np.random.default_rng(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.random.manual_seed(cfg["seed"])

    # create data file to which the run is stored
    log.info(f"   Creating output file at:\n         {cfg['output_path']}")
    h5file = h5.File(cfg["output_path"], mode="w")
    h5group = h5file.create_group(model_name)

    # Get the training data
    log.info("   Fetching training data...")
    training_data = RC_model.get_RC_circuit_data(data_cfg=model_cfg["Data"], h5group=h5group).to(
        device
    )
    log.info(f"    training_data.shape = {training_data.shape}")

    #workaround for sweeping over mlp:predict and mlp:single-input (mlp-single is a valid NN model type where an MLP with lookback = 1 is initialized)
    if model_cfg["NeuralNet"]["type"] == "mlp-single": 
        model_cfg["Training"]["mode"] = "single-input"
        log.info("detected mlp-sinlge key. Setting cfg.Training.mode to single-input")

    # tune hyperparameters using optuna
    if "Tuning" in model_cfg.keys():
        if model_cfg["Tuning"]["perform"]:
            
            mp.set_start_method("spawn", force=True)
            model_mode = model_cfg["Training"]["mode"]
            model_cfg["Training"]["mode"] = "tuning"
            n_workers = model_cfg["Tuning"]["n_jobs"]
            model_cfg["Tuning"]["study_n_trials"] = int(np.ceil(model_cfg["Tuning"]["study_n_trials"]/n_workers))

            model_cfg["Tuning"]["n_jobs"] = 1

            # every thread references the same study
            model_cfg["Tuning"]["study_name"] = model_cfg["Tuning"]["study_name"] + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log.info("   Performing hyperparameter tuning using\n         %s", pformat(model_cfg["Tuning"]).replace("\n", "\n         "))

            processes = []
            for _ in range(n_workers):
                p = mp.Process(
                    target=RC_model.tune_hyperparameters,
                    args=(model_cfg, device, rng, training_data),
                    kwargs = model_cfg["Tuning"]
                )
                p.start()
                processes.append(p)
            log.debug(f"\tstarted {processes}, waiting for them to join back...")

            for p in processes:
                p.join()

            study = optuna.load_study(
                study_name=model_cfg["Tuning"]["study_name"],
                storage="sqlite:///optuna.db",
            )
            best_hypers = RC_model.unflatten_dict_keys(study.best_params, sep = "/")
            hyper_folder = os.path.join(*os.path.split(cfg["output_path"])[:-1], "tuning")

            Path(hyper_folder).mkdir(parents=True, exist_ok=True)

            with open(os.path.join(hyper_folder, "best_hyperparameters.yml"), "w") as f:
                yaml.YAML(typ='unsafe', pure=True).dump(best_hypers, f)
            study.trials_dataframe().to_csv(os.path.join(hyper_folder, "study.csv"))
            log.info("   Best parameters found\n         %s", pformat(best_hypers).replace("\n", "\n         "))
            recursive_update(model_cfg, best_hypers)
            model_cfg["Training"]["mode"] = model_mode



    # get the physical model object used for simulating temperature.
    physical = get_from_module(RC_model.Physicals, name = model_cfg["Training"]["model_type"])()

    # try to infer model type from filename of the general estimator, iff passed 
    if "pretrained" in model_cfg["NeuralNet"].keys():
        pretrained_path = model_cfg["NeuralNet"]["pretrained"]
        if not pretrained_path == "None" and (os.path.split(pretrained_path)[-1].startswith(("mlp", "lstm"))): #if a net type can be inferred from filename
            model_cfg["NeuralNet"]["type"] = os.path.split(pretrained_path)[-1][:os.path.split(pretrained_path)[-1].find("_")]
            log.info(f"   Inferred net type {model_cfg['NeuralNet']['type']} from pretrained path {pretrained_path}.")

    # set up input dimension for NN model depending on estimation method
    if model_cfg["Training"]["mode"] == "single-input":
        input_size = physical.dynamic_variables
    elif model_cfg["NeuralNet"]["type"] == "lstm":
        input_size = training_data.shape[2]
    else: 
        input_size = model_cfg["NeuralNet"]["lookback"]*training_data.shape[2]

    log.info(f"   Initializing the {model_cfg['NeuralNet']['type']} in {model_cfg['Training']['mode']} mode (inpsize: {input_size}, outpsize {len(physical.parameter_names)}) ...")
    log.info("   Using NeuralNet parameters\n         %s", pformat(model_cfg["NeuralNet"]).replace("\n", "\n         "))
    log.info(f"   Using physical {model_cfg['Training']['model_type']} and train_data.shape = {training_data.shape}")

    # remove numbers from optimizer specifications. Enables the user to do coupled sweeps using the same optimizer multiple times
    # without triggering utopyas "duplicate encountered" error by counting optimizers (e.g. "Adam0", "Adam1", ...)
    if not model_cfg["NeuralNet"]["type"].startswith("external"):
        model_cfg["NeuralNet"]["optimizer"] = ''.join(filter(lambda x: x.isalpha(), model_cfg["NeuralNet"]["optimizer"]))

    # initialize the NN model used for estimation or an optimizer as replacement
    if model_cfg["NeuralNet"]["type"].startswith("optimizer"):
        # if optimizer is used instead of a neural net, always process the whole timeseries at once
        if isinstance(model_cfg["Training"]["train_range"], float) and isinstance(model_cfg["Training"]["val_range"], float):
            model_cfg["NeuralNet"]["lookback"] = int((model_cfg["Training"]["train_range"] + model_cfg["Training"]["val_range"])*training_data.shape[1])-1
        elif isinstance(model_cfg["Training"]["train_range"], list) and isinstance(model_cfg["Training"]["val_range"], list):
            model_cfg["NeuralNet"]["lookback"] = model_cfg["Training"]["val_range"][1] - model_cfg["Training"]["train_range"][0]-1
        else:
            raise ValueError("Both train and val range need to both be passed either as slices or float")
        model_cfg["Training"]["batch_size"] = 1
        model_cfg["Training"]["loss_function"] = {"name": "MSELoss"}
        net = base.Optimizer(input_size=input_size,
            output_size=len(physical.parameter_names),
            **model_cfg["NeuralNet"],
        ).to(device)
    elif model_cfg["NeuralNet"]["type"] == "lstm":
        net = base.Lstm(input_size = input_size,
            output_size=len(physical.parameter_names),
            **model_cfg["NeuralNet"],
            )
    elif model_cfg["NeuralNet"]["type"] == "external":
        net = base.External(input_size = input_size,
            output_size = len(physical.parameter_names),
            **model_cfg["NeuralNet"],
            )
        model_cfg["Training"]["mode"] = "external"
        cfg["num_epochs"] = 1
        log.info("External optimization algorithm selected. Number of epochs set to one, since epoch control is done externally")
    else:
        net = base.NeuralNet(
            input_size=input_size,
            output_size=len(physical.parameter_names),
            **model_cfg["NeuralNet"],
        ).to(device)

    # load the weights of a general estimator, if configured
    if "pretrained" in model_cfg["NeuralNet"].keys():
        if not model_cfg["NeuralNet"]["pretrained"].startswith("None"): #for sweeping over the "pretrained" entry, yml entries are parsed as strings
            net.load_state_dict(torch.load(model_cfg["NeuralNet"]["pretrained"], weights_only = True))
            log.info(f"    loaded pretrained model from {model_cfg['NeuralNet']['pretrained']}")

    log.info("   Initializing Training using\n         %s", pformat(model_cfg["Training"]).replace("\n", "\n         "))

    # Initialise the training loop, using the initialized neural_net and Physical model
    model = RC_model.NN(
        rng=rng,
        h5group=h5group,
        neural_net=net,
        write_every=cfg["write_every"],
        write_start=cfg["write_start"],
        num_steps=training_data.shape[1],
        training_data=training_data[:, :, :physical.dynamic_variables, :],
        external_data=training_data[:, :, physical.dynamic_variables:, :],
        physical = physical,
        lookback = model_cfg["NeuralNet"]["lookback"],
        dt=torch.tensor(model_cfg["Data"]["dt"], dtype = torch.float64),
        **model_cfg["Training"],
    )

    # store some metadata in the dataset to later retrieve for evaluation
    model.dset_parameters.attrs["model_type"] = model_cfg["Training"]["model_type"]
    model.dset_parameters.attrs["dt"] = model_cfg["Data"]["dt"]
    log.info(f"   Initialized model '{model_name}'.")


    # filter epochs passed as strings (remove characters and convert to int)
    if isinstance(cfg["num_epochs"], str):
        cfg["num_epochs"] = int(''.join(filter(lambda x: not x.isalpha(), cfg["num_epochs"])))

    # start training loop for specified amount of epochs
    num_epochs = cfg["num_epochs"]
    log.info(f"   Now commencing training for {num_epochs} epochs ...")
    log.progress(
        f"   Epochs: {num_epochs}"
        f"  \t\t\t\t\t"
        f"  Parameters: {physical.parameter_names}"
        )
    for i in range(num_epochs):
        model.epoch()
        log.progress(
            f"   Epoch {i+1} / {num_epochs}; "
            f"   Loss: {np.float(model.current_loss)}"
            f"   Parameters: {model.current_predictions}"
        )

    log.info("   Simulation run finished.")
    log.info("   Wrapping up ...")

    # store generalized model
    if model_cfg["Training"]["mode"] == "generalize":
        net_out_path = os.path.join(*os.path.split(cfg["output_path"])[:-1], "trained_net.pth")
        log.info(f"   {net_out_path}")

        torch.save(net.state_dict(), net_out_path)
        log.info(f"   Saved Neural Net to {net_out_path}")

    h5file.close()
    log.success("   All done.")
