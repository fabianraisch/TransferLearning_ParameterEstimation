#!/usr/bin/env python3
import sys
from os.path import dirname as up
import os

import coloredlogs
import h5py as h5
import numpy as np
import ruamel.yaml as yaml
import torch
from dantro import logging
from dantro._import_tools import import_module_from_path, get_from_module
import optuna
from copy import deepcopy
import yaml
import datetime


sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

RC_model = import_module_from_path(mod_path=up(up(__file__)), mod_str="RC_model")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)

class Objective:
	'''
	Object class that is used for specifying the objective for optuna to minimize.

	'''
    
	def __init__(self, config, device, rng, training_data):
		'''
		args:
			- config: The whole run config that would be used for training
			- device: torch.device to run tuning on
			- rng: numpy.rng object used if hyperparameter tuning is done seed-sensitive
			- training_data: external data used for training
		'''
		self.config = config

		self.hyperparam_ranges_dict = config["Tuning"]["hyperparam_ranges"]

		self.device = device

		self.training_data = training_data

		self.rng = rng

        

	def get_fresh_NN(self, config, rng):
		'''
		initializes a fresh neural network, using the sizes and model types from the config (same as run.py)
		args:
			- config: the run config
			- rng: numpy.rng object for random weight generation
		returns:
			- the new model
		'''
		physical = deepcopy(get_from_module(RC_model.Physicals, name = config["Training"]["model_type"]))()
		# Initialise the neural net
		if config["NeuralNet"]["type"] == "mlp-single": #workaround for sweeping over mlp:predict and mlp:single-input
			config["Training"]["mode"] = "single-input"

		if config["Training"]["mode"] == "single-input":
			input_size = physical.dynamic_variables
		elif config["NeuralNet"]["type"] == "lstm":
			input_size = self.training_data.shape[2]
		else: 
			input_size = config["NeuralNet"]["lookback"]*self.training_data.shape[2]

		if config["NeuralNet"]["type"] == "optimizer":
			net = base.Optimizer(input_size=input_size,
								output_size=len(physical.parameter_names),
								**config["NeuralNet"],
								).to(self.device)
		elif config["NeuralNet"]["type"] == "lstm":
			net = base.Lstm(input_size = input_size,
							output_size=len(physical.parameter_names),
							**config["NeuralNet"],
								)
		else:
			net = base.NeuralNet(
				input_size=input_size,
				output_size=len(physical.parameter_names),
				**config["NeuralNet"],
			).to(self.device)

		model = RC_model.NN(
			rng=rng,
			h5group=None,
			neural_net=net,
			num_steps=self.training_data.shape[1],
			training_data=self.training_data[:, :, :physical.dynamic_variables, :],
			external_data=self.training_data[:, :, physical.dynamic_variables:, :],
			physical = physical,
			lookback = config["NeuralNet"]["lookback"],
			dt=config["Data"]["dt"],
			**config["Training"],
		)
		return model


	def __call__(self, trial):

		'''
		call function used by optuna. A NeuralNet is used for parameter estimation and the lowest loss recorded.
		If sweeps over multiple seeds (rng initializations) is specified, multiple runs are performed using
		different rngs and the losses meaned.
		args:
			- trial: An optuna trial object that contains mutations to the hyperparams specified in the config
		returns:
			- lowest loss of run or mean over lowest losses for each rng
		'''

        #------------------------------------------------------------------------
        #              Specify Hyperparameter Search Features
        #------------------------------------------------------------------------

		log.info(f"{datetime.datetime.now().strftime('%d%m%y %X')}: starting trial {trial.number}...")

        # v[0]: min, v[1]: max, v[2]: section in config (key), v[3] (opt): numtype, v[4] (opt) step_size. If v[3] == float and no step_size: log steps

		for k, v in self.hyperparam_ranges_dict.items():

			if len(v) < 3: 
				log.info("Please give lists of length 3 in the hyperparameter ranges. The first entry is the min value, the second the max value, and the third specifies the section in the config this hyperparam is in.")
			if len(v) > 5: 
				log.info("List for hyperparameter too long.")
			hyp_type = "int"
			step_size = None
			if len(v) > 3: 
				hyp_type = v[3]
			if len(v) == 5: 
				step_size = v[4]


            # Set Hyperparameter in Config. Note: k is the name of the hyperparam, 
            # v[2] specifies the section in the config of this hyperparam v[0] is the min and v[1] the max value.
			log.debug(f"Hyperparameter {k} in section {v[2]} with min value {v[0]} and max value {v[1]}")
			hyperparam_name = f"{v[2]}/{k}"
			if step_size:
				if hyp_type == "int":
					self.config[v[2]][k] = trial.suggest_int(hyperparam_name, v[0], v[1], step=step_size)
				elif hyp_type == "float":
					self.config[v[2]][k] = trial.suggest_float(hyperparam_name, v[0], v[1], step=step_size)
			else:
				if hyp_type == "int":
					self.config[v[2]][k] = trial.suggest_int(hyperparam_name, v[0], v[1])
				elif hyp_type == "float":
					self.config[v[2]][k] = trial.suggest_float(hyperparam_name, v[0], v[1], log=True)

        
        #------------------------------------------------------------------------
        #                   Do Trainig and Evaluations
        #------------------------------------------------------------------------
		if "seeds_per_trial" in self.config["Tuning"].keys():
			if self.config["Tuning"]["seeds_per_trial"] > 1:
				rngs = [np.random.default_rng(i) for i in range(self.config["Tuning"]["seeds_per_trial"])]
		else:
			rngs = [self.rng]
		

		
		losses_over_rng = 0
		for i, rng in enumerate(rngs):
			log.debug(f"Hyperparameter.Objective.__call__(): Trying new seed {i}")
			model = self.get_fresh_NN(self.config, rng)
			min_loss = float('Inf')
			for ep in range(self.config["num_epochs"]):
				model.epoch()
				if model.current_loss < min_loss:
					min_loss = model.current_loss
				log.info(f"{datetime.datetime.now().strftime('%d%m%y %X')}: trial {trial.number} finished epoch with {model.current_loss}")
			losses_over_rng += min_loss
	        
		return losses_over_rng/len(rngs)
    
    

def tune_hyperparameters(config: dict = {}, device: torch.device = None, rng = None, training_data: torch.tensor = None, n_jobs = 1, **kwargs):
    '''
    Function that performs hyperparameter tuning using the optuna framework

    args:
		- config: full run configuration file
		- device: torch.device to perform tuning on
		- rng: custom rng if seed sweeping is used
		- training_data: the data the parameters are to be tuned on
		- n_jobs: the amount of parallel jobs otpuna is allowed to start. -1 for as many as Threads are available
	returns:
		- the optuna study created for the tuning
    '''
    # copy the config
    config_ = deepcopy(config)

    if not config_.get("Tuning"):
    	log.exception("Config does not contain the hyperparam block -> Please add it.")
    	exit()

    config_["num_epochs"]=config_["Tuning"]["epochs_tuning"]     # different epoch size for hyp tuning compared to later training is possible

    study_name = config_["Tuning"]["study_name"]


    # create the study object
    try:
    	study = optuna.create_study(
    		study_name=study_name,
    		storage="sqlite:///optuna.db",
    		load_if_exists=True,
    		direction="minimize",
    		)
    except optuna.exceptions.DuplicatedStudyError:
    	log.exception(f"Could not crate study because a study with the name {study_name} already exists in the database. Please specify another name.")
    	exit()

    # initialize tuning objective
    objective = Objective(config=config_, device = device, rng = rng, training_data = training_data)

    # perform hyperparameter tuning
    study.optimize(
    	objective,                                                                                                          # Specifies what objective function should be optimized.
    	n_trials=config_["Tuning"]["study_n_trials"] if "study_n_trials" in config_["Tuning"].keys() else 1,           # How many trials should be made.
    	n_jobs = n_jobs
    )

    return study



def unflatten_dict_keys(params: dict, sep: str = "/") -> dict:
    """Convert flat Optuna-style keys (with slashes or dots) into nested dicts.

	args:
		- params: parameter dictionary with flat/nested/keys
		- seperator: the nesting seperator like the "/" in the example above
	returns:
		- parameter dictionary {regular: {nested: {keys}}]

    """
    result = {}
    for key, value in params.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result