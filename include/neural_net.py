from typing import Any, List, Sequence, Union

import torch
from torch import nn
import pygad
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .utils import random_tensor


import coloredlogs
from dantro import logging
log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)

# ----------------------------------------------------------------------------------------------------------------------
# -- NN utility functions ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def sigmoid(beta=torch.tensor(1.0)):
    """Extends the torch.nn.sigmoid activation function by allowing for a slope parameter."""

    return lambda x: torch.sigmoid(beta * x)


# Pytorch activation functions.
# Pairs of activation functions and whether they are part of the torch.nn module, in which case they must be called
# via func(*args, **kwargs)(x).


ACTIVATION_FUNCS = {
    "abs": [torch.abs, False],
    "celu": [torch.nn.CELU, True],
    "cos": [torch.cos, False],
    "cosine": [torch.cos, False],
    "elu": [torch.nn.ELU, True],
    "gelu": [torch.nn.GELU, True],
    "hardshrink": [torch.nn.Hardshrink, True],
    "hardsigmoid": [torch.nn.Hardsigmoid, True],
    "hardswish": [torch.nn.Hardswish, True],
    "hardtanh": [torch.nn.Hardtanh, True],
    "leakyrelu": [torch.nn.LeakyReLU, True],
    "linear": [None, False],
    "logsigmoid": [torch.nn.LogSigmoid, True],
    "mish": [torch.nn.Mish, True],
    "prelu": [torch.nn.PReLU, True],
    "relu": [torch.nn.ReLU, True],
    "rrelu": [torch.nn.RReLU, True],
    "selu": [torch.nn.SELU, True],
    "sigmoid": [sigmoid, True],
    "silu": [torch.nn.SiLU, True],
    "sin": [torch.sin, False],
    "sine": [torch.sin, False],
    "softplus": [torch.nn.Softplus, True],
    "softshrink": [torch.nn.Softshrink, True],
    "swish": [torch.nn.SiLU, True],
    "tanh": [torch.nn.Tanh, True],
    "tanhshrink": [torch.nn.Tanhshrink, True],
    "threshold": [torch.nn.Threshold, True],
}

OPTIMIZERS = {
        "Adagrad": torch.optim.Adagrad,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SparseAdam": torch.optim.SparseAdam,
        "Adamax": torch.optim.Adamax,
        "ASGD": torch.optim.ASGD,
        "LBFGS": torch.optim.LBFGS,
        "NAdam": torch.optim.NAdam,
        "RAdam": torch.optim.RAdam,
        "RMSprop": torch.optim.RMSprop,
        "Rprop": torch.optim.Rprop,
        "SGD": torch.optim.SGD,
}

MINIMIZE_FUNCTIONS = {
        "modestga.minimize": modestga.minimize,
        "scipy.optimize.minimize": scipy.optimize.minimize
}




def get_architecture(
    input_size: int, output_size: int, n_layers: int, cfg: dict
) -> List[int]:
    # Apply default to all hidden layers
    _nodes = [cfg.get("default")] * n_layers

    # Update layer-specific settings
    _layer_specific = cfg.get("layer_specific", {})
    for layer_id, layer_size in _layer_specific.items():
        _nodes[layer_id] = layer_size

    return [input_size] + _nodes + [output_size]


def get_activation_funcs(n_layers: int, cfg: dict) -> List[callable]:
    """Extracts the activation functions from the config. The config is a dictionary containing the
    default activation function, and a layer-specific entry detailing exceptions from the default. 'None' entries
    are interpreted as linear layers.

    .. Example:
        activation_funcs:
          default: relu
          layer_specific:
            0: ~
            2: tanh
            3:
              name: HardTanh
              args:
                - -2  # min_value
                - +2  # max_value
    """

    def _single_layer_func(layer_cfg: Union[str, dict]) -> callable:
        """Return the activation function from an entry for a single layer"""

        # Entry is a single string
        if isinstance(layer_cfg, str):
            _f = ACTIVATION_FUNCS[layer_cfg.lower()]
            if _f[1]:
                return _f[0]()
            else:
                return _f[0]

        # Entry is a dictionary containing args and kwargs
        elif isinstance(layer_cfg, dict):
            _f = ACTIVATION_FUNCS[layer_cfg.get("name").lower()]
            if _f[1]:
                return _f[0](*layer_cfg.get("args", ()), **layer_cfg.get("kwargs", {}))
            else:
                return _f[0]

        elif layer_cfg is None:
            _f = ACTIVATION_FUNCS["linear"][0]

        else:
            raise ValueError(f"Unrecognized activation function {cfg}!")

    # Use default activation function on all layers
    _funcs = [_single_layer_func(cfg.get("default"))] * (n_layers + 1)

    # Change activation functions on specified layers
    _layer_specific = cfg.get("layer_specific", {})
    for layer_id, layer_cfg in _layer_specific.items():
        _funcs[layer_id] = _single_layer_func(layer_cfg)

    return _funcs


def get_bias(n_layers: int, cfg: dict) -> List[Any]:
    """Extracts the bias initialisation settings from the config. The config is a dictionary containing the
    default, and a layer-specific entry detailing exceptions from the default. 'None' entries
    are interpreted as unbiased layers.

    .. Example:
        biases:
          default: ~
          layer_specific:
            0: [-1, 1]
            3: [2, 3]
    """

    # Use the default value on all layers
    biases = [cfg.get("default")] * (n_layers + 1)

    # Amend bias on specified layers
    _layer_specific = cfg.get("layer_specific", {})
    for layer_id, layer_bias in _layer_specific.items():
        biases[layer_id] = layer_bias

    return biases


# -----------------------------------------------------------------------------
# -- Neural net class ---------------------------------------------------------
# -----------------------------------------------------------------------------


class NeuralNet(nn.Module):
    

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        num_layers: int,
        nodes_per_layer: dict,
        activation_funcs: dict,
        biases: dict,
        prior: Union[list, dict] = None,
        prior_max_iter: int = 2000,
        prior_tol: float = 1e-5,
        optimizer: str = "Adam",
        learning_rate: float = 0.002,
        optimizer_kwargs: dict = {},
        **__,
    ):
        """

        :param input_size: the number of input values
        :param output_size: the number of output values
        :param num_layers: the number of hidden layers
        :param nodes_per_layer: a dictionary specifying the number of nodes per layer
        :param activation_funcs: a dictionary specifying the activation functions to use
        :param biases: a dictionary containing the initialisation parameters for the bias
        :param prior (optional): initial prior distribution of the parameters. If given, the neural net will
            initially output a random value within that distribution.
        :param prior_tol (optional): the tolerance with which the prior distribution should be met
        :param prior_max_iter (optional): maximum number of training iterations to hit the prior target
        :param optimizer: the name of the optimizer to use. Default is the torch.optim.Adam optimizer.
        :param learning_rate: the learning rate of the optimizer. Default is 1e-3.
        :param __: Additional model parameters (ignored)
        """

        super().__init__()
        self.flatten = nn.Flatten()

        self.input_dim = input_size
        self.output_dim = output_size
        self.hidden_dim = num_layers

        # Get architecture, activation functions, and layer bias
        self.architecture = get_architecture(
            input_size, output_size, num_layers, nodes_per_layer
        )
        self.activation_funcs = get_activation_funcs(num_layers, activation_funcs)
        self.bias = get_bias(num_layers, biases)

        # Add the neural net layers
        self.layers = nn.ModuleList()
        for i in range(len(self.architecture) - 1):
            layer = nn.Linear(
                self.architecture[i],
                self.architecture[i + 1],
                bias=self.bias[i] is not None,
            )

            # Initialise the biases of the layers with a uniform distribution
            if self.bias[i] is not None:
                # Use the pytorch default if indicated
                if self.bias[i] == "default":
                    torch.nn.init.uniform_(layer.bias)
                # Initialise the bias on explicitly provided intervals
                else:
                    torch.nn.init.uniform_(layer.bias, self.bias[i][0], self.bias[i][1])

            self.layers.append(layer)

        # Get the optimizer
        self.optimizer = OPTIMIZERS[optimizer](
            self.parameters(), lr=learning_rate, **optimizer_kwargs
        )

        # Get the initial distribution and initialise
        self.prior_distribution = prior
        self.initialise_to_prior(tol=prior_tol, max_iter=prior_max_iter)

        self.type = "mlp"

    def initialise_to_prior(self, *, tol: float = 1e-5, max_iter: int = 500) -> None:
        """Initialises the neural net to output values following a prior distribution. The random tensor is drawn
        following a prior distribution and the neural network trained to output that value. Training is performed
        until the neural network output matches the drawn value (which typically only takes a few seconds), or until
        a maximum iteration count is reached.

        :param tol: the target error on the neural net initial output and drawn value.
        :param max_iter: maximum number of training steps to perform in the while loop
        """

        # If not initial distribution is given, nothing happens
        if self.prior_distribution is None:
            return

        # Draw a target tensor following the given prior distribution
        target = random_tensor(self.prior_distribution, size=(self.output_dim,))

        # Generate a prediction and train the net to output the given target
        prediction = self.forward(torch.rand(self.input_dim))
        iter = 0

        # Use a separate optimizer for the training
        optim = torch.optim.Adam(self.parameters(), lr=0.002)
        while torch.norm(prediction - target) > tol and iter < max_iter:
            prediction = self.forward(torch.rand(self.input_dim))
            loss = torch.nn.functional.mse_loss(target, prediction, reduction="sum")
            loss.backward()
            optim.step()
            optim.zero_grad()
            iter += 1

    # ... Evaluation functions .........................................................................................

    # The model forward pass
    def forward(self, x):
        for i in range(len(self.layers)):
            if self.activation_funcs[i] is None:
                x = self.layers[i](x)
            else:
                x = self.activation_funcs[i](self.layers[i](x))
        return x


class Optimizer(nn.Module):

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        num_layers: int,
        nodes_per_layer: dict,
        activation_funcs: dict,
        biases: dict,
        prior: Union[list, dict] = None,
        prior_max_iter: int = 500,
        prior_tol: float = 1e-5,
        optimizer: str = "Adam",
        learning_rate: float = 0.002,
        optimizer_kwargs: dict = {},
        **__,
    ):
        """
        :param input_size: the number of input values
        :param output_size: the number of output values
        :param num_layers: the number of hidden layers
        :param nodes_per_layer: a dictionary specifying the number of nodes per layer
        :param activation_funcs: a dictionary specifying the activation functions to use
        :param biases: a dictionary containing the initialisation parameters for the bias
        :param prior (optional): initial prior distribution of the parameters. If given, the neural net will
            initially output a random value within that distribution.
        :param prior_tol (optional): the tolerance with which the prior distribution should be met
        :param prior_max_iter (optional): maximum number of training iterations to hit the prior target
        :param optimizer: the name of the optimizer to use. Default is the torch.optim.Adam optimizer.
        :param learning_rate: the learning rate of the optimizer. Default is 1e-3.
        :param __: Additional model parameters (ignored)

        
        This Class implements a direct interface from the NeuralABM optimization loop and the optimizer itself.
        Normally, the optimizer would optimize a neural net which in return provides parameter estimates. Using
        this class, the optimizer directly provides and optimizes the parameters, bypassing the neural network.
        """
        super().__init__()
        if prior is not None:
            target = random_tensor(prior, size=(output_size,))
            self.param_list = [nn.Parameter(i.unsqueeze(0)) for i in target]
        else:
            self.param_list = [nn.Parameter(torch.rand(1)) for _ in range(output_size)]
        self.optimizer = OPTIMIZERS[optimizer](
            self.param_list, lr=learning_rate,
            **optimizer_kwargs
        )

        self.type = "optimizer"


    def forward(self, x):
        return torch.cat(self.param_list)

class Lstm(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        num_layers: int,
        nodes_per_layer: dict,
        activation_funcs: dict,
        biases: dict,
        prior: Union[list, dict] = None,
        prior_max_iter: int = 500,
        prior_tol: float = 1e-5,
        optimizer: str = "Adam",
        learning_rate: float = 0.002,
        optimizer_kwargs: dict = {},
        **__,
    ):
        """
        :param input_size: the number of input values
        :param output_size: the number of output values
        :param num_layers: the number of hidden layers
        :param nodes_per_layer: a dictionary specifying the number of nodes per layer
        :param activation_funcs: a dictionary specifying the activation functions to use
        :param biases: a dictionary containing the initialisation parameters for the bias
        :param prior (optional): initial prior distribution of the parameters. If given, the neural net will
            initially output a random value within that distribution.
        :param prior_tol (optional): the tolerance with which the prior distribution should be met
        :param prior_max_iter (optional): maximum number of training iterations to hit the prior target
        :param optimizer: the name of the optimizer to use. Default is the torch.optim.Adam optimizer.
        :param learning_rate: the learning rate of the optimizer. Default is 1e-3.
        :param __: Additional model parameters (ignored)


        This class implements an LSTM neural network using the standart implementation of torch.
        It can be used instead of a standard MLP and uses the same interface for the configs.
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size, 
                            num_layers = num_layers,
                            hidden_size = nodes_per_layer["default"]
                            )
        self.fc = nn.Linear(nodes_per_layer["default"], output_size    )
        self.optimizer = OPTIMIZERS[optimizer](
            self.parameters(), lr=learning_rate, **optimizer_kwargs
        )
        self.output_activation = ACTIVATION_FUNCS[activation_funcs["layer_specific"][-1]][0]

        self.input_dim = input_size
        self.output_dim = output_size
        self.hidden_dim = num_layers

         # Get the initial distribution and initialise
        self.prior_distribution = prior
        self.initialise_to_prior(tol=prior_tol, max_iter=prior_max_iter)

        self.type = "lstm"

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        return self.output_activation(self.fc(output[-1]).squeeze(0))



    def initialise_to_prior(self, *, tol: float = 1e-5, max_iter: int = 500) -> None:
        """Initialises the neural net to output values following a prior distribution. The random tensor is drawn
        following a prior distribution and the neural network trained to output that value. Training is performed
        until the neural network output matches the drawn value (which typically only takes a few seconds), or until
        a maximum iteration count is reached.

        :param tol: the target error on the neural net initial output and drawn value.
        :param max_iter: maximum number of training steps to perform in the while loop
        """

        # If not initial distribution is given, nothing happens
        if self.prior_distribution is None:
            return

        # Draw a target tensor following the given prior distribution
        target = random_tensor(self.prior_distribution, size=(self.output_dim,))

        # Generate a prediction and train the net to output the given target
        prediction = self.forward(torch.rand(self.input_dim).unsqueeze(0).unsqueeze(0))
        iter = 0

        # Use a separate optimizer for the training
        optim = torch.optim.Adam(self.parameters(), lr=0.002)
        while torch.norm(prediction - target) > tol and iter < max_iter:
            prediction = self.forward(torch.rand(self.input_dim).unsqueeze(0).unsqueeze(0))
            loss = torch.nn.functional.mse_loss(target, prediction, reduction="sum")
            loss.backward()
            optim.step()
            optim.zero_grad()
            iter += 1


class GAOptimizer(torch.optim.Optimizer):

    '''
    This class implements a Genetic Algorithm as optimizer. This GA can be used as any other optimizer offered in the
     NeuralABM framework. Therefore, it can either directly optimize the ODE's parameters or be used to fit a neural network
     to in return provide parameters.

    Please refer to the documentation of pygad for further details about the algorithm.

    Please note that, due to the setting in the framework, the GA can only be run in single-thread mode and that
     the number of epochs set in the config refers to the amount of optimization calls = generations/population_size.

    '''

    
    # mock function for the GA constructor. Since I execute run() manually, it is never accessed (except for type checking)
    def fitness(self, ga_instance, solution, solution_idx):
        pass

    def __init__(self,
                params,
                population_size=40,
                mutation_rate=0.05,
                crossover_rate=0.5,
                lr=1e-3,
                num_parents_mating=2,
                init_mutation_range = 0.1,
                **__):

        # unroll parameter iterator for repeated access, store for later
        params = list(params)
        self.model_params = params

        super().__init__(params, defaults = {"lr": lr})

        # convert param dict to 1D-vector for GA individual size
        params = parameters_to_vector(params)

        
        # read out 1D-parameters for initial GA individual
        parameters = np.array([i.detach().item() for i in params])
        self.current_loss = 0.0
        print("initializing GAOptimizer")
        self.ga = pygad.GA(
                    num_generations = 1,                # generations are manually evolved
                    num_parents_mating = num_parents_mating,
                    sol_per_pop = population_size,
                    num_genes = len(parameters),
                    init_range_low = parameters * (1-init_mutation_range),    # <- array same shape as base_params
                    init_range_high = parameters * (1+init_mutation_range),  # <- array same shape as base_params
                    mutation_type = "random",
                    mutation_probability = mutation_rate,
                    fitness_func = self.fitness,
                    parent_selection_type = "tournament",
                    K_tournament = 6,
                    crossover_type="uniform",
                    crossover_probability=0.5
                    )

        self.generation = 1
        self.evaluated = 0
        self.pop_size = population_size
        self.ga.last_generation_fitness = np.zeros(population_size)
        #print(self.ga.population)


    @torch.no_grad()
    def step(self):
        self.ga.last_generation_fitness[self.evaluated] = -self.current_loss
        self.evaluated += 1
        

        # one whole generation has been evaluated
        if self.evaluated == self.pop_size:
            #print("Completed GA generation - ineriting and mutating")
            self.ga.run_select_parents()
            self.ga.run_crossover()
            self.ga.run_mutation()
            self.ga.run_update_population()
            self.evaluated = 0
            self.generation += 1
            #print(self.ga.population)

        # get the next individual in line
        individual_tensor = torch.tensor(self.ga.population[self.evaluated], dtype=torch.float32)
        # update all model parameters at once
        vector_to_parameters(individual_tensor, self.model_params)

    def zero_grad(*arg):
        pass



# add custom GA Optimizer to list of optimizers
OPTIMIZERS["GA"] = GAOptimizer



