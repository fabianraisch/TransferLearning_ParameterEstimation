import sys
from os.path import dirname as up

import coloredlogs
import h5py as h5
import numpy as np
import torch
from torchdiffeq import odeint
from dantro import logging
from dantro._import_tools import import_module_from_path
import random
import tplot
import types


sys.path.append(up(up(__file__)))
sys.path.append(up(up(up(__file__))))

RC_model = import_module_from_path(mod_path=up(up(__file__)), mod_str="RC_model")
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

log = logging.getLogger(__name__)
coloredlogs.install(fmt="%(levelname)s %(message)s", level="INFO", logger=log)


class RC_model_NN:

    ''' Object that implements the whole NeuralABM optimization cycle, modified for the RC model application.
        
        A neural_net is used to predict model parameters, which in return are used to evolve an ODE.
        The ODE's trajectory is used to compute a loss with respect to the ground truth which is stored
        and used for optimizing the neural_net's internal parameters.

    '''



    def __init__(
        self,
        *,
        rng: np.random.Generator,
        h5group: h5.Group,
        neural_net: base.NeuralNet,
        loss_function: dict,
        dt: torch.Tensor,
        true_parameters: dict = {},
        write_every: int = 1,
        write_start: int = 1,
        training_data: torch.Tensor,
        external_data: torch.Tensor,
        physical,
        batch_size: int,
        lookback: int,
        mode: str,
        scaling_factors: dict = {},
        train_range = 0.7,
        val_range = 0.3,
        slicing_difference = 1,
        **__,
    ):
        """Initialize the model instance with a previously constructed RNG and
        HDF5 group to write the output data to.

        :param rng (np.random.Generator): The shared RNG
        :param h5group (h5.Group): The output file group to write data to
        :param neural_net: The neural network
        :param loss_function (dict): the loss function to use
        :param to_learn: the list of parameter names to learn
        :param true_parameters: the dictionary of true parameters
        :param training_data: the time series of T_in data to calibrate
        :param external_data: the time series of the external data (T_out, Q_H, Q_O)
        :param write_every: write every iteration
        :param write_start: iteration at which to start writing
        :param batch_size: epoch batch size: instead of calculating the entire time series,
            only a subsample of length batch_size can be used. The likelihood is then
            scaled up accordingly.
        :param scaling_factors: factors by which the parameters are to be scaled
        """
        self._h5group = h5group
        self._rng = rng

        self.neural_net = neural_net
        if not mode == "external":
            self.neural_net.optimizer.zero_grad()
        self.loss_function = base.LOSS_FUNCTIONS[loss_function.get("name").lower()](
            loss_function.get("args", None), **loss_function.get("kwargs", {})
        )

        self.current_loss = torch.tensor(0.0)

        self.to_learn = {key: idx for idx, key in enumerate(physical.parameter_names)}
        self.true_parameters = {
            key: torch.tensor(val, dtype=torch.float)
            for key, val in true_parameters.items()
        }
        self.current_predictions = torch.zeros(len(self.to_learn))

        # Training data (time series of T_in) and external forcing (T_out, Q_H, Q_O)
        self.training_data = training_data
        self.external_data = external_data

        

        if not (isinstance(train_range, float) or isinstance(train_range, list)) or not (isinstance(val_range, float) or isinstance(val_range, list)):
            raise indexError("Wrong Training.train_range or Taining.val_range format. Please provide as float in (0, 1) or tuple (start, stop)")

        if isinstance(train_range, float):
            self.train_range = slice(*[0, int(training_data.shape[1]*train_range)])
        else:
            self.train_range = slice(*train_range)

        if isinstance(val_range, float):
            self.val_range = slice(*[self.train_range.stop, self.train_range.stop + int(training_data.shape[1]*val_range)])
        else:
            self.val_range = slice(*val_range)

        if mode == "generalize":
            self.best_model = None
            self.best_val_loss = float('Inf')
        elif mode == "predict" or mode == "single-input":
            self.training_data = training_data[:1, :, :, :]
            self.train_range = slice(self.train_range.start, self.val_range.stop)
            self.val_range = slice(self.train_range.stop, self.train_range.stop)
   
        self.physical = physical

        # Time differential to use for the numerical solver
        self.dt = dt

        # Generate the batch ids
        self.batch_size = batch_size

        # range of data for NN input (lookback) and steps the parameters are simulated over (horizon) should be same.
        self.horizon = lookback
        self.lookback = lookback
        self.slicing_difference = int(slicing_difference)

        
        self.mode = mode
        self.external_data = external_data

        physical.dt = dt

        # Scaling factors to use for the parameters, if given
        self.scaling_factors = scaling_factors

        self.current_hidden_states = None

        # --- Set up chunked dataset to store the state data in --------------------------------------------------------
        # Write the loss after every batch
        if self._h5group is not None:
            self._dset_loss = self._h5group.create_dataset(
                "loss",
                (0,),
                maxshape=(None,),
                chunks=True,
                compression=3,
            )
            self._dset_loss.attrs["dim_names"] = ["batch"]
            self._dset_loss.attrs["coords_mode__batch"] = "start_and_step"
            self._dset_loss.attrs["coords__batch"] = [write_start, write_every]


            self.dset_hidden_states = self._h5group.create_dataset(
                "hidden_states",
                (0,),
                maxshape=(None,),
                chunks=True,
                compression=3,
            )
            self.dset_hidden_states.attrs["dim_names"] = ["batch"]
            self.dset_hidden_states.attrs["coords_mode__batch"] = "start_and_step"
            self.dset_hidden_states.attrs["coords__batch"] = [write_start, write_every]

            # Write the computation time of every epoch
            self.dset_time = self._h5group.create_dataset(
                "computation_time",
                (0,),
                maxshape=(None,),
                chunks=True,
                compression=3,
            )
            self.dset_time.attrs["dim_names"] = ["epoch"]
            self.dset_time.attrs["coords_mode__epoch"] = "trivial"

            # Write the parameter predictions after every batch
            self.dset_parameters = self._h5group.create_dataset(
                "parameters",
                (0, len(self.to_learn.keys())),
                maxshape=(None, len(self.to_learn.keys())),
                chunks=True,
                compression=3,
            )
            self.dset_parameters.attrs["dim_names"] = ["batch", "parameter"]
            self.dset_parameters.attrs["coords_mode__batch"] = "start_and_step"
            self.dset_parameters.attrs["coords__batch"] = [write_start, write_every]
            self.dset_parameters.attrs["coords_mode__parameter"] = "values"
            self.dset_parameters.attrs["coords__parameter"] = physical.parameter_names
            self.dset_parameters.attrs["data_used_up_to"] = self.val_range.stop


        self._time = 0
        self._write_every = write_every
        self._write_start = write_start

        # disable torch threading to not overload cpu scheduler for many parallel tuning tasks
        if mode == "tuning":
            torch.set_num_threads(1)
#            torch.set_num_interop_threads(1)



    def shuffle_data(self, subset):

        '''
        calculates the training example slices based on config parameters
        args:
            - subset: slice - the subset of the data that should be used (can be valiadtion or training set)
        returns:
            - a randomly ordered list of tuples of shape (building, slice) to be used as training examples

        the returned tuples map onto the generated training data to get the training examples as such:
            training_example = training_data[tuple[0], tuple[1]] and external_te_data = external_data[tuple[0], tuple[1]]

        '''
        if abs(subset.stop-subset.start) < self.horizon or abs(subset.stop-subset.start) < self.lookback:
            raise IndexError(f"Error: sample size or lookback lie out of provided data range {subset}!")
        
        # this window size refers to the required data per training example, not necessarily the horizon or input to the neural net
        window_size = max(self.horizon+1, self.lookback)
        # [][][][][][][][][][][][][][][][][]
        # | lookback       |
        # | |-------------horizon ---------|
        # ^ initial condition for horizon simulation -> ODE is evalated <horizon> steps from the initial condition
        #   => resulting trajectory that is compared against truth has length horizon+1 ([T_init, simulation])
        out = []
        for k in range(self.training_data.shape[0]): # for every building in the data
            index = 0
            while index <= abs(subset.stop-subset.start)-window_size:
                out.append((k, slice(index, index + window_size, 1)))
                index += self.slicing_difference
        random.shuffle(out)
        return out
        
    def predict_and_simulate(self, training_example, te_range):
        '''
        predicts parameters based on the data contained in self.data[te_range], simulates the ODE and returns a loss
        args:
            - training_example: int - ID of training_example, deprecated
            - te_range: range of data that sets the specific training example (building, slice of timesteps)
        returns:
            - loss that resulted from the predicted parameters produced by the current state of self.neural_net
        '''
        
        # format training_example (concat dynamic and external data)
        newdata = torch.cat((self.training_data[te_range][:self.lookback],
                                            self.external_data[te_range][:self.lookback]),1)

        # format the shape to match the type of neural_net
        if self.mode == "single-input":
            nn_input = torch.flatten(self.training_data[te_range][0])
            
        elif self.neural_net.type == "lstm":
            nn_input = newdata[:,:,0].unsqueeze(1)
        else: #predict or generalize mlp or optimizer
            nn_input = torch.flatten(newdata)

        # predict some parameters
        predicted_parameters = self.neural_net(
            nn_input.float()
        )


        # Scale predicted parameters if scale is passed or get true values from config
        parameters = torch.stack([self.true_parameters[key]
                        if key in self.true_parameters.keys()
                        else self.scaling_factors.get(key , 1.0)*predicted_parameters[self.to_learn[key]]
                        for key in self.physical.parameter_names]).double()

        # set up ODE model
        self.physical.set_params(parameters)
        self.physical.reset()
               
        # Get current initial condition and make traceable
        initial_condition = self.training_data[te_range][0].clone()
        initial_condition.requires_grad_(True)

        # set up loss and time vector, pass external data to ODE object
        loss = torch.tensor(0.0, requires_grad=True)
        time = torch.arange(0, self.horizon+1, dtype = torch.float32)
        self.physical.external_data = self.external_data[te_range]
        
        # simulate the ODE over horizon
        trajectory = odeint(self.physical,
                                initial_condition,
                                time,
                                method = 'euler',
                                options={'step_size': 1.0}
                                )
       
        ground_truth = self.training_data[te_range[0], te_range[1].start:te_range[1].start+self.horizon+1]
        loss = self.loss_function(trajectory, ground_truth)# / (te_range[1].stop- te_range[1].start) #normalize loss over te_range
    
        return loss, parameters, self.physical.last_hidden_init


    def epoch(self):
        """
        An epoch is a pass over the entire dataset. The data was sliced into training examples (their indices
        are stored). After iterating over batch_size training examples, a backpropagation and optimization
        process is started.

        """

        # if the neural net is an optimize function (e.g. scipy.minimize or Modestpy.minimize())
        if self.mode == "external":
            return self.neural_net(self.optimize_func)



        # Process the training data in batches

        #for training_example, batch_idx in enumerate(self.batches[:-1]):
        batch_loss = 0
        for training_example, te_range in enumerate(self.shuffle_data(self.train_range)):


            loss, parameters, last_hidden_init = self.predict_and_simulate(training_example, te_range)

            batch_loss += loss

            self.neural_net.optimizer.current_loss = loss.item()
            
            # record loss if it is supposed to be collected from training set (only durin estimation)

            if not (self.mode == "generalize" or self.mode == "tuning"):
                self.current_loss = loss.clone().detach().cpu().numpy().item()
                self.current_predictions = parameters.clone().detach().numpy()
                self.current_hidden_states = last_hidden_init
                #self.current_predictions = predicted_parameters.clone().detach().cpu().numpy()

                self._time += 1
                self.write_data()
            if (training_example+1)%self.batch_size == 0:
            #    log.info(f"finished batch {int(training_example/self.batch_size)} with loss {batch_loss}")
                batch_loss.backward()
                self.neural_net.optimizer.step()
                self.neural_net.optimizer.zero_grad()
                batch_loss = 0

        # collect loss from validation set if tuning or generalizing
        if self.mode == "generalize" or self.mode == "tuning":
            loss = 0
            for training_example, te_range in enumerate(self.shuffle_data(self.val_range)):
                l, params, _ = self.predict_and_simulate(training_example, te_range)
                loss += l/(self.batch_size*(self.val_range.stop - self.val_range.start))
            self.current_loss = loss.clone().detach().cpu().numpy().item()
            self.current_parameters = params.clone().detach().numpy()
            self._time += 1
            self.write_data()

            # optuna handles best model selection during tuning
            if self.mode == "generalize":
                if self.current_loss < self.best_val_loss: 
                    log.info(f"Found new best model with val loss {self.current_loss} < last best loss {self.best_val_loss}")
                    self.best_model = self.neural_net.state_dict()
                    self.best_val_loss = self.current_loss
                    


            
            

    def write_data(self):
        """Write the current state (loss and parameter predictions) into the state dataset.

        In the case of HDF5 data writing that is used here, this requires to
        extend the dataset size prior to writing; this way, the newly written
        data is always in the last row of the dataset.
        """
        if self._time >= self._write_start and (self._time % self._write_every == 0) and self._h5group is not None:
            self._dset_loss.resize(self._dset_loss.shape[0] + 1, axis=0)
            self._dset_loss[-1] = self.current_loss
            #self._dset_loss[-1] = 1
            self.dset_parameters.resize(self.dset_parameters.shape[0] + 1, axis=0)
            self.dset_parameters[-1, :] = [
                self.current_predictions[self.to_learn[p]] for p in self.to_learn.keys()
            ]

            self.dset_hidden_states.resize(self.dset_hidden_states.shape[0] + 1, axis = 0)
            self.dset_hidden_states[-1] = self.current_hidden_states if self.current_hidden_states else float('Nan')



