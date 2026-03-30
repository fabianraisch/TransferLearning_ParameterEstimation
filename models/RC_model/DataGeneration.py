import h5py as h5
import logging
import sys
import torch
from os.path import dirname as up
import numpy as np
import pandas as pd
import Physicals
import copy
from dantro._import_tools import get_from_module
import os
import random

sys.path.append(up(up(up(__file__))))

from dantro._import_tools import import_module_from_path

base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")
print(up(up(up(__file__))))

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
# Data loading and generation utilities
# ----------------------------------------------------------------------------------------------------------------------


def apply_controller(cfg, data, heatPower_index):
    '''
    Function that implements and applies rudimentary control algorithms for heating control
    args:
        - cfg: The run config defining simulation parameters (max heat Power, temperature setpoints, ...)
        - data: external data like internal temperature or weather, only for one timestep
        - heatPower_index: the index of data where the heating Power can be found (for two PointControl)

    return:
        - heatPower: the numerical value of the Power the building is heated with

    '''
    if cfg["controller"] == "TwoPointControl":
        if data[0] < cfg["T_min"]:
            heatPower = cfg["maxHeatingPower"]
        elif data[0] >= cfg["T_max"]:
            heatPower = 0
        else:
            heatPower = data[heatPower_index]
    elif cfg["controller"] == "PControl":
        heatPower = int(data[0]<=cfg["T_max"])*(cfg["T_max"]-data[0])*cfg["maxHeatingPower"]/(cfg["T_max"]-cfg["T_min"])
        if heatPower > cfg["maxHeatingPower"]:
            heatPower = cfg["maxHeatingPower"]
    elif cfg["controller"] == "RampUp":
        heatPower = data[heatPower_index] + cfg["maxHeatingPower"]/cfg["num_steps"]
    return heatPower


def generate_weather_based_data(cfg: dict, *, dt: torch.Tensor) -> torch.Tensor:
    """ Function that generates weather data based time series of length num_steps based on a defined Physical.

    :param cfg: configuration of data settings
    :param dt: time differential for the numerical solver (Euler in this case)
    :return: torch.Tensor of the time series for the values specified in the physical. Tensor has shape (num_states, num_steps)
    """

    # read the provided weather data
    wdata, _ = read_mos(cfg["weather_data"])
    wdata = wdata.to_numpy(dtype = float)

    # initialize dt in hours for reading the weather data, which should be provided in hourly steps
    dt_hours = float(dt.item() / 3600)

    # if raw heating power is provided and not to be computed, load it and set flag
    pred_heat = False
    if "heating_data" in cfg:
        heat_data = np.array(pd.read_csv(cfg["heating_data"])["heatPower"])
        pred_heat = True

    # initialize the physical ODE object
    model = get_from_module(Physicals, name = cfg["model_type"])()
    initial_condition = torch.Tensor(model.initial_condition(cfg, wdata))

    # if lists of parameters are passed, set up the simulation of multi
    parameters = [cfg[param] for param in model.parameter_names]
    if isinstance(parameters[0], int) or isinstance(parameters[0], float):
        parameters = [[parameters]]
    out = []

    # Generate some synthetic time series. Generate multiple if specified
    for params in zip(*parameters):
        data = []
        state = torch.tensor(initial_condition[:model.dynamic_variables], dtype = torch.float64).unsqueeze(-1)
        heatPower = 0
        # if parameter list is not set up correctly, use first entry of lists
        if isinstance(params[0], list):
            params = params[0]

        # convert parameters into torch tensor for compatibility with odeint (used in the 'physical' object)
        params = torch.tensor(params, dtype = torch.float64)

        # integrate over specified number of timesteps (+1 for inital condition)
        for i in range(cfg['num_steps'] + 1):

            # calculate the index, where current weather data is stored
            weather_index = int(i*dt_hours)

            # fetch external data (external temperature, solar gains)
            dat = [wdata[weather_index][2]+273.15, heatPower, wdata[weather_index][8]*cfg["effWinArea"]]

            # fetch external heating power or calculate using specified controller
            if pred_heat:
                heatPower = heat_data[i]
            else:
                heatPower = apply_controller(cfg, torch.cat((state.squeeze(1), torch.tensor(dat))), model.data_names.index("heatPower"))

            # these format acrobatics are done to use the same step function in the NN and here
            dat = torch.tensor([wdata[weather_index][2]+273.15, heatPower, wdata[weather_index][8] if cfg["store_raw_QSolar"] else wdata[int(i*dt/3600)][8]*cfg["effWinArea"]], dtype = torch.float64).unsqueeze(-1)
            
            # store current state of dynamic variables
            data.append(torch.cat((
                state,
                dat
                )))

            # integrate the timestep and resume
            state = data[-1][:model.dynamic_variables] + model.step(state, dat, params, dt)

        # append timeseries for current set of parameters to the data to be returned
        out.append(torch.reshape(torch.stack(copy.deepcopy(data)), (len(data), len(model.data_names), 1)))

    # reformat tensor to shape (param_set, timestep, features, 1)
    return torch.reshape(torch.stack(out), (len(out), len(out[0]), len(model.data_names), 1))


def get_RC_circuit_data(*, data_cfg: dict, h5group: h5.Group):
    """Returns the training data for the RC_circuit model:

    Either data is synthetically generated by not providing a "load_from_dir" key in the config
    Or data is loaded from a csv file if specified through the "load_from_dir" key

    :param data_cfg: dictionary of config keys
    :param h5group: h5.Group to write the training data to
    :return: torch.Tensor of training data

    """
    if "load_from_dir" in data_cfg.keys():

        # if source csv is provided, check for list of files
        if not isinstance(data_cfg["load_from_dir"]["path"], list):

            # if passed filename is a csv, wrap into list for further processing
            if data_cfg["load_from_dir"]["path"].endswith(".csv"):
                data_cfg["load_from_dir"]["path"] = [data_cfg["load_from_dir"]["path"]]

            # if passed filename does not contain ".csv", it is interpreted as a directory and all underlying csvs
            # starting with a "_" are appended to the list of files to be loaded
            else:
                rootpath = data_cfg["load_from_dir"]["path"]
                data_cfg["load_from_dir"]["path"] = []
                for root, dirs, files in os.walk(rootpath):
                    for file in files:
                        if file.startswith("_"):
                            log.debug(f"found file {os.path.join(root, file)}")
                            data_cfg["load_from_dir"]["path"].append(os.path.join(root, file))

                # shuffle the loaded csv paths for generalization training
                random.shuffle(data_cfg["load_from_dir"]["path"])
                log.info(f"\tDetected folder as dirpath, loading {len(data_cfg['load_from_dir']['path'])} csvs that start with \"_\".")

        out = []

        # load the whole list of filenames, read specified columns (see data.load_from_dir.csv_keys key in config)
        # and take the subset of timesteps specified through the data.load_from_dir.subset key
        for path in data_cfg["load_from_dir"]["path"]:
            with open(path, "r") as f:
                df = pd.read_csv(f)
                keys = data_cfg["load_from_dir"]["csv_keys"]
                data = torch.from_numpy(np.array([[df[keys[0]],
                                                    df[keys[1]],
                                                    df[keys[2]],
                                                    np.array(df[keys[3]])*data_cfg["load_from_dir"]["effWinArea"]]]
                                                    )).float()[:, :, :data_cfg["load_from_dir"]["subset"]].T.unsqueeze(0)
            out.append(data)

        data = torch.cat(out, axis  = 0)

        # store names of columns in h5 dataset
        attributes = data_cfg["load_from_dir"]["csv_keys"]

    # else if load_from_dir is not specified, synthetic data is generated from weather data contained in a .mos file, an RC architecture and inital condition
    elif "synthetic_data" in data_cfg.keys() and not "load_from_dir" in data_cfg.keys():

        # copy physical model information into synthetic data config
        data_cfg["synthetic_data"]["model_type"] = data_cfg["model_type"]

        # generate one or multiple timeseries based on configuration
        data = generate_weather_based_data(
            cfg=data_cfg["synthetic_data"],
            dt=torch.tensor(data_cfg["dt"], dtype = torch.float64)
        )

        # store names of columns specified through implementation of physical model
        attributes = get_from_module(Physicals, name = data_cfg["model_type"])().data_names
    
    else:
        raise ValueError(
            f"You must supply one of 'load_from_dir' or 'synthetic data' keys!"
        )

    if h5group is not None:
        # Store the synthetically generated data in an h5 file
        dset = h5group.create_dataset(
            "RC_data",
            data.shape,
            maxshape=data.shape,
            chunks=True,
            compression=3,
            dtype=float,
        )

        dset.attrs["dim_names"] = ["permut", "time", "kind", "dim_name__0"]
        dset.attrs["coords_mode__time"] = "trivial"
        dset.attrs["coords_mode__kind"] = "values"
        dset.attrs["coords__kind"] = attributes
        dset.attrs["coords_mode__dim_name__0"] = "trivial"

        dset[:, :] = data
    
    
    return data

def read_mos(filename):
    '''
    Function that reads weather data from a .mos file formatted like the ones used in the
    https://github.com/fabianraisch/BuilDa.git project from arXiv:2512.00483
    args:
        - filename: The path to the mos file the weather data is loaded from
    returns:
        - a pd.DataFrame containing the whole mos data
        - the header of the file as string

    '''

    print(f"Reading reference file {filename} for data generation")
    with open(filename, "r") as f:
        data = f.read()

    # scans for header and data and splits it
    n_cols = int(data[data.find(",")+1: data.find(")")])  #start of header contains "*double tab1(rows,cols)\n*"
    last_header_line = data.find(f"C{n_cols}")
    header_end = data[last_header_line:].find("\n")+last_header_line
    header = data[:header_end+1] 
    dat = data[header_end+1:]

    #converts the data to a numpy array
    arr1 = dat.split("\n")[:-1]
    arr2 = np.array([i.split("\t") for i in arr1])
    header_shape = (int(header[header.find("(")+1:header.find(",")]), int(header[header.find(",")+1:header.find(")")]))
    if arr2.shape != header_shape:
        print(f"ERROR while reading .mos file: list dimensions {arr2.shape} do not match header {header_shape}!")
        exit()
    df = pd.DataFrame(arr2, columns = [f"C{i+1}" for i in range(n_cols)])
    return df, header
