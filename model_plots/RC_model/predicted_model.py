import logging

import sys

import xarray as xr
from os.path import dirname as up
import os

from utopya.eval import is_operation

from dantro._import_tools import import_module_from_path, get_from_module

import numpy as np
import torch
from torchdiffeq import odeint


sys.path.append(os.path.join(up(up(up(__file__))), "models", "RC_model"))

Physicals = import_module_from_path(mod_path=os.path.join(up(up(up(__file__))), "models", "RC_model"), mod_str="RC_model")



log = logging.getLogger(__name__)


@is_operation("simulate_model")
def simulate(
    marginals: xr.DataArray,
    *args,
    model_type: str,
    dt: int,
    rc_data: xr.Dataset,
    mode: str = None,
    horizon: int = 10e10,
    simulate_from_index: int = 0,
    export: str = None,
    show_global_errs = True,
    mean_over: int = 1,
    marginals_key: str  = "y",

):
    #print(f"simulate model called with {len(rc_data)} datasets and model type = {model_type}")
    # read attributes and real data from multiverse data if called from multiverse
    if isinstance(model_type, xr.DataArray):
        model_type = model_type.values.flat[0]
        
        dt = dt.values.flat[0]
    elif isinstance(model_type, list):
        dt = dt.values.flat[0]
        

    if isinstance(simulate_from_index, xr.DataArray):
        simulate_from_index = simulate_from_index.values.flat[0]


    if not isinstance(marginals, list):
        marginals = [marginals]

   #print(simulate_from_index)


    # get physical obkect for sim
    if isinstance(model_type, list):
        physical = [get_from_module(Physicals, name = mod)() for mod in model_type]
    else:
        physical = [get_from_module(Physicals, name = model_type)()]

    # remove all multiverse dimensions
    #rc_data = np.array(rc_data.values.flat)[:rc_data.sizes["time"]*rc_data.sizes["kind"]]
    #rc_data = rc_data.reshape((int(rc_data.shape[0]/len(physical.plot_args)), len(physical.plot_args)))

    if isinstance(rc_data, list):
        dims_to_reduce = [dim for dim in rc_data[0].dims if dim not in ("time", "kind")]
        rc_data = [np.array(data.isel({dim: 0 for dim in dims_to_reduce})) for data in rc_data]
    else: 
        dims_to_reduce = [dim for dim in rc_data.dims if dim not in ("time", "kind")]
        rc_data = [np.array(rc_data.isel({dim: 0 for dim in dims_to_reduce}))]

    if not show_global_errs:
        rc_data = [data[simulate_from_index:, :] for data in rc_data]
    longest_data = rc_data[np.argmax([data.shape[0] for data in rc_data])]
    
    all_sims = []

    #print("rc_shapes", [data.shape for data in rc_data])

    for i, phys in enumerate(physical):

        # read best parameters (currently from min(loss) or max(neg_exp) - might make sense to fit gauss?
        params = []
        best_params = []
        for k, key in enumerate(marginals[i].coords["parameter"].values):
            params.append(key)
            best_idx = marginals[i][marginals_key][k].argmax()
            best_params.append(marginals[i]["x"][k][best_idx].values.item())
        # reorder params to match the physical's order
        best_params = [best_params[params.index(param_name)] for param_name in phys.parameter_names]

        if i%mean_over == 0:
            log.info(f"============= Meaning over a new set of {mean_over} sims... ====================== ")

        log.info(f" {model_type[i]} using {[f'{phys.parameter_names[l]}: {best_params[l]}' for l in range(len(params))]} over {rc_data[i].shape[0]}, {simulate_from_index}:")

        # set the physical up for simulation
        phys.set_params(torch.Tensor(best_params))
        phys.dt = dt
        phys.external_data = torch.Tensor(rc_data[i])[:,phys.dynamic_variables:]

        # set up simulation and run
        time = []
        if rc_data[i].shape[0] <= horizon:
            time.append(torch.arange(0, rc_data[i].shape[0], 1.0))
        else:
            for k in range(int((rc_data[i].shape[0])/horizon)+int((rc_data[i].shape[0])/horizon%1!=0)):
                time.append(torch.arange(k*horizon, min((k+1)*horizon, rc_data[i].shape[0]), 1.0))
        sims = []
        for t in time:
            sim = odeint(phys,
                        torch.Tensor(rc_data[i])[int(t[0]), :phys.dynamic_variables],
                        t,
                        method = 'euler',
                        options={'step_size': 1.0}
                        ).unsqueeze(0).unsqueeze(2)
            sims.append(sim)
            phys.reset()
        #print(sim)
        #print(np.isnan(sim).any())
        sim = torch.cat(sims, dim = 1)
        log.info(f"=> test MAE {np.mean(np.abs(np.array((sim[0, :, :physical[i].dynamic_variables, 0] - torch.Tensor(rc_data[i])[:,:physical[i].dynamic_variables])[:, 0])[simulate_from_index:]))}")
        all_sims.append(sim)

    


    delta = [(sim[0, :, :physical[i].dynamic_variables, 0] - torch.Tensor(rc_data[i])[:,:physical[i].dynamic_variables])[:, 0].numpy() for i, sim in enumerate(all_sims)]
    #print(f"{delta[0][simulate_from_index:]}")
    #print(np.max(np.abs(np.array(delta)[:, simulate_from_index:])))
    #print(np.array(delta[0][simulate_from_index:]) - np.array(delta)[:, simulate_from_index:][0])


    mae = np.array([np.mean(np.abs(d)) for d in np.array(delta)[:, simulate_from_index:]])
    rmse = np.array([np.sqrt(np.mean(d**2)) for d in np.array(delta)[:, simulate_from_index:  ]])
    global_mae = mae
    global_rmse = rmse
    if show_global_errs:
        global_mae = np.array([np.sum(np.abs(d)/d.shape[0]) for d in delta])
        global_rmse = np.array([np.sqrt((d**2).sum()/d.shape[0]) for d in delta])
    if mean_over != None or mean_over != "None":
        print(mae)
        print(rmse)
        print(global_mae)
        print(global_rmse)
        mae = np.array([np.mean(mae[i*mean_over:(i+1)*mean_over]) for i in range(int(len(mae)/mean_over))])
        rmse = np.array([np.mean(rmse[i*mean_over:(i+1)*mean_over]) for i in range(int(len(rmse)/mean_over))])
        global_mae = np.array([np.mean(global_mae[i*mean_over:(i+1)*mean_over]) for i in range(int(len(global_mae)/mean_over))])
        global_mae = np.array([np.mean(mae[i*mean_over:(i+1)*mean_over]) for i in range(int(len(global_rmse)/mean_over))])
        print(mae)
        print(rmse)
        print(mae.shape)
    sim_out = np.array([torch.Tensor(rc_data[np.argmax([data.shape[0] for data in rc_data])][:, 0])] + [pad_arr_with_nans(sim[0, :, 0, 0], longest_data.shape[0]) for i, sim in enumerate(all_sims)]).T
    delta_out = np.array([pad_arr_with_nans(delt, longest_data.shape[0]) for delt in delta]).T
    if mode == None:
        return sim_out, delta_out, mae, rmse, global_mae, global_rmse
    # add real data along axis 2 or compute delta depending on mode
    if mode == "sim":

        out = torch.cat([torch.Tensor(rc_data[i])[:, :1]] + [sim[0, :, :1, 0] for i, sim in enumerate(all_sims)], axis = 1)
        #print(out.shape)
        return out
    elif mode == "mae":
        delta = [(sim[0, :, :physical[i].dynamic_variables, 0] - torch.Tensor(rc_data[i])[:,:physical[i].dynamic_variables])[:, 0] for i, sim in enumerate(all_sims)]
        mae = np.array([np.sum(np.abs(np.array(d))/d.shape[0]) for d in delta])
        #print(mae)
        return mae
    else:
        delta = [(sim[0, :, :physical[i].dynamic_variables, 0] - torch.Tensor(rc_data[i])[:,:physical[i].dynamic_variables])[:, 0].numpy() for i, sim in enumerate(all_sims)]
        mae = np.array([np.ones(d.shape[0])*np.sum(np.abs(np.array(d))/d.shape[0]) for d in delta])
        #print(mae)
        log.info(delta)
        log.info(mae)
        return np.concatenate([delta, mae], axis = 0).T

def pad_arr_with_nans(arr, target_len):
    return np.concatenate((np.array(arr), np.full((target_len-len(arr), *arr.shape[1:]), np.nan)))





