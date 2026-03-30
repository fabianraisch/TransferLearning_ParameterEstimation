import copy
from utopya.eval import is_operation

from typing import Sequence, Union

import xarray as xr

from utopya.eval import PlotHelper, is_plot_func
import numpy as np
from itertools import product

from math import prod


@is_operation("select_first_uni_time_coords")
def select(data):
    ''' return the time coordinates of the first universe contained in data
    args:
        - data: raw xr.DataSet read from an estimation run's h5 file
    returns:
        - out: np.array of the time axis. Corresponds to the time column in the original ground truth csv

    NOTE:
        this function can probably be implemented in clean dantro/yaml logic, tho the current author does not know how to do it.

    '''
    # select the time coords alon the right axis, this one's pretty hardcoded
    dims_to_reduce = [dim for dim in data.dims if dim not in ("time")]

    # Step 2: Index the dataset at 0 along those dims
    out = np.array(data.isel({dim: 0 for dim in dims_to_reduce}))
    return out

@is_operation("flatten_dims_except")
def flatten_dims(
    ds: Union[xr.Dataset, xr.DataArray, list],
    dims_to_keep,
    dim_name,
    *,
    new_coords: Sequence = None,
) -> Union[xr.Dataset, xr.DataArray]:
    ''' Flattens all dimensions except the specified ones into a newly generated dimension.
        If a list is passed, the function is applied to each list element independantly.
    args:
        - ds: xr data (or list thereof) containing arbitrary dimensions
        - dims_to_keeo: a list of dimensions that are to be kept intact
        - dim_name: name of the new dimension all leftovers are to be concatenated into
        - new_coords: OPTIONAL - manually set the coordinates of the new dimension
    returns:
        - out: a new xr object (or list thereof) containing the original data in reformatted dimensions

    '''

    # if the passed data is a list, apply the function to each list element individually
    if isinstance(ds, list):
        return [flatten_dims(element, dims_to_keep, dim_name, new_coords = new_coords) for element in ds]

    # parse the dimensions to be removed
    new_dim, dims_to_stack = dim_name, [dim for dim in list(ds.dims) if dim not in dims_to_keep]

    # Check if the new dimension name already exists. If it already exists, use a temporary name for the new dimension
    # switch back later
    _renamed = False
    if new_dim in list(ds.coords.keys()):
        new_dim = f"__{new_dim}__"
        _renamed = True

    # Stack and drop the dimensions
    ds = ds.stack({new_dim: dims_to_stack})
    q = set(dims_to_stack)
    q.add(new_dim)
    ds = ds.drop_vars(q)

    # Name the stacked dimension back to the originally intended name
    if _renamed:
        ds = ds.rename({new_dim: dim_name})
        new_dim = dim_name
    # Add coordinates to new dimension and return
    if new_coords is None:
        out =  ds.assign_coords({new_dim: np.arange(len(ds.coords[new_dim]))})
    else:
        out =  ds.assign_coords({new_dim: new_coords})

    return out

@is_operation("broadcast_dims")
def broadcast_e(
    ds1: xr.DataArray, ds2: xr.DataArray, broadcast_dims: list, *, x: str = "x", p: str = "loss", **kwargs
) -> xr.Dataset:
    all_coords = list(dict.fromkeys(list(ds1.coords) + list(ds2.coords)))
    exclude_dims = list(set(all_coords) - set(broadcast_dims))

    return xr.broadcast(xr.Dataset({x: ds1, p: ds2}), exclude = exclude_dims, **kwargs)[0]

@is_operation("dims2list")
def split_dataset_along_dim(ds, dim):
    ''' Splits the entries of an xr.DataSet along a dimension into a list.
        If passed data is a list, the function is applied to each list element independently.
    args:
        - ds: xr.DataSet to convert (or list thereof)
        - dim: dimension that is converted to a list
    returns:
        - out: a list of datasets. Each dataset is one entry along the dimension specified.
    '''
    if isinstance(ds, list):
        nested = [split_dataset_along_dim(i, dim) for i in ds]
        return [item for sublist in nested for item in sublist]

    out = [
        drop_fully_nan_entries(ds.isel({dim: i}).drop_vars(dim, errors="ignore"))
        for i in range(ds.sizes[dim])
        ]

    return out


def drop_fully_nan_entries(data):
    """
    Iteratively drops entries from all dimensions where values are fully NaN.
    Works for both DataArray and Dataset.
    """
    for dim in data.dims:
        # Get all other dims
        reduce_dims = [d for d in data.dims if d != dim]

        if reduce_dims:  # skip reduction if scalar
            mask = ~data.isnull().all(dim=reduce_dims)
            data = data.sel({dim: data[dim][mask]})

    return data

@is_operation("get_model_types_from_multiverse")
def get_mtypes(ds: xr.DataArray, groupings: list):
    ''' Get the physical model type for each estimation run.
        Format the list to match the scheduled simulations for later evaluation.
    args:
        - ds: dataArray containing raw estimation run data (from the h5 file)
        - groupings: List of sweep dimensions that are independantly evaluated.
    returns:
        - out: a list of string representations of the physical model types used in each estimation run.

    NOTE: running a multiverse sweep over different model types HAS to result in them being compared or meaned over
          (meaning over a dimension internally results in it being added to the "groupings" list). Otherwise parametrizations
          of different architectures would have to be mixed which is impossible and thus not implemented.
    '''

    group_shape = [len(ds.coords[dim]) for dim in groupings if not (dim == "None" or dim == None)]

    # if model type is not to be compared, return an array with the same model for all runs.
    if not "model_type" in groupings:
        return [ds.attrs["model_type"] for _ in range(np.array(group_shape).prod())]

    
    # check the amount of simulations that are scheduled with each physical model type and duplicate entries accordingly
    dim = "model_type"
    out = []
    padding = prod(group_shape[groupings.index("model_type")+1:])
    repeat = prod(group_shape[:groupings.index("model_type")])


    for _ in range(repeat):
        for model_type in ds[dim].values:
            for _ in range(padding):
                out.append(model_type)

    return out

@is_operation("cart_prod_string")
def cart_prod_string(a, b):
    ''' Returns the cartesian product of strings from two lists. Used for generating short descriptions and legends
    '''
    return [elem[0] + elem[1] for elem in product(a, b)]


@is_operation("bar_plot_groups")
def compute_grouped_bar_positions(arr: xr.DataArray, dimensions: list, base=1.2, step=0.1):
    """
    Computes bar positions for arbitrarily nested groups.
    
    Args:
        arr: an xr.DataArray of all multiverses
        dimensions: a list of dimensions that are supposed to be grouped togther, from biggest group to smallest
        base (float): Starting base position for the outermost group.
        step (float): Step size between nested group levels (affects spacing).
        
    Returns:
        List of float positions.
    """

    group_shape = [len(arr.coords[dim]) for dim in dimensions]
    # Create all index combinations, e.g. (0, 1, 2)
    index_ranges = [range(n) for n in group_shape]
    index_combinations = list(product(*index_ranges))

    positions = []
    for combo in index_combinations:
        pos = 0.0
        for i, idx in enumerate(combo):
            pos += idx * (step ** (i))
        positions.append(base + pos)

    return positions

@is_operation("pad_array")
def pad_array(arr, padding = 1):
    ''' Function that duplicates each entry of an array [padding] times
    '''
    out = []
    for element in arr:
        for _ in range(padding):
            out.append(element)
    return out

@is_operation("repeat_array")
def repeat_array(arr, rep = 1):
    ''' function that repeats an array [rep] times
    '''
    out = []
    for _ in range(rep):
        for element in arr:
            out.append(element)
    return out


@is_operation("slice_array")
def slice_array(array, start, stop = None):
    ''' function that applies the slice operator since I have not found a way to do that in yaml-dantro syntax
    '''
    return array[start:stop]

@is_operation("list_by_dims")
def list_by_dims(data, groupings):
    ''' Creates an ordered list following the specification in groupings
    args:
        - data: an xr.DataSet from a parameter estimation run containing multiple sweep dimensions
        - groupings: List of sweep dimensions setting up the order of results. 
            First grouping element is the outermost order, last one is split closest together. This also applies to the resulting bar plots.
    retuns:
        - out: a list of datasets that were split along the groupings dimensions.
    '''

    out = data
    for dim in groupings:
        if not dim == "None" or dim == None:
            out = split_dataset_along_dim(out, dim)
    return out

@is_operation("get_permut_description")
def get_permut_description(data, groupings):
    ''' generate labels for each compared multiverse run.
    args:
        - data: some dataset from the entire multiverse run containing all dimension info
        - groupings: the dimensions that are to be compared later.
    returns:
        - out: a list of strings containing the information about each coordinate combination across all dimensions
    '''

    names = list(product(*_map_pathnames([data.coords[dim].values.tolist() for dim in groupings if not (dim == "None" or dim == None)])))
    return [str(name) for name in names]

@is_operation("filter_for_eval_method")
def filter_for_eval_method(paramloss, groupings, eval_method = None, dimdata = None, mean_over_count = 1):
    ''' This function applies different parameter selection strategies, like last sample, best sample, etc.
        This is done by replacing list entries (hence all the list arithmetic upto this point) with single parameter samples if
        configured by the eval_method dict.
    args:
        - paramloss: list of joint xr.DataSet containing parameter samples alongside their respective loss
        - groupings: List of dimensions used for grouping the bar plots, from which the order of sweeps in the paramloss list is contained.
        - eval_method: sweep keys and the desired evaluation method (e.g. {'mlp': ['best', 'first'], 'optimizer': 'best'})
        - dimdata: arbitrary untouched raw dataset from the estimation's h5 file containing all the dimension information
        - mean_over_count: the amount of simulations (one entry in paramloss results in one simulation later) the final error is meaned over.
                This is relevant as paramloss entries are duplicated wrt mean_over_count but eval_method entries are not.
    returns:
        - paramloss: same as arg paramloss except the entries for which no marginalization is desired are replaced by the single respective sample
    '''
    if eval_method == None or eval_method == "None":
        return paramloss
    
    # generate an array containing the important sweep dimensions corresponding to the paramloss list, which does not contain this information
    dims = pad_array(list(product(*[dimdata.coords[dim].values.tolist() for dim in groupings])), mean_over_count)

    # iterate over every parameter set that is to be evaluated for a simulation
    for index in range(len(paramloss)):

        # check if all loss values for a set of samples are the same, which would interfere with marginalisation/sample selection.
        # This might happen vor variuos reasons, such as ill configured parameter scales, wrong exp(-L) transformation, underflows etc.
        if paramloss[index].loss.min().item() == paramloss[index].loss.max().item():
            log.warning(f"No probabilities gradient detected! Maybe rescale loss before transforming!")

        # check for every relevant dimension
        for eval_dim in eval_method.keys():

            # check for every dimension if there is a specific eval method attached to it
            for dimname in dims[index]:
                for method in eval_method[eval_dim]:
                    if isinstance(dimname, str):
                        if dimname.endswith(method):
                            # Apply the specified sample selection strategy. Other/more elaborate ones can easily be implemented here.
                            # Just note that list elements with more than one sample (loss dimension > 1) will be marginalized down the line.
                            if method == "best":
                                best_sample = paramloss[index].loss.mean(dim="parameter").argmax().item() #loss is the same for all parameters of one sample, meaning doesnt change anything
                                paramloss[index] = paramloss[index].sel(sample=[best_sample], drop=False)
                                log.debug(f"evaluating list element {index} using best result {np.array(paramloss[index]['x']).T}")

                            elif method == "last":
                                log.debug(f"evaluating list element {index} using last result")
                                last_index = len(paramloss[index].coords["sample"].values)-1
                                paramloss[index] = paramloss[index].sel(sample=[last_index], drop = False)

                            elif method == "first":
                                log.debug(f"evaluating list element {index} using first result")
                                paramloss[index] = paramloss[index].sel(sample=[0], drop = False)

                            else:
                                log.debug(f"evaluating list element {index} using std marginalization")
    return paramloss


def _anystartswith(string, dic):
    # Helper function that checks if the passed string starts with any of the dictionary keys.
    for key in dic.keys():
        if string.startswith(key):
            return True
    return False


@is_operation("prepare_data_for_eval_method")
def prepare_data_for_eval_method(data: xr.DataArray, eval_method: dict):
    ''' This function renames estimation runs for which non-standard evaluation methods have been specified.
        Entries that have more than one method assigned to them are also dupliated.
    args:
        - data: raw joint xr.DataArray from the run h5 file with parameter samples and respective loss joined
        - eval_method: dict {'sweep_key': 'eval_method'} containing entries of swept dimensions and the respective evaluation methods
    returns:
        - data: shaped like input data but with possibly renamed coordinates and possibly dublicated entries.

    '''

    # return if no special eval methods are desired
    if eval_method == None or eval_method == "None":
        return data

    # parse all coordinates where a special evaluation is desired
    dim_dict = {}
    for k, v in data.coords.items():
        for value in v.values.tolist():
            if _anystartswith(str(value), eval_method):
                if not k in dim_dict.keys():
                    dim_dict[k] = [value]
                else:
                    dim_dict[k].append(value)

    # for all special evaluation coordinates, add the method to the coordinate name and duplicate element, if there are multiple methods
    for key, items in dim_dict.items():
        for item in items:
            coords = data[key].values.tolist()
            index = coords.index(item)
            duplicates = []
            for value in eval_method[''.join(filter(lambda x: x.isalpha(), item))]:
                dup_label = f"{item}-{value}"
                duplicates.append(data.isel({key: index}).assign_coords({key: dup_label}))
            before_slice = slice(0, index)
            after_slice = slice(index + 1, None)
            data = xr.concat([data.isel({key: before_slice}), *duplicates, data.isel({key: after_slice})], dim = key)

    return data

@is_operation("get_color_array")
def get_color_array(base_colors: list, groupings: list, dimref: xr.DataArray):
    ''' Repeat the base_colors color list for the smalles group size in groupings. Pull the dimension shapes from dimref
    '''
    group_shape = [len(dimref.coords[dim]) for dim in groupings]
    while len(base_colors) < group_shape[-1]:
        base_colors = base_colors + base_colors
    group_colors = base_colors[:group_shape[-1]]

    return repeat_array(group_colors, prod(group_shape[:-1]))




@is_plot_func(use_dag = True)
def export_simulation_to_csv(*, simulation, hlpr: PlotHelper, csv_heads = None, **kwargs):
    '''
    brutal workaround for storing the raw simulation data as a csv. unfortunately the out_path is not available in the
    dantro yaml transform section and multiple plot_funcs are not definable, therefore a second simulation and call to this
    mock plot func has to be made. Please make me never update this logic ever again D:
    '''
    log.info(hlpr.out_path[:-4] + "csv")
    sim_out = simulation[0]
    delta_out = simulation[1]
    mae = simulation[2]
    rmse = simulation[3]
#    log.info(f"trying to assemble result dataframe using shapes {sim_out.shape}, {delta_out.shape}, {(mae*np.ones(delta_out.shape[0])).shape} and {(rmse*np.ones(delta_out.shape[0])).shape}")
#    pd.DataFrame(np.array([sim_out[:, 0], sim_out[:, 1], delta_out[:, 0], np.ones(delta_out.shape[0])*mae, np.ones(delta_out.shape[0])*rmse]).T, columns = ["ground truth", "simulation", "residuals", "mae", "rmse"]).to_csv(hlpr.out_path[:-4] + ".csv", index = False)
    pd.DataFrame(np.concatenate([sim_out[:, 1:], delta_out], axis = 1), columns = csv_heads).to_csv(hlpr.out_path[:-4] + ".csv", index = False)



@is_operation("filter_loss")
def filter_loss(data: xr.DataArray, max_loss: int):
    ''' function that filters out any loss-parameter samples above a certain loss threshold
    '''
    return data.where(data.loss < max_loss, drop = True)

@is_operation("pick_specific_sample")
def pick_specific_sample(data, method):
    ''' small self sufficient sample picking strategy that can be used like a filter for single estimation runs.
    '''
    if method == "best":
        best_sample = data.loss.mean(dim="parameter").argmax().item() #loss is the same for both parameters, meaning doesnt change anything
        data = data.sel(sample=[best_sample], drop=False)
    elif method == "last":
        last_index = len(data.coords["sample"].values)-1
        data = data.sel(sample=[last_index], drop = False).expand_dims("x")
    return data

@is_operation("map_pathnames")
def _map_pathnames(string):
    ''' Paper-specific filename mapping to shorten legends in plots
    '''
    if isinstance(string, list):
        return [_map_pathnames(i) for i in string]
    match string:
        case "data/RC_model/NN_parameter_estimation/8targets_nnEstim_noWin_inhom_noAct/#Uex_0.25#HeaWal_280000#FawSou_0.19#ZonLen_10#Roo_21#Roo_19#Wea_Amsterdam#Int_NoActivity":
            return "T1"
        case "data/RC_model/NN_parameter_estimation/8targets_nnEstim_noWin_inhom_noAct/#Uex_0.25#HeaWal_40000#FawSou_0.16#ZonLen_7#Roo_22#Roo_21#Wea_Bratislava#Int_NoActivity":
            return "T2"
        case "data/RC_model/NN_parameter_estimation/8targets_nnEstim_noWin_inhom_noAct/#Uex_0.55#HeaWal_150000#FawSou_0.16#ZonLen_7#Roo_23#Roo_20#Wea_Amsterdam#Int_NoActivity":
            return "T3"
        case "data/RC_model/NN_parameter_estimation/8targets_nnEstim_noWin_inhom_noAct/#Uex_0.55#HeaWal_280000#FawSou_0.19#ZonLen_10#Roo_20.5#Roo_19#Wea_Munich#Int_NoActivity":
            return "T4"
        case "data/RC_model/NN_parameter_estimation/8targets_nnEstim_noWin_inhom_noAct/#Uex_0.85#HeaWal_150000#FawSou_0.19#ZonLen_10#Roo_22.5#Roo_22#Wea_Bratislava#Int_NoActivity":
            return "T5"
        case "data/RC_model/NN_parameter_estimation/8targets_nnEstim_noWin_inhom_noAct/#Uex_0.85#HeaWal_40000#FawSou_0.16#ZonLen_7#Roo_22#Roo_19.5#Wea_Munich#Int_NoActivity":
            return "T6"
        case "data/RC_model/NN_parameter_estimation/8targets_nnEstim_noWin_inhom_noAct/#Uex_1.15#HeaWal_280000#FawSou_0.16#ZonLen_7#Roo_23#Roo_23#Wea_Bratislava#Int_NoActivity":
            return "T7"
        case "data/RC_model/NN_parameter_estimation/8targets_nnEstim_noWin_inhom_noAct/#Uex_1.15#HeaWal_40000#FawSou_0.19#ZonLen_10#Roo_23#Roo_21.5#Wea_Amsterdam#Int_NoActivity":
            return "T8"
        case _:
            return string


@is_operation("cut_nans_from_dataset")
def cut_nans(data):
    ''' function that removes nans that sometimes are used to fill big joint datasets containing multiple physical models
        (e.g. when an RC and a 2R2C model are collected into one dataset, the RC model does not contain R2 and C2, hence they are filled with nans)

        if a list instead of a xr.dataset is passed, the function is applied to each list element individually
    '''
    if isinstance(data, list):
        return [cut_nans(item) for item in data]
    #print(data.max().item())
    no_nans = data
    for dim in data.dims:
        no_nans = no_nans.dropna(dim = dim)
    #   print(no_nans.max().item(), no_nans.shape)
    return no_nans

@is_operation("get_mean_over_count")
def get_mean_over_count(ds, dim):
    ''' simple function that returns the size of a dimension to parse the amount of simulations, whose error is meaned over if desired through config
    '''
    if dim[0] == None or dim[0] == "None":
        return 1
    return ds.sizes[dim[0]]


@is_operation("loss_to_probs")
def loss_to_probs(data, **kwargs):
    ''' earlier implemented in the yaml syntax, loss_to_probs converts raw loss values to probabilities by transforming them through exp(-L).

    This implementation allows to separate optimizer-generated losses, which are only inverted to pick the best sample and mlp-generated losses,
    which are still converted using exp(-L) by passing the corresponding names and the method as kwargs.
    '''
    data = data.astype("float64")


    for typ, method in kwargs.items():
        for t in data.coords["type"].values:
            if t.startswith(typ):
                if method == "divide_by_median":
                    median = data.sel(type=t).median(['batch', 'seed'])
                    data.loc[dict(type=t)] = np.exp(- data.sel(type=t)/median)
                elif method == "subtract_min":
                    minimum = data.sel(type=t).min(['batch', 'seed'])
                    data.loc[dict(type=t)] = np.exp(- data.sel(type=t) + minimum)
                elif method == "pure_negexp":
                    data.loc[dict(type=t)] = np.exp(- data.sel(type=t))
                elif method == "just_negate_loss":
                    data.loc[dict(type=t)] = -data.sel(type=t)
                
    return data

@is_plot_func(use_dag = True)
def marginals_to_csv(
    *,
    data: dict,
    list_data: xr.Dataset,
    list_dims: list,
    hlpr: PlotHelper,
    **plot_kwargs,
):
    ''' another brutal workaround to store data into a csv. This probably has a more elegant solution in dantro/yaml, which I could not find.

        Pretty hardcoded to store an xr.Dataset produced by the model_plots.data_ops.marginals_from_ds() function.
    '''
    dfs = []
    for i, dat in enumerate(list_data):
        for parameter in dat.coords["parameter"].values:
            param_ds = dat.sel(parameter=parameter)
            df_param = {}
            # one x column per path
            df_param[f"{list_dims[i]}_{parameter}"] = param_ds["x"].values

            # one y column per (path, method)
            y_vals = param_ds["y"].values
            kde_vals = param_ds["kde"].values
            df_param[f"{list_dims[i]}_{parameter}_marginals"] = y_vals
            df_param[f"{list_dims[i]}_{parameter}_kde"] = kde_vals

            dfs.append(pd.DataFrame(df_param))
    df_wide = pd.concat(dfs, axis=1)
    df_wide.to_csv(hlpr.out_path[:-4] + ".csv", index=False)