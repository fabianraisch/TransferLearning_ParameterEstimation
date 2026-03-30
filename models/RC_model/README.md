# The RC_model surrogates for building estimation

## Model description
The RC_model incorporates different RC model ODEs taken from [Identifying suitable models for the heat dynamics of buildings](https://doi.org/10.1016/j.enbuild.2011.02.005). 


The parameters that can be calibrated are All Rs and Cs as well as the effective window area $A_eff$. The model produces marginals for all parameters aswell as a simulation using the estimated parameters and the residuals in comparison to the source data. 

As source data, an RC circuit can be evolved over time using provided weather data and some arbitrary parameters or external data can be used.

## Model parameters

```yaml
Data:
  model_type: RC #TwoR2C, RC
  synthetic_data:
    initial_conditions:
      T_in: 291
    store_raw_QSolar: true # if the provided solar radiation is stored raw or as input power scaled with A_eff
    effWinArea: 5 #[m2]
    maxHeatingPower: 5000 #[W]
    controller: PControl #PControl | TwoPointControl
    T_min: 290 #[K]
    T_max: 294 #[K]
    num_steps: 3000
    C: 7452000 # first entries taken from  Identifying suitable models for the heat dynamics of buildings: Peder Bacher, Henrik Madsen
    R: 0.00529
    A_eff: 7.89
Training:
 
  mode: predict # predict | generalize | single-input 
  model_type: TwoR2C #TwoR2C, RC
  
  train_range: 0.6         # either define in percent or absolute indices
  val_range: 0.2
  test_range: [6912, 8640]

  slicing_difference: 96
```
The key `mode` determines the way the neural net is treated. `predict` passes a lookback window to the neural net, `single-input` only feeds one single feature to the net. `generalize` uses the provided data to train the neural net to predict different parameters based on the provided data and stores the weights dict after training.
`model_type` determines the RC circuit the parameters are fit for. Which parameters are to be fit is automatically determined from the provided model. The keys `train_range`, `val_range` and `test_range` refer to the subset of the data's time indices to be used for training (parameter estimation), validation (hyperparameter tuning and generalization training) and testing (final test of the estimated parameters). The `slicing_difference` refers to the distance in between two training examples (in indices: e.g. a slicing_difference of 96 with a dt = 15min means, every training example is taken from a new day)



### Loading data
Instead of generating synthetic data, you can also load data from an `.csv` File.

```yaml
Data:
  load_from_dir: path/to/csv
```

the csv keys and effective window area to scale the radiation with need to be passed along, a subset of sequential timestamps can be defined:

```yaml
Data:
  effWinArea: 5 # [m2]
  csv_keys: [T_in, T_out, heatPower, HGloHor]
  subset: 1440
```

### Configuring the neural net
A type and lookback window can be specified for the neural net. `lstm` and `mlp` are standard implementations of torch, `mlp-single` is an alias for running the mlp in single-input mode. `optimizer` uses the optimization algorithm to directly fit the parameters.
```yaml
NeuralNet: 
  type: lstm #lstm, mlp, mlp-single, optimizer
  learning_rate: 0.002
  lookback: 288
```


### Sweeping hyperparameters

using the `model_sweep` config, a sweep over as many hyperparameters as one desires can be run. Its `eval.yml` is written to merge all samples into one joint and compute the marginals from it.

For coupled sweeps, to undermine the duplication-detection utopya implements, Optimizer names and Neural net types can be passed with following numbers, which are ignored by run.py. Similarly, epoch numbers can be passes as strings and slicing differences as floats where the decimals are ignored.

## Evaluation

### Advanced evaluation

While in the original code, the estimated parameters were only presented by means of densities, our RC_model also uses these parameters to evaluate their performance by running a simulation and benchmarking it against a testset. These results are usually shown in a `model_pred.pdf` and give further insight to the quality of the parameters estimated.

### Misuse of sweeping

Conversely to Th. Gaskins implementation did we also use the sweep mechanism to immediately compare different hyperparameter configs (e.g. directly benchmarking a direct optimization using a Genetic Algorithm agains the MLP approach). These extra sweeping dimensions are later separated again in the eval.yml files, using either the `select_and_combine.subspace` keys or converting the dimension into a list and processing the data in parallel downstream. This all can also happen behind the scenes by using the `evaluate_to_bars` transform function. The kwargs `groupings` and `mean_over` refer to the run.yml keys that are swept over and respectively plot the results next to each other or mean over them.
