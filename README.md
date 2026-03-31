# Neural parameter calibration of ODEs and SDEs
### Fabian Raisch, Timo Germann

---

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)


This project estimates parameters of electrical RC networks to some synthetic and real data using neural parameter estimation. This repository contains all the code used in the publication aswell as some frame to implement your own ODE models for which parameters can be estimated. The source of the framework - everything except \*/RC_model/ folders - was, with slight changes, written and published by Th. Gaskin and used extensively in his work. Feel free to pay the original [NeuralABM](https://github.com/ThGaskin/NeuralABM) a visit and take a look at his work and his models.

As a simulation framework, this code uses the [utopya package](https://docs.utopia-project.org/html/index.html). The following README, written by Th. Gaskin and slightly edited by us (for more personal comments, please refer to `models/RC_model/README.md`), shows some use cases of the code. A complete guide on utopya can be found [here](https://docs.utopia-project.org/html/getting_started/tutorial.html#tutorial):


### Contents of this README
* [How to install](#installation)
* [Tutorial](#tutorial)
  * [How to run a model](#how-to-run-a-model)
  * [Parameter sweeps](#parameter-sweeps)
  * [Evaluation and plotting](#evaluation-and-plotting)
  * [Adjusting the neural net configuration](#adjusting-the-neural-net-configuration)
  * [Training settings](#training-settings)
  * [Changing the loss function](#changing-the-loss-function)
  * [Loading data](#loading-data)
* [Models overview](#models-overview)
* [Building your own model](#building-your-own-model)

---
# Installation
> [!WARNING]
> utopya is currently only fully supported on Unix systems (macOS and Ubuntu).
> Be aware that utopya for Windows is currently work in progress and might raise issues.

#### 1. Clone this repository
Clone this repository using a link obtained from 'Code' button (for non-developers, use HTTPS):

```commandline
git clone <GIT-CLONE-URL>
```

#### 2. Install requirements
Make sure, you have access to the conda environment manager installed. If you don't know what conda is or how to install it, please refer to their ['Getting Started with conda'](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) guide.

Move to the repository folder you just cloned, and build the conda environment:
```commandline
conda nev create -f env-[your_os].yml
```
(replace \[your_os\] with either "win" or "linux")
On linux, comment the following line in `cfg/multiverse_project_cfg` out:
```yaml
executable_control:
  prefix: !if-windows-else [[python], ~]
```

when that is done, activate your environment:
```commandline
conda activate neuralabmrc
```

You should now be able to invoke the utopya CLI:
```commandline
utopya --help
```

> [!NOTE] 
> Enabling CUDA for PyTorch requires additional packages, e.g. `torchvision` and `torchaudio`.
> Follow [these](https://pytorch.org/get-started/locally/) instructions to enable GPU training.
> For Apple Silicon, follow [these](https://PyTorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
> installation instructions. Note that GPU acceleration for Apple Silicon is still work in progress and many functions have not
> yet been implemented.

#### 3. Register the project and all models with utopya

In the project directory (i.e. this one), register the entire project and all its models using the following command:
```commandline
utopya projects register . --with-models
```
You should get a positive response from the utopya CLI and your project should appear in the project list when calling:
```commandline
utopya projects ls
```
Done! 🎉

> [!IMPORTANT]
> Any changes to the project info file need to be communicated to utopya by calling the registration command anew.
> You will then have to additionally pass the `````--exists-action overwrite````` flag, because a project of that name already exists.
> See ```utopya projects register --help``` for more information.



---
# Tutorial 
> [!TIP]
> At any stage and for any command, you can use the `--help` flag to show a description of the command, syntax details, and valid arguments, e.g.
> ```commandline
> utopya eval --help
> ```

## How to run a model
Now you have set up the repository, let's run a model. We'll use the `RC` model as an example. Running a model is a simple command:
```commandline
utopya run RC_model
```
You can call 
```commandline
utopya models ls
```
to see a full list of all the registered models. Replace `RC_model` with any of the registered model names to run that model
instead.

For all models, this command will generate some synthetic data, train the neural net to calibrate the model equations on it, and generate a series of plots in the 
`utopya_output` directory, located by default in your home directory (but this can be [changed](#changing-the-output-directory)). Once everything is done, you should see an output like this in your terminal:

```commandline
SUCCESS  logging           Performed plots from 5 plot configurations in 37.5s.

SUCCESS  logging           All done.
```

Navigate to your `utopya_output` directory and open the `RC_model` folder. In it you should see a time-stamped folder
containing a `config`, a `data`, and an `eval` folder. One of the most important benefits of using utopya is that it automatically
stores data, plots, and all the configuration files used to generate them in a unique folder, and outputs are never overwritten. This makes reproducing
and repeating runs easy, and keeps all the data organised. We will shortly see how you can easily re-evaluate the data 
from a given run without having to re-run the simulation.

This directory structure already hints at the three basic steps that are executed during a model run:

- Combine different configurations, prepare the simulation run(s) and start them.
- Store the data
- Read in the data and automatically evaluate it by calling plot functions.

Open the `eval` folder — in it there will be a further time-stamped folder. Every time you evaluate a simulation, a new folder is created. This way, no evaluation result is ever overwritten. In the `eval/YYMMDD-hhmmss` folder, you should find five plots. Take a look at `model_pred.pdf`, which should look something like this:

<img width="1321" height="708" alt="image" src="https://github.com/user-attachments/assets/266da1f6-ea42-4d50-9fbe-efe37bae9437" />



You can see the true data in blue together with the neural net predictions in black.
The results aren't great; you will also notice from the `loss.pdf` plot that the training loss has barely decreased. Why? Well, 
take a look at the `RC_model_cfg.yml` file. This file holds all the default parameters for the model run. Scroll down to the `Training` entry: you will notice the `lookback` is set to 1. This means that the neural network performs a gradient descent step every time it has reproduced a single frame of the time series. Further above, you will notice that the synthetic dataset used to train the model has a length of `num_steps: 1152` (with a dt of 900s = 15min that accounts for 12 days). For these thermal dynamics, let's see if letting the neural network see one day of data for each gradient descent step would improve things. You could change the lookback in the `RC_model_cfg.yml` file directly, but actually this is not recommended: this file holds all the default values the model will fall back on, should something go wrong. Instead create a new `run.yml` file, somewhere on your computer, and copy the following entries into it:

```yaml
parameter_space:
  num_epochs: 30
  RC_model:
    NeuralNet:
      lookback: 96
```
We are now using a lookback of 96, i.e. the length of the time series, and are also training the model for a little bit longer (30 epochs instead of the default 10). Now, run the model again and pass the path to this file to the model:

```commandline
utopya run RC_model path/to/run.yml
```
Here, we are *only* updating those entries of the base configuration which are also given in the `run.yml` file; the remaining ones are taken from the default configuration file. The results in the output folder should look something like this:

<img width="1334" height="705" alt="image" src="https://github.com/user-attachments/assets/457fc8e5-42b7-4392-bc94-8912a6718cce" />

Much better, though we can still improve this result. Right now, we are only training our neural network from a single initialisation, and letting it find one of the possible parameters that fit the problem. This doesn't give us an accurate representation of the parameter space. What we really need to be doing is training it multiple times, in parallel, from different initialisations, so that it can see more of the parameter space. This is what we will do in the next section.

> [!TIP]
> #### Changing the output directory
> If you wish to save the model output to a different directory, add the following entry to your run configuration:
> ```yaml
> paths:
>   out_dir: ... # path/to/dir
> ```
> or run the model with 
> ```commandline
> utopya run <model_name> -p paths.out_dir path/to/out_dir
> ```

## Parameter sweeps

Take a look at the `models/RC_model/cfgs` folder. In it you will find lots of subfolders, each containing a pair of `run.yml` and `eval.yml` files. These are called *configuration sets*: pre-fabricated run files and corresponding evaluation configurations. Try running the following command:

```commandline
utopya run SIR --cs example_estimation
```
The `--cs` ('configuration set') command tells utopya to use the `run.yml` and later the `eval.yml` file for the plotting routine (we will get to the plots [a little later on](#evaluation-and-plotting)). In the `run.yml` file, take note of the following entries:

```yaml
perform_sweep: True
parameter_space:
  seed: !sweep
    default: 1
    range: [8]
```
The `seed` entry controls the random initialisation of the neural network, and we are 'sweeping' over 8 different initialisations (`range: [8]`) and training the model on the same dataset each time! The `perform_sweep` entry tells the model to run the sweep – set it to `False` to just perform a single run again. The `seed` would then be set to its `default` value, in this case 1. utopya will automatically parallelise the runs over as many cores as your computer makes available (you can [change](#adjusting-the-parallelisation) how many workers it can use). A single run is called a 'universe' run, a sweep run over many 'universes' is called a 'multiverse' run.

Once the run is complete, the plot output should look like this:

<img width="1325" height="711" alt="image" src="https://github.com/user-attachments/assets/f35d7703-7b22-42cf-9469-b97d50550724" />



A lot better! You can see that the predicted temperatures are closer to the true data. The folder also contains the marginal densities on the parameters we are estimating:

<img width="1434" height="419" alt="image" src="https://github.com/user-attachments/assets/ea60113f-2eb5-4fd0-ba89-be882c1efd13" />


These too look good: we obtain a capacity of about 7.420MF, a resistance of about 5.28mOhm and an effective window size of about 7.87qm – these are very similar to the values of 7.452MF, 5.29mOhm and 7.89qm used to generate the synthetic data.

> [!TIP]
> You can also configure sweeps by adding a `--run-mode sweep` or `--run-mode single` flag to the command in the CLI:
> ```commandline
> utopya run RC_model --run-mode sweep`
> ```
> This will overwrite the settings in the configuration file. In general, paths to `run.yml` files will overwrite the default entries, and CLI flags will overwrite the
> entries in the config file. You can also change parameters right from the CLI:
> ```commandline
> utopya run RC_model --pp num_epochs=30
> ```
> See [here](https://docs.utopia-project.org/html/usage/run/config.html) for details. 

In your output folder you will also find the following plot:

<img width="1310" height="426" alt="image" src="https://github.com/user-attachments/assets/e5c51048-f81d-40eb-92f0-26be9241d59f" />


Each line represents a trajectory taken by the neural net during training; as you can see, we are training the net multiple times in parallel, each time initialising the neural network at a different value of the initial distribution – see [the corresponding section](#specifying-the-prior-on-the-output) on how to adjust this distribution. The colour of each line repressents the training loss at that sample.
The number of initialisations is controlled by the `seed` entry of the run config.

> [!TIP]
> As an exercise, play around with the `seed.range` argument of the `run.yml` config. How does the quality of the time series prediction and marginal densities change as you increase or decrease the number of runs?

### Sweep configurations and multiple sweeps
You can sweep over as many parameters and entries as you like; any key in the run configuration can be swept over. Any sweep entry must take the following form:
```yaml
parameter: !sweep
   default: 0
   values: [1, 2, 3, 4]
```
Any configuration file must be compatible with *both* a multiverse ('sweep') and a universe ('single') run. The `default` entry is used whenever a universe run is performed, 
the `values` entry used for the sweep. Instead of specifying a list of `values`, you can also provide a `range`, a `linspace`, or a `logspace`:
```yaml
parameter: !sweep
   default: default_value
   range: [1, 4] # passed to python range()
                 # Other ways to specify sweep values:
                 #   values: [1,2,3,4]  # taken as they are
                 #   range: [1, 4]      # passed to python range()
                 #   linspace: [1,4,4]  # passed to np.linspace
                 #   logspace: [-5, -2, 7]  # 7 log-spaced values in [10^-5, 10^-2], passed to np.logspace
```

Once you have set up your sweep configuration file, enable a multiverse run either by setting `perform_sweep: True` to the top-level of the file, or by passing `--run-mode sweep` to the CLI command when you run your model. Without one of these, the model will be run as a universe run.

There is no limit to how many parameters you can sweep over. Take a look, for instance, at the `models/RC_model/cfgs/paper_estimation_72d/run.yml` file. Here, we are sweeping over the type of the estimation method (`NeuralNet.type`) as well as the building we are estimating (`Data.load_from_dir.path`) and the `seed`. Sweeping over more parameters takes longer, of course, since the volume of parameters increases exponentially.

> [!TIP]
> Read the full guide on running parameter sweeps [here](https://docs.utopia-project.org/html/getting_started/tutorial.html#parameter-sweeps).

### Coupled sweeps
If you want to sweep over one parameter but vary some others along with it, you can perform a [coupled sweep](https://docs.utopia-project.org/html/about/features.html?highlight=target_name#id31):
```yaml
param1: !sweep
  default: 1
  values: [1, 2, 3, 4]
param2: !coupled-sweep
  default: foo
  values: [bar, baz, foo, fab]
  target_name: param1
```
Here, `param2` is being varied along `param1` – the dimension of the parameter space remains 1. You can couple as many parameters to sweep parameters as you like.

### Adjusting the parallelisation
When running a sweep, you will see the following logging entry in your terminal:
```commandline
PROGRESS logging           Initializing WorkerManager ...
NOTE     logging             Number of available CPUs:  8
NOTE     logging             Number of workers:         8
NOTE     logging             Non-zero exit handling:    raise
PROGRESS logging           Initialized WorkerManager.
```

As you can see, here utopya is using 8 CPU cores as individual workers to run universes in parallel. If you wish to adjust this, e.g. to reduce the load on the CPU, you can adjust the `worker_manager` settings in your configuration file:

```yaml
worker_manager:
  num_workers: 4
```

## YAML configuration files and changing the parameters
As you have seen, there are multiple configuration layers that are recursively updated: at the bottom, there are default configuration entries for each model, stored in `<model_name>_cfg.yml`. These are default values that will, broadly speaking, be useful in most situations. For this reason, it is best to not change them when performaing 
a specific run. The default configuration file should include *all* the defaults used for a model, but you wouldn't want to have to copy-paste *all* of them into a new file if you only want to change a few. For this purpose there are *run-specific* configuration files, which you can pass to the model CLI via 
```commandline
utopya run <model> path/to/run.yml
```
You can pass a relative or an absolute path, it's up to you. Entries in these files will overwrite the default values. Remember that you only need to provide those entries of the default config you wish to update! Finally, you can also change parameters directly by passing a `--pp` flag from the CLI:
```commandline
utopya run <model> --pp num_epochs=100 --pp entry.key=2
```

Note that, when using the CLI, you can set sublevel entries of outer scopes by connecting them with a `.`: `key.subkey.subsubkey`. YAML offers a wide range of functionality within the configuration file. Take a look e.g. at the [learnXinYminutes](https://learnxinyminutes.com/docs/yaml/) YAMl tutorial for an overview – but since it is an intuitive and humand-readable configuration language, most things should seem very familiar to you already.

> [!IMPORTANT]
> YAML is sensitive to indentation levels! In utopya, nearly every option can be set through a configuration parameter. With these, it is important to take care of the correct indentation level. If you place a parameter at the wrong location, it will often be ignored, sometimes even without warning! A common mistake at the beginning is to place model specific parameters outside of the <model_name> scope:
> ```yaml
> parameter_space:
>   SIR:
>     model_parameter: 1   # Parameters in this scope are passed to the model!
> ```

In general, every aspect of running, evaluation, and configuring models is controllable from the configuration file. Take a look at the [documentation entry](https://docs.utopia-project.org/html/ref/mv_base_cfg.html?highlight=worker%20manager#utopya-multiverse-base-configuration) for a full overview of the keys and controls at your disposal. 

### Automatic parameter validation
Take a look at, for example, the `models/RC_model/RC_model_cfg.yml` file. You will notice lots of little `!is-positive` or `!is-positive-or-zero` flags. These are so-called *validation flags*, and can only be used in the default configuration. They are optional, but their function is to make sure you do not pass invalid parameters to the model (e.g. negative values where only positive ones are allowed), and to catch such errors before the model is run. Running a model with invalid parameters can sometimes lead to cryptic error messages or are even not caught at all, leading to unpredictable behaviour which can be a nightmare to debug. For this reason, you can add these validation flags to the default configuration, along with possible values, ranges, or datatypes for each parameter.

> [!TIP]
> See the [full tutorial entry](https://docs.utopia-project.org/html/usage/run/config-validation.html) for a guide on how to use these. They are useful if you wish to implement your own model.

### Full model configuration
Inside our `utopya_output/RC_model` output folder, take a look at the `config` folder. You will see a whole bunch of configuration files. Every single level of the configuration hierarchy is backed up to this folder, allowing you to always reconstruct which parameters you used to run a model. A couple of useful pointers:

- the `model_cfg.yml` file contains the default configuration
- the `run_cfg.yml` is the run configuration
- the `update_cfg.yml` contains any additional parameters you passed from the CLI
- the `meta_cfg.yml` is the combination of all three, plus all the other defaults (many provided by utopya itself) used to run the model. This file will probably seem very large and overwhelming, and you don't really need to worry about it. However, when in doubt, you can refer to it to check where in your custom configuration you need to place certain keys.

> [!TIP]
> Almost every aspect of running, evaluation, and configuring models is controllable from the configuration file. Take a look at the [documentation entry](https://docs.utopia-project.org/html/ref/mv_base_cfg.html?highlight=worker%20manager#utopya-multiverse-base-configuration) for a full overview of the keys and controls at your disposal.

## Evaluation and plotting 

As you saw, calling 
```commandline
utopya run <model_name>
```
performs a series of tasks:

1. It collects all the configuration files, parameters passed, backs up the files, validates parameters, and prepares sweep runs (if configured)
1. It passes the parameters to the model (or models, if running a sweep)
1. It then collects and bundles the output data and stores it
1. Finally, it loads all the data into a so-called `DataManager` and plots the files.

Running a simulation and plotting the data are seperate steps that can be run indepedently of one another. For instance, if you call
```commandline
utopya run <model_name> --no-eval
```
the evaluation step will be skipped. A common use case however will be re-evaluating a model run you have already performed. This can easily be done by running the command
```commandline
utopya eval <model_name>
```
This will re-evaluate the *last* simulation run that was performed. If you wish to evaluate a different run, simply pass the path to that folder in the CLI:
```commandline
utopya eval <model_name> path/to/folder
```
Calling this will use all the plots given in the *default plot configuration file* `<model_name>_plots.yml`. This is the default behaviour; you can pass a different plot configuration using the `--plots-cfg` flat in the CLI:

```commandline
utopya eval <model_name> --plots-cfg path/to/config.yml
```
Take a look at the `RC_model_plots.yml` file: you will see a list of entries, each corresponding to one plot. In each of the configuration folders, you will notice an `eval.yml` file. These are plot configurations used for these specific configuration sets; thus, all the configuration set `--cs` flag is is a shorthand for the command

```commandline
utopya run <model_name> path/to/run.yml --plots-cfg path/to/eval.yml
```

Many of these plots are based on a *base plot*: these are default plots given in the `RC_model_base_plots.yml` file and which are available throughout the model, i.e. to any other plot configuration. This is handy, since you may wish to share plots throughout the model and not want to have to copy the configuration each time. Take a look at the `RC_model_base_plots.yml` file, and scroll all the way down to the `loss` baseplot:

```yaml
loss:
  based_on:
    - .creator.universe
    - .plot.facet_grid.line
  select:
    data: loss
```
This function plots the training loss for each batch, and is available throughout the model. Let's go through it line by line: the `based_on` argument tells the `PlotManager` which configurations to use as the base. Remember that in utopya, a single run is called a `universe`, and that sweeping over multiple parameters creates multiple universes, or `multiverses`. The two plot creators to use are thus the `.creator.universe` and the `.creator.multiverse`. The universe creator creates plots for each individual universe, whereas the multiverse creator creates plots for the multiverse. The `.plot.facet_grid.line` is the plot function to use to plot a line. Finally, the `select` key tells the `PlotManager` which data to plot. It's that simple. Everything else shown in the configuration entry is just styling, which you can also control right from the configuration (and this backed up and reconstructible later on). If you now wish to use this function in your model evaluation, create an `eval.yml` and simply add
```yaml
loss:
  based_on: loss  # This is the 'loss' plot from the base configuration
```

> [!TIP]
> Read the [full tutorial entry](https://docs.utopia-project.org/html/usage/eval/plotting/index.html) on plotting before continuing to the next steps.

The advantage of configuration-based plotting is twofold: for one, it once again means the configuration files are stored alongside plots, meaning any given plot can be quickly recreated, and you will always be able to understand what you did to create a specific plot long after you first made it. This is invaluable for scientific research, where workflows often involve a lot of experimenting and playing around with numerical settings, and you may wish to return to a previous configuration weeks or months later. The other advantage is that utopya supports data transformation right from the configuration file: this means that data analysis and data plotting are kept seperate, and you can always reconstruct the analysis steps later. 

> [!TIP]
> Read the [full tutorial entry](https://docs.utopia-project.org/html/usage/eval/dag/index.html) on configuration-based analysis using a DAG (directed acyclic graph). utopya uses [xarray](https://docs.xarray.dev/en/stable/) for data handling and transformation.

## Adjusting the neural net configuration
### Adjusting the architecture
You can vary the size of the neural net and the activation functions
right from the config. The size of the input layer is inferred from
the data passed to it, and the size of the output layer is
determined by the number of parameters you wish to learn — all the hidden layers
can be determined by the user. The net is configured from the ``NeuralNet`` key of the
config:

```yaml
NeuralNet:
  num_layers: 2
  nodes_per_layer:
    default: 140
    layer_specific:
  activation_funcs:
    default: sigmoid
    layer_specific:
      -1: abs
  biases:
    default: [0, 1]
  learning_rate: 0.0029
```
``num_layers`` sets the number of hidden layers. ``nodes_per_layer``, ``activation_funcs``, and ``biases`` are
dictionaries controlling the structure of the hidden layers. Each requires a ``default`` key
giving the default value, applied to all layers. An optional ``layer_specific`` entry
controls any deviations from the default on specific layers; in the above example,
all layers have 140 nodes by default, use a sigmoid activation function, and have a bias
which is initialised uniformly at random on [0, 1]. Layer-specific settings are then provided.
You can also set the bias initialisation interval to `default`: this will initialise the bias using the [PyTorch default](https://github.com/pytorch/pytorch/blob/9a575e77ca8a0be7a3f3625c4dfdc6321d2a0c2d/torch/nn/modules/linear.py#L72)
Xavier uniform distribution.

### Setting the activation functions
Any [PyTorch activation function](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
is supported, such as ``relu``, ``linear``, ``tanh``, ``sigmoid``, etc. Some activation functions take arguments and
keyword arguments; these can be provided like this:

```yaml
NeuralNet:
  num_layers: 2
  nodes_per_layer: 140
  activation_funcs:
    default:
      name: Hardtanh
      args:
        - -2 # min_value
        - +2 # max_value
      kwargs:
        # any kwargs here ...
```

### Specifying the prior on the output
For many applications, you will want control over the prior distribution of the parameters. To this
end, you can add a `prior` entry that gives a distribution over the parameters you wish to learn:
```yaml
NeuralNet:
  prior:
    distribution: uniform
    parameters:
      lower: 0
      upper: 1
```
This will train the neural network to initially output values uniformly within [0, 1], for all
parameters you wish to learn. If you want individual parameters to have their own priors, you can do so by passing a
list as the argument to `prior`. For instance, assume you wish to learn 2 parameters; the configuration entry then could
be:
```yaml
NeuralNet:
  prior:
    - distribution: normal
      parameters:
        mean: 0.5
        std: 0.1
    - distribution: uniform
      parameters:
        lower: 0
        upper: 5
```

## Training settings
You can modify the training settings, such as the batch size or the training device, from the
`Training` entry of the config:

```yaml
Training:
  batch_size: 1
  loss_function:
    name: MSELoss
  true_parameters:
    param4: 0.5
  device: cpu
  num_threads: ~
```
The `to_learn` entry lists the parameters you wish to learn. The naming of the parameters can be derived from the `models/RC_model/Physicals.py` file.

The `device` entry sets the training device. The default here is the `cpu`; you can set it to any
supported PyTorch training device; for instance, set it to `cuda` to use the GPU for training. Make sure your platform
is configured to support the selected device.
On Apple Silicon, set the device to `mps` to enable GPU training, provided you have followed the corresponding
installation instructions (see above). Note that PyTorch for Apple Silicon is still work in progress at this stage,
and some functions have not yet been fully implemented.

`utopya` automatically parallelises multiple runs; the number of CPU cores available to do this
can be specified under `worker_managers/num_workers` on the root-level configuration (i.e. on the same level as
`parameter_space`). The `Training/num_threads` entry controls the number of threads *per model run* to be used during training.
If you thus set `num_workers` to 4 and `num_threads` to 3, you will in total be able to use 12 threads.

### Changing the loss function
You can set the ``loss_function/name`` argument to point to any supported
[Pytorch loss function](https://pytorch.org/docs/stable/nn.html#loss-functions). Additional arguments to
the loss function can be passed via an optional ``args`` and ``kwargs`` entry:

```yaml
loss_function:
  name: CTCLoss
  args:
    - 1  # blank
    - 'sum' # reduction to use
```
### Loading data
By default, new synthetic data is produced during every run, but this is often not desired. For one, when performing a multiverse run, we want each universe to calibrate the same data. For another, we will want to be able to load in real data. The specific loading syntax for each model is slightly (unifying this is still WIP), but the general concept is always the same: to your run config, add the following entry (here using SIR as an example):

```yaml
SIR:
  Data:
    load_from_dir: load_from_dir: data/RC_model/Varennes_paper_version.csv
    csv_keys: [thermalZone.TAir, weaBus.TDryBul, totalHeatingPower.y, weaDat.weaBus.HGloHor]
```
This will load in the training data from the given `csv` file and use it across universes. See the model-specific README files to see the syntax for each model. Data is stored in the `data/` folder.

## Models overview
This repository contains the following models:
- [**RC_model**](models/RC_model/README.md): Multiple RC surrogates for thermal gray-box modelling of buildings. 
  
See the model-specific README files for a guide to each model. The README files are located in the respective `<model_name>` folders.

## Building your own model
If you are ready to build your own `NeuralABM` model, there is an easy command you can use to get started:
```commandline
utopya models copy <model_name>
```
This command will duplicate an existing model and rename it to whatever name you give when prompted. You can then successively change an existing model to your own requirements.
