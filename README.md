<h1 align='center'>Alopex</h1>

Alopex is a small library that accelerates the prototyping of deep learning projects with JAX. **Currently, Alopex is in alpha version, and API will be changed in the future.**


## Installation

1. Install JAX for your environment. See details in [installation section of JAX project](https://github.com/google/jax#installation).
2. Install Alopex via pip:
```bash
$ pip install git+https://github.com/h-terao/Alopex
```

## Overview of Modules

### Epoch Loop (epochs.py)

Training and evaluation loops are common routines for experimenting with deep neural networks.

Alopex provides easy-to-use functions that create epoch functions from your step functions. The epoch functions automatically use multiple GPUs/TPUs and summarize scalars using batch size weighted average.

Example:

```python
def train_step(train_state, batch):
    new_train_state = ...  # update train_state here.
    scalars = {"loss": 0} # dict of metrics to summarize.
    return new_train_state, scalars

def eval_step(train_state, batch):
    scalars = {"loss": 1}
    return scalars  # For evaluation, new_train_state is unnecessary.

# Transform step function into epoch function.
train_fun = train_epoch(train_step, prefix="train/")
eval_fun = eval_epoch(eval_step, prefix="test/")

# Loop loaders until StopIteration is raised.
train_state, summary = train_fun(train_state, train_loader)
summary |= eval_fun(train_state, test_loader)

# Summary is a dict of correctly averaged scalars with the specified prefix.
assert summary == {"train/loss": 0, "test/loss": 1}
```

### Loggers (loggers.py, *_logger.py)

*NOTE: To use CometLogger, install comet-ml.*

Logging is another common routine.

Alopex provides some loggers to achieve logging with a few lines. Because loggers are designed simply, you can quickly implement your loggers.

Currently, Alopex supports the following loggers:
- LoggerCollection: Concat multiple loggers.
- ConsoleLogger: Print log on the console.
- DiskLogger: Dump JSON formatted log in the disk.
- CometLogger: Log values to Comet.ml

Example:
```python
logger = LoggerCollection(
    ConsoleLogger(),
    DiskLogger(save_dir="logs/"),
)

logger.log_hparams(hyperparams)  # pass dict of hyperparams.
logger.log_summary(summary)  # pass summary

lg_state = logger.state_dict()  # create state_dict to restore loggers again.
logger.load_state_dict(lg_state)  # restore loggers using state_dict.
```

### Statistic Transformation (stats.py)

Statistic transformation creates functions that compute statistics of the inner functions. Currently, Alopex provides `flop`, `mac`, `latency`, and `memory_access` to transform functions. In addition, `count_params` is provided to count the number of parameters stored in a PyTree.

Example:
```python
@jax.jit
def add(x, y):
    return x + y

FLOPs = flop(add)(1, 1)
GFLOPs = flop(add, unit="G")(1, 1)
MACs = mac(add)(1, 1)
Latency = latency(add, num_iters=100, warmup_iters=100)(1, 1)  # seconds / forward pass.
MemoryAccess = memory_access(add)(1, 1)

param_size = count_params(variables)  # count number of elements in variables.
```

### Harvest Transformation (harvest.py)

Because JAX employs the functional programming style, collecting and rewriting intermediate variables in functions are difficult. Such functions can create new bugs but are sometimes helpful. Harvest transformations create a function that can collect or rewrite intermediate variables of wrapped functions. This feature is a reimplementation of the harvest transformation of [Oryx](https://github.com/jax-ml/oryx).

The first step of the harvest transformation is tagging intermediate variables using `sow` or `sow_grad.` The `sow` performs as an identity function as usual but collects or rewrites the tagged values if the harvest transformations wrap the outer function. `sow_grad` is similar to `sow,` but it collects gradients of the tagged values. Note that you need to wrap the outer function by `jax.grad` to obtain gradients using `sow_grad.`


Example:
```python
def fun(x, y):
    x = sow(2 * x, tag="tracked", name="x")
    return x + y

assert fun(1, 2) == 4
assert reap(fun, tag="tracked")(1, 2) == {"x": 2}  # `reap` collects intermediate variables.
assert plant(fun, tag="tracked")({"x": 10}, 1, 2) == 12  # `plant` changes intermediate variables.
```

### Plotting (plotting.py)

*NOTE: install matlotlib and seaborn to use this module.*

Visualizing the metric curve is an important step to check whether training works well. `alopex.plot_log_on_disk` draws the curve of your specified metrics.


### Functions (functions.py)

Alopex also provides some useful operations. This module will be extended in the future.
