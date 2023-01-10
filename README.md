<h1 align='center'>Alopex</h1>

Alopex is a small library that accelerates prototyping of deep learning projects with JAX. Some features are inspired by other awesome libraies. **Currently, Alopex is in alpha version and API will be changed in the future.**


## Installation

1. Setup JAX for your environment. See details in [jax reporitory](https://github.com/google/jax#installation).
2. Install Alopex via pip:
```bash
$ pip install git+https://github.com/h-terao/Alopex
```

## Overview of Modules

### Epoch Loop (epochs.py)

Training/evaluation loop is a common routine to update model parameters and summarize metrics on datasets.

Alopex provided easy-to-use functions that transform step functions into the epoch functions. These epoch functions automatically use multiple GPUs/TPUs and summarize scalars using batch size weighted average.

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

# Summary is a correctly averaged scalars with the specified prefix.
assert summary == {"train/loss": 0, "test/loss": 1}
```

### Loggers (loggers.py, *_logger.py)

Logging is another common routine of deep neural network training.

Alopex provides a simple logger classes such as `ConsoleLogger` and `DiskLogger`. The important point is that we can concat multiple loggers using `LoggerCollection` and log summary by calling `log_summary` once.

Currently, the following loggers are implemented:
- LoggerCollection: Treat multiple loggers.
- ConsoleLogger: Print log on the console.
- DiskLogger: Dump json formatted log in the disk.
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

Statistic transformation creates functions that computes statistics of functions.

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
```

Note that only `count_params` directly count parameters from PyTrees, does not transform any functions.

### Harvest Transformation (harvest.py)

Because JAX employs the functional programming style, it is very difficult to collect and change intermediate variables. Such functions are useful for debugging or collecting variables where the complex position. Alopex provides the harvest transformations that create function to collect and change intermediate variables in the specified function. This feature is a reimplementation of [Oryx](https://github.com/jax-ml/oryx).

Example:
```python
def fun(x, y):
    x = sow(2 * x, tag="tracked", name="x")
    return x + y

assert fun(1, 2) == 4
assert plant(fun, tag="tracked")({"x": 10}, 1, 2) == 12  # `plant` changes intermediate variables.
assert reap(fun, tag="tracked")(1, 2) == {"x": 2}  # `reap` collects intermediate variables.
```


### Registry (registry.py)

Registry is widely used object that holds registered objects and easy to access them without considering where the object is implemented.
