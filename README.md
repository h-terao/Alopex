# Alopex

A small library to accelerate the development of your JAX projects.


## Installation

Before installing Hyrax, you must setup JAX for your environment. After that, type the following command to install Hyrax.

```bash
$ pip install hyrax
```

## Training and Evaluation Loop

In deep learning, epoch loop is common. Hyrax provides `train_epoch` and `eval_epoch` that perform epoch loop using step functions.

The advantage of Hyrax's epoch functions is that they automatically use `jax.pmap` and train/evaluate your models in parallel. Then, train_state and batches are automatically modificated for `jax.pmap` in Hyrax's epoch functions. Thus, you can debug step function on a single device and run on multiple devices without any modification of step function, train_state and batches.

First, implement step function.
```python
def train_step(train_state, batch):
    # NOTE: usually, train_state is updated by gradients.
    scalars = {"loss": 0, "accuracy": 1.0}
    return train_state, scalars

def eval_step(train_state, batch):
    # eval_step is similar to train_step, but only returns scalars.
    scalars = {"loss": 0, "accuracy": 1.0}
    return scalars
```

Then, pass step functions to corresponding epoch functions.
```python
train_state, summary = hyrax.train_epoch(train_state, iterable, train_fn)
```
where `iterable` is your favorite dataloader. eval_epoch also has similar interface, but only returns `summary`.

## Model Stats

To evaluate the efficiency of functions or models, flops, params and time to compute are common. Hyrax provides `count_flops`, `count_macs`, `count_params`, `time`. They are also useful for debugging.


## Harvest Transformation
Harvest transformation is available to get intermediate values for debugging. This feature is reimplementation of `oryx.core`.

`reap` collects `sow`-ed values. `plant` replaces `sow`-ed values with the specified values.

```python
def f(x):
    y = 2 * x + 1
    y = hyrax.sow(y, tag="v", name="a")
    return 3 * y

assert f(2) == 15
assert hyrax.reap(f, tag="v")(2) == {"a": 5}
assert hyrax.plant(f, tag="v")({"a": 3}, 2) == 9

# jax.jit, jax.vmap, jax.pmap are available.
assert hyrax.reap(f, tag="v")(2) == jax.jit(hyrax.reap(f, tag="v")(2))
```

## Registry
Hyrax provides a simple registry system. This feature can be used as follows:
```python
registry = hyrax.Registry()

# set default value of y=0.
registry.register("add", lambda x, y: x+y, y=0)
assert registry.get("add")(1) == 1
assert registry.get("add")(1, 2) == 3

# decorator.
@registry.register("mul", y=1)
def f(x, y):
    return x * y

assert registry.get("mul")(2) == 2
assert registry.get("mul")(2, 3) == 6
```

Registry is a dict object, so you can use any methods of dict as like:
```python
# Listup all registered keys
assert list(registry) == ["add", "mul"]

# Remove `add` function.
del registry["add"]

for key, f in registry.items():
    print(key, f)
```

## TODO
- Borrow traverse_util from flax to drop flax dependency.
    - Reimplement flatten_dict and unflatten_dict?