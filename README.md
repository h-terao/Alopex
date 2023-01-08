# Alopex

Accelerate prototyping of deep learning projects with JAX


## Installation

1. Setup JAX for your environment. See details in [jax reporitory](https://github.com/google/jax#installation).
2. Install Alopex via pip:
```bash
$ pip install git+https://github.com/h-terao/Alopex
```

## Examples

### Epoch loop

Epoch loop is a common routine of most deep learning experiments. Alopex provides `train_epoch`, `eval_epoch` and `predict_epoch` to support various epoch loops. They automatically replicate/unreplicate train_state, reshape batches and feed the modified objects into the `pmap`-ed step function.

```python
import numpy
import jax.numpy as jnp
import alopex as ap

def train_step(train_state, batch)
    # usually, update train_state here.
    scalars = {"acc": jnp.mean(batch)}
    return train_state, scalars

iterable = numpy.zeros((5, 32, 8))  # 5 steps/epoch, batch_size is 32.
iterable += numpy.arange(5).reshape(-1, 1, 1)

train_state = tuple()  # usually, train_state is a PyTree object that contains states of model, optimizer and others.

train_state, summary = ap.train_epoch(train_state, iterable, train_step, prefix="train/")
assert summary == {"train/acc": 2.0}  # average of (0,1,2,3,4) reported as scalars.
```

### Statistic transformation

Some statistics are useful to evaluate the efficiency of model inference. Alopex provides some functions to compute such major statistics.
`flop`, `mac` and `memory_access` support `unit` arguments, and you can easily obtain KFLOPs, GFLOPs, TFLOPs and so on.

```python
add = lambda x, y: x + y

ap.flop(add)(1, 2)  # FLOPs
ap.flop(add, unit="G")(1,2)  # compute GFLOPs
ap.mac(add)(1,2)  # MACs
ap.latency(add, num_iters=1000)(1,2) # Seconds per forward pass.
ap.memory_access(add)(1,2)  # Momory access cost
```


### Harvest transformation

Harvest transformation collects or changes the intermediate values without changing input and output of functions.
The target values are marked by `sow`, and then transform entire function using `plant` or `reap`. `plant` changes the marked values via dict and `reap` collects all marked values as a dict. If you do not transform the entire function, `sow` performs a simple identity function. Harvest transformation is useful to debug functios or take variables that are too difficult to get (such as gradients of not-leaf variables).

```python
import alopex as ap

def func(x):
    y = ap.sow(2 * x, tag="test", name="y")
    z = 4 * z
    return z

assert func(1) == 8  # sow performs as an identity function because func is not transformed by any harvest transformations.
assert ap.plant(func, tag="test")({"y": 10}, 1) == 40  # y is replaced with 10.
assert ap.reap(func, tag="test")(1) == {"y": 2}  # take intermediate variables.
assert ap.reap(func, tag="test")(1) == jax.jit(ap.reap(func, tag="test"))(1)  # NOTE: harvest transformations are jit-able.
```

### Registry

Registry holds callable objects with default arguments. You can register objects via a `Registry.register` method as follows:

```python
import alopex as ap

REGISTRY = ap.Registry()
REGISTRY.register("add", lambda x, y: x+y, y=0)

@REGISTRY.register("mul", y=1)  # decorator.
@REGISTRY.register("twice", y=2)  # decorator.
def multiply(x, y):
    return x * y

assert REGISTRY.get("add")(1, 2) == 3  # 1+2
assert REGISTRY.get("add")(1) == 1  # 1+0, y=0 is default.
assert REGISTRY.get("mul")(2, 3) == 6  # 2x3
assert REGISTRY.get("mul")(2) == 2  # 2x1
assert REGISTRY.get("twice")(5) == 10  # 5x2
```

Actually, Registry is a Python dict object. Thus, you can use all methods of dict such as `Registry.keys`, `Registry.pop` and `Registry.merge`. Alopex also defines `registry` and `register` that are aliases of the instantiated `Registry` object and its register method.


### Visualization

***This function is not implemented yet.**

Visualization module provides utilities to visualize model information. For example, GradCAM and landscape of loss values will be implemented.
