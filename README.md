# Alopex

Alopex provides common routines and utilities to accelerate the prototyping deep learning projects built on JAX. 


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
import alopex as alx

def train_step(train_state, batch)
    # usually, update train_state here.
    scalars = {"acc": jnp.mean(batch)}
    return train_state, scalars
    
iterable = numpy.zeros((5, 32, 8))  # 5 steps/epoch, batch_size is 32.
iterable += numpy.arange(5).reshape(-1, 1, 1)

train_state = tuple()  # usually, train_state is a PyTree object that contains states of model, optimizer and others.

train_state, summary = alx.train_epoch(train_state, iterable, train_step, prefix="train/")
assert summary == {"train/acc": 2.0}  # average of (0,1,2,3,4) reported as scalars.  
```

### Harvest transformation

Harvest transformation collects or changes the intermediate values without changing input and output of functions. 
The target values are marked by `sow`, and then transform entire function using `plant` or `reap`. `plant` changes the marked values via dict and `reap` collects all marked values as a dict. If you do not transform the entire function, `sow` performs a simple identity function. Harvest transformation is useful to debug functios or take variables that are too difficult to get (such as gradients of not-leaf variables).

```python
import alopex as alx

def func(x):
    y = alx.sow(2 * x, tag="test", name="y")
    z = 4 * z
    return z
    
assert func(1) == 8  # sow performs as an identity function because func is not transformed by any harvest transformations.
assert alx.plant(func, tag="test")({"y": 10}, 1) == 40  # y is replaced with 10.
assert alx.reap(func, tag="test")(1) == {"y": 2}  # take intermediate variables.
assert alx.reap(func, tag="test")(1) == jax.jit(alx.reap(func, tag="test"))(1)  # NOTE: harvest transformations are jit-able.
```
