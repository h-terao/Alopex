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
import alopex

def train_step(train_state, batch)
    # usually, update train_state here.
    scalars = {"acc": jnp.mean(batch)}
    return train_state, scalars
    
iterable = numpy.zeros((5, 32, 8))  # 5 steps/epoch, batch_size is 32.
iterable += numpy.arange(5).reshape(-1, 1, 1)

train_state = tuple()  # usually, train_state is a PyTree object that contains states of model, optimizer and others.

train_state, summary = alopex.train_epoch(train_state, iterable, train_step, prefix="train/")
assert summary == {"train/acc": 2.0}  # average of (0,1,2,3,4) reported as scalars.  
```

### Harvest transformation
