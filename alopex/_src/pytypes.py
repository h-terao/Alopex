import typing as tp
import chex


TrainState = chex.ArrayTree
Batch = chex.ArrayTree
Summary = tp.Mapping[str, chex.Array]
Scalars = tp.Mapping[str, chex.Array]
Prediction = chex.ArrayTree

TrainFn = tp.Callable[[TrainState, Batch], tp.Tuple[TrainState, Scalars]]
EvalFn = tp.Callable[[TrainState, Batch], Scalars]
PredFn = tp.Callable[[TrainState, Batch], Prediction]
