from .scheduler import  ConstantValueScheduler, LinearStepScheduler , LinearEpochScheduler, ExponentialScheduler, StepwiseExponentialScheduler,TransformerScheduler, LinearWarmUpAndExpDecayScheduler
import torch
from kws.bin.utils import save_checkpoint, load_checkpoint
from kws.bin.executor import Executor


BuildOptimizer = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD
}

BuildScheduler = {
    'constant': ConstantValueScheduler,
    'step-linear': LinearStepScheduler,
    'epoch-linear': LinearEpochScheduler,
    'exp': ExponentialScheduler,
    'step-exp': StepwiseExponentialScheduler,
    'transformer': TransformerScheduler,
    'linear-warmup-exp-decay': LinearWarmUpAndExpDecayScheduler
}

