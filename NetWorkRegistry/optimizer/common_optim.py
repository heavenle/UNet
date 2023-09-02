import torch.optim as optim
from utils.Registry import OPTIMIZER


@OPTIMIZER.registry()
def sgd_optim(params, **kwargs):
    return optim.SGD(params, **kwargs)

@OPTIMIZER.registry()
def adam_optim(params, **kwargs):
    return optim.Adam(params, **kwargs)
