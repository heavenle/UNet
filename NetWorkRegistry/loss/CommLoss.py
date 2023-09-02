import torch
import torch.nn as nn
from utils.Registry import LOSS


@LOSS.registry()
def BCELoss(**kwargs):
    return nn.BCELoss(**kwargs)


@LOSS.registry()
def CrossEntropyLoss(**kwargs):
    return nn.CrossEntropyLoss(**kwargs)
