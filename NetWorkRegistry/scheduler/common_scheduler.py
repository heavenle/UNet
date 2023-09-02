from torch.optim import lr_scheduler
from utils.Registry import SCHEDULER


@SCHEDULER.registry()
def CosineAnnealingLR(optimizer, **kwargs):
    return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)


@SCHEDULER.registry()
def ReduceLROnPlateau(optimizer, **kwargs):
    return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)


@SCHEDULER.registry()
def MultiStepLR(optimizer, **kwargs):
    return lr_scheduler.MultiStepLR(optimizer, **kwargs)
