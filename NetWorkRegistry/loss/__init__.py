from NetWorkRegistry.loss.CommLoss import BCELoss, CrossEntropyLoss
from NetWorkRegistry.loss.FocalLoss import FocalLoss
from NetWorkRegistry.loss.iouLoss import IOULoss
from NetWorkRegistry.loss.msssimLoss import MSSSIMLoss

__all__ = ["BCELoss",
           "CrossEntropyLoss",
           "FocalLoss",
           "IOULoss",
           "LovaszHingeLoss",
           "BCEDiceLoss",
           "MSSSIMLoss"]
