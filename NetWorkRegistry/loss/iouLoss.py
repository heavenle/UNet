import torch
from utils.Registry import LOSS


def _iou(pred, target, size_average=True):
    if len(target.shape) == 3:
        target = target.view(target.shape[0], 1, target.shape[1], target.shape[2])
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


@LOSS.registry()
class IOULoss(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOULoss, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)
