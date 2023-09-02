import torch
import torch.nn as nn
from utils.Registry import LOSS


@LOSS.registry()
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # if isinstance(alpha, (float, int)):
        #     self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        if len(target.shape) == 4:
            target = target.view(input.size(0), target.size(1), -1)  # N,C,H,W => N,C,H*W
            target = target.transpose(1, 2)  # N,C,H*W => N,H*W,C
            target = target.contiguous().view(-1, target.size(2))  # N,H*W,C => N*H*W,C
        elif len(target.shape) == 3:
            target = target.contiguous().view(-1, 1).to(torch.float32)
        else:
            print("target shape error")
            exit(0)
        class_pt = input * target
        class_neg_pt = (1 - input) * target

        class_pt = class_pt.clamp(min=0.0001, max=1.0)
        class_neg_pt = class_neg_pt.clamp(min=0.0001, max=1.0)
        self.alpha = self.alpha.to(input.device)
        # self.alpha = self.alpha.reshape(1, input.size[1])
        loss = -1 * self.alpha * torch.pow(class_neg_pt, self.gamma) * torch.log(class_pt)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
