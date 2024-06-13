import torch
import torch.nn as nn
import numpy as np

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_probs = torch.nn.functional.log_softmax(input, dim=-1)
        target = torch.zeros_like(log_probs).scatter(-1, target.unsqueeze(-1), 1)
        target = target * (1 - self.smoothing) + self.smoothing / input.size(-1)
        loss = (-target * log_probs).mean()
        return loss

class NegatievSumCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_probs = torch.nn.functional.log_softmax(input, dim=-1)
        target = torch.zeros_like(log_probs).scatter(-1, target.unsqueeze(-1), 1)
        target = target * (1 - self.smoothing) + self.smoothing / input.size(-1)
        loss = (-target * log_probs).mean()
        return loss



class DLRLoss(nn.Module):
    def __init__(self,use_softmax=False):
        super(DLRLoss, self).__init__()
        self.use_softmax = use_softmax
    def forward(self, input, target):
        if self.use_softmax:
            x = torch.nn.functional.softmax(input)
        else:
            x = input
        y = target
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        loss_indiv = -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
        return loss_indiv.mean()
