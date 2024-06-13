import torch
import torch.nn as nn
import torch.nn.functional as F


class ComModel(nn.Module):
    def __init__(self, encoder,classifier):
        super(ComModel, self).__init__()
        self.enc = encoder
        self.cls = classifier

    def forward(self, x):
        out = self.enc(x)
        out = self.cls(out)
        return out
