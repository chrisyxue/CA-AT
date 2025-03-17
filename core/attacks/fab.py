import numpy as np
import torch
import torch.nn as nn

from torchattacks import FAB



class FABAttack():
    def __init__(self, model, eps=8/255, steps=10, multi_targeted=False):
        self.attack = FAB(model, eps=eps, steps=10, multi_targeted=False)
    
    def perturb(self, x, y=None):
        x_adv = self.attack(x, y)
        r_adv = x_adv - x
        return x_adv, r_adv
       