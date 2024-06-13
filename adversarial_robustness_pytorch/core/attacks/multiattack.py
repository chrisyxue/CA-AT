import numpy as np

import torch
# from autoattack.autopgd_pt import APGDAttack
import torchattacks

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiAttack():
    """
    Zhiyu:

    This is a warp of attacks aims to generate adversarial examples with different attacks
    Arguments:
        --attacks: A list of names of attacks
        --model: The model to attack
    """
    def __init__(self,model,attacks=None,steps=10):
        if attacks == None:
            self.attacks = ['PGD-CE','APGD-CE','APGD-DLR','TAPGD-DLR','T-FAB']
        
        self.model = model
        self.steps = steps

        # init the list of attacks
        self.create_attk_funcs()

    def create_attk_funcs(self):
        attk_funcs = []
        for att in self.attacks:
            if att == 'PGD-CE':
                att_func = torchattacks.PGD(self.model, eps=8/255)
            elif att == 'APGD-CE':
                att_func = torchattacks.APGD(self.model, norm='Linf', eps=8/255, loss='ce')
            elif att == 'APGD-DLR':
                att_func = torchattacks.APGD(self.model, norm='Linf', eps=8/255, loss='dlr')
            elif att == 'TAPGD-DLR':
                att_func = torchattacks.APGDT(self.model, norm='Linf', eps=8/255, n_classes=10)
            elif att == 'T-FAB':
                att_func = torchattacks.FAB(self.model, norm='Linf', eps=8/255, multi_targeted=True, n_classes=10)
            else:
                raise ValueError(str(att)+ ' is not in the implemented list')
            attk_funcs.append(att_func)
        self.attk_funcs = attk_funcs

    def perturb(self, x, y):
        x_adv_lst = []
        for idx in range(len(self.attacks)):
            att = self.attacks[idx]
            attk_func = self.attk_funcs[idx]
            x_adv = attk_func(x,y)
            x_adv_lst.append(x_adv)
        return x_adv_lst, self.attacks