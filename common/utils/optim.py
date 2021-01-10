#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch

"""
Optimizer functions are separately defined here to provide a common interface to access any optimizer in the functions/train file
"""

def optim_SGD(model=None, initial_lr=0.1, momentum=0.9, weight_decay=0.05, **kwargs):

    # return an SGD optimizer object
    return torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)

def optim_SGD_Nesterov(model=None, initial_lr=0.1, momentum=0.9, weight_decay=0.05, **kwargs):

    # return an SGD optimizer object
    return torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

def optim_AdamW(model=None, initial_lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05, **kwargs):

    # return AdamW object
    return torch.optim.AdamW(model.parameters(), lr=initial_lr, betas=betas, eps=eps, weight_decay=weight_decay)
