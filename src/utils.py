from copy import deepcopy
import numpy as np
import random
import torch
from torch import nn
import os

from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import LogisticRegression as LogitReg
from sklearn.linear_model import Ridge

import torch
from torch.nn.utils import parameters_to_vector

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn



def get_obj(regression, reduction='mean'):
    if regression:
        return nn.MSELoss(reduction=reduction)
    else:
        return nn.CrossEntropyLoss(reduction=reduction)


def flatten(tensor):
    return torch.cat(list(tensor))


class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
#     __getattr__ = dict.__getitem__
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_device(device=None):
    if device is None:
        return torch.cuda()
    return torch.device(device)



def mkdir(path):
    try: 
        os.mkdir(path) 
    except OSError as error: 
        pass



def get_latest_checkpoint(checkpoint_loc):
    # Is checkpoint_loc a file?
    if os.path.isfile(checkpoint_loc):
        return checkpoint_loc
    
    assert os.path.isdir(checkpoint_loc), 'No such directory exists: {0}'.format(checkpoint_loc)

    # checkpoint_loc is a directory
    for fname in os.listdir(checkpoint_loc):
        if 'final' in fname:
            return os.path.join(checkpoint_loc, fname)

    class NotFoundError(Exception):
        pass
    raise NotFoundError('No final checkpoint found in {0}'.format(checkpoint_loc))


def copy_model(model, grad=False):
    if grad:
        model_deepcopy = deepcopy(model)
    else:
        with torch.no_grad():
            model_deepcopy = deepcopy(model)
    return model_deepcopy
#     if deep:
#     for param, param_deepcopy in zip(model.parameters(), model_deepcopy.parameters()):
#         param_deepcopy = param.detach().clone()
#     return model_deepcopy


def copy_params(model, param_vec):
    params = vec_to_params(model, param_vec)
    with torch.no_grad():
        for param, p in zip(model.parameters(), params):
            param.copy_(p)


def params_to_vec(params):
    return torch.cat([param.contiguous().view(-1) for param in params])


def vec_to_params(model, vec):
    pointer = 0
    res = []
    for param in model.parameters():
        num_param = param.numel()
        res.append(vec[pointer:pointer+num_param].view_as(param).data)
        pointer += num_param
    return res


def grad_means(grads):
    means = []
    for param_list in zip(*grads):
        means.append(torch.mean(torch.stack(param_list), dim=0))
    return means


def apply_grad(model, grad):
    '''
    assign gradient to model(nn.Module) instance. return the norm of gradient
    '''
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
    return


def set_seed(args):
    if args.debug:
        print('Setting numpy seed = {0}, torch seed = {1}'.format(args.np_seed, args.torch_seed))
    np.random.seed(args.np_seed)
    torch.manual_seed(args.torch_seed)
    if not args.torch_seed is None:
#         torch.set_deterministic(True)
        torch.backends.cudnn.deterministic = True
    random.seed(args.np_seed)
