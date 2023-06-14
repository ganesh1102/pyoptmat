#!/usr/bin/env python3

import sys
sys.path.append('../../..')

import numpy as np
import numpy.random as ra

import torch

import xarray as xr

from pyoptmat import flowrules, hardening, models, neuralode, ode, optimize, solvers, temperature
from pyoptmat.temperature import ConstantParameter as CP

from tqdm import tqdm

import tqdm

import warnings

import torch.nn as nn

import itertools

from NODEIntegrator import NODEIntegrator

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)


# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def make_model(model, n_hidden, n_layers, n_inter, **kwargs):
    
    '''
    Args:
      model:                               a neural ODE of type nn.Sequential

    Returns:
      NODEIntegrator:                      a NODEIntegrator class that contains useful helper functions to integrate a neural ODE
    '''

    print(model)

    return NODEIntegrator(model, n_hidden, n_layers, n_inter)


def make_nn_ode(n_hidden, n_layers, n_inter, **kwargs):
    n_in = n_hidden + 2     # Input layer: hidden variables, stress, and strain
    n_out = n_hidden + 1    # Output layer: hidden variables and stress
    n_inter += 1
    
    layers = [nn.Linear(n_in, n_inter), nn.ReLU()] + list(itertools.chain(*[[nn.Linear(n_inter, n_inter), nn.ReLU()] for i in range(n_layers)])) + [nn.Linear(n_inter, n_out), nn.ReLU()]

    # Return a model with `layers` as defined above using `nn.Sequential`
    return nn.Sequential(*layers)


def downsample(rawdata, nkeep, nrates, nsamples):
    """
    Return fewer than the whole number of samples for each strain rate
    """
    ntime = rawdata[0].shape[1]
    return tuple(
        data.reshape(data.shape[:-1] + (nrates, nsamples))[..., :nkeep].reshape(
            data.shape[:-1] + (-1,)
        )
        for data in rawdata
    )