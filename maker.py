import sys

sys.path.append("../../..")

import numpy as np
import numpy.random as ra

import xarray as xr

import torch
from pyoptmat import models, flowrules, hardening, optimize
from pyoptmat.temperature import ConstantParameter as CP

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Define true values

E_true = 150000.0
weights_true = torch.rand((3,3,3))
offsets_true = torch.rand((3,3,3))
weight_last_true = torch.rand((1,3))
bias_last_true = torch.rand((1))
n_true = 7.0
eta_true = 300.0
s0_true = 50.0

# Scale factor used in the model definition
sf = 0.5

device = torch.device("cpu")

def make_model(E, n, eta, s0, weights, bias, weight_last, bias_last, device=device, **kwargs):    
    isotropic = hardeningNeuralNetHardeningModel(weights, bias, weight_last, bias_last)
    kinematic = hardening.NoKinematicHardeningModel()

    flowrule = flowrules.IsoKinViscoplasticity(
        CP(
            n,
            scaling = optimize.bounded_scale_function(
            (torch.tensor(n_true * (1 - sf), device = device),
                torch.tensor(n_true * (1 + sf), device = device),)
        ),
        ),
        CP(
            eta,
            scaling = optimize.bounded_scale_function(
                (torch.tensor(eta_true * (1 - sf), device = device),
                torch.tensor(eta_true * (1 + sf), device = device),)
            )
        ),
        CP(
            s0,
            scaling = optimize.bounded_scale_function(
                (torch.tensor(s0_true * (1 - sf), device = device),
                torch.tensor(s0_true * (1 + sf), device = device),)
            )
        ),
        isotropic,
        kinematic,
    )
    model = models.InelasticModel(
        CP(
            E,
            scaling = optimize.bounded_scale_function(
                (torch.tensor(E_true * (1 - sf), device = device),
                torch.tensor(E_true * (1 + sf), device = device),)
            )
        ),
        flowrule
    )    
    return models.ModelIntegrator(model, **kwargs)
    
def generate_input(erates, emax, ntime):
    """
        Generate the times and strains given strain rates, max strain, and number of time steps.
    """
    strain = torch.repeat_interleave(
        torch.linspace(0, emax, ntime, device = device)[None, :], len(erates), 0
    ).T.to(device)   
    time = strain / erates
    return time, strain

def downsample(rawdata, nkeep, nrates, nsamples):
    """
        Downsample the data to retrun fewer number of samples than initially passed
    """
    ntime = rawdata[0].shape[1]
    return tuple(
        data.reshape(data.shape[:-1] + (nrates, nsamples))[..., :nkeep].reshape(
            data.shape[:-1] + (-1,)
        )
        for data in rawdata
    )
