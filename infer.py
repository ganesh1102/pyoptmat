# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.functional as grad_F

# file handling
import sys
import os.path
sys.path.append('../../../..')
sys.path.append('..')

# for n-dimensional arrays
import numpy as np
import numpy.random as ra

# importing datasets
import xarray as xr

# pyoptmat: material response
from pyoptmat import optimize, experiments
from tqdm import tqdm

# Bayesian statistical analysis
import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim

# graphing the results
import matplotlib.pyplot as plt

# some warnings happen because of PyOptMat structure, but everything is really fine
import warnings
warnings.filterwarnings("ignore")

# torch set up
torch.set_default_tensor_type(torch.DoubleTensor) # Use double data type for ease of computation
dev = "cpu"
device = torch.device(dev)

# set up and process the data
scale = 0.15
nsamples = 1 # at each strain rate
input_data = xr.open_dataset(os.path.join('..', "opt/anaconda3/lib/python3.9/site-packages/pyoptmat_master/examples/structural-inference/tension/scale-%3.2f.nc" % scale)) # change this according to where your data is located
data, results, cycles, types, control = downsample(experiments.load_results(
    input_data, device = device),
    nsamples, input_data.nrates, input_data.nsamples)

def make(n, eta, s0, weight, bias, weight_last, bias_last, **kwargs):
    """
        Maker with Young's modulus fixed
    """
    return make_model(torch.tensor(0.5), n, eta, s0, weight, bias, weight_last, bias_last, device=device, **kwargs).to(
        device)
  
# fix the priors for analysis  
names = ["n", "eta", "s0", "weights", "bias", "weights_last", "bias_last"]
loc_loc_priors = [torch.tensor(0., device = device),
                     torch.tensor(0., device = device),
                     torch.tensor(0., device = device),
                     torch.zeros((3,3,3), device = device),
                     torch.zeros((3,3), device = device),
                     torch.zeros((1,3), device = device),
                     torch.zeros((1), device = device)]

loc_scale_priors = [torch.tensor(0.1, device = device),
                   torch.tensor(0.1, device = device),
                   torch.tensor(0.1, device = device),
                   torch.full((3,3,3), 0.1, device = device),
                   torch.full((3,3), 0.1, device = device),
                   torch.full((1,3), 0.1, device = device),
                   torch.full((1,), 0.1, device = device)]    

scale_scale_priors = [torch.tensor(0.1, device = device),
                   torch.tensor(0.1, device = device),
                   torch.tensor(0.1, device = device),
                   torch.full((3,3,3), 0.1, device = device),
                   torch.full((3,3), 0.1, device = device),
                   torch.full((1,3), 0.1, device = device),
                   torch.full((1,), 0.1, device = device)]

eps = torch.tensor(1.0e-4, device = device)

model = optimize.HierarchicalStatisticalModel(make, names, loc_loc_priors,
    loc_scale_priors, scale_scale_priors, eps).to(device)

# 4) Get the guide
guide = model.make_guide()

# 5) Setup the optimizer and loss
lr = 1.0e-1
g = 1.0
niter = 20
lrd = g**(1.0 / niter)
num_samples = 1
optimizer = optim.ClippedAdam({"lr": lr, 'lrd': lrd})
ls = pyro.infer.Trace_ELBO(num_particles = num_samples)

# 6) Set up the SVI model
svi = SVI(model, guide, optimizer, loss = ls)

# 7) Infer!
t = tqdm(range(niter), total = niter, desc = "Loss:    ")
loss_hist = []
for i in t:
  loss = svi.step(data, cycles, types, control, results)
  loss_hist.append(loss)
  t.set_description("Loss %3.2e" % loss)
