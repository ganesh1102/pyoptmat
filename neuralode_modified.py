#!/usr/bin/env python3

import itertools

import torch
from torch import nn
from torch.func import vmap, jacrev

import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append('../..')

from pyoptmat import ode, utility

if torch.cuda.is_available():
	dev = 'cuda:0'
else:
	dev = 'cpu'

device = torch.device(dev)



class NeuralODE(nn.Module):
	def __init__(self, model, force, n_hidden, n_layers, n_inter):
		super().__init__()
		self.model = model
		self.force = force
		self.n_in = n_hidden + 2 
		self.n_out = n_hidden + 1
		self.n_inter = n_inter + 1
		
	def forward(self, t, y):
		print('Shape of y in `neuralode_modified`: ', y.shape)
		inp = torch.empty(y.shape[:-1] + (self.n_in), device = device)
		inp[..., :-1] = y
		inp[..., -1] = self.force(t)

		print('Shape of inp in `neuralode_modified`: ', inp.shape)

		return self.model(inp), vmap(vmap(jacrev(self.model)))(inp)[..., :-1]

	def initial(self, nsamples):
		return torch.zeros(nsamples, self.n_out, device = device)



