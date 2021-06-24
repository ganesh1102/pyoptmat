"""
  Objects and helper functions to help with deterministic model calibration
  and statistical inference.
"""

from collections import defaultdict
import warnings

import numpy as np
import scipy.interpolate as inter
import scipy.optimize as opt

import torch
from torch.nn import Module, Parameter

from skopt.space import Space
from skopt.sampler import Lhs

import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
from pyro.contrib.autoguide import AutoDelta, init_to_mean

from tqdm import tqdm

def construct_weights(etypes, weights, normalize = True):
  """
    Construct an array of weights 

    Args:
      etypes:               strings giving the experiment type
      weights:              dictionary mapping etype to weight
      normalize (optional): normalize by the number of experiments of each type
  """
  warray = torch.ones(len(etypes))

  count = defaultdict(int)
  for i,et in enumerate(etypes):
    warray[i] = weights[et]
    count[et] += 1

  if normalize:
    for i, et in enumerate(etypes):
      warray[i] /= count[et]

  return warray

def grid_search(model, time, strain, stress, loss, bounds, 
    ngrid, method = "lhs-maximin", save_grid = None,
    rbf_function = "inverse"):
  """
    Use a coarse grid search to find a good starting point

    Args:
      model:                    forward model
      time:                     time data
      strain:                   strain data
      stress:                   stress data
      loss:                     loss function
      bounds:                   bounds on each parameter
      ngrid:                    number of points
      method (optional):        method for generating the grid
      save_grid (optional):     save the parameter grid to a file for future use
      rbf_function (optional):  kernel for radial basis interpolation
  """
  # It's wasteful to do the adjoint propagation here, disable it
  use_adjoint = model.use_adjoint
  model.use_adjoint = False

  # Helpful later
  device = list(bounds.values())[0][0].device

  # Parameter order
  params = [(n, p.shape, torch.flatten(p).shape[0]) for n, p
      in list(model.named_parameters())]
  offsets = [0] + list(np.cumsum([i[2] for i in params]))
  
  # Generate the space
  # Need to do this on the cpu
  space = []
  for n, shp, sz in params:
    for i in range(sz):
      space.append((bounds[n][0].cpu().numpy().flatten()[i],
        bounds[n][1].cpu().numpy().flatten()[i]))
  sspace = Space(space)

  # Actually sample
  if method == "lhs-maximin":
    sampler = Lhs(criterion = "maximin")
  else:
    raise ValueError("Unknown sampling method %s" % method)
  
  samples = torch.tensor(sampler.generate(sspace.dimensions, ngrid),
      device = device)
  results = torch.zeros(samples.shape[0], device = samples.device)

  # Here we go
  pdict = {n:p for n,p in model.named_parameters()}
  for i,sample in tqdm(enumerate(samples), total = len(samples),
      desc = "Grid sampling"):
    for k,(name, shp, sz) in enumerate(params):
      pdict[name].data = samples[i][offsets[k]:offsets[k+1]].reshape(shp)
    with torch.no_grad():
      pred = model.solve(time, strain)
      lv = loss(pred[:,:,0], stress)
      results[i] = lv

  data = torch.hstack((samples, results[:,None]))
  # Store the results if we want
  if save_grid is not None:
    torch.save(data, save_grid)

  # Uncache
  model.use_adjoint = use_adjoint

  # We now need to do this on the CPU again
  data = data.cpu().numpy()
  
  # Get the surrogate and minimize it
  ifn = inter.Rbf(*(data[:,i] for i in range(data.shape[1])), 
      function = rbf_function)
  res = opt.minimize(lambda x: ifn(*x), [(i+j)/2 for i,j in space],
      method = 'L-BFGS-B', bounds = space)
  if not res.success:
    warnings.warn("Surrogate model minimization did not succeed!")

  # Setup the parameter dict and alter the model
  result = {}
  for k, (name, shp, sz) in enumerate(params):
    pdict[name].data = torch.tensor(res.x[offsets[k]:offsets[k+1]]).reshape(shp).to(device)
    result[name] = res.x[offsets[k]:offsets[k+1]].reshape(shp)
  
  return result

def bounded_scale_function(bounds):
  """
    Sets up a scaling function that maps `(0,1)` to `(bounds[0], bounds[1])`
    and clips the values to remain in that range

    Args:
      bounds:   tuple giving the parameter bounds
  """
  return lambda x: torch.clamp(x, 0, 1)*(bounds[1]-bounds[0]) + bounds[0]

class DeterministicModel(Module):
  """
    Wrap a material model to provide a :py:mod:`pytorch` deterministic model

    Args:
      maker:      function that returns a valid Module, given the 
                  input parameters
      names:      names to use for the parameters
      ics:        initial conditions to use for each parameter
  """
  def __init__(self, maker, names, ics):
    super().__init__()

    self.maker = maker
    
    # Add all the parameters
    self.params = names
    for name, ic in zip(names, ics):
      setattr(self, name, Parameter(torch.tensor(ic)))

  def get_params(self):
    """
      Return the parameters for input to the model
    """
    return [getattr(self, name) for name in self.params]

  def forward(self, times, strains):
    """
      Integrate forward and return the stress

      Args:
        times:      time points to hit
        strains:    input strain data
    """
    model = self.maker(*self.get_params())
    
    return model.solve(times, strains)[:,:,0]

class StatisticalModel(PyroModule):
  """
    Wrap a material model to provide a py:mod:`pyro` single-level 
    statistical model

    Single level means each parameter is sampled once before running
    all the tests -- i.e. each test is run on the "heat" of material

    Args:
      maker:      function that returns a valid Module, given the 
                  input parameters
      names:      names to use for the parameters
      loc:        parameter location priors
      scales:     parameter scale priors
      eps:        random noise, could be a constant value or a parameter
  """
  def __init__(self, maker, names, locs, scales, eps):
    super().__init__()

    self.maker = maker
    
    # Add all the parameters
    self.params = names
    for name, loc, scale in zip(names, locs, scales):
      setattr(self, name, PyroSample(dist.Normal(loc, scale)))

    self.eps = eps

  def get_params(self):
    """
      Return the sampled parameters for input to the model
    """
    return [getattr(self, name) for name in self.params]

  def forward(self, times, strains, true = None):
    """
      Integrate forward and return the result

      Args:
        times:      time points to hit in integration
        strains:    input strains

      Additional Args:
        true:       true values of the stress, if we're using this
                    model in inference
    """
    model = self.maker(*self.get_params())
    stresses = model.solve(times, strains)[:,:,0]
    with pyro.plate("trials", times.shape[1]):
      with pyro.plate("time", times.shape[0]):
        return pyro.sample("obs", dist.Normal(stresses, self.eps), obs = true)
    
class HierarchicalStatisticalModel(PyroModule):
  """
    Wrap a material model to provide a hierarchical :py:mod:`pyro` statistical
    model

    This type of statistical model does two levels of sampling for each
    parameter in the base model.

    First it samples a random variable to select the mean and scale of the
    population of parameter values

    Then, based on this "top level" location and scale it samples each parameter
    independently -- i.e. each experiment is drawn from a different "heat",
    with population statistics given by the top samples

    At least for the moment the population means are selected from 
    normal distributions, the population standard deviations from HalfNormal
    distributions, and then each parameter population comes from a
    Normal distribution

    Args: 
      maker:                    function that returns a valid material Module, 
                                given the input parameters
      names:                    names to use for each parameter
      loc_loc_priors:           location of the prior for the mean of each
                                parameter
      loc_scale_priors:         scale of the prior of the mean of each
                                parameter
      scale_scale_priors:       scale of the prior of the standard
                                deviation of each parameter
      noise_priors:             prior on the white noise
      loc_suffix (optional):    append to the variable name to give the top
                                level sample for the location
      scale_suffix (optional):  append to the variable name to give the top
                                level sample for the scale
      include_noise (optional): if true include white noise in the inference
  """
  def __init__(self, maker, names, loc_loc_priors, loc_scale_priors,
      scale_scale_priors, noise_prior, loc_suffix = "_loc",
      scale_suffix = "_scale", include_noise = False):
    super().__init__()
    
    # Store things we might later 
    self.maker = maker
    self.loc_suffix = loc_suffix
    self.scale_suffix = scale_suffix
    self.include_noise = include_noise

    # Setup both the top and bottom level variables
    self.bot_vars = names
    self.top_vars = []
    for var, loc_loc, loc_scale, scale_scale, in zip(
        names, loc_loc_priors, loc_scale_priors, scale_scale_priors):
      # These set standard PyroSamples with names of var + suffix
      dim = loc_loc.dim()
      self.top_vars.append(var + loc_suffix)
      setattr(self, self.top_vars[-1], PyroSample(dist.Normal(loc_loc,
        loc_scale).to_event(dim)))
      self.top_vars.append(var + scale_suffix)
      setattr(self, self.top_vars[-1], PyroSample(dist.HalfNormal(scale_scale
        ).to_event(dim)))
      
      # The tricks are: 1) use lambda self and 2) remember how python binds...
      setattr(self, var, PyroSample(
        lambda self, var = var, dim = loc_loc.dim(): dist.Normal(
          getattr(self, var + loc_suffix),
          getattr(self, var + scale_suffix)).to_event(dim)))

    # Setup the noise
    if self.include_noise:
      self.eps = PyroSample(dist.HalfNormal(noise_prior))
    else:
      self.eps = torch.tensor(noise_prior)

    # This annoyance is required to make the adjoint solver work
    self.extra_param_names = []

  def sample_top(self):
    """
      Sample the top level variables
    """
    return [getattr(self, name) for name in self.top_vars]

  def sample_bot(self):
    """
      Sample the bottom level variables
    """
    return [getattr(self, name) for name in self.bot_vars]

  def make_guide(self):
    """
      Make the guide and cache the extra parameter names the adjoint solver
      is going to need

      Currently this uses a Delta distribution for both the top and
      bottom level parameters.  This could be altered in the future
      to have the bottom level use a normal.
    """
    guide = AutoDelta(self, init_loc_fn = init_to_mean())
    self.extra_param_names = ["AutoDelta." + name for name in self.bot_vars]
    return guide

  def get_extra_params(self):
    """
      Actually list the extra parameters required for the adjoint solve.

      We can't determine this by introspection on the base model, so
      it needs to be done here
    """
    if len(self.extra_param_names) == 0:
      return []
    elif self.extra_param_names[0] not in pyro.get_param_store().keys():
      return []
    else:
      return [pyro.param(name) for name in self.extra_param_names]

  def forward(self, times, strains, true_stresses = None):
    """
      Evaluate the forward model, conditioned by the true stresses if
      they are available

      Args:
        times:                      time points to hit
        strains:                    input strains
        true_stresses (optional):   actual stress values, if we're conditioning
                                    for inference
    """
    # Sample the top level parameters
    curr = self.sample_top()
    eps = self.eps

    with pyro.plate("trials", times.shape[1]):
      # Sample the bottom level parameters
      bmodel = self.maker(*self.sample_bot(),
          extra_params = self.get_extra_params())
      # Generate the stresses
      stresses = bmodel.solve(times, strains)[:,:,0]
      # Sample!
      with pyro.plate("time", times.shape[0]):
        pyro.sample("obs", dist.Normal(stresses, eps), obs = true_stresses)

    return stresses