#!/usr/bin/env python3

import torch
from torch import nn

from pyoptmat import utility, ode, solvers

class NODEIntegrator(nn.Module):
	"""
	This class provides infrastructure for integrating neural ODE models for strain control.

	Args:
	  model:                    base neural ODE model
	  method:                   integration method used to solve the 
	                            equations, defaults to 
	                            `'backward-euler'`
	  use_adjoint (optional):   if `True` use adjoint apporach in 
	  							the odeint method
	  **kwargs:					passed on to odeint method
	"""

	def __init__(self, model, n_hidden, n_layers, n_inter, *args, use_adjoint = True, **kwargs):
		super().__init__(*args)
		self.model = model
		print('Inside __init__')
		print(model)
		print(self.model)
		self.use_adjoint = use_adjoint
		self.kwargs_for_integration = kwargs

		self.n_hidden = n_hidden
		self.n_layers = n_layers
		self.n_inter = n_inter

		if self.use_adjoint:
			self.imethod = ode.odeint_adjoint
		else:
			self.imethod = ode.odeint

	def solve_both(self, times, temperatures, idata, control):
		"""
        Solve for either strain or stress control at once

        Args:
          times:          input times, (ntime,nexp)
          temperatures:   input temperatures (ntime,nexp)
          idata:          input data (ntime,nexp)
          control:        signal for stress/strain control (nexp,)
        """
		print('Inside solve_both')
		print(self.model)

		rates = torch.cat(
        	(
                torch.zeros(1, idata.shape[1], device=idata.device),
                (idata[1:] - idata[:-1]) / (times[1:] - times[:-1]),
        	)
        )

		rates[torch.isnan(rates)] = 0

		rates[torch.isnan(rates)] = 0

		rate_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(times, rates)
		base_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(times, idata)
		temperature_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            times, temperatures
        )

		init = torch.zeros(
        	times.shape[1], 2 + self.n_hidden, device = idata.device
        )

		

		bmodel = BothBasedModel(
            self.model,
            rate_interpolator,
            base_interpolator,
            temperature_interpolator,
            control,
        )

		return self.imethod(bmodel, init, times, **self.kwargs_for_integration)


class BothBasedModel(nn.Module):
	"""
	Provides both the strain rate and stress rate form at once, for better vectorization

	Args:
	  model:    base InelasticModel
	  rate_fn:  controlled quantity rate interpolator
	  base_fn:  controlled quantity base interpolator
	  T_fn:     temperature interpolator
	  indices:  split into strain and stress control
	"""

	def __init__(self, model, rate_fn, base_fn, T_fn, control, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model = model
		self.rate_fn = rate_fn
		self.base_fn = base_fn
		self.T_fn = T_fn
		self.control = control

		self.emodel = StrainBasedModel(self.model, self.rate_fn, self.T_fn)
		self.smodel = StressBasedModel(
		    self.model, self.rate_fn, self.base_fn, self.T_fn
		)

	def forward(self, t, y):
		"""
		Evaluate both strain and stress control and paste into the right
		locations.

		Args:
		    t:  input times
		    y:  input state
		"""
		strain_rates, strain_jacs = self.emodel(t, y)
		stress_rates, stress_jacs = self.smodel(t, y)

		actual_rates = torch.zeros_like(strain_rates)

		e_control = self.control == 0
		s_control = self.control == 1

		actual_rates[..., e_control, :] = strain_rates[..., e_control, :]
		actual_rates[..., s_control, :] = stress_rates[..., s_control, :]

		actual_jacs = torch.zeros_like(strain_jacs)
		actual_jacs[..., e_control, :, :] = strain_jacs[..., e_control, :, :]
		actual_jacs[..., s_control, :, :] = stress_jacs[..., s_control, :, :]

		return actual_rates, actual_jacs


class StrainBasedModel(nn.Module):
	"""
	Provides the strain rate form

	Args:
	  model:        base InelasticModel
	  erate_fn:     erate(t)
	  T_fn:         T(t)
	"""

	def __init__(self, model, erate_fn, T_fn, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model = model
		self.erate_fn = erate_fn
		self.T_fn = T_fn

	def forward(self, t, y):
		"""
		Strain rate as a function of t and state

		Args:
		t:  input times
		y:  input state
		"""
		inp = torch.cat((self.erate_fn(t), y))
		return self.model(inp)  # Don't need the extras


class StressBasedModel(nn.Module):
	"""
	Provides the stress rate form

	Args:
	  model:        base InelasticModel
	  srate_fn:     srate(t)
	  T_fn:         T(t)
	"""

	def __init__(self, model, srate_fn, stress_fn, T_fn, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.model = model
		self.srate_fn = srate_fn
		self.stress_fn = stress_fn
		self.T_fn = T_fn

	def forward(self, t, y):
		"""
		Stress rate as a function of t and state

		Args:
		    t:  input times
		    y:  input state
		"""
		csr = self.srate_fn(t)
		cs = self.stress_fn(t)
		cT = self.T_fn(t)

		erate_guess = torch.zeros_like(y[..., 0])[..., None]

		def RJ(erate):
			yp = y.clone()
			yp[..., 0] = cs
			ydot, _, Je, _ = self.model(t, yp, erate[..., 0], cT)

			R = ydot[..., 0] - csr
			J = Je[..., 0]

			return R[..., None], J[..., None, None]

		erate, _ = solvers.newton_raphson(RJ, erate_guess)
		yp = y.clone()
		yp[..., 0] = cs
		ydot, J, Je, _ = self.model(t, yp, erate[..., 0], cT)

		# Rescale the jacobian
		J[..., 0, :] = -J[..., 0, :] / Je[..., 0][..., None]
		J[..., :, 0] = 0

		# Insert the strain rate
		ydot[..., 0] = erate[..., 0]

		return ydot, J
