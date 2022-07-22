from pyoptmat import temperature, hardening
IsotropicHardeningModel = hardening.IsotropicHardeningModel

import numpy as np
import torch
from torch import nn

class LinearHardeningModel(IsotropicHardeningModel):

	"""
	Linear Hardening Model defined by:
	
	"""

	def __init__(self, kappa):

		super().__init__()
		self.kappa = kappa

	def value(self, h):

		"""
		Map the internal variables to the value of the isotropic hardening value.

		Args:
		- h (torch.tensor): vector of internal variables for this model

	  	Returns:
	    - torch.tensor: tensor of the isotropic hardening values

		"""

		return h[:, 0]

	def dvalue(self, h):

		"""
		Map a vector of internal variables to the isotropic hardening value

		Args:
		- h (torch.tensor): vector of internal variables for this model

		Returns:
		- torch.tensor: isotropic hardening value

		"""
		
		return torch.ones((h.shape[0], 1), device=h.device)

	@property    
	def nhist(self):

 		"""
		
		Returns the number of internal variables

 		"""

 		return 1


	def history_rate(self, s, h, t, ep, T):

		"""

		The rate evolving the internal variables.

		Internal variables:
		- s (torch.tensor): stress
		- h (torch.tensor): history
		- t (torch.tensor): time
		- ep (torch.tensor): inelasting strain rate
		- T (torch.tensor): temperature

		Returns:
		- torch.tensor: internal variable rate

		"""

		return torch.unsqueeze(
			self.kappa(T) * torch.abs(ep), 1
		)

	def dhistory_rate_dstress(self, s, h, t, ep, T):
		"""
		The derivative of history rate w.r.t. stress

		Parameters:
		- s (torch.tensor): stress
		- h (torch.tensor): history
		- t (torch.tensor): time
		- ep (torch.tensor): the inelastic strain rate
		- T (torch.tensor): the temperature

		Returns:
		- torch.tensor: derivative with respect to stress

		"""
    	
		return torch.zeros_like(h)

	def dhistory_rate_dhistory(self, s, h, t, ep, T):
		"""
        The derivative of the history rate with respect to the internal variables
        
        Args:
        - s (torch.tensor): stress
        - h (torch.tensor): history
        - t (torch.tensor): time
        - ep (torch.tensor): the inelastic strain rate
        - T (torch.tensor): the temperature
        Returns:
          torch.tensor: derivative with respect to history
        
        """
		return torch.unsqueeze(torch.unsqueeze(torch.tensor(self.kappa(T)), -1) * torch.zeros_like(h) * torch.abs(ep)[:, None], 1)

	def dhistory_rate_derate(self, s, h, t, ep, T):
		"""
		The derivative of the history rate with respect to the inelastic
		strain rate

		Args:
		- s (torch.tensor): stress
		- h (torch.tensor): history
		- t (torch.tensor): time
		- ep (torch.tensor): the inelastic strain rate
		- T (torch.tensor): the temperature

		Returns:
		- torch.tensor: derivative with respect to the inelastic rate

		"""
		return torch.unsqueeze(
        	torch.unsqueeze(self.kappa(T) * torch.sign(ep), 1),1,
        )









