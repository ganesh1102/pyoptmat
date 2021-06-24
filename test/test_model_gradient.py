import unittest

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn

from pyoptmat import ode, models, flowrules, hardening, utility, damage

torch.set_default_tensor_type(torch.DoubleTensor)
torch.autograd.set_detect_anomaly(True)

def differ(mfn, p0, eps = 1.0e-6):
  v0 = mfn(p0).numpy()

  puse = p0.numpy()

  result = np.zeros(puse.shape)

  for ind, val in np.ndenumerate(puse):
    dp = np.abs(val) * eps
    if dp < eps:
      dp = eps
    pcurr = np.copy(puse)
    pcurr[ind] += dp
    v1 = mfn(torch.tensor(pcurr)).numpy()
    result[ind] = (v1 - v0) / dp
    
  return result

def simple_diff(fn, p0):
  res = []
  for i in range(len(p0)):
    def mfn(pi):
      ps = [pp for pp in p0]
      ps[i] = pi
      return fn(ps)

    res.append(differ(mfn, p0[i]))

  return res


class CommonGradient:
  def test_gradient(self):
    bmodel = self.model_fn([Variable(pi,
      requires_grad = True) for pi in self.p])
    res = torch.norm(bmodel.solve(self.times, self.strains))
    res.backward()
    grad = self.extract_grad(bmodel)
    
    ngrad = simple_diff(
        lambda p: torch.norm(self.model_fn(p).solve(self.times, self.strains)),
        self.p)

    for i,(p1, p2) in enumerate(zip(grad, ngrad)):
      print(i,p1, p2)
      self.assertTrue(np.allclose(p1, p2, rtol = 1e-4))
    
class TestPerfectViscoplasticity(unittest.TestCase, CommonGradient):
  def setUp(self):
    self.ntime = 10
    self.nbatch = 10

    self.E = torch.tensor(100000.0)
    self.n = torch.tensor(5.2)
    self.eta = torch.tensor(110.0)

    self.p = [self.E, self.n, self.eta]

    self.model_fn = lambda p: models.InelasticModel(p[0],
        flowrules.PerfectViscoplasticity(p[1], p[2]))

    self.extract_grad = lambda m: np.array(
        [m.E.grad.numpy(), m.flowrule.n.grad.numpy(), m.flowrule.eta.grad.numpy()])

    self.times = torch.transpose(
        torch.tensor([np.linspace(0,1,self.ntime) for i in range(self.nbatch)]), 1, 0)
    self.strains = torch.transpose(
        torch.tensor([np.linspace(0,0.003,self.ntime) for i in range(self.nbatch)]), 1, 0)

class TestIsotropicOnly(unittest.TestCase, CommonGradient):
  def setUp(self):
    self.ntime = 10
    self.nbatch = 10

    self.E = torch.tensor(100000.0)
    self.n = torch.tensor(5.2)
    self.eta = torch.tensor(110.0)
    self.R = torch.tensor(100.0)
    self.d = torch.tensor(5.1)
    self.s0 = torch.tensor(10.0)

    self.p = [self.E, self.n, self.eta, self.s0, self.R, self.d]

    self.model_fn = lambda p: models.InelasticModel(p[0],
        flowrules.IsoKinViscoplasticity(p[1], p[2], p[3],
          hardening.VoceIsotropicHardeningModel(p[4],p[5]),
          hardening.NoKinematicHardeningModel()))

    self.extract_grad = lambda m: np.array(
        [m.E.grad.numpy(), m.flowrule.n.grad.numpy(), m.flowrule.eta.grad.numpy(),
          m.flowrule.s0.grad.numpy(), m.flowrule.isotropic.R.grad.numpy(),
          m.flowrule.isotropic.d.grad.numpy()])

    self.times = torch.transpose(
        torch.tensor([np.linspace(0,1,self.ntime) for i in range(self.nbatch)]), 1, 0)
    self.strains = torch.transpose(
        torch.tensor([np.linspace(0,0.003,self.ntime) for i in range(self.nbatch)]), 1, 0)

class TestHardeningViscoplasticity(unittest.TestCase, CommonGradient):
  def setUp(self):
    self.ntime = 10
    self.nbatch = 10

    self.E = torch.tensor(100000.0)
    self.n = torch.tensor(5.2)
    self.eta = torch.tensor(110.0)
    self.R = torch.tensor(100.0)
    self.d = torch.tensor(5.1)
    self.C = torch.tensor(1000.0)
    self.g = torch.tensor(10.0)
    self.s0 = torch.tensor(10.0)

    self.p = [self.E, self.n, self.eta, self.s0, self.R, self.d, self.C, self.g]

    self.model_fn = lambda p: models.InelasticModel(p[0],
        flowrules.IsoKinViscoplasticity(p[1], p[2], p[3],
          hardening.VoceIsotropicHardeningModel(p[4],p[5]),
          hardening.FAKinematicHardeningModel(p[6],p[7])))

    self.extract_grad = lambda m: np.array(
        [m.E.grad.numpy(), m.flowrule.n.grad.numpy(), m.flowrule.eta.grad.numpy(),
          m.flowrule.s0.grad.numpy(), m.flowrule.isotropic.R.grad.numpy(),
          m.flowrule.isotropic.d.grad.numpy(),
          m.flowrule.kinematic.C.grad.numpy(), m.flowrule.kinematic.g.grad.numpy()])

    self.times = torch.transpose(
        torch.tensor([np.linspace(0,1,self.ntime) for i in range(self.nbatch)]), 1, 0)
    self.strains = torch.transpose(
        torch.tensor([np.linspace(0,0.003,self.ntime) for i in range(self.nbatch)]), 1, 0)

class TestHardeningViscoplasticityDamage(unittest.TestCase, CommonGradient):
  def setUp(self):
    self.ntime = 10
    self.nbatch = 10

    self.E = torch.tensor(100000.0)
    self.n = torch.tensor(5.2)
    self.eta = torch.tensor(110.0)
    self.R = torch.tensor(100.0)
    self.d = torch.tensor(5.1)
    self.C = torch.tensor(1000.0)
    self.g = torch.tensor(10.0)
    self.s0 = torch.tensor(10.0)
    self.A = torch.tensor(2000.0)
    self.xi = torch.tensor(6.5)
    self.phi = torch.tensor(1.7)

    self.p = [self.E, self.n, self.eta, self.s0, self.R, self.d, self.C, self.g, self.A, self.xi, self.phi]

    self.model_fn = lambda p: models.InelasticModel(p[0],
        flowrules.IsoKinViscoplasticity(p[1], p[2], p[3],
          hardening.VoceIsotropicHardeningModel(p[4],p[5]),
          hardening.FAKinematicHardeningModel(p[6],p[7])),
        dmodel = damage.HayhurstLeckie(p[8], p[9], p[10]))

    self.extract_grad = lambda m: np.array(
        [m.E.grad.numpy(), m.flowrule.n.grad.numpy(), m.flowrule.eta.grad.numpy(),
          m.flowrule.s0.grad.numpy(), m.flowrule.isotropic.R.grad.numpy(),
          m.flowrule.isotropic.d.grad.numpy(),
          m.flowrule.kinematic.C.grad.numpy(), m.flowrule.kinematic.g.grad.numpy(),
          m.dmodel.A.grad.numpy(), m.dmodel.xi.grad.numpy(), 
          m.dmodel.phi.grad.numpy()])

    self.times = torch.transpose(
        torch.tensor([np.linspace(0,1,self.ntime) for i in range(self.nbatch)]), 1, 0)
    self.strains = torch.transpose(
        torch.tensor([np.linspace(0,0.03,self.ntime) for i in range(self.nbatch)]), 1, 0)
   
class TestChabocheViscoplasticity(unittest.TestCase, CommonGradient):
  def setUp(self):
    self.ntime = 10
    self.nbatch = 4

    self.E = torch.tensor(100000.0)
    self.n = torch.tensor(5.2)
    self.eta = torch.tensor(110.0)
    self.R = torch.tensor(100.0)
    self.d = torch.tensor(5.1)
    self.C = torch.tensor([1000.0,750.0,100.0])
    self.g = torch.tensor([10.0, 1.2, 8.6])
    self.s0 = torch.tensor(10.0)

    self.p = [self.E, self.n, self.eta, self.s0, self.R, self.d, self.C, self.g]

    self.model_fn = lambda p: models.InelasticModel(p[0],
        flowrules.IsoKinViscoplasticity(p[1], p[2], p[3],
          hardening.VoceIsotropicHardeningModel(p[4],p[5]),
          hardening.ChabocheHardeningModel(p[6],p[7])))

    self.extract_grad = lambda m: np.array(
        [m.E.grad.numpy(), m.flowrule.n.grad.numpy(), m.flowrule.eta.grad.numpy(),
          m.flowrule.s0.grad.numpy(), m.flowrule.isotropic.R.grad.numpy(),
          m.flowrule.isotropic.d.grad.numpy(),
          m.flowrule.kinematic.C.grad.numpy(), m.flowrule.kinematic.g.grad.numpy()])

    self.times = torch.transpose(
        torch.tensor([np.linspace(0,1,self.ntime) for i in range(self.nbatch)]), 1, 0)
    self.strains = torch.transpose(
        torch.tensor([np.linspace(0,0.003,self.ntime) for i in range(self.nbatch)]), 1, 0)