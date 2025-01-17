from pyoptmat import chunktime

import torch

import unittest
import warnings

warnings.filterwarnings("ignore")

torch.set_default_tensor_type(torch.DoubleTensor)


class TestBackwardEulerChunkTimeOperator(unittest.TestCase):
    def setUp(self):
        self.sblk = 6
        self.max_nblk = 31
        self.sbat = 5

    def _gen_operators(self):
        self.blk_A = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk)
        self.blk_B = (
            torch.rand(self.nblk - 1, self.sbat, self.sblk, self.sblk) / 10
        )  # Diagonal dominance

        self.A = chunktime.BidiagonalForwardOperator(self.blk_A, self.blk_B)
        self.b = torch.rand(self.sbat, self.nblk * self.sblk)

    def test_inv_mat_vec_thomas(self):
        for self.nblk in range(1,self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalThomasFactorization(self.blk_A, self.blk_B)
            one = torch.linalg.solve(self.A.to_diag().to_dense(), self.b)
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_inv_mat_vec_pcr(self):
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalPCRFactorization(self.blk_A, self.blk_B)
            one = torch.linalg.solve(self.A.to_diag().to_dense(), self.b)
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_inv_mat_vec_hybrid_pcr(self):
        """Hybrid method, but set min_size so it always uses PCR
        """
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalHybridFactorization(self.blk_A, self.blk_B)
            one = torch.linalg.solve(self.A.to_diag().to_dense(), self.b)
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_inv_mat_vec_hybrid_thomas(self):
        """Hybrid method, but set min_size so it always uses Thomas
        """
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalHybridFactorization(self.blk_A, self.blk_B, 
                    min_size = self.max_nblk + 1)
            one = torch.linalg.solve(self.A.to_diag().to_dense(), self.b)
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_inv_mat_vec_hybrid_actual(self):
        """Hybrid method actually set to do something
        """
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            M = chunktime.BidiagonalHybridFactorization(self.blk_A, self.blk_B, 
                    min_size = self.nblk//2)
            one = torch.linalg.solve(self.A.to_diag().to_dense(), self.b)
            two = M(self.b)

            self.assertTrue(torch.allclose(one, two))

    def test_mat_vec(self):
        for self.nblk in range(1, self.max_nblk):
            self._gen_operators()
            one = self.A.to_diag().to_dense().matmul(self.b.unsqueeze(-1)).squeeze(-1)
            two = self.A(self.b)

            self.assertTrue(torch.allclose(one, two))


class TestBasicSparseSetup(unittest.TestCase):
    def setUp(self):
        self.sblk = 4
        self.nblk = 3
        self.sbatch = 4

        self.u = torch.zeros(self.nblk - 2, self.sbatch, self.sblk, self.sblk)
        self.d = torch.zeros(self.nblk, self.sbatch, self.sblk, self.sblk)
        self.l = torch.zeros(self.nblk - 1, self.sbatch, self.sblk, self.sblk)

        for i in range(self.nblk - 2):
            self.u[i] = (i + 2) * 1.0
        for i in range(self.nblk):
            self.d[i] = 2.0 * i - 1.0
        for i in range(self.nblk - 1):
            self.l[i] = -(i + 1) * 1.0

        self.sp = chunktime.SquareBatchedBlockDiagonalMatrix(
            [self.d, self.l, self.u], [0, -1, 2]
        )

    def test_coo(self):
        coo = self.sp.to_batched_coo()
        d = coo.to_dense().movedim(-1, 0)
        od = self.sp.to_dense()

        self.assertTrue(torch.allclose(d, od))

    def test_csr(self):
        csr_list = self.sp.to_unrolled_csr()
        od = self.sp.to_dense()

        for i in range(self.sbatch):
            self.assertTrue(torch.allclose(csr_list[i].to_dense(), od[i]))
