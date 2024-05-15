import torch
from torch import Tensor
from torch.autograd import Function
from src.utils.sparse import bilinear_diag_torch
from src.sparse_torch import CSRMatrix3d, CSCMatrix3d
import scipy.sparse as ssp
import numpy as np


def construct_aff_mat(Ke: Tensor, Kp: Tensor, KroG: CSRMatrix3d, KroH: CSCMatrix3d,
                      KroGt: CSRMatrix3d=None, KroHt: CSCMatrix3d=None) -> Tensor:
    return RebuildFGM.apply(Ke, Kp, KroG, KroH, KroGt, KroHt)


def kronecker_torch(t1: Tensor, t2: Tensor) -> Tensor:
    batch_num = t1.shape[0]
    t1dim1, t1dim2 = t1.shape[1], t1.shape[2]
    t2dim1, t2dim2 = t2.shape[1], t2.shape[2]
    if t1.is_sparse and t2.is_sparse:
        tt_idx = torch.stack(t1._indices()[0, :] * t2dim1, t1._indices()[1, :] * t2dim2)
        tt_idx = torch.repeat_interleave(tt_idx, t2._nnz(), dim=1) + t2._indices().repeat(1, t1._nnz())
        tt_val = torch.repeat_interleave(t1._values(), t2._nnz(), dim=1) * t2._values().repeat(1, t1._nnz())
        tt = torch.sparse.FloatTensor(tt_idx, tt_val, torch.Size(t1dim1 * t2dim1, t1dim2 * t2dim2))
    else:
        t1 = t1.reshape(batch_num, -1, 1)
        t2 = t2.reshape(batch_num, 1, -1)
        tt = torch.bmm(t1, t2)
        tt = tt.reshape(batch_num, t1dim1, t1dim2, t2dim1, t2dim2)
        tt = tt.permute([0, 1, 3, 2, 4])
        tt = tt.reshape(batch_num, t1dim1 * t2dim1, t1dim2 * t2dim2)
    return tt


def kronecker_sparse(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    s1 = ssp.coo_matrix(arr1)
    s2 = ssp.coo_matrix(arr2)
    ss = ssp.kron(s1, s2)
    return ss


class RebuildFGM(Function):
    @staticmethod
    def forward(ctx, Ke: Tensor, Kp: Tensor, Kro1: CSRMatrix3d, Kro2: CSCMatrix3d,
                Kro1T: CSRMatrix3d=None, Kro2T: CSCMatrix3d=None) -> Tensor:
        ctx.save_for_backward(Ke, Kp)
        if Kro1T is not None and Kro2T is not None:
            ctx.K = Kro1T, Kro2T
        else:
            ctx.K = Kro1.transpose(keep_type=True), Kro2.transpose(keep_type=True)
        batch_num = Ke.shape[0]

        Kro1Ke = Kro1.dotdiag(Ke.transpose(1, 2).contiguous().view(batch_num, -1))
        Kro1KeKro2 = Kro1Ke.dot(Kro2, dense=True)

        K = torch.empty_like(Kro1KeKro2)
        for b in range(batch_num):
            K[b] = Kro1KeKro2[b] + torch.diag(Kp[b].transpose(0, 1).contiguous().view(-1))

        return K

    @staticmethod
    def backward(ctx, dK):
        device = dK.device
        Ke, Kp = ctx.saved_tensors
        Kro1t, Kro2t = ctx.K
        dKe = dKp = None
        if ctx.needs_input_grad[0]:
            dKe = bilinear_diag_torch(Kro1t, dK.contiguous(), Kro2t)
            dKe = dKe.view(Ke.shape[0], Ke.shape[2], Ke.shape[1]).transpose(1, 2)
        if ctx.needs_input_grad[1]:
            dKp = torch.diagonal(dK, dim1=-2, dim2=-1)
            dKp = dKp.view(Kp.shape[0], Kp.shape[2], Kp.shape[1]).transpose(1, 2)

        return dKe, dKp, None, None, None, None
