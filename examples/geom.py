"""
Tools for analyzing and comparing geometries
"""

import numpy as np
import torch
from torch.utils.data import DataLoader


BATCH_SIZE = 3000


def quaternion_to_matrix(q):

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    R_q = torch.stack([q0**2 + q1**2 - q2**2 - q3**2,
                       2 * (q1 * q2 - q0 * q3),
                       2 * (q1 * q3 + q0 * q2),
                       2 * (q1 * q2 + q0 * q3),
                       q0**2 - q1**2 + q2**2 - q3**2,
                       2 * (q2 * q3 - q0 * q1),
                       2 * (q1 * q3 - q0 * q2),
                       2 * (q2 * q3 + q0 * q1),
                       q0**2 - q1**2 - q2**2 + q3**2]
                      ).transpose(0, 1).reshape(-1, 3, 3)

    return R_q


def rotation_matrix_from_points(m0, m1):

    v0 = torch.clone(m0)[:, None, :, :]
    v1 = torch.clone(m1)

    out_0 = (v0 * v1).sum(-1).reshape(-1, 3)
    R11 = out_0[:, 0]
    R22 = out_0[:, 1]
    R33 = out_0[:, 2]

    out_1 = torch.sum(v0 * torch.roll(v1, -1, dims=1), dim=-1
                      ).reshape(-1, 3)
    R12 = out_1[:, 0]
    R23 = out_1[:, 1]
    R31 = out_1[:, 2]

    out_2 = torch.sum(v0 * torch.roll(v1, -2, dims=1), dim=-1
                      ).reshape(-1, 3)
    R13 = out_2[:, 0]
    R21 = out_2[:, 1]
    R32 = out_2[:, 2]

    f = torch.stack([R11 + R22 + R33, R23 - R32, R31 - R13, R12 - R21,
                     R23 - R32, R11 - R22 - R33, R12 + R21, R13 + R31,
                     R31 - R13, R12 + R21, -R11 + R22 - R33, R23 + R32,
                     R12 - R21, R13 + R31, R23 + R32, -R11 - R22 + R33]
                    ).transpose(0, 1).reshape(-1, 4, 4)

    # Really slow on a GPU / with torch for some reason.
    # See https://github.com/pytorch/pytorch/issues/22573:
    # the slow-down is significant in PyTorch, and is particularly
    # bad for small matrices.

    # Use numpy on cpu instead

    # w, V = torch.symeig(f, eigenvectors=True)

    f_np = f.detach().cpu().numpy()
    nan_idx = np.isnan(f_np).any(-1).any(-1)
    good_idx = np.bitwise_not(nan_idx)
    f_good = f_np[good_idx]

    if f_good.shape[0] != 0:
        # Only do this if we have any good idx
        # Otherwise we'll run into issues with
        # taking the argmax of an empty
        # sequence

        w, V = np.linalg.eigh(f_good)
        w = torch.Tensor(w).to(f.device)
        V = torch.Tensor(V).to(f.device)

        arg = w.argmax(dim=1)
        idx = list(range(len(arg)))
        q = V[idx, :, arg]

        R = quaternion_to_matrix(q)

    counter = 0
    r_with_nan = []

    for i, is_nan in enumerate(nan_idx):
        if is_nan:
            r_with_nan.append(torch.diag(torch.ones(3)))
            counter += 1
        else:
            r_with_nan.append(R[i - counter])
    r_with_nan = torch.stack(r_with_nan)

    return r_with_nan


def minimize_rotation_and_translation(targ_nxyz, this_nxyz):

    p = this_nxyz[:, :, 1:]
    p0 = targ_nxyz[:, :, 1:]

    c = p.mean(1).reshape(-1, 1, 3)
    p -= c

    c0 = p0.mean(1).reshape(-1, 1, 3)
    p0 -= c0

    R = rotation_matrix_from_points(p.transpose(1, 2),
                                    p0.transpose(1, 2))

    num_repeats = targ_nxyz.shape[0]
    p_repeat = torch.repeat_interleave(p, num_repeats, dim=0)

    new_p = torch.einsum("ijk,ilk->ijl", p_repeat, R)

    return new_p, p0, R


def compute_rmsd(targ_nxyz, this_nxyz):

    targ_nxyz = torch.Tensor(targ_nxyz).reshape(1, -1, 4)
    this_nxyz = torch.Tensor(this_nxyz).reshape(1, -1, 4)

    (new_atom, new_targ, _
     ) = minimize_rotation_and_translation(
        targ_nxyz=targ_nxyz,
        this_nxyz=this_nxyz)
    xyz_0 = new_atom

    num_mols_1 = targ_nxyz.shape[0]
    num_mols_0 = this_nxyz.shape[0]

    xyz_1 = new_targ.repeat(num_mols_0, 1, 1)

    delta_sq = (xyz_0 - xyz_1) ** 2
    num_atoms = delta_sq.shape[1]
    distances = (((delta_sq.sum((1, 2)) / num_atoms) ** 0.5)
                 .reshape(num_mols_0, num_mols_1)
                 .cpu().reshape(-1).item())

    return distances


