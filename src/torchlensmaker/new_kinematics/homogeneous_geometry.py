# This file is part of Torch Lens Maker
# Copyright (C) 2024-present Victor Poughon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import functools
from typing import TypeAlias

import torch
from jaxtyping import Float

from torchlensmaker.core.rot3d import euler_angles_to_matrix

HomMatrix2D: TypeAlias = Float[torch.Tensor, "3 3"]
HomMatrix3D: TypeAlias = Float[torch.Tensor, "4 4"]
HomMatrix: TypeAlias = HomMatrix2D | HomMatrix3D


def hom_matrix_2d(M: Float[torch.Tensor, "2 2"]) -> HomMatrix2D:
    "Extend a 2x2 matrix to a homonegenous coordinates 3x3 matrix"
    right = torch.zeros((2, 1), dtype=M.dtype, device=M.device)
    bottom = torch.tensor([[0.0, 0.0, 1.0]], dtype=M.dtype, device=M.device)
    return torch.cat((torch.cat((M, right), dim=1), bottom), dim=0)


def hom_matrix_3d(M: Float[torch.Tensor, "3 3"]) -> HomMatrix3D:
    "Extend a 3x3 matrix to a homonegenous coordinates 4x4 matrix"
    right = torch.zeros((3, 1), dtype=M.dtype, device=M.device)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=M.dtype, device=M.device)
    return torch.cat((torch.cat((M, right), dim=1), bottom), dim=0)


def hom_matrix(M: Float[torch.Tensor, "D D"]) -> HomMatrix:
    "Extend a 2x2 or 3x3 matrix to homogenerous coordinates matrix"
    assert M.shape == (2, 2) or M.shape == (3, 3)
    return hom_matrix_2d(M) if M.shape[0] == 2 else hom_matrix_3d(M)


def hom_identity_2d(
    dtype: torch.dtype, device: torch.device
) -> tuple[HomMatrix2D, HomMatrix2D]:
    "Identity 2D homogeneous transform matrices"
    return (
        torch.eye(3, dtype=dtype, device=device),
        torch.eye(3, dtype=dtype, device=device),
    )


def hom_identity_3d(
    dtype: torch.dtype, device: torch.device
) -> tuple[HomMatrix3D, HomMatrix3D]:
    "Identity 3D homogeneous transform matrices"
    return (
        torch.eye(4, dtype=dtype, device=device),
        torch.eye(4, dtype=dtype, device=device),
    )


def hom_identity(
    dim: int, dtype: torch.dtype, device: torch.device
) -> tuple[HomMatrix, HomMatrix]:
    "Identity homogeneous transform matrices in 2D or 3D"
    return (
        hom_identity_2d(dtype, device) if dim == 2 else hom_identity_3d(dtype, device)
    )


def hom_translate_2d(T: Float[torch.Tensor, "2"]) -> tuple[HomMatrix2D, HomMatrix2D]:
    "Homogeneous transform matrices for a 2D translation"
    eye = torch.eye(2, dtype=T.dtype, device=T.device)
    top_direct = torch.cat((eye, T.unsqueeze(1)), dim=1)
    top_inverse = torch.cat((eye, -T.unsqueeze(1)), dim=1)
    bottom = torch.tensor([[0.0, 0.0, 1.0]], dtype=T.dtype, device=T.device)
    return (
        torch.cat((top_direct, bottom), dim=0),
        torch.cat((top_inverse, bottom), dim=0),
    )


def hom_translate_3d(T: Float[torch.Tensor, "3"]) -> tuple[HomMatrix3D, HomMatrix3D]:
    "Homogeneous transform matrices for a 3D translation"
    eye = torch.eye(3, dtype=T.dtype, device=T.device)
    top_direct = torch.cat((eye, T.unsqueeze(1)), dim=1)
    top_inverse = torch.cat((eye, -T.unsqueeze(1)), dim=1)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=T.dtype, device=T.device)
    return (
        torch.cat((top_direct, bottom), dim=0),
        torch.cat((top_inverse, bottom), dim=0),
    )


def hom_rotate_2d(theta: Float[torch.Tensor, ""]) -> tuple[HomMatrix2D, HomMatrix2D]:
    "Homogeneous transform matrices for a 2D rotation in radians"
    zero = torch.zeros((), dtype=theta.dtype, device=theta.device)
    bottom = torch.tensor([0.0, 0.0, 1.0], dtype=theta.dtype, device=theta.device)
    rot = torch.stack(
        (
            torch.stack((torch.cos(theta), -torch.sin(theta), zero)),
            torch.stack((torch.sin(theta), torch.cos(theta), zero)),
            bottom,
        ),
    )
    return rot, rot.T


def hom_rotate_3d(
    y: Float[torch.Tensor, ""], z: Float[torch.Tensor, ""]
) -> tuple[HomMatrix3D, HomMatrix3D]:
    """
    Transform for a 2 axis euler angles 3D rotation in degrees

    Args:
        y: rotation around the Y axis in degrees
        z: rotation around the Z axis in degrees

    Returns:
        Pair of homogeneous transform matrices representing the transform
    """

    zero = torch.zeros((), dtype=y.dtype, device=y.device)
    euler_angles = torch.deg2rad(torch.stack((zero, y, z)))
    M = euler_angles_to_matrix(
        euler_angles,
        "XYZ",
    ).to(dtype=y.dtype)  # TODO need to support dtype in euler_angles_to_matrix

    return hom_matrix_3d(M), hom_matrix_3d(M.T)


def kinematic_chain_append(
    dfk: HomMatrix, ifk: HomMatrix, hom: HomMatrix, hom_inv: HomMatrix
) -> tuple[HomMatrix, HomMatrix]:
    """
    Append a joint to a forward kinematic chain

    The existing kinematic chain is represented by the pair of matrices (dfk,
    ifk) which model the composed forward transform of the existing kinematic
    chain and its inverse.

    Args:
        dfk: Direct Forward Kinematic transform
        ifk: Inverse Forward Kinematic transform
        hom: Direct transform of the new joint
        hom_inv: Inverse transform of the new joint
    """
    assert hom.shape == hom_inv.shape == dfk.shape == ifk.shape
    assert hom.dtype == hom_inv.dtype == dfk.dtype == ifk.dtype

    return (
        dfk @ hom,
        hom_inv @ ifk,
    )


def hom_compose(
    homs: list[HomMatrix], homs_inv: list[HomMatrix]
) -> tuple[HomMatrix, HomMatrix]:
    """
    Compose a list of transforms represented by direct and inverse homogeneous
    matrices, in the order of the input lists.

    Args:
        homs: list of direct transform homogeneous matrices
        homs_inv: list of inverse transform homogeneous matrices

    Returns:
        the composed direct and inverse transform as homogeneous matrices
    """

    assert len(homs) == len(homs_inv)
    assert len(homs) >= 1

    dtype, device = homs[0].dtype, homs[0].device
    assert all(dtype == h.dtype for h in homs)
    assert all(dtype == h.dtype for h in homs_inv)
    assert all(device == h.device for h in homs)
    assert all(device == h.device for h in homs_inv)
    assert all(h.shape == (3, 3) or h.shape == (4, 4) for h in homs)
    assert all(h.shape == (3, 3) or h.shape == (4, 4) for h in homs_inv)

    composed_hom = functools.reduce(lambda t1, t2: t2 @ t1, homs)
    composed_hom_inv = functools.reduce(lambda t1, t2: t1 @ t2, homs_inv)
    return composed_hom, composed_hom_inv


def kinematic_chain_extend(
    dfk: HomMatrix, ifk: HomMatrix, homs: list[HomMatrix], homs_inv: list[HomMatrix]
) -> tuple[HomMatrix, HomMatrix]:
    """
    Extend a kinematic chain with a list of joints

    The existing kinematic chain is represented by the pair of matrices (dfk,
    ifk) which model the composed forward transform of the existing kinematic
    chain and its inverse.

    Args:
        dfk: Direct Forward Kinematic transform
        ifk: Inverse Forward Kinematic transform
        homs: List of new joints direct transforms
        homs_inv: List of new joints inverse transforms
    """
    assert dfk.dtype == ifk.dtype
    assert dfk.shape == ifk.shape

    assert len(homs) == len(homs_inv)
    assert all([dfk.dtype == h.dtype for h in homs])
    assert all([dfk.dtype == h.dtype for h in homs_inv])
    assert all([dfk.device == h.device for h in homs])
    assert all([dfk.device == h.device for h in homs_inv])

    new_dfk = functools.reduce(lambda t1, t2: t1 @ t2, [dfk, *homs])
    new_ifk = functools.reduce(lambda t1, t2: t2 @ t1, [ifk, *homs_inv])
    return new_dfk, new_ifk


def transform_points(
    hom: HomMatrix, points: Float[torch.Tensor, "*B D"]
) -> Float[torch.Tensor, "*B D"]:
    """
    Apply a homogeneous transform matrix to points
    """

    D = points.shape[-1]
    assert hom.shape == (D + 1, D + 1), (
        f"Transform matrix ({hom.shape}) and tensor shape mismatch ({points.shape})"
    )

    ones = torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)
    hom_points = torch.cat((points, ones), dim=-1)
    transformed = torch.einsum("ij,...j->...i", hom, hom_points)
    return transformed[..., :-1]


def transform_vectors(
    hom: HomMatrix, vectors: Float[torch.Tensor, "*B D"]
) -> Float[torch.Tensor, "*B D"]:
    """
    Apply a homogeneous transform matrix to vectors
    """

    D = vectors.shape[-1]
    assert hom.shape == (D + 1, D + 1), (
        f"Transform matrix ({hom.shape}) and tensor shape mismatch ({vectors.shape})"
    )

    return torch.einsum("ij,...j->...i", hom[:-1, :-1], vectors)


def transform_rays(
    hom: HomMatrix, P: Float[torch.Tensor, "*B D"], V: Float[torch.Tensor, "*B D"]
) -> tuple[Float[torch.Tensor, "*B D"], Float[torch.Tensor, "*B D"]]:
    """
    Apply a homogeneous transform matrix to a pair of tensors representing rays
    """
    return transform_points(hom, P), transform_vectors(hom, V)
