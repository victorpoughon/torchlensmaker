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

from torchlensmaker.types import (
    HomMatrix2D,
    HomMatrix3D,
    HomMatrix,
    Tf2D,
    Tf3D,
    Tf,
    BatchNDTensor,
)


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


def hom_identity_2d(dtype: torch.dtype, device: torch.device) -> Tf2D:
    "Identity 2D homogeneous transform matrices"
    return Tf2D(
        torch.eye(3, dtype=dtype, device=device),
        torch.eye(3, dtype=dtype, device=device),
    )


def hom_identity_3d(dtype: torch.dtype, device: torch.device) -> Tf3D:
    "Identity 3D homogeneous transform matrices"
    return Tf3D(
        torch.eye(4, dtype=dtype, device=device),
        torch.eye(4, dtype=dtype, device=device),
    )


def hom_identity(dim: int, dtype: torch.dtype, device: torch.device) -> Tf:
    "Identity homogeneous transform matrices in 2D or 3D"
    return (
        hom_identity_2d(dtype, device) if dim == 2 else hom_identity_3d(dtype, device)
    )


def hom_translate_2d(T: Float[torch.Tensor, "2"]) -> Tf2D:
    "Homogeneous transform matrices for a 2D translation"
    eye = torch.eye(2, dtype=T.dtype, device=T.device)
    top_direct = torch.cat((eye, T.unsqueeze(1)), dim=1)
    top_inverse = torch.cat((eye, -T.unsqueeze(1)), dim=1)
    bottom = torch.tensor([[0.0, 0.0, 1.0]], dtype=T.dtype, device=T.device)
    return Tf2D(
        torch.cat((top_direct, bottom), dim=0),
        torch.cat((top_inverse, bottom), dim=0),
    )


def hom_translate_3d(T: Float[torch.Tensor, "3"]) -> Tf3D:
    "Homogeneous transform matrices for a 3D translation"
    eye = torch.eye(3, dtype=T.dtype, device=T.device)
    top_direct = torch.cat((eye, T.unsqueeze(1)), dim=1)
    top_inverse = torch.cat((eye, -T.unsqueeze(1)), dim=1)
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=T.dtype, device=T.device)
    return Tf3D(
        torch.cat((top_direct, bottom), dim=0),
        torch.cat((top_inverse, bottom), dim=0),
    )


def hom_translate(T: Float[torch.Tensor, " D"]) -> Tf:
    "Homogeneous transform matrix for a 2D or 3D translation"

    assert T.shape == (2,) or T.shape == (3,)
    return hom_translate_2d(T) if T.shape[0] == 2 else hom_translate_3d(T)


def hom_rotate_2d(theta: Float[torch.Tensor, ""]) -> Tf2D:
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
    return Tf2D(rot, rot.T)


def hom_rotate_3d(y: Float[torch.Tensor, ""], z: Float[torch.Tensor, ""]) -> Tf3D:
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

    return Tf3D(hom_matrix_3d(M), hom_matrix_3d(M.T))


def hom_scale_2d(scale: Float[torch.Tensor, ""]) -> Tf2D:
    """
    Homogeneous transform matrices for a 2D scale on all axes
    """

    H = hom_matrix(scale * torch.eye(2, dtype=scale.dtype, device=scale.device))
    H_inv = hom_matrix(
        torch.reciprocal(scale) * torch.eye(2, dtype=scale.dtype, device=scale.device)
    )
    return Tf2D(H, H_inv)


def hom_scale_3d(scale: Float[torch.Tensor, ""]) -> Tf3D:
    """
    Homogeneous transform matrices for a 3D scale on all axes
    """

    H = hom_matrix(scale * torch.eye(3, dtype=scale.dtype, device=scale.device))
    H_inv = hom_matrix(
        torch.reciprocal(scale) * torch.eye(3, dtype=scale.dtype, device=scale.device)
    )
    return Tf3D(H, H_inv)


def hom_scale(dim: int, scale: Float[torch.Tensor, ""]) -> Tf:
    """
    Homogeneous transform matrices for a scale on all axes
    """

    H = hom_matrix(scale * torch.eye(dim, dtype=scale.dtype, device=scale.device))
    H_inv = hom_matrix(
        torch.reciprocal(scale) * torch.eye(dim, dtype=scale.dtype, device=scale.device)
    )
    return Tf2D(H, H_inv) if dim == 2 else Tf3D(H, H_inv)


def kinematic_chain_append_2d(fk: Tf2D, joint: Tf2D) -> Tf2D:
    """
    Append a joint to a 2D forward kinematic chain

    Args:
        fk: Forward Kinematic transform
        joint: New joint to append to the chain
    """
    assert fk.dtype == joint.dtype
    assert fk.device == joint.device

    return Tf2D(
        fk.direct @ joint.direct,
        joint.inverse @ fk.inverse,
    )


def kinematic_chain_append_3d(fk: Tf3D, joint: Tf3D) -> Tf3D:
    """
    Append a joint to a 3D forward kinematic chain

    Args:
        fk: Forward Kinematic transform
        joint: New joint to append to the chain
    """
    assert fk.dtype == joint.dtype
    assert fk.device == joint.device

    return Tf3D(
        fk.direct @ joint.direct,
        joint.inverse @ fk.inverse,
    )


def kinematic_chain_append(fk: Tf, joint: Tf) -> Tf:
    """
    Append a joint to a forward kinematic chain

    Args:
        fk: Forward Kinematic transform
        joint: New joint to append to the chain
    """
    assert fk.shape == joint.shape
    assert fk.dtype == joint.dtype

    return type(fk)(
        fk.direct @ joint.direct,
        joint.inverse @ fk.inverse,
    )


def kinematic_chain_extend_2d(fk: Tf2D, joints: list[Tf2D]) -> Tf2D:
    """
    Extend a 2D kinematic chain with a list of joints

    Args:
        fk: Forward Kinematic transform
        joints: New joints to append to the chain
    """
    for joint in joints:
        fk = kinematic_chain_append_2d(fk, joint)
    return fk


def kinematic_chain_extend_3d(fk: Tf3D, joints: list[Tf3D]) -> Tf3D:
    """
    Extend a 3D kinematic chain with a list of joints

    Args:
        fk: Forward Kinematic transform
        joints: New joints to append to the chain
    """
    for joint in joints:
        fk = kinematic_chain_append_3d(fk, joint)
    return fk


def kinematic_chain_extend(fk: Tf, joints: list[Tf]) -> Tf:
    """
    Extend a kinematic chain with a list of joints

    Args:
        fk: Forward Kinematic transform
        joints: New joints to append to the chain
    """
    for joint in joints:
        fk = kinematic_chain_append(fk, joint)
    return fk


def hom_compose_2d(tfs: list[Tf2D]) -> Tf2D:
    """
    Compose a list of 2D transforms
    """

    assert len(tfs) >= 1
    shape, dtype, device = tfs[0].shape, tfs[0].dtype, tfs[0].device
    assert all(shape == tf.shape for tf in tfs)
    assert all(dtype == tf.dtype for tf in tfs)
    assert all(device == tf.device for tf in tfs)

    return functools.reduce(
        lambda a, b: Tf2D(b.direct @ a.direct, a.inverse @ b.inverse), tfs
    )


def hom_compose_3d(tfs: list[Tf3D]) -> Tf3D:
    """
    Compose a list of 3D transforms
    """

    assert len(tfs) >= 1
    shape, dtype, device = tfs[0].shape, tfs[0].dtype, tfs[0].device
    assert all(shape == tf.shape for tf in tfs)
    assert all(dtype == tf.dtype for tf in tfs)
    assert all(device == tf.device for tf in tfs)

    return functools.reduce(
        lambda a, b: Tf3D(b.direct @ a.direct, a.inverse @ b.inverse), tfs
    )


def hom_compose(tfs: list[Tf]) -> Tf:
    """
    Compose a list of transforms
    """

    assert len(tfs) >= 1
    shape, dtype, device = tfs[0].shape, tfs[0].dtype, tfs[0].device
    assert all(shape == tf.shape for tf in tfs)
    assert all(dtype == tf.dtype for tf in tfs)
    assert all(device == tf.device for tf in tfs)

    cls = type(tfs[0])
    return functools.reduce(
        lambda a, b: cls(b.direct @ a.direct, a.inverse @ b.inverse), tfs
    )


def transform_points(hom: HomMatrix, points: BatchNDTensor) -> BatchNDTensor:
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


def transform_vectors(hom: HomMatrix, vectors: BatchNDTensor) -> BatchNDTensor:
    """
    Apply a homogeneous transform matrix to vectors
    """

    D = vectors.shape[-1]
    assert hom.shape == (D + 1, D + 1), (
        f"Transform matrix ({hom.shape}) and tensor shape mismatch ({vectors.shape})"
    )

    return torch.einsum("ij,...j->...i", hom[:-1, :-1], vectors)


def transform_rays(
    hom: HomMatrix, P: BatchNDTensor, V: BatchNDTensor
) -> tuple[BatchNDTensor, BatchNDTensor]:
    """
    Apply a homogeneous transform matrix to a pair of tensors representing rays
    """
    return transform_points(hom, P), transform_vectors(hom, V)
