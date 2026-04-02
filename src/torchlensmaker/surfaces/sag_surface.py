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


from functools import partial
from typing import Any, Callable, Sequence, TypeAlias

import torch
from jaxtyping import Bool, Float

from torchlensmaker.types import (
    Batch2DTensor,
    BatchNDTensor,
    BatchTensor,
    MaskTensor,
    ScalarTensor,
    Tf,
)

from .implicit_solver import (
    DomainFunction,
    ImplicitSolver,
    implicit_surface_local_raytrace,
)
from .raytrace import surface_raytrace
from .sag_functions import (
    LiftFunction,
    SagFunction,
)


def sag_surface_raytrace(
    sag_function: SagFunction,
    lift_function: LiftFunction,
    domain_function: DomainFunction,
    implicit_solver: ImplicitSolver,
    P: BatchNDTensor,
    V: BatchNDTensor,
    tf_in: Tf,
    diameter: ScalarTensor,
    normalize: Bool[torch.Tensor, ""],
) -> tuple[BatchTensor, BatchNDTensor, MaskTensor, BatchNDTensor, BatchNDTensor]:
    """
    Generic raytracing for a sag surface.
    Used to implement surface kernels.
    """

    # Setup the implicit function
    one = torch.ones((), dtype=P.dtype, device=P.device)
    tau = diameter / 2
    nf = torch.where(normalize, tau, one)
    implicit_function = lift_function(sag_function, nf, tau)

    # Setup the local solver
    local_solver = partial(
        implicit_surface_local_raytrace,
        implicit_function=implicit_function,
        domain_function=domain_function,
        implicit_solver=implicit_solver,
    )

    # Perform raytrace
    return surface_raytrace(P, V, tf_in, local_solver)
