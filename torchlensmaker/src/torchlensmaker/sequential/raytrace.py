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

import torch

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.sequential.optical_trace import OpticalTrace
from torchlensmaker.types import Tf


def raytrace(
    model: BaseModule,
    dim: int,
    *,
    input_trace: OpticalTrace | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> OpticalTrace:
    """
    Evaluate an optical model into an optical trace

    The model's light sources emit the initial ray bundles. Use this for
    self-contained models that include their own light source.

    An input_trace can also be provided, in which case the output rays and
    output tf are used as input. If not provided start from an empty bundle
    and an identity root transform.

    Args:
        model: the optical model to evaluate
        dim: spatial dimension (2 or 3)
        input_trace: optional input optical trace
        dtype: floating-point dtype; defaults to torch default
        device: compute device; defaults to torch default

    Returns:
        OpticalTrace containing all intermediate and final ray data
    """
    if input_trace is None:
        trace = OpticalTrace.empty(dim, dtype, device)
    else:
        trace = OpticalTrace.from_inputs(
            input_trace.output_rays(), input_trace.output_tf()
        )

    model.trace(trace, "", "_root")
    return trace


def raytrace_with_inputs(
    model: BaseModule,
    rays: RayBundle,
    tf: Tf,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> OpticalTrace:
    """
    Evaluate an optical model into an optical trace

    Use this when the source rays are computed externally and the model is a
    sub-chain that does not include its own light source (e.g. just optics and
    a target).

    Args:
        model: the optical model to evaluate
        rays: input ray bundle placed at the root of the trace
        tf: kinematic transform at the root of the trace
        dtype: floating-point dtype; defaults to torch default
        device: compute device; defaults to torch default

    Returns:
        OpticalTrace containing all intermediate and final ray data
    """
    trace = OpticalTrace.from_inputs(rays, tf)
    model.trace(trace, "", "_root")
    return trace
