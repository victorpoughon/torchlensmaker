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
from jaxtyping import Float

import torchlensmaker as tlm
from torchlensmaker.core.tensor_manip import default_dtype_device
from torchlensmaker.kinematics.homogeneous_geometry import hom_identity_2d
from torchlensmaker.sequential.optical_trace import OpticalTrace
from torchlensmaker.sequential.raytrace import raytrace, raytrace_with_inputs


def equivalent_locus_2d(
    Pa: Float[torch.Tensor, "N 2"],
    Va: Float[torch.Tensor, "N 2"],
    Pb: Float[torch.Tensor, "N 2"],
    Vb: Float[torch.Tensor, "N 2"],
) -> Float[torch.Tensor, "N 2"]:
    """
    Equivalent refracting locus in 2D

    Given two ray bundles of the same size, compute the set of intersection
    points. Will error if the rays don't intersect.

    Args:
        Input rays :: (N, 2) and (N, 2)
        Output rays :: (N, 2) and (N, 2)

    Returns:
        (t,) :: (N, 2)
        Pair solution to (Pa + ta * Va) + (Pb + tb * Vb) = 0

        To obtain actual collision points, do either of:
            Pa + t[:, 0].unsqueeze(-1)*Va
            Pb + -t[:, 1].unsqueeze(-1)*Vb
    """

    # V is the (N, 2, 2) matrix such that Vt = P
    V = torch.stack((Va, Vb), dim=-1)
    P = Pb - Pa

    # Solve for t
    return torch.linalg.solve(V, P)


def fit_parabola_vertex_2d(
    x: Float[torch.Tensor, " N"], y: Float[torch.Tensor, " N"]
) -> Float[torch.Tensor, ""]:
    """
    Find the vertex of a set of points by fitting a parabola X = AY + C

    Find the X coordinate at which a set of 2D point crosses the X axis
    by fitting a parabolic model X = AY + C using least square and returning C
    """
    assert x.dim() == 1, y.dim() == 1
    assert x.numel() == y.numel()

    N = x.shape[0]
    sx = x.sum()
    sy4 = (y**4).sum()
    sy2 = (y**2).sum()
    sy2x = (y**2 * x).sum()

    num = sx * (sy4 / sy2) - sy2x
    denom = N * (sy4 / sy2) - sy2

    return num / denom


def principal_point_with_light_source(
    lens: tlm.Lens, light_source: tlm.LightSourceBase
) -> Float[torch.Tensor, ""]:
    # Infer dtype, device from the first gap element
    dtype, device = lens.dtype, lens.device

    # Evaluate the light source and the model to get output rays
    input_tf = hom_identity_2d(dtype, device)
    input_rays = light_source(input_tf.direct)
    trace = OpticalTrace.from_inputs(input_rays, input_tf)
    lens.trace(trace, "", "_root")
    outputs_rays = trace.output_rays()

    # Compute intersection locus of input and output ray bundles
    t = equivalent_locus_2d(input_rays.P, input_rays.V, outputs_rays.P, outputs_rays.V)
    collision_points = input_rays.points_at(t[:, 0])

    # Fit a parabola to the locus surface to obtain vertex
    return fit_parabola_vertex_2d(collision_points[:, 0], collision_points[:, 1])


def rear_principal_point(
    lens: tlm.Lens,
    wavelength: float,
    alpha: float = 0.05,
    beta: float = 0.01,
    pupil_samples=30,
) -> Float[torch.Tensor, ""]:
    """
    Rear principal point of a lens

    Compute the rear principal point of a lens by fitting a parabolic model to
    the equivalent refracting locus.

    Args:
        lens: the tlm.Lens model
        wavelength: the wavelength to use
        alpha: alpha * D is the diameter of the light source used to compute the principal plane
               where D is the minimal diameter of the lens
        beta:  beta * D is the diameter of the reverse aperture used to block
               rays too close to the optical axis
        pupil_samples: number of samples

    Returns:
        a scalar tensor that contains the X coordinate of the rear principal plane
    """

    # TODO implement beta

    # Get the lens minimal diameter and setup the light source
    mdiam = lens.minimal_diameter()
    source = tlm.PointSourceAtInfinity(
        alpha * mdiam,
        sampler_pupil_2d=tlm.LinspaceSampler1D(pupil_samples),
        sampler_wavel_2d=tlm.ZeroSampler1D(),
        wavelength=wavelength,
    )

    return principal_point_with_light_source(lens, source)


def front_principal_point(
    lens: tlm.Lens,
    wavelength: float,
    alpha: float = 0.05,
    beta: float = 0.01,
    pupil_samples=30,
) -> Float[torch.Tensor, ""]:
    """
    Front principal point of a lens

    Compute the front principal point of a lens by fitting a parabolic model to
    the equivalent refracting locus.

    Args:
        lens: the tlm.Lens model
        wavelength: the wavelength to use
        alpha: alpha * D is the diameter of the light source used to compute the principal plane
               where D is the minimal diameter of the lens
        beta:  beta * D is the diameter of the reverse aperture used to block
               rays too close to the optical axis
        pupil_samples: number of samples

    Returns:
        a scalar tensor that contains the X coordinate of the front principal plane
    """

    # TODO implement beta

    # Get the lens minimal diameter and setup the light source
    mdiam = lens.minimal_diameter()
    source = tlm.PointSourceAtInfinity(
        alpha * mdiam,
        sampler_pupil_2d=tlm.LinspaceSampler1D(pupil_samples),
        sampler_wavel_2d=tlm.ZeroSampler1D(),
        wavelength=wavelength,
    )

    return principal_point_with_light_source(lens.reverse(), source.reverse())


def focal_point_with_light_source(
    lens: tlm.Lens,
    light_source: tlm.BaseModule,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Float[torch.Tensor, ""]:
    """
    Compute focal point with a paraxial light source
    """

    dtype, device = default_dtype_device(dtype, device)

    # Evaluate the light source and the model to get output ray

    trace_light_source = raytrace(light_source, dim=2)
    input_rays = trace_light_source.output_rays()

    root_tf = tlm.hom_identity(dim=2)
    trace_lens = raytrace_with_inputs(lens, input_rays, root_tf)
    output_rays = trace_lens.output_rays()

    assert input_rays.P.shape == output_rays.P.shape == (1, 2)

    # Compute output ray intersection with the optical axis
    t = -output_rays.P[:, 1] / output_rays.V[:, 1]
    return output_rays.P[:, 0] + t * output_rays.V[:, 0]


def rear_focal_point(
    lens: tlm.Lens,
    wavelength: float,
    h: float,
) -> Float[torch.Tensor, ""]:
    """
    Compute rear focal length of a lens using a paraxial ray

    Args:
        lens: the tlm.Lens model
        wavelength: the wavelength to use
        h: height of the paraxial ray, normalized to the lens minimal diameter

    Returns:
        a scalar tensor that contains the X coordinate of the rear focal length
    """

    # Get the lens minimal diameter and setup the light source
    mdiam = tlm.lens_minimal_diameter(lens)
    light_source = tlm.SubChain(
        tlm.Translate2D(y=h * mdiam),
        tlm.RaySource(
            wavelength=wavelength,
            sampler_wavel_2d=tlm.ZeroSampler1D(),
        ),
    )

    return focal_point_with_light_source(lens, light_source)


def front_focal_point(
    lens: tlm.Lens,
    wavelength: float,
    h: float,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Float[torch.Tensor, ""]:
    """
    Compute front focal length of a lens using a paraxial ray

    Args:
        lens: the tlm.Lens model
        wavelength: the wavelength to use
        h: height of the paraxial ray, normalized to the lens minimal diameter

    Returns:
        a scalar tensor that contains the X coordinate of the front focal length
    """

    # Get the lens minimal diameter and setup the light source
    mdiam = lens.minimal_diameter()
    light_source = tlm.SubChain(
        tlm.Translate2D(y=h * mdiam),
        tlm.RaySource(
            wavelength=wavelength,
            sampler_wavel_2d=tlm.ZeroSampler1D(),
        ).reverse(),
    )

    return focal_point_with_light_source(lens.reverse(), light_source, dtype, device)
