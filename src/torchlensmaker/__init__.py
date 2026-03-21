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
import torch.optim as optim

from torchlensmaker.analysis.plot_magnification import plot_magnification
from torchlensmaker.analysis.plot_material_model import plot_material_models
from torchlensmaker.analysis.spot_diagram import spot_diagram
from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.geometry import (
    rotated_unit_vector,
    unit2d_rot,
    unit3d_rot,
    unit_vector,
)
from torchlensmaker.core.parameter import parameter
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.core.tensor_manip import to_tensor
from torchlensmaker.kinematics.homogeneous_geometry import (
    hom_identity,
    hom_identity_2d,
    hom_identity_3d,
    hom_target,
    hom_translate_2d,
    hom_translate_3d,
    kinematic_chain_append,
    kinematic_chain_extend,
    transform_points,
    transform_vectors,
)
from torchlensmaker.kinematics.kinematics_elements import (
    AbsolutePosition3D,
    Gap,
    KinematicElement,
    Rotate,
    Rotate2D,
    Rotate3D,
    Translate,
    Translate2D,
    Translate3D,
)
from torchlensmaker.lens.lens import Lens
from torchlensmaker.lens.lens_thickness import (
    lens_inner_thickness,
    lens_outer_thickness,
)
from torchlensmaker.lens.position_gap import (
    InnerGap,
    OuterGap,
    PositionGap,
    position_gap_to_anchors,
)
from torchlensmaker.light_sources.light_sources_elements import (
    LightSourceBase,
    Object,
    ObjectAtInfinity,
    PointSource,
    PointSourceAtInfinity,
    RaySource,
)
from torchlensmaker.light_sources.light_sources_query import (
    set_sampling2d,
    set_sampling3d,
)
from torchlensmaker.light_targets.focal_point import FocalPoint
from torchlensmaker.light_targets.image_plane import ImagePlane, linear_magnification
from torchlensmaker.materials.material_elements import (
    CauchyMaterial,
    MaterialModel,
    NonDispersiveMaterial,
    SellmeierMaterial,
)
from torchlensmaker.optical_surfaces.aperture import Aperture
from torchlensmaker.optical_surfaces.reflective_surface import ReflectiveSurface
from torchlensmaker.optical_surfaces.refractive_surface import RefractiveSurface
from torchlensmaker.optimize import (
    OptimizationRecord,
    optimize,
    plot_optimization_record,
)
from torchlensmaker.physics.physics import reflection, refraction
from torchlensmaker.sampling.sampler_elements import (
    DiskSampler2D,
    ExactSampler1D,
    ExactSampler2D,
    LinspaceSampler1D,
    LinspaceSampler2D,
    ZeroSampler1D,
    ZeroSampler2D,
)
from torchlensmaker.sampling.sampling import disk_sampling
from torchlensmaker.sequential.model_trace import ModelTrace
from torchlensmaker.sequential.sequential import Sequential, SequentialElement, SubChain
from torchlensmaker.sequential.sequential_data import SequentialData
from torchlensmaker.sequential.utils import (
    Debug,
    get_elements_by_type,
)
from torchlensmaker.surfaces.implicit_solver import implicit_solver_newton
from torchlensmaker.surfaces.sag_functions import (
    aspheric_sag_2d,
    aspheric_sag_3d,
    conical_sag_2d,
    conical_sag_3d,
    parabolic_sag_2d,
    parabolic_sag_3d,
    sag_sum_2d,
    sag_sum_3d,
    sag_to_implicit_2d,
    sag_to_implicit_3d,
    spherical_sag_2d,
    spherical_sag_3d,
    xypolynomial_sag_3d,
)
from torchlensmaker.surfaces.surface_anchor import KinematicSurface
from torchlensmaker.surfaces.surface_asphere import Asphere
from torchlensmaker.surfaces.surface_conic import Conic
from torchlensmaker.surfaces.surface_disk import Disk
from torchlensmaker.surfaces.surface_element import SurfaceElement
from torchlensmaker.surfaces.surface_parabola import Parabola
from torchlensmaker.surfaces.surface_sphere_by_curvature import SphereByCurvature
from torchlensmaker.surfaces.surface_sphere_by_radius import SphereByRadius
from torchlensmaker.surfaces.surface_square import Square
from torchlensmaker.surfaces.surface_xypolynomial import XYPolynomial
from torchlensmaker.types import (
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    BatchTensor,
    HomMatrix,
    ScalarTensor,
)
from torchlensmaker.viewer.render_model_trace import render_model_trace
from torchlensmaker.viewer.show import (
    export_json,
    render_model,
    show,
    show2d,
    show3d,
)
from torchlensmaker.viewer.tlmviewer import (
    display_scene,
    new_scene,
    render_arrows,
    render_collisions,
    render_points,
    render_rays,
    render_surface,
    render_surface_local,
)

from . import lenses, paraxial

__all__ = [
    # Analysis
    "plot_magnification",
    "plot_material_models",
    "spot_diagram",
    # Core
    "BaseModule",
    "rotated_unit_vector",
    "unit2d_rot",
    "unit3d_rot",
    "unit_vector",
    "parameter",
    "RayBundle",
    "to_tensor",
    # Kinematics
    "hom_identity",
    "hom_identity_2d",
    "hom_identity_3d",
    "hom_target",
    "hom_translate_2d",
    "hom_translate_3d",
    "kinematic_chain_append",
    "kinematic_chain_extend",
    "transform_points",
    "transform_vectors",
    "AbsolutePosition3D",
    "Gap",
    "KinematicElement",
    "Rotate",
    "Rotate2D",
    "Rotate3D",
    "Translate",
    "Translate2D",
    "Translate3D",
    # Lens
    "Lens",
    "lens_inner_thickness",
    "lens_outer_thickness",
    "InnerGap",
    "OuterGap",
    "PositionGap",
    "position_gap_to_anchors",
    # Lenses
    "lenses",
    # Light Sources
    "LightSourceBase",
    "Object",
    "ObjectAtInfinity",
    "PointSource",
    "PointSourceAtInfinity",
    "RaySource",
    "set_sampling2d",
    "set_sampling3d",
    # Light Targets
    "FocalPoint",
    "ImagePlane",
    "linear_magnification",
    # Materials
    "CauchyMaterial",
    "MaterialModel",
    "NonDispersiveMaterial",
    "SellmeierMaterial",
    # Optical Surfaces
    "Aperture",
    "ReflectiveSurface",
    "RefractiveSurface",
    # Optimization
    "OptimizationRecord",
    "optimize",
    "plot_optimization_record",
    # Paraxial
    "paraxial",
    # Physics
    "reflection",
    "refraction",
    # Sampling
    "DiskSampler2D",
    "ExactSampler1D",
    "ExactSampler2D",
    "LinspaceSampler1D",
    "LinspaceSampler2D",
    "ZeroSampler1D",
    "ZeroSampler2D",
    "disk_sampling",
    # Sequential
    "ModelTrace",
    "Sequential",
    "SequentialElement",
    "SubChain",
    "SequentialData",
    "Debug",
    "get_elements_by_type",
    # Surfaces
    "implicit_solver_newton",
    "aspheric_sag_2d",
    "aspheric_sag_3d",
    "conical_sag_2d",
    "conical_sag_3d",
    "parabolic_sag_2d",
    "parabolic_sag_3d",
    "sag_sum_2d",
    "sag_sum_3d",
    "sag_to_implicit_2d",
    "sag_to_implicit_3d",
    "spherical_sag_2d",
    "spherical_sag_3d",
    "xypolynomial_sag_3d",
    "KinematicSurface",
    "Asphere",
    "Conic",
    "Disk",
    "SurfaceElement",
    "Parabola",
    "SphereByCurvature",
    "SphereByRadius",
    "Square",
    "XYPolynomial",
    # Types
    "Batch2DTensor",
    "Batch3DTensor",
    "BatchNDTensor",
    "BatchTensor",
    "HomMatrix",
    "ScalarTensor",
    # Viewer
    "export_json",
    "render_model",
    "show",
    "show2d",
    "show3d",
    "display_scene",
    "new_scene",
    "render_arrows",
    "render_collisions",
    "render_points",
    "render_rays",
    "render_model_trace",
    "render_surface",
    "render_surface_local",
]
