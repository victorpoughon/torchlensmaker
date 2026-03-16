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

# ruff: noqa: F401
# ruff: noqa: I001

import torch

#######
# Types
#######

from torchlensmaker.types import (
    ScalarTensor,
    BatchTensor,
    Batch2DTensor,
    Batch3DTensor,
    BatchNDTensor,
    HomMatrix,
    Direction,
)

######
# Core
######

from torchlensmaker.physics.physics import reflection, refraction

from torchlensmaker.core.parameter import parameter
from torchlensmaker.core.geometry import (
    unit_vector,
    rotated_unit_vector,
    unit2d_rot,
    unit3d_rot,
)
from torchlensmaker.core.tensor_manip import to_tensor
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.core.base_module import BaseModule

############
# Kinematics
############

from torchlensmaker.kinematics.homogeneous_geometry import (
    transform_points,
    transform_vectors,
    hom_target,
    hom_identity,
    hom_identity_2d,
    hom_identity_3d,
    hom_translate_2d,
    hom_translate_3d,
    kinematic_chain_append,
    kinematic_chain_extend,
)

from torchlensmaker.kinematics.kinematics_elements import (
    KinematicElement,
    AbsolutePosition3D,
    Gap,
    Rotate3D,
    Rotate2D,
    Translate2D,
    Translate3D,
    Rotate,
    Translate,
)

##########
# Samplers
##########
from torchlensmaker.sampling.sampler_elements import (
    ZeroSampler1D,
    ZeroSampler2D,
    LinspaceSampler1D,
    LinspaceSampler2D,
    DiskSampler2D,
    ExactSampler1D,
    ExactSampler2D,
)
from torchlensmaker.sampling.sampling import disk_sampling

##########
# Surfaces
##########

from torchlensmaker.surfaces.surface_element import SurfaceElement
from torchlensmaker.surfaces.sag_functions import (
    sag_to_implicit_2d,
    sag_to_implicit_3d,
    spherical_sag_2d,
    spherical_sag_3d,
    parabolic_sag_2d,
    parabolic_sag_3d,
    conical_sag_2d,
    conical_sag_3d,
    aspheric_sag_2d,
    aspheric_sag_3d,
    xypolynomial_sag_3d,
    sag_sum_2d,
    sag_sum_3d,
)
from torchlensmaker.surfaces.implicit_solver import implicit_solver_newton
from torchlensmaker.surfaces.surface_sphere_by_curvature import SphereByCurvature
from torchlensmaker.surfaces.surface_sphere_by_radius import SphereByRadius
from torchlensmaker.surfaces.surface_parabola import Parabola
from torchlensmaker.surfaces.surface_disk import Disk
from torchlensmaker.surfaces.surface_conic import Conic
from torchlensmaker.surfaces.surface_asphere import Asphere
from torchlensmaker.surfaces.surface_xypolynomial import XYPolynomial
from torchlensmaker.surfaces.surface_square import Square
from torchlensmaker.surfaces.surface_anchor import KinematicSurface

##################
# Optical elements
##################

from torchlensmaker.optical_surfaces.reflective_surface import ReflectiveSurface
from torchlensmaker.optical_surfaces.refractive_surface import RefractiveSurface
from torchlensmaker.optical_surfaces.aperture import Aperture
from torchlensmaker.optical_surfaces.image_plane import ImagePlane, linear_magnification

from torchlensmaker.elements.sequential import (
    Sequential,
    SubChain,
    SequentialElement,
    Reversed,
)
from torchlensmaker.light_sources.light_sources_elements import (
    RaySource,
    PointSourceAtInfinity,
    PointSource,
    ObjectAtInfinity,
    Object,
    LightSourceBase,
)
from torchlensmaker.light_sources.light_sources_query import (
    set_sampling2d,
    set_sampling3d,
)
from torchlensmaker.elements.focal_point import FocalPoint

# Top level stuff - to be reorganized
from torchlensmaker.elements.sequential_data import SequentialData
from torchlensmaker.materials.material_elements import (
    MaterialModel,
    NonDispersiveMaterial,
    CauchyMaterial,
    SellmeierMaterial,
)

from torchlensmaker.elements.utils import (
    Debug,
    get_elements_by_type,
)

######
# Lens
######

from torchlensmaker.lens.position_gap import (
    position_gap_to_anchors,
    PositionGap,
    InnerGap,
    OuterGap,
)
from torchlensmaker.lens.lens import Lens
from torchlensmaker.lens.lens_thickness import (
    lens_inner_thickness,
    lens_outer_thickness,
)

########
# Lenses
########
from . import lenses

##########
# Paraxial
##########

from . import paraxial

##############
# Optimization
##############

import torch.optim as optim
from torchlensmaker.optimize import (
    optimize,
    OptimizationRecord,
    plot_optimization_record,
)

########
# Viewer
########

from torchlensmaker.viewer.tlmviewer import (
    render_rays,
    render_points,
    render_collisions,
    render_arrows,
    render_surface,
    render_surface_local,
    new_scene,
    display_scene,
)
from torchlensmaker.viewer.render_sequence import (
    show,
    show2d,
    show3d,
    export_json,
    render_sequence,
    ForwardArtist,
)

##########
# Analysis
##########

from torchlensmaker.analysis.plot_magnification import plot_magnification
from torchlensmaker.analysis.plot_material_model import plot_material_models
from torchlensmaker.analysis.spot_diagram import spot_diagram
