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
    HomMatrix2D,
    HomMatrix3D,
    HomMatrix,
)

######
# Core
######

from torchlensmaker.core.dim import Dim
from torchlensmaker.physics.physics import reflection, refraction
from torchlensmaker.core.intersect import intersect
from torchlensmaker.core.full_forward import forward_tree, full_forward

from torchlensmaker.core.parameter import parameter
from torchlensmaker.core.geometry import (
    unit_vector,
    rotated_unit_vector,
    unit2d_rot,
    unit3d_rot,
)
from torchlensmaker.core.sag_functions import (
    Spherical,
    Parabolic,
    Aspheric,
    XYPolynomial,
    XYPolynomialN,
    Conical,
    SagSum,
    SagFunction,
)
from torchlensmaker.core.tensor_manip import to_tensor

############
# Kinematics
############

from torchlensmaker.kinematics.homogeneous_geometry import (
    transform_points,
    transform_vectors,
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
    ExactKinematicElement2D,
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

###################
# Implicit surfaces
###################

from torchlensmaker.implicit_surfaces.implicit_function import (
    sag_to_implicit_2d,
    sag_to_implicit_3d,
)
from torchlensmaker.implicit_surfaces.sag import (
    spherical_sag_2d,
    spherical_sag_3d,
    parabolic_sag_2d,
    parabolic_sag_3d,
    conical_sag_2d,
    conical_sag_3d,
    aspheric_sag_2d,
    aspheric_sag_3d,
    xypolynomial_sag_3d,
)

##########
# Surfaces
##########

from torchlensmaker.surfaces.sphere_r import SphereR
from torchlensmaker.surfaces.conics import Sphere, Parabola, Conic, Asphere
from torchlensmaker.surfaces.implicit_surface import ImplicitSurface
from torchlensmaker.surfaces.local_surface import LocalSurface
from torchlensmaker.surfaces.plane import Plane, CircularPlane, SquarePlane
from torchlensmaker.surfaces.implicit_cylinder import ImplicitCylinder
from torchlensmaker.surfaces.sag_surface import SagSurface

##################
# Optical elements
##################

from torchlensmaker.elements.sequential import (
    Sequential,
    SubChain,
    SequentialElement,
)
from torchlensmaker.elements.optical_surfaces import (
    CollisionSurface,
    ReflectiveSurface,
    RefractiveSurface,
    Aperture,
    ImagePlane,
    linear_magnification,
)
from torchlensmaker.light_sources.light_sources_elements import (
    RaySource,
    RaySource2D,
    RaySource3D,
    PointSourceAtInfinity,
    PointSourceAtInfinity2D,
    PointSourceAtInfinity3D,
    PointSource,
    PointSource2D,
    PointSource3D,
    ObjectAtInfinity,
    ObjectAtInfinity2D,
    ObjectAtInfinity3D,
    Object,
    Object2D,
    Object3D,
    LightSourceBase,
)
from torchlensmaker.light_sources.light_sources_query import (
    set_sampling2d,
    set_sampling3d,
)
from torchlensmaker.elements.focal_point import FocalPoint

# Top level stuff - to be reorganized
from torchlensmaker.optical_data import OpticalData, default_input
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
