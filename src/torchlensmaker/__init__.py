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

######
# Core
######

from torchlensmaker.core.physics import reflection, refraction
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

############
# Kinematics
############

from torchlensmaker.kinematics.homogeneous_geometry import (
    HomMatrix2D,
    HomMatrix3D,
    HomMatrix,
    transform_points,
    transform_vectors,
    hom_identity,
    hom_identity_2d,
    hom_identity_3d,
)

from torchlensmaker.kinematics.kinematics_elements import (
    KinematicElement,
    MixedDimKinematic,
    AbsolutePosition,
    Gap,
    Rotate3D,
    Rotate2D,
    Translate2D,
    Translate3D,
    Rotate,
    Translate,
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

from torchlensmaker.elements.sequential import Sequential, SubChain, SequentialElement
from torchlensmaker.elements.optical_surfaces import (
    CollisionSurface,
    ReflectiveSurface,
    RefractiveSurface,
    Aperture,
    ImagePlane,
    linear_magnification,
)
from torchlensmaker.elements.light_sources import (
    LightSourceBase,
    RaySource,
    PointSourceAtInfinity,
    PointSource,
    ObjectAtInfinity,
    Object,
)
from torchlensmaker.elements.focal_point import FocalPoint

# Top level stuff - to be reorganized
from torchlensmaker.optical_data import OpticalData, default_input
from torchlensmaker.lenses import LensBase, BiLens, Lens, PlanoLens
from torchlensmaker.materials import (
    MaterialModel,
    NonDispersiveMaterial,
    CauchyMaterial,
    SellmeierMaterial,
)

##########
# Sampling
##########

from torchlensmaker.sampling import (
    dense,
    random_uniform,
    random_normal,
    exact,
    init_sampling,
    Sampler,
)

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

import torchlensmaker.viewer.tlmviewer as viewer
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

##################
# Export build123d
##################

import torchlensmaker.export_build123d as export
from torchlensmaker.export_build123d import show_part

# Double precision is the better choice for most optical applications
torch.set_default_dtype(torch.float64)
