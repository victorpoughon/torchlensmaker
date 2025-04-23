# This file is part of Torch Lens Maker
# Copyright (C) 2025 Victor Poughon
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

# Core
from torchlensmaker.core.physics import *
from torchlensmaker.core.outline import *
from torchlensmaker.surfaces.sphere_r import *
from torchlensmaker.core.sag_functions import *
from torchlensmaker.core.transforms import *
from torchlensmaker.core.intersect import *
from torchlensmaker.core.full_forward import *
from torchlensmaker.core.collision_detection import *
from torchlensmaker.core.parameter import *
from torchlensmaker.core.geometry import *

# Surfaces
from torchlensmaker.surfaces.conics import *
from torchlensmaker.surfaces.sphere_r import *
from torchlensmaker.surfaces.implicit_surface import *
from torchlensmaker.surfaces.local_surface import *
from torchlensmaker.surfaces.plane import *
from torchlensmaker.surfaces.implicit_cylinder import *

# Optics
from torchlensmaker.elements.kinematics import *
from torchlensmaker.optical_data import *
from torchlensmaker.optics import *
from torchlensmaker.lenses import *
from torchlensmaker.light_sources import *
from torchlensmaker.materials import *
from torchlensmaker.sampling import *

# Optimization
from torchlensmaker.optimize import *

# Viewer
import torchlensmaker.viewer.tlmviewer as viewer
from torchlensmaker.viewer.render_sequence import *


# Analysis
from torchlensmaker.analysis.plot_magnification import plot_magnification
from torchlensmaker.analysis.plot_material_model import plot_material_models
from torchlensmaker.analysis.spot_diagram import spot_diagram

# Export build123d
import torchlensmaker.export_build123d as export
from torchlensmaker.export_build123d import show_part
