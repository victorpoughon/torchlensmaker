# Core
from torchlensmaker.physics import *
from torchlensmaker.outline import *
from torchlensmaker.surfaces import *
from torchlensmaker.transforms import *
from torchlensmaker.intersect import *

# Optics
from torchlensmaker.optical_data import *
from torchlensmaker.optics import *
from torchlensmaker.lenses import *
from torchlensmaker.light_sources import *

from torchlensmaker.materials import *

# Optimization
from torchlensmaker.parameter import parameter
from torchlensmaker.full_forward import *
from torchlensmaker.optimize import optimize

# Viewer
import torchlensmaker.viewer as viewer
from torchlensmaker.viewer import show, show2d, show3d, export_json

from torchlensmaker.sampling import *

# Analysis
from torchlensmaker.analysis.plot_magnification import plot_magnification
from torchlensmaker.analysis.plot_material_model import plot_material_models
from torchlensmaker.analysis.spot_diagram import spot_diagram

# Export build123d
import torchlensmaker.export_build123d as export

__all__ = [
    "viewer",
    "export",
    "show",
    "show2d",
    "show3d",
    "export_json",
    "full_forward",
    "parameter",
]
