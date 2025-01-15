from torchlensmaker.optics import (
    FocalPoint,
    Gap,
    RefractiveSurface,
    ReflectiveSurface,
    Aperture,
    PointSource,
    PointSourceAtInfinity,
    OpticalSurface,
    default_input,
    ObjectAtInfinity,
    Image,
    ImagePlane,
)

from torchlensmaker.shapes import (
    BaseShape,
    BezierSpline,
    CircularArc,
    Line,
    Parabola,
    PiecewiseLine,
)

from torchlensmaker.training import (
    optimize,
)

from torchlensmaker.export3d import (
    lens_to_part,
)

from torchlensmaker.surface import (
    Surface,
)

from torchlensmaker.render_plt import render_plt

from torchlensmaker.module import Module

from torchlensmaker.lenses import (
    AsymmetricLens,
    SymmetricLens,
    PlanoLens,
)

from torchlensmaker.torch_extensions import (
    full_forward,
    OpticalSequence,
    Parameter,
)

from torchlensmaker.plot_magnification import plot_magnification


## new 3D stuff

from torchlensmaker.outline import *
import torchlensmaker.surfaces as surfaces
from torchlensmaker.transforms3D import *
from torchlensmaker.transforms2D import *
from torchlensmaker.physics import *

import torchlensmaker.tlmviewer.tlmviewer as viewer
from torchlensmaker.intersect import *
