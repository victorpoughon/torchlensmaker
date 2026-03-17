import torch
import torch.nn as nn

import torchlensmaker as tlm


def test_query0(dtype: torch.dtype, device: torch.device):
    optics = tlm.Sequential(
        tlm.ObjectAtInfinity(beam_diameter=10, angular_size=20, wavelength=(400, 800)),
        tlm.Gap(15),
        tlm.RefractiveSurface(
            tlm.SphereByCurvature(diameter=25, C=1 / -45.759), materials=("air", "BK7")
        ),
        tlm.Gap(3.419),
        tlm.RefractiveSurface(
            tlm.SphereByCurvature(diameter=25, C=1 / -24.887), materials=("BK7", "air")
        ),
        tlm.Gap(97.5088),
        tlm.ImagePlane(50),
    )

    optics.set_sampling2d(pupil=10, field=5, wavel=3)
    outputs = optics.raytrace(dim=2)

    scene = tlm.render_sequence(optics, dim=2)

    tlm.show2d(optics)

    tlm.set_sampling2d(optics, pupil=10, field=5, wavel=3)
    tlm.set_sampling3d(optics, pupil=100, field=10, wavel=3)
