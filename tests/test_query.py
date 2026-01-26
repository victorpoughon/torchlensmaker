import torch
import torch.nn as nn
import torchlensmaker as tlm


def test_query0():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(torch.device("cpu"))

    optics = tlm.Sequential(
        tlm.ObjectAtInfinity(beam_diameter=10, angular_size=20, wavelength=(400, 800)),
        tlm.Gap(15),
        tlm.RefractiveSurface(tlm.Sphere(diameter=25, R=-45.759), material="BK7"),
        tlm.Gap(3.419),
        tlm.RefractiveSurface(tlm.Sphere(diameter=25, R=-24.887), material="air"),
        tlm.Gap(97.5088),
        tlm.ImagePlane(50),
    )

    optics.set_sampling2d(pupil=10, field=5, wavelength=3)
    outputs = optics(tlm.default_input(dim=2, dtype=torch.float64))

    scene = tlm.render_sequence(optics, dim=2, dtype=torch.float64)
    
    tlm.show2d(optics)
    
    tlm.set_sampling2d(optics, pupil=10, field=5, wavelength=3)
    tlm.set_sampling3d(optics, pupil=100, field=10, wavelength=3)

    # Query
    print(tlm.get_light_source2d(optics))
    print(tlm.get_light_source3d(optics))