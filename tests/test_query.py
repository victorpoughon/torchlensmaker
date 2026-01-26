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

    sampling = {"base": 10, "object": 5, "wavelength": 10}
    outputs = optics(tlm.default_input(dim=2, dtype=torch.float64, sampling=sampling))
    scene = tlm.render_sequence(optics, dim=2, dtype=torch.float64, sampling=sampling)

    # Query
    # print(optics)

    source2d = tlm.get_light_source2d(optics)
    source3d = tlm.get_light_source3d(optics)

    print(source2d)
    print(source3d)
