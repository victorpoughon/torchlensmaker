import torch
import torch.nn as nn
import torchlensmaker as tlm


def test_elements0():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(torch.device("cpu"))

    optics = tlm.Sequential(
        tlm.Gap(15),
    )

    sampling = {"base": 10, "object": 5, "wavelength": 10}
    outputs = optics(tlm.default_input(dim=2, dtype=torch.float64, sampling=sampling))
    scene = tlm.render_sequence(optics, dim=2, dtype=torch.float64, sampling=sampling)


def test_elements1():
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


def test_elements2():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(torch.device("cpu"))

    S = 5
    R = S / 2
    A = 30

    optics = tlm.Sequential(
        tlm.Rotate2D(20),
        tlm.RaySource(material="air", wavelength=(400, 800)),
        tlm.Gap(10),
        tlm.Rotate2D(-20),
        tlm.SubChain(
            tlm.Rotate2D(-A),
            tlm.RefractiveSurface(tlm.CircularPlane(S), material="K5"),
        ),
        tlm.Gap(R),
        tlm.SubChain(
            tlm.Rotate2D(A),
            tlm.RefractiveSurface(tlm.CircularPlane(S), material="air"),
        ),
    )

    optics.to(dtype=torch.float64)

    sampling = {"wavelength": 10}
    output = optics(tlm.default_input(dim=2, dtype=torch.float64, sampling=sampling))
    scene = tlm.render_sequence(optics, dim=2, dtype=torch.float64, sampling=sampling)


def test_rainbow():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(torch.device("cpu"))

    # Use half spheres to model interface boundaries
    radius = 5
    halfsphere = tlm.SphereR(diameter=2 * radius, R=radius)

    optics = tlm.Sequential(
        # Position the light source just above the optical axis
        tlm.SubChain(
            tlm.Translate(y=5.001),
            tlm.ObjectAtInfinity(10, 0.5, wavelength=(400, 660)),
        ),
        # Move the droplet of water some distance away from the source
        tlm.Gap(50),
        # First interface: half sphere (pointing left), refractive air to water
        tlm.RefractiveSurface(
            halfsphere, material="water", anchors=("extent", "extent")
        ),
        # Second interface: half sphere (pointing right), reflective
        tlm.SubChain(
            tlm.Rotate((-180, 0)),
            tlm.ReflectiveSurface(halfsphere, anchors=("extent", "extent")),
        ),
        # Third interface: half sphere (pointing down), refractive water to air
        tlm.SubChain(
            tlm.Rotate((60, 0)),
            tlm.RefractiveSurface(
                halfsphere, material="air", anchors=("extent", "origin")
            ),
        ),
    )

    optics.to(dtype=torch.float64)

    sampling = {"base": 3, "object": 2, "wavelength": 3}
    output = optics(tlm.default_input(dim=2, dtype=torch.float64, sampling=sampling))
    scene = tlm.render_sequence(optics, dim=2, dtype=torch.float64, sampling=sampling)


def test_elements3():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(torch.device("cpu"))

    optics = tlm.Sequential(
        tlm.Translate(y=5.001),
        tlm.ObjectAtInfinity(10, 0.5),
    )

    optics.to(dtype=torch.float64)

    sampling = {"base": 10, "object": 5, "wavelength": 10}
    output = optics(tlm.default_input(dim=2, dtype=torch.float64, sampling=sampling))
    scene = tlm.render_sequence(optics, dim=2, dtype=torch.float64, sampling=sampling)



def test_elements3d():
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

    optics.to(dtype=torch.float64)

    sampling = {"base": 10, "object": 5, "wavelength": 10}
    output = optics(tlm.default_input(dim=3, dtype=torch.float64, sampling=sampling))
    scene = tlm.render_sequence(optics, dim=3, dtype=torch.float64, sampling=sampling)
