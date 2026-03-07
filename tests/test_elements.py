import torch
import torch.nn as nn
import torchlensmaker as tlm


def test_elements0():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(torch.device("cpu"))

    optics = tlm.Sequential(
        tlm.Gap(15),
    )

    optics.set_sampling2d(pupil=10, field=5, wavel=1)
    outputs = optics(tlm.default_input(dim=2, dtype=torch.float64))
    scene = tlm.render_sequence(optics, dim=2, dtype=torch.float64)


def test_elements1():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(torch.device("cpu"))

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

    optics.set_sampling2d(pupil=10, field=5, wavel=10)
    outputs = optics(tlm.default_input(dim=2, dtype=torch.float64))
    scene = tlm.render_sequence(optics, dim=2, dtype=torch.float64)


def test_elements2():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(torch.device("cpu"))

    S = 5
    R = S / 2
    A = 30

    optics = tlm.Sequential(
        tlm.Rotate2D(20),
        tlm.RaySource(wavelength=(400, 800)),
        tlm.Gap(10),
        tlm.Rotate2D(-20),
        tlm.SubChain(
            tlm.Rotate2D(-A),
            tlm.RefractiveSurface(tlm.Disk(S), materials=("air", "K5")),
        ),
        tlm.Gap(R),
        tlm.SubChain(
            tlm.Rotate2D(A),
            tlm.RefractiveSurface(tlm.Disk(S), materials=("K5", "air")),
        ),
    )

    optics.to(dtype=torch.float64)

    optics.set_sampling2d(wavel=10)
    output = optics(tlm.default_input(dim=2, dtype=torch.float64))
    scene = tlm.render_sequence(optics, dim=2, dtype=torch.float64)


def test_rainbow():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(torch.device("cpu"))

    # Use half spheres to model interface boundaries
    radius = 5
    halfsphere = tlm.SphereByRadius(diameter=2 * radius, R=radius)

    optics = tlm.Sequential(
        # Position the light source just above the optical axis
        tlm.SubChain(
            tlm.Translate(y=5.001),
            tlm.ObjectAtInfinity(10, 0.5, wavelength=(400, 660)),
        ),
        # Move the droplet of water some distance away from the source
        tlm.Gap(50),
        # First interface: half sphere (pointing left), refractive air to water
        tlm.RefractiveSurface(halfsphere.clone(anchors=(1, 1)), materials=("air", "water")),
        # Second interface: half sphere (pointing right), reflective
        tlm.SubChain(
            tlm.Rotate((-180, 0)),
            tlm.ReflectiveSurface(halfsphere.clone(anchors=(1, 1))),
        ),
        # Third interface: half sphere (pointing down), refractive water to air
        tlm.SubChain(
            tlm.Rotate((60, 0)),
            tlm.RefractiveSurface(halfsphere.clone(anchors=(1, 0)), materials=("water", "air")),
        ),
    )

    optics.to(dtype=torch.float64)

    optics.set_sampling2d(pupil=3, field=2, wavel=3)
    output = optics(tlm.default_input(dim=2, dtype=torch.float64))
    scene = tlm.render_sequence(optics, dim=2, dtype=torch.float64)


def test_elements3():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(torch.device("cpu"))

    optics = tlm.Sequential(
        tlm.Translate(y=5.001),
        tlm.ObjectAtInfinity(10, 0.5),
    )

    optics.to(dtype=torch.float64)

    optics.set_sampling2d(pupil=10, field=5, wavel=10)
    output = optics(tlm.default_input(dim=2, dtype=torch.float64))
    scene = tlm.render_sequence(optics, dim=2, dtype=torch.float64)


def test_elements3d():
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(torch.device("cpu"))

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

    optics.to(dtype=torch.float64)

    optics.set_sampling2d(pupil=10, field=5, wavel=10)
    output = optics(tlm.default_input(dim=3, dtype=torch.float64))
    scene = tlm.render_sequence(optics, dim=3, dtype=torch.float64)


def test_cooke():
    d1, d2 = 30, 25

    r1 = tlm.SphereByCurvature(d1, 1 / 26.4)
    r2 = tlm.SphereByCurvature(d1, 1 / -150.7)
    r3 = tlm.SphereByCurvature(d2, 1 / -29.8)
    r4 = tlm.SphereByCurvature(d2, 1 / 24.2)
    r5 = tlm.SphereByCurvature(d1, 1 / 150.7)
    r6 = tlm.SphereByCurvature(d1, 1 / -26.4)

    material1 = tlm.NonDispersiveMaterial(1.5108)
    material2 = tlm.NonDispersiveMaterial(1.6042)

    L1 = tlm.lenses.singlet(r1, tlm.InnerGap(5.9), r2, material=material1)
    L2 = tlm.lenses.singlet(r3, tlm.InnerGap(0.2), r4, material=material2)
    L3 = tlm.lenses.singlet(r5, tlm.InnerGap(5.9), r6, material=material1)

    focal = tlm.parameter(85)

    optics = tlm.Sequential(
        tlm.ObjectAtInfinity(15, 25),
        L1,
        tlm.Gap(10.9),
        L2,
        tlm.Gap(3.1),
        tlm.Aperture(18),
        tlm.Gap(9.4),
        L3,
        tlm.Gap(focal),
        tlm.ImagePlane(65),
    )

    tlm.show2d(optics, wavelength=3)

    optics.set_sampling3d(pupil=100, wavel=4)
    tlm.show3d(optics)
    # f, _ = tlm.spot_diagram(optics, sampling=sampling, row="object", figsize=(12, 12))


def test_nolens():
    surface = tlm.Parabola(diameter=15, A=tlm.parameter(0.02))  # y = a*x^2

    optics = tlm.Sequential(
        tlm.PointSourceAtInfinity(beam_diameter=18.5),
        tlm.Gap(10),
        tlm.RefractiveSurface(surface.clone(anchors=(0, 1)), materials=("air", "water")),
        tlm.Gap(2),
        tlm.RefractiveSurface(
            surface.clone(
                anchors=(0, 1),
                scale=-1,
            ),
            materials=("water", "air"),
        ),
        tlm.Gap(50),
        tlm.FocalPoint(),
    )

    tlm.show(optics, dim=2)
    tlm.show(optics, dim=3)


def test_surface_reuse() -> None:
    surface = tlm.Parabola(diameter=15, A=-0.05)

    optics = tlm.Sequential(
        # tlm.PointSourceAtInfinity(beam_diameter=18.5),
        tlm.Gap(10),
        tlm.RefractiveSurface(surface, materials=("air", "water")),
        tlm.Gap(2),
        tlm.RefractiveSurface(surface, materials=("air", "water")),
        tlm.Gap(1),
        tlm.RefractiveSurface(surface, materials=("air", "water")),
        tlm.Gap(1),
        tlm.RefractiveSurface(surface, materials=("air", "water")),
    )

    tlm.show(optics, dim=2)
    tlm.show(optics, dim=3)


def test_elements_reuse() -> None:
    surface = tlm.SphereByCurvature(10, 1 / 50)
    material = tlm.NonDispersiveMaterial(1.5108)
    lens = tlm.lenses.symmetric_singlet(surface, tlm.InnerGap(5.9), material=material)
    gap = tlm.Gap(10)

    manual_lens = tlm.Sequential(
        tlm.RefractiveSurface(surface, materials=("air", "water")),
        tlm.Gap(2),
        tlm.RefractiveSurface(surface, materials=("air", "water")),
    )

    optics = tlm.Sequential(
        # tlm.PointSourceAtInfinity(beam_diameter=18.5),
        gap,
        lens,
        gap,
        manual_lens,
        gap,
        manual_lens,
        gap,
        lens,
        gap,
        lens,
    )

    print(optics)

    print(lens.inner_thickness())
    print(lens.outer_thickness())
    print(lens.minimal_diameter())
    print(tlm.paraxial.rear_focal_point(lens, 500, 0.01))

    tlm.show(optics, dim=2)
    tlm.show(optics, dim=3)
