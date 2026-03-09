import torch
import torch.nn as nn
import torchlensmaker as tlm


def test_show_dtype_device(dtype: torch.dtype, device: torch.device):
    print("device", device)
    print("dtype", dtype)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())


def assert_ray_bundle_dtype(rays: tlm.RayBundle, expected_dtype: torch.dtype) -> None:
    assert rays["P"].dtype == expected_dtype
    assert rays["V"].dtype == expected_dtype
    assert rays["pupil"].dtype == expected_dtype
    assert rays["field"].dtype == expected_dtype
    assert rays["wavel"].dtype == expected_dtype
    assert rays["pupil_idx"].dtype == torch.int64
    assert rays["field_idx"].dtype == torch.int64
    assert rays["wavel_idx"].dtype == torch.int64


def assert_ray_bundle_device(rays: tlm.RayBundle, expected_device: torch.dtype) -> None:
    assert rays["P"].device == expected_device
    assert rays["V"].device == expected_device
    assert rays["pupil"].device == expected_device
    assert rays["field"].device == expected_device
    assert rays["wavel"].device == expected_device
    assert rays["pupil_idx"].device == expected_device
    assert rays["field_idx"].device == expected_device
    assert rays["wavel_idx"].device == expected_device


def check_sample_and_render_2d(
    optics: tlm.BaseModule, expected_dtype: torch.dtype, expected_device: torch.device
) -> None:
    # Sample and render in 2D
    optics.set_sampling2d(pupil=10, field=5, wavel=2)
    outputs_2d = optics(tlm.default_input(dim=2))
    _ = tlm.render_sequence(optics, dim=2)

    # Check dtype, device
    assert_ray_bundle_dtype(outputs_2d.rays, expected_dtype)
    assert_ray_bundle_device(outputs_2d.rays, expected_device)


def check_sample_and_render_3d(
    optics: tlm.BaseModule, expected_dtype: torch.device, expected_device: torch.device
) -> None:
    # Sample and render in 3D
    optics.set_sampling3d(pupil=10, field=5, wavel=2)
    outputs_3d = optics(tlm.default_input(dim=3))
    _ = tlm.render_sequence(optics, dim=3)

    # Check dtype, device
    assert_ray_bundle_dtype(outputs_3d.rays, expected_dtype)
    assert_ray_bundle_device(outputs_3d.rays, expected_device)


def test_basic_sequential_models(dtype: torch.dtype, device: torch.device):
    """
    Run simple checks for dtype and device correctness on basic sequential systems
    that are expected to work in both 2D and 3D.

    More complex models should go into their own test functions.
    """

    sequential_test_models = [
        tlm.Sequential(),
        tlm.Sequential(
            tlm.Gap(15),
        ),
        tlm.Sequential(
            tlm.ObjectAtInfinity(
                beam_diameter=10, angular_size=20, wavelength=(400, 800)
            ),
            tlm.Gap(15),
            tlm.RefractiveSurface(
                tlm.SphereByCurvature(diameter=25, C=1 / -45.759),
                materials=("air", "BK7"),
            ),
            tlm.Gap(3.419),
            tlm.RefractiveSurface(
                tlm.SphereByCurvature(diameter=25, C=1 / -24.887),
                materials=("BK7", "air"),
            ),
            tlm.Gap(97.5088),
            tlm.ImagePlane(50),
        ),
        tlm.Sequential(
            tlm.Gap(-100),
            tlm.PointSourceAtInfinity(beam_diameter=30),
            tlm.Gap(100),
            tlm.ReflectiveSurface(tlm.Parabola(35.0, A=-0.0001, trainable=True)),
            tlm.Gap(-80),
            tlm.ReflectiveSurface(
                tlm.SphereByCurvature(35.0, C=1 / 450.0, trainable=True)
            ),
            tlm.Gap(100),
            tlm.FocalPoint(),
        ),
    ]

    for model in sequential_test_models:
        # Can evaluate forward and render
        check_sample_and_render_2d(model, dtype, device)
        check_sample_and_render_3d(model, dtype, device)

        # Can convert to another dtype or device
        model.to(dtype=torch.float32)
        model.to(dtype=torch.float64)
        model.to(device=torch.device("cpu"))
        if torch.cuda.is_available():
            model.to(device=torch.device("cuda:0"))


def test_rainbow(dtype: torch.dtype, device: torch.device):

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
        tlm.RefractiveSurface(
            halfsphere.clone(anchors=(1, 1)), materials=("air", "water")
        ),
        # Second interface: half sphere (pointing right), reflective
        tlm.SubChain(
            tlm.Rotate((-180, 0)),
            tlm.ReflectiveSurface(halfsphere.clone(anchors=(1, 1))),
        ),
        # Third interface: half sphere (pointing down), refractive water to air
        tlm.SubChain(
            tlm.Rotate((60, 0)),
            tlm.RefractiveSurface(
                halfsphere.clone(anchors=(1, 0)), materials=("water", "air")
            ),
        ),
    )

    check_sample_and_render_2d(optics, dtype, device)
    check_sample_and_render_3d(optics, dtype, device)


def test_elements3(dtype: torch.dtype, device: torch.device):
    optics = tlm.Sequential(
        tlm.Translate(y=5.001),
        tlm.ObjectAtInfinity(10, 0.5),
    )

    check_sample_and_render_2d(optics, dtype, device)
    check_sample_and_render_3d(optics, dtype, device)


def test_cooke(dtype: torch.dtype, device: torch.device):
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


def test_nolens(dtype: torch.dtype, device: torch.device):
    surface = tlm.Parabola(diameter=15, A=tlm.parameter(0.02))  # y = a*x^2

    optics = tlm.Sequential(
        tlm.PointSourceAtInfinity(beam_diameter=18.5),
        tlm.Gap(10),
        tlm.RefractiveSurface(
            surface.clone(anchors=(0, 1)), materials=("air", "water")
        ),
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


def test_surface_reuse(dtype: torch.dtype, device: torch.device) -> None:
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


def test_elements_reuse(dtype: torch.dtype, device: torch.device) -> None:
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
