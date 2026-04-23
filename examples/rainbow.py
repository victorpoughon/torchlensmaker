import torchlensmaker as tlm
import torch

import math

# Use half spheres to model interface boundaries
radius = 5
halfsphere = tlm.SphereByRadius(diameter=2 * radius, R=radius)

model = tlm.Sequential(
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
        tlm.RotateMixed(-180),
        tlm.ReflectiveSurface(halfsphere.clone(anchors=(1, 1))),
    ),
    # Third interface: half sphere (pointing down), refractive water to air
    tlm.SubChain(
        tlm.RotateMixed(60),
        tlm.RefractiveSurface(
            halfsphere.clone(anchors=(1, 0)), materials=("water", "air")
        ),
    ),
)

# Use rays opacity and thickness to give some illusion of real color
controls = {
    "opacity": 0.05,
    "thickness": 2.1,
    "valid_rays": "wavelength (true color)",
    "output_rays": "wavelength (true color)",
}

# tlm.show2d(model, pupil=50, field=5, wavelength=10, end=50, controls=controls)

# tlm.show3d(model, pupil=200, field=15, wavelength=10, end=30, controls=controls)

tlm.set_sampling2d(model, pupil=50, field=5, wavel=10)

scene = tlm.render_model(model, 2, end=50, title="Rainbow", controls=controls)

from tlmviewer import push_scene

push_scene(scene)
