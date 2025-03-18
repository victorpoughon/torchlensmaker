# Test local collide


```python
import torchlensmaker as tlm
import torch
import torch.nn as nn
from pprint import pprint

from torchlensmaker.testing.basic_transform import basic_transform
from torchlensmaker.core.transforms import IdentityTransform
from torchlensmaker.testing.collision_datasets import (
    NormalRays,
    TangentRays,
    RandomRays,
    FixedRays,
    make_offset_rays,
)

from torchlensmaker.core.collision_detection import CollisionMethod, Newton, LM

from torchlensmaker.core.surfaces import CircularPlane, Sphere, SphereR

from torchlensmaker.core.geometry import rotated_unit_vector, unit3d_rot, unit2d_rot

import matplotlib.pyplot as plt

import sys
import traceback

from torchlensmaker.testing.dataset_view import dataset_view, convergence_plot
from torchlensmaker.testing.check_local_collide import check_local_collide


test_cases = [
    # Failing cases so far with LM
    (tlm.Sphere(30, 30), TangentRays(dim=2, N=15, distance=-0.6, epsilon=0.05), True),
    (tlm.Sphere(30, 30), FixedRays(dim=2, N=15, direction=unit2d_rot(45), offset=30, epsilon=0.05), True),
    (tlm.Sphere(30, 30), FixedRays(dim=2, N=15, direction=unit2d_rot(65), offset=30, epsilon=0.05), True),
    (tlm.Sphere(30, 30), FixedRays(dim=2, N=15, direction=unit2d_rot(85), offset=30, epsilon=0.05), True),

    # Failing with Newton init_best_axis, because of nan in dot product
    #(tlm.Sphere(30, 30), FixedRays(direction=torch.tensor([0., 1.0]), offset=30, N=15), True),
]

test_cases = [
    # Fails weirdly with float32, not 64!
    (tlm.SphereR(5, 10, dtype=torch.float32), NormalRays(dim=2, N=5, offset=10.0, epsilon=0.1), True),
    (tlm.SphereR(5, 10, dtype=torch.float64), NormalRays(dim=2, N=5, offset=10.0, epsilon=0.1), True)
]

test_cases = [
    (tlm.Asphere(10, 50, 1.0, 0.005, dtype=torch.float64),
    FixedRays(dim=2, N=15, direction=torch.tensor([0.1736, 0.9848], dtype=torch.float64), offset=0.0, epsilon=0.01),
    True)
]

show_all = True

offset_space = torch.cat(
        (
            torch.logspace(-6, 3, 20),
            torch.linspace(0.0, 100.0, 20),
            #-torch.logspace(-6, 2, 20),
            #-torch.linspace(0.0, 100.0, 20),
        ),
        dim=0,
    )


for surface, gen, expected_collide in test_cases:
    genP, genV = gen(surface)

    #P, V = make_offset_rays(genP, genV, offset_space)
    P, V = genP + 68.4211*genV, genV

    print(P.shape)

    if show_all:
        dataset_view(surface, P, V, rays_length=1000)

    # check collisions
    try:
        check_local_collide(surface, P, V, expected_collide)
    except AssertionError as err:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)

        # tlmviewer view
        print("Test failed")
        print(gen)
        print("expected_collide:", expected_collide)
        print("AssertionError:", err)
        dataset_view(surface, P, V)

        if isinstance(surface, tlm.ImplicitSurface):
            convergence_plot(surface, P, V, dataset_name=str(gen.__class__.__name__), methods=[surface.collision_method])


```

    torch.Size([15, 2])



<TLMViewer src="./collision_detection_view_testcase_files/collision_detection_view_testcase_0.json?url" />

