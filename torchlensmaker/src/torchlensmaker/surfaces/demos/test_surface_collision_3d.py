#!/usr/bin/env python
# coding: utf-8


import tlmviewer as tlmv
import torch
import torch.nn as nn

import torchlensmaker as tlm


def display_hit_miss_3d(title, source, surface, dist, pupil=50, field=10, wavel=3):
    """
    Given a light source (sequential system) and a surface
    position the surface after the light source and show hit / miss rays in tlmviewer
    """

    # Sample the light source
    tlm.set_sampling3d(source, pupil, field, wavel)
    data = source(tlm.SequentialData.empty(dim=3))

    # Raytrace the surface
    outs = surface(data.rays.P, data.rays.V, data.fk)
    t, normals, valid, stf, ntf = (
        outs.t,
        outs.normals,
        outs.valid,
        outs.tf_surface,
        outs.tf_next,
    )

    if not t.isfinite().all():
        print("Warning: surface collision returned some non-finite t values")
    if valid.all():
        print("All rays hit the surface")
    elif (~valid).all():
        print("All rays miss the surface")

    # Compute end points for colliding and non colliding rays
    hit_start = data.rays.P[valid]
    hit_end = (data.rays.P + t.unsqueeze(-1) * data.rays.V)[valid]
    miss_start = data.rays.P[~valid]
    miss_end = (data.rays.P + dist * data.rays.V)[~valid]

    # Render both ray groups
    hit = tlm.render_rays(hit_start, hit_end, "rays-valid", default_color="lightgreen")
    miss = tlm.render_rays(miss_start, miss_end, "rays-blocked", default_color="orange")

    # Render collision normals
    arrows = tlm.render_arrows(hit_end, normals[valid])

    # Render points_global
    points = tlm.render_points(outs.points_global, color="white", radius=0.05)

    # Render surface
    surf = surface.render(stf.direct)

    # Render manually
    scene = tlm.new_scene("3D")
    scene.data = [surf, hit, miss, arrows, points, tlmv.SceneTitle(title=title)]
    scene.controls = {
        "show_axis_x": True,
        "show_axis_y": True,
        "show_axis_z": True,
        "show_blocked_rays": True,
    }

    tlmv.push_scene(scene)


# display_hit_miss_3d(
#     "ImplicitDisk",
#     tlm.Sequential(
#         tlm.Gap(-6),
#         tlm.PointSource(50),
#         tlm.Gap(10),
#     ),
#     tlm.ImplicitDisk(5, solver_config=dict(num_iter=7, damping=1.0)),
#     10,
# )


# display_hit_miss_3d(
#     "XYPolynomial",
#     tlm.Sequential(
#         tlm.Gap(-1),
#         tlm.PointSourceAtInfinity(10),
#         tlm.Gap(2),
#     ),
#     tlm.XYPolynomial(
#         5,
#         C=0,
#         K=0,
#         coefficients=[[1, 0.5, 0.1], [0.2, -0.3, 0.01]],
#         solver_config=dict(num_iter=12),
#     ),
#     10,
# )


# display_hit_miss_3d(
#     "Asphere",
#     tlm.Sequential(
#         tlm.Gap(-1),
#         tlm.PointSourceAtInfinity(10),
#         tlm.Gap(2),
#     ),
#     tlm.Asphere(5, C=0, K=0, alphas=[-0.1, 0.02], solver_config=dict(num_iter=12)),
#     10,
# )


# display_hit_miss_3d(
#     "Disk",
#     tlm.Sequential(
#         tlm.Gap(-1),
#         tlm.PointSource(80),
#         tlm.Gap(5),
#     ),
#     tlm.Disk(5),
#     10,
# )


# display_hit_miss_3d(
#     "SphereByCurvature",
#     tlm.Sequential(
#         tlm.Gap(-1),
#         tlm.PointSource(80),
#         tlm.Gap(5),
#     ),
#     tlm.SphereByCurvature(5, 0.30),
#     10,
# )


# display_hit_miss_3d(
#     "SphereByRadius",
#     tlm.Sequential(
#         tlm.Gap(-1),
#         tlm.PointSourceAtInfinity(10),
#         tlm.Gap(2),
#         # tlm.Rotate((10, 0)),
#     ),
#     tlm.SphereByRadius(5, 2.5),
#     10,
# )


# display_hit_miss_3d(
#     "Parabola",
#     tlm.Sequential(
#         tlm.Gap(-1),
#         tlm.PointSourceAtInfinity(10),
#         tlm.Gap(2),
#         tlm.RotateMixed(10),
#     ),
#     tlm.Parabola(5, 0.5, solver_config=dict(num_iter=12)),
#     10,
# )


# display_hit_miss_3d(
#     "Conic",
#     tlm.Sequential(
#         tlm.Gap(-1),
#         tlm.PointSourceAtInfinity(10),
#         tlm.Gap(2),
#     ),
#     tlm.Conic(5, C=0.15, K=0, solver_config=dict(num_iter=12)),
#     10,
# )

display_hit_miss_3d(
    "Rainbow - sphere by radius",
    tlm.Sequential(
        tlm.SubChain(
            tlm.Translate(y=5.001),
            tlm.ObjectAtInfinity(10, 0.5, wavelength=(400, 660)),
        ),
        tlm.Gap(50),
    ),
    tlm.SphereByRadius(diameter=2 * 5, R=5),
    80,
    pupil=100,
    field=1,
    wavel=1,
)


display_hit_miss_3d(
    "Rainbow - sphere newton1",
    tlm.Sequential(
        tlm.SubChain(
            tlm.Translate(y=5.001),
            tlm.ObjectAtInfinity(10, 0.5, wavelength=(400, 660)),
        ),
        tlm.Gap(50),
    ),
    tlm.Sphere(R=5),
    80,
    pupil=100,
    field=1,
    wavel=1,
)

display_hit_miss_3d(
    "Rainbow - sphere newton2",
    tlm.Sequential(
        tlm.SubChain(
            tlm.Translate(y=5.001),
            tlm.ObjectAtInfinity(10, 0.5, wavelength=(400, 660)),
        ),
        tlm.Gap(50),
    ),
    tlm.Sphere(
        R=5,
        solver_config=dict(
            implicit_solver="newton2",
            num_iter=12,
            damping=0.95,
            tol=1e-4,
            init="0",
            clamp_positive=True,
        ),
    ),
    80,
    pupil=100,
    field=1,
    wavel=1,
)

tlmv.push_source(__file__)
