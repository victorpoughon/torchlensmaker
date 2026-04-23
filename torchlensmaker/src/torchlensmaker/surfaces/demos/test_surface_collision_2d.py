#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

import torchlensmaker as tlm
import tlmviewer as tlmv

def display_hit_miss_2d(source, surface, dist):
    """
    Given a light source (sequential system) and a surface
    position the surface after the light source and show hit / miss rays in tlmviewer
    """

    # Sample the light source
    source.set_sampling2d(pupil=50)
    data = source(tlm.SequentialData.empty(dim=2))

    # Raytrace the surface
    outs = surface(
        data.rays.P, data.rays.V, data.fk
    )
    t, normals, valid, stf, ntf = (outs.t, outs.normals, outs.valid, outs.tf_surface, outs.tf_next)

    if not t.isfinite().all():
        print("Warning: surface collision returned some non-finite t values")
    if valid.all():
        print("All rays hit the surface")
    elif (~valid).all():
        print("All rays miss the surface")

    print("rsm: ", outs.rsm.min(), outs.rsm.mean(), outs.rsm.max())

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

    # Build scene
    scene = tlm.new_scene("2D")
    scene.data = [surf, hit, miss, arrows, points]
    scene.controls = {"show_axis_x": True, "show_axis_y": True, "show_axis_z": True}

    print(scene.data)
    tlmv.push_scene(scene)


# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-4),
        tlm.PointSource(180),
        tlm.Gap(10),
        tlm.Rotate2D(40),
    ),
    tlm.ImplicitDisk(5, solver_config=dict(num_iter=12, damping=0.9, implicit_solver="newton2")),
    15,
)

# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-4),
        tlm.PointSource(180),
        tlm.Gap(10),
        tlm.Rotate2D(40),
    ),
    tlm.SphereByCurvature(5, 0.1, solver_config={"implicit_solver": "newton", "lift_function": "raw"}),
    15,
)

display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-4),
        tlm.PointSource(180),
        tlm.Gap(10),
        tlm.Rotate2D(40),
    ),
    tlm.SphereByCurvature(5, 0.1, solver_config={"implicit_solver": "newton2", "lift_function": "euclid"}),
    15,
)


# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSourceAtInfinity(10),
        tlm.Gap(2),
    ),
    tlm.Asphere(5, C=0, K=0, alphas=[-0.1, 0.02], solver_config=dict(num_iter=12)),
    10,
)


# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.Rotate2D(85),
        tlm.PointSourceAtInfinity(3),
        tlm.Rotate2D(-45),
        tlm.Gap(2),
    ),
    tlm.SphereByRadius(5, 2.5),
    10,
)


# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSourceAtInfinity(10),
        tlm.Gap(2),
        tlm.Translate2D(y=-2),
        tlm.Rotate2D(25),
    ),
    tlm.SphereByCurvature(5, 0.1, scale=-1),
    10,
)

display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSourceAtInfinity(10),
        tlm.Gap(2),
        tlm.Translate2D(y=-2),
        tlm.Rotate2D(25),
    ),
    tlm.SphereByRadius(5, 10, scale=-1),
    10,
)


# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSource(50),
        tlm.Gap(2),
        tlm.Translate2D(y=-2),
        tlm.Rotate2D(100),
    ),
    tlm.SphereByCurvature(5, 0.12, solver_config=dict(num_iter=1)),
    10,
)

display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSource(50),
        tlm.Gap(2),
        tlm.Translate2D(y=-2),
        tlm.Rotate2D(100),
    ),
    tlm.SphereByCurvature(5, 0.12, solver_config=dict(num_iter=2)),
    10,
)

display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSource(50),
        tlm.Gap(2),
        tlm.Translate2D(y=-2),
        tlm.Rotate2D(100),
    ),
    tlm.SphereByCurvature(5, 0.12, solver_config=dict(num_iter=3)),
    10,
)

display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSource(50),
        tlm.Gap(2),
        tlm.Translate2D(y=-2),
        tlm.Rotate2D(100),
    ),
    tlm.SphereByCurvature(5, 0.12, solver_config=dict(num_iter=6)),
    10,
)


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSource(50),
        tlm.Gap(2),
        tlm.Translate2D(y=-2),
        tlm.Rotate2D(100),
    ),
    tlm.SphereByCurvature(5, 0.12, solver_config=dict(num_iter=24, damping=0.5)),
    10,
)


# In[ ]:


# lift function: raw vs euclid

display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSource(50),
        tlm.Gap(2),
        tlm.Translate2D(y=-2),
        tlm.Rotate2D(100),
    ),
    tlm.SphereByCurvature(5, 0.12, solver_config=dict(num_iter=6, lift_function="raw")),
    10,
)

display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSource(50),
        tlm.Gap(2),
        tlm.Translate2D(y=-2),
        tlm.Rotate2D(100),
    ),
    tlm.SphereByCurvature(5, 0.12, solver_config=dict(num_iter=6, damping=0.8, lift_function="euclid")),
    10,
)


# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.Rotate2D(85),
        tlm.PointSourceAtInfinity(3),
        tlm.Rotate2D(-85),
        tlm.Gap(2),
    ),
    tlm.SphereByCurvature(5, 0.10, solver_config=dict(num_iter=12)),
    10,
)


# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSourceAtInfinity(3),
        tlm.Gap(1),
    ),
    tlm.SphereByCurvature(5, 0.10, anchors=(1.0, 0.0)),
    10,
)


# ## inner case

# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.Translate2D(y=-2),
        tlm.Rotate2D(90),
        tlm.PointSourceAtInfinity(3),
        tlm.Rotate2D(-90),
        tlm.Translate2D(y=2),
        tlm.Gap(2),
    ),
    tlm.SphereByCurvature(5, -0.39, solver_config={"init": 20, "damping": 0.5, "lift_function": "raw", "num_iter": 20}),
    10,
)


# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.Translate2D(y=-6),
        tlm.Rotate2D(85),
        tlm.PointSourceAtInfinity(3),
        tlm.Rotate2D(-85),
        tlm.Translate2D(y=6),
        tlm.Gap(2),
    ),
    tlm.SphereByCurvature(5, -0.39, solver_config=dict(num_iter=1, damping=0.5, tol=1e-3)),
    10,
)

display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.Translate2D(y=-6),
        tlm.Rotate2D(85),
        tlm.PointSourceAtInfinity(3),
        tlm.Rotate2D(-85),
        tlm.Translate2D(y=6),
        tlm.Gap(2),
    ),
    tlm.SphereByCurvature(5, -0.39, solver_config=dict(num_iter=2, damping=0.5, tol=1e-3)),
    10,
)

display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.Translate2D(y=-6),
        tlm.Rotate2D(85),
        tlm.PointSourceAtInfinity(3),
        tlm.Rotate2D(-85),
        tlm.Translate2D(y=6),
        tlm.Gap(2),
    ),
    tlm.SphereByCurvature(5, -0.39, solver_config=dict(num_iter=12, damping=0.5, tol=1e-3)),
    10,
)


# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSourceAtInfinity(10),
        tlm.Gap(2),
    ),
    tlm.SphereByCurvature(5, 0.12, normalize=False),
    10,
)

display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSourceAtInfinity(10),
        tlm.Gap(2),
    ),
    tlm.SphereByCurvature(5, 0.12, normalize=True),
    10,
)


# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSourceAtInfinity(10),
        tlm.Gap(2),
        tlm.Translate2D(y=-2),
        tlm.Rotate2D(25),
    ),
    tlm.Parabola(5, 0.5, solver_config=dict(num_iter=12)),
    10,
)


# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-1),
        tlm.PointSourceAtInfinity(10),
        tlm.Gap(2),
    ),
    tlm.Conic(5, C=0, K=0, solver_config=dict(num_iter=12)),
    10,
)


# In[ ]:


display_hit_miss_2d(
    tlm.Sequential(
        tlm.Gap(-5),
        tlm.PointSource(10),
        tlm.Gap(5),
    ),
    tlm.Parabola(8.0, A=-0.1, normalize=False, anchors=(1, 0)),
    10,
)


# In[ ]:




