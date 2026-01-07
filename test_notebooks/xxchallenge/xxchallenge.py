# This file is part of Torch Lens Maker
# Copyright (C) 2024-present Victor Poughon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import math
import torch
import torch.nn as nn
import torchlensmaker as tlm
import numpy as np

from torchlensmaker.core.collision_detection import LM

from typing import Any
from itertools import product
from stl import mesh

import matplotlib.pyplot as plt

from torchlensmaker.viewer import tlmviewer
from torchlensmaker.core.rot3d import euler_angles_to_matrix
from torchlensmaker.core.transforms import (
    LinearTransform,
)
from scipy.spatial import Delaunay


def find_closest_points(surface, P, V):
    N = P.shape[0]

    t = torch.zeros(N).unsqueeze(0)
    algo = LM(0.5)

    with torch.no_grad():
        for i in range(30):
            t = t - algo.delta(surface, P, V, t, max_delta=1000.0)

    t = t - algo.delta(surface, P, V, t, max_delta=1000.0)

    return t.squeeze(0)


# def loss(surface, P, V):
#     t = find_closest_points(surface, P, V)
#     points = P + t.unsqueeze(-1).expand_as(V) * V
#     F = surface.F(points)
#     return F.sum() / points.shape[0]


class XXLightSource(nn.Module):
    def __init__(self, all_rays):
        super().__init__()
        self.all_rays = all_rays
        self.positive = all_rays[all_rays[:, 3] > 0.0]
        self.negative = all_rays[all_rays[:, 3] < 0.0]

    @classmethod
    def load(cls, half: bool = False):
        data = np.loadtxt("xx_lighttools.txt", usecols=(0, 1, 2, 3, 4, 5))
        all_rays = torch.from_numpy(data).float()
        assert all_rays.shape == (1000000, 6)
        return cls(all_rays)

    def forward(self, inputs):
        # Pick random rays along the 'xx' dimension
        N = inputs.sampling["xx"].N
        letter = inputs.sampling["letter"]

        match letter:
            case "both":
                source = self.all_rays
            case "positive":
                source = self.positive
            case "negative":
                source = self.negative
            case _:
                raise RuntimeError(f"unknown xx letter {letter}")

        indices = torch.randint(0, source.shape[0], (N,))
        rays = source[indices]

        P = 1000 * torch.stack(
            (torch.zeros_like(rays[:, 0]), rays[:, 0], rays[:, 1]), -1
        )
        V = torch.stack((-rays[:, 5], rays[:, 3], rays[:, 4]), -1)

        V = torch.nn.functional.normalize(V, dim=-1)

        # Apply kinematic transform
        P, V = inputs.tf().direct_rays(P, V)

        return inputs.replace(
            P=torch.cat((inputs.P, P), dim=0),
            V=torch.cat((inputs.V, V), dim=0),
        )


class NonImagingRod(nn.Module):
    def __init__(self, surface):
        self.surface = surface
        super().__init__()

    def forward(self, inputs: tlm.OpticalData) -> tlm.OpticalData:
        N, dim = inputs.P.shape
        assert dim == 3

        if N == 0:
            print("NonImagingRod: warning, no rays")
            return inputs

        # Convert rays to surface local frame
        transform = inputs.tf()
        P_local = transform.inverse_points(inputs.P)
        V_local = transform.inverse_vectors(inputs.V)

        t = find_closest_points(self.surface, P_local, V_local)

        # Closest points
        points_local = P_local + t.unsqueeze(-1).expand_as(V_local) * V_local

        # Normalize by surface scale
        scale = torch.mean(
            torch.abs(
                torch.as_tensor(
                    [self.surface.xmin, self.surface.xmax, self.surface.tau]
                )
            )
        )

        F = self.surface.F(points_local) / scale

        # loss = torch.clamp(F, min=0.0).pow(2).sum() / N
        loss = F.pow(2).sum() / N

        return inputs.replace(loss=inputs.loss + loss)


class NonImagingRodNoSq(nn.Module):
    def __init__(self, surface):
        self.surface = surface
        super().__init__()

    def forward(self, inputs: tlm.OpticalData) -> tlm.OpticalData:
        N, dim = inputs.P.shape
        assert dim == 3

        if N == 0:
            print("NonImagingRod: warning, no rays")
            return inputs

        # Convert rays to surface local frame
        transform = inputs.tf()
        P_local = transform.inverse_points(inputs.P)
        V_local = transform.inverse_vectors(inputs.V)

        t = find_closest_points(self.surface, P_local, V_local)

        # Closest points
        points_local = P_local + t.unsqueeze(-1).expand_as(V_local) * V_local

        # Normalize by surface scale
        scale = torch.mean(
            torch.abs(
                torch.as_tensor(
                    [self.surface.xmin, self.surface.xmax, self.surface.tau]
                )
            )
        )

        F = self.surface.F(points_local) / scale

        # loss = torch.clamp(F, min=0.0).pow(2).sum() / N
        loss = torch.log(torch.clamp(F, min=0.1)).sum() / N

        return inputs.replace(loss=inputs.loss + loss)


class RodArtist:
    def render_module(self, collective, module) -> list[Any]:
        tf = collective.input_tree[module].tf()
        s = module.surface
        return [
            {
                "type": "surface",
                "bcyl": [s.xmin.item(), s.xmax.item(), s.tau.item()],
                "matrix": tf.hom_matrix().tolist(),
            }
        ]

    def render_rays(self, collective, module) -> list[Any]:
        inputs = collective.input_tree[module]
        N, dim = inputs.P.shape

        if N == 0:
            print("RodArtist: warning, no rays")
            return []

        # Convert rays to surface local frame
        transform = tlm.forward_kinematic(inputs.transforms)
        P_local = transform.inverse_points(inputs.P)
        V_local = transform.inverse_vectors(inputs.V)

        t = find_closest_points(module.surface, P_local, V_local)
        closest_points = inputs.P + t.unsqueeze(-1).expand_as(inputs.V) * inputs.V

        rays = [
            tlmviewer.render_rays(
                inputs.P,
                closest_points,
                default_color="#ffa724",
                layer=0,
            )
        ]

        return rays


class RaysViewerPlane(nn.Module):
    "Utility component to vizualize rays distribution"

    def __init__(self, diameter, title):
        super().__init__()
        self.title = title
        surface = tlm.CircularPlane(diameter, dtype=torch.float64)
        self.collision_surface = tlm.CollisionSurface(surface)

    def forward(self, inputs):
        # Collision detection
        t, normals, valid_collision, new_chain = self.collision_surface(inputs)
        collision_points = inputs.P + t.unsqueeze(-1).expand_as(inputs.V) * inputs.V

        if "disable_viewer" not in inputs.sampling:
            with torch.inference_mode():
                # compute rays intersection with X=0 plane
                # plot YZ linear distribution
                # plot MN angular distribution

                X, Y, Z = collision_points.unbind(-1)
                L, M, N = inputs.V.unbind(-1)

                f, (ax1, ax2) = plt.subplots(1, 2)

                ax1.scatter(Y, Z, s=0.2)
                ax1.set_aspect("equal")
                ax1.set_title("linear")
                ax2.scatter(M, N, s=0.2)
                ax2.set_aspect("equal")
                ax2.set_title("angular")
                f.suptitle(self.title)

        return inputs


class Focus(nn.Module):
    "X axis translation equal to the focal distance of a parabola"

    def __init__(self, mirror):
        super().__init__()
        self.mirror = mirror

    def forward(self, data):
        dim, dtype = data.dim, data.dtype
        f = 1.0 / (self.mirror._sag.unnorm(self.mirror.diameter / 2) * 4)
        translate_vector = torch.cat(
            (
                f.unsqueeze(0).to(dtype=dtype),
                torch.zeros(dim - 1, dtype=dtype),
            )
        )
        return data.replace(
            transforms=data.transforms + [tlm.TranslateTransform(translate_vector)]
        )


class XXBounds(nn.Module):
    def __init__(self, margin, scale, beta):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.s1 = nn.Softplus(beta)
        self.s2 = nn.Softplus(beta)

    def forward(self, val):
        return self.scale * (self.s1(val - self.margin) + self.s2(-self.margin - val))


class BoxLoss(nn.Module):
    "Loss term for joint with the XX box"

    def __init__(self, margin, scale, beta):
        super().__init__()
        self.xxbounds = XXBounds(margin, scale, beta)

    def forward(self, data):
        target = data.target()

        x, y, z = target.unbind(-1)
        term = self.xxbounds(y) + self.xxbounds(z)

        return data.replace(loss=data.loss + term)


# Custom render XX
def xxrender(optics, sampling, extra_artists={}, end=0):
    controls = {"show_bounding_cylinders": True, "show_optical_axis": True}

    scene = tlm.render_sequence(
        optics,
        dim=3,
        dtype=torch.float64,
        sampling=sampling,
        end=end,
        extra_artists={
            NonImagingRod: RodArtist(),
            NonImagingRodNoSq: RodArtist(),
            **extra_artists,
        },
    )
    scene["controls"] = controls
    boxtf = tlm.TranslateTransform(torch.tensor([-500, 0, 0]))
    scene["data"].append(
        {
            "type": "box3D",
            "size": [1000, 1000, 1000],
            "matrix": boxtf.hom_matrix().tolist(),
        }
    )
    tlmviewer.display_scene(scene, 8)


def gridsearch1d(model, parameter, space, sampling):
    loss = torch.zeros_like(space)

    for i, p in enumerate(space):
        with torch.no_grad():
            inputs = tlm.default_input(sampling, dim=3, dtype=torch.float64)
            parameter.fill_(p.item())
            output = model(inputs)
            loss[i] = output.loss.item()

    fig, ax = plt.subplots(1, 1)
    ax.plot(space.tolist(), loss.tolist())
    ax.set_ylim([0, ax.get_ylim()[1]])


def gridsearchnd(model, parameters, spaces, sampling):
    inputs = tlm.default_input(sampling, dim=3, dtype=torch.float64)

    with torch.no_grad():
        for vals in product(*[s.tolist() for s in spaces]):
            for p, v in zip(parameters, vals):
                p.fill_(v)
            output = model(inputs)
            print(vals)
            print(output.loss.item())
            print()


def param_to_name(param, named_parameters):
    for n, p in named_parameters:
        if p is param:
            return n


def plot_record(record, param_groups, optics, ylim=None) -> None:
    if record.num_iter == 0:
        return

    # Create subplots
    M = len(param_groups)
    fig, all_axes = plt.subplots(M + 1, 1, figsize=(10, 8))
    ax_loss = all_axes[0]
    axes = all_axes[1:]

    epoch_range = torch.arange(0, record.num_iter)

    # Plot parameters history
    for i, pdict in enumerate(param_groups):
        ax = axes[i]

        for param in pdict["params"]:
            pname = param_to_name(param, optics.named_parameters())
            precord = record.parameters[pname]
            if param.dim() == 0:
                data = torch.stack(precord).detach().numpy()
            else:
                data = (
                    torch.stack([torch.linalg.matrix_norm(p) for p in precord])
                    .detach()
                    .numpy()
                )
            ax.plot(epoch_range.detach(), data, label=pname)
            ax.legend()

    # Plot loss
    ax_loss.plot(epoch_range, record.loss.detach(), label="loss")
    ax_loss.set_title("loss")
    if not ylim:
        ax_loss.set_ylim([0, ax_loss.get_ylim()[1]])
    else:
        ax_loss.set_ylim(ylim)
    ax_loss.legend()
    plt.show()


class StoreVar(nn.Module):
    def __init__(self, foo):
        super().__init__()
        self.foo = foo
        self.value = None

    def forward(self, data):
        self.value = self.foo(data)
        return data


class StaticRays(nn.Module):
    def __init__(self, P, V):
        super().__init__()
        self.P = P
        self.V = V

    def forward(self, data):
        return data.replace(P=self.P, V=self.V)


def addsides(part):
    # Define cube side vertices (8 triangles total)
    cube_vectors = np.array(
        [
            # Right side (X=0.5)
            [[0.5, -0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 1]],
            [[0.5, -0.5, 0], [0.5, 0.5, 1], [0.5, -0.5, 1]],
            # Left side (X=-0.5)
            [[-0.5, -0.5, 0], [-0.5, 0.5, 0], [-0.5, 0.5, 1]],
            [[-0.5, -0.5, 0], [-0.5, 0.5, 1], [-0.5, -0.5, 1]],
            # Front side (Y=0.5)
            [[0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, 0.5, 1]],
            [[0.5, 0.5, 0], [-0.5, 0.5, 1], [0.5, 0.5, 1]],
            # Back side (Y=-0.5)
            [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, -0.5, 1]],
            [[-0.5, -0.5, 0], [0.5, -0.5, 1], [-0.5, -0.5, 1]],
        ],
        dtype=part.vectors.dtype,
    )

    # Create new mesh section for cube walls
    cube_mesh = mesh.Mesh(np.zeros(cube_vectors.shape[0], dtype=mesh.Mesh.dtype))
    cube_mesh.vectors = cube_vectors

    # Combine with original geometry
    part = mesh.Mesh(np.concatenate([part.data, cube_mesh.data]))

    return part


def makesides(dtype):
    # Define cube side vertices (8 triangles total)
    cube_vectors = np.array(
        [
            # Right side (X=0.5)
            [[0.5, -0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 1]],
            [[0.5, -0.5, 0], [0.5, 0.5, 1], [0.5, -0.5, 1]],
            # Left side (X=-0.5)
            [[-0.5, -0.5, 0], [-0.5, 0.5, 0], [-0.5, 0.5, 1]],
            [[-0.5, -0.5, 0], [-0.5, 0.5, 1], [-0.5, -0.5, 1]],
            # Front side (Y=0.5)
            [[0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, 0.5, 1]],
            [[0.5, 0.5, 0], [-0.5, 0.5, 1], [0.5, 0.5, 1]],
            # Back side (Y=-0.5)
            [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, -0.5, 1]],
            [[-0.5, -0.5, 0], [0.5, -0.5, 1], [-0.5, -0.5, 1]],
        ],
        dtype=dtype,
    )

    # Create new mesh section for cube walls
    cube_mesh = mesh.Mesh(np.zeros(cube_vectors.shape[0], dtype=mesh.Mesh.dtype))
    cube_mesh.vectors = cube_vectors

    return cube_mesh


def xxgrid(r, N):
    x = np.linspace(-r, r, N)
    y = np.linspace(-r, r, N)
    X, Y = np.meshgrid(x, y)
    return np.stack((X, Y), -1).reshape(-1, 2)


def xxtess(points, tf, primary, filename: str):
    "XX challenge tessalate"

    # sampling rays:
    N = points.shape[0]
    P = torch.stack(
        (
            torch.full((N,), -1000, dtype=torch.float64),
            torch.as_tensor(points[:, 0]),
            torch.as_tensor(points[:, 1]),
        ),
        -1,
    )
    V = torch.tile(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), (N, 1))

    # Collision detection to primary mirror
    optics = tlm.Sequential(
        StaticRays(P, V),
        tlm.AbsoluteTransform(tf),
        tlm.ReflectiveSurface(primary),
    )

    output = optics(tlm.default_input({}, dim=3, dtype=torch.float64))
    vertices_tlm = output.P

    # Perform Delaunay triangulation on the YZ grid
    tri = Delaunay(vertices_tlm[:, 1:])
    faces = tri.simplices

    # convert coordinate frame convention
    vertices = (
        torch.stack(
            [
                vertices_tlm[:, 1],
                vertices_tlm[:, 2],
                -vertices_tlm[:, 0],
            ],
            dim=-1,
        )
        .detach()
        .numpy()
    )

    # convert mm to m
    vertices *= 0.001

    # Create the mesh
    part = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            part.vectors[i][j] = vertices[f[j], :]

    # Add four sides to the cube
    part = addsides(part)

    part.save(filename)


def settobest(model, record, param_group):
    best_loss, idx = record.loss.min(dim=0)
    print(f"Best loss {best_loss.item()} at iteration {idx + 1} / {record.num_iter}")

    with torch.no_grad():
        for ps in param_group:
            for p in ps["params"]:
                name = param_to_name(p, model.named_parameters())
                val = record.parameters[name][idx].item()
                print(f"Set {name} to {val}")
                p.fill_(val)


def tess_mirror(points, tf, surface, flipy=False):
    "tesselate mirror by raytracing"

    # make sampling rays from the given YZ grid
    N = points.shape[0]
    P = torch.stack(
        (
            torch.full((N,), -1000, dtype=torch.float64),
            torch.as_tensor(points[:, 0]),
            torch.as_tensor(points[:, 1]),
        ),
        -1,
    )
    V = torch.tile(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), (N, 1))

    # Collision detection to the mirror
    optics = tlm.Sequential(
        StaticRays(P, V),
        tlm.AbsoluteTransform(tf),
        tlm.CollisionSurface(surface),
    )

    t, _, collision_valid, new_chain = optics(
        tlm.default_input({}, dim=3, dtype=torch.float64)
    )

    # Collision vertices in tlm frame convention
    vertices_tlm = P + t.unsqueeze(1).expand_as(V) * V
    vertices_tlm = vertices_tlm[collision_valid]

    # Filter out of box points
    inbox = torch.logical_and(vertices_tlm[:, 0] > -1000, vertices_tlm[:, 0] < 0)
    print("not in box", (~inbox).sum())
    vertices_tlm = vertices_tlm[inbox]

    if flipy:
        vertices_tlm[:, 1:] = -vertices_tlm[:, 1:]

    # Perform Delaunay triangulation on the YZ grid
    tri = Delaunay(vertices_tlm[:, 1:])
    faces = tri.simplices

    # convert coordinate frame convention
    vertices = (
        torch.stack(
            [
                vertices_tlm[:, 1],
                vertices_tlm[:, 2],
                -vertices_tlm[:, 0],
            ],
            dim=-1,
        )
        .detach()
        .numpy()
    )

    # convert mm to m
    vertices *= 0.001

    # Create the mesh
    part = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            part.vectors[i][j] = vertices[f[j], :]

    return part


class Secondary(nn.Module):
    def __init__(self, surface):
        super().__init__()
        self.collision_surface = tlm.CollisionSurface(surface)

    def forward(self, data):
        # Collision detection
        t, normals, valid, new_chain = self.collision_surface(data)
        collision_points = data.P + t.unsqueeze(1).expand_as(data.V) * data.V

        # Compute reflection for colliding rays
        reflected = tlm.reflection(data.V[valid], normals[valid])

        # Compute implicit cylinder loss for non colliding rays
        nmiss = (~valid).sum()
        if nmiss > 0:
            surface = tlm.ImplicitCylinder(
                *self.collision_surface.surface.bcyl().unbind(-1)
            )
            rod = NonImagingRod(surface)

            data_miss = data.replace(P=data.P[~valid], V=data.V[~valid])

            loss = rod(data_miss).loss

            return data.filter_variables(valid).replace(
                P=collision_points[valid],
                V=reflected,
                transforms=new_chain,
                loss=data.loss + loss,
            )

        else:
            return data.filter_variables(valid).replace(
                P=collision_points[valid],
                V=reflected,
                transforms=new_chain,
            )


def makecanards(dtype, xspace, depth):
    Y = 0.5
    D = depth / 1000
    Xspace = xspace / 1000

    triangles = []
    for X in Xspace:
        triangles.append(
            np.array(
                [
                    [[X, -Y, 1], [X, -Y, 1.0 - D], [X, +Y, 1]],
                    [[X, -Y, 1 - D], [X, +Y, 1 - D], [X, +Y, 1]],
                ],
                dtype=dtype,
            )
        )

    all_triangles = np.concatenate(triangles)

    cube_mesh = mesh.Mesh(np.zeros(all_triangles.shape[0], dtype=mesh.Mesh.dtype))
    cube_mesh.vectors = all_triangles

    return cube_mesh
