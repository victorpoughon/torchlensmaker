import torch
import torch.nn as nn
import torchlensmaker as tlm

from typing import Literal, Any, Optional, Type, Dict, TypeVar, Iterable, Generic
from collections import defaultdict
from dataclasses import dataclass

from torchlensmaker.core.tensor_manip import filter_optional_mask

from torchlensmaker.analysis.colors import (
    color_valid,
    color_focal_point,
    color_blocked,
)

import torchlensmaker.viewer as viewer

import matplotlib as mpl
import json
import os

Tensor = torch.Tensor


LAYER_VALID_RAYS = 1
LAYER_BLOCKED_RAYS = 2
LAYER_OUTPUT_RAYS = 3
LAYER_JOINTS = 4


@dataclass
class RayVariables:
    "Available ray variables and their min/max domain"

    variables: list[str]
    domain: dict[str, list[float]]

    @classmethod
    def from_optical_data(
        cls, optical_data: Iterable[tlm.OpticalData]
    ) -> "RayVariables":
        variables: set[str] = set()
        domain: defaultdict[str, list[float]] = defaultdict(
            lambda: [float("+inf"), float("-inf")]
        )

        def update(var: Optional[Tensor], name: str) -> None:
            if var is not None:
                variables.add(name)
                if var.numel() > 0 and var.min() < domain[name][0]:
                    domain[name][0] = var.min().item()
                if var.numel() > 0 and var.max() > domain[name][1]:
                    domain[name][1] = var.max().item()

        for inputs in optical_data:
            update(inputs.rays_base, "base")
            update(inputs.rays_object, "object")
            update(inputs.rays_wavelength, "wavelength")

        return cls(list(variables), dict(domain))


def ray_variables_dict(
    data: tlm.OpticalData, variables: list[str], valid: Optional[Tensor] = None
) -> dict[str, Tensor]:
    "Convert ray variables from an OpticalData object to a dict of Tensors"
    d = {}

    def update(tensor: Optional[Tensor], name: str) -> None:
        if tensor is not None:
            d[name] = filter_optional_mask(tensor, valid)

    # TODO no support for 2D colormaps in tlmviewer yet
    # but base and object are 2D variables in 3D
    if data.dim == 2:
        update(data.rays_base, "base")
        update(data.rays_object, "object")
    update(data.rays_wavelength, "wavelength")

    return d


def render_rays_until(
    P: Tensor,
    V: Tensor,
    end: Tensor,
    variables: dict[str, Tensor],
    domain: dict[str, list[float]],
    default_color: str,
    layer: int,
) -> list[Any]:
    "Render rays until an absolute X coordinate"
    assert end.dim() == 0
    # div by zero here for vertical rays
    t = (end - P[:, 0]) / V[:, 0]
    ends = P + t.unsqueeze(1).expand_as(V) * V
    return [
        viewer.render_rays(
            P,
            ends,
            variables=variables,
            domain=domain,
            default_color=default_color,
            layer=layer,
        )
    ]


def render_rays_length(
    P: Tensor,
    V: Tensor,
    length: float | Tensor,
    variables: dict[str, Tensor],
    domain: dict[str, list[float]],
    layer: int,
    default_color: str = color_valid,
) -> list[Any]:
    "Render rays with fixed length"

    if isinstance(length, Tensor):
        assert length.dim() in {0, 1}

    if isinstance(length, Tensor) and length.dim() == 1:
        length = length.unsqueeze(1).expand_as(V)

    return [
        viewer.render_rays(
            P,
            P + length * V,
            variables=variables,
            domain=domain,
            default_color=default_color,
            layer=layer,
        )
    ]


class KinematicSurfaceArtist:
    @staticmethod
    def render_module(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        ray_variables: RayVariables,
    ) -> list[Any]:

        dim, dtype = (
            input_tree[module].transforms[0].dim,
            input_tree[module].transforms[0].dtype,
        )
        chain = input_tree[module].transforms + module.surface_transform(dim, dtype)
        transform = tlm.forward_kinematic(chain)

        # TODO find a way to group surfaces together?
        return [
            viewer.render_surfaces(
                [module.surface], [transform], dim=transform.dim, N=100
            )
        ]

    @staticmethod
    def render_rays(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        ray_variables: RayVariables,
    ) -> list[Any]:
        return render_rays(module.element, input_tree, output_tree, ray_variables)


class CollisionSurfaceArtist:
    @staticmethod
    def render_module(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        ray_variables: RayVariables,
    ) -> list[Any]:

        points = output_tree[module].P
        normals = output_tree[module].normals

        # return viewer.render_collisions(points, normals)
        return []

    @staticmethod
    def render_rays(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        ray_variables: RayVariables,
    ) -> list[Any]:
        inputs = input_tree[module]
        outputs = output_tree[module]
        # If rays are not blocked, render simply all rays from collision to collision
        if outputs.blocked is None:
            return [
                viewer.render_rays(
                    inputs.P,
                    outputs.P,
                    variables=ray_variables_dict(inputs, ray_variables.variables),
                    domain=ray_variables.domain,
                    default_color=color_valid,
                    layer=LAYER_VALID_RAYS,
                )
            ]

        # Else, split into colliding and non colliding rays using blocked mask
        else:
            valid = ~outputs.blocked

            group_valid = (
                [
                    viewer.render_rays(
                        inputs.P[valid],
                        outputs.P,
                        variables=ray_variables_dict(
                            inputs, ray_variables.variables, valid
                        ),
                        domain=ray_variables.domain,
                        default_color=color_valid,
                        layer=LAYER_VALID_RAYS,
                    )
                ]
                if inputs.P[valid].numel() > 0
                else []
            )

            P, V = inputs.P[outputs.blocked], inputs.V[outputs.blocked]
            if P.numel() > 0:
                group_blocked = render_rays_until(
                    P,
                    V,
                    inputs.target()[0],
                    variables=ray_variables_dict(
                        inputs, ray_variables.variables, outputs.blocked
                    ),
                    domain=ray_variables.domain,
                    default_color=color_blocked,
                    layer=LAYER_BLOCKED_RAYS,
                )

            else:
                group_blocked = []

            return group_valid + group_blocked


class FocalPointArtist:
    @staticmethod
    def render_module(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        ray_variables: RayVariables,
    ) -> list[Any]:

        target = input_tree[module].target().unsqueeze(0)
        return [viewer.render_points(target, color_focal_point)]

    @staticmethod
    def render_rays(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        ray_variables: RayVariables,
    ) -> list[Any]:
        inputs = input_tree[module]

        # Distance from ray origin P to target
        dist = torch.linalg.vector_norm(inputs.P - inputs.target(), dim=1)

        # Always draw rays in their positive t direction
        t = torch.abs(dist)
        return render_rays_length(
            inputs.P,
            inputs.V,
            t,
            layer=LAYER_VALID_RAYS,
            variables=ray_variables_dict(inputs, ray_variables.variables),
            domain=ray_variables.domain,
            default_color=color_valid,
        )


def render_joints(
    module: nn.Module,
    input_tree: dict[nn.Module, tlm.OpticalData],
    output_tree: dict[nn.Module, tlm.OpticalData],
    ray_variables: RayVariables,
) -> list[Any]:
    dim, dtype = (
        input_tree[module].transforms[0].dim,
        input_tree[module].transforms[0].dtype,
    )

    # Final transform list
    tflist = output_tree[module].transforms

    points = []

    for i in range(len(tflist)):
        tf = tlm.forward_kinematic(tflist[: i + 1])
        joint = tf.direct_points(torch.zeros((dim,), dtype=dtype))

        points.append(joint.tolist())

    return [{"type": "points", "data": points, "layers": [LAYER_JOINTS]}]


class EndArtist:
    def __init__(self, end: float):
        self.end = end

    def render_rays(
        self,
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        ray_variables: RayVariables,
    ) -> list[Any]:
        return render_rays_length(
            output_tree[module].P,
            output_tree[module].V,
            self.end,
            variables=ray_variables_dict(output_tree[module], ray_variables.variables),
            domain=ray_variables.domain,
            default_color=color_valid,
            layer=LAYER_OUTPUT_RAYS,
        )


class SequentialArtist:
    @staticmethod
    def render_module(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        ray_variables: RayVariables,
    ) -> list[Any]:
        nodes = []
        for child in module.children():
            nodes.extend(render_module(child, input_tree, output_tree, ray_variables))
        return nodes

    @staticmethod
    def render_rays(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        ray_variables: RayVariables,
    ) -> list[Any]:
        nodes = []
        for child in module.children():
            nodes.extend(render_rays(child, input_tree, output_tree, ray_variables))
        return nodes


class LensArtist:
    @staticmethod
    def render_module(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        ray_variables: RayVariables,
    ) -> list[Any]:
        nodes = []
        nodes.extend(
            render_module(module.surface1, input_tree, output_tree, ray_variables)
        )
        nodes.extend(
            render_module(module.surface2, input_tree, output_tree, ray_variables)
        )
        return nodes

    @staticmethod
    def render_rays(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        ray_variables: RayVariables,
    ) -> list[Any]:
        nodes = []
        nodes.extend(
            render_rays(module.surface1, input_tree, output_tree, ray_variables)
        )
        nodes.extend(
            render_rays(module.surface2, input_tree, output_tree, ray_variables)
        )
        return nodes


class SubTransformArtist:
    @staticmethod
    def render_module(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        ray_variables: RayVariables,
    ) -> list[Any]:
        nodes = []
        nodes.extend(
            render_module(module.element, input_tree, output_tree, ray_variables)
        )
        return nodes

    @staticmethod
    def render_rays(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        ray_variables: RayVariables,
    ) -> list[Any]:
        nodes = []
        nodes.extend(
            render_rays(module.element, input_tree, output_tree, ray_variables)
        )
        return nodes


artists_dict: Dict[type, type] = {
    nn.Sequential: SequentialArtist,
    tlm.FocalPoint: FocalPointArtist,
    tlm.LensBase: LensArtist,
    tlm.Offset: SubTransformArtist,
    tlm.Rotate: SubTransformArtist,
    tlm.KinematicSurface: KinematicSurfaceArtist,
    tlm.CollisionSurface: CollisionSurfaceArtist,
}


def inspect_stack(execute_list: list[tuple[nn.Module, Any, Any]]) -> None:
    for module, inputs, outputs in execute_list:
        print(type(module))
        print("inputs.transform:")
        for t in inputs.transforms:
            print(t)
        print()
        print("outputs.transform:")
        for t in outputs.transforms:
            print(t)
        print()


def render_module(
    module: nn.Module,
    input_tree: dict[nn.Module, tlm.OpticalData],
    output_tree: dict[nn.Module, tlm.OpticalData],
    ray_variables: RayVariables,
) -> list[Any]:
    # find matching artists for this module, use the first one for rendering
    artists: list[Any] = [
        a for typ, a in artists_dict.items() if isinstance(module, typ)
    ]

    if len(artists) == 0:
        return []

    artist = artists[0]
    nodes = []

    # Render element itself
    nodes.extend(artist.render_module(module, input_tree, output_tree, ray_variables))

    return nodes


def render_rays(
    module: nn.Module,
    input_tree: dict[nn.Module, tlm.OpticalData],
    output_tree: dict[nn.Module, tlm.OpticalData],
    ray_variables: RayVariables,
) -> list[Any]:
    # find matching artists for this module, use the first one for rendering
    artists: list[Any] = [
        a for typ, a in artists_dict.items() if isinstance(module, typ)
    ]

    if len(artists) == 0:
        return []

    artist = artists[0]
    nodes = []

    # Render rays
    nodes.extend(artist.render_rays(module, input_tree, output_tree, ray_variables))

    return nodes


def render_sequence(
    optics: nn.Module,
    dim: int,
    dtype: torch.dtype,
    sampling: dict[str, Any],
    end: Optional[float] = None,
    title: str = "",
) -> Any:
    input_tree, output_tree = tlm.forward_tree(
        optics, tlm.default_input(sampling, dim, dtype)
    )

    # Figure out available ray variables and their range, this will be used for coloring info by tlmviewer
    ray_variables = RayVariables.from_optical_data(input_tree.values())

    scene = viewer.new_scene("2D" if dim == 2 else "3D")

    # Render the top level module
    scene["data"].extend(render_module(optics, input_tree, output_tree, ray_variables))

    # Render rays
    scene["data"].extend(render_rays(optics, input_tree, output_tree, ray_variables))

    # Render kinematic chain joints
    scene["data"].extend(render_joints(optics, input_tree, output_tree, ray_variables))

    # Render output rays with end argument
    if end is not None:
        scene["data"].extend(
            EndArtist(end).render_rays(optics, input_tree, output_tree, ray_variables)
        )

    if title != "":
        scene["title"] = title

    return scene


def default_sampling(
    optics: nn.Module,
    dim: int,
    dtype: torch.dtype = torch.float64,
) -> dict[str, Any]:
    "Default sampling values"

    # TODO could be improved by looking at stack content, etc.
    return {"base": 10, "object": 5, "wavelength": 3}


def show(
    optics: nn.Module,
    sampling: Optional[Dict[str, Any]] = None,
    dim: int = 2,
    dtype: torch.dtype = torch.float64,
    end: Optional[float] = None,
    title: str = "",
    ndigits: int | None = 8,
    return_scene: bool = False,
) -> None:
    "Render an optical stack and show it with ipython display"

    if sampling is None:
        sampling = default_sampling(optics, dim, dtype)

    scene = render_sequence(optics, dim, dtype, sampling, end, title)

    viewer.display_scene(scene, ndigits)

    return scene if return_scene else None


def show2d(*args, **kwargs):
    kwargs["dim"] = 2
    return show(*args, **kwargs)


def show3d(*args, **kwargs):
    kwargs["dim"] = 3
    return show(*args, **kwargs)


def export_json(
    optics: nn.Module,
    filename: str,
    dim: int = 2,
    dtype: torch.dtype = torch.float64,
    sampling: Optional[Dict[str, Any]] = None,
    end: Optional[float] = None,
    title: str = "",
    ndigits: int | None = 8,
) -> None:
    "Render and export an optical stack to a tlmviewer json file"

    if sampling is None:
        # TODO figure out a better default based on stack content?
        sampling = {"base": 10, "object": 5, "wavelength": 8}

    scene = render_sequence(optics, dim, dtype, sampling, end, title)

    if ndigits is not None:
        scene = viewer.truncate_scene(scene, ndigits)

    with open(filename, "w") as f:
        json.dump(scene, f)
