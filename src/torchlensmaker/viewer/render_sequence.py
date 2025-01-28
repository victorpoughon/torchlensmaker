import torch
import torch.nn as nn
import torchlensmaker as tlm

from typing import Literal, Any, Optional, Type, Dict, TypeVar

from torchlensmaker.tensorframe import TensorFrame

import matplotlib as mpl
import colorcet as cc

Tensor = torch.Tensor

# Color theme
color_valid = "#ffa724"
color_blocked = "red"
color_focal_point = "red"

default_colormap = cc.cm.CET_I2


def render_rays_until(
    P: Tensor, V: Tensor, end: Tensor, default_color: str
) -> list[Any]:
    "Render rays until an absolute X coordinate"

    t = (end - P[:, 0]) / V[:, 0]
    ends = P + t.unsqueeze(1).expand_as(V) * V
    return [tlm.viewer.render_rays(P, ends, default_color=default_color)]


def render_rays_length(
    P: Tensor,
    V: Tensor,
    length: float | Tensor,
    color_data: Optional[Tensor] = None,
    default_color: str = color_valid,
) -> list[Any]:
    "Render rays with fixed length"

    if isinstance(length, Tensor):
        assert length.dim() in {0, 1}

    if isinstance(length, Tensor) and length.dim() == 1:
        length = length.unsqueeze(1).expand_as(V)

    return [
        tlm.viewer.render_rays(
            P, P + length * V, color_data=color_data, default_color=default_color
        )
    ]


def color_rays_tensor(data: tlm.OpticalData, color_dim: str) -> Tensor:
    if color_dim == "base":
        return data.rays_base
    elif color_dim == "object":
        return data.rays_object
    else:
        raise RuntimeError(f"Unknown color dimension '{color_dim}'")


def color_rays(
    data: tlm.OpticalData,
    color_dim: Optional[str],
    colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
) -> Optional[Tensor]:
    if color_dim is None:
        return None

    color_tensor = color_rays_tensor(data, color_dim)

    # unsqueeze to 2D
    if color_tensor.dim() == 1:
        color_tensor = color_tensor.unsqueeze(1)

    assert color_tensor.dim() == 2
    assert color_tensor.shape[1] in {1, 2}

    # Ray variables that we use for coloring can be 2D when simulating in 3D
    # TODO more configurability here

    if color_tensor.shape[1] == 1:
        var = color_tensor[:, 0]
    else:
        # TODO 2D colormap?
        var = torch.linalg.vector_norm(color_tensor, dim=1)

    # normalize color variable to [0, 1]
    c = (var - var.min()) / (var.max() - var.min())

    # convert to rgb using color map
    return torch.tensor(colormap(c))


class SurfaceArtist:
    @staticmethod
    def render_element(
        element: nn.Module, inputs: Any, _outputs: Any, color_dim: Optional[str] = None
    ) -> list[Any]:

        dim, dtype = inputs.transforms[0].dim, inputs.transforms[0].dtype
        chain = inputs.transforms + element.surface_transform(dim, dtype)
        transform = tlm.forward_kinematic(chain)

        # TODO find a way to group surfaces together?
        return [
            tlm.viewer.render_surfaces(
                [element.surface], [transform], dim=transform.dim, N=100
            )
        ]

    @staticmethod
    def render_rays(
        element: nn.Module,
        inputs: Any,
        outputs: Any,
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:

        # If rays are not blocked, render simply all rays from collision to collision
        if outputs.blocked is None:
            return [
                tlm.viewer.render_rays(inputs.P, outputs.P, default_color=color_valid)
            ]

        # Else, split into colliding and non colliding rays using blocked mask
        else:
            valid = ~outputs.blocked
            color_data = color_rays(inputs, color_dim, colormap)[valid] if color_dim is not None else None
            group_valid = (
                [
                    tlm.viewer.render_rays(
                        inputs.P[valid],
                        outputs.P,
                        color_data=color_data,
                        default_color=color_valid,
                    )
                ]
                if inputs.P[valid].numel() > 0
                else []
            )

            P, V = inputs.P[outputs.blocked], inputs.V[outputs.blocked]
            if P.numel() > 0:
                dim, dtype = inputs.transforms[0].dim, inputs.transforms[0].dtype
                chain = inputs.transforms + element.surface_transform(dim, dtype)
                transform = tlm.forward_kinematic(chain)
                target = transform.direct_points(torch.zeros(1, dim, dtype=dtype))[0]

                group_blocked = render_rays_until(
                    P, V, target[0], default_color=color_blocked
                )

            else:
                group_blocked = []

            return group_valid + group_blocked
            # Render non blocked rays


class FocalPointArtist:
    @staticmethod
    def render_element(
        element: nn.Module, inputs: tlm.OpticalData, _outputs: tlm.OpticalData
    ) -> list[Any]:

        target = inputs.target().unsqueeze(0)
        return [tlm.viewer.render_points(target, color_focal_point)]

    @staticmethod
    def render_rays(
        element: nn.Module,
        inputs: Any,
        outputs: Any,
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:

        # Distance from ray origin P to target
        dist = torch.linalg.vector_norm(inputs.P - inputs.target(), dim=1)

        # Always draw rays in their positive t direction
        t = torch.abs(dist)
        return render_rays_length(inputs.P, inputs.V, t, default_color=color_valid)


class ApertureArtist:
    @staticmethod
    def render_element(
        element: nn.Module, inputs: tlm.OpticalData, _outputs: tlm.OpticalData
    ) -> list[Any]:

        target = inputs.target().unsqueeze(0)
        return [tlm.viewer.render_points(target, color_focal_point)]

    @staticmethod
    def render_rays(
        element: nn.Module,
        inputs: Any,
        outputs: Any,
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:
        return SurfaceArtist.render_rays(element, inputs, outputs, color_dim, colormap)


class JointArtist:
    @staticmethod
    def render_element(element: nn.Module, inputs: Any, _outputs: Any) -> list[Any]:

        dim, dtype = inputs.transforms[0].dim, inputs.transforms[0].dtype
        transform = tlm.forward_kinematic(inputs.transforms)
        joint = transform.direct_points(torch.zeros((dim,), dtype=dtype))

        return [{"type": "points", "data": [joint.tolist()]}]


class EndArtist:
    def __init__(self, end: float):
        self.end = end

    def render_rays(
        self,
        element: nn.Module,
        inputs: Any,
        outputs: Any,
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:
        return render_rays_length(
            outputs.P,
            outputs.V,
            self.end,
            color_data=color_rays(outputs, color_dim, colormap),
            default_color=color_valid,
        )


artists_dict: Dict[type, type] = {
    tlm.OpticalSurface: SurfaceArtist,
    tlm.FocalPoint: FocalPointArtist,
    # tlm.Aperture: ApertureArtist,
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


def render_sequence(
    optics: nn.Module,
    dim: int,
    dtype: torch.dtype,
    sampling: dict[str, Any],
    color_dim: Optional[str] = None,
    colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    end: Optional[float] = None,
) -> Any:
    execute_list, top_output = tlm.full_forward(
        optics, tlm.default_input(dim, dtype, sampling)
    )

    scene = tlm.viewer.new_scene("2D" if dim == 2 else "3D")

    # Render elements
    for module, inputs, outputs in execute_list:

        # render chain join position for every module
        scene["data"].extend(JointArtist.render_element(module, inputs, outputs))

        # find matching artists for this module, use the first one for rendering
        artists = [a for typ, a in artists_dict.items() if isinstance(module, typ)]

        if len(artists) > 0:
            artist = artists[0]
            scene["data"].extend(artist.render_element(module, inputs, outputs))

            if inputs.P.numel() > 0:
                scene["data"].extend(
                    artist.render_rays(module, inputs, outputs, color_dim, colormap)
                )

    # Render output rays
    if end is not None:
        scene["data"].extend(
            EndArtist(end).render_rays(module, inputs, outputs, color_dim, colormap)
        )

    return scene


def ipython_show(
    optics: nn.Module,
    dim: int,
    dtype: torch.dtype = torch.float64,
    sampling: Optional[Dict[str, Any]] = None,
    color_dim: Optional[str] = None,
    colormap: mpl.colors.Colormap = default_colormap,
    end: Optional[float] = None,
    dump: bool = False,
    ndigits: int | None = 4,
) -> None:

    if sampling is None:
        sampling = {"base": 10, "object": 5}

    scene = tlm.viewer.render_sequence(
        optics, dim, dtype, sampling, color_dim, colormap, end
    )

    if dump:
        tlm.viewer.dump(scene, ndigits=2)

    tlm.viewer.ipython_display(scene, ndigits)
