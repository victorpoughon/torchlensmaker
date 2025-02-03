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


LAYER_VALID_RAYS = 1
LAYER_BLOCKED_RAYS = 2
LAYER_OUTPUT_RAYS = 3
LAYER_JOINTS = 4


def render_rays_until(
    P: Tensor,
    V: Tensor,
    end: Tensor,
    default_color: str,
    layer: Optional[int] = None,
) -> list[Any]:
    "Render rays until an absolute X coordinate"
    assert end.dim() == 0
    t = (end - P[:, 0]) / V[:, 0]
    ends = P + t.unsqueeze(1).expand_as(V) * V
    return [tlm.viewer.render_rays(P, ends, default_color=default_color, layer=layer)]


def render_rays_length(
    P: Tensor,
    V: Tensor,
    length: float | Tensor,
    color_data: Optional[Tensor] = None,
    default_color: str = color_valid,
    layer: Optional[int] = None,
) -> list[Any]:
    "Render rays with fixed length"

    if isinstance(length, Tensor):
        assert length.dim() in {0, 1}

    if isinstance(length, Tensor) and length.dim() == 1:
        length = length.unsqueeze(1).expand_as(V)

    return [
        tlm.viewer.render_rays(
            P,
            P + length * V,
            color_data=color_data,
            default_color=default_color,
            layer=layer,
        )
    ]


def color_rays_tensor(data: tlm.OpticalData, color_dim: str) -> Tensor:
    if color_dim == "base":
        return data.rays_base
    elif color_dim == "object":
        return data.rays_object
    elif color_dim == "wavelength":
        return data.rays_wavelength
    # TODO check that returned tensor is not None?
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
    # unless the data range is too small, then use 0.5
    denom = var.max() - var.min()
    if denom > 1e-4:
        c = (var - var.min()) / denom
    else:
        c = torch.full_like(var, 0.5)

    # convert to rgb using color map
    return torch.tensor(colormap(c))


class KinematicSurfaceArtist:
    @staticmethod
    def render_module(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:

        dim, dtype = (
            input_tree[module].transforms[0].dim,
            input_tree[module].transforms[0].dtype,
        )
        chain = input_tree[module].transforms + module.surface_transform(dim, dtype)
        transform = tlm.forward_kinematic(chain)

        # TODO find a way to group surfaces together?
        return [
            tlm.viewer.render_surfaces(
                [module.surface], [transform], dim=transform.dim, N=100
            )
        ]

    @staticmethod
    def render_rays(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:
        return render_rays(module.element, input_tree, output_tree, color_dim, colormap)


class CollisionSurfaceArtist:
    @staticmethod
    def render_module(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:

        points = output_tree[module].P
        normals = output_tree[module].normals

        # return tlm.viewer.render_collisions(points, normals)
        return []

    @staticmethod
    def render_rays(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:
        inputs = input_tree[module]
        outputs = output_tree[module]
        # If rays are not blocked, render simply all rays from collision to collision
        if outputs.blocked is None:
            return [
                tlm.viewer.render_rays(
                    inputs.P,
                    outputs.P,
                    default_color=color_valid,
                    layer=LAYER_VALID_RAYS,
                )
            ]

        # Else, split into colliding and non colliding rays using blocked mask
        else:
            valid = ~outputs.blocked
            color_data = (
                color_rays(inputs, color_dim, colormap)[valid]
                if color_dim is not None
                else None
            )
            group_valid = (
                [
                    tlm.viewer.render_rays(
                        inputs.P[valid],
                        outputs.P,
                        color_data=color_data,
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
                    default_color=color_blocked,
                    layer=LAYER_BLOCKED_RAYS,
                )

            else:
                group_blocked = []

            return group_valid + group_blocked
            # Render non blocked rays


class FocalPointArtist:
    @staticmethod
    def render_module(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:

        target = input_tree[module].target().unsqueeze(0)
        return [tlm.viewer.render_points(target, color_focal_point)]

    @staticmethod
    def render_rays(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:

        # Distance from ray origin P to target
        dist = torch.linalg.vector_norm(
            input_tree[module].P - input_tree[module].target(), dim=1
        )

        # Always draw rays in their positive t direction
        t = torch.abs(dist)
        return render_rays_length(
            input_tree[module].P, input_tree[module].V, t, default_color=color_valid
        )


def render_joints(
    module: nn.Module,
    input_tree: dict[nn.Module, tlm.OpticalData],
    output_tree: dict[nn.Module, tlm.OpticalData],
    color_dim: Optional[str] = None,
    colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
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
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:
        return render_rays_length(
            output_tree[module].P,
            output_tree[module].V,
            self.end,
            color_data=color_rays(output_tree[module], color_dim, colormap),
            default_color=color_valid,
            layer=LAYER_OUTPUT_RAYS,
        )


class SequentialArtist:
    @staticmethod
    def render_module(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:
        nodes = []
        for child in module.children():
            nodes.extend(
                render_module(child, input_tree, output_tree, color_dim, colormap)
            )
        return nodes

    @staticmethod
    def render_rays(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:
        nodes = []
        for child in module.children():
            nodes.extend(
                render_rays(child, input_tree, output_tree, color_dim, colormap)
            )
        return nodes


class LensArtist:
    @staticmethod
    def render_module(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:
        nodes = []
        nodes.extend(
            render_module(module.surface1, input_tree, output_tree, color_dim, colormap)
        )
        nodes.extend(
            render_module(module.surface2, input_tree, output_tree, color_dim, colormap)
        )
        return nodes

    @staticmethod
    def render_rays(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:
        nodes = []
        nodes.extend(
            render_rays(module.surface1, input_tree, output_tree, color_dim, colormap)
        )
        nodes.extend(
            render_rays(module.surface2, input_tree, output_tree, color_dim, colormap)
        )
        return nodes


class SubTransformArtist:
    @staticmethod
    def render_module(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:
        nodes = []
        nodes.extend(
            render_module(module.element, input_tree, output_tree, color_dim, colormap)
        )
        return nodes

    @staticmethod
    def render_rays(
        module: nn.Module,
        input_tree: dict[nn.Module, tlm.OpticalData],
        output_tree: dict[nn.Module, tlm.OpticalData],
        color_dim: Optional[str] = None,
        colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    ) -> list[Any]:
        nodes = []
        nodes.extend(
            render_rays(module.element, input_tree, output_tree, color_dim, colormap)
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
    color_dim: Optional[str] = None,
    colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
) -> list[Any]:
    # find matching artists for this module, use the first one for rendering
    artists = [a for typ, a in artists_dict.items() if isinstance(module, typ)]

    if len(artists) == 0:
        return []

    artist = artists[0]
    nodes = []

    # Render element itself
    nodes.extend(
        artist.render_module(module, input_tree, output_tree, color_dim, colormap)
    )

    return nodes


def render_rays(
    module: nn.Module,
    input_tree: dict[nn.Module, tlm.OpticalData],
    output_tree: dict[nn.Module, tlm.OpticalData],
    color_dim: Optional[str] = None,
    colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
) -> list[Any]:
    # find matching artists for this module, use the first one for rendering
    artists = [a for typ, a in artists_dict.items() if isinstance(module, typ)]

    if len(artists) == 0:
        return []

    artist = artists[0]
    nodes = []

    # Render rays
    nodes.extend(
        artist.render_rays(module, input_tree, output_tree, color_dim, colormap)
    )

    return nodes


def render_sequence(
    optics: nn.Module,
    dim: int,
    dtype: torch.dtype,
    sampling: dict[str, Any],
    color_dim: Optional[str] = None,
    colormap: mpl.colors.LinearSegmentedColormap = default_colormap,
    end: Optional[float] = None,
) -> Any:
    input_tree, output_tree = tlm.forward_tree(
        optics, tlm.default_input(dim, dtype, sampling)
    )

    scene = tlm.viewer.new_scene("2D" if dim == 2 else "3D")

    # Render the top level module
    scene["data"].extend(
        render_module(optics, input_tree, output_tree, color_dim, colormap)
    )

    # Render rays
    scene["data"].extend(
        render_rays(optics, input_tree, output_tree, color_dim, colormap)
    )

    # Render kinematic chain joints
    scene["data"].extend(
        render_joints(optics, input_tree, output_tree, color_dim, colormap)
    )

    # Render output rays with end argument
    if end is not None:
        scene["data"].extend(
            EndArtist(end).render_rays(
                optics, input_tree, output_tree, color_dim, colormap
            )
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
        # TODO figure out a better default based on stack content?
        sampling = {"base": 10, "object": 5, "wavelength": 8}

    scene = tlm.viewer.render_sequence(
        optics, dim, dtype, sampling, color_dim, colormap, end
    )

    if dump:
        tlm.viewer.dump(scene, ndigits=2)

    tlm.viewer.ipython_display(scene, ndigits)
