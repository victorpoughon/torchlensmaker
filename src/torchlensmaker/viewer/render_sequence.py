import torch
import torch.nn as nn
import torchlensmaker as tlm

from typing import Literal, Any


class SurfaceArtist:
    @staticmethod
    def render_element(element: nn.Module, inputs: Any, _outputs: Any) -> Any:

        dim, dtype = inputs.transforms[0].dim, inputs.transforms[0].dtype
        chain = inputs.transforms + element.surface_transform(dim, dtype)
        transform = tlm.forward_kinematic(chain)

        # TODO find a way to group surfaces together?
        return tlm.viewer.render_surfaces(
            [element.surface], [transform], dim=transform.dim, N=10
        )

    @staticmethod
    def render_rays(element: nn.Module, inputs: Any, outputs: Any) -> Any:

        # TODO dim, dtype as argument
        dim, dtype = inputs.transforms[0].dim, inputs.transforms[0].dtype

        if dim == 2:
            rays_start = inputs.rays.get(["RX", "RY"])
            rays_end = outputs.rays.get(["RX", "RY"])
        else:
            rays_start = inputs.rays.get(["RX", "RY", "RZ"])
            rays_end = outputs.rays.get(["RX", "RY", "RZ"])

        return tlm.viewer.render_rays(rays_start, rays_end)


class JointArtist:
    @staticmethod
    def render_element(element: nn.Module, inputs: Any, _outputs: Any) -> Any:

        dim, dtype = inputs.transforms[0].dim, inputs.transforms[0].dtype
        transform = tlm.forward_kinematic(inputs.transforms)
        joint = transform.direct_points(torch.zeros((dim,), dtype=dtype))

        return {"type": "points", "data": [joint.tolist()]}


artists_dict = {
    tlm.OpticalSurface: SurfaceArtist,
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


def render_sequence(optics: nn.Module, sampling: dict[str, Any]) -> Any:
    dim, dtype = sampling["dim"], sampling["dtype"]
    execute_list, top_output = tlm.full_forward(optics, tlm.default_input(sampling))

    scene = tlm.viewer.new_scene("2D" if dim == 2 else "3D")

    # inspect_stack(execute_list)

    for module, inputs, outputs in execute_list:

        # render chain join position for every module
        scene["data"].append(JointArtist.render_element(module, inputs, outputs))

        # find matching artists for this module, use the first one for rendering
        artists = [a for typ, a in artists_dict.items() if isinstance(module, typ)]

        if len(artists) > 0:
            artist = artists[0]
            group = artist.render_element(module, inputs, outputs)
            scene["data"].append(group)

            if inputs.rays.numel() > 0:
                group = artist.render_rays(module, inputs, outputs)
                scene["data"].append(group)

    return scene
