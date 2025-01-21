import torch

from torchlensmaker.surfaces import LocalSurface

from torchlensmaker.transforms import (
    TransformBase,
    LinearTransform,
    TranslateTransform,
    ComposeTransform,
)

from typing import Callable

from torchlensmaker.rot3d import euler_angles_to_matrix
from torchlensmaker.rot2d import rotation_matrix_2D


def basic_transform(
    scale: float,
    anchor: str,
    thetas: float | list[float],
    translate: list[float],
    dtype: torch.dtype = torch.float64,
) -> Callable[[LocalSurface], TransformBase]:
    """
    Compound transform used for testing

    Transform is of the form: Y = RS(X - A) + T
    
    Returns a function foo(surface)
    """

    if isinstance(thetas, list) and len(translate) == 3:
        dim = 3
    elif isinstance(thetas, (float, int)) and len(translate) == 2:
        dim = 2
    else:
        raise RuntimeError("invalid arguments to basic_transform")

    def makeit(surface: LocalSurface) -> TransformBase:
        # anchor
        anchor_translate = surface.extent(dim, dtype)
        transforms: list[TransformBase] = (
            [TranslateTransform(-anchor_translate)] if anchor == "extent" else []
        )

        # scale
        Md = torch.eye(dim, dtype=dtype) * scale
        Mi = torch.eye(dim, dtype=dtype) * 1 / scale
        transforms.append(LinearTransform(Md, Mi))

        # rotate
        if dim == 2:
            Mr = rotation_matrix_2D(torch.as_tensor(thetas, dtype=dtype))
        else:
            Mr = euler_angles_to_matrix(
                torch.deg2rad(torch.as_tensor(thetas, dtype=dtype)), "XYZ"
            ).to(
                dtype=dtype
            )  # TODO need to support dtype in euler_angles_to_matrix

        transforms.append(LinearTransform(Mr, Mr.T))

        # translate
        transforms.append(TranslateTransform(torch.as_tensor(translate, dtype=dtype)))

        return ComposeTransform(transforms)

    return makeit
