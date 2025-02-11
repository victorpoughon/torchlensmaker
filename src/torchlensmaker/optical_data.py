import torch

from dataclasses import dataclass, replace

from typing import Any, Optional, TypeAlias

from torchlensmaker.transforms import (
    TransformBase,
    IdentityTransform,
    forward_kinematic,
)

from torchlensmaker.materials import (
    MaterialModel,
    get_material_model,
)

from torchlensmaker.sampling.samplers import Sampler, init_sampling

Tensor: TypeAlias = torch.Tensor


@dataclass
class OpticalData:
    # dim is 2 or 3
    # dtype default is torch.float64
    dim: int
    dtype: torch.dtype

    # Sampling configuration for each variable
    sampling: dict[str, Sampler]

    # Transform kinematic chain
    transforms: list[TransformBase]

    # Parametric light rays P + tV
    # Tensors of shape (N, 2|3)
    P: Tensor
    V: Tensor

    # Surface normals at the rays origin
    # Present if rays collided with a surface in the previous element
    normals: Optional[Tensor]

    # Rays variables
    # Tensors of shape (N, 2|3) or None
    rays_base: Optional[Tensor]
    rays_object: Optional[Tensor]
    rays_image: Optional[Tensor]
    rays_wavelength: Optional[Tensor]

    # Basis of each sampling variable
    # Tensors of shape (*, 2|3)
    # number of rows is the size of each sampling dimension
    var_base: Optional[Tensor]
    var_object: Optional[Tensor]
    var_wavelength: Optional[Tensor]

    # Material model for this batch of rays
    material: MaterialModel

    # Mask array indicating which rays from the previous data in the sequence
    # were blocked by the previous optical element. "blocked" includes hitting
    # an absorbing surface but also not hitting anything
    # None or Tensor of shape (N,)
    blocked: Optional[Tensor]

    # Loss accumulator
    # Tensor of dim 0
    loss: torch.Tensor

    def tf(self) -> TransformBase:
        return forward_kinematic(self.transforms)

    def target(self) -> Tensor:
        return self.tf().direct_points(torch.zeros((self.dim,), dtype=self.dtype))

    def replace(self, /, **changes: Any) -> "OpticalData":
        return replace(self, **changes)


def default_input(
    sampling: dict[str, Any],
    dim: int,
    dtype: torch.dtype = torch.float64,
) -> OpticalData:
    return OpticalData(
        dim=dim,
        dtype=dtype,
        sampling=init_sampling(sampling),
        transforms=[IdentityTransform(dim, dtype)],
        P=torch.empty((0, dim), dtype=dtype),
        V=torch.empty((0, dim), dtype=dtype),
        normals=None,
        rays_base=None,
        rays_object=None,
        rays_image=None,
        rays_wavelength=None,
        var_base=None,
        var_object=None,
        var_wavelength=None,
        material=get_material_model("vacuum"),
        blocked=None,
        loss=torch.tensor(0.0, dtype=dtype),
    )
