import torch

from torchimplicit.types import ImplicitFunction

_all_functions_definitions: dict[str, ImplicitFunction] = {}


def register_function(f: ImplicitFunction):
    _all_functions_definitions[f.name] = f


def get_functions(dim: int | None = None) -> dict[str, ImplicitFunction]:
    if dim is None:
        return dict(_all_functions_definitions)
    return {k: v for k, v in _all_functions_definitions.items() if v.dim == dim}


def example_empty():
    def f(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.tensor((), dtype=dtype, device=device)

    return f


def example_scalar(x: float):
    def f(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.tensor([x], dtype=dtype, device=device)

    return f
