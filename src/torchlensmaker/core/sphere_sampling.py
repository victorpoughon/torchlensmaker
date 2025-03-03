import torch

Tensor = torch.Tensor


def sphere_samples_angular(radius: float, start: float, end: float, N: int, dtype: torch.dtype) -> Tensor:
    "Angular sampling of a circular arc defined by radius"
    R = radius
    
    if R > 0:
        theta = torch.linspace(torch.pi - end, torch.pi - start, N, dtype=dtype)
    else:
        theta = torch.linspace(start, end, N, dtype=dtype)

    X = torch.abs(R) * torch.cos(theta) + R
    Y = torch.abs(R) * torch.sin(theta)

    return torch.stack((X, Y), dim=-1)


def sphere_samples_linear(curvature: float, start: float, end: float, N: int, dtype: torch.dtype) -> Tensor:
    "Linear sampling of a circular arc defined by curvature"
    Y = torch.linspace(start, end, N, dtype=dtype)
    Y2 = Y**2
    C = curvature

    X = torch.div(C * Y2, 1 + torch.sqrt(1 - Y2 * C**2))
    return torch.stack((X, Y), dim=-1)
