import torch

from torchlensmaker.core.tensor_manip import is_integral


"""
intari: Interval arithmetic

Functions for estimating and performing algebra operations on inf and sup
bounds.

Bounds are represented as a batch tensor, where the last dimension is 2 and
contains (inf, sup) bounds. The other dimensions are user-defined.
"""

def scalar(c: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
    """
    Interval arithmetic: scalar multiply

    Args:
        c: Tensor of scalars of shape (...), same batch dimensions as bounds
        bounds: Input tensor of shape (..., 2), where the last dimension are (inf, sup) bounds
    """

    assert bounds.shape[-1] == 2
    assert bounds.shape[:-1] == c.shape

    scaled = c.unsqueeze(-1) * bounds

    # Compute new bounds (handles sign flip via min/max)
    new_inf = torch.min(scaled, dim=-1).values
    new_sup = torch.max(scaled, dim=-1).values

    return torch.stack((new_inf, new_sup), dim=-1)


def sum(bounds: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Interval arithmetic: sum

    Args:
        bounds: Batch tensor of shape (..., 2) representing bounds [inf, sup]
        dim: Dimension to reduce

    Returns:
        Tensor: A tensor with one fewer dimension than the input,
                containing [inf_sum, sup_sum] for the specified dimension.
    """
    # Sum lower bounds (inf) and upper bounds (sup) along the specified dimension
    lower_sum = torch.sum(bounds[..., 0], dim=dim)
    upper_sum = torch.sum(bounds[..., 1], dim=dim)

    # Stack results into a new tensor with shape (..., 2)
    return torch.stack((lower_sum, upper_sum), dim=-1)


def product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Interval arithmetic: product

    Args:
        a: Batched tensor of shape (..., 2) representing bounds [a_inf, a_sup]
        b: Batched tensor of shape (..., 2) representing bounds [b_inf, b_sup]

    Returns:
        Batched tensor of shape (..., 2) representing [inf, sup] bounds of the product a*b
    """
    # Compute all combinations of products between bounds
    products = torch.stack(
        [
            a[..., 0] * b[..., 0],  # a_inf * b_inf
            a[..., 0] * b[..., 1],  # a_inf * b_sup
            a[..., 1] * b[..., 0],  # a_sup * b_inf
            a[..., 1] * b[..., 1],  # a_sup * b_sup
        ],
        dim=-1,
    )  # Shape (..., 4)

    # Determine new bounds by taking min and max along the last dimension
    new_inf = torch.min(products, dim=-1).values
    new_sup = torch.max(products, dim=-1).values

    return torch.stack([new_inf, new_sup], dim=-1)  # Shape (..., 2)


def monomial(p: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Interval arithmetic: bounds of x^p for x ∈ [-tau, tau]

    Args:
        p: Non-negative integer exponents, tensor of shape (...)
        tau: Zero-dim tensor such that x ∈ [-tau, tau]

    Returns:
        Tensor of shape (..., 2) representing [inf, sup] bounds for each exponent in p
    """

    if not is_integral(p.dtype):
        raise ValueError(f"Expected integral dtype for p, got {p.dtype}")

    if not torch.all(p >= 0):
        raise ValueError("All exponents must be non negative")

    zero_int = torch.zeros((1), dtype=p.dtype)

    taup = torch.pow(tau, p)
    bounds_zero = torch.tensor([1.0, 1.0], dtype=tau.dtype)
    bounds_even = torch.stack((torch.zeros_like(p), taup), dim=-1)
    bounds_odd = torch.stack((-taup, taup), dim=-1)

    return torch.where(
        (p == zero_int).unsqueeze(-1).expand_as(bounds_even),
        bounds_zero,
        torch.where(
            (p % 2 == zero_int).unsqueeze(-1).expand_as(bounds_even),
            bounds_even,
            bounds_odd,
        ),
    )

