import torch


def safe_sign(x: torch.Tensor) -> torch.Tensor:
    "Like torch.sign() but equals 1 at 0"
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)


def safe_sqrt(radicand: torch.Tensor) -> torch.Tensor:
    """
    Gradient safe version of torch.sqrt() that returns 0 where radicand <= 0
    """
    ok = radicand > 0
    safe = torch.zeros_like(radicand)
    return torch.sqrt(torch.where(ok, radicand, safe))


def safe_div(dividend: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    """
    Gradient safe version of torch.div() that returns dividend where divisor == 0
    """

    ok = divisor != torch.zeros((), dtype=divisor.dtype, device=divisor.device)
    safe = torch.ones_like(divisor)
    return torch.div(dividend, torch.where(ok, divisor, safe))


def bbroad(vector: torch.Tensor, nbatch: int) -> torch.Tensor:
    """
    Expands a tensor to be compatible with the dimensions of a batched tensor
    by appending batch dimensions as needed.

    Args:
    * vector: A tensor of shape M
    * nbatch: Number of dimensions of some batched tensor

    Returns:
    * A view of the vector tensor with shape (*M, ...) that is broadcastable
      with the batched tensor.
    """
    return vector.view(*vector.shape, *([1] * nbatch))
