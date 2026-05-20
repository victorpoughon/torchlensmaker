import torch


def safe_sign(x: torch.Tensor) -> torch.Tensor:
    "Like torch.sign() but equals 1 at 0"
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)


def safe_sqrt(radicand: torch.Tensor) -> torch.Tensor:
    """
    Gradient safe version of torch.sqrt() that returns 0 where radicand <= 0.
    Uses the double-where trick: substitute 1 (not 0) so sqrt always sees a
    positive value and its gradient stays finite, then zero out the output
    for non-positive inputs. This avoids 0*inf=NaN in backward.
    """
    ok = radicand > 0
    safe = torch.sqrt(torch.where(ok, radicand, torch.ones_like(radicand)))
    return torch.where(ok, safe, torch.zeros_like(radicand))


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
