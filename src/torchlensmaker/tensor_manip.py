import torch
from typing import Optional

Tensor = torch.Tensor


def to_tensor(
    val: int | float | torch.Tensor,
    default_dtype: torch.dtype = torch.float64,
) -> Tensor:
    if not isinstance(val, torch.Tensor):
        return torch.as_tensor(val, dtype=default_dtype)
    return val


def cat_optional(a: Optional[Tensor], b: Optional[Tensor]) -> Optional[Tensor]:
    "something something monad"

    if a is None and b is None:
        return None
    else:
        if a is None:
            return b
        elif b is None:
            return a
        else:
            return torch.cat((a, b), dim=0)


def filter_optional_tensor(t: Optional[Tensor], valid: Tensor) -> Optional[Tensor]:
    if t is None:
        return None
    else:
        return t[valid]


def cartesian_prod2d(A: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    """
    Cartesian product of 2 batched coordinate tensors of shape (N, D) and (M, E)
    returns 2 Tensors of shape (N*M , D) and (N*M, E)
    """

    if A.dim() == 1:
        A = A.unsqueeze(1)
    if B.dim() == 1:
        B = B.unsqueeze(1)

    assert A.dim() == B.dim() == 2
    N, M = A.shape[0], B.shape[0]
    D, E = A.shape[1], B.shape[1]

    PA = torch.repeat_interleave(A, M, dim=0)
    PB = torch.tile(B, (N, 1))

    assert PA.shape == (M * N, D)
    assert PB.shape == (M * N, E)
    return PA, PB
