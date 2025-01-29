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


def cartesian_prod2d(A: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    """
    Cartesian product of 2 batched coordinate tensors of shape (N, D) and (M, D)
    returns 2 Tensors of shape ( N*M , D )
    """

    if A.dim() == 1:
        A = A.unsqueeze(1)
    if B.dim() == 1:
        B = B.unsqueeze(1)

    assert A.dim() == B.dim() == 2
    assert A.shape[1] == B.shape[1]
    N, M = A.shape[0], B.shape[0]
    D = A.shape[1]

    A = torch.repeat_interleave(A, M, dim=0)
    B = torch.tile(B, (N, 1))

    assert A.shape == B.shape == (M * N, D)
    return A, B
