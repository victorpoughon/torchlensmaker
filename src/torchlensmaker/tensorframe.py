from __future__ import annotations

import torch
from typing import Iterable, List, Dict
import numbers


class TensorFrame:
    "A 2D tensor with named columns"

    data: torch.Tensor
    columns: List[str]

    def __init__(self, data: torch.Tensor, columns: Iterable[str]):
        self.data = data
        self.columns = list(columns)
        assert len(columns) == data.shape[1]

    def __repr__(self) -> str:
        return f"TensorFrame:\ndata:\n{repr(self.data)}\ncolumns:\n{self.columns}"

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def numel(self) -> int:
        return self.data.numel()

    def get(self, names: str | Iterable[str]) -> torch.Tensor:
        try:
            idx: int | list[int]
            if isinstance(names, str):
                idx = self.columns.index(names)
            else:
                idx = [self.columns.index(n) for n in names]
            return self.data[:, idx]
        except:
            raise KeyError(f"TensorFrame doesn't have column(s): {names}")

    def masked(self, mask: torch.Tensor) -> TensorFrame:
        return TensorFrame(self.data[mask], self.columns)

    def stack(self, other: TensorFrame) -> TensorFrame:
        """
        Return a new TensorFrame that stacks self with other
        Both columns must be identical, unless one of the tensor frames is empty
        """

        if self.numel() == 0:
            return other
        elif other.numel() == 0:
            return self
        else:
            assert self.columns == other.columns
            return TensorFrame(torch.cat((self.data, other.data), dim=0), self.columns)

    def update(self, **kwargs: float | torch.Tensor) -> TensorFrame:
        "Return a new TensorFrame with updated or inserted columns"

        N = self.data.shape[0]
        new_cols = kwargs.keys() - set(self.columns)
        merged_cols = list(self.columns) + list(new_cols)

        # convert all values to tensor of shape (N,)
        kwargs_tensor: Dict[str, torch.Tensor] = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                kwargs_tensor[k] = v.unsqueeze(0).expand((N,))
            elif isinstance(v, torch.Tensor) and v.shape == (1,):
                kwargs_tensor[k] = v.expand((N,))
            elif isinstance(v, torch.Tensor):
                kwargs_tensor[k] = v
            elif isinstance(v, numbers.Number):
                kwargs_tensor[k] = torch.full((N,), v)
            else:
                raise TypeError(
                    "Unsupported type in TensorFrame.update(): " + type(v).__name__
                )

            assert kwargs_tensor[k].shape == (N,)

        new_data = torch.stack(
            tuple(
                (
                    kwargs_tensor[col]
                    if col in kwargs_tensor
                    else self.data[:, self.columns.index(col)]
                )
                for col in merged_cols
            ),
            dim=1,
        )

        return TensorFrame(new_data, merged_cols)
