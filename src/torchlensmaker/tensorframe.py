import torch
from typing import Iterable


class TensorFrame:
    "A 2D tensor with named columns"

    data: torch.Tensor
    columns: Iterable[str]

    def __init__(self, data: torch.Tensor, columns: Iterable[str]):
        self.data = data
        self.columns = list(columns)

    def __repr__(self):
        return f"TensorFrame:\ndata:\n{self.data.__repr__()}\ncolumns:\n{self.columns}"

    @property
    def shape(self):
        return self.data.shape

    def numel(self) -> int:
        return self.data.numel()

    def get(self, names: str | Iterable[str]):
        try:
            if isinstance(names, str):
                idx = self.columns.index(names)
            else:
                idx = [self.columns.index(n) for n in names]
            return self.data[:, idx]
        except:
            raise KeyError(f"TensorFrame doesn't have column(s): {names}")

    def masked(self, mask):
        return TensorFrame(self.data[mask], self.columns)
    
    def stack(self, other):
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


    def update(self, **kwargs):
        "Return a new TensorFrame with updated or inserted columns"

        new_cols = kwargs.keys() - set(self.columns)
        merged_cols = list(self.columns) + list(new_cols)

        new_data = torch.stack(
            tuple(
                kwargs[col] if col in kwargs else self.data[:, self.columns.index(col)]
                for col in merged_cols
            ),
            dim=1,
        )

        return TensorFrame(new_data, merged_cols)
