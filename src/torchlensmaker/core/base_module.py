# This file is part of Torch Lens Maker
# Copyright (C) 2024-present Victor Poughon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import TYPE_CHECKING, Any, Callable, Self

import torch.nn as nn
from torch.nn.modules.container import Sequential

from torchlensmaker.sequential.sequential_data import SequentialData

if TYPE_CHECKING:
    from torchlensmaker.sequential.model_trace import ModelTrace


class BaseModule(nn.Module):
    """
    Base class for tlm modules
    """

    def __init__(self):
        super().__init__()
        self.trace_key: str | None = None

    def clone(self, **overrides: Any) -> Self:
        raise NotImplementedError(
            f"clone() not implemented for {type(self).__qualname__}"
        )

    def reverse(self) -> Self:
        raise NotImplementedError(
            f"reverse() not implemented for {type(self).__qualname__}"
        )

    def sequential(self, data: SequentialData) -> SequentialData:
        raise NotImplementedError(
            f"sequential() not implemented for {type(self).__qualname__}"
        )

    def trace(self, trace: "ModelTrace", key: str, inputs: Any, outputs: Any) -> Any:
        return


class MultiForwardModule(BaseModule):
    """
    Enable defining multiple forward functions with the @multiforward decorator
    and still have hooks called correctly
    """

    def forward(
        self, actual_forward: Callable[[Any], Any], *args: Any, **kwargs: Any
    ) -> Any:
        if not callable(actual_forward):
            raise TypeError(
                f"'{type(actual_forward).__qualname__}' is not callable. Did you forget to call the multiforward function?"
            )
        return actual_forward(*args, **kwargs)


def multiforward(new_forward_func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """
    Decorator to create a multiforward function
    """

    def wrapper(self: MultiForwardModule, *args: Any, **kwargs: Any) -> Any:
        return self(new_forward_func, self, *args, **kwargs)

    return wrapper
