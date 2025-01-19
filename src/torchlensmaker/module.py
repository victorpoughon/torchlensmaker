import torch
import torch.nn as nn

import torchlensmaker as tlm

from typing import Any, Union


# TODO could this class be removed is surfaces are also modules?
# that way when assigning parameter to surface, then surface to opticalsurface,
# it all ends up working with standard pytorch?

class Module(nn.Module):
    """
    Overload of nn.Module to automatically register parameters of surfaces

    This is similar to how PyTorch's nn.Module automatically registers nn.Parameters
    that are assigned to it. But here, we register tlm surfaces and their inner parameters.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.__dict__["_shapes"] = {}

    def __getattr__(self, name: str) -> Any:
        # if you get an error here, check that you called super().__init__
        # in your custom tlm.Module
        if name in self.__dict__["_shapes"]:
            return self.__dict__["_shapes"][name]
        else:
            return super().__getattr__(name)

    def __setattr__(
        self, name: str, value: Union[torch.Tensor, torch.nn.modules.module.Module]
    ) -> None:
        if isinstance(value, tlm.surfaces.LocalSurface):
            for parameter_name, parameter in value.parameters().items():
                self.register_parameter(name + "_" + parameter_name, parameter)
            self.__dict__["_shapes"][name] = value
        else:
            super().__setattr__(name, value)
