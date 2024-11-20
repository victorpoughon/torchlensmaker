import torch
import torch.nn as nn

import torchlensmaker as tlm


class Module(nn.Module):
    """
    Custom nn.Module to automatically register parameters of shapes

    This is similar to how PyTorch's nn.Module automatically registers nn.Parameters
    that are assigned to it. But here, we register tlm shapes and their inner parameters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().__setattr__("_shapes", {})

    def __getattr__(self, name: str):
        if name in self.__dict__["_shapes"]:
            return self.__dict__["_shapes"][name]
        else:
            return super().__getattr__(name)
    
    def __setattr__(self, name: str, value):
        if isinstance(value, tlm.Parabola):
            for parameter_name, parameter in value.parameters().items():
                self.register_parameter(name + "_" + parameter_name, parameter)
            self.__dict__["_shapes"][name] = value
        else:
            super().__setattr__(name, value)
