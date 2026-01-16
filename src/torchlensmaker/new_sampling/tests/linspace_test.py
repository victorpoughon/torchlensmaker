import torch
import torch.onnx


class LinspaceModel(torch.nn.Module):
    def forward(self, start, end, steps):
        return torch.linspace(start, end, steps)


x = torch.tensor(3, dtype=torch.float)
y = torch.tensor(10, dtype=torch.float)
z = torch.tensor(5, dtype=torch.int)


# Example usage and export
model = LinspaceModel()
example_input = (x, y, z)
samples = model(x, y, z)
torch.onnx.export(model, example_input, "linspace.onnx", dynamo=False)
