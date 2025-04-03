from torchlensmaker.materials import default_material_models_dict
import torch


def test_refractive_index() -> None:
    W = torch.linspace(400, 800, 10)
    for name, model in default_material_models_dict.items():
        print(name)
        idx = model.refractive_index(W)
        print(idx)
        assert torch.all(idx >= 1.0)
        assert torch.all(idx <= 10.0)
