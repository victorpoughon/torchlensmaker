import torch
from copy import copy

Tensor = torch.Tensor

default_material_model_name = "(unnamed)"


class MaterialModel:
    def __init__(self, name: str = default_material_model_name):
        self.name = name

    def is_dispersive(self) -> bool:
        raise NotImplementedError

    def refractive_index(self, W: Tensor) -> Tensor:
        "Refractive index function of wavelength (in nm)"
        raise NotImplementedError


class NonDispersiveMaterial(MaterialModel):
    def __init__(self, n: float, name: str = default_material_model_name):
        super().__init__(name)
        self.n = n

    def is_dispersive(self):
        return False

    def refractive_index(self, W: Tensor) -> Tensor:
        return torch.full_like(W, self.n)

    def __str__(self) -> str:
        return f"{type(self).__name__}(name={repr(self.name)}, n={self.n})"


class CauchyMaterial(MaterialModel):
    def __init__(
        self,
        A: float,
        B: float,
        C: float = 0,
        D: float = 0,
        name: str = default_material_model_name,
    ):
        super().__init__(name)
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def is_dispersive(self):
        return True

    def refractive_index(self, W: Tensor) -> Tensor:
        Wmicro = W / 1000
        return (
            torch.full_like(Wmicro, self.A)
            + self.B / torch.pow(Wmicro, 2)
            + self.C / torch.pow(Wmicro, 4)
            + self.D / torch.pow(Wmicro, 6)
        )

    def __str__(self) -> str:
        return f"{type(self).__name__}(name={repr(self.name)}, A={self.A}, B={self.B}, C={self.C}, D={self.D})"


# TODO add:
# SellmeierMaterial
# LinearSegmentedMaterial


# These are "default" materials in the sense that they are the models
# that will be picked when using a string material argument, as in: material="water"
# any material argument can also be a MaterialModel object directly for more advanced usage
default_material_models = [
    NonDispersiveMaterial(1.0, name="vacuum"),
    NonDispersiveMaterial(1.00027, name="air"),
    # Bashkatov, Alexey & Genina, Elina. (2003).
    # Water refractive index in dependence on temperature and wavelength:
    # A simple approximation. Proceedings of SPIE
    # The International Society for Optical Engineering. 5068. 10.1117/12.518857.
    CauchyMaterial(1.31984, 0.005190553, name="water20C"),
    CauchyMaterial(
        1.31044,
        0.0050572226,
        name="water80C",
    ),
    CauchyMaterial(
        1.31984,
        0.005190553,
        name="water",
    ),
    CauchyMaterial(1.31044, 0.0050572226, name="Tea Earl Grey Hot"),
    # https://en.wikipedia.org/wiki/Cauchy%27s_equation
    CauchyMaterial(1.4580, 0.00354, name="Fused Silica"),
    CauchyMaterial(1.5046, 0.00420, name="BK7"),
    CauchyMaterial(1.5220, 0.00459, name="K5"),
    CauchyMaterial(1.5690, 0.00531, name="BaK4"),
    CauchyMaterial(1.6700, 0.00743, name="BaF10"),
    CauchyMaterial(1.7280, 0.013420, name="SF10"),
]

# Generate -nd suffix versions for "non dispersive" alternatives
# by taking the refractive index at 500nm
for model in copy(default_material_models):
    if model.is_dispersive():
        name = model.name + "-nd"
        n = model.refractive_index(torch.tensor(500))
        default_material_models.append(NonDispersiveMaterial(n, name=name))

default_material_models_dict = {
    model.name: model
    for model in default_material_models
    if model.name != default_material_model_name
}


def get_material_model(material: str | MaterialModel) -> MaterialModel:
    if isinstance(material, MaterialModel):
        return material

    try:
        return default_material_models_dict[material]
    except KeyError:
        raise ValueError(f"No material model for '{material}'")
