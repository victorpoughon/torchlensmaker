import torch

Tensor = torch.Tensor


class MaterialModel:
    def __init__(self, name: str):
        self.name = name

    def refractive_index(self, W: Tensor) -> Tensor:
        "Refractive index function of wavelength (in nm)"
        raise NotImplementedError


class NonDispersiveMaterial(MaterialModel):
    def __init__(self, name: str, n: float):
        super().__init__(name)
        self.n = n

    def refractive_index(self, W: Tensor) -> Tensor:
        return torch.full_like(W, self.n)

    def __str__(self) -> str:
        return f"{type(self).__name__}(name={repr(self.name)}, n={self.n})"


class CauchyMaterial(MaterialModel):
    def __init__(self, name: str, A: float, B: float, C: float = 0, D: float = 0):
        super().__init__(name)
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def refractive_index(self, W: Tensor) -> Tensor:
        Wmicro = W/1000
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
    NonDispersiveMaterial("vacuum", 1.0),
    NonDispersiveMaterial("air", 1.00027),

    # Bashkatov, Alexey & Genina, Elina. (2003).
    # Water refractive index in dependence on temperature and wavelength:
    # A simple approximation. Proceedings of SPIE
    # The International Society for Optical Engineering. 5068. 10.1117/12.518857.
    CauchyMaterial("water20C", 1.31984, 0.005190553),
    CauchyMaterial("water80C", 1.31044, 0.0050572226,),
    CauchyMaterial("water", 1.31984, 0.005190553,),
    CauchyMaterial("Tea Earl Grey Hot", 1.31044, 0.0050572226),
    
    # https://en.wikipedia.org/wiki/Cauchy%27s_equation
    CauchyMaterial("Fused Silica", 1.4580, 0.00354),
    CauchyMaterial("BK7", 1.5046, 0.00420),
    CauchyMaterial("K5", 1.5220, 0.00459),
    CauchyMaterial("BaK4", 1.5690, 0.00531),
    CauchyMaterial("BaF10", 1.6700, 0.00743),
    CauchyMaterial("SF10", 1.7280, 0.013420),
]

default_material_models_dict = {model.name: model for model in default_material_models}


def get_material_model(material: str | MaterialModel) -> MaterialModel:
    if isinstance(material, MaterialModel):
        return material

    try:
        return default_material_models_dict[material]
    except KeyError:
        raise ValueError(f"No material model for '{material}'")
