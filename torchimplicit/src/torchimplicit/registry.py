from torchimplicit.types import FunctionDefinition

_all_functions_definitions: dict[str, FunctionDefinition] = {}


def register_function(f: FunctionDefinition):
    _all_functions_definitions[f.name] = f


def get_functions(dim: int | None = None) -> dict[str, FunctionDefinition]:
    if dim is None:
        return dict(_all_functions_definitions)
    return {k: v for k, v in _all_functions_definitions.items() if v.dim == dim}
