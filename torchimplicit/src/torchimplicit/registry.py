from torchimplicit.types import FunctionDefinition

_all_functions_definitions = {}


def register_function(f: FunctionDefinition):
    _all_functions_definitions[f.name] = f
