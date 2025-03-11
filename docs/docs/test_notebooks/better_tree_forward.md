```python
## earlier attempt
## doesn't work because multiple hooks get added to the same module instance :(
from typing import Any, Callable

import torch
import torch.nn as nn

def forward_tree(
    module: nn.Module, inputs: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:

    input_tree: dict[str, Any] = {}
    output_tree: dict[str, Any] = {}
    module_tree: dict[str, nn.Module] = {}

    def make_hook(key: str) -> Callable[[nn.Module, Any, Any], None]:
        def hook(mod: nn.Module, inp: Any, out: Any) -> None:
            # inp[0] here restricts us to forward() first argument
            # so this only works with single argument forward() functions
            if key not in input_tree:
                print("hook called on ", key, type(key))
                input_tree[key] = inp[0]
                output_tree[key] = out
                module_tree[key] = mod
        return hook


    # Register forward hooks to every module recursively
    hooks = []
    for key, mod in module.named_modules(remove_duplicate=False, prefix="root"):
        print(".")
        hook = make_hook(key)

        hooks.append(mod.register_forward_hook(hook))

    # Evaluate the full model, then remove all hooks
    try:
        _ = module(inputs)
    finally:
        for h in hooks:
            h.remove()

    return input_tree, output_tree, module_tree
```


```python


class Group(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return x + 10

class Mod(nn.Module):
    def forward(self, x):
        return x + 1

lens = Mod()

model = nn.Sequential(
    lens,
    lens,
    nn.Sequential(
        lens,
        lens
    ),
    Group(lens, lens),
)

print(model)

input_tree, output_tree, module_tree = forward_tree(model, 0)

print(input_tree)
print(output_tree)
print(module_tree)
```

    Sequential(
      (0): Mod()
      (1): Mod()
      (2): Sequential(
        (0): Mod()
        (1): Mod()
      )
      (3): Group(
        (a): Mod()
        (b): Mod()
      )
    )
    .
    .
    .
    .
    .
    .
    .
    .
    .
    hook called on  root.0 <class 'str'>
    hook called on  root.1 <class 'str'>
    hook called on  root.2.0 <class 'str'>
    hook called on  root.2.1 <class 'str'>
    hook called on  root.3.a <class 'str'>
    hook called on  root.3.b <class 'str'>
    hook called on  root.2 <class 'str'>
    hook called on  root.3 <class 'str'>
    hook called on  root <class 'str'>
    {'root.0': 0, 'root.1': 0, 'root.2.0': 0, 'root.2.1': 0, 'root.3.a': 0, 'root.3.b': 0, 'root.2': 2, 'root.3': 4, 'root': 0}
    {'root.0': 1, 'root.1': 1, 'root.2.0': 1, 'root.2.1': 1, 'root.3.a': 1, 'root.3.b': 1, 'root.2': 4, 'root.3': 14, 'root': 14}
    {'root.0': Mod(), 'root.1': Mod(), 'root.2.0': Mod(), 'root.2.1': Mod(), 'root.3.a': Mod(), 'root.3.b': Mod(), 'root.2': Sequential(
      (0): Mod()
      (1): Mod()
    ), 'root.3': Group(
      (a): Mod()
      (b): Mod()
    ), 'root': Sequential(
      (0): Mod()
      (1): Mod()
      (2): Sequential(
        (0): Mod()
        (1): Mod()
      )
      (3): Group(
        (a): Mod()
        (b): Mod()
      )
    )}

