```python
import torch

import torchlensmaker.core.intari as intari

Tensor = torch.Tensor

# given input tensor batched, with last dim 2 (inf, sup)


c = torch.tensor([
    [-1, 0.5, 0.1],
    [-0.1, 0, 5],
    [-2, 4, 1],
    [1, 1, 0]
], dtype=torch.float64)


from torchlensmaker.core.intari import *

tau = torch.tensor(2., dtype=torch.float64)

p = torch.arange(4).unsqueeze(1).tile((1, 3))
q = torch.arange(3).unsqueeze(0).tile((4, 1))

yp = intari.monomial(p, tau)
zq = intari.monomial(q, tau)
prod = intari.product(yp, zq)
final = intari.scalar(c, prod)

print(final.sum(dim=0).sum(dim=0))
```

    tensor([-106.2000,  112.6000], dtype=torch.float64)

