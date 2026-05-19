# This file is part of Torch Lens Maker
# Copyright (C) 2024-present Victor Poughon
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Iterator, Self

import torch

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.kinematics.homogeneous_geometry import hom_identity
from torchlensmaker.surfaces import SurfaceElement
from torchlensmaker.types import BatchNDTensor, BatchTensor, MaskTensor, Tf


@dataclass
class ModelTraceNode:
    record: Any  # the output of the node's module
    module: BaseModule  # the model module that produced this node
    bundle_in: RayBundle
    bundle_out: RayBundle
    tf_in: Tf
    tf_out: Tf


@dataclass
class ModelTrace:
    """
    A concrete realization of a model after sampling and forward evaluation,
    including intermediate data in the model sequence
    """

    dim: int
    root_bundle: RayBundle
    root_tf: Tf

    nodes: OrderedDict[str, ModelTraceNode] = field(default_factory=OrderedDict)
    _linear_latest_bundle: RayBundle | None = None
    _linear_latest_tf: Tf | None = None

    @classmethod
    def empty(
        cls,
        dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> Self:
        root_bundle = RayBundle.empty(dim, dtype, device)
        root_tf = hom_identity(dim=dim)
        return cls(
            dim,
            root_bundle,
            root_tf,
            OrderedDict(),
            root_bundle,
            root_tf,
        )

    def add_node(
        self,
        key: str,
        record: Any,
        module: BaseModule,
        new_bundle: RayBundle | None = None,  # None = share bundle_in as bundle_out
        new_tf: Tf | None = None,  # None = share tf_in as tf_out
    ) -> None:
        if key in self.nodes:
            raise ValueError(f"ModelTrace already contains a node for key {key}")

        if self._linear_latest_bundle is None or self._linear_latest_tf is None:
            raise ValueError(
                "ModelTrace internal error, linear latest tracking is None"
            )

        linear_next_bundle = (
            new_bundle if new_bundle is not None else self._linear_latest_bundle
        )
        linear_next_tf = new_tf if new_tf is not None else self._linear_latest_tf

        new_node = ModelTraceNode(
            record=record,
            module=module,
            bundle_in=self._linear_latest_bundle,
            bundle_out=linear_next_bundle,
            tf_in=self._linear_latest_tf,
            tf_out=linear_next_tf,
        )

        self.nodes[key] = new_node

        self._linear_latest_bundle = linear_next_bundle
        self._linear_latest_tf = linear_next_tf

    def iter_nodes_by_module_type(
        self, typ: type[BaseModule]
    ) -> Iterator[tuple[str, ModelTraceNode]]:
        for key, node in self.nodes.items():
            if isinstance(node.module, typ):
                yield (key, node)

    def iter_nodes_by_record_type(
        self, typ: Any
    ) -> Iterator[tuple[str, ModelTraceNode]]:
        for key, node in self.nodes.items():
            if isinstance(node.record, typ):
                yield (key, node)


def trace_model(optics: BaseModule, dim: int, *inputs: Any) -> ModelTrace:
    trace = ModelTrace.empty(dim=dim)

    # keep track of how many times we called each module type to produce unique default keys
    counts: DefaultDict[Any, int] = defaultdict(int)

    def hookfn(mod: BaseModule, ins: Any, outs: Any) -> None:
        default_key = f"{type(mod).__qualname__}.{counts[type(mod)]}"
        counts[type(mod)] += 1
        try:
            key = mod.trace_key or default_key
        except Exception:
            key = default_key

        if hasattr(mod, "trace") and callable(mod.trace):
            mod.trace(trace, key, ins, outs)

    hooks: list[Any] = []
    for _, module in optics.named_modules():
        hooks.append(module.register_forward_hook(hookfn))

    _ = optics(*inputs)

    for h in hooks:
        h.remove()

    return trace
