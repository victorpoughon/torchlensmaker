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
from itertools import islice
from typing import Any, DefaultDict, Iterator, Self

import torch

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.kinematics.homogeneous_geometry import hom_identity
from torchlensmaker.surfaces import SurfaceElement
from torchlensmaker.types import BatchNDTensor, BatchTensor, MaskTensor, Tf


@dataclass
class OpticalTraceNode:
    record: Any  # the output of the node's module
    module: BaseModule | None  # the model module that produced this node
    parents: set[str]
    bundle_in: RayBundle
    bundle_out: RayBundle
    tf_in: Tf
    tf_out: Tf


@dataclass
class OpticalTrace:
    """
    A concrete realization of a model after sampling and forward evaluation,
    including intermediate data in the model sequence
    """

    dim: int
    dtype: torch.dtype
    device: torch.device

    nodes: OrderedDict[str, OpticalTraceNode] = field(default_factory=OrderedDict)
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

        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.get_default_device()

        root = OpticalTraceNode(
            record=None,
            module=None,
            parents=set(),
            bundle_in=root_bundle,
            bundle_out=root_bundle,
            tf_in=root_tf,
            tf_out=root_tf,
        )

        return cls(
            dim,
            dtype,
            device,
            OrderedDict([("_root", root)]),
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
            raise ValueError(f"OpticalTrace already contains a node for key {key}")

        if self._linear_latest_bundle is None or self._linear_latest_tf is None:
            raise ValueError(
                "OpticalTrace internal error, linear latest tracking is None"
            )

        linear_next_bundle = (
            new_bundle if new_bundle is not None else self._linear_latest_bundle
        )
        linear_next_tf = new_tf if new_tf is not None else self._linear_latest_tf

        new_node = OpticalTraceNode(
            record=record,
            module=module,
            parents=set(),
            bundle_in=self._linear_latest_bundle,
            bundle_out=linear_next_bundle,
            tf_in=self._linear_latest_tf,
            tf_out=linear_next_tf,
        )

        self.nodes[key] = new_node

        self._linear_latest_bundle = linear_next_bundle
        self._linear_latest_tf = linear_next_tf

    def append(
        self,
        key: str,
        record: Any,
        module: BaseModule,
        parents: set[str],
        bundle_in: RayBundle,
        tf_in: Tf,
        new_bundle: RayBundle | None = None,  # None = share bundle_in as bundle_out
        new_tf: Tf | None = None,  # None = share tf_in as tf_out
    ) -> None:
        if key in self.nodes:
            raise ValueError(f"OpticalTrace already contains a node for key {key}")

        linear_next_bundle = new_bundle if new_bundle is not None else bundle_in
        linear_next_tf = new_tf if new_tf is not None else tf_in

        new_node = OpticalTraceNode(
            record=record,
            module=module,
            parents=parents,
            bundle_in=bundle_in,
            bundle_out=linear_next_bundle,
            tf_in=tf_in,
            tf_out=linear_next_tf,
        )

        self.nodes[key] = new_node

    def is_linear(self):
        parents_list = [(key, list(node.parents)) for key, node in self.nodes.items()]

        # Check every node has the previous as a parent
        for i in range(1, len(parents_list)):
            prev_key = parents_list[i - 1][0]
            curr_parents = parents_list[i][1]
            linear = len(curr_parents) == 1 and curr_parents[0] == prev_key
            if not linear:
                return False

        return True

    def iter_nodes_by_module_type(
        self, typ: type[BaseModule]
    ) -> Iterator[tuple[str, OpticalTraceNode]]:
        for key, node in self.nodes.items():
            if isinstance(node.module, typ):
                yield (key, node)

    def iter_nodes_by_record_type(
        self, typ: Any
    ) -> Iterator[tuple[str, OpticalTraceNode]]:
        for key, node in self.nodes.items():
            if isinstance(node.record, typ):
                yield (key, node)


def trace_model(
    optics: BaseModule,
    dim: int,
    *inputs: Any,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> OpticalTrace:
    trace = OpticalTrace.empty(dim, dtype, device)

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
