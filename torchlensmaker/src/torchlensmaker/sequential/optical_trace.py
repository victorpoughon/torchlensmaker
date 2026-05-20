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

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Self

import torch

from torchlensmaker.core.base_module import BaseModule
from torchlensmaker.core.ray_bundle import RayBundle
from torchlensmaker.kinematics.homogeneous_geometry import hom_identity
from torchlensmaker.types import Tf


@dataclass
class OpticalTraceNode:
    record: Any  # the output of the node's module
    module: BaseModule | None  # the model module that produced this node
    upstream: set[str]  # set set of nodes where input rays come from for this node
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
    nodes: OrderedDict[str, OpticalTraceNode]

    @classmethod
    def from_inputs(
        cls,
        rays: RayBundle,
        tf: Tf,
    ) -> Self:
        root = OpticalTraceNode(
            record=None,
            module=None,
            upstream=set(),
            bundle_in=rays,
            bundle_out=rays,
            tf_in=tf,
            tf_out=tf,
        )

        return cls(rays.dim, rays.dtype, rays.device, OrderedDict([("_root", root)]))

    @classmethod
    def empty(
        cls,
        dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> Self:

        root_bundle = RayBundle.empty(dim, dtype, device)
        root_tf = hom_identity(dim=dim)

        return cls.from_inputs(root_bundle, root_tf)

    def append(
        self,
        key: str,
        record: Any,
        module: BaseModule,
        upstream: set[str],
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
            upstream=upstream,
            bundle_in=bundle_in,
            bundle_out=linear_next_bundle,
            tf_in=tf_in,
            tf_out=linear_next_tf,
        )

        self.nodes[key] = new_node

    def is_linear(self):
        parents_list = [(key, list(node.upstream)) for key, node in self.nodes.items()]

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

    def output_rays(self) -> RayBundle:
        return next(reversed(self.nodes.values())).bundle_out

    def output_tf(self) -> Tf:
        return next(reversed(self.nodes.values())).tf_out

    def output_record(self) -> Any:
        return next(reversed(self.nodes.values())).record
