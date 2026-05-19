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

from dataclasses import dataclass

import torch

from torchlensmaker.light_sources.light_sources_elements import LightSourceBase
from torchlensmaker.optical_surfaces.optical_surface import OpticalSurfaceRecord
from torchlensmaker.sequential.optical_trace import OpticalTrace, OpticalTraceNode


@dataclass
class OpticalPath:
    """
    A linear walk through an OpticalTrace collecting nodes with spatial positions.
    Segments run between consecutive waypoints (source and optical surfaces).
    """

    nodes: list[OpticalTraceNode]
    keys: list[str]

    def _waypoints(self) -> list[tuple[OpticalTraceNode, torch.Tensor]]:
        result = []
        for node in self.nodes:
            if isinstance(node.record, OpticalSurfaceRecord):
                result.append((node, node.record.surface_record.points_global))
            elif isinstance(node.module, LightSourceBase):
                result.append((node, node.bundle_out.P))
        return result

    def segment_lengths(self) -> torch.Tensor:
        """Euclidean distance between consecutive waypoints. Shape: (K-1, N)"""
        waypoints = self._waypoints()
        lengths = []
        for i in range(len(waypoints) - 1):
            _, p1 = waypoints[i]
            _, p2 = waypoints[i + 1]
            lengths.append(torch.linalg.vector_norm(p2 - p1, dim=-1))
        return torch.stack(lengths, dim=0)

    def segment_n(self) -> torch.Tensor:
        """Refractive index in medium for each segment. Shape: (K-1, N)"""
        waypoints = self._waypoints()
        ns = []
        for i in range(len(waypoints) - 1):
            node, _ = waypoints[i]
            ns.append(node.bundle_out.n)
        return torch.stack(ns, dim=0)

    def segment_valid(self) -> torch.Tensor:
        """Valid mask for each segment: True iff ray reached the destination. Shape: (K-1, N)"""
        waypoints = self._waypoints()
        valids = []
        for i in range(len(waypoints) - 1):
            node, _ = waypoints[i + 1]
            valids.append(node.bundle_out.valid)
        return torch.stack(valids, dim=0)

    def opl(self) -> torch.Tensor:
        """Optical path length per ray. Shape: (N,)"""
        lengths = self.segment_lengths()
        n = self.segment_n()
        valid = self.segment_valid()
        return (n * lengths * valid.float()).sum(dim=0)


def linear_path(trace: OpticalTrace, start_key: str, end_key: str) -> OpticalPath:
    """
    Walk parent pointers from end_key back to start_key, returning an OpticalPath
    over the nodes in that linear chain. Raises if the path is non-linear or if
    either key is missing.
    """
    if start_key not in trace.nodes:
        raise KeyError(f"start_key '{start_key}' not found in trace")
    if end_key not in trace.nodes:
        raise KeyError(f"end_key '{end_key}' not found in trace")

    keys: list[str] = []
    current = end_key
    while True:
        keys.append(current)
        if current == start_key:
            break
        parents = trace.nodes[current].upstream
        if len(parents) == 0:
            raise ValueError(
                f"Reached root node '{current}' before finding start_key '{start_key}'"
            )
        if len(parents) > 1:
            raise ValueError(
                f"Non-linear path at '{current}': node has {len(parents)} parents"
            )
        current = next(iter(parents))

    keys.reverse()
    return OpticalPath(nodes=[trace.nodes[k] for k in keys], keys=keys)
