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


import pytest
import torch
import torch.nn as nn

from torchlensmaker.core.deep_forward import (
    deep_forward,
)

from torchlensmaker.core.base_module import MultiForwardModule, multiforward


def test_deep_forward() -> None:
    "Test DeepForward without any multiforward module"

    class SmallNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Linear(4, 3),
                nn.ReLU(),
            )
            self.block2 = nn.Sequential(
                nn.Linear(3, 2),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            return x

    batch = 5
    x = torch.randn(batch, 4)
    model = SmallNet()

    with deep_forward(model) as trace:
        _ = model(x)

    # Expected entries
    expected_entries = [
        model,
        model.block1,
        model.block2,
        model.block1[0],
        model.block1[1],
        model.block2[0],
        model.block2[1],
    ]
    assert set(expected_entries) == set(trace.inputs.keys())
    assert set(expected_entries) == set(trace.outputs.keys())

    # Compare with normal evaluation output
    expected_final = model(x)
    assert torch.allclose(trace.outputs[model], expected_final)

    # Expected shapes
    assert trace.inputs[model].shape == (batch, 4)
    assert trace.outputs[model].shape == (batch, 2)

    assert trace.inputs[model.block1].shape == (batch, 4)
    assert trace.outputs[model.block1].shape == (batch, 3)

    assert trace.inputs[model.block2].shape == (batch, 3)
    assert trace.outputs[model.block2].shape == (batch, 2)

    assert trace.inputs[model.block1[0]].shape == (batch, 4)
    assert trace.outputs[model.block1[0]].shape == (batch, 3)

    assert trace.inputs[model.block1[1]].shape == (batch, 3)
    assert trace.outputs[model.block1[1]].shape == (batch, 3)

    assert trace.inputs[model.block2[0]].shape == (batch, 3)
    assert trace.outputs[model.block2[0]].shape == (batch, 2)

    assert trace.inputs[model.block2[1]].shape == (batch, 2)
    assert trace.outputs[model.block2[1]].shape == (batch, 2)


def test_multi_deep_forward() -> None:
    class Model(MultiForwardModule):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Linear(2, 2),
                nn.ReLU(),
            )
            self.block2 = nn.Sequential(
                nn.Linear(2, 1),
                nn.Sigmoid(),
            )
            self.translation = nn.Parameter(torch.zeros(2))

        @multiforward
        def forward_kinematic(self, point):
            x = self.block1(point)
            x = self.block2(point)
            x = x.sum()
            return point + self.translation + x

        @multiforward
        def reverse_kinematic(self, point):
            return point - self.translation

    model = Model()
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    target_forward = torch.tensor([[11.0, 7.0], [13.0, 9.0]])
    target_reverse = torch.tensor([[-9.0, -3.0], [-7.0, -1.0]])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    print(model)
    print()

    for i in range(10):
        print("> ", i)

        with deep_forward(model) as tree_forward:
            model.forward_kinematic(data)

        with deep_forward(model) as tree_reverse:
            model.reverse_kinematic(data)

        print(tree_forward.inputs)
        print(tree_forward.outputs)

        print(tree_reverse.inputs)
        print(tree_reverse.outputs)

        translated = tree_forward.outputs[model]
        reverse = tree_reverse.outputs[model]

        loss_fwd = nn.MSELoss()(translated, target_forward)
        loss_rev = nn.MSELoss()(reverse, target_reverse)

        # Compute loss dependent on both forward and reverse
        total_loss = loss_fwd + loss_rev

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print(model.translation)


def test_deep_forward_detects_reuse() -> None:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            relu = nn.ReLU()
            self.block1 = nn.Sequential(
                nn.Linear(2, 2),
                relu,
            )
            self.block2 = nn.Sequential(
                nn.Linear(2, 1),
                relu,
            )

        def forward(self, point):
            x = self.block1(point)
            x = self.block2(point)
            return x

    model = Model()
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Expect an error due to reused relu module
    with pytest.raises(RuntimeError):
        with deep_forward(model) as trace:
            _ = model(data)


def test_deep_forward_detects_multiple_forward_call() -> None:
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Linear(2, 2),
                nn.ReLU(),
            )

        def forward(self, point):
            x = self.block1(point)
            x = self.block1(point)
            return x

    model = Model()
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Expect an error due to multiple forward calls
    with pytest.raises(RuntimeError):
        with deep_forward(model) as trace:
            _ = model(data)


def test_deep_multi_forward_detects_multiple_forward_call() -> None:
    class Model(MultiForwardModule):
        def __init__(self):
            super().__init__()
            relu = nn.ReLU()
            self.block1 = nn.Sequential(
                nn.Linear(2, 2),
                relu,
            )
            self.block2 = nn.Sequential(
                nn.Linear(2, 1),
                relu,
            )

        @multiforward
        def forward1(self, point):
            point = self.block1(point)
            point = self.block1(point)
            return point

        @multiforward
        def forward2(self, point):
            point = self.block1(point)
            point = self.block1(point)
            return point

    model = Model()
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Expect an error due to reused block1 module
    with pytest.raises(RuntimeError):
        with deep_forward(model) as trace:
            _ = model.forward1(data)

def test_multi_deep_forward_delegate() -> None:
    class Model(MultiForwardModule):
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Linear(2, 2),
                nn.ReLU(),
            )
            self.block2 = nn.Sequential(
                nn.Linear(2, 1),
                nn.Sigmoid(),
            )
            self.translation = nn.Parameter(torch.zeros(2))

        @multiforward
        def forward_kinematic(self, point):
            x = self.block1(point)
            x = self.block2(point)
            x = x.sum()
            return point + self.translation + x

        @multiforward
        def reverse_kinematic(self, point):
            return 0.7 * self.forward_kinematic(point)

    model = Model()
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    target_forward = torch.tensor([[11.0, 7.0], [13.0, 9.0]])
    target_reverse = torch.tensor([[-9.0, -3.0], [-7.0, -1.0]])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    print(model)
    print()

    for i in range(10):
        print("> ", i)

        with deep_forward(model) as tree_forward:
            model.forward_kinematic(data)

        with deep_forward(model) as tree_reverse:
            model.reverse_kinematic(data)

        print(tree_forward.inputs)
        print(tree_forward.outputs)

        print(tree_reverse.inputs)
        print(tree_reverse.outputs)

        translated = tree_forward.outputs[model]
        reverse = tree_reverse.outputs[model]

        loss_fwd = nn.MSELoss()(translated, target_forward)
        loss_rev = nn.MSELoss()(reverse, target_reverse)

        # Compute loss dependent on both forward and reverse
        total_loss = loss_fwd + loss_rev

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print(model.translation)
