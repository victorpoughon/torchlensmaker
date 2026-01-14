import pytest
import torch

from torchlensmaker.surfaces.conics import Sphere
from torchlensmaker.testing.basic_transform import basic_transform


def test_basic_transform():
    surface = Sphere(10, 50)
    tfs = [
        basic_transform(1.0, "origin", 0.1, [1.0, 2.0]),
        basic_transform(-1.0, "origin", 0.1, [1.0, 2.0]),
        basic_transform(1.0, "extent", 0.1, [1.0, 2.0]),
        basic_transform(-1.0, "extent", 0.1, [1.0, 2.0]),
    ]

    for tf in tfs:
        _ = tf(surface)
