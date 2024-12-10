import math
import torch
from torchlensmaker.raytracing import rot2d


def test_single_vector_single_angle():
    v1 = torch.tensor([1.0, 0.0])
    theta1 = torch.tensor(torch.pi/4)
    result1 = rot2d(v1, theta1)
    assert torch.allclose(result1, torch.tensor([1./math.sqrt(2), 1./math.sqrt(2)]))

def test_batch_vectors_single_angle():
    v2 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    theta2 = torch.tensor(torch.pi/3)
    result2 = rot2d(v2, theta2)
    
    assert torch.allclose(result2, torch.tensor([
        [1/2, math.sqrt(3)/2],
        [-math.sqrt(3)/2, 1/2],
        [-1/2, -math.sqrt(3)/2],
    ]))

def test_batch_vectors_batch_angles():
    v3 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    theta3 = torch.tensor([-torch.pi/3, torch.pi/4, torch.pi/2])
    result3 = rot2d(v3, theta3)

    expected = torch.tensor([
        [1/2, -math.sqrt(3)/2],
        [-1./math.sqrt(2), 1./math.sqrt(2)],
        [0.0, -1.0],
    ])
    
    assert torch.allclose(result3, expected, rtol=1e-05, atol=1e-07)
