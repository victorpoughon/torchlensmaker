import torch
from torchlensmaker.tensor_manip import cartesian_prod2d
import itertools


def test_shapes() -> None:
    Nspace = [1, 2, 3, 10]
    Mspace = [1, 2, 3, 10]
    Dspace = [1, 2, 3, 10]
    Espace = [1, 2, 3, 10]

    for N, M, D, E in itertools.product(Nspace, Mspace, Dspace, Espace):
        A = torch.rand((N, D))
        B = torch.rand((M, E))

        PA, PB = cartesian_prod2d(A, B)
        assert PA.shape == (N * M, D)
        assert PB.shape == (N * M, E)


def test_A1D() -> None:
    Nspace = [1, 2, 3, 10]
    Mspace = [1, 2, 3, 10]
    # no D (A.dim() == 1)
    Espace = [1, 2, 3, 10]

    for N, M, E in itertools.product(Nspace, Mspace, Espace):
        A = torch.rand((N,))
        B = torch.rand((M, E))

        PA, PB = cartesian_prod2d(A, B)
        assert PA.shape == (N * M, 1)
        assert PB.shape == (N * M, E)


def test_B1D() -> None:
    Nspace = [1, 2, 3, 10]
    Mspace = [1, 2, 3, 10]
    Dspace = [1, 2, 3, 10]
    # no E (B.dim() == 1)

    for N, M, D in itertools.product(Nspace, Mspace, Dspace):
        A = torch.rand((N, D))
        B = torch.rand((M,))

        PA, PB = cartesian_prod2d(A, B)
        assert PA.shape == (N * M, D)
        assert PB.shape == (N * M, 1)


def test_content() -> None:
    A = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )

    B = torch.tensor(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ]
    )

    PA, PB = cartesian_prod2d(A, B)

    assert torch.all(
        PA
        == torch.tensor(
            [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
                [4, 5, 6],
                [4, 5, 6],
                [4, 5, 6],
            ]
        )
    )

    assert torch.all(
        PB
        == torch.tensor(
            [
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90],
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90],
            ]
        )
    )


def test_content2() -> None:
    A = torch.tensor(
        [
            [1, 2],
            [4, 5],
        ]
    )

    B = torch.tensor(
        [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ]
    )

    PA, PB = cartesian_prod2d(A, B)

    assert torch.all(
        PA
        == torch.tensor(
            [
                [1, 2],
                [1, 2],
                [1, 2],
                [4, 5],
                [4, 5],
                [4, 5],
            ]
        )
    )

    assert torch.all(
        PB
        == torch.tensor(
            [
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90],
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90],
            ]
        )
    )
