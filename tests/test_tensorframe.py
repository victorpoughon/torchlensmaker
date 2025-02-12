import torch
from torchlensmaker.core.tensorframe import TensorFrame


def test_print() -> None:
    tf = TensorFrame(torch.rand(10, 3), ["a", "b", "c"])
    print(tf)


def test_get() -> None:
    tf = TensorFrame(torch.rand(10, 3), ["a", "b", "c"])

    print(tf.get("a"))
    print(tf.get(["a", "b"]))


def test_update() -> None:
    N = 2
    tf = TensorFrame(
        torch.column_stack(
            (
                torch.full((N,), 1),
                torch.full((N,), 2),
                torch.full((N,), 3),
            )
        ),
        ["a", "b", "c"],
    )

    tf2 = tf.update(a=torch.full((N,), 0), d=torch.full((N,), 10))

    assert torch.all(tf2.data == torch.tensor([[0, 2, 3, 10], [0, 2, 3, 10]]))
    assert tf2.columns == ["a", "b", "c", "d"]
    assert tf2.shape == tf2.data.shape == (N, 4)


def test_update_broadcast() -> None:
    N = 2
    tf = TensorFrame(
        torch.column_stack(
            (
                torch.full((N,), 1),
                torch.full((N,), 2),
                torch.full((N,), 3),
            )
        ),
        ["a", "b", "c"],
    )

    # broadcast single number
    tf = tf.update(a=42, d=45)

    # broadcast tensor of dim 0
    tf = tf.update(e=torch.tensor(0.))

    # broadcast tensor of dim 1, size 1
    tf = tf.update(f=torch.tensor([-10]))

    assert torch.all(tf.data == torch.tensor([[42, 2, 3, 45, 0, -10], [42, 2, 3, 45, 0, -10]]))
    assert tf.columns == ["a", "b", "c", "d", "e", "f"]
    assert tf.shape == tf.data.shape == (N, 6)


def test_stack() -> None:
    N = 2
    tf1 = TensorFrame(
        torch.column_stack(
            (
                torch.full((N,), 1),
                torch.full((N,), 2),
            )
        ),
        ["a", "b"],
    )

    tf2 = TensorFrame(
        torch.column_stack(
            (
                torch.full((N,), 3),
                torch.full((N,), 4),
            )
        ),
        ["a", "b"],
    )

    tf3 = tf1.stack(tf2)

    assert torch.all(tf3.data == torch.tensor([[1, 2], [1, 2], [3, 4], [3, 4]]))
    assert tf3.columns == tf1.columns == tf2.columns
