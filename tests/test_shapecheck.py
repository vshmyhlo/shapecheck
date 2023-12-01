from typing import NamedTuple, Tuple
import pytest

from shapecheck import shapecheck, UnknownDimError, MismatchedDimError


def test_batch_matmul():
    s = shapecheck()

    s.check("NAB", array(10, 3, 5))
    s.check("NBC", array(10, 5, 7))
    s.check("NAC", array(10, 3, 7))

    assert s["N"] == 10
    assert s["A"] == 3
    assert s["B"] == 5
    assert s["C"] == 7


def test_ndim_doesnt_match():
    s = shapecheck()

    s.check("ABC", array(1, 2, 3))

    with pytest.raises(MismatchedDimError):
        s.check("AB", array(1, 2, 3))

    with pytest.raises(MismatchedDimError):
        s.check("ABCD", array(1, 2, 3))

    s.check("ABC", array(1, 2, 3))


def test_unknown_dim():
    s = shapecheck()

    with pytest.raises(UnknownDimError):
        assert s["A"] == 10

    s.check("A", array(10))

    assert s["A"] == 10


def test_mismatched_dim():
    s = shapecheck()

    s.check("ABC", array(1, 2, 3))

    with pytest.raises(MismatchedDimError):
        s.check("BC", array(1, 2))


# @pytest.mark.skip('requires torch')
def test_matmul_readme_example():
    import torch
    from shapecheck import shapecheck

    def some_code_that_does_batch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        s = shapecheck()

        # N matrices of shape (A, B)
        s.check("NAB", a)

        # N matrices of shape (B, C)
        s.check("NBC", b)

        c = torch.bmm(a, b)

        # N matrices of shape (A, C)
        s.check("NAC", c)

        return c

    some_code_that_does_batch_matmul(torch.zeros(10, 3, 5), torch.zeros(10, 5, 7))


# @pytest.mark.skip('requires torch')
def test_images_readme_example():
    import torch
    from torchvision.models import resnet18
    import torch.nn as nn
    from shapecheck import shapecheck

    # batch of pairs of (3, 128, 256) images for contrastive training
    image_pairs = torch.zeros(10, 2, 3, 128, 256)

    s = shapecheck()
    s.check("BNCHW", image_pairs)
    assert s["N"] == 2  # check those are pairs indeed
    assert s["C"] == 3  # check they have 3 channels

    # flatten to feed into model
    image_pairs = image_pairs.flatten(0, 1)

    # forward pass
    model = resnet18()
    model.fc = nn.Identity()
    outputs = model(image_pairs)

    # unflatten back into pairs
    outputs = outputs.unflatten(0, (s["B"], s["N"]))

    # batch of pairs of (512) vectors
    s.check("BNc", outputs)
    assert s["c"] == 512


class DummyArray(NamedTuple):
    shape: Tuple[int, ...]


def array(*dim: int) -> DummyArray:
    return DummyArray(dim)
