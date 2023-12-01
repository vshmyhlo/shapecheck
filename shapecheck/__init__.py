import inspect
import textwrap
from typing import Callable, Dict, Optional, Protocol, Tuple


class UnknownDimError(Exception):
    pass


class MismatchedDimError(Exception):
    pass


class Shaped(Protocol):
    @property
    def shape(self) -> Tuple[int, ...]:
        ...


class ShapeCheck:
    _name_to_dim: Dict[str, int]

    def __init__(self) -> None:
        self._name_to_dim = {}

    def check(self, shape_spec: str, array: Shaped):
        assert shape_spec.isalpha()

        shape = array.shape
        del array

        # check shape spec is same length as shape
        if len(shape_spec) != len(shape):
            raise MismatchedDimError(f"expected {shape_spec=}, got {shape=}")

        for dim_name, dim in zip(shape_spec, shape):
            # if dim_name has already been seen - compare against what was seen before
            if dim_name in self._name_to_dim:
                expected = self._name_to_dim[dim_name]
                if dim != expected:
                    raise MismatchedDimError(
                        f"expected {expected} for {dim_name=}, got {dim} ({shape_spec=}, {shape=})"
                    )
            # if dim_name is seen first name - remember it's value
            else:
                self._name_to_dim[dim_name] = dim

    def __getitem__(self, dim_name: str):
        if dim_name not in self._name_to_dim:
            raise UnknownDimError(dim_name)
        return self._name_to_dim[dim_name]


def shapecheck() -> ShapeCheck:
    return ShapeCheck()
