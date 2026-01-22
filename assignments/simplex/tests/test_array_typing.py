import re

import jaxtyping
import numpy as np
import pytest
from beartype import beartype

# We use the explicit decorator @jaxtyping.jaxtyped here. In the simplex modules
# we use install_import_hook in the __init__.py to automatically add decorators to all functions.


@jaxtyping.jaxtyped(typechecker=beartype)
def parametrized_return_type_wrong(n: int) -> jaxtyping.Float[np.ndarray, "n n"]:
    # using wrong dimensions here
    return np.eye(n, n + 1)


def test_parametrized_return_type() -> None:
    with pytest.raises(
        jaxtyping.TypeCheckError, match=re.escape("Actual value: f64[5,6](numpy)")
    ):
        parametrized_return_type_wrong(5)


@jaxtyping.jaxtyped(typechecker=beartype)
def parametrized_return_type_correct(n: int) -> jaxtyping.Float[np.ndarray, "n n"]:
    return np.eye(n, n)


def test_parametrized_return_type_correct() -> None:
    assert parametrized_return_type_correct(5).shape == (5, 5)


@jaxtyping.jaxtyped(typechecker=beartype)
def fun_with_local_variable_annotations() -> float:
    """
    jaxtype only checks function arguments and return values. It will not error on the annotation of the local variable id_matrix
    """
    id_matrix: jaxtyping.Float[np.ndarray, "4 5"] = np.eye(5)
    assert id_matrix.shape == (5, 5)

    return float(id_matrix.trace())


@jaxtyping.jaxtyped(typechecker=beartype)
def fun_with_arg_annotations(matrix: jaxtyping.Float[np.ndarray, "4 5"]) -> float:
    return float(matrix.trace())


def test_fun_with_internal_type_annotations() -> None:
    assert fun_with_local_variable_annotations() == 5.0
    with pytest.raises(jaxtyping.TypeCheckError):
        fun_with_arg_annotations(np.eye(5))
