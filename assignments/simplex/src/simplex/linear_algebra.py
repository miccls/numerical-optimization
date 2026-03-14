import jaxtyping
import numpy as np
from common.numpy_type_aliases import ArrayF


def update_inverse(
    a: jaxtyping.Float[ArrayF, "m n"],
    b_inv: jaxtyping.Float[ArrayF, "m m"],
    entering_variable: int,
    exiting_index: int,
) -> jaxtyping.Float[ArrayF, "m m"]:
    """
    Computes `B_new^-1` where `B_new` is formed by replacing column `exiting_index`
    in the matrix `B` with the column `A[:, entering_variable]`.

    Args:
        a: constraint matrix.
        b_inv: inverse of the current basis matrix.
        entering_variable: variable index for the variable entering the basis.
        exiting_index: index of the column in the basis matrix that should be replaced.

    Returns:
        inverse of the updated basis matrix, `B_new^-1`.
    """
    # TODO(you): Implement a numerically efficient way to calculate the inverse
    # of the new basis matrix after performing a pivot.
    return np.array([])


def update_inverse_gaussian(
    a: jaxtyping.Float[ArrayF, "m n"],
    b_inv: jaxtyping.Float[ArrayF, "m m"],
    entering_variable: int,
    exiting_index: int,
) -> jaxtyping.Float[ArrayF, "m m"]:
    """
    Computes `B_new^-1` where `B_new` is formed by replacing column `exiting_index`
    in the matrix `B` with the column `A[:, entering_variable]`.

    Args:
        a: constraint matrix.
        b_inv: inverse of the current basis matrix.
        entering_variable: variable index for the variable entering the basis.
        exiting_index: index of the column in the basis matrix that should be replaced.

    Returns:
        inverse of the updated basis matrix, `B_new^-1`.
    """
    # TODO(you): Implement the gaussian elimination method to update the inverse.
    return np.array([])
