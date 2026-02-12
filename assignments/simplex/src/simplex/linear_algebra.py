import jaxtyping
import numpy as np

from common.numpy_type_aliases import ArrayF


def update_inverse(
    a: jaxtyping.Float[ArrayF, "m n"],
    b_inv: jaxtyping.Float[ArrayF, "m m"],
    entering_variable: int,
    exiting_index: int,
) -> jaxtyping.Float[ArrayF, "m m"]:
    # TODO(you): Implement a numerically efficient way to calculate the inverse
    # of the new basis matrix after performing a pivot.
    return np.array([])
