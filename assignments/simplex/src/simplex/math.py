import jaxtyping
import numpy as np


def update_inverse(
    a: jaxtyping.Float[np.ndarray, "m n"],
    b_inv: jaxtyping.Float[np.ndarray, "m m"],
    entering_variable: int,
    exiting_index: int,
) -> jaxtyping.Float[np.ndarray, "m m"]:
    # TODO(you): Implement a numerically efficient way to calculate the inverse
    # of the new basis matrix after performing a pivot.
    return np.array([])
