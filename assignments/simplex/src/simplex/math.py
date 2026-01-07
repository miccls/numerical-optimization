import numpy as np
from jaxtyping import Float

def update_inverse(
        A: Float[np.ndarray, "constraints variables"], 
        Binv: Float[np.ndarray, "constraints constraints"], 
        entering_variable: int, 
        exiting_index: int
        ) -> Float[np.ndarray, "constraints constraints"]:
    # TODO: Implement a numerically efficient way to calculate the inverse
    #       of the new basis matrix after performing a pivot.
    pass