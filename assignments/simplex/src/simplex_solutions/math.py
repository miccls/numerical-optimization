import numpy as np
from jaxtyping import Float

def update_inverse(
        A: Float[np.ndarray, "constraints variables"], 
        Binv: Float[np.ndarray, "constraints constraints"], 
        entering_variable: int, 
        exiting_index: int
        ) -> Float[np.ndarray, "constraints constraints"]:
    d: Float[np.ndarray, "constraints"] = Binv @ A[:, entering_variable]
    dl: float = d[exiting_index]
    rowl: np.ndarray = Binv[exiting_index]
    Binv = np.array([row - (rowl * d[i]/dl) if i != exiting_index else row / dl for i, row in enumerate(Binv)])
    return Binv