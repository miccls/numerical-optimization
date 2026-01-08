import numpy as np
from jaxtyping import Float

class LpProblem:
    
    def __init__(
            self, 
            A: Float[np.ndarray, "constraints variables"], 
            b: Float[np.ndarray, "constraints"], 
            c: Float[np.ndarray, "variables"]):
        self.A_ = A
        self.b_ = b
        self.c_ = c
    
    @property
    def constraint_matrix(self) -> Float[np.ndarray, "constraints variables"]:
        return self.A_
    
    @property
    def rhs(self) -> Float[np.ndarray, "constraints"]:
        return self.b_
    
    @property
    def objective(self) -> Float[np.ndarray, "variables"]:
        return self.c_
    
    @property
    def variables(self) -> int:
        return len(self.c_)