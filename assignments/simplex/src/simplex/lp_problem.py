import numpy as np

class LpProblem:
    
    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray):
        self.A_ = A
        self.b_ = b
        self.c_ = c
    
    @property
    def constraint_matrix(self) -> np.ndarray:
        return self.A_
    
    @property
    def rhs(self) -> np.ndarray:
        return self.b_
    
    @property
    def objective(self) -> np.ndarray:
        return self.c_
    
    @property
    def variables(self) -> int:
        return len(self.c_)