from dataclasses import dataclass

import jaxtyping
import numpy as np


@dataclass(frozen=True)
class LpProblem:
    constraint_matrix: jaxtyping.Float[np.ndarray, "m n"]
    rhs: jaxtyping.Float[np.ndarray, " m"]
    objective: jaxtyping.Float[np.ndarray, " n"]

    @property
    def num_variables(self) -> int:
        return len(self.objective)
