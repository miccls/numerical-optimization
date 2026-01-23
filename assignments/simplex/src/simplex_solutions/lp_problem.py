from dataclasses import dataclass

import jaxtyping

from simplex_solutions.numpy_type_aliases import ArrayF


@dataclass(frozen=True)
class LpProblem:
    constraint_matrix: jaxtyping.Float[ArrayF, "m n"]
    rhs: jaxtyping.Float[ArrayF, " m"]
    objective: jaxtyping.Float[ArrayF, " n"]

    @property
    def num_variables(self) -> int:
        return len(self.objective)
