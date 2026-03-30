from dataclasses import dataclass

import jaxtyping
from common.numpy_type_aliases import ArrayF, ArrayI


@dataclass(frozen=True)
class SolveResult:
    basis: jaxtyping.Int[ArrayI, " m"]
    solution: jaxtyping.Float[ArrayF, " n"]
    objective_value: float


class SolveFailedError(RuntimeError):
    pass


class InfeasibleLpError(SolveFailedError):
    pass


class UnboundedLpError(SolveFailedError):
    pass


class SimplexCyclingError(SolveFailedError):
    pass


class IterationLimitError(SolveFailedError):
    pass


MIN_CYCLE_LEN = 2
OBJECTIVE_IMPROVEMENT_TOL = 1e-9
OPTIMALITY_TOL = 1e-9
NON_NEGATIVITY_TOLERANCE = 1e-9
INVERSE_RECOMPUTE_INTERVAL = 1000


class SolveHistory:
    def __init__(self) -> None:
        self.basis_: list[tuple[int, ...]] = []
        self.objective_: list[float] = []
        self.basis_cycle_: set[tuple[int, ...]] = set()

    @property
    def basis_history(self) -> list[tuple[int, ...]]:
        return self.basis_

    @property
    def objective_history(self) -> list[float]:
        return self.objective_

    def update(
        self, basis: jaxtyping.Int[ArrayI, " m"], objective_value: float
    ) -> None:
        self.objective_.append(objective_value)

        if (
            len(self.objective_) > 1
            and abs(self.objective_[-1] - self.objective_[-2])
            > OBJECTIVE_IMPROVEMENT_TOL
        ):
            # If the objective improved, we are not cycling, so we can clear the history.
            self.basis_cycle_.clear()

        basis_signature = tuple(int(b) for b in sorted(basis))
        self.basis_.append(basis_signature)

        if basis_signature in self.basis_cycle_:
            raise SimplexCyclingError(
                f"Basis cycle of length {len(self.basis_cycle_)} detected"
            )
        self.basis_cycle_.add(basis_signature)
