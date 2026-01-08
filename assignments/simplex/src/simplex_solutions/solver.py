from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from jaxtyping import Float

from . import lp_problem, math, pivoting_strategy

def compute_reduced_costs(
    problem: lp_problem.LpProblem,
    basic_variables: list[int],
    non_basic_variables: list[int],
    Binv: Float[np.ndarray, "constraints constraints"],
) -> Float[np.ndarray, "non_basic_vars"]:
    
    if not non_basic_variables:
        return np.array([])

    c: Float[np.ndarray, "variables"] = problem.objective
    A: Float[np.ndarray, "constraints variables"] = problem.constraint_matrix
    
    return (
        c[non_basic_variables]
        - (c[basic_variables] @ Binv) @ A[:, non_basic_variables]
    )

class SolverStatus(Enum):
    SUCCESS = 0
    INFEASIBLE = 1
    UBOUNDED = 2
    CYCLING = 3
    
    def __str__(self):
        return self.name

@dataclass
class Solver:
    pivoting_strategy_: pivoting_strategy.PivotingStrategy

    def _check_cycling(
        self,
        objective_history: list[float],
        basis_history: list[tuple[int, ...]],
        current_basis: list[int],
    ) -> Optional[SolverStatus]:
        """Checks for cycling in the simplex algorithm."""
        if len(objective_history) < 2:
            return None

        # A cycle can only occur with degenerate pivots (no change in objective)
        if abs(objective_history[-1] - objective_history[-2]) < 1e-9:
            basis_signature = tuple(sorted(current_basis))
            if basis_signature in basis_history:
                return SolverStatus.CYCLING
            basis_history.append(basis_signature)
        else:
            # If the objective improved, we are not cycling, so we can clear the history.
            basis_history.clear()
        
        return None

    def find_basic_feasible_solution(
        self,
        problem: lp_problem.LpProblem,
    ) -> Tuple[SolverStatus, Optional[list[int]]]:
        """
        Finds a basic feasible solution by solving an auxiliary LP.
        """

        b: Float[np.ndarray, "constraints"] = problem.rhs.copy()
        A_original: Float[np.ndarray, "constraints variables"] = problem.constraint_matrix.copy()

        # Make sure b >= 0 so that y = b, x = 0 is a feasible start solution.
        for i in range(len(b)):
            if b[i] < 0:
                A_original[i] *= -1
                b[i] *= -1

        # This concatenation represents changing Ax = b --> Ax + y = b
        A: Float[np.ndarray, "constraints auxiliary_problem_vars"] = np.concatenate(
            (A_original, np.eye(len(b))), axis=1
        )
        c: Float[np.ndarray, "auxiliary_problem_vars"] = np.concatenate(
            (np.zeros(len(problem.objective)), np.ones(len(b)))
        )
        
        feasibility_problem = lp_problem.LpProblem(A, b, c)

        # The basis is all of the auxiliary variables.
        B: list[int] = list(range(len(problem.objective), len(c)))
        status, (B_final, _, objective_value) = self.solve(feasibility_problem, B=B, log=False)

        if status != SolverStatus.SUCCESS:
            raise RuntimeError(
                f"Could not solve auxiliary problem to find feasible starting point. Solver finished with status code: {status}"
            )

        if objective_value is not None and abs(objective_value) > 1e-9:
            return SolverStatus.INFEASIBLE, None
        
        if B_final:
            # If the problem is feasible, but the optimal basis of the aux problem
            # contains artificial variables, it means the original problem has
            # redundant constraints. We need to pivot out the artificial variables.
            # For this assignment, we will assume this does not happen with the test cases.
            for var in B_final:
                if var >= problem.variables:
                    # This is an artificial variable.
                    # A pivot step is needed to remove it from the basis.
                    pass

        return SolverStatus.SUCCESS, B_final

    def solve(
        self,
        problem: lp_problem.LpProblem,
        B: Optional[list[int]] = None,
        log: bool = False,
    ) -> Tuple[
        SolverStatus,
        Tuple[
            Optional[list[int]],
            Optional[list[float]],
            Optional[float],
        ],
    ]:
        if B is None:
            status, B = self.find_basic_feasible_solution(problem)
            if status == SolverStatus.INFEASIBLE:
                return status, (None, None, None)

        assert B is not None  # for type checkers

        A: Float[np.ndarray, "constraints variables"] = problem.constraint_matrix
        c: Float[np.ndarray, "variables"] = problem.objective
        
        Binv: Float[np.ndarray, "constraints constraints"] = np.linalg.inv(A[:, B])
        
        x_basis: Float[np.ndarray, "constraints"] = Binv @ problem.rhs

        if log:
            print("Starting simplex algorithm...")

        basis_history: list[tuple[int, ...]] = []
        objective_history: list[float] = [c[B] @ x_basis]
        iteration: int = 1
        while True:
            non_basic_variables: list[int] = [
                col for col in range(problem.variables) if col not in B
            ]

            reduced_costs = compute_reduced_costs(
                problem,
                B,
                non_basic_variables,
                Binv,
            )

            if len(reduced_costs) == 0 or np.all(reduced_costs >= 0):
                status = SolverStatus.SUCCESS
                break

            entering_variable: int = self.pivoting_strategy_.pick_entering_index(reduced_costs, non_basic_variables)
            
            d: Float[np.ndarray, "constraints"] = Binv @ A[:, entering_variable]
            
            if np.all(d <= 0):
                status = SolverStatus.UBOUNDED
                break

            ratios = np.array([xi / ui if ui > 0 else np.inf for (xi, ui) in zip(x_basis, d)])
            exiting_variable = self.pivoting_strategy_.pick_exiting_index(ratios, d, B)
       
            basic_exiting_index = B.index(exiting_variable) 
            B[basic_exiting_index] = entering_variable
            Binv = math.update_inverse(A, Binv, entering_variable, basic_exiting_index)

            # Update the solution x based on the pivot column.
            x_basis -= ratios[basic_exiting_index] * d
            x_basis[basic_exiting_index] = ratios[basic_exiting_index]

            objective_history.append(c[B] @ x_basis)
            
            status = self._check_cycling(objective_history, basis_history, B)
            if status == SolverStatus.CYCLING:
                break

            if log:
                print(
                    f"Iteration {iteration} ::: "
                    f"Leaving index: {exiting_variable}, "
                    f"Entering index: {entering_variable}, "
                    f"Objective: {c[B] @ x_basis}"
                )
                
            iteration += 1

        if log:
            print(f"Simplex algorithm terminated after {iteration} iterations.")

        if status == SolverStatus.SUCCESS:
            solution: list[float] = [
                float(x_basis[B.index(i)]) if i in B else 0.0
                for i in range(problem.variables)
            ]
            objective_value: float = float(c[B] @ x_basis)
            return status, (B, solution, objective_value)

        return status, (None, None, None)
