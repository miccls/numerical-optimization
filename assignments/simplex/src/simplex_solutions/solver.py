from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from jaxtyping import Float

from simplex import lp_problem, math, pivoting_strategy

def compute_reduced_costs(
    problem: lp_problem.LpProblem,
    basic_variables: list[int],
    non_basic_variables: list[int],
    Binv: Float[np.ndarray, "constraints constraints"],
) -> Float[np.ndarray, "non_basic_vars"]:
    
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
        success, (_, _, objective_value) = self.solve(feasibility_problem, B=B, log=False)

        if success != SolverStatus.SUCCESS:
            if success == SolverStatus.INFEASIBLE:
                code = "infeasible"
            elif success == SolverStatus.UBOUNDED:
                code = "unbounded"
            elif success == SolverStatus.CYCLING:
                code = "cycling"

            raise RuntimeError(
                f"Could not solve auxiliary problem to find feasible starting point. Solver finished with status code: {code}"
            )

        if objective_value != 0:
            return SolverStatus.INFEASIBLE, None
        return SolverStatus.SUCCESS, B

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
        
        # Binv is the inverse of the Basis matrix (subset of columns of A)
        Binv: Float[np.ndarray, "constraints constraints"] = np.linalg.inv(A[:, B])
        
        # x_basis is the values of the basic variables
        x_basis: Float[np.ndarray, "constraints"] = Binv @ problem.rhs

        if log:
            print("Starting simplex algorithm...")

        basis_history: list[list[int]] = [] # Used to check for cycling, only degenerate steps are recorded.
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

            if np.all(reduced_costs >= 0):
                status = SolverStatus.SUCCESS
                break

            entering_variable: int = self.pivoting_strategy_.pick_entering_index(reduced_costs, non_basic_variables)
            
            # u is the pivot column in the current basis
            u: Float[np.ndarray, "constraints"] = Binv @ A[:, entering_variable]
            
            # -u is the change in the current basic variables per unit change of the entering variable.
            # If all elements of -u are >= 0, then we can decrease the objective by an arbitrarily large amount
            # since the entering variable has a negative reduced cost and can be increased forever without ever
            # violating any constraint (basic variables reacing 0).
            if np.all(u <= 0):
                status = SolverStatus.UBOUNDED
                break

            ratios = np.array([xi / ui for (xi, ui) in zip(x_basis, u)])
            exiting_variable = self.pivoting_strategy_.pick_exiting_index(ratios, u, B)
       
            basic_exiting_index = B.index(exiting_variable) 
            B[basic_exiting_index] = entering_variable
            Binv = math.update_inverse(A, Binv, entering_variable, basic_exiting_index)

            # Update the solution x based on the pivot column.
            x_basis -= ratios[basic_exiting_index] * u
            x_basis[basic_exiting_index] = ratios[basic_exiting_index]

            objective_history.append(c[B] @ x_basis)

            # Check for cycling
            if objective_history[-1] == objective_history[-2]:
                # Update basis history if objective doesn't change
                basis_signature = tuple(sorted(B))
                if basis_signature in basis_history:
                    status = SolverStatus.CYCLING
                    break
                basis_history.append(basis_signature)
                pass

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