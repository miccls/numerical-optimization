from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from jaxtyping import Float

from simplex import lp_problem, math, pivoting_strategy

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
        # TODO: Set up an auxiliary LP whose solution is a basic feasible solution to the original problem
        # See page 378 in the book, for example.
        pass

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
            
            # Step 1: Compute reduced costs
            reduced_costs = []

            if np.all(reduced_costs >= 0):
                status = SolverStatus.SUCCESS
                break

            # Step 2: Determine the entering variable
            entering_variable: int = self.pivoting_strategy_.pick_entering_index()
            
            # d is the "basic direction" of the entering variable
            d: Float[np.ndarray, "constraints"] = Binv @ A[:, entering_variable]
            
            # -d is the change in the current basic variables per unit change of the entering variable.
            # If all elements of -d are >= 0, then we can decrease the objective by an arbitrarily large amount
            # since the entering variable has a negative reduced cost and can be increased forever without ever
            # violating any constraint (basic variables reacing 0).
            if np.all(d <= 0):
                status = SolverStatus.UBOUNDED
                break

            # Step 3: Determine the exiting variable
            exiting_variable = self.pivoting_strategy_.pick_exiting_index()
   
            # Step 4: Update the inverse of the basis matrix (feel free to change input args to update_inverse if desirable)
            basic_exiting_index = B.index(exiting_variable) 
            B[basic_exiting_index] = entering_variable
            Binv = math.update_inverse(A, Binv, entering_variable, basic_exiting_index) # <-- TODO in here.

            # Step 5: Update the basic solution from the basic direction
            # TODO...

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