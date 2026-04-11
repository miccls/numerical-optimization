import logging
import time

import numpy as np
from common import lp_problem

from ipm import ipm_tools

logger = logging.getLogger(__name__)


def calculate_starting_point(
    problem: lp_problem.LpProblem,
) -> ipm_tools.PrimalDualTuple:
    """Calculates a suitable starting point for the predictor corrector algorithm
    according to the recipe on p. 410 in Nocedal & Wright"""

    # TODO(you): Calculate a suitable starting point according to the recipe on p. 410.

    return ipm_tools.PrimalDualTuple(
        x=np.zeros(problem.constraint_matrix.shape[1]),
        lam=np.zeros(problem.constraint_matrix.shape[0]),
        s=np.zeros(problem.constraint_matrix.shape[1]),
    )


def update_point(
    point: ipm_tools.PrimalDualTuple,
    step: ipm_tools.PrimalDualTuple,
    primal_step_size: float,
    dual_step_size: float,
) -> ipm_tools.PrimalDualTuple:
    """Updates the current solution using a primal dual step and
    corresponding step sizes. Corresponds to the last two steps in the
    for-loop of algorithm 14.3 on p. 411 in the book."""

    # TODO(you): Update the current solution with the provided step and step sizes.

    return ipm_tools.PrimalDualTuple(
        x=np.zeros(point.x.shape),
        lam=np.zeros(point.lam.shape),
        s=np.zeros(point.s.shape),
    )


class PredictorCorrector:
    def __init__(self, max_iterations: int, optimality_tolerance: float) -> None:
        self.max_iterations = max_iterations
        self.optimality_tolerance = optimality_tolerance

    def solve(self, problem: lp_problem.LpProblem) -> ipm_tools.PrimalDualTuple:
        """Solves the LP `problem` using algorithm 14.3 on p. 411 in Nocedal & Wright."""

        point = calculate_starting_point(problem)

        logger.info("                Objective              Residual")
        logger.info(
            "Iter       Primal       Dual       Primal      Dual     Compl       Time"
        )

        iteration = 0
        start_time = time.time()
        while (
            ipm_tools.calculate_duality_measure(point.x, point.s)
            > self.optimality_tolerance
            and iteration < self.max_iterations
        ):
            # TODO(you): Solve for Predictor-Corrector direction
            predictor_corrector_direction = (
                ipm_tools.solve_predictor_corrector_direction(problem, point)
            )

            # TODO(you): Calculate step sizes (according to (14.38), p. 409)
            primal_step_size = 0.0
            dual_step_size = 0.0

            # TODO(you): Update the point.
            point = update_point(
                point, predictor_corrector_direction, primal_step_size, dual_step_size
            )

            primal_obj = problem.objective.T @ point.x
            dual_obj = problem.rhs.T @ point.lam
            primal_res = np.linalg.norm(
                problem.constraint_matrix @ point.x - problem.rhs
            )
            dual_res = np.linalg.norm(
                problem.constraint_matrix.T @ point.lam + point.s - problem.objective
            )
            compl = ipm_tools.calculate_duality_measure(point.x, point.s)
            duration = time.time() - start_time

            logger.info(
                f"{iteration:4d}   {primal_obj:10.3e}  {dual_obj:10.3e}  "
                f"{primal_res:10.3e} {dual_res:10.3e}  {compl:8.3e}    {duration:5.2}s"
            )
            iteration += 1

        return point
