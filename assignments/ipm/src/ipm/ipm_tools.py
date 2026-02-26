from dataclasses import dataclass

import jaxtyping
import numpy as np
from common import lp_problem
from common.numpy_type_aliases import ArrayF


@dataclass(frozen=True)
class PrimalDualTuple:
    x: jaxtyping.Float[ArrayF, " n"]
    lam: jaxtyping.Float[ArrayF, " m"]
    s: jaxtyping.Float[ArrayF, " n + m"]


def calculate_max_step_size(
    x: jaxtyping.Float[ArrayF, " n"], dx: jaxtyping.Float[ArrayF, " n"]
) -> float:
    """Calculates the maximum possible step size in the direction `dx`
    such that no variable in `x` becomes negative. Used for both primal and
    dual variables. For details, see p. 409, (14.36) in Nocedal & Wright."""
    # TODO(you): Calculate the maximum allowed step size
    return 0.0


def calculate_affine_step_size(
    x: jaxtyping.Float[ArrayF, " n"], dx: jaxtyping.Float[ArrayF, " n"]
) -> float:
    """Calculates the maximum allowable step length for the affine scaling algorithm. Used for both primal and
    dual variables. For details, see p. 408, (14.32) in Nocedal & Wright."""
    # TODO(you): Calculate the maximum allowable step size in the affine scaling (Newton) direction
    return 0.0


def calculate_duality_measure(
    primal_vars: jaxtyping.Float[ArrayF, " n"], dual_vars: jaxtyping.Float[ArrayF, " n"]
) -> float:
    r"""Calculate the duality measure, `mu`, which is the average complementary slackness."""
    # TODO(you): Calculate the duality measure
    return 0.0


def calculate_mu_after_step(
    point: PrimalDualTuple,
    step: PrimalDualTuple,
) -> float:
    """Calculates the value of the duality measure if one steps the
    primal and dual variables according to the input data.
    See p. 408, (14.33) in Nocedal & Wright for details."""
    # TODO(you): Calculate the duality measure resulting from the step `step`
    return 0.0


def solve_ipm_system(
    a: jaxtyping.Float[ArrayF, "n m"],
    point: PrimalDualTuple,
    r_c: jaxtyping.Float[ArrayF, " n"],
    r_b: jaxtyping.Float[ArrayF, " m"],
    r_xs: jaxtyping.Float[ArrayF, " n"],
) -> PrimalDualTuple:
    """Solves the system (14.41) in the book.
    Input args:
        a: The LP constraint matrix
        point: The primal and dual variables, `(x, lambda, s)`
        r_c: The dual feasibility residual, see p. 398 (14.7)
        r_b: The primal feasibility residual, see p. 398 (14.7)
        r_xs: Algorithm dependent duality residual, see for example (14.8), (14.9), (14.35)

    Output:
        Solution of system (14.41)
    """
    # TODO(you): Solve the system (14.41)
    return PrimalDualTuple(
        x=np.zeros(a.shape[1]), lam=np.zeros(a.shape[0]), s=np.zeros(a.shape[1])
    )


def calculate_centering_parameter(
    point: PrimalDualTuple, step: PrimalDualTuple
) -> float:
    """Calculates the centering parameter used to set up the system solved
    for the predictor-corrector step. See p. 408, (14.34).
    """
    # TODO(you): Calculate the centering parameter used in the Predictor Corrector algorithm
    return 0.0


def solve_newtown_direction(
    lp_problem: lp_problem.LpProblem,
    point: PrimalDualTuple,
) -> PrimalDualTuple:
    """Solves the system (14.35) in Nocedal & Wright for the Newton direction"""

    a = lp_problem.constraint_matrix

    # TODO(you): Compute the proper residuals used to solve for the Newton direction
    r_c = np.zeros(a.shape[1])
    r_b = np.zeros(a.shape[0])
    r_xs = np.zeros(a.shape[1])

    return solve_ipm_system(a, point, r_c, r_b, r_xs)


def solve_affine_scaling_step(
    lp_problem: lp_problem.LpProblem,
    point: PrimalDualTuple,
) -> PrimalDualTuple:
    """Solves the system (14.30) for the Newton direction and scales
    with the affine scaling step sizes."""

    # TODO(you): Construct the affine scaling step

    return PrimalDualTuple(
        x=np.zeros(lp_problem.constraint_matrix.shape[1]),
        lam=np.zeros(lp_problem.constraint_matrix.shape[0]),
        s=np.zeros(lp_problem.constraint_matrix.shape[1]),
    )


def solve_predictor_corrector_direction(
    lp_problem: lp_problem.LpProblem,
    point: PrimalDualTuple,
) -> PrimalDualTuple:
    """Solves the system (14.35) in Nocedal & Wright for the predictor corrector direction"""

    a = lp_problem.constraint_matrix

    # TODO(you): Compute the proper residuals used to solve for the Predictor Corrector direction
    r_c = np.zeros(a.shape[1])
    r_b = np.zeros(a.shape[0])
    r_xs = np.zeros(a.shape[1])

    return solve_ipm_system(a, point, r_c, r_b, r_xs)
