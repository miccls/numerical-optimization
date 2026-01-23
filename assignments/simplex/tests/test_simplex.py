import time
from typing import TYPE_CHECKING

import jaxtyping
import numpy as np
import pytest

from simplex_solutions import linear_algebra, lp_problem, pivoting_strategy, solver
from simplex_solutions.numpy_type_aliases import ArrayI

if TYPE_CHECKING:
    from simplex_solutions.numpy_type_aliases import ArrayF


class TestPivoting:
    def test_jaxtyping(self) -> None:
        """Test that we get a runtime error from jaxtyping due to calling the
        function with arrays with dimensions that are not consistent with the annotations.
        """

        with pytest.raises(jaxtyping.TypeCheckError):
            pivoting_strategy.SmallestSubscriptRule().pick_entering_index(
                np.array([1.0, 2.0, 3.0]), np.array([1, 2])
            )

    def test_smallest_subscript_entering(self) -> None:
        reduced_costs = np.array([0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
        non_basic_vars = np.array([0, 4, 5, 6, 3, 1, 2])
        assert (
            pivoting_strategy.SmallestSubscriptRule().pick_entering_index(
                reduced_costs, non_basic_vars
            )
            == 1
        )

    def test_smallest_subscript_exiting(self) -> None:
        x_basis = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        d = np.array(
            [1.0, -4.0, 1.0, 8.0, 1.0, 12.0],
        )
        assert x_basis / d == pytest.approx(np.array([1.0, -0.5, 3.0, 0.5, 5.0, 0.5]))

        basic_vars = np.array([10, 2, 4, 20, 7, 6])
        exiting_index = pivoting_strategy.SmallestSubscriptRule().pick_exiting_index(
            basic_vars, x_basis, d
        )
        exiting_variable = basic_vars[exiting_index]

        assert exiting_index == 5
        assert exiting_variable == 6


class TestInverseComputation:
    def test_update_inverse(self) -> None:
        a = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
        basis = np.array([1, 2], dtype=int)
        basis_matrix = a[:, basis]
        assert basis_matrix == pytest.approx(
            np.array(
                [
                    [2.0, 3.0],
                    [3.0, 2.0],
                ]
            )
        )
        b_inv = np.linalg.inv(basis_matrix)

        # Switch out variable 2 for variable 3
        entering_variable = 3
        exiting_index = 1
        assert basis[exiting_index] == 2
        basis[exiting_index] = entering_variable

        b_inv = linear_algebra.update_inverse(
            a, b_inv, entering_variable, exiting_index
        )
        b_inv_expected = np.linalg.inv(a[:, basis])

        assert b_inv == pytest.approx(b_inv_expected)

    def test_update_speed(self) -> None:
        rng = np.random.default_rng(1337)
        random_column = rng.uniform(low=0, high=1, size=(500, 1))

        constraint_matrix = np.concatenate((np.eye(500), random_column), axis=1)
        basis = list(range(500))
        entering_variable = 500
        exiting_index = 499
        new_basis = [*basis[:-1], 500]

        inv_basis_matrix: ArrayF = np.eye(500)

        start_time = time.perf_counter()
        inv_basis_matrix = linear_algebra.update_inverse(
            constraint_matrix,
            inv_basis_matrix,
            entering_variable,
            exiting_index,
        )
        end_time = time.perf_counter()

        time_in_ms = (end_time - start_time) * 1000

        assert inv_basis_matrix @ constraint_matrix[:, new_basis] == pytest.approx(
            np.eye(500)
        )

        time_limit_ms = 10
        assert time_in_ms < time_limit_ms, (
            f"Should take less than {time_limit_ms} ms, took {time_in_ms}"
        )


@pytest.fixture()
def example_problem_35() -> lp_problem.LpProblem:
    """Example 3.5 in "Introduction to Linear Programming", page 101."""
    a = np.array(
        [
            [1, 2, 2, 1, 0, 0],
            [2, 1, 2, 0, 1, 0],
            [2, 2, 1, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    b = np.array([20, 20, 20], dtype=np.float64).T

    c = np.array([-10, -12, -12, 0, 0, 0], dtype=np.float64)

    return lp_problem.LpProblem(a, b, c)


@pytest.fixture()
def example_problem_131() -> tuple[lp_problem.LpProblem, ArrayI]:
    """Example 13.1 in "Introduction to Linear Programming", page 371."""
    a = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 0.5, 0.0, 1.0],
        ],
    )
    b = np.array([5.0, 8.0])
    c = np.array([-4.0, -2.0, 0.0, 0.0])

    initial_basis = np.array([2, 3])

    return lp_problem.LpProblem(a, b, c), initial_basis


class TestSolver:
    def test_problem_with_start_solution(
        self, example_problem_35: lp_problem.LpProblem
    ) -> None:
        # Starting basis.
        basis = np.array([3, 4, 5])

        inv_basis_matrix = np.linalg.inv(example_problem_35.constraint_matrix[:, basis])
        x = inv_basis_matrix @ example_problem_35.rhs

        # Assert feasibility of starting point.
        assert example_problem_35.constraint_matrix[:, basis] @ x == pytest.approx(
            example_problem_35.rhs
        )
        assert (x >= 0).all()

        solve_result = solver.Solver().solve(example_problem_35, initial_basis=basis)

        assert sorted(solve_result.basis) == [0, 1, 2]
        assert solve_result.solution == pytest.approx(np.array([4, 4, 4, 0, 0, 0]))
        assert solve_result.objective_value == pytest.approx(-136)

    def test_find_initial_basis(self, example_problem_35: lp_problem.LpProblem) -> None:
        assert len(solver.Solver().find_initial_basis(example_problem_35)) == 3

    def test_problem_without_start_solution(
        self, example_problem_35: lp_problem.LpProblem
    ) -> None:
        solve_result = solver.Solver().solve(example_problem_35)

        assert sorted(solve_result.basis) == [0, 1, 2]
        assert solve_result.solution == pytest.approx(np.array([4, 4, 4, 0, 0, 0]))
        assert solve_result.objective_value == pytest.approx(-136)

    def test_chvatals_example_for_cycling(self) -> None:
        # See for ref: https://www.matem.unam.mx/~omar/math340/degenerate.html or
        # https://sites.math.washington.edu/~vinzant/teaching/407/Chvatal.pdf, page 31.
        # This is an example of an LP which cycles under some pivot rules.
        a = np.array(
            [
                [-0.5, 5.5, 2.5, -9, -1, 0, 0],
                [-0.5, 1.5, 0.5, -1, 0, -1, 0],
                [1, 0, 0, 0, 0, 0, 1],
            ]
        )
        b = np.array([0, 0, 1]).T
        c = np.array([-10, 57, 9, 24, 0, 0, 0])

        with pytest.raises(solver.SimplexCyclingError):
            solver.Solver().solve(lp_problem.LpProblem(a, b, c))

    def test_infeasible_problem(self) -> None:
        # min -x1 s.t. x1 <= 1, x1 >= 2
        # Standard form: x1 + s1 = 1, x1 - s2 = 2
        a = np.array([[1, 1, 0], [1, 0, -1]])
        b = np.array([1, 2])
        c = np.array([-1, 0, 0])

        test_problem = lp_problem.LpProblem(a, b, c)
        with pytest.raises(solver.InfeasibleLpError):
            solver.Solver().solve(test_problem)

    def test_unbounded_problem(self) -> None:
        # min -2*x1 - x2 s.t. x1 - x2 <= 10, 2*x1 <= 40
        # Standard form: x1 - x2 + s1 = 10, 2*x1 + s2 = 40
        a = np.array([[1, -1, 1, 0], [2, 0, 0, 1]])
        b = np.array([10, 40])
        c = np.array([-2, -1, 0, 0])

        test_problem = lp_problem.LpProblem(a, b, c)
        with pytest.raises(solver.UnboundedLpError):
            solver.Solver().solve(test_problem)

    def test_example_13_1_from_book(
        self, example_problem_131: tuple[lp_problem.LpProblem, ArrayI]
    ) -> None:
        # Lukas spotted typo in book, test to verify!
        simplex_solver = solver.Solver()
        solve_result = simplex_solver.solve(
            problem=example_problem_131[0], initial_basis=example_problem_131[1]
        )
        assert solve_result.objective_value == pytest.approx(-52 / 3)
        assert simplex_solver.history.basis_history == [(2, 3), (0, 2), (0, 1)]
        assert simplex_solver.history.objective_history == pytest.approx(
            [0.0, -16.0, -52 / 3]
        )

    def test_example_13_1_from_book_without_initial_basis(
        self, example_problem_131: tuple[lp_problem.LpProblem, ArrayI]
    ) -> None:
        solve_result = solver.Solver().solve(example_problem_131[0])
        assert solve_result.objective_value == pytest.approx(-52 / 3)
