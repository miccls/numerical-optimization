import time

import numpy as np
import pytest

from simplex import lp_problem, math, pivoting_strategy, solver


class TestPivoting:
    def test_smallest_subscript_entering(self) -> None:
        reduced_costs = np.array([1, 2, 3, 4, -1, -2, -3], dtype=float)
        non_basic_vars = np.array([3, 4, 5, 6, 0, 1, 2])
        assert (
            pivoting_strategy.SmallestSubscriptRule().pick_entering_index(
                reduced_costs, non_basic_vars
            )
            == 0
        )

    # def test_smallest_subscript_exiting(self) -> None:
    #     x = np.array([1, 2, 3, 4, 5, 6])
    #     u = np.array([1, -4, 1, 8, 1, 12])
    #     basic_vars = [10, 2, 4, 6, 7, 20]
    #     ratios = np.array([xi / ui for xi, ui in zip(x, u, strict=True)])

    #     assert (
    #         pivoting_strategy.SmallestSubscriptRule().pick_exiting_index(
    #             ratios, u, basic_vars
    #         )
    #         == 6
    #     )


class TestInverseComputation:
    def test_update_inverse(self) -> None:
        a = np.array(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
            ],
            dtype=float,
        )
        basis = np.array([1, 2], dtype=int)
        basis_matrix = a[:, basis]
        assert basis_matrix == pytest.approx(
            np.array(
                [
                    [2, 3],
                    [3, 2],
                ],
                dtype=float,
            )
        )
        b_inv = np.linalg.inv(basis_matrix)

        # Switch out variable 2 for variable 3
        entering_variable = 3

        exiting_index = 1
        exiting_variable = basis[exiting_index]
        assert exiting_variable == 2
        basis[exiting_index] = entering_variable

        b_inv = math.update_inverse(a, b_inv, entering_variable, exiting_index)

        assert np.allclose(b_inv, np.linalg.inv(a[:, basis]))

    def test_update_speed(self) -> None:
        rng = np.random.default_rng(1337)
        random_column = rng.uniform(low=0, high=1, size=(500, 1))

        constraint_matrix = np.concatenate((np.eye(500), random_column), axis=1)
        basis = list(range(500))
        entering_variable = 500
        exiting_index = 499
        new_basis = [*basis[:-1], 500]

        inv_basis_matrix = np.eye(500)

        start_time = time.perf_counter()
        inv_basis_matrix = math.update_inverse(
            constraint_matrix,
            inv_basis_matrix,
            entering_variable,
            exiting_index,
        )
        end_time = time.perf_counter()

        time_in_ms = (end_time - start_time) * 1000

        assert np.allclose(
            inv_basis_matrix @ constraint_matrix[:, new_basis], np.eye(500)
        )
        assert time_in_ms < 10, f"Should take less than 10 ms, took {time_in_ms}"


@pytest.fixture()
def example_problem_35() -> lp_problem.LpProblem:
    # Example 3.5 in "Introduction to Linear Programming", page 101.
    a = np.array(
        [
            [1, 2, 2, 1, 0, 0],
            [2, 1, 2, 0, 1, 0],
            [2, 2, 1, 0, 0, 1],
        ]
    )

    b = np.array([20, 20, 20]).T

    c = np.array([-10, -12, -12, 0, 0, 0])

    return lp_problem.LpProblem(a, b, c)


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

        simplex_solver = solver.Solver(pivoting_strategy.SmallestSubscriptRule())

        (status, solve_result) = simplex_solver.solve(
            example_problem_35, initial_basis=basis
        )

        assert status == solver.SolverStatus.SUCCESS
        assert solve_result is not None
        assert sorted(solve_result.basis) == [0, 1, 2]
        assert solve_result.solution == pytest.approx(np.array([4, 4, 4, 0, 0, 0]))
        assert solve_result.objective_value == pytest.approx(-136)

    def test_problem_without_start_solution(
        self, example_problem_35: lp_problem.LpProblem
    ) -> None:
        simplex_solver = solver.Solver(pivoting_strategy.SmallestSubscriptRule())

        (status, solve_result) = simplex_solver.solve(example_problem_35)

        assert status == solver.SolverStatus.SUCCESS
        assert solve_result is not None
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

        simplex_solver = solver.Solver(pivoting_strategy.SmallestSubscriptRule())
        status, _ = simplex_solver.solve(lp_problem.LpProblem(a, b, c))

        assert status == solver.SolverStatus.CYCLING

    def test_infeasible_problem(self) -> None:
        # min -x1 s.t. x1 <= 1, x1 >= 2
        # Standard form: x1 + s1 = 1, x1 - s2 = 2
        a = np.array([[1, 1, 0], [1, 0, -1]])
        b = np.array([1, 2])
        c = np.array([-1, 0, 0])

        simplex_solver = solver.Solver(pivoting_strategy.SmallestSubscriptRule())
        test_problem = lp_problem.LpProblem(a, b, c)
        status, _ = simplex_solver.solve(test_problem)
        assert status == solver.SolverStatus.INFEASIBLE

    def test_unbounded_problem(self) -> None:
        # min -2*x1 - x2 s.t. x1 - x2 <= 10, 2*x1 <= 40
        # Standard form: x1 - x2 + s1 = 10, 2*x1 + s2 = 40
        a = np.array([[1, -1, 1, 0], [2, 0, 0, 1]])
        b = np.array([10, 40])
        c = np.array([-2, -1, 0, 0])

        simplex_solver = solver.Solver(pivoting_strategy.SmallestSubscriptRule())
        test_problem = lp_problem.LpProblem(a, b, c)
        status, _ = simplex_solver.solve(test_problem)

        assert status == solver.SolverStatus.UNBOUNDED

    def test_example_13_1_from_book(self) -> None:
        # Lukas spotted typo in book, test to verify!

        a = np.array([[1, 1, 1, 0], [2, 0.5, 0, 1]])
        b = np.array([5, 8]).T
        c = np.array([-4, -2, 0, 0])

        simplex_solver = solver.Solver(pivoting_strategy.SmallestSubscriptRule())
        status, solve_result = simplex_solver.solve(lp_problem.LpProblem(a, b, c))
        print(f"{solve_result=}")

        assert status == solver.SolverStatus.SUCCESS
        assert solve_result is not None
        assert solve_result.objective_value == pytest.approx(-52 / 3)
