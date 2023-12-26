from board import Board
from solver import Solver, Status

import unittest


class SolverTest(unittest.TestCase):

    def _solve(self, input_board: Board):
        solver = Solver(input_board)
        return solver.solve(input_board)

    def _solve_and_check_solution(self, input_string_board: str, expected_string_board: str):
        input_board = Board.from_string(input_string_board)
        expected_board = Board.from_string(expected_string_board)

        solver_result = self._solve(input_board)
        self.assertEqual(solver_result.solver_status, Status.SOLVED)
        self.assertEqual(len(solver_result.solutions), 1)

        solution = solver_result.solutions[0]
        self.assertTrue(
            solution.is_solved(),
            f"""Did not solve puzzle.
input:
{input_board}

unsolved output:
{solution.board}""")
        self.assertEqual(solution.board.grid, expected_board.grid,
                         f"{solution}\n{expected_board}")

    def test_9x9_puzzle_1(self):
        input_board = """
          4 1 _ _ 6 _ _ 7 _
          _ _ 3 _ 8 5 _ _ 9
          _ 2 _ 3 7 _ 5 _ 1
          _ 3 _ 6 _ 9 2 5 _
          6 _ _ 5 _ 1 _ _ _
          _ _ 9 _ 2 _ _ _ 3
          _ _ 6 2 _ _ 7 4 5
          _ _ _ 4 _ 6 8 _ _
          2 8 4 _ _ _ 1 9 6
        """
        expected_board = """
          4 1 5 9 6 2 3 7 8
          7 6 3 1 8 5 4 2 9
          9 2 8 3 7 4 5 6 1
          8 3 1 6 4 9 2 5 7
          6 7 2 5 3 1 9 8 4
          5 4 9 8 2 7 6 1 3
          3 9 6 2 1 8 7 4 5
          1 5 7 4 9 6 8 3 2
          2 8 4 7 5 3 1 9 6
        """
        self._solve_and_check_solution(input_board, expected_board)

    def test_9x9_very_difficult_1(self):
        input_board = """
          4 5 . . . . . . .
          . . 2 . 7 . 6 3 .
          . . . . . . . 2 8
          . . . 9 5 . . . .
          . 8 6 . . . 2 . .
          . 2 . 6 . . 7 5 .
          . . . . . . 4 7 6
          . 7 . . 4 5 . . .
          . . 8 . . 9 . . .
        """
        expected_board = """
          4 5 3 8 2 6 1 9 7
          8 9 2 5 7 1 6 3 4
          1 6 7 4 9 3 5 2 8
          7 1 4 9 5 2 8 6 3
          5 8 6 1 3 7 2 4 9
          3 2 9 6 8 4 7 5 1
          9 3 5 2 1 8 4 7 6
          6 7 1 3 4 5 9 8 2
          2 4 8 7 6 9 3 1 5
        """
        self._solve_and_check_solution(input_board, expected_board)

    def test_9x9_very_difficult_30(self):
        input_board = """
          8 . . . . . . 7 .
          . 6 . . . . 5 . 1
          . . . 3 1 9 6 . .
          . . . . . . . . 4
          . 9 8 . . 1 . . .
          4 . . 2 9 5 . . .
          . . 1 6 . 8 . 5 .
          . 3 2 . . . . 8 .
          6 . 9 . . 3 . . .
        """
        expected_board = """
          8 1 3 5 6 4 9 7 2
          9 6 4 7 8 2 5 3 1
          2 5 7 3 1 9 6 4 8
          1 2 5 8 3 6 7 9 4
          3 9 8 4 7 1 2 6 5
          4 7 6 2 9 5 8 1 3
          7 4 1 6 2 8 3 5 9
          5 3 2 9 4 7 1 8 6
          6 8 9 1 5 3 4 2 7
        """
        self._solve_and_check_solution(input_board, expected_board)

    def test_9x9_nyt_12_24_2023(self):
        # Not trivially solvable!
        input_board = """
          . 2 . . 3 . . . .
          4 9 . . . . . . 5
          . 1 . 9 . 7 . . .
          1 7 . 5 . . . 9 4
          . . 2 . . 3 . 7 .
          . . . . . 6 . . .
          . 3 . . . 8 . 1 .
          . . . . 6 . . . .
          . . . 4 . . . . 7
        """
        expected_board = """
          5 2 7 6 3 4 9 8 1
          4 9 6 8 2 1 7 3 5
          3 1 8 9 5 7 2 4 6
          1 7 3 5 8 2 6 9 4
          9 6 2 1 4 3 5 7 8
          8 5 4 7 9 6 1 2 3
          6 3 5 2 7 8 4 1 9
          7 4 1 3 6 9 8 5 2
          2 8 9 4 1 5 3 6 7
        """
        self._solve_and_check_solution(input_board, expected_board)
    
    def test_no_solution_invalid_board_9x9(self):
        # Bottom right subgrid contains two 7's.
        input_string_board = """
          . 2 . . 3 . . . .
          4 9 . . . . . . 5
          . 1 . 9 . 7 . . .
          1 7 . 5 . . . 9 4
          . . 2 . . 3 . 7 .
          . . . . . 6 . . .
          . 3 . . . 8 7 1 .
          . . . . 6 . . . .
          . . . 4 . . . . 7
        """
        input_board = Board.from_string(input_string_board)
        solver_result = self._solve(input_board)
        self.assertEqual(solver_result.solver_status, Status.NO_SOLUTION)
    
    def test_no_solution_valid_board_4x4(self):
        # Board is valid but still cannot be solved.
        input_string_board = """
          3 . . 1
          1 . . .
          4 . . .
          . . 2 .
        """
        input_board = Board.from_string(input_string_board)
        self.assertTrue(input_board.is_valid())
        solver_result = self._solve(input_board)
        self.assertEqual(solver_result.solver_status, Status.NO_SOLUTION)

    def test_4x4_difficult_1(self):
        input_board = """
          3 . . 1
          1 . . .
          . . . .
          . . 2 .
        """
        expected_board = """
          3 2 4 1
          1 4 3 2
          2 3 1 4
          4 1 2 3
        """
        self._solve_and_check_solution(input_board, expected_board)

    def test_4x4_difficult_50(self):
        input_board = """
          . 4 . .
          . . . 2
          . . . .
          3 2 . .
        """
        expected_board = """
          2 4 3 1
          1 3 4 2
          4 1 2 3
          3 2 1 4
        """
        self._solve_and_check_solution(input_board, expected_board)


if __name__ == '__main__':
    unittest.main()
