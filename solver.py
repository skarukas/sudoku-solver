from board import Board, SudokuIndexer
import utils
from copy import deepcopy
from typing import TypeAlias
import enum
import dataclasses
import random
from queue import PriorityQueue, SimpleQueue, LifoQueue

RawBoard: TypeAlias = list[list[int | None]]
CandidateBoard: TypeAlias = list[list[set[int]]]


class SearchMethod(enum.Enum):
    DFS = 'dfs'
    BFS = 'bfs'
    RANDOM_PRIORITY = 'random'


@dataclasses.dataclass
class TilePlacement:
    i: int
    j: int
    value: int
    is_trivial: bool = True


class InvalidBoardException(Exception):
    pass


class Status(enum.Enum):
    SOLVED = 0
    NO_SOLUTION = 1


_EMPTY_LIST_FIELD = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class BoardState:
    board: Board
    steps: list[TilePlacement] = _EMPTY_LIST_FIELD

    def has_trivial_solution(self) -> bool:
        """Maybe move this so we know it's solved."""
        return all(map(lambda s: s.is_trivial, self.steps))

    def is_solved(self) -> bool:
        return self.board.is_solved()

    def set_tile(self, i: int, j: int, value: int, is_trivial_step: bool) -> 'BoardState':
        if self.get_tile(i, j):
            raise ValueError(
                f"Cannot set ({i}, {j}) as there is already a tile there. board=\n{self.board}")
        new_grid = deepcopy(self.board.grid)
        new_grid[i][j] = value
        step = TilePlacement(i, j, value, is_trivial_step)
        return BoardState(
            Board(new_grid),
            [*self.steps, step]
        )

    def get_tile(self, i: int, j: int) -> int:
        return self.board.grid[i][j]

    def __hash__(self) -> int:
        return hash(self.board.to_string(include_separators=False))


@dataclasses.dataclass(frozen=True)
class SolverResult:
    solver_status: Status
    method: SearchMethod
    solutions: list[BoardState] = _EMPTY_LIST_FIELD


class Solver:
    def __init__(self, board: Board):
        self.indexer = SudokuIndexer(board.num_subgrids, board.num_subgrids)

    def _get_full_candidate_board(self, board_size: int) -> CandidateBoard:
        return [[set(range(1, board_size+1))
                 for _ in range(board_size)]
                for _ in range(board_size)]

    def get_candidates(self, state: BoardState) -> CandidateBoard:
        """Returns a grid containing each possible number that could be placed in each spot according to the constraints. A spot that already has a number will have an empty candidate set."""
        grid = state.board.grid
        candidates = self._get_full_candidate_board(len(grid))

        for (i, j) in self.indexer.all_coords():
            tile_value = grid[i][j]
            if tile_value:  # (is not blank)
                candidates[i][j] = set()
                tile = TilePlacement(i, j, tile_value)
                self._update_candidates_from_tile_placement(tile, candidates)
        return candidates

    def _update_candidates_from_tile_placement(
            self,
            tile_placement: TilePlacement,
            candidates: CandidateBoard):
        for ii, jj in self.indexer.get_all_containing_sequences(tile_placement.i, tile_placement.j):
            st = candidates[ii][jj]
            if tile_placement.value in st:
                st.remove(tile_placement.value)

    def apply_single_candidates(self, state: BoardState, candidates: CandidateBoard) -> BoardState:
        """Wherever there is only one candidate, place that number there."""
        for (i, j), candidate_set in utils.flattened(candidates):
            if len(candidate_set) == 1:
                value, *_ = candidate_set
                state = state.set_tile(i, j, value, is_trivial_step=True)
                self._update_candidates_from_tile_placement(
                    TilePlacement(i, j, value), candidates)

        return state

    def apply_elimination_candidates(self, state: BoardState, candidates: CandidateBoard) -> BoardState:
        """Wherever only one tile in a sequence contains a candidate, apply that candidate.

        For example, pretend there's a 4-tile row of candidates:
        [1, 2] [1, 2] [] [4, 1, 2]

        Actual row numbers: _ _ 3 _

        Logic:
         - [4, 1, 2] is the only candidate set that contains a 4 
         - Each sequence (including this row) must contain all numbers from 1 to 4
        Therefore, the number that goes there MUST be a 4.

        So we can turn that row into _ _ 3 4.
        """
        for sequence in self.indexer.get_all_sequences():
            # Need to loop over it twice, so save the elements.
            sequence = [*sequence]
            for i, j in sequence:
                if not state.get_tile(i, j):
                    # Get all other candidate sets in the sequence.
                    other_sets = [candidates[ii][jj]
                                  for ii, jj in sequence if (ii, jj) != (i, j)]
                    # Any elements that are ONLY in this candidate set.
                    diff = candidates[i][j].difference(*other_sets)
                    if len(diff) == 1:
                        value, *_ = diff
                        state = state.set_tile(
                            i, j, value, is_trivial_step=True)
                        self._update_candidates_from_tile_placement(
                            TilePlacement(i, j, value), candidates)
                    elif diff:
                        raise InvalidBoardException(
                            f"According to candidate analysis, tile ({i}, {j}) must take on more than one value: {diff}. This board has no solution.")
        return state

    def assert_has_valid_child_states(self, candidates: CandidateBoard, state: BoardState) -> bool:
        """Validates that all spots without a number have at least one candidate number that can be put down."""
        for (i, j), candidate_set in utils.flattened(candidates):
            # There either must be 1+ candidates, or a number placed.
            if not (state.get_tile(i, j) or candidate_set):
                raise InvalidBoardException(
                    f"Tile {(i, j)} is empty but has no valid values, so the board cannot be solved.")

    def resolve_trivial_constraints(self, state: BoardState) -> BoardState:
        """Resolve all tiles where constraints lead to a definite solution."""
        prev_state = None
        updated_state = state
        while updated_state != prev_state:
            prev_state = updated_state
            candidates = self.get_candidates(prev_state)
            self.assert_has_valid_child_states(candidates, prev_state)
            updated_state = self.apply_single_candidates(
                updated_state, candidates)
            updated_state = self.apply_elimination_candidates(
                updated_state, candidates)
        return updated_state

    def get_child_states(self, state: BoardState, do_random=False) -> list[BoardState]:
        candidates = self.get_candidates(state)
        children = []
        for (i, j) in self.indexer.all_coords():
            if not state.get_tile(i, j):
                # Each candidate for each tile must be tried out.
                children.extend([
                    state.set_tile(i, j, candidate, is_trivial_step=False)
                    for candidate in candidates[i][j]
                ])
        if do_random:
            random.shuffle(children)
        return children

    def solve(
            self,
            board: Board,
            find_all_solutions: bool = False,
            method: SearchMethod = SearchMethod.DFS,
            debug=False) -> SolverResult:
        """Solve the Sudoku board. With find_all_solutions, the assumption is that the board has multiple solutions (not "proper"), and all possibile DFS branches will be explored."""
        # TODO: how to figure out whether a state is part of a branch already explored?
        # One way to see if they're NOT part of the same branch is if any of the filled-in tiles are different. However, this also applies to parent-child relationships, and we would want to keep the child state.
        local_print = print if debug else lambda *s: None
        solutions = set()
        initial_state = BoardState(board)
        do_random = method == SearchMethod.RANDOM_PRIORITY
        if method == SearchMethod.DFS:
            queue = LifoQueue()
        elif method == SearchMethod.BFS:
            queue = SimpleQueue()
        else:
            queue = PriorityQueue()

        priority = random.random()
        queue.put((priority, initial_state))
        seen_states = set([initial_state])

        if not board.is_valid():
            return SolverResult(Status.NO_SOLUTION, method)
        num_initial_filled_in_tiles = sum([
            1
            for i, j in self.indexer.all_coords()
            if initial_state.get_tile(i, j)
        ])
        total_tiles = self.indexer.num_digits**2
        while not queue.empty():
            _priority, prev_state = queue.get()
            try:
                updated_state = self.resolve_trivial_constraints(prev_state)
            except InvalidBoardException as e:
                local_print(e)
                continue
            if updated_state.is_solved():
                if updated_state not in solutions:
                    solutions.add(updated_state)
                    if not find_all_solutions:
                        break
                continue
            local_print("Unsolved state:", updated_state)
            # No trivial steps to take, use DFS
            child_states = self.get_child_states(updated_state, do_random)
            depth = num_initial_filled_in_tiles + len(updated_state.steps)
            local_print(
                f"Forking to {len(child_states)} possible branches at depth {depth}/{total_tiles}")
            for next_state in child_states:
                if next_state not in seen_states:
                    priority = random.random()
                    queue.put((priority, next_state))
                    seen_states.add(next_state)

        status = Status.SOLVED if solutions else Status.NO_SOLUTION
        return SolverResult(status, method, list(solutions))

    def generate_board(self, num_missing_tiles=60, ensure_unique=False, size=9) -> Board:
        """Generate a board by "solving" an empty board."""
        if ensure_unique:
            raise ValueError("Not sure how to do that yet...")

        grid = [[0 for _ in range(size)] for _ in range(size)]
        solver_result = self.solve(Board(grid), find_all_solutions=False)
        final_state = solver_result.solutions[0]
        board = final_state.board
        steps = final_state.steps
        steps_to_undo = list(reversed(steps))[:num_missing_tiles]
        for step in steps_to_undo:
            board.grid[step.i][step.j] = 0

        return board


if __name__ == "__main__":
    _ = None
    board = Board.from_string("""
          . . . . . . . 7 .
          . 6 . . . . 5 . 1
          . . . . . 9 . . .
          . . . . . . . . 4
          . 9 . . . 1 . . .
          4 . . 2 . 5 . . .
          . . 1 . . . . 5 .
          . . 2 . . . . 8 .
          6 . 9 . . 3 . . .
        """)

    solver = Solver(board)
    solved = solver.solve(board, find_all_solutions=False)
    print(solved)

    for _ in range(10):
        gen_board = solver.generate_board(56, size=9)
        print(gen_board)
