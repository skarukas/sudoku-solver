# Definitions:
# tile:
#  A space on the Sudoku board.
# subgrid:
#  The minigrid within the sudoku grid (standard is 3x3).
# subgrid size:
#  The width and height of a subgrid (standard = 3).
# num subgrids:
#  The number of subgrids within each dimension of the board (standard = 3).
# num digits:
#  Digits are numbered from 1 to num_digits+1. This is also the width and height of the board.
# sequence:
#  Any set of tiles that must contain all numbers (a sequence is a row, column, or subgrid).

from typing import Iterator, TypeAlias
import math
import utils

# Either a digit 1-9 or 0/None which means blank.
Num: TypeAlias = int | None
Coord: TypeAlias = tuple[int, int]
CoordIterator: TypeAlias = Iterator[Coord]


class SudokuIndexer:
    """Utility methods for iterating over a Sudoku board."""
    def __init__(self, num_subgrids: int, subgrid_size: int):
        self.num_subgrids = num_subgrids
        self.subgrid_size = subgrid_size
        self.num_digits = self.num_subgrids * self.subgrid_size

    def _idx_to_subgrid_idx(self, i: int) -> int:
        """Returns the index of the subgrid the true index is found in."""
        return i // self.subgrid_size

    def _subgrid_idx_to_idx(self, i: int) -> int:
        """Returns the min true index for given the subgrid index."""
        return i * self.subgrid_size

    def get_containing_subgrid(self, i, j) -> CoordIterator:
        """Returns an iterator over the subgrid containing (i, j)."""
        return self._get_subgrid_by_subgrid_idx(
            self._idx_to_subgrid_idx(i),
            self._idx_to_subgrid_idx(j)
        )

    def _get_subgrid_by_subgrid_idx(self, sub_i, sub_j) -> CoordIterator:
        """Returns all indices within the subgrid index (sub_i, sub_j). Note the inputs are *subgrid* indices, not original indices."""
        # Bottom of subgrid -> top of subgrid.
        grid_i = range(self._subgrid_idx_to_idx(sub_i),
                       self._subgrid_idx_to_idx(sub_i+1))
        grid_j = range(self._subgrid_idx_to_idx(sub_j),
                       self._subgrid_idx_to_idx(sub_j+1))

        for k in grid_i:
            for l in grid_j:
                yield k, l

    def get_containing_row(self, i, _j) -> CoordIterator:
        """Returns an iterator over the row containing (i, j)."""
        for jj in range(self.num_digits):
            yield i, jj

    def get_containing_column(self, _i, j) -> CoordIterator:
        """Returns an iterator over the column containing (i, j)."""
        for ii in range(self.num_digits):
            yield ii, j

    def get_all_containing_sequences(self, i, j) -> CoordIterator:
        """Returns all indices that are in the same row / column / subgrid."""
        its = [
            self.get_containing_column(i, j),
            self.get_containing_row(i, j),
            self.get_containing_subgrid(i, j)
        ]
        for it in its:
            for tup in it:
                yield tup

    def get_all_sequences(self) -> list[CoordIterator]:
        """Returns all rows, columns, and subgrids, each as a CoordIterator."""
        rows = [
            self.get_containing_row(i, i)
            for i in range(self.num_digits)
        ]
        columns = [
            self.get_containing_column(j, j)
            for j in range(self.num_digits)
        ]
        subgrids = [
            self._get_subgrid_by_subgrid_idx(sub_i, sub_j)
            for sub_i in range(self.num_subgrids)
            for sub_j in range(self.num_subgrids)
        ]
        return [*rows, *columns, *subgrids]

    def all_coords(self) -> CoordIterator:
        """Get all coords in the grid."""
        for i in range(self.num_digits):
            for j in range(self.num_digits):
                yield i, j


class Board(SudokuIndexer):
    def __init__(self, grid: list[list[Num]]):
        self.grid = grid
        self.num_digits, self.num_subgrids, self.subgrid_size = self._get_game_size()
        self.all_digits = set(range(1, self.num_digits+1))
        super().__init__(self.num_subgrids, self.subgrid_size)
    
    @classmethod
    def from_string(cls, s: str) -> 'Board':
        """Builds a board from a string where numbers are separated by spaces."""
        def _decode(c: str) -> int:
            c = c.strip()
            return 0 if c in ".-_" else int(c)

        grid = [
            [_decode(c) for c in line.strip().split(" ")] 
            for line in s.split("\n") if line.strip()
        ]
        return cls(grid)

    def _get_game_size(self) -> tuple[int, int, int]:
        """Return (num_digits, num_subgrids, subgrid_size). Assume num_subgrids == subgrid_size."""
        # num_digits must be a square.
        num_digits = len(self.grid)
        num_subgrids = int(math.sqrt(num_digits))
        assert num_subgrids**2 == num_digits, f"{num_subgrids=}**2 should be {num_digits=}"
        for row in self.grid:
            assert len(row) == num_digits
        # Assume num_subgrids == subgrid_size but may not be the case
        # (e.g. 12x12 grid with 4x4 subgrids)
        return num_digits, num_subgrids, num_subgrids

    def all_tiles(self) -> Iterator[tuple[Coord, int]]:
        """Return an enumerated iterator over all tiles."""
        for i, j in self.all_coords():
            yield (i, j), self.grid[i][j]
    
    def is_solved(self) -> bool:
        """Check whether the board is solved by checking all sequences to make sure they have all digits."""
        for seq in self.get_all_sequences():
            if set(self.grid[i][j] for i, j in seq) != self.all_digits:
                return False
        return True
    
    def _should_add_lines(self, i) -> bool:
        return (i + 1) % self.num_subgrids == 0 and i != self.num_digits - 1

    def to_string(self, include_separators: bool = True) -> str:
        lines = []
        for i, row in enumerate(self.grid):
          line = []
          for j, x in enumerate(row):
              line.append(str(x) if x else "_")
              if include_separators and self._should_add_lines(j):
                  line.append("|")
          lines.append(" ".join(line))
          if include_separators and self._should_add_lines(i):
              num_dashes = 2 * (self.num_digits + self.num_subgrids - 1) - 1
              lines.append("-" * num_dashes)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.to_string(include_separators=True)
        
    def is_valid(self) -> bool:
        for sequence in self.get_all_sequences():
            seen_digits = set()
            for i, j in sequence:
                digit = self.grid[i][j]
                if digit and digit in seen_digits:
                    return False
                seen_digits.add(digit)
        
        return True