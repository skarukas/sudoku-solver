from typing import Iterator

def flattened(
    grid: list[list[float]]
) -> Iterator[tuple[tuple[int, int], float]]:
    for i, col in enumerate(grid):
        for j, v in enumerate(col):
            yield (i, j), v
