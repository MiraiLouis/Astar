from __future__ import annotations

import heapq
import multiprocessing as mp
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from modele_astar import modele

Position = Tuple[int, int]
Maze = List[List[int]]
Path = List[Position]

MOVES_8: Tuple[Position, ...] = (
    (0, -1),
    (0, 1),
    (-1, 0),
    (1, 0),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
)


@dataclass
class SolveResult:
    path: Optional[Path]
    explored: Optional[int]


@dataclass
class TimedRun:
    result: SolveResult
    avg_ms: Optional[float]
    timed_out: bool


def in_bounds(maze: Maze, pos: Position) -> bool:
    rows = len(maze)
    cols = len(maze[0]) if rows else 0
    return 0 <= pos[0] < rows and 0 <= pos[1] < cols


def passable(maze: Maze, pos: Position) -> bool:
    return maze[pos[0]][pos[1]] == 0


def neighbors(maze: Maze, pos: Position) -> List[Position]:
    out: List[Position] = []
    for dx, dy in MOVES_8:
        nxt = (pos[0] + dx, pos[1] + dy)
        if in_bounds(maze, nxt) and passable(maze, nxt):
            out.append(nxt)
    return out


def reconstruct(parent: Dict[Position, Position], end: Position) -> Path:
    path = [end]
    cur = end
    while cur in parent:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def chebyshev(a: Position, b: Position) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def bfs_solver(maze: Maze, start: Position, end: Position) -> SolveResult:
    if not passable(maze, start) or not passable(maze, end):
        return SolveResult(path=None, explored=0)

    q = deque([start])
    seen = {start}
    parent: Dict[Position, Position] = {}
    explored = 0

    while q:
        cur = q.popleft()
        explored += 1
        if cur == end:
            return SolveResult(path=reconstruct(parent, end), explored=explored)
        for nxt in neighbors(maze, cur):
            if nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = cur
            q.append(nxt)

    return SolveResult(path=None, explored=explored)


def dijkstra_solver(maze: Maze, start: Position, end: Position) -> SolveResult:
    if not passable(maze, start) or not passable(maze, end):
        return SolveResult(path=None, explored=0)

    pq: List[Tuple[int, Position]] = [(0, start)]
    dist: Dict[Position, int] = {start: 0}
    parent: Dict[Position, Position] = {}
    closed = set()
    explored = 0

    while pq:
        cost, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        closed.add(cur)
        explored += 1
        if cur == end:
            return SolveResult(path=reconstruct(parent, end), explored=explored)
        for nxt in neighbors(maze, cur):
            new_cost = cost + 1
            if new_cost < dist.get(nxt, 10**12):
                dist[nxt] = new_cost
                parent[nxt] = cur
                heapq.heappush(pq, (new_cost, nxt))

    return SolveResult(path=None, explored=explored)


def greedy_best_first_solver(maze: Maze, start: Position, end: Position) -> SolveResult:
    if not passable(maze, start) or not passable(maze, end):
        return SolveResult(path=None, explored=0)

    pq: List[Tuple[int, Position]] = [(chebyshev(start, end), start)]
    seen = {start}
    parent: Dict[Position, Position] = {}
    explored = 0

    while pq:
        _, cur = heapq.heappop(pq)
        explored += 1
        if cur == end:
            return SolveResult(path=reconstruct(parent, end), explored=explored)
        for nxt in neighbors(maze, cur):
            if nxt in seen:
                continue
            seen.add(nxt)
            parent[nxt] = cur
            heapq.heappush(pq, (chebyshev(nxt, end), nxt))

    return SolveResult(path=None, explored=explored)


def astar_solver(maze: Maze, start: Position, end: Position) -> SolveResult:
    if not passable(maze, start) or not passable(maze, end):
        return SolveResult(path=None, explored=0)

    pq: List[Tuple[int, int, Position]] = [(chebyshev(start, end), 0, start)]
    g_score: Dict[Position, int] = {start: 0}
    parent: Dict[Position, Position] = {}
    closed = set()
    explored = 0

    while pq:
        _, g_cur, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        closed.add(cur)
        explored += 1

        if cur == end:
            return SolveResult(path=reconstruct(parent, end), explored=explored)

        for nxt in neighbors(maze, cur):
            tentative_g = g_cur + 1
            if tentative_g < g_score.get(nxt, 10**12):
                g_score[nxt] = tentative_g
                parent[nxt] = cur
                f = tentative_g + chebyshev(nxt, end)
                heapq.heappush(pq, (f, tentative_g, nxt))

    return SolveResult(path=None, explored=explored)


def your_model_solver(maze: Maze, start: Position, end: Position) -> SolveResult:
    try:
        path = modele([row[:] for row in maze], start, end)
    except Exception:
        return SolveResult(path=None, explored=None)

    if isinstance(path, str) or path is None:
        return SolveResult(path=None, explored=None)
    return SolveResult(path=path, explored=None)


def path_len(path: Optional[Path]) -> Optional[int]:
    if not path:
        return None
    return len(path) - 1


def run_solver(
    solver: Callable[[Maze, Position, Position], SolveResult],
    maze: Maze,
    start: Position,
    end: Position,
    repeats: int = 10,
    timeout_s: float = 2.0,
) -> TimedRun:
    q: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_run_solver_worker,
        args=(q, solver, maze, start, end, repeats),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return TimedRun(result=SolveResult(path=None, explored=None), avg_ms=None, timed_out=True)

    if q.empty():
        return TimedRun(result=SolveResult(path=None, explored=None), avg_ms=None, timed_out=True)

    result, avg_ms = q.get()
    return TimedRun(result=result, avg_ms=avg_ms, timed_out=False)


def _run_solver_worker(
    q: mp.Queue,
    solver: Callable[[Maze, Position, Position], SolveResult],
    maze: Maze,
    start: Position,
    end: Position,
    repeats: int,
) -> None:
    result = solver([row[:] for row in maze], start, end)
    t0 = time.perf_counter()
    for _ in range(repeats):
        solver([row[:] for row in maze], start, end)
    t1 = time.perf_counter()
    avg_ms = ((t1 - t0) / repeats) * 1000.0
    q.put((result, avg_ms))


def benchmark_mazes() -> List[Tuple[str, Maze, Position, Position]]:
    return [
        (
            "maze_small_open",
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0],
            ],
            (0, 0),
            (4, 4),
        ),
        (
            "maze_small_dense",
            [
                [0, 1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 1],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 0],
            ],
            (0, 0),
            (5, 5),
        ),
        (
            "maze_medium_corridor",
            [
                [0, 0, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 1, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, 0, 1],
                [1, 1, 1, 0, 1, 0, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 1, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
            ],
            (0, 0),
            (7, 7),
        ),
        (
            "maze_medium_no_path",
            [
                [0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
            ],
            (0, 0),
            (5, 5),
        ),
        (
            "maze_large",
            [
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
                [1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            ],
            (0, 0),
            (9, 9),
        ),
    ]


def format_value(v: Optional[int]) -> str:
    return "-" if v is None else str(v)


def main() -> None:
    solvers: Sequence[Tuple[str, Callable[[Maze, Position, Position], SolveResult]]] = (
        ("YourModel", your_model_solver),
        ("BFS", bfs_solver),
        ("Dijkstra", dijkstra_solver),
        ("GreedyBestFirst", greedy_best_first_solver),
        ("AStar", astar_solver),
    )

    print("Benchmark: Your model vs BFS, Dijkstra, Greedy Best-First, A*")
    print("Moves: 8-directional, unit step cost")
    print()

    header = f"{'Maze':<22} {'Solver':<16} {'Found':<6} {'PathLen':<8} {'Explored':<9} {'AvgMs':<10}"
    print(header)
    print("-" * len(header))

    for maze_name, maze, start, end in benchmark_mazes():
        for solver_name, solver in solvers:
            timed = run_solver(solver, maze, start, end, repeats=20, timeout_s=2.0)
            result = timed.result
            found = "timeout" if timed.timed_out else ("yes" if result.path else "no")
            avg_ms_str = "-" if timed.avg_ms is None else f"{timed.avg_ms:.3f}"
            print(
                f"{maze_name:<22} "
                f"{solver_name:<16} "
                f"{found:<6} "
                f"{format_value(path_len(result.path)):<8} "
                f"{format_value(result.explored):<9} "
                f"{avg_ms_str:<10}"
            )
        print()


if __name__ == "__main__":
    main()
