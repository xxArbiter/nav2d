import random
import numpy as np
from typing import List, Tuple

from ..elements import Point, Vector
from ..regions import NoEntryRegion, SlipperyRegion, BlackHoleRegion
from ..map import CellMap
from .polygon import generate_polygon


def select(grid_map: np.ndarray, rng: np.random.Generator):
    zero_idxes = np.argwhere(grid_map == 0)
    if zero_idxes.size == 0:
        raise ValueError("No zero index found")

    i, j = zero_idxes[rng.choice(zero_idxes.shape[0])]
    grid_map[i, j] = 1
    return [i, j]


def generate_map(
    width: int,
    height: int,
    init_cells: List[Tuple[int, int]],
    goal_cells: List[Tuple[int, int]],
    num_blocks: int,
    num_dynamics: int,
    v_max: float,
    seed: int,
    bounce: str | None = None,
) -> CellMap:
    """
    Generate a cell map given parameters

    Args:
        width (int): refer to CellMap
        height (int): refer to CellMap
        init_cells (List[Tuple[int, int]]): refer to CellMap
        goal_cells (List[Tuple[int, int]]): refer to CellMap
        num_blocks (int): number of cells that will be placed with block regions
        num_dynamics (int): number of cells that will be placed with dynamic regions
        v_max (float): maximum allowed movement in the map
            * The forces in dynamic regions will be scaled by this value
        seed (int): RNG seed
        bounce (str | None): refer to NoEntryRegion. None,  'reflection' or 'back'.

    Returns:
        CellMap: the returned VALID map
    """
    random.seed(seed)
    rng = np.random.default_rng(seed)

    while True:
        grid_map = np.zeros((width, height))
        
        block = []
        dynamics = []
        for _ in range(num_dynamics):
            x, y = select(grid_map, rng)
            polygon = generate_polygon(Point(x, y), 1, .3, .3, 6)
            polygon.scale([Point(x, y), Point(x + 1, y + 1)])
            region_cls = random.choice([SlipperyRegion, BlackHoleRegion])
            force = Vector(*rng.uniform(-1, 1, 2))
            force *= 0.6 * v_max / force.length
            force = force.length if region_cls == BlackHoleRegion else force
            center = Point(x + 0.5, y + 0.5)
            dynamics.append(region_cls(zone=polygon, center=center, force=force))

        for cell in init_cells + goal_cells:
            grid_map[cell[0], cell[1]] = 1

        for _ in range(num_blocks):
            x, y = select(grid_map, rng)
            polygon = generate_polygon(Point(x, y), 1, .3, .3, 6)
            polygon.scale([Point(x, y), Point(x + 1, y + 1)])
            region = NoEntryRegion(polygon, bounce=bounce)
            block.append(region)
    
        map = CellMap.make_map(width, height, init_cells, goal_cells, block, dynamics)
        if map is not None:
            print(map)
            # Reset to random seeds
            # random.seed()
            # np.random.seed(None)
            return map
        else:
            print("Invalid map, retrying...")
