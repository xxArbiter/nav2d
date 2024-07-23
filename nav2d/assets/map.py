import math
import numpy as np
from typing import List, Tuple
from collections import deque

from .elements import Element, Point, Polygon
from .regions import Region


class CellMap(Element):
    """
    A cell map that partitions the canvas into 1x1 cells, where each cell can be either:
        - free
        - init cell
        - goal cell
        - block cell: contains a NoEntryRegion
        - dynamic cell: contains a SlipperyRegion or BlackHoleRegion
    A region is a polygon bounded by the boundary of the cell.
    
    Call the static method make_map to generate a valid map, that has valid paths
    between any init-goal pairs.
    
    Map axis representation:
    0  ------------>  x

    |   a -- width -- b
    |
    |   |             |
    |   |           height
    v   |             |

    y   c ----------- d

    Args:
        width (int): the width of the map
        height (int): the height of the map
        init_cells (List[Tuple[int, int]]): list of cells for the agent to start from
        goal_cells (List[Tuple[int, int]]): list of cells for candidate goals
        block_regions (List[Region]): list of block regions
        dynamic_regions (List[Region]): list of dynamic regions
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        init_cells: List[Tuple[int, int]],
        goal_cells: List[Tuple[int, int]],
        block_regions: List[Region],
        dynamic_regions: List[Region],
    ) -> None:
        
        self._w = width
        self._h = height
        
        # check if the init and goal cells are within the map
        for cell in init_cells + goal_cells:
            assert 0 <= cell[0] < self._w
            assert 0 <= cell[1] < self._h
        
        # check if the init and goal cells are overlapping
        for init_cell in init_cells:
            assert init_cell not in goal_cells
            
        # check if the regions are within the map and record their cells
        self._block_cells = []
        self._dynamic_cells = []
        for region in block_regions:
            envelope = region.zone.envelope
            (min_x, min_y), (max_x, max_y) = envelope[0].pos, envelope[1].pos
            assert 0 <= min_x < max_x <= self._w
            assert 0 <= min_y < max_y <= self._h
            # a region should not exceed a cell
            assert max_x <= math.floor(min_x) + 1
            assert max_y <= math.floor(min_y) + 1
            self._block_cells.append([math.floor(min_x), math.floor(min_y)])
        for region in dynamic_regions:
            envelope = region.zone.envelope
            (min_x, min_y), (max_x, max_y) = envelope[0].pos, envelope[1].pos
            assert 0 <= min_x < max_x <= self._w
            assert 0 <= min_y < max_y <= self._h
            # a region should not exceed a cell
            assert max_x <= math.floor(min_x) + 1
            assert max_y <= math.floor(min_y) + 1
            self._dynamic_cells.append([math.floor(min_x), math.floor(min_y)])
            
        
        self._init_cells = init_cells
        self._goal_cells = goal_cells
        self._block_regions = block_regions
        self._dynamic_regions = dynamic_regions
        self._regions = block_regions + dynamic_regions
        self._generate_grid_map()
        
    @staticmethod
    def make_map(*args, **kwargs):
        map = CellMap(*args, **kwargs)
        
        # check if there are valid paths between all init-goal pairs that do not intersect with block cells
        for init_cell in map._init_cells:
            for goal_cell in map._goal_cells:
                if not map.is_valid_map(init_cell, goal_cell):
                    return None
                
        return map
                
    def _generate_grid_map(self):
        """
        Generate a grid map, where:
            0: free cell
            1: init cell
            2: goal cell
            3: block cell
            4: dynamic cell
        """
        self.grid_map = np.zeros((self._w, self._h))
        for (x, y) in self._init_cells:
            self.grid_map[x, y] = 1
        for (x, y) in self._goal_cells:
            self.grid_map[x, y] = 2
        for (x, y) in self._block_cells:
            self.grid_map[x, y] = 3
        for (x, y) in self._dynamic_cells:
            self.grid_map[x, y] = 4
        print(self.grid_map)
            
    def __is_valid_move(self, x, y, visited):
    # Check if (x, y) is within the grid bounds and is not blocked and not visited
        return (
            0 <= x < self._w and
            0 <= y < self._h and
            self.grid_map[x, y] != 3 and
            not visited[x, y]
        )
        
    def is_valid_map(self, start, goal):
        # Define the possible movements: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Initialize the visited matrix
        visited = np.zeros((self._w, self._h))
        
        # Queue for BFS
        queue = deque([start])
        visited[start[0], start[1]] = 1
        
        while queue:
            x, y = queue.popleft()
            
            # Explore the neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if self.__is_valid_move(nx, ny, visited):
                    if (nx, ny) == goal:
                        return True
                    queue.append((nx, ny))
                    visited[nx, ny] = 1
        
        return False
    
    def __repr__(self):
        return f"CellMap(width={self._w}, height={self._h}, init_cells={self._init_cells}, goal_cells={self._goal_cells}, block_regions={self._block_regions}, dynamic_regions={self._dynamic_regions})"
    
    def __str__(self):
        return (
            f"Cell map:\n" +
            f"self.grid_map" + "\n" +
            f"Init cells: {self._init_cells}\n" +
            f"Goal cells: {self._goal_cells}\n" +
            f"Block regions:\n" +
            '\n'.join([str(region) for region in self._block_regions]) + "\n" +
            '\n'.join([str(region) for region in self._dynamic_regions])
        )
