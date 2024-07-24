import numpy as np
from typing import List, Tuple

from nav2d.assets.elements import Point, Vector, Circle, Polygon, Box
from nav2d.assets.regions import NoEntryRegion, BlackHoleRegion, SimpleRewardRegion
from nav2d.assets.generator.map import generate_map
from nav2d.envs.base import Navigation, MultiNavigation


class HomoNav(Navigation):
    def __init__(
            self,
            init_x_normal: tuple = None,
            init_y_normal: tuple = None,
            # screen_scale: int = 600,
            # margin: float = 0.1,
            # dynamic_zones: List[DynamicRegion] = [],
            # reward_zones: List[RewardRegion] = [],
            draw_init: bool = True,
            is_eval: bool = False,
    ):
        reward_zone = SimpleRewardRegion(
            zone=Polygon([Point(3, 7.5), Point(4, 8.5), Point(8.5, 4), Point(7.5, 3)]),
            reward=-100,
        )

        init_x_normal_default = (4.5, 0.5) if is_eval else (1.0, 0.5) 
        init_y_normal_default = (4.5, 0.5) if is_eval else (7.5, 1.25) 
        init_x_normal = init_x_normal_default if init_x_normal is None else init_x_normal
        init_y_normal = init_y_normal_default if init_y_normal is None else init_y_normal
                
        super().__init__(
            init_x_normal=init_x_normal,
            init_y_normal=init_y_normal,
            reward_zones=[reward_zone],
            draw_init=draw_init,
        )

        self.goal = np.array([9., 9.])


class HeteroNav(Navigation):
    def __init__(
            self,
            init_x_normal: tuple = None,
            init_y_normal: tuple = None,
            # screen_scale: int = 600,
            # margin: float = 0.1,
            # dynamic_zones: List[DynamicRegion] = [],
            # reward_zones: List[RewardRegion] = [],
            is_eval: bool = False,
            draw_init: bool = True,
    ):
        dynamic_zone = BlackHoleRegion(
            zone=Polygon([Point(3, 5), Point(3, 8), Point(8, 8), Point(8, 3), Point(5, 3)]),
            # center=Point(5.5, 5.5),
            center=Point(4, 4),
            force=0.5,
        )
        reward_zone = SimpleRewardRegion(
            zone=Box(Point(4, 4), 3, 3),
            reward=-10,
        )

        init_x_normal_default = (4.0, 0.5) if is_eval else (1.0, 0.5) 
        init_y_normal_default = (4.0, 0.5) if is_eval else (7.5, 1.25) 
        init_x_normal = init_x_normal_default if init_x_normal is None else init_x_normal
        init_y_normal = init_y_normal_default if init_y_normal is None else init_y_normal
        
        super().__init__(
            init_x_normal=init_x_normal,
            init_y_normal=init_y_normal,
            dynamic_zones=[dynamic_zone],
            reward_zones=[reward_zone],
            draw_init=draw_init,
        )

        self.goal = np.array([9., 9.])


class WallNav(Navigation):
    def __init__(
            self,
            init_x_normal: tuple = None,
            init_y_normal: tuple = None,
            version: int = 0,       # 0 - No example zones, 1 - Three example zones
            # screen_scale: int = 600,
            # margin: float = 0.1,
            # dynamic_zones: List[DynamicRegion] = [],
            # reward_zones: List[RewardRegion] = [],
            is_eval: bool = False,
            draw_init: bool = True,
    ):
        assert version in (0, 1)
        if version == 0:
            example_zones = []
        elif version == 1:
            example_zones = [
                NoEntryRegion(
                    zone=Polygon([Point(0.5, 0.5), Point(0.8, 0.5), Point(0.8, 0.8), Point(0.5, 0.8)])
                ),  
                NoEntryRegion(
                    zone=Polygon([Point(1.2, 1.0), Point(1.4, 0.8), Point(1.6, 1.0), Point(1.4, 1.2)])
                ),  
                NoEntryRegion(
                    zone=Polygon([Point(0.8, 2.0), Point(0.7, 2.4), Point(0.9, 2.5), Point(1.0, 2.1)])
                ),
            ]
        dynamic_zone = NoEntryRegion(
            zone=Polygon([Point(3, 7.5), Point(4, 8.5), Point(8.5, 4), Point(7.5, 3)])
        )

        init_x_normal_default = (4.5, 0.5) if is_eval else (1.0, 0.5) 
        init_y_normal_default = (4.5, 0.5) if is_eval else (7.5, 1.25) 
        init_x_normal = init_x_normal_default if init_x_normal is None else init_x_normal
        init_y_normal = init_y_normal_default if init_y_normal is None else init_y_normal
        
        super().__init__(
            init_x_normal=init_x_normal,
            init_y_normal=init_y_normal,
            dynamic_zones=[dynamic_zone] + example_zones,
            draw_init=draw_init,
        )

        self.goal = np.array([9., 9.])


class CellMapMultiNav(MultiNavigation):
    def __init__(
        self,
        train_init_cells: List[Tuple[int, int]],
        eval_init_cells: List[Tuple[int, int]],
        goal_cells: List[Tuple[int, int]],
        num_blocks: int,
        num_dynamics: int,
        v_max: float,
        map_seed: int,
        seed: int,
        sparse: bool = True,
        eval: bool = False,
    ):
        init_cells = train_init_cells + eval_init_cells
        self.cell_map = generate_map(
            width=10,
            height=10,
            init_cells=init_cells,
            goal_cells=goal_cells,
            num_blocks=num_blocks,
            num_dynamics=num_dynamics,
            v_max=v_max,
            seed=map_seed,
        )
        
        self.train_init_zones = [Box(Point(*cell), 1, 1, seed=seed+i) for i, cell in enumerate(train_init_cells)]
        self.eval_init_zones = [Box(Point(*cell), 1, 1, seed=seed+i) for i, cell in enumerate(eval_init_cells)]
        goals = [Circle(Point(*cell) + Vector(.5, .5), 0.5) for cell in goal_cells]
        
        super().__init__(
            init_zones=self.train_init_zones + self.eval_init_zones,
            goals=goals,
            dynamic_zones=self.cell_map._block_regions + self.cell_map._dynamic_regions,
            reward_zones=[],
            v_max=v_max,
            sparse=sparse,
        )
        self.eval = eval
        
    def reset(self, eval: bool = False):
        if eval or self.eval:
            self.init_zones = self.eval_init_zones
        else:
            self.init_zones = self.train_init_zones
        return super().reset()

