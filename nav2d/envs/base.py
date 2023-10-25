import gym
import numpy as np
from gym import spaces
import pygame
from pygame.locals import Rect
from typing import Dict, List

from nav2d.assets.elements import Point, Vector, Line, DirectEdge, RelativePos, Polygon, Box
from nav2d.assets.regions import Region, DynamicRegion, SimpleDynamicRegion, PunchRegion, NoEntryRegion, SlipperyRegion, BlackHoleRegion, RewardRegion, SimpleRewardRegion

from . import EPS_GOAL


class Navigation(gym.Env):
    def __init__(
            self,
            init_x_normal: tuple = (1., 0.5),
            init_y_normal: tuple = (5., 1.25),
            screen_scale: int = 600,
            margin: float = 0.1,
            dynamic_zones: List[DynamicRegion] = [],
            reward_zones: List[RewardRegion] = [],
            draw_init: bool = True,
            dim: int = 2,
            **kwargs,
    ):
        #
        self.v_max = 0.5
        self._draw_init = draw_init
        self._dim = dim

        # The map spans (0, 0) to (10, 10)
        self.low_state = np.array([0, 0] + [0] * (self._dim - 2) + [0, 0], dtype=np.float32)
        self.high_state = np.array([10, 10] + [5] * (self._dim - 2) + [10, 10], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(self._dim+2, ),
            dtype=np.float32,
        )

        # The largest step size of the agent in one timestep is capped at v_max
        self.min_actions = np.array(
            [-self.v_max] * self._dim, dtype=np.float32,
        )
        self.max_actions = np.array(
            [self.v_max] * self._dim, dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=self.min_actions,
            high=self.max_actions,
            dtype=np.float32,
        )

        self.goal = np.array([9.0, 9.0])
        self.centers = np.array([5, 5])
        self.init_center = np.array([init_x_normal[0], init_y_normal[0]])

        # Special zones
        self.dynamic_zones = dynamic_zones
        self.reward_zones = reward_zones

        # Initial state distribution
        x_normal, y_normal = np.array(init_x_normal), np.array(init_y_normal)
        self.init_dist = [x_normal, y_normal]

        # Reward setup
        self.time_penalty = 0.5                      # Reward penalty per each timestep
        self.control_penalty = 0.4
        self.goal_dist_penalty = 0.5

        # For rendering
        self.margin = int(screen_scale * margin)
        self.screen_size = [screen_scale + 2 * self.margin, screen_scale + 2 * self.margin]
        self.screen_scale = screen_scale
        self.normalized_state_scale = 1 / (self.high_state[0] - self.low_state[0])
        self.background_color = [255, 255, 255]     # White background
        self.box_color = [128, 128, 128]
        self.init_color = [0, 0, 255]
        self.init_radius = 30                       # The radius of the init region (in pixels)
        self.goal_color = [0, 255, 0]
        self.goal_radius = 15                       # The radius of the goal region (in pixels)
        # self.dynamic_color = [160, 160, 160]        # Set these two in nav_ed.py
        # self.reward_color = [255, 102, 102]
        self.agent_color = [0, 0, 0]                # Black
        self.action_color = [255, 0, 0]
        self.action_mul = 1                         # Action force vector multiplier
        self.movement_color = [0, 0, 0]
        self.movement_mul = 1                       # Movement vector multiplier
        self.agent_radius = 9                       # The radius of the agent represented as a circle (in pixels)
        self.agent_width = 3                        # The width of the line of the circle representing the agent
        self.viewer = None

        self.pos_dim = 2
        self.obs_dim = self.observation_space.low.size
        self.action_dim = self.action_space.low.size

        self.np_random = None

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def __need_resample(self, pos):
        if not all((
            (self.low_state[:self.pos_dim] <= pos).all(),
            (pos <= self.high_state[:self.pos_dim]).all(),
        )):
            return True
        for zone in self.dynamic_zones:
            if isinstance(zone, NoEntryRegion):
                if zone._zone.point_relative_pos(Point(*pos)) == Polygon.IN:
                    return True
        return False

    def reset(self, eval=False):
        if self.np_random is None:
            self.seed(0)

        # Sample initial position
        sampled = False
        while not sampled:
            # Normal initial state distribution
            self.init_pos = np.array([self.np_random.normal(self.init_dist[0][0], self.init_dist[0][1]),
                                      self.np_random.normal(self.init_dist[1][0], self.init_dist[1][1])], dtype=np.float32)
            if not self.__need_resample(self.init_pos):
                sampled = True
            # self.init_pos = np.clip(self.init_pos, self.low_state, self.high_state)
            # sampled = True
        
        # Add auxiliary dimensions randomly sampled from the standard normal
        if self._dim > 2:
            obs_ = self.np_random.random(size=(self._dim-2, )) \
                    * (self.high_state[self._dim-1] - self.low_state[self._dim-1]) \
                        + self.low_state[self._dim-1]
            self.init_pos = np.concatenate([self.init_pos, obs_])
        
        if eval:
            pass

        # For rendering
        self.last_state = None
        self.action = None

        self.state = np.concatenate([self.init_pos, self.goal])
        return self.state.copy()

    def set_state_from_obs(
            self,
            obs: np.ndarray,
    ):
        self.set_state(obs.copy())

    def get_dist_to_goal(self):
        return np.linalg.norm(self.goal_pos - self.agent_pos)

    def step(self, action: np.ndarray):
        self.last_state = self.state.copy()
        self.action = action.copy()[:self.pos_dim]
        # action = np.clip(action, self.min_actions, self.max_actions)      # Handled by ``NormalizedBoxEnv``
        # assert self.action_space.contains(action)

        # print(f"s: {self.state}, ", end='')
        # Handle the agent's movement
        obs = self.state.copy()
        obs[:self.pos_dim], add_rew = self.transition(self.agent_pos.copy(), 
                                                      self.action.copy())
        
        # Handle dimensions with no semantics
        if self._dim > self.pos_dim:
            obs[self.pos_dim: self._dim] += action.copy()[self.pos_dim: self._dim] + \
                                            self.np_random.normal(0, 0.01, size=(self._dim-self.pos_dim, ))
        
        obs = np.clip(obs, self.low_state, self.high_state)
        self.set_state(obs.copy())

        d_goal = self.get_dist_to_goal()                # R(s', a)
        action_cost = np.linalg.norm(action[:self.pos_dim])     # only the first 2 dimensions are considered
        reward = (
            - self.time_penalty
            - self.control_penalty * action_cost
            - self.goal_dist_penalty * d_goal
            + add_rew
        )

        done = False
        if d_goal < EPS_GOAL:
            done = True
        # print(f"a: {action}, s': {obs}, r: {reward}.")
        return obs.copy(), reward, done, {}

    def transition(self, pos, action: np.array):
        """
        From the current agent position (``pos``) and given an action (``action``),
        computes the next position of the agent. By default, a linear dynamics is assumed.
        But, subject to the current position or the predicted next position,
        non-linear dynamics can also be implemented.
        """
        assert pos.size == action.size
        action = action.squeeze()
        p0 = Point(*pos)
        proposition = [Vector(*action)]
        
        # Applies dynamics according to the distance to the contact point.
        if len(self.dynamic_zones) > 0:
            dist_to_contacts = []
            for zone in self.dynamic_zones:
                dist_to_contacts.append(zone.first_contact(p0, proposition)[1])
            _, self.dynamic_zones = zip(*sorted(zip(dist_to_contacts, self.dynamic_zones), key=lambda x: x[0]))
            for zone in self.dynamic_zones:
                proposition = zone.apply_dynamic(p0, proposition)

        # Applies rewards with an arbitrary order.
        add_rew = 0
        for zone in self.reward_zones:
            add_rew += zone.apply_reward(p0, proposition)

        return (p0 + sum(proposition)).pos, add_rew

    @property
    def agent_pos(self):
        return self.state[:self.pos_dim]

    @property
    def goal_pos(self):
        return self.state[-self.pos_dim:]

    def get_dataset(self) -> Dict:
        """Loads the offline dataset of the related environment: Navigation, NavWithWall, or NavUnsafe."""
        import pickle
        import hydra.utils
        from pathlib import Path
        try:
            cwd = hydra.utils.get_original_cwd()
        except ValueError as e:
            cwd = Path.cwd()
        f_pickled = Path(cwd) / f"data/navigation/{self.config}/{self.config}.pkl"

        with open(f_pickled, 'rb') as file:
            dataset = pickle.load(file)
        return dataset

    def render_for_subclass(self):
        pass

    def render(self, mode='human'):
        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode(self.screen_size)
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.viewer.fill(self.background_color)

        # Draw the state space as a box
        rect = Rect(self.margin, self.margin,
                    self.screen_size[0] - 2 * self.margin,
                    self.screen_size[1] - 2 * self.margin)
        pygame.draw.rect(self.viewer, self.box_color, rect)

        # Draw the initial position
        if self._draw_init:
            init_pos = self.to_pixel(self.init_center)
            pygame.draw.circle(self.viewer, self.init_color, init_pos, self.init_radius)

        # Draw the goal position
        goal_pos = self.to_pixel(self.goal)
        pygame.draw.circle(self.viewer, self.goal_color, goal_pos, self.goal_radius)

        # Draw dynamic regions
        for region in self.dynamic_zones:
            region.render(self.viewer, self.to_pixel)

        # Draw reward regions
        for region in self.reward_zones:
            region.render(self.viewer, self.to_pixel)

        # Draw any other registered objects on screen
        self.render_for_subclass()

        # Draw the agent position
        pos = self.to_pixel(self.agent_pos)
        pygame.draw.circle(self.viewer, self.agent_color, pos, self.agent_radius, self.agent_width)

        # Try to draw the previous action
        if self.action is not None:
            start = Point(*self.last_state[:self.pos_dim])
            end = start + Vector(*self.action) * self.action_mul
            pygame.draw.line(
                self.viewer,
                self.action_color,
                self.to_pixel(start.pos),
                self.to_pixel(end.pos)
            )

        # Try to draw the movement
        if self.last_state is not None:
            start = Point(*self.last_state[:self.pos_dim])
            end = start + (Point(*self.state[:self.pos_dim]) - start) * self.movement_mul
            pygame.draw.line(
                self.viewer,
                self.movement_color,
                self.to_pixel(start.pos),
                self.to_pixel(end.pos)
            )

        # Update the screen
        pygame.display.update()

        if mode == 'rgb_array':
            data = pygame.surfarray.array3d(self.viewer)
            data = np.transpose(data, (1, 0, 2))
            return data
        # plt.imshow(np.transpose(arr, (1, 0, 2)), interpolation='nearest', origin='lower')
        # plt.show()

        # Time delay
        self.clock.tick(10)

    def to_pixel(self, x):
        return (self.screen_scale * x * self.normalized_state_scale + self.margin).astype(int).tolist()

    def set_state(self, state):
        self.state[:] = state

    def get_goal_from_obs(self, obs):
        """obs = (x, y, ..., gx, gy)"""
        return obs[..., -2:]

