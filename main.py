from nav2d.envs.envs import CellMapMultiNav
from time import sleep


if __name__ == '__main__':
    env = CellMapMultiNav(
        train_init_cells=[(0, 1), (1, 0), (1, 1)],
        eval_init_cells=[(0, 0)],
        goal_cells=[(9, 0), (9, 9), (0, 9)],
        num_blocks=5,
        num_dynamics=5,
        v_max=.1,
        map_seed=123,
    )
    env.reset()
    env.render()

    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        sleep(.1)
        if done:
            env.reset()
            env.render()
            sleep(1)
