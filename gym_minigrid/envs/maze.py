from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class MazeEnv(MiniGridEnv):
    """
    Dense maze environment
    """

    def __init__(self,
                 size=11,
                 agent_start_pos=(1, 1),
                 agent_start_dir=0,
                 goal_pos=None,
                 max_steps=1000
                 ):
        assert size % 2 == 1, "Size should be odd"
        self._agent_start_pos = agent_start_pos
        self._agent_start_dir = agent_start_dir
        if goal_pos is not None and goal_pos[0] < 0:
            goal_pos = (goal_pos[0] + size, goal_pos[1] + size)  # Negative means from bottom-right corner
        self._goal_pos = goal_pos
        super().__init__(grid_size=size, max_steps=max_steps)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Fill everything with walls
        for i in range(width // 2 + 1):
            self.grid.wall_rect(i, i, width - 2 * i, height - 2 * i)

        visited = np.ones((width, height), dtype=bool)
        visited[1:-1, 1:-1] = False

        def _gen_maze_dfs(x, y):
            self.grid.set(x, y, None)
            visited[x, y] = True
            moves = [(x - 2, y), (x + 2, y), (x, y - 2), (x, y + 2)]
            moves = self.np_random.permutation(moves)
            for tx, ty in moves:
                if 0 <= tx < width and 0 <= ty < height and not visited[tx, ty]:
                    self.grid.set((x + tx) // 2, (y + ty) // 2, None)  # set intermediate "corridor"
                    _gen_maze_dfs(tx, ty)

        _gen_maze_dfs(*self._agent_start_pos)
        self.agent_pos = self._agent_start_pos
        self.agent_dir = self._agent_start_dir

        if self._goal_pos:
            self.put_obj(Goal(), *self._goal_pos)
        else:
            self.place_obj(Goal())

        self.mission = "get to the green goal square"


register(id='MiniGrid-Maze-v0', entry_point='gym_minigrid.envs:MazeEnv')
register(id='MiniGrid-MazeS7-v0', entry_point='gym_minigrid.envs:MazeEnv', kwargs=dict(size=7))
register(id='MiniGrid-MazeS11-v0', entry_point='gym_minigrid.envs:MazeEnv', kwargs=dict(size=11))
register(id='MiniGrid-MazeS15-v0', entry_point='gym_minigrid.envs:MazeEnv', kwargs=dict(size=15))
register(id='MiniGrid-MazeS19-v0', entry_point='gym_minigrid.envs:MazeEnv', kwargs=dict(size=19))
