import numpy as np

class Ant:
    def __init__(self, x, y, color, rule, step_size=1):
        self.x = x
        self.y = y
        self.color = color  # (R, G, B)
        self.rule = rule
        self.num_states = len(rule)
        self.direction = 0  # 0: right, 1: up, 2: left, 3: down
        self.directions = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # right, up, left, down
        self.step_size = step_size
        self.ant_index = None  # Will be set by LangtonAntSimulation

    def move(self, grid):
        for _ in range(self.step_size):
            cell_state, cell_ant = grid[self.x, self.y]
            state_index = cell_state % self.num_states
            instruction = self.rule[state_index]
            if instruction == 'L':
                self.direction = (self.direction - 1) % 4
            else:
                self.direction = (self.direction + 1) % 4
            next_state = (cell_state + 1) % self.num_states
            grid[self.x, self.y] = (next_state, self.ant_index)
            dx, dy = self.directions[self.direction]
            self.x = (self.x + dx) % grid.shape[0]
            self.y = (self.y + dy) % grid.shape[1]

class LangtonAntSimulation:
    def __init__(self, width, height, cell_size, ant_configs):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid_width = width // cell_size
        self.grid_height = height // cell_size
        # Initialize grid with tuples of (state, ant_index)
        self.grid = np.zeros((self.grid_width, self.grid_height), dtype=[('state', 'i4'), ('ant', 'i4')])
        self.ants = []
        for i, cfg in enumerate(ant_configs):
            ant = Ant(cfg['x'], cfg['y'], cfg['color'], cfg['rule'], cfg.get('step_size', 1))
            ant.ant_index = i + 1  # 1-based index to distinguish from unvisited cells (0)
            self.ants.append(ant)
        self.iteration = 0

    def step(self):
        for ant in self.ants:
            ant.move(self.grid)
        self.iteration += 1

    def get_grid(self):
        return self.grid

    def get_ants(self):
        return self.ants

    def get_iteration(self):
        return self.iteration

    def reset_iteration_count(self):
        """Reset the iteration count to 0 while preserving the current state"""
        self.iteration = 0

    def reset(self):
        self.grid = np.zeros((self.grid_width, self.grid_height), dtype=[('state', 'i4'), ('ant', 'i4')])
        for ant in self.ants:
            ant.x = self.grid_width // 2
            ant.y = self.grid_height // 2
            ant.direction = 0
        self.iteration = 0 