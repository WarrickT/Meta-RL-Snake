import numpy as np
import random
import matplotlib.pyplot as plt


class SnakeEnv:

    def __init__(self, grid_size=10, variant="classic"):
        self.grid_size = grid_size
        self.variant = variant
        self.reset()
        plt.ion()

    def place_apple(self):
        empty_cells = [
            (y, x)
            for y in range(self.grid_size)
            for x in range(self.grid_size)
            if self.grid[y][x] == 0
        ]
        self.apple = random.choice(empty_cells)
        self.grid[self.apple[0]][self.apple[1]] = 2

    def reset(self):
        self.obstacles = []
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        mid = self.grid_size // 2

        # Setting the center
        self.snake = [(mid, mid), (mid, mid - 1), (mid, mid - 2)]

        # Drawing the snake
        for y, x in self.snake:
            self.grid[y][x] = 1

        self.direction = (0, 1)
        self.done = False

        if self.variant == "obstacles":
            obstacle_count = 5
            empty = [
                (y, x)
                for y in range(self.grid_size)
                for x in range(self.grid_size)
                if self.grid[y][x] == 0
            ]

            for _ in range(obstacle_count):
                oy, ox = random.choice(empty)
                self.grid[oy][ox] = -1
                self.obstacles.append((oy, ox))
                empty.remove((oy, ox))

        self.place_apple()
        return self.grid.copy()

    def step(self, action):
        # Interpret action as DIRECTION CHANGE
        # Move snakes head
        # Check collision
        # Check if apple eaten
        # Update the grid

        # This handles the moving apple variant
        if not hasattr(self, "step_count"):
            self.step_count = 0
        self.step_count += 1

        # Once every 5
        if self.variant == "moving_apple" and self.step_count % 5 == 0:
            self.grid[self.apple[0]][self.apple[1]] = 0
            self.place_apple()

        ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

        new_direction = ACTIONS[action]
        current_direction = self.direction

        if (
            new_direction[0] == -current_direction[0]
            and new_direction[1] == -current_direction[1]
        ):
            new_direction = current_direction

        self.direction = new_direction
        dy, dx = self.direction

        head_y, head_x = self.snake[0]
        new_head = (head_y + dy, head_x + dx)

        # There is a collision
        y, x = new_head
        if (
            y < 0
            or y >= self.grid_size
            or x < 0
            or x >= self.grid_size
            or self.grid[y][x] == 1  # It bit itself!
            or self.grid[y][x] == -1  # It ran into an obstacle
        ):
            self.done = True
            return self.grid.copy(), -1, True, {}

        self.snake.insert(0, new_head)

        if new_head == self.apple:
            reward = 1
            self.place_apple()  # Place a new one

        else:
            reward = 0
            tail = self.snake.pop()
            self.grid[tail[0]][tail[1]] = 0

        self.grid[y][x] = 1

        return self.grid.copy(), reward, False, {}

    def render(self):
        plt.imshow(self.grid, cmap="gray_r", vmin=-1, vmax=2)
        plt.title("RL Snake")
        plt.axis("off")
        plt.pause(0.1)
        plt.clf()  # Clearing the frame
