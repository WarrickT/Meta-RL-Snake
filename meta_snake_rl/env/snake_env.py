import numpy as np
import random
import matplotlib.pyplot as plt
from meta_snake_rl.utils.constants import DIRECTIONS, LEFT_TURN, RIGHT_TURN, ACTIONS, OPPOSITE


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


def get_state(snake, apple, direction, grid_size):
    # Define relative directions 
    # These numbers could be very problematic once we work with reward systems
   
    # There are three possible positions after this
    head_y, head_x = snake[0]
    forward = (head_y + direction[0], head_x + direction[1])
    left = (head_y + LEFT_TURN[direction][0], head_x + LEFT_TURN[direction][1])
    right = (head_y + RIGHT_TURN[direction][0], head_x + RIGHT_TURN[direction][1])

    danger_front = int(is_danger(forward, snake, grid_size))
    danger_left = int(is_danger(left, snake, grid_size))
    danger_right = int(is_danger(right, snake, grid_size))

    direction_index = DIRECTIONS[direction]

    apple_up_down = (
        1 if apple[0] > head_y else
        -1 if apple[0] < head_y else 
        0
    )
    apple_left_right = (
        1 if apple[1] > head_x else
        -1 if apple[1] < head_x else 
        0
    )

    return (danger_front, danger_left, danger_right, direction_index, apple_up_down, apple_left_right)


def is_danger(pos, snake, grid_size):
    y, x = pos
    return ((pos in snake) or (y < 0 or y >= grid_size) or (x < 0 or x >= grid_size))

    
def select_action(Q, state, epsilon, direction):
    if state not in Q:
        Q[state] = {a: 0 for a in ACTIONS}

    illegal_action = OPPOSITE[direction]
    valid_actions = [a for a in ACTIONS if a != illegal_action]

    # Initialize epsilon greedy decision
    r = random.random()

    if r < epsilon: 
        action = random.choice(valid_actions)
    else:
        max_Q_val = max(Q[state][a] for a in valid_actions)
        best_actions = [a for a in valid_actions if Q[state][a] == max_Q_val]
        action = random.choice(best_actions)

    return action




    