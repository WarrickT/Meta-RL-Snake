# Snake movement parameters

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

DIRECTIONS = {
    UP: 0,
    DOWN: 1,
    LEFT: 2,
    RIGHT: 3
}

LEFT_TURN = {
    UP: LEFT, # Up
    DOWN: RIGHT, # Down
    LEFT: DOWN, # Left
    RIGHT: UP # Right
}

RIGHT_TURN = {
        UP: RIGHT, # Up
    DOWN: LEFT, # Down
    LEFT: UP, # Left
    RIGHT: DOWN # Right
}

OPPOSITE = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT
}

# TD Parameters

# EPISLON 
# ALPHA
# GAMMA 