from typing import List, Tuple, Dict
from dataclasses import dataclass

# Grid size constant
GRID_SIZE = 10

@dataclass
class State:
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

@dataclass
class Action:
    dx: int
    dy: int
    name: str

# Define actions
left = Action(0, -1, "left")
down = Action(1, 0, "down")
right = Action(0, 1, "right")
up = Action(-1, 0, "up")

ACTIONS: List[Action] = [up, right, down, left]
ACTION_INDEX: Dict[str, int] = {a.name: i for i, a in enumerate(ACTIONS)}

# Lookup table for action name from dx and dy
DX_DY_TO_NAME = {(a.dx, a.dy): a.name for a in ACTIONS}

class GridWorldEnv:
    """Minimal deterministic grid world with static obstacles."""
    def __init__(self, start: Tuple[int, int], goal: Tuple[int, int], obstacles: List[Tuple[int, int]]):
        self.start = State(*start)
        self.goal = State(*goal)
        self.obstacles = {State(*xy) for xy in obstacles}
        self.state = self.start
        self.timestep = 0

    def reset(self):
        self.state = self.start
        self.timestep = 0
        return self.state

    def step(self, action: Action):
        self.timestep += 1
        next_state = State(self.state.x + action.dx, self.state.y + action.dy)

        off_grid = not (0 <= next_state.x < GRID_SIZE and 0 <= next_state.y < GRID_SIZE)
        blocked = next_state in self.obstacles
        is_collision = blocked          # off‑grid bumps are *not* counted as collision

        # Invalid moves → stay in place
        if off_grid or blocked:
            next_state = self.state
        self.state = next_state

        goal_reached = (next_state == self.goal)
        reward = 0 if goal_reached else -1
        done = goal_reached  # Only terminate when goal is reached
        return next_state, reward, done, {
            "is_collision": is_collision,
            "goal_reached": goal_reached,
        }

    def would_collide(self, state: State, action: Action) -> bool:
        """True if executing `action` from `state` would end in an obstacle or off-grid."""
        nx, ny = state.x + action.dx, state.y + action.dy
        off_grid = not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE)
        blocked  = State(nx, ny) in self.obstacles
        return off_grid or blocked 