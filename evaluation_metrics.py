import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass, asdict
from queue import PriorityQueue
from environment import State, Action, ACTIONS, GRID_SIZE

@dataclass
class EvaluationMetrics:
    exp_id: str
    agent_type: str
    episodes_run: int
    goal_reach_rate: float
    average_final_reward: float
    average_final_path_length: float
    average_collisions: float
    collisions_per_episode: float
    revisit_rate: float
    regret_vs_optimal: float
    cumul_regret_at_100: float
    optimality_ratio: Optional[float]
    convergence_episode: Optional[int]
    sample_complexity: Optional[int]
    acceleration_score: float
    optimal_path_length: Optional[int] = None

def _manhattan_distance(a: State, b: State) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)

def _get_neighbors(state: State, obstacles: Set[State]) -> List[State]:
    neighbors = []
    for action in ACTIONS:
        next_state = State(state.x + action.dx, state.y + action.dy)
        if (0 <= next_state.x < GRID_SIZE and 
            0 <= next_state.y < GRID_SIZE and 
            next_state not in obstacles):
            neighbors.append(next_state)
    return neighbors

def _find_optimal_length_astar(start: State, goal: State, obstacles: Set[State]) -> int:
    """Find the optimal path length using A* search."""
    frontier = PriorityQueue()
    # Use a counter to ensure unique ordering when priorities are equal
    counter = 0
    frontier.put((0, counter, start))
    counter += 1
    
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while not frontier.empty():
        current = frontier.get()[2]  # Get the state from the tuple
        
        if current == goal:
            break
            
        for next_state in _get_neighbors(current, obstacles):
            new_cost = cost_so_far[current] + 1
            if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                cost_so_far[next_state] = new_cost
                priority = new_cost + _manhattan_distance(next_state, goal)
                frontier.put((priority, counter, next_state))
                counter += 1
                came_from[next_state] = current
    
    # Reconstruct path length
    current = goal
    path_length = 0
    while current != start:
        path_length += 1
        current = came_from[current]
    return path_length

def _find_optimal_length(exp_dir: Path) -> int:
    # First try to read from CSV file
    for candidate in [exp_dir / "optimal_path_table.csv",
                     exp_dir / "p2" / "optimal_path_table.csv"]:
        if candidate.exists():
            df = pd.read_csv(candidate)
            return int(df["opt_length"].iloc[0])
    
    # If CSV not found, try to read from config
    try:
        # Look for config in the same relative path as the original config file
        config_path = Path("config") / exp_dir.name.split("-")[0] / f"{exp_dir.name}.yml"
        if not config_path.exists():
            # Fallback to old behavior for backward compatibility
            config_path = Path("config") / f"{exp_dir.name}.yml"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, "r") as f:
            import yaml
            cfg = yaml.safe_load(f)
            start = State(*cfg["agent_start"])
            goal = State(*cfg["agent_goal"])
            obstacles = {State(*xy) for xy in cfg["obstacles"]}
            return _find_optimal_length_astar(start, goal, obstacles)
    except Exception as e:
        raise FileNotFoundError(f"Could not find optimal path length in {exp_dir}: {e}")

def calculate_metrics(exp_id: str, agent_type: str, results_dir: Path) -> EvaluationMetrics:
    """Calculate evaluation metrics from experiment logs."""
    base = results_dir / agent_type.lower()
    action_log = pd.read_csv(base / "agent_action_log.csv")
    state_visits = pd.read_csv(base / "state_visitation_log.csv")

    # episodes and goal‐reach
    episodes_run    = int(action_log["episode"].max())
    goal_reach_rate = float(
        action_log.groupby("episode")["goal_reached"].max().mean()
    )

    # per‐episode stats
    ep_stats = action_log.groupby("episode").agg({
        "timestep":     "max",  # path length
        "is_collision": "sum",
        "goal_reached": "max"
    }).reset_index()

    # only successful episodes count for final‐path
    succ_eps = ep_stats[ep_stats["goal_reached"] == 1]
    avg_final_len    = float(succ_eps["timestep"].mean())
    avg_final_reward = - avg_final_len
    avg_collisions   = float(ep_stats["is_collision"].mean())

    # collisions per episode and revisit rate
    collisions_per_ep = avg_collisions
    # revisit_rate = fraction of (ep,x,y) visited >1
    vc = state_visits.groupby(["episode","x","y"])["visits"].sum()
    revisit_rate = float((vc>1).sum() / vc.size)

    # load true optimal
    opt_len = _find_optimal_length(results_dir)
    regret_vs_optimal  = avg_final_len - opt_len
    cumul_regret_100   = regret_vs_optimal * 100.0

    # optimality_ratio for P1_CM & P2
    opt_ratio = None
    if agent_type in ("P1_CM","P2"):
        opt_ratio = avg_final_len / opt_len

    # (optional) convergence + complexity + acceleration
    convergence_episode = None
    sample_complexity   = None
    acceleration_score  = 0.0
    if agent_type == "P2":
        # your existing logic (only meaningful for P2)
        window = 20
        rg = action_log.groupby("episode")["goal_reached"].max().rolling(window).mean()
        for ep in range(window, episodes_run+1):
            if rg.get(ep,0) >= 0.95:
                convergence_episode = ep - window + 1
                break
        if convergence_episode:
            sample_complexity = int(
                action_log[action_log["episode"] <= convergence_episode]["timestep"].sum()
            )
        lc = action_log.groupby("episode")["goal_reached"].mean()
        sd = lc.diff().diff()
        acceleration_score = float(sd.max())

    return EvaluationMetrics(
        exp_id=exp_id,
        agent_type=agent_type,
        episodes_run=episodes_run,
        goal_reach_rate=goal_reach_rate,
        average_final_reward=avg_final_reward,
        average_final_path_length=avg_final_len,
        average_collisions=avg_collisions,
        collisions_per_episode=collisions_per_ep,
        revisit_rate=revisit_rate,
        regret_vs_optimal=regret_vs_optimal,
        cumul_regret_at_100=cumul_regret_100,
        optimality_ratio=opt_ratio,
        optimal_path_length=opt_len,
        convergence_episode=convergence_episode,
        sample_complexity=sample_complexity,
        acceleration_score=acceleration_score
    )

def save_metrics(metrics: EvaluationMetrics, output_path: Path):
    d = asdict(metrics)
    # convert numpy types
    for k,v in d.items():
        if isinstance(v,(np.integer,)): d[k]=int(v)
        if isinstance(v,(np.floating,)): d[k]=float(v)
    output_path.write_text(json.dumps(d, indent=2))

# Example process function
def process_experiment(exp_id: str, results_dir: Path):
    for agent in ["R2","P1_CM","P2"]:
        try:
            m = calculate_metrics(exp_id, agent, results_dir)
            save_metrics(m, results_dir / f"evaluation_metrics_{agent}.json")
        except Exception as e:
            print(f"  ✗ {agent}: {e}")

if __name__=="__main__":
    # assume you call this for each experiment folder
    root = Path("results")
    for exp in root.iterdir():
        if not exp.is_dir(): continue
        print("Processing", exp.name)
        process_experiment(exp.name, exp) 