#!/usr/bin/env python3
"""
main.py – single‑file driver for every phase of the Causal‑Based Knowledge‑Transfer
experiment suite.

Implemented phases
----------------
train_p1     – pre‑train without obstacles (baseline)
train_p2     – full Q‑learning retraining **with** the obstacle mask
train_r2     – random‑exploration baseline (no learning)
discover_ra  – offline discovery & logging of recovery‑actions
build_cm     – build causal model from recovery actions
cm_validate  – validate causal model assumptions and performance
cm_infer     – inference using causal model for recovery actions
transfer     – transfer knowledge from teacher to learner

Execute with e.g.::
    python main.py --cfg config/Wall-SS-SE-1.yml  --phase train_p1
    python main.py --cfg config/Wall-SS-SE-1.yml  --phase train_p2
    python main.py --cfg config/Wall/Wall-SS-SE.yml  --phase train_r2
    python main.py --cfg config/Wall/Wall-SS-SE.yml  --phase discover_ra
    python main.py --cfg config/Wall/Wall-SS-SE.yml  --phase build_cm
    python main.py --cfg config/Wall/Wall-SS-SE.yml  --phase cm_validate
    python main.py --cfg config/Wall/Wall-SS-SE.yml  --phase cm_infer
    python main.py --cfg config/Wall/Wall-SS-SE.yml --phase transfer --overrides '{"teacher_exp_id":"Wall-SS-SE"}'

The experiment suite implements a grid-world environment where an agent learns to navigate
around obstacles using Q-learning, with causal modeling to identify effective recovery
actions when collisions occur.
"""

from __future__ import annotations
import argparse, logging, pathlib, random, sys, json, yaml, pickle
from dataclasses import dataclass
from typing import Tuple, List, Dict
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
import joblib
import warnings
import traceback
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from evaluation_metrics import calculate_metrics, save_metrics
from dowhy import CausalModel
from econml.dml import CausalForestDML

# ---------------------------------------------------------------------------
# Global hyper‑parameters – override via YAML or --overrides {}
# ---------------------------------------------------------------------------
HYPER = {
    # Q‑learning phases
    "episodes_p1":     500000,
    "episodes_p2":     500000,
    "episodes_r2":     5000,
    "episodes_ra":     5000,
    "episodes_cm":     1,
    # RL learning rates
    "alpha":           0.10,
    "gamma":           0.95,
    "epsilon_start":   0.80,
    "epsilon_final":   0.01,
    # Causal model filtering
    "cm_min_per_ra":         20,
    # CausalForestDML hyperparams
    "cf_n_estimators":       200,
    "cf_min_samples_leaf":   10,
    "cf_cv":                 3,
    "cf_max_iter":           500,
    "cf_ridge_alpha":        1.0,  # Ridge regularization strength
    # Validation
    "validation_test_frac":  0.25,
    "permutation_runs":      150,
}
GRID_SIZE = 11  # 11×11 grid


# ——————————————————————————————————————————————————————————————
# Domain objects
# ——————————————————————————————————————————————————————————————
@dataclass(frozen=True)
class State:
    x: int
    y: int

@dataclass(frozen=True)
class Action:
    dx: int
    dy: int
    name: str

left = Action(0, -1, "left")
down = Action(1, 0, "down")
right = Action(0, 1, "right")
up = Action(-1, 0, "up")
ACTIONS: List[Action] = [up, right, down, left]
ACTION_INDEX: Dict[str, int] = {a.name: i for i, a in enumerate(ACTIONS)}

# Lookup table for action name from dx and dy
DX_DY_TO_NAME = {(a.dx, a.dy): a.name for a in ACTIONS}



# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class GridWorldEnv:
    """Minimal deterministic grid world with static obstacles."""
    def __init__(self, start: Tuple[int, int], goal: Tuple[int, int], obstacles: List[Tuple[int, int]]):
        self.start = State(*start)
        self.goal = State(*goal)
        self.obstacles = {State(*xy) for xy in obstacles}
        self.state = self.start
        self.timestep = 0

    # ------------------------------------------------------------------
    def reset(self):
        self.state = self.start
        self.timestep = 0
        return self.state

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def would_collide(self, state: State, action: Action) -> bool:
        """True if executing `action` from `state` would end in an obstacle or off-grid."""
        nx, ny = state.x + action.dx, state.y + action.dy
        off_grid = not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE)
        blocked  = State(nx, ny) in self.obstacles
        return off_grid or blocked


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class BaseAgent:
    """Interface every learner must expose."""
    def begin_episode(self):
        pass
    def act(self, state: State) -> Action:
        raise NotImplementedError
    def observe(self, *args, **kwargs):
        pass

class QAgent(BaseAgent):
    def __init__(self, rng: random.Random, episodes: int):
        self.rng = rng
        self.eps_schedule = np.linspace(HYPER["epsilon_start"], HYPER["epsilon_final"], int(0.75*episodes))
        self.eps_final = HYPER["epsilon_final"]
        # table shape: (GRID_SIZE, GRID_SIZE, len(ACTIONS))
        self.q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)), dtype=float)
        self.current_episode = 0

    # --------------------------------------------------------------
    def begin_episode(self):
        self.current_episode += 1

    # --------------------------------------------------------------
    def _epsilon(self):
        idx = self.current_episode - 1
        if idx < len(self.eps_schedule):
            return float(self.eps_schedule[idx])
        return self.eps_final

    # --------------------------------------------------------------
    def act(self, state: State) -> Action:
        if self.rng.random() < self._epsilon():
            return self.rng.choice(ACTIONS)
        qs = self.q[state.x, state.y]
        return ACTIONS[int(qs.argmax())]

    # --------------------------------------------------------------
    def observe(self, s: State, a: Action, r: float, sp: State, done: bool):
        a_idx = ACTION_INDEX[a.name]
        best_next = 0.0 if done else self.q[sp.x, sp.y].max()
        td_target = r + HYPER["gamma"]*best_next
        td_err = td_target - self.q[s.x, s.y, a_idx]
        self.q[s.x, s.y, a_idx] += HYPER["alpha"]*td_err

class RandomAgent(BaseAgent):
    """Policy‑less baseline for Phase R2."""
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.current_episode = 0

    def begin_episode(self):
        self.current_episode += 1

    def act(self, state: State) -> Action:
        return self.rng.choice(ACTIONS)


# ——————————————————————————————————————————————————————————————
# Helper classes
# ——————————————————————————————————————————————————————————————
class ResultsDir:
    """Stable path helper – guarantees every run has its own folder."""

    def __init__(self, exp_id: str):
        self.base = pathlib.Path("results") / exp_id
        self.base.mkdir(parents=True, exist_ok=True)

    def phase(self, name: str) -> pathlib.Path:
        """Get or create a phase-specific subdirectory."""
        p = self.base / name
        p.mkdir(exist_ok=True)
        return p

    def csv(self, stem: str) -> pathlib.Path:
        """Get path for a CSV file in the base directory."""
        return self.base / f"{stem}.csv"

    def pkl(self, stem: str) -> pathlib.Path:
        """Get path for a pickle file in the base directory."""
        return self.base / f"{stem}.pkl"

    def save_df(self, df: pd.DataFrame, stem: str):
        """Save DataFrame to CSV in the base directory."""
        df.to_csv(self.csv(stem), index=False)

# ——————————————————————————————————————————————————————————————
# Helper functions
# ——————————————————————————————————————————————————————————————

def make_ra_id(cs: State, cd_name: str, seq_names: list[str]) -> str:
    # Example:  RA_4_2_down-left_down_right
    tail = "-".join(seq_names)
    return f"RA_{cs.x}_{cs.y}_{cd_name}-{tail}" if tail else f"RA_{cs.x}_{cs.y}_{cd_name}"


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------

# ------------------------------------------------------------------
# Re-usable helper: run a single episode and log every step
# ------------------------------------------------------------------
def _run_episode(env: GridWorldEnv, agent: BaseAgent,
                 rng: random.Random, log_rows: List[Dict], state_visits: Counter):
    state = env.reset()
    agent.begin_episode()
    state_visits.clear()  # Reset counter for new episode

    t = 0
    while True:  # No artificial step limit
        # Track state visits BEFORE the action
        state_visits[(state.x, state.y)] += 1
        
        act = agent.act(state)
        next_state, reward, done, info = env.step(act)

        if isinstance(agent, QAgent):
            agent.observe(state, act, reward, next_state, done)

        log_rows.append({
            "episode"      : agent.current_episode,
            "timestep"     : t,
            "x"            : state.x,
            "y"            : state.y,
            "action"       : act.name,
            "reward"       : reward,
            "is_collision" : info["is_collision"],
            "goal_reached" : info["goal_reached"],
            "is_recovery_action": agent.in_recovery if hasattr(agent, 'in_recovery') else False,
        })
        state = next_state
        t += 1
        if done:
            # Don't forget to count the final state
            state_visits[(state.x, state.y)] += 1
            break


# ------------------------------------------------------------------
# Phase 1.1 – P1 (already obstacle‑free env)  -------------------------

def train_p1(cfg, out: ResultsDir, rng_env, rng_agent):
    out_phase = out.phase("p1")
    log_path = out_phase / "agent_action_log.csv"
    state_visits_path = out_phase / "state_visitation_log.csv"

    env = GridWorldEnv(cfg["agent_start"], cfg["agent_goal"], [])
    agent = QAgent(rng_agent, cfg["episodes_p1"])
    rows: List[Dict] = []
    state_visits = Counter()
    state_visits_rows: List[Dict] = []

    for _ in tqdm(range(cfg["episodes_p1"]), desc="Training P1"):
        _run_episode(env, agent, rng_env, rows, state_visits)
        # Log state visits for this episode
        for (x, y), count in state_visits.items():
            state_visits_rows.append({
                "episode": agent.current_episode,
                "x": x,
                "y": y,
                "visits": count
            })

    pd.DataFrame(rows).to_csv(log_path, index=False)
    pd.DataFrame(state_visits_rows).to_csv(state_visits_path, index=False)
    with open(out_phase / "q_table.pkl", "wb") as f:
        pickle.dump(agent.q, f)

# ------------------------------------------------------------------
# Phase 1.2 – P2: full Q-learning with obstacles
# ------------------------------------------------------------------
def train_p2(cfg, out: ResultsDir, rng_env, rng_agent):
    out_phase = out.phase("p2")
    log_path = out_phase / "agent_action_log.csv"
    state_visits_path = out_phase / "state_visitation_log.csv"

    env = GridWorldEnv(cfg["agent_start"], cfg["agent_goal"], cfg["obstacles"])
    agent = QAgent(rng_agent, cfg["episodes_p2"])
    rows: List[Dict] = []
    state_visits = Counter()
    state_visits_rows: List[Dict] = []

    for _ in tqdm(range(cfg["episodes_p2"]), desc="Training P2"):
        _run_episode(env, agent, rng_env, rows, state_visits)
        # Log state visits for this episode
        for (x, y), count in state_visits.items():
            state_visits_rows.append({
                "episode": agent.current_episode,
                "x": x,
                "y": y,
                "visits": count
            })

    pd.DataFrame(rows).to_csv(log_path, index=False)
    pd.DataFrame(state_visits_rows).to_csv(state_visits_path, index=False)
    with open(out_phase / "q_table.pkl", "wb") as f:
        pickle.dump(agent.q, f)

    # Compute and save evaluation metrics
    metrics = calculate_metrics(cfg["exp_id"], "P2", out.base)
    save_metrics(metrics, out.base / "evaluation_metrics_P2.json")
    logging.info("✓ Saved P2 evaluation metrics")


# ------------------------------------------------------------------
# Phase 1.3 – R2: random-exploration baseline (no learning)
# ------------------------------------------------------------------
def train_r2(cfg, out: ResultsDir, rng_env, rng_agent):
    out_phase = out.phase("r2")
    log_path = out_phase / "agent_action_log.csv"
    state_visits_path = out_phase / "state_visitation_log.csv"
    
    env = GridWorldEnv(cfg["agent_start"], cfg["agent_goal"], cfg["obstacles"])
    agent = RandomAgent(rng_agent)
    rows: List[Dict] = []
    state_visits = Counter()
    state_visits_rows: List[Dict] = []

    for _ in tqdm(range(cfg["episodes_r2"]), desc="Training R2"):
        _run_episode(env, agent, rng_env, rows, state_visits)
        # Log state visits for this episode
        for (x, y), count in state_visits.items():
            state_visits_rows.append({
                "episode": agent.current_episode,
                "x": x,
                "y": y,
                "visits": count
            })

    pd.DataFrame(rows).to_csv(log_path, index=False)
    pd.DataFrame(state_visits_rows).to_csv(state_visits_path, index=False)
    # no Q-table for R2

    # Compute and save evaluation metrics
    metrics = calculate_metrics(cfg["exp_id"], "R2", out.base)
    save_metrics(metrics, out.base / "evaluation_metrics_R2.json")
    logging.info("✓ Saved R2 evaluation metrics")


# ------------------------------------------------------------------
# Phase 2 – discover_ra : offline discovery & logging of Recovery-Actions
# ------------------------------------------------------------------

def ra_preprocessing(out: ResultsDir):
    """Process recovery action data to create causal model dataset.
    
    Args:
        out: ResultsDir object pointing to the experiment output directory
    """
    # Read recovery action log
    ra_log_path = out.phase("ra") / "recovery_action_log.csv"
    ra_df = pd.read_csv(ra_log_path)
    
    # Calculate ra_length (number of valid moves in RA sequence)
    ra_df['ra_length'] = ra_df['ra_sequence'].apply(lambda x: len(eval(x)))
    
    # Calculate path_length_post
    ra_df['path_length_post'] = ra_df.apply(
        lambda row: 0 if row['is_terminal'] else 
        row['final_path_length'] - row['path_length_pre'] - row['ra_length'],
        axis=1
    )
    
    # Create causal model dataset
    causal_df = pd.DataFrame({
        'episode_id'            : ra_df['episode'],
        'timestep'              : ra_df.index,  # Using index as timestep for ordering
        'collision_state'       : ra_df['collision_state'],
        'collision_direction'   : ra_df['collision_direction'],
        'recovery_action'       : ra_df['ra_id'],
        'path_length_pre'       : ra_df['path_length_pre'],
        'ra_length'             : ra_df['ra_length'],
        'path_length_post'      : ra_df['path_length_post'],
        'final_path_length'     : ra_df['final_path_length'],
    })

    # —— NEW BLOCK —— compute "which RA number in this episode"  
    causal_df = causal_df.sort_values(['episode_id','timestep'])
    causal_df['ra_index'] = causal_df.groupby('episode_id').cumcount() + 1
    # optional: how many RAs have already happened
    causal_df['num_previous_ras'] = causal_df['ra_index'] - 1
    # optional: what the *previous* RA was
    causal_df['prev_recovery_action'] = (
        causal_df.groupby('episode_id')['recovery_action']
                 .shift(1)
                 .fillna('NONE')
    )
    
    # Convert categorical columns
    categorical_cols = [
       'collision_state','collision_direction','recovery_action',
       'prev_recovery_action'
    ]
    for col in categorical_cols:
        causal_df[col] = causal_df[col].astype('category')
    
    # Save causal model dataset
    causal_path = out.phase("ra") / "causal_model_data.csv"
    causal_df.to_csv(causal_path, index=False)

def discover_ra(cfg, out: ResultsDir, rng_env, rng_agent):
    out_phase = out.phase("ra")
    log_path = out_phase / "agent_action_log.csv"
    ra_log_path = out_phase / "recovery_action_log.csv"
    state_visits_path = out_phase / "state_visitation_log.csv"

    # Load P1 Q-table for pre-collision policy
    p1_q_table = None
    p1_path = out.phase("p1") / "q_table.pkl"
    if p1_path.exists():
        with open(p1_path, "rb") as f:
            p1_q_table = pickle.load(f)
    else:
        raise FileNotFoundError(f"P1 Q-table not found at {p1_path}")

    class RecoveryAgent(BaseAgent):
        def __init__(self, rng: random.Random, p1_q_table: np.ndarray):
            self.rng = rng
            self.p1_q_table = p1_q_table
            self.current_episode = 0
            self.in_recovery = False
            self.current_ra = None
            self.episode_valid_moves = 0
            self.episode_collisions = 0
            self.collision_count = 0
            self.moves_since_last_collision = 0
            self.total_valid_moves = 0
            self.collision_valid_moves = {}  # Track valid moves after each collision
            self.collision_timesteps = {}  # Track when each collision occurred

        def begin_episode(self):
            self.current_episode += 1
            self.in_recovery = False
            self.current_ra = None
            self.episode_valid_moves = 0
            self.episode_collisions = 0
            self.collision_count = 0
            self.moves_since_last_collision = 0
            self.total_valid_moves = 0
            self.collision_valid_moves = {}
            self.collision_timesteps = {}

        def act(self, state: State) -> Action:
            if not self.in_recovery:
                # Use P1 policy before collision
                qs = self.p1_q_table[state.x, state.y]
                return ACTIONS[int(qs.argmax())]
            else:
                # Pure random during recovery
                return self.rng.choice(ACTIONS)

    env = GridWorldEnv(cfg["agent_start"], cfg["agent_goal"], cfg["obstacles"])
    agent = RecoveryAgent(rng_agent, p1_q_table)
    
    # Logging structures
    step_rows: List[Dict] = []
    ra_rows: List[Dict] = []
    episode_ras: List[Dict] = []  # Store RAs for current episode
    state_visits = Counter()
    state_visits_rows: List[Dict] = []

    for _ in tqdm(range(cfg["episodes_ra"]), desc="Discovering Recovery Actions"):
        state = env.reset()
        agent.begin_episode()
        state_visits.clear()  # Reset counter for new episode
        t = 0
        current_ra = None
        episode_ras.clear()

        while True:
            act = agent.act(state)
            next_state, reward, done, info = env.step(act)

            # Track state visits BEFORE the action
            state_visits[(state.x, state.y)] += 1

            # Log step
            step_rows.append({
                "episode": agent.current_episode,
                "timestep": t,
                "x": state.x,
                "y": state.y,
                "action": act.name,
                "reward": reward,
                "is_collision": info["is_collision"],
                "goal_reached": info["goal_reached"],
                "is_recovery_action": agent.in_recovery,
            })

            # Handle collision and recovery
            if info["is_collision"] and not agent.in_recovery:
                # Start new RA
                agent.in_recovery = True
                agent.episode_collisions += 1
                agent.collision_count = agent.episode_collisions
                current_ra = {
                    "episode": agent.current_episode,
                    "collision_count": agent.collision_count,
                    "collision_x": state.x,
                    "collision_y": state.y,
                    "collision_state": f"x{state.x}_y{state.y}",
                    "collision_direction": act.name,
                    "ra_sequence": [],
                    "path_length_pre": agent.total_valid_moves,  # Valid moves before this collision
                }
                agent.moves_since_last_collision = 0
                # Initialize tracking for this collision
                agent.collision_valid_moves[agent.collision_count] = 0
                agent.collision_timesteps[agent.collision_count] = t
            elif agent.in_recovery:
                if not info["is_collision"]:
                    # Add valid move to RA sequence
                    current_ra["ra_sequence"].append((act.dx, act.dy))
                    agent.moves_since_last_collision += 1
                    agent.total_valid_moves += 1
                    # Increment valid moves for all active collisions
                    for collision in agent.collision_valid_moves:
                        agent.collision_valid_moves[collision] += 1
                else:
                    # RA ended due to collision
                    if current_ra["ra_sequence"]:  # Only save if RA has valid moves
                        current_ra.update({
                            "end_state": (state.x, state.y),
                            "is_terminal": False,
                            "path_length_post": agent.collision_valid_moves[agent.collision_count],  # Valid moves after this collision
                            "final_path_length": agent.total_valid_moves,  # Will be updated at episode end
                        })
                        # Generate RA ID
                        seq_names = [DX_DY_TO_NAME[(dx, dy)] for dx, dy in current_ra["ra_sequence"]]
                        current_ra["ra_id"] = make_ra_id(
                            State(current_ra["collision_x"], current_ra["collision_y"]),
                            current_ra["collision_direction"],
                            seq_names
                        )
                        episode_ras.append(current_ra)
                    # Start new RA immediately
                    agent.episode_collisions += 1
                    agent.collision_count = agent.episode_collisions
                    current_ra = {
                        "episode": agent.current_episode,
                        "collision_count": agent.collision_count,
                        "collision_x": state.x,
                        "collision_y": state.y,
                        "collision_state": f"x{state.x}_y{state.y}",
                        "collision_direction": act.name,
                        "ra_sequence": [],
                        "path_length_pre": agent.total_valid_moves,  # Valid moves before this collision
                    }
                    agent.moves_since_last_collision = 0
                    # Initialize tracking for this collision
                    agent.collision_valid_moves[agent.collision_count] = 0
                    agent.collision_timesteps[agent.collision_count] = t
            else:
                agent.total_valid_moves += 1

            state = next_state
            t += 1

            if done:
                # Save final RA if in recovery mode
                if agent.in_recovery and current_ra and current_ra["ra_sequence"]:
                    current_ra.update({
                        "end_state": (state.x, state.y),
                        "is_terminal": True,
                        "path_length_post": agent.collision_valid_moves[agent.collision_count],  # Valid moves after this collision
                        "final_path_length": agent.total_valid_moves,  # Total valid moves in episode
                    })
                    # Generate RA ID
                    seq_names = [DX_DY_TO_NAME[(dx, dy)] for dx, dy in current_ra["ra_sequence"]]
                    current_ra["ra_id"] = make_ra_id(
                        State(current_ra["collision_x"], current_ra["collision_y"]),
                        current_ra["collision_direction"],
                        seq_names
                    )
                    episode_ras.append(current_ra)

                # Update all RAs in this episode with final path lengths
                for ra in episode_ras:
                    ra["final_path_length"] = agent.total_valid_moves  # Total valid moves in episode
                    ra_rows.append(ra)

                # Log state visits for this episode
                for (x, y), count in state_visits.items():
                    state_visits_rows.append({
                        "episode": agent.current_episode,
                        "x": x,
                        "y": y,
                        "visits": count
                    })
                break

    # Save logs
    pd.DataFrame(step_rows).to_csv(log_path, index=False)
    pd.DataFrame(ra_rows).to_csv(ra_log_path, index=False)
    pd.DataFrame(state_visits_rows).to_csv(state_visits_path, index=False)
    
    # Process recovery actions for causal model
    ra_preprocessing(out)

# ------------------------------------------------------------------
# Phase 3 – build_cm : build causal model
# ------------------------------------------------------------------

def build_cm(cfg:Dict, out:ResultsDir, *rngs):
    src = out.phase("ra")/"causal_model_data.csv"
    df = pd.read_csv(src)
    # filtering
        # --- 2. Positivity / overlap filtering
    cnt = df.groupby(["collision_state","collision_direction","recovery_action"]).size().reset_index(name="count")
    df = df.merge(
        cnt[cnt['count'] >= HYPER["cm_min_per_ra"]][['collision_state','collision_direction','recovery_action']],
        on=['collision_state','collision_direction','recovery_action'],
        how='inner'
    )
    ctx = df.groupby(["collision_state","collision_direction"]).recovery_action.nunique()
    df = df.set_index(["collision_state","collision_direction"]).loc[ctx[ctx>=2].index].reset_index()
    # encode
    le_cs=LabelEncoder().fit(df['collision_state']); le_cd=LabelEncoder().fit(df['collision_direction'])
    le_ra=LabelEncoder().fit(df['recovery_action']); le_ep=LabelEncoder().fit(df['episode_id'])
    df['collision_state_encoded']=le_cs.transform(df['collision_state'])
    df['collision_direction_encoded']=le_cd.transform(df['collision_direction'])
    df['recovery_action_encoded']=le_ra.transform(df['recovery_action'])
    df['episode_encoded']=le_ep.transform(df['episode_id'])
    df = df.sort_values(['episode_id','timestep'])
    df['ra_index']=df.groupby('episode_id').cumcount()+1
    cm_dir=out.phase('cm')
    joblib.dump(le_cs,cm_dir/'collision_state_le.pkl'); joblib.dump(le_cd,cm_dir/'collision_direction_le.pkl')
    joblib.dump(le_ra,cm_dir/'recovery_action_le.pkl'); joblib.dump(le_ep,cm_dir/'episode_le.pkl')
    # --- 4. Fit causal model with complexity-aware estimator ----------------------
    # include ra_length as a confounder/effect-modifier
    adj = [
        'collision_state_encoded',
        'collision_direction_encoded',
        'path_length_pre',
        'ra_length',            # ← make sure this is here!
        'ra_index',
        'episode_encoded'
    ]
    model = CausalModel(
        data=df,
        treatment='recovery_action_encoded',
        outcome='final_path_length',
        common_causes=adj,
        effect_modifiers=adj
    )
    identified = model.identify_effect(proceed_when_unidentifiable=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=FutureWarning)
        # prefer causal forest for small-sample heterogeneity
        cf_init = {
            'n_estimators'     : HYPER['cf_n_estimators'],
            'min_samples_leaf' : HYPER['cf_min_samples_leaf'],
            'max_iter'         : HYPER['cf_max_iter'],
            'cv'               : HYPER['cf_cv'],
            'alpha'            : HYPER['cf_ridge_alpha']
        }
        try:
            est = model.estimate_effect(
                identified,
                method_name='backdoor.econml.dml.CausalForestDML',
                method_params={ 'init_params': cf_init }
            )
        except Exception:
            logging.warning('CausalForestDML failed; trying DRLearner')
            try:
                est = model.estimate_effect(
                    identified,
                    method_name='backdoor.econml.drlearner.DRLearner',
                    method_params={ 'init_params': cf_init }
                )
            except Exception:
                logging.warning('DRLearner failed; falling back to ridge linear regression')
                est = model.estimate_effect(
                    identified,
                    method_name='backdoor.linear_regression'
                )
    joblib.dump(model, cm_dir / 'causal_model.pkl')

    # --- 5. Generate full RA-candidates list and best-RA lookup -------------------
    rows_best = []
    rows_all  = []
    for (cs, cd), group in df.groupby(['collision_state','collision_direction'], observed=False):
        for ra in group['recovery_action'].unique():
            # extract the one fixed ra_length
            ra_len          = int(group.loc[group['recovery_action']==ra, 'ra_length'].iloc[0])
            # and the corresponding pre‐collision context
            pre_len         = float(group.loc[group['recovery_action']==ra, 'path_length_pre'].iloc[0])
            ra_idx          = int(group.loc[group['recovery_action']==ra, 'ra_index'].iloc[0])
            ep_enc          = int(group.loc[group['recovery_action']==ra, 'episode_encoded'].iloc[0])

            # now call do() with every single confounder / effect‐modifier
            try:
                val = est.do({
                    'collision_state_encoded'     : int(le_cs.transform([cs])[0]),
                    'collision_direction_encoded' : int(le_cd.transform([cd])[0]),
                    'recovery_action_encoded'     : int(le_ra.transform([ra])[0]),
                    'ra_length'                   : ra_len,
                    'path_length_pre'             : pre_len,
                    'ra_index'                    : ra_idx,
                    'episode_encoded'             : ep_enc
                })['value']
            except Exception:
                val = float(group.loc[group['recovery_action']==ra,'final_path_length'].mean())
            rows_all.append((cs, cd, ra, val))

        # pick the RA with smallest predicted final_path_length
        best = min((r for r in rows_all if r[0]==cs and r[1]==cd), key=lambda x: x[3])
        rows_best.append(best)

    # dump all candidates
    df_all = pd.DataFrame(rows_all, columns=['collision_state','collision_direction','recovery_action','expected_final_path_length'])
    df_all.to_csv(cm_dir / 'ra_candidates.csv', index=False)
    # dump only best per context
    df_best = pd.DataFrame(rows_best, columns=['collision_state','collision_direction','recovery_action','expected_final_path_length'])
    df_best.to_csv(cm_dir / 'ra_selection_table.csv', index=False)
    logging.info(f"build_cm: wrote {len(df_best)} best-RA entries and {len(df_all)} total candidates")




# ------------------------------------------------------------------
# Scrubs
# ------------------------------------------------------------------


def validate_cm(cfg: Dict, out: 'ResultsDir'):
    """
    Run sanity‐check, hold‐out validation, placebo permutation, and positivity diagnostics
    on the causal model data and outputs in results/<exp_id>/cm/.
    """
    cm_dir = out.phase('cm')
    # Load datasets
    df_data = pd.read_csv(out.phase('ra') / 'causal_model_data.csv')
    df_best = pd.read_csv(cm_dir / 'ra_selection_table.csv')
    df_all  = pd.read_csv(cm_dir / 'ra_candidates.csv')

        # 0. Restrict to contexts and RAs used by the model
    df_filtered = pd.merge(
        df_data,
        df_all[['collision_state','collision_direction','recovery_action']],
        on=['collision_state','collision_direction','recovery_action'],
        how='inner'
    )

    # Load saved encoders to avoid unseen labels
    le_cs = joblib.load(cm_dir / 'collision_state_le.pkl')
    le_cd = joblib.load(cm_dir / 'collision_direction_le.pkl')
    le_ra = joblib.load(cm_dir / 'recovery_action_le.pkl')

    # 1. Sanity‐check range
    bounds = df_data.groupby(['collision_state','collision_direction'])['final_path_length'] \
                .agg(['min','max']).reset_index()
    merged = df_all.merge(bounds, on=['collision_state','collision_direction'])
    assert (merged.expected_final_path_length >= merged['min']).all() and \
           (merged.expected_final_path_length <= merged['max']).all(), \
           "Sanity‐check range failed"
    logging.info("validate_cm: sanity‐check range passed")

    # 2. Hold‐out validation
    test_frac = cfg.get('validation_test_frac', 0.2)
    # Encode full dataset before splitting
    df_enc = df_filtered.copy()
    df_enc['t_enc']  = le_ra.transform(df_enc['recovery_action'])
    df_enc['cs_enc'] = le_cs.transform(df_enc['collision_state'])
    df_enc['cd_enc'] = le_cd.transform(df_enc['collision_direction'])

    train, test = train_test_split(
        df_enc,
        test_size=test_frac,
        stratify=df_enc[['collision_state','collision_direction']],
        random_state=42
    )
    # Fit ridge regression on train
    X_train = train[['t_enc','cs_enc','cd_enc','path_length_pre']]
    y_train = train['final_path_length']
    model_lr = Ridge(alpha=1.0).fit(X_train, y_train)
    # Predict on test
    X_test = test[['t_enc','cs_enc','cd_enc','path_length_pre']]
    preds = model_lr.predict(X_test)
    rmse = mean_squared_error(test['final_path_length'], preds, squared=False)
    mae  = mean_absolute_error (test['final_path_length'], preds)
    logging.info(f"validate_cm: hold‐out RMSE={rmse:.2f}, MAE={mae:.2f}")

    # 3. Placebo / permutation test
    diffs = []
    for i in range(cfg.get('permutation_runs',100)):
        perm = train.copy()
        perm['t_enc'] = np.random.permutation(perm['t_enc'].values)
        mdl = Ridge(alpha=1.0).fit(
            perm[['t_enc','cs_enc','cd_enc','path_length_pre']], perm['final_path_length']
        )
        # random pair of treatment codes
        ra_vals = perm['t_enc'].unique()[:2]
        x0 = pd.DataFrame([{
            't_enc': ra_vals[0], 'cs_enc': perm['cs_enc'].iloc[0],
            'cd_enc': perm['cd_enc'].iloc[0], 'path_length_pre': perm['path_length_pre'].iloc[0]
        }])
        x1 = pd.DataFrame([{
            't_enc': ra_vals[1], 'cs_enc': perm['cs_enc'].iloc[0],
            'cd_enc': perm['cd_enc'].iloc[0], 'path_length_pre': perm['path_length_pre'].iloc[0]
        }])
        diff = mdl.predict(x0) - mdl.predict(x1)
        diffs.append(diff[0])
    logging.info(f"validate_cm: placebo mean effect diff ~ {np.mean(diffs):.2f} (should ~0)")

            # 4. Positivity & overlap diagnostics (warn if violations)
    freq_df = df_data.groupby([
        'collision_state','collision_direction','recovery_action'
    ]).size().reset_index(name='count')
    freq_df['prop'] = freq_df.groupby([
        'collision_state','collision_direction'
    ])['count'].transform(lambda x: x / x.sum())
    low  = freq_df['prop'].min()
    high = freq_df['prop'].max()
    if low < 0.05 or high > 0.90:
        logging.warning(
            f"validate_cm: positivity warning – min_prop={low:.2f}, max_prop={high:.2f}"  
        )
    else:
        logging.info("validate_cm: positivity & overlap passed")

# To call this function, you need to have run discover_ra first.
def cm_validate(cfg, out: ResultsDir, *rngs):
    validate_cm(cfg, out)

# To call this function, you need to have run build_cm first.
def cm_infer(cfg, out: ResultsDir, *rngs):
    """
    Run episodes using the causal-model–derived recovery actions.
    Tries best, 2nd best, … for each context, then falls back to random.
    Emits detailed print/log output for debugging.
    """
    # unpack RNGs
    rng_env, rng_agent = rngs

    # 1. Load P1 Q-table
    p1_path = out.phase("p1") / "q_table.pkl"
    logging.info(f"Loading P1 Q-table from {p1_path} …")
    if not p1_path.exists():
        raise FileNotFoundError("P1 Q-table not found; run train_p1 first.")
    with open(p1_path, "rb") as f:
        p1_q = pickle.load(f)
    logging.info(f"Loaded P1 Q-table, shape = {p1_q.shape}")

    # prepare the environment once
    env = GridWorldEnv(cfg["agent_start"], cfg["agent_goal"], cfg["obstacles"])

    # 2. Load ranked RA candidates
    cand_path = out.phase("cm") / "ra_candidates.csv"
    logging.info(f"Loading RA-candidates from {cand_path} …")
    cand_df = pd.read_csv(cand_path)
    candidates_map: Dict[Tuple[str,str], List[str]] = {}
    for (cs, cd), grp in cand_df.groupby(["collision_state","collision_direction"], observed=False):
        # sort ascending by expected_final_path_length
        sorted_ids = list(grp.sort_values("expected_final_path_length")["recovery_action"])
        candidates_map[(cs, cd)] = sorted_ids
    total_cands = len(cand_df)
    logging.info(f"Loaded {len(candidates_map)} contexts with {total_cands} total candidates")

    # 3. Load only needed RA sequences from log
    ra_log_path = out.phase("ra") / "recovery_action_log.csv"
    logging.info(f"Loading RA log from {ra_log_path} …")
    
    # Get set of unique RA-IDs we actually need
    needed = set(cand_df["recovery_action"].unique())
    logging.info(f"Found {len(needed)} unique RA-IDs needed for inference")
    
    # Build ra_map by scanning recovery_action_log.csv just once
    ra_map: Dict[str, List[Tuple[int,int]]] = {}
    for chunk in pd.read_csv(ra_log_path, usecols=["ra_id","ra_sequence"], chunksize=50_000):
        for ra_id, seq_str in zip(chunk.ra_id, chunk.ra_sequence):
            if ra_id in needed and ra_id not in ra_map:
                ra_map[ra_id] = eval(seq_str)
        if len(ra_map) == len(needed):
            break
    logging.info(f"Prepared {len(ra_map)} RA sequences as Action lists (skipped {len(needed)-len(ra_map)} unused RAs)")

    # 4. Define a CausalAgent that tries ranked RAs, then random fallback
    class CausalAgent(BaseAgent):
        def __init__(self, rng: random.Random):
            self.rng            = rng
            self.in_recovery    = False
            self.ra_seq         = []
            self.ra_idx         = 0
            self.total_ras      = 0
            # for each context, which candidate index to try next
            self.next_candidate = {}  # (cs,cd) -> int

        def begin_episode(self):
            self.in_recovery    = False
            self.ra_seq         = []
            self.ra_idx         = 0
            self.next_candidate = {}
            print("\n=== BEGIN EPISODE ===")

        def act(self, state: State) -> Action:
            # 1) continue current RA
            if self.in_recovery:
                if self.ra_idx < len(self.ra_seq):
                    dx, dy = self.ra_seq[self.ra_idx]
                    self.ra_idx += 1
                    a = next(a for a in ACTIONS if (a.dx, a.dy) == (dx, dy))
                    print(f"  → RA step {self.ra_idx}/{len(self.ra_seq)}: {a.name}")
                    return a
                else:
                    print("  → finished RA; reverting to P1 policy")
                    self.in_recovery = False

            # 2) P1 greedy action
            qs = p1_q[state.x, state.y]
            a  = ACTIONS[int(qs.argmax())]
            print(f"  P1 greedy chose: {a.name} at ({state.x},{state.y})")

            # 3) If that would collide, try next-ranked RA
            if env.would_collide(state, a):
                cs, cd = f"x{state.x}_y{state.y}", a.name
                print(f"  Collision predicted at ({cs}) via '{cd}'")
                candidates = candidates_map.get((cs, cd), [])
                idx = self.next_candidate.get((cs, cd), 0)

                if idx < len(candidates):
                    ra_id = candidates[idx]
                    self.next_candidate[(cs, cd)] = idx + 1
                    self.ra_seq        = ra_map[ra_id]
                    self.ra_idx        = 0
                    self.in_recovery   = True
                    self.total_ras    += 1
                    print(f"  Applying RA '{ra_id}' ({len(self.ra_seq)} steps), rank#{idx+1}")
                    return self.act(state)  # recurse into the first RA step

                # exhausted all learned RAs → random valid fallback
                print(f"  Exhausted {len(candidates)} RAs at {cs},{cd}; falling back randomly")
                valid = [x for x in ACTIONS if not env.would_collide(state, x)]
                return self.rng.choice(valid) if valid else a

            # 4) no collision → execute P1 action
            print(f"  No collision → executing {a.name}")
            return a

    # 5. Run inference episodes with full logging + tqdm
    n_eps = cfg["episodes_cm"]
    logging.info("Starting %d inference episodes…", n_eps)
    infer_rows = []
    state_visits = Counter()
    state_visits_rows: List[Dict] = []

    with tqdm(total=n_eps, desc="cm_infer episodes", unit="ep") as ep_bar:
        for ep in range(n_eps):
            logging.info(" → Episode %d/%d", ep+1, n_eps)
            state = env.reset()
            agent = CausalAgent(rng_agent)
            t = 0
            state_visits.clear()  # Reset counter for new episode

            # run one episode
            while True:
                # Track state visits BEFORE the action
                state_visits[(state.x, state.y)] += 1
                
                a = agent.act(state)
                s2, rew, done, info = env.step(a)
                
                # Check if this action would cause a collision
                is_collision = env.would_collide(state, a)
                
                infer_rows.append({
                    "episode": ep,
                    "timestep": t,
                    "x": state.x, "y": state.y,
                    "action": a.name,
                    "reward": rew,
                    "is_collision": is_collision,
                    "goal_reached": info["goal_reached"],
                    "in_recovery": agent.in_recovery
                })
                state, t = s2, t+1
                if done:
                    # Don't forget to count the final state
                    state_visits[(state.x, state.y)] += 1
                    break

            # Log state visits for this episode
            for (x, y), count in state_visits.items():
                state_visits_rows.append({
                    "episode": ep,
                    "x": x,
                    "y": y,
                    "visits": count
                })

            # update progress bar with this episode's stats
            ep_bar.set_postfix({
                "steps": t,
                "total_RAs": agent.total_ras
            })
            ep_bar.update(1)

    # 6. Save
    inf_dir = out.phase("p1_cm")
    inf_dir.mkdir(exist_ok=True)
    # Save action log
    action_log_path = inf_dir / "agent_action_log.csv"
    pd.DataFrame(infer_rows).to_csv(action_log_path, index=False)
    # Save state visits log
    state_visits_path = inf_dir / "state_visitation_log.csv"
    pd.DataFrame(state_visits_rows).to_csv(state_visits_path, index=False)
    logging.info("cm_infer: wrote %d steps (used %d RAs) to %s and state visits to %s",
                 len(infer_rows), agent.total_ras, action_log_path, state_visits_path)
    
    # calucalte metrics
    metrics = calculate_metrics(cfg["exp_id"], "P1_CM", out.base)
    save_metrics(metrics, out.base / "evaluation_metrics_P1_CM.json")
    logging.info("✓ Saved cm_infer evaluation metrics")



def transfer(cfg, out: ResultsDir, rng_env, rng_agent):
    """
    Transfer phase: the learner uses its P1 Q-table (free‐grid) plus
    recovery‐action suggestions from a teacher's causal model.
    Teacher exp_id is given in cfg["teacher_exp_id"].
    """
    teacher_id = cfg["teacher_exp_id"]
    teacher_out = ResultsDir(teacher_id)

    # 1. Load free-grid Q-table from teacher's P1
    # use learner's P1
    p1_path = out.phase("p1") / "q_table.pkl"

    if not p1_path.exists():
        raise FileNotFoundError(f"P1 Q-table not found in teacher {teacher_id}: {p1_path}")
    with open(p1_path, "rb") as f:
        p1_q = pickle.load(f)
    logging.info(f"Loaded teacher's P1 Q-table from {p1_path}, shape = {p1_q.shape}")

    # 2. Load teacher's causal model artifacts
    cm_dir = teacher_out.phase("cm")
    # 2a. RA candidates ranking
    cand_path = cm_dir / "ra_candidates.csv"
    logging.info(f"Loading RA-candidates from {cand_path} …")
    cand_df = pd.read_csv(cand_path)
    candidates_map: Dict[Tuple[str,str], List[str]] = {}
    for (cs, cd), grp in cand_df.groupby(["collision_state","collision_direction"], observed=False):
        # sort ascending by expected_final_path_length
        sorted_ids = list(grp.sort_values("expected_final_path_length")["recovery_action"])
        candidates_map[(cs, cd)] = sorted_ids
    total_cands = len(cand_df)
    logging.info(f"Loaded {len(candidates_map)} contexts with {total_cands} total candidates")

    # 3. Load only needed RA sequences from teacher's log
    ra_log_path = teacher_out.phase("ra") / "recovery_action_log.csv"
    logging.info(f"Loading RA log from {ra_log_path} …")
    
    # Get set of unique RA-IDs we actually need
    needed = set(cand_df["recovery_action"].unique())
    logging.info(f"Found {len(needed)} unique RA-IDs needed for transfer")
    
    # Build ra_map by scanning recovery_action_log.csv just once
    ra_map: Dict[str, List[Tuple[int,int]]] = {}
    for chunk in pd.read_csv(ra_log_path, usecols=["ra_id","ra_sequence"], chunksize=50_000):
        for ra_id, seq_str in zip(chunk.ra_id, chunk.ra_sequence):
            if ra_id in needed and ra_id not in ra_map:
                ra_map[ra_id] = eval(seq_str)
        if len(ra_map) == len(needed):
            break
    logging.info(f"Prepared {len(ra_map)} RA sequences as Action lists (skipped {len(needed)-len(ra_map)} unused RAs)")

    # 4. Prepare learner's env (start/goal/obstacles from current cfg)
    env = GridWorldEnv(cfg["agent_start"], cfg["agent_goal"], cfg["obstacles"])

    # 5. Run transfer episodes with full logging + tqdm
    n_eps = cfg.get("episodes_transfer", cfg.get("episodes_cm", 1))
    logging.info("Starting %d transfer episodes with teacher=%s …", n_eps, teacher_id)
    transfer_rows = []
    state_visits = Counter()
    state_visits_rows: List[Dict] = []


    class CausalAgent(BaseAgent):
        def __init__(self, rng: random.Random):
            self.rng            = rng
            self.in_recovery    = False
            self.ra_seq         = []
            self.ra_idx         = 0
            self.total_ras      = 0
            # for each context, which candidate index to try next
            self.next_candidate = {}  # (cs,cd) -> int

        def begin_episode(self):
            self.in_recovery    = False
            self.ra_seq         = []
            self.ra_idx         = 0
            self.next_candidate = {}
            print("\n=== BEGIN EPISODE ===")

        def act(self, state: State) -> Action:
            # 1) continue current RA
            if self.in_recovery:
                if self.ra_idx < len(self.ra_seq):
                    dx, dy = self.ra_seq[self.ra_idx]
                    self.ra_idx += 1
                    a = next(a for a in ACTIONS if (a.dx, a.dy) == (dx, dy))
                    print(f"  → RA step {self.ra_idx}/{len(self.ra_seq)}: {a.name}")
                    return a
                else:
                    print("  → finished RA; reverting to P1 policy")
                    self.in_recovery = False

            # 2) P1 greedy action
            qs = p1_q[state.x, state.y]
            a  = ACTIONS[int(qs.argmax())]
            print(f"  P1 greedy chose: {a.name} at ({state.x},{state.y})")

            # 3) If that would collide, try next-ranked RA
            if env.would_collide(state, a):
                cs, cd = f"x{state.x}_y{state.y}", a.name
                print(f"  Collision predicted at ({cs}) via '{cd}'")
                candidates = candidates_map.get((cs, cd), [])
                idx = self.next_candidate.get((cs, cd), 0)

                if idx < len(candidates):
                    ra_id = candidates[idx]
                    self.next_candidate[(cs, cd)] = idx + 1
                    self.ra_seq        = ra_map[ra_id]
                    self.ra_idx        = 0
                    self.in_recovery   = True
                    self.total_ras    += 1
                    print(f"  Applying RA '{ra_id}' ({len(self.ra_seq)} steps), rank#{idx+1}")
                    return self.act(state)  # recurse into the first RA step

                # exhausted all learned RAs → random valid fallback
                print(f"  Exhausted {len(candidates)} RAs at {cs},{cd}; falling back randomly")
                valid = [x for x in ACTIONS if not env.would_collide(state, x)]
                return self.rng.choice(valid) if valid else a

            # 4) no collision → execute P1 action
            print(f"  No collision → executing {a.name}")
            return a

    with tqdm(total=n_eps, desc="transfer episodes", unit="ep") as ep_bar:
        for ep in range(n_eps):
            logging.info(" → Episode %d/%d", ep+1, n_eps)
            state = env.reset()
            agent = CausalAgent(rng_agent)  # Reuse CausalAgent from cm_infer
            t = 0
            state_visits.clear()  # Reset counter for new episode

            # run one episode
            while True:
                # Track state visits BEFORE the action
                state_visits[(state.x, state.y)] += 1
                
                a = agent.act(state)
                s2, rew, done, info = env.step(a)
                
                # Check if this action would cause a collision
                is_collision = env.would_collide(state, a)
                
                transfer_rows.append({
                    "episode": ep,
                    "timestep": t,
                    "x": state.x, "y": state.y,
                    "action": a.name,
                    "reward": rew,
                    "is_collision": is_collision,
                    "goal_reached": info["goal_reached"],
                    "in_recovery": agent.in_recovery
                })
                state, t = s2, t+1
                if done:
                    # Don't forget to count the final state
                    state_visits[(state.x, state.y)] += 1
                    break

            # Log state visits for this episode
            for (x, y), count in state_visits.items():
                state_visits_rows.append({
                    "episode": ep,
                    "x": x,
                    "y": y,
                    "visits": count
                })

            # update progress bar with this episode's stats
            ep_bar.set_postfix({
                "steps": t,
                "total_RAs": agent.total_ras
            })
            ep_bar.update(1)

    # 6. Save under a new "transfer" phase directory
    td = out.phase("transfer")
    td.mkdir(exist_ok=True)
    # Save action log
    action_log_path = td / "agent_action_log.csv"
    pd.DataFrame(transfer_rows).to_csv(action_log_path, index=False)
    # Save state visits log
    state_visits_path = td / "state_visitation_log.csv"
    pd.DataFrame(state_visits_rows).to_csv(state_visits_path, index=False)
    logging.info("transfer: wrote %d steps (used %d RAs) to %s and state visits to %s",
                 len(transfer_rows), agent.total_ras, action_log_path, state_visits_path)
    
    # 7. Compute + save metrics
    metrics = calculate_metrics(cfg["exp_id"], "TRANSFER", out.base)
    save_metrics(metrics, out.base / "evaluation_metrics_TRANSFER.json")
    logging.info("✓ Saved transfer evaluation metrics")


# ------------------------------------------------------------------
PHASES = {
    "train_p1": train_p1,
    "train_p2": train_p2,
    "train_r2": train_r2,
    "discover_ra": discover_ra,
    "build_cm": build_cm,
    "cm_validate": cm_validate,
    "cm_infer": cm_infer,
    "transfer": transfer,
}

# ---------------------------------------------------------------------------
# CLI + entry‑point
# ---------------------------------------------------------------------------

def _load_cfg(path: str) -> Dict:
    """Load YAML config and merge with global hyperparameters.
    YAML config can override any hyperparameter, but the defaults come from HYPER.
    """
    cfg = yaml.safe_load(open(path, encoding='utf-8'))
    # Start with global hyperparameters, then override with YAML
    merged = {**HYPER, **cfg}
    return merged

def _apply_overrides(cfg: Dict, overrides: str|None) -> Dict:
    """Apply command-line overrides to the config."""
    if overrides:
        cfg.update(json.loads(overrides))
    return cfg

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="YAML config file")
    parser.add_argument("--phase", required=True, choices=PHASES.keys())
    parser.add_argument("--overrides", help="JSON string of cfg overrides")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(levelname)s | %(message)s")

    cfg = _apply_overrides(_load_cfg(args.cfg), args.overrides)

    # Initialize separate seeds for different components
    env_seed = cfg.get("env_seed", 42)
    agent_seed = cfg.get("agent_seed", 43)
    numpy_seed = cfg.get("numpy_seed", 44)

    # deterministic seeds ----------------------------------------------------------------
    rng_env = random.Random(env_seed)
    rng_agent = random.Random(agent_seed)
    np.random.seed(numpy_seed)  # for potential use in causal model later

    out = ResultsDir(cfg["exp_id"])
    logging.info("Running phase %s for %s", args.phase, cfg["exp_id"])
    PHASES[args.phase](cfg, out, rng_env, rng_agent)
    logging.info("✓ finished")

if __name__ == "__main__":
    main() 