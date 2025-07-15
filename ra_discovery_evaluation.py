#!/usr/bin/env python3
"""
Script to evaluate recovery action discovery data from all experiments.
Generates statistics about recovery actions and collision paths.
"""

import pathlib
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_to_python_types(obj: Any) -> Any:
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_python_types(item) for item in obj)
    return obj

def process_experiment(exp_dir: pathlib.Path) -> Tuple[Dict, pd.DataFrame]:
    """Process a single experiment's recovery action data.
    
    Args:
        exp_dir: Path to experiment directory
        
    Returns:
        Tuple containing:
        - Dictionary with overview statistics
        - DataFrame with collision path data
    """
    ra_dir = exp_dir / "ra"
    if not ra_dir.exists():
        logging.info(f"No 'ra' directory found in {exp_dir.name}")
        return None, None
        
    # Read recovery action log
    ra_log_path = ra_dir / "recovery_action_log.csv"
    if not ra_log_path.exists():
        logging.info(f"No recovery_action_log.csv found in {exp_dir.name}/ra")
        return None, None
        
    logging.info(f"Processing {exp_dir.name}...")
    
    # Read only necessary columns to save memory
    try:
        ra_df = pd.read_csv(ra_log_path, usecols=[
            'episode', 'collision_state', 'collision_direction', 
            'ra_id', 'final_path_length'
        ])
        logging.info(f"Loaded {len(ra_df)} rows from {exp_dir.name}")
    except Exception as e:
        logging.error(f"Error reading {ra_log_path}: {str(e)}")
        return None, None
    
    # Calculate overview statistics
    try:
        overview = {
            "episodes": int(ra_df["episode"].max()),
            "collisions": int(len(ra_df)),
            "episode_path_lengths": ra_df.groupby("episode")["final_path_length"].first().values.astype(int).tolist(),
            "contexts": [(str(cs), str(cd)) for cs, cd in zip(
                ra_df["collision_state"].unique(), 
                ra_df["collision_direction"].unique()
            )],
            "ra_s": ra_df["ra_id"].unique().tolist()
        }
        
        # Calculate collision path data
        collision_path_data = []
        for episode in ra_df["episode"].unique():
            episode_data = ra_df[ra_df["episode"] == episode]
            collision_path_data.append({
                "episode": int(episode),
                "number_of_collisions": int(len(episode_data)),
                "final_path_length": int(episode_data["final_path_length"].iloc[0])
            })
        
        collision_path_df = pd.DataFrame(collision_path_data)
        logging.info(f"Processed {exp_dir.name} successfully")
        return overview, collision_path_df
        
    except Exception as e:
        logging.error(f"Error processing data for {exp_dir.name}: {str(e)}")
        return None, None

def main():
    # Create output directory
    output_dir = pathlib.Path("results") / "ra_discovery_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    
    # Get all experiment directories
    results_dir = pathlib.Path("results")
    experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name != "ra_discovery_evaluation"]
    logging.info(f"Found {len(experiment_dirs)} experiment directories")
    
    # Process each experiment
    all_overviews = {}
    all_collision_paths = {}
    
    for exp_dir in experiment_dirs:
        logging.info(f"Starting to process {exp_dir.name}")
        overview, collision_path_df = process_experiment(exp_dir)
        if overview is not None:
            all_overviews[exp_dir.name] = overview
            all_collision_paths[exp_dir.name] = collision_path_df
            logging.info(f"Successfully processed {exp_dir.name}")
        else:
            logging.info(f"Skipped {exp_dir.name} due to missing or invalid data")
    
    # Save overview statistics
    if all_overviews:
        # Convert all NumPy types to Python native types
        all_overviews = convert_to_python_types(all_overviews)
        
        with open(output_dir / "ra_overview_table.json", "w") as f:
            json.dump(all_overviews, f, indent=2)
        logging.info(f"Saved overview statistics to {output_dir / 'ra_overview_table.json'}")
        
        # Save collision path data
        for exp_id, df in all_collision_paths.items():
            df.to_csv(output_dir / f"collision_path_table_{exp_id}.csv", index=False)
        logging.info(f"Saved collision path data for {len(all_collision_paths)} experiments")
    else:
        logging.warning("No valid data was processed")
    
    print(f"✓ Processed {len(all_overviews)} experiments with recovery action data")
    print(f"✓ Results saved to {output_dir}")

if __name__ == "__main__":
    main() 