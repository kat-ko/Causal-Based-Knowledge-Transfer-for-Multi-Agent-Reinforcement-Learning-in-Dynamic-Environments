#!/usr/bin/env python3
"""
optimal_path_table.py – compute and persist the optimal path table for each context.

The optimal path table maps each (collision_state, collision_direction) pair to
the best recovery action (RA) for that context, based on the causal model's
predictions of final path length.
"""

from __future__ import annotations
import logging
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple

def compute_optimal_path_table(df: pd.DataFrame, est, le_cs, le_cd, le_ra) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Compute the optimal path table by evaluating each RA in each context.
    
    Args:
        df: DataFrame containing the causal model data
        est: The fitted causal estimator
        le_cs: LabelEncoder for collision states
        le_cd: LabelEncoder for collision directions
        le_ra: LabelEncoder for recovery actions
        
    Returns:
        Tuple of (rows_all, rows_best) where:
        - rows_all: List of all (cs, cd, ra, predicted_value) tuples
        - rows_best: List of best (cs, cd, ra, predicted_value) tuples per context
    """
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
    
    return rows_all, rows_best

def save_optimal_path_table(rows_all: List[Tuple], rows_best: List[Tuple], cm_dir: Path):
    """
    Save the optimal path table to CSV files.
    
    Args:
        rows_all: List of all (cs, cd, ra, predicted_value) tuples
        rows_best: List of best (cs, cd, ra, predicted_value) tuples per context
        cm_dir: Directory to save the files in
    """
    # dump all candidates
    df_all = pd.DataFrame(rows_all, columns=[
        'collision_state',
        'collision_direction',
        'recovery_action',
        'expected_final_path_length'
    ])
    df_all.to_csv(cm_dir / 'ra_candidates.csv', index=False)
    
    # dump only best per context
    df_best = pd.DataFrame(rows_best, columns=[
        'collision_state',
        'collision_direction',
        'recovery_action',
        'expected_final_path_length'
    ])
    df_best.to_csv(cm_dir / 'ra_selection_table.csv', index=False)
    
    logging.info(f"Saved {len(df_best)} best-RA entries and {len(df_all)} total candidates")

def build_optimal_path_table(df: pd.DataFrame, est, le_cs, le_cd, le_ra, cm_dir: Path):
    """
    Build and save the optimal path table.
    
    Args:
        df: DataFrame containing the causal model data
        est: The fitted causal estimator
        le_cs: LabelEncoder for collision states
        le_cd: LabelEncoder for collision directions
        le_ra: LabelEncoder for recovery actions
        cm_dir: Directory to save the files in
    """
    rows_all, rows_best = compute_optimal_path_table(df, est, le_cs, le_cd, le_ra)
    save_optimal_path_table(rows_all, rows_best, cm_dir)

if __name__ == "__main__":
    # Example usage
    import sys
    from main_cf import ResultsDir
    
    if len(sys.argv) != 2:
        print("Usage: python optimal_path_table.py <exp_id>")
        sys.exit(1)
    
    exp_id = sys.argv[1]
    out = ResultsDir(exp_id)
    
    # Load the causal model and data
    cm_dir = out.phase('cm')
    model = joblib.load(cm_dir / 'causal_model.pkl')
    df = pd.read_csv(out.phase('ra') / 'causal_model_data.csv')
    
    # Load encoders
    le_cs = joblib.load(cm_dir / 'collision_state_le.pkl')
    le_cd = joblib.load(cm_dir / 'collision_direction_le.pkl')
    le_ra = joblib.load(cm_dir / 'recovery_action_le.pkl')
    
    # Build and save the optimal path table
    build_optimal_path_table(df, model, le_cs, le_cd, le_ra, cm_dir) 