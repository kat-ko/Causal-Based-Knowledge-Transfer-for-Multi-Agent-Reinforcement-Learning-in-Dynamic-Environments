#!/usr/bin/env python3
import pathlib
import logging
import yaml
from main import ra_preprocessing, ResultsDir

def get_exp_id_from_config(config_path):
    """Extract experiment ID from config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('exp_id')

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    
    # List of config files
    config_files = [
        "config/Wall/Wall-SS-SE.yml",
        "config/Wall/Wall-DS-SE-a1.yml",
        "config/Wall/Wall-DS-SE-a2.yml",
        "config/Wall/Wall-SS-DE-a1.yml",
        "config/Wall/Wall-SS-DE-a2.yml",
        "config/Wall/Wall-DS-DE-a1.yml",
        "config/Wall/Wall-DS-DE-a2.yml",
        "config/U/U-SS-SE.yml",
        "config/U/U-DS-SE-a1.yml",
        "config/U/U-DS-SE-a2.yml",
        "config/U/U-SS-DE-a1.yml",
        "config/U/U-SS-DE-a2.yml",
        "config/U/U-DS-DE-a1.yml",
        "config/U/U-DS-DE-a2.yml",
        "config/ReverseU/ReverseU-SS-SE.yml",
        "config/ReverseU/ReverseU-DS-SE-a1.yml",
        "config/ReverseU/ReverseU-DS-SE-a2.yml",
        "config/ReverseU/ReverseU-SS-DE-a1.yml",
        "config/ReverseU/ReverseU-SS-DE-a2.yml",
        "config/ReverseU/ReverseU-DS-DE-a1.yml",
        "config/ReverseU/ReverseU-DS-DE-a2.yml"
    ]
    
    for config_file in config_files:
        try:
            exp_id = get_exp_id_from_config(config_file)
            if not exp_id:
                logging.error(f"Could not find exp_id in {config_file}")
                continue
                
            logging.info(f"Processing {exp_id}...")
            out = ResultsDir(exp_id)
            ra_preprocessing(out)
            logging.info(f"✓ Completed RA preprocessing for {exp_id}")
            
        except Exception as e:
            logging.error(f"Error processing {config_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 