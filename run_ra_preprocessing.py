#!/usr/bin/env python3
import pathlib
import logging
from main import ra_preprocessing, ResultsDir

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    
    # Get experiment ID from command line
    import sys
    if len(sys.argv) != 2:
        print("Usage: python run_ra_preprocessing.py <experiment_id>")
        sys.exit(1)
    
    exp_id = sys.argv[1]
    out = ResultsDir(exp_id)
    
    try:
        ra_preprocessing(out)
        logging.info("✓ RA preprocessing completed successfully")
    except Exception as e:
        logging.error(f"Error during RA preprocessing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 