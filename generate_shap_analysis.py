#!/usr/bin/env python
"""
Generate SHAP Analysis from Pickled Models.

This script loads the trained models and data, calculates SHAP values,
and generates summary plots and data files.
"""

import pickle
import logging
import sys
import warnings
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Add the project root to the path to import local modules
sys.path.insert(0, str(Path(__file__).parent))

from config.config import load_config, Config
from src import utils
from src import shap_utils
from src.pipeline import DataProcessor

# Filter warnings
warnings.filterwarnings('ignore')

def load_pickled_models(models_dir: Path, objective_names: list) -> dict:
    """Load trained models from pickle files."""
    models = {}
    for obj_name in objective_names:
        model_path = models_dir / f"{obj_name}_model.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                models[obj_name] = model
                logging.info(f"Loaded {obj_name} model from {model_path}")
            except Exception as e:
                logging.error(f"Failed to load {obj_name} model: {e}")
        else:
            logging.warning(f"Model file not found: {model_path}")
    return models

def main():
    # 1. Load Configuration
    # Assuming config is in 'config/config.py' and we can load default or specified
    # Using default loading for now
    try:
        config = load_config("config/config.yaml") # Try loading from file if exists
    except:
        # Fallback to default Config object if yaml loading fails or isn't set up
        config = Config()
    
    # Setup logging
    utils.setup_logging(config.output_dir, level="INFO")
    logging.info("Starting SHAP Analysis Generation...")

    # 2. Load and Process Data
    # We need the DataProcessor to get the Scalers and processed Datasets
    # Note: The original pipeline fits scalers during data loading.
    # Ideally, we should load saved scalers to ensure consistency.
    # If scalers weren't saved, we re-fit them on the *same* data, which should be deterministic.
    
    data_processor = DataProcessor(config)
    logging.info("Loading and processing data to reconstruct scalers...")
    try:
        # This re-loads data and re-fits scalers. 
        # Crucial: Ensure the data file hasn't changed since model training.
        datasets, formulation_ids = data_processor.load_and_process_data()
    except Exception as e:
        logging.error(f"Failed to load and process data: {e}")
        sys.exit(1)

    if not datasets:
        logging.error("No datasets loaded. Cannot proceed.")
        sys.exit(1)

    # 3. Load Pickled Models
    models_dir = Path(config.output_dir) / "models"
    models = load_pickled_models(models_dir, config.objective_names)

    if not models:
        logging.error("No models loaded. Cannot proceed with SHAP analysis.")
        sys.exit(1)

    # 4. Run SHAP Analysis
    logging.info("Running SHAP analysis...")
    try:
        all_shap_results, all_real_unit_results = shap_utils.run_shap_analysis(
            models, 
            datasets, 
            data_processor, 
            formulation_ids, 
            config
        )
    except Exception as e:
        logging.error(f"SHAP analysis failed: {e}", exc_info=True)
        sys.exit(1)

    # 5. Generate Summary Plots
    logging.info("Generating SHAP summary plots...")
    shap_plot_dir = Path(config.output_dir) / 'shap_results' / 'plots'
    shap_plot_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        shap_utils.plot_real_unit_shap_summary_for_all_models(
            all_shap_results, 
            all_real_unit_results, 
            config.feature_columns,
            save_dir=str(shap_plot_dir)
        )
    except Exception as e:
        logging.error(f"Failed to generate summary plots: {e}", exc_info=True)

    logging.info("SHAP analysis completed successfully.")
    logging.info(f"Results saved to {Path(config.output_dir) / 'shap_results'}")

if __name__ == "__main__":
    main()

