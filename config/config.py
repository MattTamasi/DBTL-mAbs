#!/usr/bin/env python
"""
BoTorch configuration - standard settings for the optimization pipeline.
Modified to use basic k-fold CV, fixed reference point, and basic SingleTaskGP.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class Config:
    """Configuration for BoTorch optimization pipeline."""
    
    # Core settings
    random_state: int = 42
    device: str = "auto"
    data_file: str = "data/Antibody_Formulation_Data_Complete_Seed to Batch 2.xlsx"
    output_dir: str = "outputs/models/botorch_models"
    
    # Feature columns (consistent across all datasets)
    feature_columns: List[str] = field(default_factory=lambda: [
        "Molarity", "NaCl", "Sucrose", "Arginine", "pH", "Buffer", "Concentration (mg/mL)"
    ])
    
    # Target datasets and objectives (1.0 = maximize, -1.0 = minimize)
    objectives: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "tm": {
            "target_column": "Tm (C) Mean",
            "std_column": "Tm (C) Std", 
            "direction": 1.0,  # maximize
            "concentration": "concentration_tm",
            "value": "tm",
            "std": "tm_std"
        },
        "diffusion": {
            "target_column": "Diffusion Coefficient Mean (cm²/s)",
            "std_column": "Diffusion Coefficient Std (cm²/s)",
            "direction": 1.0,  # maximize
            "concentration": "concentration_diff", 
            "value": "diff",
            "std": "diff_std"
        },
        "viscosity": {
            "target_column": "Viscosity (cP) Mean",
            "std_column": "Viscosity (cP) Std",
            "direction": -1.0,  # minimize
            "concentration": "concentration_visc",
            "value": "visc", 
            "std": "visc_std"
        }
    })
    
    # Source columns for data loading
    excipient_columns: List[str] = field(default_factory=lambda: [
        "Molarity", "NaCl", "Sucrose", "Arginine", "pH", "Buffer"
    ])
    
    metadata_columns: List[str] = field(default_factory=lambda: [
        "Formulation ID", "Objective", "Generation"
    ])
    
    # Model settings
    cv_folds: int = 10
    min_samples: int = 15
    min_r2: float = 0.2
    
    # Optimization settings
    batch_size: int = 6
    mc_samples: int = 128
    num_restarts: int = 10
    reference_point: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # Feature bounds for optimization (unscaled, in real units)
    # These bounds are derived from the experimental data range
    feature_bounds = {
        "Molarity": (6.979166031, 98.02084351),
        "NaCl": (3.125, 146.875),
        "Sucrose": (0.520833313, 24.47916667),        
        "Arginine": (2.604166667, 122.3958359),
        "pH": (4.0, 7.0),
        "Buffer": (1, 2),
        # Note: Antibody Concentration is handled dynamically based on the objective
    }

    valid_buffer_pH_pairs = {
        1: [4.0, 4.5, 5.0, 5.5],  # Acetate
        2: [4.0, 5.0, 6.0, 7.0]   # Citrate
    }


    # Prediction settings
    # concentrations used for each objective when calculting objectivce predictions on candidates
    # returned by the optimization loop
    pred_conc = {
        "tm": 15.0,
        "diffusion": 15.0,
        "viscosity": 120.0
    }

    # Output flags
    save_models: bool = True
    save_candidates: bool = True
    
    @property
    def torch_device(self) -> torch.device:
        """Get torch device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
    
    @property
    def objective_names(self) -> List[str]:
        """Get ordered list of objective names."""
        return list(self.objectives.keys())
    
    @property
    def objective_directions(self) -> Dict[str, float]:
        """Get objective directions."""
        return {name: obj["direction"] for name, obj in self.objectives.items()}


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration with optional YAML override.
    
    Args:
        config_path: Optional path to YAML config file
        
    Returns:
        Config instance
    """
    config = Config()
    
    # Override with YAML if provided
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Simple override of top-level attributes
        for key, value in yaml_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config
