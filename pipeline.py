#!/usr/bin/env python
"""
Legacy BoTorch Pipeline - reverts to original notebook approaches.
Modified to use basic SingleTaskGP, Group K-fold CV, and fixed reference point.
"""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold  # CHANGE: Revert back to Group K-fold as it was in original
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# BoTorch imports
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

# GPyTorch imports
from gpytorch.mlls import ExactMarginalLogLikelihood

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from src import utils


class DataProcessor:
    """Handles data loading, preprocessing, and scaling."""
    def __init__(self, config: Config):
        self.config = config
        self.device = config.torch_device
        self.dtype = torch.float64
        
        # Objective-specific X scalers instead of a single global scaler
        self.X_scalers = {}  # Dictionary of scalers per objective
        self.y_scalers = {}  # Individual scalers per objective
        self.datasets = {}  # Store processed datasets for multi-objective access
        
    def load_and_process_data(self) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, np.ndarray]]:
        """Load and process data for all objectives."""
        logging.info("Loading and processing data...")
        
        # Load raw data
        df = utils.load_excel_data(self.config.data_file)
        
        # Process each objective
        datasets = {}
        formulation_ids = {}
        
        for obj_name, obj_config in self.config.objectives.items():
            logging.info(f"Processing {obj_name} data...")
            
            # Expand array data and filter - use the raw column names from data file
            source_target_col = obj_config["value"]  # e.g., "tm", "diff", "visc"
            source_std_col = obj_config["std"]       # e.g., "tm_std", "diff_std", "visc_std"
            source_conc_col = obj_config["concentration"]  # e.g., "concentration_tm"
            
            expanded_df = utils.expand_array_data(
                df, source_target_col, 
                source_std_col, 
                source_conc_col
            )
            
            if expanded_df.empty:
                logging.warning(f"No data found for {obj_name}")
                continue
            
            # Extract features and targets
            X, y, form_ids = self._extract_features_targets(expanded_df, obj_config)
            
            if X.shape[0] < self.config.min_samples:
                logging.warning(f"Insufficient samples for {obj_name}: {X.shape[0]}")
                continue
            
            datasets[obj_name] = {"X_raw": X, "y_raw": y}
            formulation_ids[obj_name] = form_ids
            
            # Create objective-specific X scaler
            self.X_scalers[obj_name] = MinMaxScaler()
            self.X_scalers[obj_name].fit(X)
            logging.info(f"  {obj_name}: {X.shape[0]} samples, {X.shape[1]} features, created objective-specific scaler")
        
        # Scale datasets and convert to tensors
        for obj_name in datasets.keys():
            X_scaled, y_scaled = self._scale_and_tensorize(datasets[obj_name], obj_name)
            datasets[obj_name]["X"] = X_scaled
            datasets[obj_name]["y"] = y_scaled
            self.datasets[obj_name] = datasets[obj_name]  # Store for multi-objective access
        
        return datasets, formulation_ids
    
    def load_and_process_data_with_metadata(self) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
        """Load and process data for all objectives. Includes metadata (Form IDs, Gen, and Obj) for analysis.
        Added by Chris on 2025-6-22."""
        logging.info("Loading and processing data + metadata TEST...")
        
        # Load raw data
        df = utils.load_excel_data(self.config.data_file)
        
        # Process each objective
        datasets = {}
        formulation_ids = {}
        formulation_metadata = {}
        
        for obj_name, obj_config in self.config.objectives.items():
            logging.info(f"Processing {obj_name} data...")
            
            # Expand array data and filter - use the raw column names from data file
            source_target_col = obj_config["value"]  # e.g., "tm", "diff", "visc"
            source_std_col = obj_config["std"]       # e.g., "tm_std", "diff_std", "visc_std"
            source_conc_col = obj_config["concentration"]  # e.g., "concentration_tm"
            
            expanded_df = utils.expand_array_data(
                df, source_target_col, 
                source_std_col, 
                source_conc_col
            )
            
            if expanded_df.empty:
                logging.warning(f"No data found for {obj_name}")
                continue
            
            # Extract features and targets
            X, y, std, form_ids, form_obj, form_gen = self._extract_features_targets_obj_gen(expanded_df, obj_config)

            if X.shape[0] < self.config.min_samples:
                logging.warning(f"Insufficient samples for {obj_name}: {X.shape[0]}")
                continue

            datasets[obj_name] = {"X_raw": X, "y_raw": y, "y_std": std}
            formulation_ids[obj_name] = form_ids
            formulation_metadata[obj_name] = {
                "Formulation ID": form_ids,
                "Objective": form_obj, 
                "Generation": form_gen
            }
            
            # Create objective-specific X scaler
            self.X_scalers[obj_name] = MinMaxScaler()
            self.X_scalers[obj_name].fit(X)
            logging.info(f"  {obj_name}: {X.shape[0]} samples, {X.shape[1]} features, created objective-specific scaler")
        
        # Scale datasets and convert to tensors
        for obj_name in datasets.keys():
            X_scaled, y_scaled = self._scale_and_tensorize(datasets[obj_name], obj_name)
            datasets[obj_name]["X"] = X_scaled
            datasets[obj_name]["y"] = y_scaled
            self.datasets[obj_name] = datasets[obj_name]  # Store for multi-objective access
            

        return datasets, formulation_ids, formulation_metadata

    
    def _extract_features_targets(self, df: pd.DataFrame, obj_config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract features and targets from dataframe."""
        # Features
        X = df[self.config.feature_columns].values
        
        # Target values (only mean, not std)
        y = df[obj_config["target_column"]].values.reshape(-1, 1)
        
        # Formulation IDs for grouping
        formulation_ids = df["Formulation ID"].values
        
        return X, y, formulation_ids
    
    def _extract_features_targets_obj_gen(self, df: pd.DataFrame, obj_config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract features and targets from dataframe. Added by Chris on 2025-6-21."""
        # Features
        X = df[self.config.feature_columns].values
        
        # Get std directly from obj_config, which comes from config.objectives
        std = df[obj_config["std_column"]].values.reshape(-1, 1)

        # Target values (only mean, not std)
        y = df[obj_config["target_column"]].values.reshape(-1, 1)
        
        # Formulation IDs for grouping
        formulation_ids = df["Formulation ID"].values

        # Objectives for grouping
        formulation_obj = df["Objective"].values

        # Generation for grouping
        formulation_gen = df["Generation"].values
        
        return X, y, std, formulation_ids, formulation_obj, formulation_gen

    def _scale_and_tensorize(self, dataset: Dict, obj_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale features and targets, convert to tensors."""
        X_raw = dataset["X_raw"]
        y_raw = dataset["y_raw"]
        
        # Scale features using objective-specific scaler
        X_scaled = self.X_scalers[obj_name].transform(X_raw)
        
        # Scale targets individually per objective (only target values, not std)
        self.y_scalers[obj_name] = MinMaxScaler()
        y_scaled = self.y_scalers[obj_name].fit_transform(y_raw)
        
        # Convert to tensors
        X_tensor = torch.tensor(X_scaled, dtype=self.dtype, device=self.device)
        y_tensor = torch.tensor(y_scaled, dtype=self.dtype, device=self.device)
        
        # Ensure proper tensor dimensions
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(-1)
        
        return X_tensor, y_tensor
    def get_multi_objective_targets(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get aligned multi-objective targets using simplified alignment approach.
        Returns aligned targets for common formulations across datasets.
        
        Note: For multi-objective optimization, we choose one reference objective
        for X scaling to ensure consistent feature space.
        """
        objective_names = self.config.objective_names
        
        # Check that we have data for all objectives
        available_datasets = {name: self.datasets[name] for name in objective_names 
                            if name in self.datasets and self.datasets[name]}
        
        if len(available_datasets) == 0:
            logging.warning("No datasets available for multi-objective alignment.")
            return {}, {'X_scaled_aligned': torch.empty((0,0)), 'formulation_ids_aligned': torch.empty(0)}
        
        # Find minimum sample count across all datasets
        sample_counts = {name: data['X'].shape[0] for name, data in available_datasets.items()}
        min_samples = min(sample_counts.values())
        
        if min_samples == 0:
            logging.warning("All datasets are empty for multi-objective alignment.")
            return {}, {'X_scaled_aligned': torch.empty((0,0)), 'formulation_ids_aligned': torch.empty(0)}
        
        logging.info(f"Using simplified alignment with {min_samples} samples (minimum across datasets)")
        
        # Use the first min_samples from each dataset
        aligned_Y_targets_dict = {}
        X_aligned = None
        formulation_ids_aligned = None
        reference_obj = None
        
        # First, identify reference objective for feature space
        for obj_name in objective_names:
            if obj_name in available_datasets:
                reference_obj = obj_name
                break
                
        if reference_obj is None:
            logging.warning("Could not identify reference objective for multi-objective alignment.")
            return {}, {'X_scaled_aligned': torch.empty((0,0)), 'formulation_ids_aligned': torch.empty(0)}
            
        logging.info(f"Using {reference_obj} as reference objective for feature scaling in multi-objective")
        
        # Now process each objective
        for obj_name in objective_names:
            if obj_name not in available_datasets:
                continue
                
            data = available_datasets[obj_name]
            
            # Take first min_samples - only the target column (first column)
            y_subset = data['y'][:min_samples]  # Already properly shaped from scaling fix
            aligned_Y_targets_dict[obj_name] = y_subset
            
            # Use X from the reference objective for consistency
            if X_aligned is None and obj_name == reference_obj:
                X_aligned = data['X'][:min_samples]
                
                # Create dummy formulation IDs
                formulation_ids_aligned = torch.arange(min_samples, dtype=torch.int64)
                
                # Store the reference objective for use in optimization
                self.multi_obj_reference = reference_obj
        
        all_formulation_data_dict = {
            'X_scaled_aligned': X_aligned if X_aligned is not None else torch.empty((0,0)),
            'formulation_ids_aligned': formulation_ids_aligned if formulation_ids_aligned is not None else torch.empty(0),
            'reference_obj': reference_obj
        }
        
        logging.info(f"Multi-objective alignment completed:")
        logging.info(f"  X shape: {X_aligned.shape if X_aligned is not None else 'N/A'}")
        logging.info(f"  Number of objectives: {len(aligned_Y_targets_dict)}")
        for name, tensor in aligned_Y_targets_dict.items():
            logging.info(f"  {name} Y shape: {tensor.shape}")
        
        return aligned_Y_targets_dict, all_formulation_data_dict
    def save_candidates_to_excel(self, candidates: Dict[str, torch.Tensor], output_path: str, 
                               models: Dict = None, y_scalers: Dict = None) -> pd.DataFrame:
        """Save optimization candidates to Excel with proper concentration handling."""
        all_candidates = []
        
        for opt_type, cand_tensor in candidates.items():
            if cand_tensor.numel() == 0:
                continue
            
            # Inverse transform candidates using the appropriate X scaler
            if opt_type == 'multi_objective' and hasattr(self, 'multi_obj_reference'):
                # For multi-objective use the reference objective's scaler
                reference_obj = self.multi_obj_reference
                if reference_obj in self.X_scalers:
                    scaler = self.X_scalers[reference_obj]
                    cand_unscaled = scaler.inverse_transform(cand_tensor.detach().numpy())
                    logging.info(f"Using {reference_obj}'s scaler for multi-objective candidates")
                else:
                    cand_unscaled = cand_tensor.detach().numpy()
                    logging.warning(f"No scaler found for reference objective {reference_obj}")
            elif opt_type in self.X_scalers:
                # Use objective-specific scaler
                scaler = self.X_scalers[opt_type]
                cand_unscaled = scaler.inverse_transform(cand_tensor.detach().numpy())
            else:
                cand_unscaled = cand_tensor.detach().numpy()
                logging.warning(f"No scaler found for {opt_type}, using raw values")
            
            # Create DataFrame
            cand_df = pd.DataFrame(cand_unscaled, columns=self.config.feature_columns)
            cand_df["Optimization_Type"] = opt_type
              # SET CORRECT CONCENTRATION for display based on objective type using config.pred_conc
            concentration_col_name = 'Concentration (mg/mL)'
            
            if opt_type in self.config.pred_conc:
                # Use concentration value from configuration
                pred_concentration = self.config.pred_conc[opt_type]
                cand_df[concentration_col_name] = pred_concentration
                logging.debug(f"Setting display concentration to {pred_concentration} mg/mL for {opt_type} (from config)")
            elif opt_type == 'multi_objective' and hasattr(self, 'multi_obj_reference') and self.multi_obj_reference in self.config.pred_conc:
                # For multi-objective, use the reference objective's concentration from config
                pred_concentration = self.config.pred_conc[self.multi_obj_reference]
                cand_df[concentration_col_name] = pred_concentration
                logging.debug(f"Setting display concentration to {pred_concentration} mg/mL for multi-objective (using {self.multi_obj_reference}'s config)")
            else:
                # Fallback to defaults if not in config
                if opt_type in ['tm', 'diffusion']:
                    cand_df[concentration_col_name] = 15.0
                    logging.debug(f"Setting display concentration to 15 mg/mL for {opt_type} (default)")
                elif opt_type == 'viscosity':
                    cand_df[concentration_col_name] = 120.0
                    logging.debug(f"Setting display concentration to 120 mg/mL for {opt_type} (default)")
                elif opt_type == 'multi_objective':
                    # For multi-objective, use 15 mg/mL by default
                    cand_df[concentration_col_name] = 15.0
                    logging.debug(f"Setting display concentration to 15 mg/mL for multi-objective (default)")
                else:
                    logging.warning(f"No concentration specified for {opt_type} in config.pred_conc")
            
            # Add predictions if models available
            if models and y_scalers:
                self._add_predictions(cand_df, cand_tensor, models, y_scalers, opt_type)
            
            all_candidates.append(cand_df)
        
        if all_candidates:
            final_df = pd.concat(all_candidates, ignore_index=True)
        
            # Post-process pH and Buffer as in original implementation
            if 'pH' in final_df.columns:
                final_df['pH'] = (final_df['pH'] * 2).round() / 2
                
            if 'Buffer' in final_df.columns and 'pH' in final_df.columns:
                final_df['Buffer'] = final_df['Buffer'].round()
                conditions = [
                    (final_df['pH'] >= 6),
                    (final_df['pH'] == 5.5) | (final_df['pH'] == 4.5)
                ]
                choices = [2, 1]
                final_df['Buffer'] = np.select(conditions, choices, default=final_df['Buffer'])
            elif 'Buffer' in final_df.columns:
                final_df['Buffer'] = final_df['Buffer'].round()
            
            final_df.to_excel(output_path, index=False)
            logging.info(f"[OK] Candidates saved with proper concentration settings")
            return final_df
        
        return pd.DataFrame()
    def _add_predictions(self, df: pd.DataFrame, X_scaled: torch.Tensor, 
                        models: Dict, y_scalers: Dict, source_obj_type: str = None) -> None:
        """Add model predictions to candidate DataFrame with proper concentration handling."""
        # Define prediction columns
        pred_cols = {
            'tm': {'mean': 'Predicted Tm (C)', 'std': 'Predicted Tm Std (C)'},
            'diffusion': {'mean': 'Predicted Diffusion (cm²/s)', 'std': 'Predicted Diffusion Std (cm²/s)'},
            'viscosity': {'mean': 'Predicted Viscosity (cP)', 'std': 'Predicted Viscosity Std (cP)'}
        }
        
        for obj_name, model in models.items():
            if model and obj_name in y_scalers:
                try:
                    # Determine which scaler to use for inverse transform
                    if source_obj_type == 'multi_objective' and hasattr(self, 'multi_obj_reference'):
                        # For multi-objective candidates, use reference objective scaler
                        reference_obj = self.multi_obj_reference
                        if reference_obj in self.X_scalers:
                            inverse_scaler = self.X_scalers[reference_obj]
                        else:
                            logging.warning(f"No scaler found for reference objective {reference_obj}")
                            continue
                    elif source_obj_type in self.X_scalers:
                        # For single objective candidates use source objective scaler
                        inverse_scaler = self.X_scalers[source_obj_type]
                    else:
                        logging.warning(f"No scaler found for source type {source_obj_type}")
                        continue
                    
                    # Get the candidates in original scale for concentration adjustment
                    candidates_original = inverse_scaler.inverse_transform(X_scaled.detach().numpy())
                      # Create prediction features with CORRECT CONCENTRATION for this objective
                    prediction_features = candidates_original.copy()
                    concentration_idx = self.config.feature_columns.index('Concentration (mg/mL)')
                    
                    # Set the correct concentration for each objective from config.pred_conc
                    if obj_name in self.config.pred_conc:
                        # Use concentration value from configuration
                        pred_concentration = self.config.pred_conc[obj_name]
                        prediction_features[:, concentration_idx] = pred_concentration
                        logging.debug(f"Using {pred_concentration} mg/mL concentration for {obj_name} predictions (from config)")
                    else:
                        # Fallback to defaults if not in config
                        if obj_name in ['tm', 'diffusion']:
                            prediction_features[:, concentration_idx] = 15.0
                            logging.debug(f"Using default 10 mg/mL concentration for {obj_name} predictions (not in config)")
                        elif obj_name == 'viscosity':
                            prediction_features[:, concentration_idx] = 120.0
                            logging.debug(f"Using default 120 mg/mL concentration for {obj_name} predictions (not in config)")
                        else:
                            logging.warning(f"No concentration specified for {obj_name} in config.pred_conc")
                    
                    # Scale the features with correct concentration using the objective-specific scaler
                    if obj_name in self.X_scalers:
                        obj_scaler = self.X_scalers[obj_name]
                        prediction_scaled_np = obj_scaler.transform(prediction_features)
                        prediction_scaled = torch.tensor(prediction_scaled_np, dtype=torch.float64)
                    else:
                        logging.warning(f"No X scaler found for {obj_name}, skipping prediction")
                        continue
                    
                    # Make predictions
                    model.eval()
                    with torch.no_grad():
                        posterior = model.posterior(prediction_scaled)
                        pred_mean_scaled = posterior.mean.cpu().numpy().reshape(-1)
                        pred_var_scaled = posterior.variance.cpu().numpy().reshape(-1)
                    
                    # Apply inverse transform
                    scaler = y_scalers[obj_name]
                    pred_mean_array = pred_mean_scaled.reshape(-1, 1)
                    
                    # Flag potential zero-range issue for diffusion
                    if obj_name == "diffusion" and hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
                        data_min = scaler.data_min_[0]
                        data_max = scaler.data_max_[0]
                        range_val = data_max - data_min
                        
                        if range_val < 1e-10:
                            # Zero range - use reasonable diffusion coefficient range
                            logging.warning(f"Extremely small range in diffusion data: {range_val}. Predictions may be unreliable.")
                                                                    
                    pred_mean_real = scaler.inverse_transform(pred_mean_array).flatten()
                    scale_factor = scaler.scale_[0] if hasattr(scaler, 'scale_') else 1.0
                    pred_std_real = np.sqrt(pred_var_scaled) / scale_factor
                    
                    # Add predictions to DataFrame
                    if obj_name in pred_cols:
                        df[pred_cols[obj_name]['mean']] = pred_mean_real
                        df[pred_cols[obj_name]['std']] = pred_std_real
                        
                except Exception as e:
                    logging.warning(f"Prediction failed for {obj_name}: {e}")
                    if obj_name in pred_cols:
                        df[pred_cols[obj_name]['mean']] = None
                        df[pred_cols[obj_name]['std']] = None


class ModelTrainer:
    """Handles GP model training and validation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.torch_device
        self.dtype = torch.float64
        self.models = {}
        self.model_quality_results = {}
    
    def train_models(self, datasets: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, SingleTaskGP]:
        """Train GP models for each objective."""
        logging.info("Training GP models...")
        
        models = {}
        for obj_name, data in datasets.items():
            if not data or data["X"].shape[0] < self.config.min_samples:
                logging.warning(f"Insufficient data for {obj_name} ({data['X'].shape[0]} samples)")
                continue
            
            model = self._train_single_model(data["X"], data["y"], obj_name)
            if model:
                models[obj_name] = model
                logging.info(f"Successfully trained {obj_name} model")
        
        self.models = models
        return models
    
    def _train_single_model(self, X: torch.Tensor, y: torch.Tensor, obj_name: str) -> Optional[SingleTaskGP]:
        """Train single GP model with BASIC configuration (LEGACY MODE)."""
        try:
            logging.info(f"Training model for {obj_name}...")
            logging.info(f"  Data shape for {obj_name}: X={X.shape}, y_target={y.shape}")
            
            # Log target statistics for debugging
            y_stats = {
                "mean": y.mean().item(),
                "std": y.std().item(), 
                "min": y.min().item(),
                "max": y.max().item()
            }
            logging.info(f"  y_target for {obj_name} (first 5): {y[:5].flatten().tolist()}")
            logging.info(f"  y_target for {obj_name} (stats): mean={y_stats['mean']:.4f}, std={y_stats['std']:.4f}, min={y_stats['min']:.4f}, max={y_stats['max']:.4f}")
            
            # CHANGE 1: BASIC SingleTaskGP configuration (no sophisticated kernels)
            model = SingleTaskGP(
                X, y,
                outcome_transform=Standardize(m=y.shape[-1]),
                input_transform=Normalize(d=X.shape[-1])
                # Removed: covar_module, adaptive noise constraints, priors
            ).to(device=self.device, dtype=self.dtype)
            
            # Fit model
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_gpytorch_model(mll)
            
            # Validate model quality
            quality_result = self._validate_model_quality(model, X, y, obj_name)
            self.model_quality_results[obj_name] = quality_result
            
            if quality_result["valid"]:
                return model
            else:
                logging.warning(f"Model {obj_name} failed quality validation")
                return None
                
        except Exception as e:
            logging.error(f"Error training {obj_name} model: {e}")
            return None
    
    def _validate_model_quality(self, model, X: torch.Tensor, y: torch.Tensor, obj_name: str) -> Dict[str, Any]:
        """Validate trained model quality."""
        try:
            with torch.no_grad():
                pred = model.posterior(X).mean
                r2 = r2_score(y.numpy(), pred.numpy())
                
                valid = r2 >= self.config.min_r2
                return {"valid": valid, "r2": r2}
        except:
            return {"valid": False, "r2": 0.0}
    
    def cross_validate(self, datasets: Dict[str, Dict[str, torch.Tensor]], 
                      formulation_ids: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Perform cross-validation on models using GROUP K-FOLD (REVERTED FROM BASIC K-FOLD)."""
        cv_results = {}
        
        for obj_name, data in datasets.items():
            if obj_name not in formulation_ids or data["X"].shape[0] < self.config.cv_folds:
                continue
            
            # CHANGE REVERTED: Back to Group K-fold as it was in original notebook
            gkf = GroupKFold(n_splits=self.config.cv_folds)
            groups = formulation_ids[obj_name]
            
            # Randomize group assignment
            unique_groups = np.unique(groups)
            rng = np.random.default_rng(self.config.random_state)
            shuffled_groups = rng.permutation(unique_groups)
            group_mapping = {old: new for new, old in enumerate(shuffled_groups)}
            mapped_groups = np.array([group_mapping[g] for g in groups])

            r2_scores = []
            for train_idx, test_idx in gkf.split(data["X"], data["y"], mapped_groups):  # With groups parameter
                X_train, X_test = data["X"][train_idx], data["X"][test_idx]
                y_train, y_test = data["y"][train_idx], data["y"][test_idx]
                
                # Train fold model
                fold_model = self._train_single_model(X_train, y_train, f"{obj_name}_fold")
                if fold_model:
                    with torch.no_grad():
                        pred = fold_model.posterior(X_test).mean
                        r2 = r2_score(y_test.numpy(), pred.numpy())
                        r2_scores.append(r2)
            
            cv_results[obj_name] = {
                "r2_scores": r2_scores,
                "mean_r2": np.mean(r2_scores) if r2_scores else 0.0,
                "std_r2": np.std(r2_scores) if r2_scores else 0.0
            }
            
            logging.info(f"{obj_name} CV: R² = {cv_results[obj_name]['mean_r2']:.3f} ± {cv_results[obj_name]['std_r2']:.3f}")
        
        return cv_results
    
    def cross_validate_with_preds(self, datasets: Dict[str, Dict[str, torch.Tensor]], 
                  formulation_ids: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Chris Method: Perform cross-validation on models using GROUP K-FOLD and return predictions and scores."""
        cv_results = {}
    
        for obj_name, data in datasets.items():
            if obj_name not in formulation_ids or data["X"].shape[0] < self.config.cv_folds:
                continue
            
            gkf = GroupKFold(n_splits=self.config.cv_folds)
            groups = formulation_ids[obj_name]

            # Randomize group assignment
            unique_groups = np.unique(groups)
            rng = np.random.default_rng(self.config.random_state)
            shuffled_groups = rng.permutation(unique_groups)
            group_mapping = {old: new for new, old in enumerate(shuffled_groups)}
            mapped_groups = np.array([group_mapping[g] for g in groups])
            
            r2_scores = []
            mse_scores = []
            fold_preds = []
            fold_truth = []

            for train_idx, test_idx in gkf.split(data["X"], data["y"], mapped_groups):
                X_train, X_test = data["X"][train_idx], data["X"][test_idx]
                y_train, y_test = data["y"][train_idx], data["y"][test_idx]
                
                fold_model = self._train_single_model(X_train, y_train, f"{obj_name}_fold")
                if fold_model:
                    with torch.no_grad():
                        pred = fold_model.posterior(X_test).mean
                        pred_np = pred.numpy().flatten()
                        y_test_np = y_test.numpy().flatten()
                        
                        r2 = r2_score(y_test_np, pred_np)
                        mse = mean_squared_error(y_test_np, pred_np)
                        
                        r2_scores.append(r2)
                        mse_scores.append(mse)
                        fold_preds.append(pred_np)
                        fold_truth.append(y_test_np)

            # Combine all fold predictions
            all_preds = np.concatenate(fold_preds) if fold_preds else np.array([])
            all_truth = np.concatenate(fold_truth) if fold_truth else np.array([])

            cv_results[obj_name] = {
                "r2_scores": r2_scores,
                "mean_r2": np.mean(r2_scores) if r2_scores else 0.0,
                "std_r2": np.std(r2_scores) if r2_scores else 0.0,
                "mse_scores": mse_scores,
                "mean_mse": np.mean(mse_scores) if mse_scores else 0.0,
                "predictions": all_preds,
                "true_values": all_truth
            }
            
            logging.info(
                f"{obj_name} CV: R² = {cv_results[obj_name]['mean_r2']:.3f} ± {cv_results[obj_name]['std_r2']:.3f}, "
                f"MSE = {cv_results[obj_name]['mean_mse']:.3f}"
            )
        
        return cv_results

    
    def create_model_list(self, datasets: Dict[str, Dict[str, torch.Tensor]]) -> Optional[ModelListGP]:
        """Create ModelListGP for multi-objective optimization with active model keys."""
        if len(self.models) < 2:
            logging.warning(f"Need at least 2 models for multi-objective, got {len(self.models)}")
            return None
        
        try:
            # Get models in the order of objective names
            model_list = []
            active_keys = []
            for obj_name in self.config.objective_names:
                if obj_name in self.models:
                    model_list.append(self.models[obj_name])
                    active_keys.append(obj_name)
            
            if len(model_list) < 2:
                logging.warning(f"Insufficient trained models for multi-objective: {len(model_list)}")
                return None
            
            model_list_gp = ModelListGP(*model_list)
            # Add active model keys for optimization engine
            model_list_gp.active_model_keys = active_keys
            
            logging.info(f"Created ModelListGP with {len(model_list)} models: {active_keys}")
            return model_list_gp
            
        except Exception as e:
            logging.error(f"Error creating ModelListGP: {e}")
            return None
    
    def get_best_observed_values(self, datasets: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Get best observed values for each objective."""
        best_values = {}
        for obj_name, data in datasets.items():
            if data and data["y"].numel() > 0:
                direction = self.config.objective_directions[obj_name]
                if direction > 0:  # maximize
                    best_values[obj_name] = data["y"].max().item()
                else:  # minimize
                    best_values[obj_name] = data["y"].min().item()
        return best_values
    
    def save_models(self, output_dir: str) -> None:
        """Save trained models to disk."""
        model_dir = Path(output_dir) / "models"
        model_dir.mkdir(exist_ok=True)
        
        for obj_name, model in self.models.items():
            model_path = model_dir / f"{obj_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logging.info(f"Saved {obj_name} model to {model_path}")


class OptimizationEngine:
    """Handles acquisition function optimization."""
    
    def __init__(self, config: Config):
        self.config = config

    def _get_scaled_bounds(self, data_processor: DataProcessor, objective: str) -> torch.Tensor:
            """
            Return scaled bounds for each feature using objective-specific MinMaxScaler.
            Handles custom Antibody Conc bounds depending on objective.
            """
            lower_bounds = []
            upper_bounds = []

            # Use the objective-specific scaler
            if objective in data_processor.X_scalers:
                scaler = data_processor.X_scalers[objective]
                logging.info(f"Using {objective}-specific scaler for bounds scaling")
            elif hasattr(data_processor, 'multi_obj_reference') and data_processor.multi_obj_reference in data_processor.X_scalers:
                # Fallback to reference objective for multi-objective
                scaler = data_processor.X_scalers[data_processor.multi_obj_reference]
                logging.info(f"Using reference objective {data_processor.multi_obj_reference} scaler for bounds scaling")
            else:
                logging.warning(f"No scaler found for {objective}, cannot scale bounds properly")
                # Return default [0,1] bounds as fallback
                return torch.tensor(
                    [[0.0] * len(self.config.feature_columns), [1.0] * len(self.config.feature_columns)], 
                    dtype=torch.float64, 
                    device=self.config.torch_device
                )            # Create arrays to hold the original bounds for all features
            lower_bound_array = np.zeros(len(self.config.feature_columns))
            upper_bound_array = np.zeros(len(self.config.feature_columns))
            
            # First, populate these arrays with the original bounds
            for feat_idx, feat in enumerate(self.config.feature_columns):
                if feat == "Antibody Conc" or feat == "Concentration (mg/mL)":
                    # For concentration, use max value from the scaler's data range
                    # This ensures it will be scaled to 1.0
                    if hasattr(scaler, 'data_max_') and hasattr(scaler, 'data_min_'):
                        lower_bound_array[feat_idx] = scaler.data_max_[feat_idx]  # Use max value for lower bound too
                        upper_bound_array[feat_idx] = scaler.data_max_[feat_idx]  # Use max value
                else:
                    # For all other features, get bounds from config
                    min_val, max_val = self.config.feature_bounds[feat]
                    lower_bound_array[feat_idx] = min_val
                    upper_bound_array[feat_idx] = max_val

            logging.info(f"Lower bounds before scaling: {lower_bound_array}")
            logging.info(f"Upper bounds before scaling: {upper_bound_array}")
            
            # Now scale the entire arrays using the scaler
            scaled_lower = scaler.transform(lower_bound_array.reshape(1, -1)).flatten()
            scaled_upper = scaler.transform(upper_bound_array.reshape(1, -1)).flatten()
            
            # Override concentration to exactly 1.0 to be safe
            for feat_idx, feat in enumerate(self.config.feature_columns):
                if feat == "Antibody Conc" or feat == "Concentration (mg/mL)":
                    scaled_lower[feat_idx] = 1.0
                    scaled_upper[feat_idx] = 1.0
            
            # Convert to lists for tensor creation
            lower_bounds = scaled_lower.tolist()
            upper_bounds = scaled_upper.tolist()

            bounds_tensor = torch.tensor(
                [lower_bounds, upper_bounds],
                dtype=torch.float64,
                device=self.config.torch_device
            )

            return bounds_tensor
    
    def optimize_single_objectives(self, data_processor: DataProcessor, models: Dict, best_values: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Optimize single objectives using qEI with proper objective direction handling."""
        logging.info("Running single-objective optimization...")
        candidates = {}
        
        for obj_name, model in models.items():
            if not model:
                continue
            
            try:
                logging.info(f"Optimizing for single objective: {obj_name}")
                
                # Get objective direction
                obj_direction = self.config.objective_directions.get(obj_name, 1.0)
                best_f_raw = best_values[obj_name]
                
                # For qEI, which assumes maximization:
                # - For maximization objectives: use the value directly
                # - For minimization objectives: negate the value
                if obj_direction == -1.0:  # Minimization (like viscosity)
                    best_f_for_qei = -best_f_raw
                    logging.info(f"  Target {obj_name} (minimization): best_f for qEI = {best_f_for_qei:.4f} (negated from {best_f_raw:.4f})")
                else:  # Maximization (like tm, diffusion)
                    best_f_for_qei = best_f_raw
                    logging.info(f"  Target {obj_name} (maximization): best_f for qEI = {best_f_for_qei:.4f}")

                # Add safety check for best_f value
                if torch.isnan(torch.tensor(best_f_for_qei)) or torch.isinf(torch.tensor(best_f_for_qei)):
                    logging.warning(f"  Invalid best_f value for {obj_name}: {best_f_for_qei}. Skipping optimization.")
                    continue
                
                # Create acquisition function
                best_f = torch.tensor([best_f_for_qei], dtype=torch.float64)
                sampler = SobolQMCNormalSampler(
                    sample_shape=torch.Size([self.config.mc_samples]),
                    seed=self.config.random_state
                )
                qei = qExpectedImprovement(model, best_f, sampler=sampler)
                
                # Define bounds for optimization using objective-specific scaler
                bounds = self._get_scaled_bounds(data_processor, obj_name)
                logging.info(f"  Using {obj_name}-specific bounds for optimization")
                logging.info(f"  Bounds for {obj_name}: {bounds.tolist()}")
                if bounds is None or bounds.numel() == 0:
                    logging.warning(f"  No valid bounds found for {obj_name}. Skipping optimization.")
                    continue

                # Optimize acquisition function
                candidate, acq_value = optimize_acqf(
                    qei, bounds=bounds, q=self.config.batch_size,
                    num_restarts=self.config.num_restarts, raw_samples=500,
                    options={"batch_limit": 5, "maxiter": 200}
                )
                
                # Validate acquisition value
                if acq_value < -100:
                    logging.warning(f"  Extremely negative acquisition value for {obj_name}: {acq_value:.4f}. This may indicate numerical instability.")
                elif acq_value < -10:
                    logging.info(f"  Large negative acquisition value for {obj_name}: {acq_value:.4f}. Consider checking model quality.")
                
                candidates[obj_name] = candidate
                logging.info(f"  Optimization for {obj_name} completed. Found {candidate.shape[0]} candidates (acq_value: {acq_value:.4f}).")
                
            except Exception as e:
                logging.error(f"Single objective optimization failed for {obj_name}: {e}")
        
        return candidates
    
    def optimize_multi_objective(self, model_list: ModelListGP, data_processor: DataProcessor) -> Optional[torch.Tensor]:
        """
        Optimize multi-objective using qEHVI with FIXED reference point and 1-VISCOSITY transformation (LEGACY MODE).
        """
        if not model_list:
            logging.warning("No ModelListGP provided for multi-objective optimization.")
            return None
        
        logging.info("Starting multi-objective optimization with qEHVI...")
        
        try:
            # Get active model keys - use all objective names if not available
            if hasattr(model_list, 'active_model_keys') and model_list.active_model_keys:
                active_keys = model_list.active_model_keys
            else:
                active_keys = self.config.objective_names
                logging.warning("ModelListGP doesn't have active_model_keys, using all configured objectives")

            logging.info(f"Active model keys for MOO: {active_keys}")

            if not active_keys:
                logging.warning("No active keys available. Skipping multi-objective optimization.")
                return None

            # Get aligned data using the method
            try:
                aligned_Y_targets_dict, all_formulation_data = data_processor.get_multi_objective_targets()
            except Exception as e:
                logging.warning(f"Could not retrieve aligned data from data_processor: {e}. Skipping MOO.")
                return None
            
            if not aligned_Y_targets_dict or not all_formulation_data or 'X_scaled_aligned' not in all_formulation_data:
                logging.warning("Could not retrieve aligned data from data_processor. Skipping MOO.")
                return None
                
            X_baseline = all_formulation_data['X_scaled_aligned'].to(device=self.config.torch_device, dtype=torch.float64)
            
            if X_baseline.shape[0] == 0:
                logging.warning("X_baseline is empty. Skipping MOO.")
                return None

            # Construct Y tensor for active objectives with LEGACY DIRECTION HANDLING
            y_tensors_for_active_models = []
            for key in active_keys:
                if key in aligned_Y_targets_dict:
                    y_tensor = aligned_Y_targets_dict[key].to(device=self.config.torch_device, dtype=torch.float64)
                    if y_tensor.shape[0] != X_baseline.shape[0]:
                        logging.error(f"Sample count mismatch for Y target {key} ({y_tensor.shape[0]}) vs X_baseline ({X_baseline.shape[0]}). Aborting MOO.")
                        return None
                    
                    # CHANGE: Use 1-viscosity transformation instead of negation (LEGACY MODE)
                    # qEHVI assumes ALL objectives are to be maximized
                    obj_direction = self.config.objective_directions.get(key, 1.0)
                    if obj_direction == -1.0:  # Minimization objective (like viscosity)
                        y_tensor = 1 - y_tensor  # Use 1-x transformation instead of negation
                        logging.info(f"  Applied 1-x transformation to {key} targets for multi-objective (minimization -> maximization)")
                    else:
                        logging.info(f"  Using {key} targets as-is for multi-objective (maximization)")
                    
                    y_tensors_for_active_models.append(y_tensor)
                else:
                    logging.error(f"Key {key} not found in aligned_Y_targets_dict. Aborting MOO.")
                    return None

            if not y_tensors_for_active_models:
                logging.warning("No Y data found for active models. Skipping MOO.")
                return None

            train_Y_active = torch.cat(y_tensors_for_active_models, dim=1).to(device=self.config.torch_device, dtype=torch.float64)
            
            logging.info(f"Multi-objective setup: X_baseline {X_baseline.shape}, train_Y_active {train_Y_active.shape}")

            # CHANGE 3: Use FIXED reference point (LEGACY MODE)
            ref_point = torch.tensor(self.config.reference_point, device=self.config.torch_device, dtype=torch.float64)
            
            # Ensure reference point matches number of active objectives
            if ref_point.shape[0] != len(active_keys):
                if ref_point.shape[0] > len(active_keys):
                    ref_point = ref_point[:len(active_keys)]
                else:
                    # Extend with zeros
                    extended_ref = torch.zeros(len(active_keys), device=self.config.torch_device, dtype=torch.float64)
                    extended_ref[:ref_point.shape[0]] = ref_point
                    ref_point = extended_ref

            logging.info(f"Using FIXED reference point: {ref_point.tolist()}")

            # Configure sampler
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.config.mc_samples]))
            
            # Create partitioning for qEHVI
            partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_Y_active)
              # Use basic qEHVI (removed sophisticated partitioning for legacy mode)
            qehvi = qExpectedHypervolumeImprovement(
                model=model_list,
                ref_point=ref_point.tolist(),
                partitioning=partitioning,  # Required argument
                sampler=sampler
            )
            
            # Optimize acquisition function using reference objective's scaler
            if hasattr(data_processor, 'multi_obj_reference'):
                reference_obj = data_processor.multi_obj_reference
                bounds = self._get_scaled_bounds(data_processor, reference_obj)
                logging.info(f"Using {reference_obj} as reference for multi-objective bounds")
                logging.info(f"  Bounds for multi-objective optimization: {bounds.tolist()}")
            else:
                # Fallback to first active key if no reference is set
                bounds = self._get_scaled_bounds(data_processor, active_keys[0])
                logging.info(f"No reference objective set, using {active_keys[0]} for multi-objective bounds")
            
            candidates, acq_value = optimize_acqf(
                acq_function=qehvi,
                bounds=bounds,
                q=self.config.batch_size,
                num_restarts=self.config.num_restarts,
                raw_samples=500,
                options={"batch_limit": 5, "maxiter": 200}
            )
            
            logging.info(f"Generated {candidates.shape[0]} multi-objective candidates using qEHVI (acq_value: {acq_value:.4f}).")
            return candidates
            
        except Exception as e:
            logging.error(f"Error during qEHVI optimization: {e}", exc_info=True)
            return None


class BoTorchPipeline:
    """Main pipeline orchestrator for BoTorch optimization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = None
        self.setup_logging()
        
        # Initialize components
        self.data_processor = DataProcessor(config)
        self.model_trainer = ModelTrainer(config)
        self.optimization_engine = OptimizationEngine(config)
        
        # Pipeline state
        self.datasets = {}
        self.formulation_ids = {}
        self.models = {}
        self.model_list = None
        self.cv_results = {}
        self.best_values = {}
        self.optimization_results = {}
    
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Configure logging
        log_file = output_dir / "pipeline.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete optimization pipeline."""
        logging.info("="*60)
        logging.info("STARTING LEGACY BOTORCH OPTIMIZATION PIPELINE")
        logging.info("="*60)
        
        results = {
            "success": False,
            "datasets": {},
            "models": {},
            "cv_results": {},
            "optimization_results": {}
        }
        
        try:
            # 1. Load and process data
            logging.info("Step 1: Loading and processing data...")
            self.datasets, self.formulation_ids = self.data_processor.load_and_process_data()
            
            if not self.datasets:
                logging.error("No datasets loaded. Exiting pipeline.")
                return results
            
            results["datasets"] = {name: {"samples": data["X"].shape[0], "features": data["X"].shape[1]} 
                                 for name, data in self.datasets.items()}
            
            # 2. Train models
            logging.info("Step 2: Training GP models...")
            self.models = self.model_trainer.train_models(self.datasets)
            
            if not self.models:
                logging.error("No models trained successfully. Exiting pipeline.")
                return results
            
            results["models"] = {name: "trained" for name in self.models.keys()}
            
            # 3. Cross-validate models
            logging.info("Step 3: Cross-validating models...")
            self.cv_results = self.model_trainer.cross_validate(self.datasets, self.formulation_ids)
            results["cv_results"] = self.cv_results
            
            # 4. Get best observed values
            self.best_values = self.model_trainer.get_best_observed_values(self.datasets)
            
            # 5. Single-objective optimization
            logging.info("Step 4: Running single-objective optimization...")
            single_obj_candidates = self.optimization_engine.optimize_single_objectives(
                self.data_processor, self.models, self.best_values
            )
            self.optimization_results.update(single_obj_candidates)
            
            # 6. Multi-objective optimization
            logging.info("Step 5: Setting up multi-objective optimization...")
            if len(self.models) >= 2:
                self.model_list = self.model_trainer.create_model_list(self.datasets)
                if self.model_list:
                    multi_obj_candidates = self.optimization_engine.optimize_multi_objective(
                        self.model_list, self.data_processor
                    )
                    if multi_obj_candidates is not None:
                        self.optimization_results["multi_objective"] = multi_obj_candidates
                        logging.info("Multi-objective optimization completed successfully.")
                    else:
                        logging.warning("Multi-objective optimization failed.")
                else:
                    logging.warning("Could not create ModelListGP for multi-objective optimization.")
            else:
                logging.warning(f"Need at least 2 models for multi-objective optimization, got {len(self.models)}")
            
            results["optimization_results"] = {
                name: f"{tensor.shape[0]} candidates" for name, tensor in self.optimization_results.items()
            }
            
            # 7. Save results
            if self.config.save_candidates and self.optimization_results:
                output_path = Path(self.config.output_dir) / "optimization_candidates.xlsx"
                candidates_df = self.data_processor.save_candidates_to_excel(
                    self.optimization_results, str(output_path), self.models, self.data_processor.y_scalers
                )
                logging.info(f"Saved {len(candidates_df)} candidates to {output_path}")
            
            if self.config.save_models:
                self.model_trainer.save_models(self.config.output_dir)
            
            results["success"] = True
            logging.info("Pipeline completed successfully!")
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}", exc_info=True)
            results["error"] = str(e)
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline execution summary."""
        summary = {
            "datasets": len(self.datasets),
            "models_trained": len(self.models),
            "optimization_candidates": {
                name: tensor.shape[0] for name, tensor in self.optimization_results.items()
            }
        }
        
        if self.cv_results:
            summary["cv_performance"] = {
                name: f"R² = {results['mean_r2']:.3f} ± {results['std_r2']:.3f}"
                for name, results in self.cv_results.items()
            }
        
        return summary 