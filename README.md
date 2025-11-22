# Automation and Active Learning for Multi-Objective Optimization of Antibody Formulations

This repository contains the code and data for the paper "Automation and Active Learning for the Multi-Objective Optimization of Antibody Formulations". It provides a machine learning pipeline for optimizing antibody formulations using Bayesian Optimization with BoTorch.

## Overview

The pipeline performs the following steps:
1.  **Data Loading & Preprocessing**: Loads formulation data from Excel, processes array-based measurements, and standardizes features.
2.  **Model Training**: Trains Gaussian Process (GP) models for each objective (Tm, Diffusion, Viscosity) using `SingleTaskGP`.
3.  **Model Validation**: Performs Group K-Fold Cross-Validation to ensure model robustness.
4.  **Optimization**:
    *   **Single-Objective**: Optimizes each target individually using q-Expected Improvement (qEI).
    *   **Multi-Objective**: Optimizes all targets simultaneously using q-Expected Hypervolume Improvement (qEHVI) with a fixed reference point.

## Repository Structure

```
├── config/
│   └── config.py       # Configuration settings (paths, objectives, bounds)
├── data/
│   └── Antibody_...    # Input Excel data
├── outputs/            # Generated models, logs, and candidate spreadsheets
├── src/
│   ├── pipeline.py     # Core optimization logic (BoTorchPipeline)
│   └── utils.py        # Utility functions (data loading, logging)
├── main.py             # Entry point script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the optimization pipeline:

```bash
python main.py
```

You can customize the execution using command-line arguments:

```bash
python main.py --config config/custom_config.yaml --output_dir outputs/experiment_1
```

*   `--config`: Path to a YAML configuration file to override defaults.
*   `--output_dir`: Directory to save results (models, logs, candidates).
*   `--debug`: Enable debug logging.

## Results

The pipeline outputs:
*   **optimization_candidates.xlsx**: Suggested formulation candidates for experimental validation.
*   **models/**: Saved PyTorch/GPyTorch models.
*   **pipeline.log**: Detailed execution log.
