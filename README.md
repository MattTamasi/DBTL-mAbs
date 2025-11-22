     1|# Automation and Active Learning for Multi-Objective Optimization of Antibody Formulations
     2|
     3|![Automation Pipeline](Figure%201%20-%20mAb%20-%20Gemini.png)
     4|
     5|## Authors
     6|
     7|**D. Christopher Radford**#, **Matthew Tamasi**#, **Elena Di Mare**, **Adam J. Gormley***
     8|
     9|*Department of Biomedical Engineering, Rutgers, The State University of New Jersey, Piscataway, New Jersey 08854, USA*
    10|
    11|## Paper
    12|
    13|[Read the full paper on ChemRxiv](LINK_TO_PAPER)
    14|
    15|## Abstract
    16|
    17|Over the last forty years, biologics such as monoclonal antibodies have become an increasingly important therapeutic agent in the treatment of numerous diseases. Between 1986 and 2025, over 200 antibody-based treatments have been approved globally, most of which are manufactured as preformulated solutions for subsequent administration to patients. However, bioformulation of complex proteins is a difficult engineering challenge; formulations must be tailored to individual therapies, necessitating time- and material-intensive campaigns to select a combination of excipients to simultaneously optimize an array of design criteria. These many interacting additives complicate formulation design with unintuitive and non-linear relationships, thus creating a vast and multidimensional design space that is intrinsically difficult to navigate through and optimize using traditional techniques. To address this challenge, we investigated a high-throughput discovery pipeline using machine learning to model and predict formulation behavior of Generally Recognized as Safe (GRAS) excipients on a model antibody. This was supported by automation-assisted “on-demand” formulation to produce dozens of uniquely formulated antibody solutions with high reproducibility for downstream evaluation and biophysical characterization. This pipeline was then integrated into an iterative closed-loop cycle of automated Design-Build-Test-Learn (DBTL), where new rounds of experiments are designed by the ML model. The process yielded both optimized formulations as well as highly accurate predictive models of formulation behavior. This validates the utility of this technique to both map the underlying property-function landscape and effectively guide formulation development while balancing multiple competing design requirements.
    18|
    19|---
    20|
    21|This repository contains the code and data for the paper "Automation and Active Learning for the Multi-Objective Optimization of Antibody Formulations". It provides a machine learning pipeline for optimizing antibody formulations using Bayesian Optimization with BoTorch.
    22|
    23|## Overview
    24|
    25|The pipeline performs the following steps:
    26|1.  **Data Loading & Preprocessing**: Loads formulation data from Excel, processes array-based measurements, and standardizes features.
    27|2.  **Model Training**: Trains Gaussian Process (GP) models for each objective (Tm, Diffusion, Viscosity) using `SingleTaskGP`.
    28|3.  **Model Validation**: Performs Group K-Fold Cross-Validation to ensure model robustness.
    29|4.  **Optimization**:
    30|    *   **Single-Objective**: Optimizes each target individually using q-Expected Improvement (qEI).
    31|    *   **Multi-Objective**: Optimizes all targets simultaneously using q-Expected Hypervolume Improvement (qEHVI) with a fixed reference point.
    32|5.  **Explainability**:
    33|    *   **SHAP Analysis**: Generates SHAP values and plots to explain model predictions in real units.
    34|
    35|## Repository Structure
    36|
    37|```
    38|├── config/
    39|│   └── config.py       # Configuration settings (paths, objectives, bounds)
    40|├── data/
    41|│   └── Antibody_...    # Input Excel data
    42|├── outputs/            # Generated models, logs, candidates, and SHAP results
    43|├── src/
    44|│   ├── pipeline.py     # Core optimization logic (BoTorchPipeline)
    45|│   ├── utils.py        # Utility functions (data loading, logging)
    46|│   └── shap_utils.py   # SHAP analysis and visualization utilities
    47|├── main.py             # Entry point script
    48|├── generate_shap_analysis.py # Script to run SHAP analysis on trained models
    49|├── requirements.txt    # Python dependencies
    50|└── README.md           # This file
    51|```
    52|
    53|## Installation
    54|
    55|We recommend using `uv` for fast and reliable dependency management.
    56|
    57|1.  Clone the repository.
    58|2.  Install `uv` (if not already installed):
    59|    ```bash
    60|    pip install uv
    61|    ```
    62|3.  Create a virtual environment and install dependencies:
    63|    ```bash
    64|    # Create virtual environment
    65|    uv venv
    66|
    67|    # Install dependencies
    68|    uv pip install -r requirements.txt
    69|    ```
    70|
    71|Alternatively, you can use standard pip:
    72|```bash
    73|pip install -r requirements.txt
    74|```
    75|
    76|## Usage
    77|
    78|To run the optimization pipeline using the virtual environment:
    79|
    80|**Windows:**
    81|```bash
    82|.venv\Scripts\python main.py
    83|```
    84|
    85|**macOS/Linux:**
    86|```bash
    87|.venv/bin/python main.py
    88|```
    89|
    90|### SHAP Analysis
    91|
    92|To generate SHAP analysis for trained models:
    93|
    94|```bash
    95|.venv\Scripts\python generate_shap_analysis.py
    96|```
    97|
    98|### Customization
    99|
   100|You can customize the execution using command-line arguments:
   101|
   102|```bash
   103|.venv\Scripts\python main.py --config config/custom_config.yaml --output_dir outputs/experiment_1
   104|```
   105|
   106|*   `--config`: Path to a YAML configuration file to override defaults.
   107|*   `--output_dir`: Directory to save results (models, logs, candidates).
   108|*   `--debug`: Enable debug logging.
   109|
   110|## Results
   111|
   112|The pipeline outputs:
   113|*   **optimization_candidates.xlsx**: Suggested formulation candidates for experimental validation.
   114|*   **models/**: Saved PyTorch/GPyTorch models.
   115|*   **pipeline.log**: Detailed execution log.
   116|*   **shap_results/**: SHAP values (CSV/Pickle) and summary plots.
   117|