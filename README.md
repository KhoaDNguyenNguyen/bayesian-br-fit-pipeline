# Charmonium Decay Analysis Pipeline

This repository contains a comprehensive physics analysis pipeline designed to determine the core model parameters, $m_c$ (the charm quark mass) and $\rho$ (a parameter related to the meson wave function), by fitting to experimental data on charmonium decays. The analysis employs both Bayesian and frequentist statistical methods to ensure robust and cross-validated results.

The pipeline is fully automated, from data loading and model fitting to uncertainty propagation and the generation of publication-quality figures and summary reports.

## Key Features

- **Bayesian Inference**: Utilizes `dynesty` for nested sampling to derive posterior distributions for model parameters, including nuisance parameters from experimental uncertainties.
- **Frequentist Cross-Check**: Implements a profile likelihood analysis to provide an independent, frequentist determination of confidence intervals.
- **Goodness-of-Fit**: Calculates and visualizes pull values for each fitted channel to assess the model's agreement with experimental data.
- **Posterior Predictive Checks (PPC)**: Generates detailed plots comparing the distribution of model predictions against experimental measurements.
- **Uncertainty Decomposition**: Quantifies the contribution of each parameter ($m_c$, $\rho$, and systematic/nuisance parameters) to the total uncertainty of predictions.
- **Automated Plotting**: Produces a full suite of professional, publication-quality plots in `.eps` format using `matplotlib` with LaTeX rendering.
- **Caching for Efficiency**: Implements caching for computationally intensive steps (e.g., predictions, profile likelihoods) to allow for rapid re-analysis and figure generation.
- **Parallel Processing**: Leverages `multiprocessing` to significantly speed up computationally expensive tasks like nested sampling and grid scans.

## Prerequisites

Before running the analysis, you will need:

- Python 3.8+
- A Fortran compiler (e.g., gfortran).
  - On macOS, this can be installed with Homebrew: `brew install gcc`.
  - On Debian/Ubuntu: `sudo apt-get install gfortran`.
- A LaTeX distribution (e.g., MiKTeX, TeX Live, MacTeX) for generating publication-quality plots. If not found, the script will gracefully fall back to standard font rendering.

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/KhoaDNguyenNguyen/bayesian-br-fit-pipeline.git
    cd bayesian-br-fit-pipeline
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Compile the Fortran model:**
    The core physics calculations are performed by a Fortran model, which needs to be compiled into a Python-callable module using `numpy.f2py`. Run the following command from the root directory of the project.
    ```bash
    python -m numpy.f2py -c --f77flags="-ffixed-line-length-none" master_combined_for_f2py.f -m fqmodel --f90flags="-mcmodel=large --no-pie"
    ```
    This command creates a file named `fqmodel.cpython-<...>.so` (or similar), which makes the Fortran subroutines available for import within Python.

## Running the Analysis

To execute the entire analysis pipeline, simply run the main script from the root directory:
```bash
python run_full_analysis.py
```
The script will execute all analysis modules in sequence and print progress updates to the console.

## Project Structure
```
.
├── pkl/
│   ├── dynesty_results.pkl         # Cached results from the Bayesian fit
│   ├── ppc_predictions.pkl         # Cached posterior predictive check data
│   ├── profile_mc.pkl              # Cached profile likelihood data for mc
│   ├── profile_rho.pkl             # Cached profile likelihood data for rho
│   └── ...                         # Other cached files
├── analysis_results/
│   ├── figures/eps/                # Output directory for all .eps figures
│   │   ├── Figure1_CornerPlot.eps
│   │   └── ...
│   └── log_data/                   # Output directory for text summaries and logs
│       ├── bayesian_summary.txt
│       ├── frequentist_summary.txt
│       └── ...
├── run_full_analysis.py            # Main driver script to run the entire analysis
├── master_fitting.py               # Core logic for fitting, physics model, and Bayesian setup
├── config.py                       # Central configuration for constants, data, and fit settings
├── master_combined_for_f2py.f      # The core Fortran source code for physics calculations
├── fqmodel.cpython-....so          # Compiled Fortran module (generated after setup)
├── requirements.txt                # Python package dependencies
└── README.md                       # This file
```

## Analysis Workflow

The `run_full_analysis.py` script orchestrates the following four modules:

1.  **Module A: Bayesian Results Analysis**
    - Loads the results from the `dynesty` nested sampling run.
    - Calculates and saves median parameter values and 68% credible intervals.
    - Computes the correlation between $m_c$ and $\rho$.
    - Generates and saves the primary corner plot (`Figure1_CornerPlot.eps`).

2.  **Module B: Goodness-of-Fit and PPC**
    - Calculates the "pull" (deviation in units of sigma) for each observable used in the fit.
    - Generates a summary bar chart of all pulls (`Figure2_PullSummary.eps`).
    - Creates detailed Posterior Predictive Check (PPC) plots for each fitted channel, comparing the model's predicted distribution to the experimental measurement (`Figure_PPC_*.eps`).

3.  **Module C: Predictions and Uncertainty Analysis**
    - Uses the posterior samples to generate predictions for other, non-fitted, decay channels.
    - Generates a summary bar chart of all predicted branching ratios (`Figure5_PredictionSummary.eps`).
    - Performs an uncertainty decomposition to determine the fractional contribution of $m_c$, $\rho$, and nuisance parameters to the total prediction uncertainty.

4.  **Module D: Frequentist Cross-Check**
    - Performs a profile likelihood scan for $m_c$ and $\rho$.
    - Calculates the frequentist 1-sigma confidence intervals based on where $\Delta\chi^2 = 1$.
    - Generates and saves the profile likelihood plots (`Figure6_ProfileLikelihood.eps`).

## Outputs

After a successful run, the `analysis_results/` directory will be populated with:

-   **EPS Figures (`figures/eps/`)**: A set of publication-ready figures visualizing all aspects of the analysis.
-   **Text Summaries (`log_data/`)**: A set of `.txt` files containing detailed numerical results, including:
    -   `bayesian_summary.txt`: Best-fit values, credible intervals, and correlation from the Bayesian fit.
    -   `frequentist_summary.txt`: Confidence intervals from the profile likelihood cross-check.
    -   `gof_pulls.txt`: Pull values for all fitted channels.
    -   `uncertainty_decomposition.txt`: The percentage contribution of different error sources to the final prediction uncertainties.
    -   `analysis_errors.log`: A log of any errors encountered during the run.
