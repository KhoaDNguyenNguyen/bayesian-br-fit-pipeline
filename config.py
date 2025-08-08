import numpy as np
import math

# =============================================================================
# 1. PHYSICAL CONSTANTS
# =============================================================================
PHYSICAL_CONSTANTS = {
    "alpEM": 1.0 / 137.036,
    "ec": 2.0 / 3.0,
    "pi": math.pi,
}

# =============================================================================
# 2. PARTICLE PROPERTIES (MASSES AND TOTAL WIDTHS)
# =============================================================================
# Data sourced from PDG 2024.
hc_width_error = math.sqrt((0.255e-3)**2 + (0.12e-3)**2)
PARTICLE_PROPERTIES = {
    'Jpsi':  {'mass': {'mean': 3096.9e-3,   'error': 0.006e-3},
              'width':{'mean': 92.6e-6,    'error': 1.7e-6}},
    'etac':  {'mass': {'mean': 2984.1e-3,   'error': 0.04e-3},
              'width':{'mean': 30.5e-3,    'error': 0.5e-3}},
    'psi2S': {'mass': {'mean': 3686.097e-3, 'error': 0.011e-3},
              'width':{'mean': 293.0e-6,   'error': 9e-6}},
    'chiC0': {'mass': {'mean': 3414.71e-3,  'error': 0.30e-3},
              'width':{'mean': 10.7e-3,    'error': 0.6e-3}},
    'hc':    {'mass': {'mean': 3525.37e-3,  'error': 0.14e-3},
              'width':{'mean': 0.78e-3,    'error': hc_width_error}},
    'chiC1': {'mass': {'mean': 3510.67e-3,  'error': 0.05e-3},
              'width':{'mean': 0.84e-3,    'error': 0.04e-3}},
    'chiC2': {'mass': {'mean': 3556.17e-3,  'error': 0.07e-3},
              'width':{'mean': 1.98e-3,    'error': 0.09e-3}},
}

# =============================================================================
# 3. FORTRAN MODEL PARAMETERS
# =============================================================================
FORTRAN_MODEL_DEFAULTS = {
    "eps": 1.0e-4,   
    "conf": 1.0,        
    "xlam": 0.181,  
}
# =============================================================================
# 4. EXPERIMENTAL DATA FOR FITTING
# =============================================================================
# Branching ratios (br) and their uncertainties (sigma).
EXPERIMENTAL_DATA = {
    'jpsi':  {'br': 0.0140, 'sigma': 0.0014, 'name': 'Br(J/ψ→ηcγ)'},
    'chic0': {'br': 0.0141, 'sigma': 0.0009, 'name': 'Br(χc0→J/ψγ)'},
    'psi2s': {'br': 0.0977, 'sigma': 0.0023, 'name': 'Br(ψ(2S)→χc0γ)'},
}
# =============================================================================
# 5. FITTER CONFIGURATION
# =============================================================================
# Configuration for the 2-parameter (mc, rho) fit
FIT_CONFIG_2D = {
    "bounds": [(1.6, 1.9), (0.3, 1.2)],
    "initial_guess": [1.8, 0.7],
}
FIT_CONFIG_3D = {
    "bounds": [
        (1.6, 1.9),                                 # Bounds for mc
        (0.3, 1.2),                                 # Bounds for rho
        (FORTRAN_MODEL_DEFAULTS["xlam"] * 0.9, FORTRAN_MODEL_DEFAULTS["xlam"] * 1.1), # Bounds for lambda
    ],
    "initial_guess": [1.8, 0.7, FORTRAN_MODEL_DEFAULTS["xlam"]],
    "lambda_central": FORTRAN_MODEL_DEFAULTS["xlam"],
    "lambda_relative_error": 0.10,
}

# =============================================================================
# 6. MONTE CARLO ERROR PROPAGATION CONFIGURATION
# =============================================================================
ERROR_PROP_CONFIG = {
    "run_propagation": True,
    "n_toys": 20000,
    "correlation_mc_rho": -0.8,
    "unscaled_errors": {
        "mc_upper": 0.009,
        "mc_lower": 0.004,
        "rho_upper": 0.016,
        "rho_lower": 0.019,
    },
    "systematics": {
        "lambda": {
            "central": FIT_CONFIG_3D["lambda_central"],
            "relative_error": FIT_CONFIG_3D["lambda_relative_error"]
        },
    }
}