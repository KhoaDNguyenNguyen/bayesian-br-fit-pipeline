import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import corner
import dynesty
from scipy.stats import norm, pearsonr
from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
import time
import multiprocessing
from scipy.stats import gaussian_kde
from functools import partial
import logging
from matplotlib.lines import Line2D
from scipy.optimize import brentq
try:
    import config
    from master_fitting import PhysicsModel, DummyLogger, NUISANCE_PARAM_ORDER
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import necessary local files. Make sure 'config.py' and 'master_fitting.py' are in the same directory.")
    print(f"Details: {e}")
    exit()

# =============================================================================
# 1. GLOBAL CONFIGURATION & SETUP
# =============================================================================
eps_DIR = "analysis_results/figures/eps"
TXT_DIR = "analysis_results/log_data"
N_CPU = cpu_count()

try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 20,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.linestyle": ':',
        "grid.alpha": 0.6,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
        "xtick.top": True,
        "ytick.right": True,
    })
    HAS_LATEX = True
except Exception as e:
    print(f"Warning: LaTeX not found. Falling back to default font rendering. Details: {e}")
    HAS_LATEX = False

PALETTE = {
    'primary': '#0072B2',    # A nice blue
    'secondary': '#009E73',  # A green
    'accent_pos': '#56B4E9', # Light blue
    'accent_neg': '#D55E00', # A vermillion/orange
    'neutral': 'gray',
    'fit_data': '#1f77b4', 
    'prediction': '#9467bd' 
}


def setup_directories():
    """Create output directories if they don't exist."""
    if not os.path.exists(eps_DIR):
        os.makedirs(eps_DIR)
    if not os.path.exists(TXT_DIR):
        os.makedirs(TXT_DIR)

def save_text(filename, content):
    """Helper function to save text content to the TXT directory."""
    with open(os.path.join(TXT_DIR, filename), 'w') as f:
        f.write(content)

# =============================================================================
# 2. DATA LOADING
# =============================================================================
def load_data():
    """Load all necessary .pkl files into a dictionary."""
    print("Loading all data from .pkl files...")
    data = {}
    try:
        with open('pkl/dynesty_results.pkl', 'rb') as f:
            data['dynesty'] = pickle.load(f)
        with open('pkl/ppc_predictions.pkl', 'rb') as f:
            ppc_preds = pickle.load(f)
            data['ppc'] = {
                'jpsi':  [p[0] for p in ppc_preds if not np.isnan(p[0])],
                'chic0': [p[1] for p in ppc_preds if not np.isnan(p[1])],
                'psi2s': [p[2] for p in ppc_preds if not np.isnan(p[2])]
            }
        with open('pkl/profile_mc.pkl', 'rb') as f:
            data['profile_mc'] = pickle.load(f)
        with open('pkl/profile_rho.pkl', 'rb') as f:
            data['profile_rho'] = pickle.load(f)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Cannot find a required data file: {e.filename}")
        exit()

# =============================================================================
# 3. ANALYSIS MODULES
# =============================================================================

# -------------------------
# MODULE A: BAYESIAN RESULTS
# -------------------------
def analyze_bayesian_results(results):
    """
    Processes dynesty results to produce the main parameter constraints,
    correlation info, and the corner plot.
    """
    print("\n[Module A] Analyzing Bayesian inference results...")
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])

    # --- Parameter Estimation (Text Summary) ---
    param_names = ['mc', 'rho'] + [f"{p}_{t}" for p, t in NUISANCE_PARAM_ORDER]
    summary_text = "BAYESIAN PARAMETER ESTIMATION SUMMARY\n"
    summary_text += "=" * 40 + "\n\n"
    summary_text += f"{'Parameter':>15} | {'Median & 68% Credible Interval'}\n"
    summary_text += "-" * 55 + "\n"

    for i in range(len(param_names)):
        quantiles = dynesty.utils.quantile(samples[:, i], [0.1587, 0.5, 0.8413], weights=weights)
        median, lower_err, upper_err = quantiles[1], quantiles[1] - quantiles[0], quantiles[2] - quantiles[1]
        summary_text += f"{param_names[i]:>15} | {median:.5f} (+{upper_err:.5f} / -{lower_err:.5f})\n"

    # --- Correlation and Best-fit ---
    mc_samples = samples[:, 0]
    rho_samples = samples[:, 1]

    mean, cov = dynesty.utils.mean_and_cov(samples[:, :2], weights)
    correlation = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    max_logl_idx = np.argmax(results.logl)
    best_fit_params = results.samples[max_logl_idx]
    chi2_min = -2 * results.logl[max_logl_idx]
    dof = len(config.EXPERIMENTAL_DATA) - 2
    chi2_per_dof = chi2_min / dof if dof > 0 else float('nan')

    summary_text += "\n\nADDITIONAL FIT METRICS\n"
    summary_text += "=" * 40 + "\n"
    summary_text += f"Best-fit (max likelihood) mc:  {best_fit_params[0]:.5f} GeV\n"
    summary_text += f"Best-fit (max likelihood) rho: {best_fit_params[1]:.5f}\n"
    summary_text += f"Pearson Correlation (mc, rho): {correlation:.4f}\n"
    summary_text += f"Minimum Chi-Squared (from dynesty): {chi2_min:.4f}\n"
    summary_text += f"Degrees of Freedom (d.o.f): {dof}\n"
    summary_text += f"Chi-Squared / d.o.f: {chi2_per_dof:.4f}\n"
    save_text("bayesian_summary.txt", summary_text)
    print(" -> Saved bayesian_summary.txt")

    corner_samples = dynesty.utils.resample_equal(samples, weights)[:, :2]
    latex_labels = [r'$m_c~[\mathrm{GeV}]$', r'$\rho$']
    
    mc_q = np.quantile(corner_samples[:, 0], [0.16, 0.5, 0.84])
    rho_q = np.quantile(corner_samples[:, 1], [0.16, 0.5, 0.84])
    mc_title = r"$m_c = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(mc_q[1], mc_q[2]-mc_q[1], mc_q[1]-mc_q[0])
    rho_title = r"$\rho = {:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$".format(rho_q[1], rho_q[2]-rho_q[1], rho_q[1]-rho_q[0])

    fig = corner.corner(
        corner_samples,
        labels=latex_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 16},
        color="darkblue",
        truths=[mean[0], mean[1]],
        truth_color="red"
    )
    fig.axes[0].set_title(mc_title, fontsize=12)
    fig.axes[3].set_title(rho_title, fontsize=12)

    # fig.suptitle('Posterior Distributions for Core Model Parameters', y=1.02)
    plt.savefig(os.path.join(eps_DIR, "Figure1_CornerPlot.eps"), bbox_inches='tight')
    plt.close(fig)
    print(" -> Saved Figure1_CornerPlot.eps")
    return best_fit_params

# -------------------------------------
# MODULE B: GOODNESS-OF-FIT & PPC PLOTS
# -------------------------------------
def analyze_gof_and_ppc(ppc_data, best_fit_params):
    """
    (Publication Version)
    Creates the pull plot and detailed Posterior Predictive Check plots,
    addressing all professional formatting recommendations.
    """
    print("\n[Module B] Analyzing Goodness-of-Fit and PPC...")
    exp_data = config.EXPERIMENTAL_DATA
    
    pulls = {}
    pull_text_content = "GOODNESS-OF-FIT PULL SUMMARY\n" + "=" * 40 + "\n\n"
    pull_text_content += f"Pull is defined as (Median_model - Exp) / sqrt(sigma_model^2 + sigma_exp^2)\n\n"
    for key, data in exp_data.items():
        if key in ppc_data:
            model_preds, exp_val, sigma_exp = ppc_data[key], data['br'], data['sigma']
            median_model, sigma_model = np.median(model_preds), np.std(model_preds)
            pull = (median_model - exp_val) / np.sqrt(sigma_model**2 + sigma_exp**2)
            pulls[key] = pull
            pull_text_content += f"Channel {data['name']}: {pull:.3f} sigma\n"
    save_text("gof_pulls.txt", pull_text_content)
    print(" -> Saved gof_pulls.txt")
    
    def to_latex_label(name):
        name = name.replace('χc0', r'$\chi_{c0}$').replace('psi(2S)', r'$\psi(2S)$')
        name = name.replace('J/psi', r'$J/\psi$').replace('→', r'$\to$')
        name = name.replace('γ', r'$\gamma$').replace('ψ', r'$\psi$')
        name = name.replace('χ', r'$\chi$').replace('ηc', r'$\eta_c$').replace('η', r'$\eta$')
        return name

    print(" -> Recreating the original bar chart style for Pull Summary (Figure 2)...")

    sorted_pulls = sorted(pulls.items(), key=lambda item: item[1], reverse=True)
    
    pull_channels = [to_latex_label(exp_data[k]['name']) for k, v in sorted_pulls]
    pull_values = [v for k, v in sorted_pulls]
    sorted_keys = [k for k, v in sorted_pulls]
    y_pos = np.arange(len(pull_channels))

    bar_colors = ['red' if key == 'jpsi' else PALETTE['fit_data'] for key in sorted_keys]

    fig, ax = plt.subplots(figsize=(10, 6))


    ax.barh(y_pos, pull_values, color=bar_colors, edgecolor='black', height=0.5)

    ax.axvline(0, color='black', linestyle='-', lw=1)
    for x_val in [-2, -1, 1, 2]:
        ax.axvline(x_val, color='gray', linestyle='--', lw=1)

    top_y = ax.get_ylim()[1]
    ax.text(-2, top_y, r'$-2\sigma$', ha='center', va='bottom', color='black', fontsize=12)
    ax.text(2, top_y, r'$2\sigma$', ha='center', va='bottom', color='black', fontsize=12)


    ax.set_yticks(y_pos)
    ax.set_yticklabels(pull_channels, fontsize=14)
    ax.invert_yaxis() 

    ax.set_xlabel(r'Pull ($\sigma$) = (Median$_{\mathrm{model}}$ $-$ Exp) / $\sqrt{\sigma^2_{\mathrm{model}} + \sigma^2_{\mathrm{exp}}}$', fontsize=14)
    # ax.set_title('Pull Summary for Fitted Decay Channels', fontsize=18)

    ax.set_xlim(-2.5, 2.5)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', which='both', length=0) 
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(os.path.join(eps_DIR, "Figure2_PullSummary.eps"), bbox_inches='tight')
    plt.close(fig)
    print(" -> Saved Figure2_PullSummary.eps")

    # --- Revamped Individual PPC Plots (Fig 3 & 4) ---
    for key, data in exp_data.items():
        if key in ppc_data:
            fig, ax = plt.subplots(figsize=(8, 6))
            exp_mean, exp_sigma, model_preds = data['br'], data['sigma'], np.array(ppc_data[key])

            raw_name = data['name']

            if raw_name.startswith('Br(') and raw_name.endswith(')'):
                core_reaction = raw_name[3:-1]
            else:
                core_reaction = raw_name
            
            tex_str = core_reaction.replace('→', r' \to ')
            tex_str = tex_str.replace('χc0', r'\chi_{c0}')
            tex_str = tex_str.replace('psi(2S)', r'\psi(2S)')
            tex_str = tex_str.replace('J/psi', r'J/\psi')
            tex_str = tex_str.replace('γ', r'\gamma')
            tex_str = tex_str.replace('ψ', r'\psi')
            tex_str = tex_str.replace('χ', r'\chi')
            tex_str = tex_str.replace('ηc', r'\eta_c')
            tex_str = tex_str.replace('η', r'\eta')
            
            tex_str = ' '.join(tex_str.split())

            final_x_label = f'$\\mathcal{{B}}({tex_str})$'

            kde = gaussian_kde(model_preds, bw_method='silverman')
            x_range = np.linspace(model_preds.min() * 0.9, model_preds.max() * 1.1, 500)
            y_kde = kde(x_range)
            ax.plot(x_range, y_kde, color=PALETTE['primary'], label='Model Posterior Predictive')
            ax.fill_between(x_range, y_kde, color=PALETTE['primary'], alpha=0.2)

            ax.axvspan(exp_mean - exp_sigma, exp_mean + exp_sigma, color=PALETTE['accent_neg'], alpha=0.3, zorder=2, label=r'Experimental $1\sigma$ Band')
            ax.axvline(exp_mean, color=PALETTE['accent_neg'], linestyle='--', lw=2, zorder=3, label='Experimental Central Value')

            ax.set_xlabel(final_x_label)
            ax.set_ylabel('Probability Density')

            pull_val = pulls.get(key, float('nan'))
            ax.text(0.95, 0.95, rf'Pull = {pull_val:.2f}$\sigma$', transform=ax.transAxes,
                    fontsize=12, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8, ec='gray'))

            ax.legend(loc='best')
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_ylim(bottom=0)
            plt.tight_layout()
            plt.savefig(os.path.join(eps_DIR, f"Figure_PPC_{key}.eps"), bbox_inches='tight')
            plt.close(fig)
            print(f" -> Saved Figure_PPC_{key}.eps ")
# -----------------------------------
# MODULE C: PREDICTIONS & UNCERTAINTY
# -----------------------------------
def predict_worker(params):
    """
    Calculates all branching ratios for a given parameter vector.
    THIS IS THE ORIGINAL WORKER. IT IS KEPT FOR REFERENCE BUT THE ROBUST
    VERSION BELOW IS USED INSTEAD.
    """
    mc, rho = params[0], params[1]
    
    sampled_properties = {}
    for i, (particle, prop_type) in enumerate(NUISANCE_PARAM_ORDER):
        key = f"{particle}_{prop_type}"
        sampled_properties[key] = params[i + 2]
    
    model = PhysicsModel(logger=DummyLogger(), sampled_properties=sampled_properties)
    
    predictions = {
        'Br(chic0->Jpsig)': model.get_chic0_br(mc, rho),
        'Br(psi(2S)->chic0g)': model.get_psi2s_br(mc, rho),
        'Br(chic1->Jpsig)': model.get_chic1_to_jpsi_br(mc, rho),
        'Br(chic2->Jpsig)': model.get_chic2_to_jpsi_br(mc, rho),
        'Br(psi(2S)->chic1g)': model.get_psi2s_to_chic1_br(mc, rho),
        'Br(psi(2S)->chic2g)': model.get_psi2s_to_chic2_br(mc, rho)
    }
    return predictions

def predict_worker_robust(params, max_retries=3, retry_delay=0.5):
    """
    A robust worker function that wraps the original `predict_worker`.
    It retries on any failure and returns None if it fails permanently.
    This prevents a single crash (e.g., from Fortran) from killing the whole pool.
    """
    for attempt in range(max_retries):
        try:
            mc, rho = params[0], params[1]
            sampled_properties = {}
            for i, (particle, prop_type) in enumerate(NUISANCE_PARAM_ORDER):
                key = f"{particle}_{prop_type}"
                sampled_properties[key] = params[i + 2]
            
            model = PhysicsModel(logger=DummyLogger(), sampled_properties=sampled_properties)
            
            predictions = {
                'Br(chic0->Jpsig)': model.get_chic0_br(mc, rho),
                'Br(psi(2S)->chic0g)': model.get_psi2s_br(mc, rho),
                'Br(chic1->Jpsig)': model.get_chic1_to_jpsi_br(mc, rho),
                'Br(chic2->Jpsig)': model.get_chic2_to_jpsi_br(mc, rho),
                'Br(psi(2S)->chic1g)': model.get_psi2s_to_chic1_br(mc, rho),
                'Br(psi(2S)->chic2g)': model.get_psi2s_to_chic2_br(mc, rho),
                'Br(Jpsi->etacg)': model.get_jpsi_br(mc, rho),    
                'Br(hc->etacg)': model.get_hc_br(mc, rho),        
                'Br(etac->gg)': model.get_etac_br(mc, rho)  
            }
            return predictions
        except Exception as e:

            print(f"Warning: Worker failed on attempt {attempt + 1}/{max_retries} for params: {params[:2]}. Error: {e}. Retrying after {retry_delay}s...")
            time.sleep(retry_delay)
    
    error_message = f"CRITICAL: Worker permanently failed for parameters {params}. This data point will be excluded."
    print(error_message)
    logging.error(error_message)
    return None

def process_parallel_results(results_list, total_samples):
    """
    Processes the list of results from the parallel pool, filtering out failures
    and reporting statistics.
    """
    failures = results_list.count(None)
    successes = [r for r in results_list if r is not None]
    
    print(f"\n  Parallel processing report:")
    print(f"  - Total samples attempted: {total_samples}")
    print(f"  - Successful calculations: {len(successes)}")
    print(f"  - Failed calculations (after retries): {failures}")
    
    if failures > 0:
        print("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"  !!! WARNING: {failures} SAMPLES FAILED TO COMPUTE.            !!!")
        print(  "  !!! The resulting distributions might be biased. Check    !!!")
        print(f"  !!! 'analysis_errors.log' for details on failing parameters. !!!")
        print(  "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        
    return successes

def run_parallel_job_with_timeout(pool, worker_func, job_samples, timeout_seconds=60):
    """
    Submits jobs asynchronously and collects results with a timeout for each job.
    This is the core of the solution to prevent hanging.
    """
    total_tasks = len(job_samples)
    print(f"  -> Submitting {total_tasks} tasks to the pool...")
    
    async_results = [pool.apply_async(worker_func, (s,)) for s in job_samples]
    results_raw = []
    
    print(f"  -> Collecting results with a {timeout_seconds}s timeout per task...")
    start_collect_time = time.time()
    
    for i, res in enumerate(async_results):
        try:
            result = res.get(timeout=timeout_seconds)
            results_raw.append(result)
        except multiprocessing.TimeoutError:
            error_message = f"CRITICAL: Worker timed out on task {i+1}/{total_tasks}. Parameters: {job_samples[i][:4]}... This data point will be excluded."
            print(f"\n{error_message}\n")
            logging.error(error_message)
            results_raw.append(None)
        except Exception as e:
            error_message = f"CRITICAL: An unexpected error occurred collecting task {i+1}/{total_tasks}. Error: {e}. Parameters: {job_samples[i][:4]}... This data point will be excluded."
            print(f"\n{error_message}\n")
            logging.error(error_message)
            results_raw.append(None)

        if (i + 1) % 1000 == 0 or (i + 1) == total_tasks:
            elapsed = time.time() - start_collect_time
            print(f"     Collected {i + 1}/{total_tasks} results... ({elapsed:.1f}s elapsed)", end='\r')

    print("\n  -> Collection complete.")
    return process_parallel_results(results_raw, total_tasks)


def analyze_predictions_and_uncertainty(results, pool):
    """
    Generates predictions for all channels, with timeout and restart capability.
    This version includes caching for both the main predictions and the 
    computationally expensive uncertainty decomposition step.
    """
    print("\n[Module C] Generating predictions and decomposing uncertainty...")
    weights = np.exp(results.logwt - results.logz[-1])
    samples = dynesty.utils.resample_equal(results.samples, weights)
    
    predictions_cache_file = "pkl/predictions_list.pkl"
    if os.path.exists(predictions_cache_file):
        print(f"  -> Found main prediction cache in '{predictions_cache_file}'. Loading.")
        with open(predictions_cache_file, 'rb') as f:
            predictions_list = pickle.load(f)
    else:
        print(f"  -> No main prediction cache found. Calculating all predictions...")
        start_time = time.time()
        predictions_list = run_parallel_job_with_timeout(pool, predict_worker_robust, samples)
        end_time = time.time()
        print(f"  ...total prediction step done in {end_time - start_time:.2f} seconds.")
        if predictions_list:
            print(f"  -> Saving successful predictions to '{predictions_cache_file}'.")
            with open(predictions_cache_file, 'wb') as f:
                pickle.dump(predictions_list, f)

    if not predictions_list:
        print("CRITICAL: No predictions could be generated. Aborting Module C.")
        return

    pred_dist = {key: np.array([p[key] for p in predictions_list if p and not np.isnan(p[key])])
                for key in predictions_list[0]}


    print(" -> Generating re-ordered Bar Chart with fitted channels grouped...")

    all_considered_channels = [
        'Br(hc->etacg)',
        'Br(chic1->Jpsig)',
        'Br(chic2->Jpsig)',
        'Br(psi(2S)->chic1g)',
        'Br(psi(2S)->chic0g)',
        'Br(psi(2S)->chic2g)',
        'Br(Jpsi->etacg)',
        'Br(chic0->Jpsig)',
    ]

    prediction_to_ppc_map = {
        'Br(chic0->Jpsig)': 'jpsi',
        'Br(psi(2S)->chic0g)': 'chic0',
        'Br(Jpsi->etacg)': 'psi2s'
    }
    fitted_ppc_keys = set(config.EXPERIMENTAL_DATA.keys())

    medians_dict = {k: np.median(pred_dist.get(k, [np.nan])) for k in all_considered_channels}
    fitted_channels = []
    prediction_only_channels = []
    for channel in all_considered_channels:
        ppc_key = prediction_to_ppc_map.get(channel)
        if ppc_key in fitted_ppc_keys:
            fitted_channels.append(channel)
        else:
            prediction_only_channels.append(channel)

    fitted_channels.sort(key=lambda k: medians_dict.get(k, 0), reverse=True)
    prediction_only_channels.sort(key=lambda k: medians_dict.get(k, 0), reverse=True)

    final_channel_order = fitted_channels + prediction_only_channels
    
    channel_names_tex = {
        'Br(chic0->Jpsig)': r'$\chi_{c0} \to J/\psi\gamma$',
        'Br(psi(2S)->chic0g)': r'$\psi(2S) \to \chi_{c0}\gamma$',
        'Br(chic1->Jpsig)': r'$\chi_{c1} \to J/\psi\gamma$',
        'Br(chic2->Jpsig)': r'$\chi_{c2} \to J/\psi\gamma$',
        'Br(psi(2S)->chic1g)': r'$\psi(2S) \to \chi_{c1}\gamma$',
        'Br(psi(2S)->chic2g)': r'$\psi(2S) \to \chi_{c2}\gamma$',
        'Br(Jpsi->etacg)': r'$J/\psi \to \eta_c\gamma$',
        'Br(hc->etacg)': r'$h_c \to \eta_c\gamma$'
    }

    y_labels = final_channel_order
    y_pos = np.arange(len(y_labels))
    
    medians_plot = [medians_dict[k] for k in y_labels]
    errors_low = [medians_plot[i] - np.quantile(pred_dist.get(k, [np.nan]), 0.16) for i, k in enumerate(y_labels)]
    errors_high = [np.quantile(pred_dist.get(k, [np.nan]), 0.84) - medians_plot[i] for i, k in enumerate(y_labels)]
    asymmetric_error = [errors_low, errors_high]

    bar_colors = []
    for key in y_labels:
        ppc_key = prediction_to_ppc_map.get(key)
        is_fitted = ppc_key in fitted_ppc_keys
        bar_colors.append(PALETTE['fit_data'] if is_fitted else PALETTE['prediction'])

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(y_pos, medians_plot, 
            height=0.7, 
            xerr=asymmetric_error, 
            align='center', 
            color=bar_colors, 
            edgecolor='black',
            linewidth=0.5,
            error_kw=dict(ecolor='black', lw=1, capsize=3, capthick=1))

    ax.set_yticks(y_pos)
    ax.set_yticklabels([channel_names_tex.get(k, k) for k in y_labels], fontsize=12)
    ax.invert_yaxis()
    # ax.set_xlabel(r'Predicted Branching Ratio, Br$_{\mathrm{model}}$ (log scale)', fontsize=14)
    ax.set_xlabel(r'$\mathcal{B}_{\mathrm{model}}$ (log scale)', fontsize=14)

    # ax.set_title('Model Predictions for All Channels', fontsize=16)
    ax.set_xscale('log')

    ax.grid(True, which='both', axis='x', linestyle=':', linewidth=0.7, color='gray')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', which='both', length=0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE['fit_data'], edgecolor='black', linewidth=0.5, label='Fitted to Data'),
        Patch(facecolor=PALETTE['prediction'], edgecolor='black', linewidth=0.5, label='Prediction Only')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(eps_DIR, "Figure5_PredictionSummary.eps"), bbox_inches='tight')
    plt.close(fig)
    print(" -> Saved Figure5_PredictionSummary.eps ")

    print("\n  Decomposing uncertainty budget...")
    decomp_cache_file = "pkl/uncertainty_decomposition_cache.pkl"

    if os.path.exists(decomp_cache_file):
        print(f"  -> Found decomposition cache in '{decomp_cache_file}'. Loading.")
        with open(decomp_cache_file, 'rb') as f:
            decomp_results = pickle.load(f)
        pred_mc_list = decomp_results.get('mc', [])
        pred_rho_list = decomp_results.get('rho', [])
        pred_syst_list = decomp_results.get('syst', [])
        print("  -> Decomposition results loaded successfully.")

    else:
        print(f"  -> No decomposition cache found. Calculating contributions...")
        q50 = np.quantile(samples, 0.5, axis=0)
        best_fit_mc, best_fit_rho = q50[0], q50[1]
        best_fit_nuisance = q50[2:]
        

        samples_mc_only = np.copy(samples); samples_mc_only[:, 1] = best_fit_rho; samples_mc_only[:, 2:] = best_fit_nuisance
        samples_rho_only = np.copy(samples); samples_rho_only[:, 0] = best_fit_mc; samples_rho_only[:, 2:] = best_fit_nuisance
        samples_syst_only = np.copy(samples); samples_syst_only[:, 0] = best_fit_mc; samples_syst_only[:, 1] = best_fit_rho

        pred_mc_list = run_parallel_job_with_timeout(pool, predict_worker_robust, samples_mc_only)
        pred_rho_list = run_parallel_job_with_timeout(pool, predict_worker_robust, samples_rho_only)
        pred_syst_list = run_parallel_job_with_timeout(pool, predict_worker_robust, samples_syst_only)
        
        results_to_cache = {
            'mc': pred_mc_list,
            'rho': pred_rho_list,
            'syst': pred_syst_list
        }
        print(f"  -> Saving new decomposition results to '{decomp_cache_file}'.")
        with open(decomp_cache_file, 'wb') as f:
            pickle.dump(results_to_cache, f)

    # total_vars = {k: np.var(v) for k, v in pred_dist.items()}
    total_vars = {k: np.var(v) for k, v in pred_dist.items()}
    mc_vars = {k: np.var([p[k] for p in pred_mc_list if p and not np.isnan(p[k])]) for k in pred_dist} if pred_mc_list else {k: 0 for k in pred_dist}
    rho_vars = {k: np.var([p[k] for p in pred_rho_list if p and not np.isnan(p[k])]) for k in pred_dist} if pred_rho_list else {k: 0 for k in pred_dist}
    syst_vars = {k: np.var([p[k] for p in pred_syst_list if p and not np.isnan(p[k])]) for k in pred_dist} if pred_syst_list else {k: 0 for k in pred_dist}
    medians = {k: np.median(v) for k, v in pred_dist.items()}

    decomp_text = "UNCERTAINTY BUDGET DECOMPOSITION\n"
    decomp_text += "=" * 75 + "\n"
    decomp_text += f"{'Decay Channel':<25} | {'Prediction (Br)':<20} | {'% Var (mc)':<10} | {'% Var (rho)':<10} | {'% Var (Sys)':<10}\n"
    decomp_text += "-" * 75 + "\n"

    for key in pred_dist:
        total_v = total_vars[key]
        if total_v > 1e-30:
            mc_p = 100 * mc_vars.get(key, 0) / total_v
            rho_p = 100 * rho_vars.get(key, 0) / total_v
            syst_p = 100 * syst_vars.get(key, 0) / total_v
        else:
            mc_p, rho_p, syst_p = 0, 0, 0
        
        pred_val = medians.get(key, 0)
        pred_err = np.sqrt(total_v)
        pred_str = f"{pred_val:.3e} +/- {pred_err:.3e}"
        decomp_text += f"{key:<25} | {pred_str:<20} | {mc_p:<10.1f} | {rho_p:<10.1f} | {syst_p:<10.1f}\n"
    
    save_text("uncertainty_decomposition.txt", decomp_text)
    print(" -> Saved uncertainty_decomposition.txt")


# ------------------------------------
# MODULE D: FREQUENTIST CROSS-CHECK
# ------------------------------------
def analyze_frequentist_crosscheck(profile_mc, profile_rho):
    """
    (Final Publication Version)
    Plots the profile likelihood with direct confidence level labeling and
    fine-tuned legend placement as requested.
    """
    print("\n[Module D] Analyzing Frequentist cross-check results...")

    results_pkl_path = os.path.join('pkl/frequentist_analysis_results.pkl')
    if os.path.exists(results_pkl_path):
        with open(results_pkl_path, 'rb') as f:
            frequentist_results = pickle.load(f)
        profiles_data = frequentist_results.get('profiles', {})
        intervals = frequentist_results.get('intervals', {})
        print("  -> Cached results loaded. Skipping recalculation.")
    else:
        print(f"  -> No cached results found. Running full calculation...")
        data_map = {'mc': profile_mc, 'rho': profile_rho}
        profiles_data, intervals = {}, {}
        for key, data in data_map.items():
            param_vals = np.array([p[0] for p in data])
            chi2_vals = np.array([p[1] for p in data])
            sort_idx = np.argsort(param_vals)
            param_vals, chi2_vals = param_vals[sort_idx], chi2_vals[sort_idx]
            delta_chi2 = chi2_vals - np.min(chi2_vals)
            profiles_data[key] = {'param_vals': param_vals, 'delta_chi2': delta_chi2}
            interp_func = interp1d(param_vals, delta_chi2 - 1, bounds_error=False, fill_value=np.max(delta_chi2))
            roots = [brentq(interp_func, param_vals[j], param_vals[j+1])
                     for j in range(len(param_vals) - 1)
                     if np.sign(interp_func(param_vals[j])) != np.sign(interp_func(param_vals[j+1]))]
            if len(roots) >= 2: intervals[key] = (roots[0], roots[-1])
            else: intervals[key] = (float('nan'), float('nan'))
        with open(results_pkl_path, 'wb') as f:
            pickle.dump({'profiles': profiles_data, 'intervals': intervals}, f)
        print(f" -> Saved new frequentist calculation results to '{results_pkl_path}'")

    if not profiles_data:
        print(" -> No profile data to plot. Skipping."); return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    # fig.suptitle('Frequentist Profile Likelihood Cross-Check', fontsize=18)
    
    latex_labels = {'mc': r'$m_c~[\mathrm{GeV}]$', 'rho': r'$\rho$'}
    plot_colors = {'mc': PALETTE['primary'], 'rho': PALETTE['secondary']}

    for i, key in enumerate(['mc', 'rho']):
        ax = axes[i]
        if key not in profiles_data:
            ax.text(0.5, 0.5, "Data unavailable", ha='center', va='center'); continue
        
        param_vals = profiles_data[key]['param_vals']
        delta_chi2 = profiles_data[key]['delta_chi2']
        ax.plot(param_vals, delta_chi2, '-', color=plot_colors[key], lw=2.5, label='Profile Likelihood')
        best_fit_param = param_vals[np.argmin(delta_chi2)]
        ax.plot(best_fit_param, 0, 'o', mfc='white', mec=plot_colors[key], ms=10, zorder=5)


        xlim = ax.get_xlim() 
        ax.axhline(1, color='gray', linestyle='--', lw=1)
        ax.text(xlim[1], 1, r' $\Delta\chi^2=1\;(1\sigma)$', va='center', ha='left', color='black')
        ax.axhline(4, color='gray', linestyle=':', lw=1)
        ax.text(xlim[1], 4, r' $\Delta\chi^2=4\;(2\sigma)$', va='center', ha='left', color='black')
        ax.axhline(9, color='gray', linestyle=':', lw=1)
        ax.text(xlim[1], 9, r' $\Delta\chi^2=9\;(3\sigma)$', va='center', ha='left', color='black')
        ax.set_xlim(xlim) 

        if key in intervals and not np.any(np.isnan(intervals[key])):
            low, high = intervals[key]
            ax.plot([low, high], [1, 1], 'k-', lw=1.5, zorder=4)
            ax.plot([low, low], [0, 1], 'k:', lw=1.5, zorder=4)
            ax.plot([high, high], [0, 1], 'k:', lw=1.5, zorder=4, label=r'$1\sigma$ Interval')

        handles, labels = ax.get_legend_handles_labels()
        legend_location = 'upper left' if key == 'mc' else 'upper right'
        ax.legend(handles, labels, loc=legend_location)

        ax.set_xlabel(latex_labels[key])
        if i == 0: ax.set_ylabel(r'$\Delta\chi^2 = \chi^2 - \chi^2_{\min}$')

        ax.set_ylim(-0.5, 10)
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', alpha=0.3)
    

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(os.path.join(eps_DIR, "Figure6_ProfileLikelihood.eps"), bbox_inches='tight')
    plt.close(fig)
    print(" -> Saved Figure6_ProfileLikelihood.eps")


    summary_text = (f"FREQUENTIST CROSS-CHECK SUMMARY\n{'=' * 40}\n\n"
                    "68.3% (1-sigma) Confidence Level Intervals from Profile Likelihood:\n\n"
                    f"mc:  [{intervals.get('mc', (0,0))[0]:.3f}, {intervals.get('mc', (0,0))[1]:.3f}] GeV\n"
                    f"rho: [{intervals.get('rho', (0,0))[0]:.3f}, {intervals.get('rho', (0,0))[1]:.3f}]\n")
    save_text("frequentist_summary.txt", summary_text)
    print(" -> Saved frequentist_summary.txt")

# =============================================================================
# 4. MAIN EXECUTION DRIVER
# =============================================================================
def main():
    """Main driver for the analysis pipeline."""
    start_total_time = time.time()
    setup_directories()
    logging.basicConfig(filename=os.path.join(TXT_DIR, 'analysis_errors.log'),
                    level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    
    all_data = load_data()

    # Context manager for the pool ensures it's closed properly
    with Pool(processes=N_CPU) as pool:
        # Module A: Bayesian primary results
        best_fit_params = analyze_bayesian_results(all_data['dynesty'])
        
        # Module B: Goodness-of-fit
        analyze_gof_and_ppc(all_data['ppc'], best_fit_params)

        # Module C: Predictions for unfitted channels & uncertainty budget
        analyze_predictions_and_uncertainty(all_data['dynesty'], pool)

        # Module D: Frequentist cross-check
        analyze_frequentist_crosscheck(all_data['profile_mc'], all_data['profile_rho'])

    end_total_time = time.time()
    print("\n=====================================================")
    print("===      ANALYSIS PIPELINE COMPLETE           ===")
    print(f"===  Total execution time: {end_total_time - start_total_time:.2f} seconds     ===")
    print(f"===  Results saved in '{eps_DIR}/' and '{TXT_DIR}/'     ===")
    print("=====================================================")


if __name__ == '__main__':
    main()