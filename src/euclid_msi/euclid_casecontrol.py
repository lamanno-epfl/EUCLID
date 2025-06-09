import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from jax import random
import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import optax
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import statsmodels.formula.api as smf
import arviz as az
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import cKDTree
import scipy.stats as stats
from jax.ops import segment_sum
from numba import njit
import matplotlib.gridspec as gridspec
import random as py_random
from scipy.stats import norm
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from PyPDF2 import PdfMerger
import os
import scanpy as sc
from jax.nn import sigmoid
from scipy.special import expit
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy as sch
from matplotlib.colors import ListedColormap, to_rgba
from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
from typing import List, Optional, Tuple

CASECONTROL_DIR = Path(os.getcwd()) / "casecontrol_analysis"
CASECONTROL_DIR.mkdir(exist_ok=True)

class LipidAnalysisConfig:
    def __init__(self):
        # Lipids to analyze
        self.lipids_to_analyze = ["SM 34:1;O2"]  # Default lipid
        
        # Model hyperparameters
        self.learning_rate = 0.05
        self.num_epochs = 2000
        self.adaptive_lr = False  # If True, will use learning rate scheduler
        
        # Priors for the model
        self.supertype_prior_std = 1.0
        self.supertype_shift_prior_std = 1.0
        self.sample_prior_std = 1.0
        self.section_prior_std = 5.0
        
        # Data processing
        self.downsampling = 1  # Use every nth point
        self.random_seed = 42
        self.normalize_percentiles = (0.1, 99.9)  # Lower and upper percentiles for normalization
        
        # Guide parameters
        self.guide_supertype_unconst_scale = 0.1
        self.guide_supertype_shift_scale = 0.1
        
    def display_config(self):
        """Display the current configuration."""
        print("=== Lipid Analysis Configuration ===")
        print(f"Lipids to analyze: {self.lipids_to_analyze}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Adaptive learning rate: {self.adaptive_lr}")
        print(f"Downsampling: {self.downsampling}")
        print(f"Random seed: {self.random_seed}")
        print(f"Normalization percentiles: {self.normalize_percentiles}")
        print("Prior standard deviations:")
        print(f"  - Supertype: {self.supertype_prior_std}")
        print(f"  - Supertype shift: {self.supertype_shift_prior_std}")
        print(f"  - Sample: {self.sample_prior_std}")
        print(f"  - Section: {self.section_prior_std}")
        print("Guide parameters:")
        print(f"  - Supertype unconst scale: {self.guide_supertype_unconst_scale}")
        print(f"  - Supertype shift scale: {self.guide_supertype_shift_scale}")

def cfg_string(cfg):
    return (
        f"lr{cfg.learning_rate}"
        f"_ep{cfg.num_epochs}"
        f"_ds{cfg.downsampling}"
        f"_seed{cfg.random_seed}"
        f"_priorS{cfg.sample_prior_std}"
        f"_priorSec{cfg.section_prior_std}"
        f"_suscPrior{cfg.supertype_shift_prior_std}"
        f"_guideU{cfg.guide_supertype_unconst_scale}"
        f"_guideS{cfg.guide_supertype_shift_scale}"
    )

config = LipidAnalysisConfig()

def normalize_lipid_column(df, column, lower_percentile=0.1, upper_percentile=99.9):
    """Normalize a lipid column to the range [0, 1] after clipping outliers."""
    values = df[column].values.astype(np.float32)
    lower_bound = np.percentile(values, lower_percentile)
    upper_bound = np.percentile(values, upper_percentile)
    clipped = np.clip(values, lower_bound, upper_bound)
    normalized = (clipped - lower_bound) / (upper_bound - lower_bound)
    df[column] = normalized
    return df

@njit
def sample_section(xs, ys, rand_idxs, max_x, max_y):
    """
    Fast sampling for a single section: no two points within a 3x3 neighborhood
    
    Parameters:
    - xs, ys: integer pixel coordinates arrays
    - rand_idxs: shuffled indices into xs/ys
    - max_x, max_y: maximum coordinate values in section
    
    Returns:
    - array of selected positions
    """
    occ = np.zeros((max_x + 3, max_y + 3), np.uint8)
    selected = np.empty(len(xs), np.int64)
    count = 0
    for i in range(len(rand_idxs)):
        idx = rand_idxs[i]
        x = xs[idx]
        y = ys[idx]
        free = True
        # check 3x3 neighborhood
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                xi = x + dx
                yi = y + dy
                if 0 <= xi < occ.shape[0] and 0 <= yi < occ.shape[1]:
                    if occ[xi, yi]:
                        free = False
                        break
            if not free:
                break
        if free:
            selected[count] = idx
            count += 1
            # mark occupied neighborhood
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    xi = x + dx
                    yi = y + dy
                    if 0 <= xi < occ.shape[0] and 0 <= yi < occ.shape[1]:
                        occ[xi, yi] = 1
    return selected[:count]

def random_subsample_no_neighbors(coords, seed=None):
    """
    Returns a random subsample of coords ensuring that no two points are within
    a 3x3 neighborhood (adjacent in x or y).
    Sampling is done independently for each SectionID.

    Parameters:
    - coords: DataFrame with columns ['x', 'y', 'SectionID']
      and arbitrary index labeling each point.
    - seed: optional random seed for reproducibility.

    Returns:
    - DataFrame subset of coords with the same columns and index,
      containing the selected points.
    """
    import numpy as _np
    _np.random.seed(seed)
    result_indices = []
    # loop per SectionID
    for section_id, group in coords.groupby('SectionID'):
        xs = group['x'].astype(np.int64).to_numpy()
        ys = group['y'].astype(np.int64).to_numpy()
        max_x = int(xs.max())
        max_y = int(ys.max())
        rand_idxs = _np.arange(len(xs), dtype=_np.int64)
        _np.random.shuffle(rand_idxs)
        sel = sample_section(xs, ys, rand_idxs, max_x, max_y)
        orig_idx = group.index.to_numpy()
        result_indices.append(orig_idx[sel])
    # combine all sections
    selected_all = np.concatenate(result_indices)
    return coords.loc[selected_all]

def analyze_nearest_neighbors(subsample):
    """Analyze nearest neighbor distances for the subsampled data."""
    # Store NN distances per section
    nn_by_section = {}

    for sec_id, grp in subsample.groupby('SectionID'):
        coords_arr = grp[['x','y']].to_numpy()
        if len(coords_arr) < 2:
            # can't compute NN for a singleton
            nn_by_section[int(sec_id)] = np.array([])
            print(f"Section {int(sec_id)}: only {len(coords_arr)} point(s), skipped.")
            continue

        # build and query tree
        tree = cKDTree(coords_arr)
        dists, idxs = tree.query(coords_arr, k=2)  # [0]=self, [1]=nearest neighbor
        nn_dists = dists[:, 1]

        nn_by_section[int(sec_id)] = nn_dists

    # If you want one big array of all NN distances across sections:
    all_nn = np.concatenate([d for d in nn_by_section.values() if len(d)])
    
    return nn_by_section, all_nn

def create_train_test_sets(coords, seed=42, downsampling=1):
    """Create training and test sets with spatial subsampling."""
    # Generate subsamples with no neighbors
    subsample = random_subsample_no_neighbors(coords, seed=seed)
    testset = random_subsample_no_neighbors(coords.loc[~coords.index.isin(subsample.index),:], seed=seed)
    
    # Apply downsampling if specified
    if downsampling > 1:
        subsample = subsample[::downsampling]
        testset = testset[::downsampling]
    
    return subsample, testset

def visualize_subsampling(coords, subsample):
    """Visualize a section to verify spatial subsampling."""
    section_id = coords['SectionID'].unique()[0]

    orig = coords[coords['SectionID'] == section_id]
    sub = subsample[subsample['SectionID'] == section_id]

    plt.figure(figsize=(6, 6))
    plt.scatter(orig['x'], orig['y'], s=0.1, alpha=0.3, label='Original points')
    plt.scatter(sub['x'], sub['y'], s=0.1, label='Subsampled')

    plt.legend()
    plt.title(f'Spatial subsample check — Section {int(section_id)}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().invert_yaxis()  # optional, if your pixel origin is top-left
    plt.tight_layout()
    
def model_CASE_CONTROL_hierarchical(
    condition_code, section_code, supertype_code,
    map_section_to_sample, map_sample_to_condition,
    lipid_x=None
):
    """Hierarchical Bayesian model for analyzing lipid expression patterns in CASE_CONTROL.

    This model implements a hierarchical Bayesian structure to analyze lipid expression
    patterns across different biological levels (supertypes, samples, and sections) while
    accounting for CASE_CONTROL condition effects. The model uses a non-centered parameterization
    for better sampling efficiency.

    Parameters
    ----------
    condition_code : array-like
        Binary codes indicating CASE_CONTROL condition (0 for control, 1 for CASE_CONTROL)
    section_code : array-like
        Integer codes identifying tissue sections
    supertype_code : array-like
        Integer codes identifying cell supertypes
    map_section_to_sample : array-like
        Mapping from section codes to sample codes
    map_sample_to_condition : array-like
        Mapping from sample codes to condition codes
    lipid_x : array-like, optional
        Observed lipid expression values. If None, the model runs in prior predictive mode.

    Notes
    -----
    The model implements a three-level hierarchy:
    1. Supertype level: Models baseline expression and condition shift
    2. Sample level: Captures inter-individual variation
    3. Section level: Models section-specific variations within samples (batch effects)

    The model uses sum-to-zero constraints on section effects within each condition
    to ensure identifiability and model batch effects.
    """
    # number of levels
    n_sections   = len(np.unique(section_code))
    n_samples    = len(np.unique(map_section_to_sample))
    n_conditions = len(np.unique(map_sample_to_condition))
    n_supertypes = len(np.unique(supertype_code))

    # ----------------------------
    # Supertypes main effects
    # ----------------------------
    with numpyro.plate("plate_supertype", n_supertypes):
        alpha_supertype_shift = numpyro.sample(
            "alpha_supertype_shift",
            dist.Normal(0.0, config.supertype_shift_prior_std)
        )
        alpha_supertype_unconst = numpyro.sample(
            "alpha_supertype_unconst",
            dist.Normal(0.0, config.supertype_prior_std)
        )
        alpha_supertype = jax.nn.sigmoid(alpha_supertype_unconst)

    # ----------------------------
    # Sample-level means
    # ----------------------------
    with numpyro.plate("plate_sample", n_samples):
        mu_alpha_sample_unconst = numpyro.sample(
            "mu_alpha_sample_unconst",
            dist.Normal(0.0, config.sample_prior_std)
        )
        alpha_sample = mu_alpha_sample_unconst

        # per-sample section‐level std (heteroskedasticity)
        log_sigma_sample = numpyro.sample(
            "log_sigma_sample",
            dist.Normal(0.0, config.section_prior_std)
        )
    sigma_sample = jnn.softplus(log_sigma_sample)

    # ----------------------------
    # Section-level effects (non-centered)
    # ----------------------------
    with numpyro.plate("plate_section", n_sections):
        z_section = numpyro.sample("z_section", dist.Normal(0.0, 1.0))
        alpha_section_unconst = (
            alpha_sample[map_section_to_sample]
            + z_section * sigma_sample[map_section_to_sample]
        )

    # sum-to-zero constraint by condition to break identifiability
    section_condition = map_sample_to_condition[map_section_to_sample]
    sum_by_cond    = segment_sum(alpha_section_unconst, section_condition, 2)
    count_by_cond  = segment_sum(jnp.ones_like(alpha_section_unconst), section_condition, 2)
    mean_by_cond   = sum_by_cond / count_by_cond
    alpha_section  = alpha_section_unconst - mean_by_cond[section_condition]

    # ----------------------------
    # Linear predictor & likelihood
    # ----------------------------
    mu = (
        alpha_section[section_code]
        + alpha_supertype[supertype_code]
        + jnp.where(
            condition_code == 1,
            alpha_supertype_shift[supertype_code],
            0.0
        )
    )
    with numpyro.plate("data", len(section_code)):
        numpyro.sample("obs", dist.Normal(mu, 0.1), obs=lipid_x)

def manual_guide(
    condition_code, section_code, supertype_code,
    map_section_to_sample, map_sample_to_condition, 
    lipid_x=None
):
    """Manual guide for variational inference of the CASE_CONTROL hierarchical model.

    This guide implements a structured variational approximation for the hierarchical
    CASE_CONTROL model. It uses a mean-field approximation with specific parameterizations
    for each level of the hierarchy.

    Parameters
    ----------
    condition_code : array-like
        Binary codes indicating CASE_CONTROL condition (0 for control, 1 for CASE_CONTROL)
    section_code : array-like
        Integer codes identifying tissue sections
    supertype_code : array-like
        Integer codes identifying cell supertypes
    map_section_to_sample : array-like
        Mapping from section codes to sample codes
    map_sample_to_condition : array-like
        Mapping from sample codes to condition codes
    lipid_x : array-like, optional
        Observed lipid expression values. If None, the guide runs in prior predictive mode.

    Notes
    -----
    The guide implements variational distributions for:
    1. Supertype effects: Normal distributions with learnable location and scale
    2. Sample effects: Delta distributions for mean and log-scale parameters
    3. Section effects: Delta distributions for non-centered parameters

    The guide uses configurable scale parameters (guide_supertype_unconst_scale and
    guide_supertype_shift_scale) to control the initial uncertainty in the
    variational approximation.
    """
    n_sections   = len(np.unique(section_code))
    n_samples    = len(np.unique(map_section_to_sample))
    n_supertypes = len(np.unique(supertype_code))
    
    # ---------------------------- 
    # Supertypes 
    # ---------------------------- 
    alpha_supertype_unconst_loc  = numpyro.param(
        "alpha_supertype_unconst_loc", jnp.zeros((n_supertypes,)))
    alpha_supertype_unconst_scale = numpyro.param(
        "alpha_supertype_unconst_scale", jnp.full((n_supertypes,), config.guide_supertype_unconst_scale),
        constraint=dist.constraints.positive
    )
    alpha_supertype_susc_loc = numpyro.param(
        "alpha_supertype_shift_loc", jnp.zeros((n_supertypes,)))
    alpha_supertype_susc_scale = numpyro.param(
        "alpha_supertype_shift_scale", jnp.full((n_supertypes,), config.guide_supertype_shift_scale),
        constraint=dist.constraints.positive
    )
    
    with numpyro.plate("plate_supertype", n_supertypes):
        numpyro.sample(
            "alpha_supertype_unconst",
            dist.Normal(alpha_supertype_unconst_loc, alpha_supertype_unconst_scale)
        )
        numpyro.sample(
            "alpha_supertype_shift",
            dist.Normal(alpha_supertype_susc_loc, alpha_supertype_susc_scale)
        )
    
    # ----------------------------
    # Sample-level means
    # ----------------------------
    mu_alpha_sample_unconst_loc = numpyro.param(
        "mu_alpha_sample_unconst_loc", jnp.zeros((n_samples,)))
    log_sigma_sample_loc = numpyro.param(
        "log_sigma_sample_loc", jnp.zeros((n_samples,)))
    
    with numpyro.plate("plate_sample", n_samples):
        numpyro.sample(
            "mu_alpha_sample_unconst",
            dist.Delta(mu_alpha_sample_unconst_loc)
        )
        numpyro.sample(
            "log_sigma_sample",
            dist.Delta(log_sigma_sample_loc)
        )
        
    # ----------------------------
    # Section non-centered parameter
    # ----------------------------
    z_section_loc = numpyro.param(
        "z_section_loc", jnp.zeros((n_sections,)))
    with numpyro.plate("plate_section", n_sections):
        numpyro.sample(
            "z_section",
            dist.Delta(z_section_loc)
        )

def prepare_data(df, lipid_name):
    """
    Prepare data for model training.
    """
    train = df.copy()
    
    label_encoder_condition = LabelEncoder()
    label_encoder_sample = LabelEncoder()
    label_encoder_supertype = LabelEncoder()
    label_encoder_section = LabelEncoder()
    
    train["Condition_code"] = label_encoder_condition.fit_transform(train["Condition"].values)
    train["Sample_code"] = label_encoder_sample.fit_transform(train["Sample"].values)
    train["supertype_code"] = label_encoder_supertype.fit_transform(train["supertype"].values)
    train["SectionID_code"] = label_encoder_section.fit_transform(train["SectionID"].values)
    
    map_sample_to_condition = (
        train[["Sample_code", "Condition_code"]]
        .drop_duplicates()
        .set_index("Sample_code", verify_integrity=True)
        .sort_index()["Condition_code"]
        .values
    )
    
    map_section_to_sample = (
        train[["SectionID_code", "Sample_code"]]
        .drop_duplicates()
        .set_index("SectionID_code", verify_integrity=True)
        .sort_index()["Sample_code"]
        .values
    )
    
    lipid_x = train[lipid_name].values
    
    return (
        train, 
        lipid_x, 
        map_sample_to_condition, 
        map_section_to_sample,
        train["supertype_code"].values, 
        train["SectionID_code"].values,
        train["Condition_code"].values
    )

def train_lipid_model(train_df, lipid_name, num_epochs=2000, learning_rate=0.05):
    """
    Train the model for a specific lipid.
    
    Parameters:
    - train_df: DataFrame with training data
    - lipid_name: Name of the lipid column
    - num_epochs: Number of training epochs
    - learning_rate: Learning rate for the optimizer
    
    Returns:
    - Trained model state and training metrics
    """
    
    # Prepare the data
    train, lipid_x, map_sample_to_condition, map_section_to_sample, supertype_code, section_code, condition_code = prepare_data(train_df, lipid_name)
    
    # Create mapping table for supertype codes
    mappingtable = train[['supertype', 'supertype_code']].drop_duplicates().reset_index().iloc[:,1:]
    mappingtable.index = mappingtable.supertype_code
    
    # Initialize optimizer and SVI
    optimizer = optax.adam(learning_rate=learning_rate)
    
    svi = SVI(
        model_CASE_CONTROL_hierarchical, 
        manual_guide, 
        optimizer, 
        loss=Trace_ELBO()
    )
    
    # Initialize SVI state
    rng_key = random.PRNGKey(0)
    svi_state = svi.init(
        rng_key,
        condition_code=condition_code,
        section_code=section_code,
        supertype_code=supertype_code,
        map_section_to_sample=map_section_to_sample,
        map_sample_to_condition=map_sample_to_condition,
        lipid_x=lipid_x
    )
    
    # Extract initial parameter names
    initial_params = svi.get_params(svi_state)
    param_traces = {name: [] for name in initial_params}
    
    losses = []
    
    # Training loop with parameter recording
    for i in tqdm(range(num_epochs), desc=f"Training {lipid_name}"):
        svi_state, loss = svi.update(
            svi_state,
            condition_code=condition_code,
            section_code=section_code,
            supertype_code=supertype_code,
            map_section_to_sample=map_section_to_sample,
            map_sample_to_condition=map_sample_to_condition,
            lipid_x=lipid_x
        )
        losses.append(loss)
        
        params = svi.get_params(svi_state)
        for name, val in params.items():
            param_traces[name].append(np.array(val))
    
    # Convert lists to arrays
    for name in param_traces:
        param_traces[name] = np.stack(param_traces[name])
    losses = np.array(losses)
    
    return svi, svi_state, param_traces, losses, train, mappingtable

def analyze_posterior(svi, svi_state, train, lipid_name, mappingtable):
    final_params = svi.get_params(svi_state)
    (_, _,
     map_s2c, map_sec2samp,
     super_code, sec_code, cond_code) = prepare_data(train, lipid_name)

    # ── Draw variational samples (rename 'samples' → 'samples_params') ────────
    samples_params = Predictive(
        model_CASE_CONTROL_hierarchical,
        guide=manual_guide,
        params=final_params,
        num_samples=1000,
        return_sites=[
            "alpha_supertype_shift",
            "alpha_supertype_unconst",
            "mu_alpha_sample_unconst",
            "log_sigma_sample",
            "z_section"
        ]
    )(
        random.PRNGKey(1),
        condition_code=cond_code,
        section_code=sec_code,
        supertype_code=super_code,
        map_section_to_sample=map_sec2samp,
        map_sample_to_condition=map_s2c,
        lipid_x=None
    )

    # ── Reconstruct section effects ──────────────────────────────────────────
    z_sec      = samples_params["z_section"]                     # [draws, n_sections]
    mu_samp    = samples_params["mu_alpha_sample_unconst"]       # [draws, n_samples]
    log_sig    = samples_params["log_sigma_sample"]              # [draws, n_samples]
    sigma_samp = jnn.softplus(log_sig)

    sec2samp = jnp.array(map_sec2samp)                            # [n_sections]
    mu_sec   = mu_samp[:, sec2samp]                              # [draws, n_sections]
    sig_sec  = sigma_samp[:, sec2samp]                           # [draws, n_sections]
    sections = mu_sec + z_sec * sig_sec                          # [draws, n_sections]

    # ── Center by condition ───────────────────────────────────────────────
    cond_mask = jnp.array(map_s2c)[sec2samp] == 1                  # [n_sections]
    pm   = jnp.mean(sections[:, cond_mask],  axis=1, keepdims=True)
    npm  = jnp.mean(sections[:, ~cond_mask], axis=1, keepdims=True)
    offset = jnp.where(cond_mask[None, :], pm, npm)
    sections_centered = sections - offset

    sectionmeans   = sections_centered.mean(axis=0)
    supertypemeans = jnn.sigmoid(samples_params["alpha_supertype_unconst"]).mean(axis=0)
    suscmeans      = samples_params["alpha_supertype_shift"].mean(axis=0)

    # ── Reconstruction vs ground truth (unchanged) ─────────────────────────
    gts, recons, colors = [], [], []
    for secnow in range(len(np.unique(train["SectionID_code"]))):
        for supertypenow in range(len(np.unique(train["supertype_code"]))):
            gt = train.loc[
                (train["SectionID_code"] == secnow) &
                (train["supertype_code"] == supertypenow),
                lipid_name
            ].mean()
            recon = sectionmeans[secnow] + supertypemeans[supertypenow]
            if secnow > 13:
                recon += suscmeans[supertypenow]
                col = "red"
            else:
                col = "blue"
            gts.append(gt); recons.append(recon); colors.append(col)
    plt.figure(figsize=(10,8))
    plt.scatter(gts, recons, c=colors, s=5, alpha=0.5)
    plt.title(f"Reconstruction vs Ground Truth for {lipid_name}")
    plt.xlabel("Ground Truth"); plt.ylabel("Reconstruction")
    plt.grid(alpha=0.3)
    plt.savefig(PDF_DIR /f"{lipid_name}_reconstruction.pdf")
    
    """ outdated...
    # ── Statistical analysis on shift ─────────────────────────────
    loc   = np.array(samples_params["alpha_supertype_shift"].mean(axis=0))
    scale = np.array(jnp.std(samples_params["alpha_supertype_shift"], axis=0))
    z_stat = loc / scale
    p_values = 2 * norm.sf(np.abs(z_stat))
    p_plus   = norm.cdf(z_stat)
    lfsr     = np.minimum(p_plus, 1 - p_plus)

    # BFDR q=0.05
    order     = np.argsort(lfsr)
    cum_mean  = np.cumsum(lfsr[order]) / np.arange(1, len(lfsr)+1)
    k_opt     = (np.where(cum_mean <= 0.05)[0].max() + 1) if np.any(cum_mean <= 0.05) else 0
    selected  = np.zeros_like(lfsr, dtype=bool)
    if k_opt > 0:
        selected[order[:k_opt]] = True

    names = mappingtable["supertype"].values
    df_stats = pd.DataFrame({
        "posterior_mean": loc,
        "posterior_sd":   scale,
        "p_sign_pos":     p_plus,
        "lfsr":           lfsr,
        "selected_bfdr05": selected,
        "ci_2.5%":        loc - 1.96*scale,
        "ci_97.5%":       loc + 1.96*scale,
        "p-value":        p_values
    }, index=names).sort_values("lfsr")
    df_stats.to_csv(f"{lipid_name}_CASE_CONTROL_shifts.csv")

    # ── Sample-effects vs observed means ───────────────────────────────────
    obs_means = train.groupby("Sample_code")[lipid_name].mean()
    plt.figure(figsize=(6,6))
    plt.scatter(obs_means.values, mu_samp.mean(axis=0))
    plt.xlabel("Observed sample mean")
    plt.ylabel("Variational mean sample effect")
    plt.title(f"Sample Effects vs Observed Means for {lipid_name}")
    plt.tight_layout()
    """
    # ── Posterior Error Probabilities & q-values for 5% FDR vs 0 ──────────
    # REFERENCE: http://varianceexplained.org/r/bayesian_fdr_baseball/
    # assume `samples` is shape [n_samples, n_supertypes]
    samples = np.array(samples_params["alpha_supertype_shift"])

    # 1) probability each effect is positive vs negative
    p_pos = (samples > 0).mean(axis=0)
    p_neg = (samples < 0).mean(axis=0)

    # 2) Posterior Error Probability: min tail beyond zero
    PEP = np.minimum(p_pos, p_neg)

    # 3) sort by PEP ascending and compute running mean → q-values
    order    = np.argsort(PEP)
    cum_PEP  = np.cumsum(PEP[order]) / np.arange(1, len(PEP) + 1)

    # 4) restore original order
    qvalue = np.empty_like(PEP)
    qvalue[order] = cum_PEP

    # 5) pick all supertypes with q < 0.05
    selected_fdr05 = qvalue < 0.05

    loc   = samples.mean(axis=0)
    scale = samples.std(axis=0)

    df_stats = pd.DataFrame({
        "posterior_mean":  loc,
        "posterior_sd":    scale,
        "p(>0)":           p_pos,
        "p(<0)":           p_neg,
        "PEP":             PEP,
        "qvalue":          qvalue,
        "selected_fdr05":  selected_fdr05,
        "ci_2.5%":         loc - 1.96*scale,
        "ci_97.5%":        loc + 1.96*scale,
    }, index=mappingtable["supertype"].values) \
        .sort_values("qvalue")

    df_stats.to_csv(f"{lipid_name}_CASE_CONTROL_shifts_fdr5_vs0.csv")

    return samples_params, df_stats

def evaluate_model(svi, svi_state, train_df, test_df, lipid_name):
    """
    Evaluate model performance on test data.
    
    Parameters:
    - svi: SVI object
    - svi_state: Trained SVI state
    - train_df: Training data DataFrame
    - test_df: Test data DataFrame
    - lipid_name: Name of the lipid
    
    Returns:
    - Test predictions and evaluation metrics
    """
    # Get final parameters
    final_params = svi.get_params(svi_state)
    
    # Prepare test data
    test, test_lipid_x, map_sample_to_condition, map_section_to_sample, supertype_code, section_code, condition_code = prepare_data(test_df, lipid_name)
    
    # Create predictive object
    num_samples = 1000
    predictive = Predictive(
        model_CASE_CONTROL_hierarchical,
        guide=manual_guide,
        params=final_params,
        num_samples=num_samples
    )
    
    # Generate predictions for test data
    samples_predictive = predictive(
        random.PRNGKey(1),
        condition_code=condition_code,
        section_code=section_code,
        supertype_code=supertype_code,
        map_section_to_sample=map_section_to_sample,
        map_sample_to_condition=map_sample_to_condition,
        lipid_x=None
    )
    
    # Calculate mean predictions
    predictions = samples_predictive["obs"].mean(axis=0)
    predictions = np.array(predictions)
    predictions[predictions < 0] = 0
    
    # Add predictions to test DataFrame
    test['estimate'] = predictions
    
    # Plot histogram of predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, density=True, bins=30, alpha=0.5, label='Predicted')
    plt.hist(test_lipid_x, density=True, bins=30, alpha=0.5, label='Actual')
    plt.legend()
    plt.xlabel(lipid_name)
    plt.ylabel('Density')
    plt.title(f'Posterior Predictive Check for {lipid_name} (Test Set)')
    plt.savefig(PDF_DIR /f"{lipid_name}_test_posterior_predictive.pdf")
    
    
    # Plot scatterplot of predictions vs actual
    plt.figure(figsize=(8, 8))
    plt.scatter(test[lipid_name], test['estimate'], s=2, alpha=0.5)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Predicted vs Actual for {lipid_name} (Test Set)')
    plt.savefig(PDF_DIR /f"{lipid_name}_test_scatter.pdf")
    
    
    # Q-Q Plot
    sorted_preds = np.sort(test['estimate'])
    sorted_actual = np.sort(test[lipid_name])
    plt.figure(figsize=(8, 5))
    plt.scatter(sorted_preds, sorted_actual, alpha=0.6, s=1, rasterized=True)
    plt.plot([min(sorted_preds), max(sorted_preds)],
             [min(sorted_preds), max(sorted_preds)],
             color='red', linestyle='--', label='y = x')
    plt.xlabel('Predicted Quantiles')
    plt.ylabel('Actual Quantiles')
    plt.title(f'Q–Q Plot for {lipid_name} (Test Set)')
    plt.savefig(PDF_DIR /f"{lipid_name}_qqplot_testset.pdf")
    
    
    # Calculate correlation
    corr = np.corrcoef(test[lipid_name], test['estimate'])[0, 1]
    
    return test, corr


def visualize_distribution_grid(samples_params, train, lipid_name, num_sections=10, num_supertypes=10):
    """
    Create a grid visualization of fitted distributions for random section/supertype combinations.
    """
    # Re-extract necessary data components
    _, _, map_sample_to_condition, map_section_to_sample, _, _, _ = prepare_data(train, lipid_name)
    
    # Reconstruct section effects from latent samples
    z_section = samples_params["z_section"]                             # [draws, n_sections]
    mu_alpha_sample = samples_params["mu_alpha_sample_unconst"]        # [draws, n_samples]
    log_sigma_sample = samples_params["log_sigma_sample"]              # [draws, n_samples]
    sigma_sample = jnn.softplus(log_sigma_sample)
    
    sec2samp = jnp.array(map_section_to_sample)                         # [n_sections]
    mu_sec   = mu_alpha_sample[:, sec2samp]                             # [draws, n_sections]
    sig_sec  = sigma_sample[:,    sec2samp]                             # [draws, n_sections]
    alpha_section_unc = mu_sec + z_section * sig_sec                   # [draws, n_sections]
    alpha_section_means = alpha_section_unc.mean(axis=0)                # [n_sections]
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(num_sections, num_supertypes)
    gs.update(wspace=0.3, hspace=0.4)
    
    # Choose random sections and supertypes
    max_section = max(train['SectionID_code']) + 1
    max_supertype = max(train['supertype_code']) + 1
    random_sections = py_random.sample(range(max_section), num_sections) if max_section >= num_sections else py_random.choices(list(set(train['SectionID_code'])), k=num_sections)
    random_supertypes = py_random.sample(range(max_supertype), num_supertypes) if max_supertype >= num_supertypes else py_random.choices(list(set(train['supertype_code'])), k=num_supertypes)
    
    # Plot each grid cell
    for i in range(num_sections):
        for j in range(num_supertypes):
            ax = plt.subplot(gs[i, j])
            secnow = random_sections[i]
            supertypenow = random_supertypes[j]
            
            # Map to sample and condition
            this_condition = map_sample_to_condition[map_section_to_sample[secnow]]
            
            # Use reconstructed section mean
            alpha_sec = float(alpha_section_means[secnow])
            alpha_st = float(jnn.sigmoid(samples_params["alpha_supertype_unconst"]).mean(axis=0)[supertypenow])
            alpha_st_susc = float(samples_params["alpha_supertype_shift"].mean(axis=0)[supertypenow])
            sigma = 0.1
            
            mu = alpha_sec + alpha_st + (alpha_st_susc if this_condition == 1 else 0.0)
            
            # Ground truth data
            gt_data = train.loc[
                (train['SectionID_code'] == secnow) &
                (train['supertype_code'] == supertypenow),
                lipid_name
            ].values
            
            if len(gt_data) == 0:
                ax.text(0.5, 0.5, f"No data\nS{secnow}, T{supertypenow}",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            
            ax.hist(gt_data, bins=10, density=True, alpha=0.6)
            x = np.linspace(gt_data.min(), gt_data.max(), 100)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=1.5)
            
            ax.set_title(f"S{secnow}, T{supertypenow}\nμ={mu:.2f}, σ={sigma:.2f}", fontsize=8)
            ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=6)
    
    plt.tight_layout()
    plt.savefig(PDF_DIR /f"{lipid_name}_distribution_grid.pdf", dpi=300)
    
def plot_parameter_traces(param_traces, losses, lipid_name):
    """
    Plot parameter traces and ELBO loss.
    
    Parameters:
    - param_traces: Parameter traces dictionary
    - losses: ELBO loss array
    - lipid_name: Name of the lipid
    """
    # Plot ELBO
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.title(f"ELBO Trace Plot for {lipid_name}")
    plt.savefig(PDF_DIR /f"{lipid_name}_elbo_trace.pdf")
    
    
    # Plot last 200 iterations of ELBO
    plt.figure(figsize=(10, 6))
    plt.plot(losses[-200:])
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.title(f"ELBO Trace Plot for {lipid_name} (Last 200 Iterations)")
    plt.savefig(PDF_DIR /f"{lipid_name}_elbo_trace_last200.pdf")
    
    
    # Initialize plot counters and grid
    plot_count = 0
    current_fig = None
    current_axes = None
    rows_per_figure = 5
    cols = 4
    
    # Function to setup a new figure when needed
    def setup_new_figure():
        fig_height = rows_per_figure * 2
        fig, axes = plt.subplots(rows_per_figure, cols, figsize=(16, fig_height))
        axes = axes.flatten()
        # Hide all axes initially
        for ax in axes:
            ax.set_visible(False)
        plt.tight_layout(pad=3.0)
        return fig, axes
    
    # Setup initial figure
    current_fig, current_axes = setup_new_figure()
    
    # Process each parameter
    for name, traces in param_traces.items():
        if traces.ndim == 1:
            # If we've filled the current figure, create a new one
            if plot_count >= rows_per_figure * cols:
                plt.tight_layout(pad=3.0)
                plt.savefig(PDF_DIR /f"{lipid_name}_param_traces_{plot_count//16}.pdf")
                
                current_fig, current_axes = setup_new_figure()
                plot_count = 0
                
            # Plot on the current subplot
            ax = current_axes[plot_count]
            ax.set_visible(True)
            ax.plot(traces)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel('Iteration', fontsize=8)
            ax.set_ylabel('Value', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)
            plot_count += 1
        else:
            # For multi-dimensional parameters, plot each component (every 20th)
            for idx in range(traces.shape[1]):
                if idx % 20 == 0:
                    # If we've filled the current figure, create a new one
                    if plot_count >= rows_per_figure * cols:
                        plt.tight_layout(pad=3.0)
                        plt.savefig(PDF_DIR /f"{lipid_name}_param_traces_{plot_count//16}.pdf")
                        
                        current_fig, current_axes = setup_new_figure()
                        plot_count = 0
                    
                    # Plot on the current subplot
                    ax = current_axes[plot_count]
                    ax.set_visible(True)
                    ax.plot(traces[:, idx])
                    ax.set_title(f"{name}[{idx}]", fontsize=10)
                    ax.set_xlabel('Iteration', fontsize=8)
                    ax.set_ylabel('Value', fontsize=8)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    plot_count += 1
    
    # Show the final figure if it has any plots
    if plot_count > 0:
        plt.tight_layout(pad=3.0)
        plt.savefig(PDF_DIR /f"{lipid_name}_param_traces_final.pdf")
        
def analyze_lipids(lipids, config, sub_alldata, subsample, testset):
    """
    Run the complete analysis for multiple lipids.
    
    Parameters:
    - lipids: List of lipid names to analyze
    - config: Configuration object
    - sub_alldata: Full data DataFrame
    - subsample: Training data indices
    - testset: Test data indices
    
    Returns:
    - Dictionary of results for each lipid
    """
    results = {}
    
    for lipid_name in lipids:
        print(f"\n{'='*50}\nAnalyzing lipid: {lipid_name}\n{'='*50}")
        
        # Normalize the lipid column
        sub_alldata_norm = normalize_lipid_column(
            sub_alldata.copy(), 
            lipid_name,
            lower_percentile=config.normalize_percentiles[0],
            upper_percentile=config.normalize_percentiles[1]
        )
        
        # Extract relevant columns
        sub_alldata_use = sub_alldata_norm[[lipid_name, "Condition", "Sample", "supertype", "SectionID"]]
        
        # Split into train and test sets
        test_df = sub_alldata_use.loc[testset.index,:]
        train_df = sub_alldata_use.loc[subsample.index,:]
        
        # Train the model
        svi, svi_state, param_traces, losses, train, mappingtable = train_lipid_model(
            train_df, 
            lipid_name, 
            num_epochs=config.num_epochs,
            learning_rate=config.learning_rate
        )
        
        # Plot parameter traces and ELBO
        plot_parameter_traces(param_traces, losses, lipid_name)
        
        # Analyze posterior
        samples_params, df_stats = analyze_posterior(svi, svi_state, train, lipid_name, mappingtable)
        
        # Evaluate on test set
        test_predictions, test_corr = evaluate_model(svi, svi_state, train_df, test_df, lipid_name)
        
        # Visualize distribution grid
        visualize_distribution_grid(samples_params, train, lipid_name)
        
        # Store results
        results[lipid_name] = {
            'svi': svi,
            'svi_state': svi_state,
            'param_traces': param_traces,
            'losses': losses,
            'samples_params': samples_params,
            'df_stats': df_stats,
            'test_predictions': test_predictions,
            'test_corr': test_corr,
            'train_df': train,
            'mappingtable': mappingtable
         }

        
        # Save model state (optional)
        final_params = svi.get_params(svi_state)
        np.save(str(PDF_DIR / f"{lipid_name}_model_params.npy"), final_params)
    
    return results

def prior_predictive_check(train_df, lipid_name):
    """
    Perform a prior predictive check for a given lipid.
    
    Parameters:
    - train_df: Training data DataFrame
    - lipid_name: Name of the lipid
    """
    # Prepare the data
    train, lipid_x, map_sample_to_condition, map_section_to_sample, supertype_code, section_code, condition_code = prepare_data(train_df, lipid_name)
    
    # Create predictive object
    predictive = Predictive(model_CASE_CONTROL_hierarchical, num_samples=25)
    
    # Generate samples from the prior
    prior_samples = predictive(
        random.PRNGKey(0),
        condition_code=condition_code,
        section_code=section_code,
        supertype_code=supertype_code,
        map_section_to_sample=map_section_to_sample,
        map_sample_to_condition=map_sample_to_condition,
        lipid_x=None
    )
    
    # Extract predictions
    predictions = prior_samples["obs"].mean(axis=0)
    predictions = np.array(predictions)
    predictions[predictions < 0] = 0
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, density=True, bins=30, alpha=0.5, label='Predicted')
    plt.hist(lipid_x, density=True, bins=30, alpha=0.5, label='Actual')
    plt.legend()
    plt.xlabel(lipid_name)
    plt.ylabel('Density')
    plt.title(f'Prior Predictive Check for {lipid_name}')
    plt.savefig(PDF_DIR /f"{lipid_name}_prior_predictive.pdf")
    
def main(sub_alldata, coords, config):
    """Main function to run the full analysis workflow."""
    # Set up the configuration
    #config.display_config()
    
    # Normalize lipid columns
    sub_alldata_processed = sub_alldata.copy()
    for lipid_name in config.lipids_to_analyze:
        sub_alldata_processed = normalize_lipid_column(
            sub_alldata_processed, 
            lipid_name,
            lower_percentile=config.normalize_percentiles[0],
            upper_percentile=config.normalize_percentiles[1]
        )
    
    # Create train/test sets
    subsample, testset = create_train_test_sets(
        coords, 
        seed=config.random_seed, 
        downsampling=config.downsampling
    )
    
    # Analyze nearest neighbors in subsampled data
    analyze_nearest_neighbors(subsample)
    
    # Visualize subsampling
    visualize_subsampling(coords, subsample)
    
    # Run prior predictive checks
    for lipid_name in config.lipids_to_analyze:
        sub_alldata_use = sub_alldata_processed[[lipid_name, "Condition", "Sample", "supertype", "SectionID"]]
        train_df = sub_alldata_use.loc[subsample.index,:]
        prior_predictive_check(train_df, lipid_name)
    
    # Run the full analysis
    results = analyze_lipids(
        config.lipids_to_analyze,
        config,
        sub_alldata_processed,
        subsample,
        testset
    )
    
    for lipid_name, res in results.items():
        # (a) merge all the individual PDFs into one
        prefix = f"{lipid_name.replace(' ','_')}_{cfg_string(config)}"
        pattern = f"{lipid_name}_*.pdf"
        files = sorted(PDF_DIR.glob(pattern))
        merger = PdfMerger()
        for p in files:
            merger.append(str(p))
        out_pdf = PDF_DIR / f"{prefix}_merged.pdf"
        merger.write(str(out_pdf))
        merger.close()
        # remove the now-redundant single-page files
        for p in files:
            p.unlink()

        train_df      = res['train_df']
        mappingtable  = res['mappingtable']
        draws         = res['samples_params']

        _, _, map_s2c, map_sec2samp, super_code, sec_code, cond_code = prepare_data(train_df, lipid_name)

        import numpy as _np
        import pandas as _pd

        def summarize_draws(arr, name, index_labels):
            """
            arr: np.ndarray, shape (n_draws, n_items)
            returns DataFrame with columns [parameter,index,mean,sd,ci_2.5,ci_97.5]
            """
            draws2d = arr.reshape(arr.shape[0], -1)  # (draws, N)
            rows = []
            for i in range(draws2d.shape[1]):
                col = draws2d[:, i]
                rows.append({
                    "parameter": name,
                    "index":   index_labels[i] if index_labels is not None else i,
                    "mean":    col.mean(),
                    "sd":      col.std(),
                    "ci_2.5":  _np.percentile(col, 2.5),
                    "ci_97.5": _np.percentile(col, 97.5),
                })
            return _pd.DataFrame(rows)
    
    return results

class CaseControlAnalysis:
    """
    A class for case-control analysis on spatial lipidomics data.
    """
    
    def __init__(self, adata: sc.AnnData, analysis_name="analysis"):
        """
        Initialize with an AnnData object containing spatial lipidomics data.
        
        Parameters
        ----------
        adata : sc.AnnData
            AnnData object with spatial lipidomics data
        analysis_name : str, optional
            Prefix for all output files. Default is "analysis".
        """
        self.adata = adata
        self.analysis_name = analysis_name
        # Set unified output directory
        self.casecontrol_dir = Path(os.getcwd()) / f"casecontrol_analysis_{self.analysis_name}"
        self.casecontrol_dir.mkdir(exist_ok=True)
        self.PDF_DIR = self.casecontrol_dir
    
    def run_case_control_analysis(
        self,
        lipids_to_analyze,
        learning_rate=0.05,
        num_epochs=2000,
        adaptive_lr=False,
        supertype_prior_std=1.0,
        supertype_shift_prior_std=1.0,
        sample_prior_std=1.0,
        section_prior_std=5.0,
        downsampling=1,
        random_seed=42,
        normalize_percentiles=(0.1, 99.9),
        guide_supertype_unconst_scale=0.1,
        guide_supertype_shift_scale=0.1,
        x_col="x",
        y_col="y",
        sectionid_col="SectionID",
        sample_col="Sample",
        condition_col="Condition",
        supertype_col="supertype",
        verbose=True
    ):
        """Run case-control analysis on spatial lipidomics data.

        This method performs a comprehensive case-control analysis on spatial lipidomics data stored in an AnnData object.
        It analyzes lipid expression patterns across different conditions, taking into account spatial information and
        lipizone supertypes.

        Parameters
        ----------
        lipids_to_analyze : list
            List of lipid names to analyze from the AnnData object.
        learning_rate : float, default=0.05
            Learning rate for the optimization process.
        num_epochs : int, default=2000
            Number of training epochs for the model.
        adaptive_lr : bool, default=False
            Whether to use adaptive learning rate during training.
        supertype_prior_std : float, default=1.0
            Standard deviation for the supertype-level prior distribution.
        supertype_shift_prior_std : float, default=1.0
            Standard deviation for the supertype shift prior distribution.
        sample_prior_std : float, default=1.0
            Standard deviation for the sample-level prior distribution.
        section_prior_std : float, default=5.0
            Standard deviation for the section-level prior distribution.
        downsampling : int, default=1
            Factor by which to downsample the data (1 means no downsampling).
        random_seed : int, default=42
            Random seed for reproducibility.
        normalize_percentiles : tuple, default=(0.1, 99.9)
            Percentiles for data normalization (lower, upper).
        guide_supertype_unconst_scale : float, default=0.1
            Scale parameter for unconstrained supertype guide.
        guide_supertype_shift_scale : float, default=0.1
            Scale parameter for supertype shift guide.
        x_col : str, default="x"
            Column name for x-coordinates in the AnnData object.
        y_col : str, default="y"
            Column name for y-coordinates in the AnnData object.
        sectionid_col : str, default="SectionID"
            Column name for section identifiers in the AnnData object.
        sample_col : str, default="Sample"
            Column name for sample identifiers in the AnnData object.
        condition_col : str, default="Condition"
            Column name for condition labels in the AnnData object.
        supertype_col : str, default="supertype"
            Column name for supertype labels in the AnnData object.
        verbose : bool, default=True
            Whether to print progress information.

        Returns
        -------
        dict
            Dictionary containing the analysis results, including:
            - Model parameters
            - Posterior distributions
            - Statistical summaries
            - Visualization data

        Raises
        ------
        ValueError
            If required columns are missing from the AnnData object.
            If no level columns are found to create supertype information.

        Notes
        -----
        - The method automatically creates a supertype column if it doesn't exist, using level columns
          from the AnnData object.
        - All outputs are saved to the casecontrol directory specified in the class initialization.
        """
        import pandas as pd
        obs = self.adata.obs.copy()
        lipid_df = pd.DataFrame(self.adata.X, columns=self.adata.var_names, index=self.adata.obs_names)
        df = pd.concat([obs, lipid_df], axis=1)
        if supertype_col not in df.columns:
            level_cols = [col for col in df.columns if col.startswith('level_')]
            if not level_cols:
                raise ValueError("No level columns found in AnnData.obs to create supertype.")
            level_cols = sorted(level_cols)
            max_level = min(8, len(level_cols))
            level_cols = level_cols[:max_level]
            df[supertype_col] = df[level_cols].apply(lambda x: '_'.join(x.astype(str).values), axis=1)
            print(f"Created supertype column from levels: {', '.join(level_cols)}")
        for col in [x_col, y_col, sectionid_col, sample_col, condition_col, supertype_col]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in AnnData.obs.")
        coords = df[[x_col, y_col, sectionid_col]].copy()
        coords.columns = ["x", "y", "SectionID"]
        class Config:
            pass
        config = Config()
        config.lipids_to_analyze = lipids_to_analyze
        config.learning_rate = learning_rate
        config.num_epochs = num_epochs
        config.adaptive_lr = adaptive_lr
        config.supertype_prior_std = supertype_prior_std
        config.supertype_shift_prior_std = supertype_shift_prior_std
        config.sample_prior_std = sample_prior_std
        config.section_prior_std = section_prior_std
        config.downsampling = downsampling
        config.random_seed = random_seed
        config.normalize_percentiles = normalize_percentiles
        config.guide_supertype_unconst_scale = guide_supertype_unconst_scale
        config.guide_supertype_shift_scale = guide_supertype_shift_scale
        global CASECONTROL_DIR
        global PDF_DIR
        CASECONTROL_DIR = self.casecontrol_dir
        PDF_DIR = self.casecontrol_dir
        results = main(df, coords, config)
        if verbose:
            print(f"All outputs saved to: {self.casecontrol_dir}")
        return results

    def summarize_case_control_results(
        self,
        lipids_to_analyze: List[str],
        supertypes: Optional[List[str]] = None,
        model_dir: Optional[str] = None,
        normalize_percentiles: Optional[Tuple[float, float]] = None,
        output_prefix: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Summarize case-control analysis results for specific lipids and supertypes."""
        from pathlib import Path
        if 'supertype' not in self.adata.obs.columns:
            level_cols = [col for col in self.adata.obs.columns if col.startswith('level_')]
            if not level_cols:
                raise ValueError("No level columns found in AnnData.obs to create supertype.")
            level_cols = sorted(level_cols)
            max_level = min(8, len(level_cols))
            level_cols = level_cols[:max_level]
            self.adata.obs['supertype'] = self.adata.obs[level_cols].apply(
                lambda x: '_'.join(x.astype(str).values), axis=1
            )
            print(f"Created supertype column from levels: {', '.join(level_cols)}")
        if supertypes is None:
            supertypes = np.sort(self.adata.obs['supertype'].unique())
        # Use the unified output directory by default
        if model_dir is None:
            model_dir = Path(self.casecontrol_dir)
        else:
            model_dir = Path(model_dir)
        aaaa_list, bbbb_list, lipid_names = [], [], []
        for lipid in lipids_to_analyze:
            param_path = model_dir / f"{lipid}_model_params.npy"
            if not param_path.exists():
                print(f"Warning: {param_path} not found, skipping.")
                continue
            params = np.load(param_path, allow_pickle=True).item()
            baseline = pd.DataFrame(
                sigmoid(params['alpha_supertype_unconst_loc']),
                index=supertypes, columns=[lipid]
            )
            shift = pd.DataFrame(
                params['alpha_supertype_shift_loc'],
                index=supertypes, columns=[lipid]
            )
            aaaa_list.append(shift)
            bbbb_list.append(baseline)
            lipid_names.append(lipid)
        if not aaaa_list:
            raise ValueError("No valid model parameters found. Please ensure you have run run_case_control_analysis with the specified lipids before calling summarize_case_control_results.")
        shift_df = pd.concat(aaaa_list, axis=1)
        baseline_df = pd.concat(bbbb_list, axis=1)
        shift_df.columns = lipid_names
        baseline_df.columns = lipid_names
        foldchange_df = shift_df / baseline_df
        # Save results in the unified output directory
        shift_df.to_parquet(model_dir / f"shift_{output_prefix}.parquet")
        baseline_df.to_parquet(model_dir / f"baseline_{output_prefix}.parquet")
        foldchange_df.to_parquet(model_dir / f"foldchange_{output_prefix}.parquet")
        shift_df.to_csv(model_dir / f"shift_{output_prefix}.csv")
        baseline_df.to_csv(model_dir / f"baseline_{output_prefix}.csv")
        foldchange_df.to_csv(model_dir / f"foldchange_{output_prefix}.csv")
        print(f"Saved shift, baseline, and foldchange to {model_dir}")
        return shift_df, baseline_df, foldchange_df

    def summarize_xsupertypes(
        self,
        lipids_to_analyze,
        supertypes=None,
        model_dir=None,
        output_prefix="CASE_CONTROL",
        baseline_condition="naive",
        upreg_threshold=0.2,
        prob_threshold=0.98,
        n_samples=1000
    ):
        """
        For each lipid, compute upregulated, downregulated, and expressed supertypes using model parameter files.
        Returns four DataFrames: upreg, downreg, expressed, and their union (all supertypes with any shift),
        as well as lists: mean_score, ci_lowers, ci_uppers for the fraction of expressed supertypes that are also shifted (with 95% bootstrap CI).
        Saves all as parquet files in the casecontrol_analysis directory.
        """
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        from pathlib import Path
        from scipy.special import expit

        # Use the unified output directory by default
        if model_dir is None:
            model_dir = self.casecontrol_dir
        model_dir = Path(model_dir)

        if supertypes is None:
            supertypes = np.sort(self.adata.obs['supertype'].unique())

        lipid_upreg_xsupertypes = []
        lipid_downreg_xsupertypes = []
        lipid_expressed_xsupertypes = []
        lipid_names = []

        for lipid in tqdm(lipids_to_analyze):
            param_path = model_dir / f"{lipid}_model_params.npy"
            if not param_path.exists():
                print(f"Warning: {param_path} not found, skipping.")
                continue
            params = np.load(param_path, allow_pickle=True).item()
            loc_susc = params['alpha_supertype_shift_loc']
            scale_susc = params['alpha_supertype_shift_scale']
            loc_unconst = params['alpha_supertype_unconst_loc']
            scale_unconst = params['alpha_supertype_unconst_scale']

            # Draw samples for shift and baseline
            samples_shift = np.random.default_rng(1234).normal(
                loc=loc_susc[None, :],
                scale=scale_susc[None, :],
                size=(n_samples, loc_susc.shape[0])
            )
            samples_unconst = np.random.default_rng(1234).normal(
                loc=loc_unconst[None, :],
                scale=scale_unconst[None, :],
                size=(n_samples, loc_unconst.shape[0])
            )
            samples_baseline = expit(samples_unconst)

            # Upregulation: shift > threshold * baseline
            upregulation = samples_shift > (upreg_threshold * samples_baseline)
            downregulation = -samples_shift > (upreg_threshold * samples_baseline)
            expressed = samples_baseline > 0.05

            lipid_upreg_xsupertypes.append((np.mean(upregulation, axis=0) > prob_threshold).astype(float))
            lipid_downreg_xsupertypes.append((np.mean(downregulation, axis=0) > prob_threshold).astype(float))
            lipid_expressed_xsupertypes.append((np.mean(expressed, axis=0) > prob_threshold).astype(float))
            lipid_names.append(lipid)

        stindex = supertypes
        lipid_upreg_xsupertypes_df = pd.DataFrame(lipid_upreg_xsupertypes, columns=stindex, index=lipid_names)
        lipid_downreg_xsupertypes_df = pd.DataFrame(lipid_downreg_xsupertypes, columns=stindex, index=lipid_names)
        lipid_expressed_xsupertypes_df = pd.DataFrame(lipid_expressed_xsupertypes, columns=stindex, index=lipid_names)
        
        # Convert to boolean before OR operation
        shifted_xsupertypes_df = (lipid_upreg_xsupertypes_df.astype(bool) | lipid_downreg_xsupertypes_df.astype(bool))

        # Save results in the unified output directory
        lipid_upreg_xsupertypes_df.to_parquet(model_dir / f"upreg_xsupertypes_{output_prefix}.parquet")
        lipid_downreg_xsupertypes_df.to_parquet(model_dir / f"downreg_xsupertypes_{output_prefix}.parquet")
        lipid_expressed_xsupertypes_df.to_parquet(model_dir / f"expressed_xsupertypes_{output_prefix}.parquet")
        shifted_xsupertypes_df.to_parquet(model_dir / f"shifted_xsupertypes_{output_prefix}.parquet")
        # Also save as CSV for convenience
        lipid_upreg_xsupertypes_df.to_csv(model_dir / f"upreg_xsupertypes_{output_prefix}.csv")
        lipid_downreg_xsupertypes_df.to_csv(model_dir / f"downreg_xsupertypes_{output_prefix}.csv")
        lipid_expressed_xsupertypes_df.to_csv(model_dir / f"expressed_xsupertypes_{output_prefix}.csv")
        shifted_xsupertypes_df.to_csv(model_dir / f"shifted_xsupertypes_{output_prefix}.csv")

        print(f"Saved upreg, downreg, expressed, and shifted xsupertypes to {model_dir}")

        # --- Downstream: Bootstrap mean and CI for fraction of expressed supertypes that are also shifted ---
        mean_score = []
        ci_lowers = []
        ci_uppers = []
        for yyy in tqdm(range(lipid_expressed_xsupertypes_df.shape[0])):
            exp = lipid_expressed_xsupertypes_df.iloc[yyy, :].values.astype(bool)
            shif = shifted_xsupertypes_df.iloc[yyy, :].values.astype(bool)
            n, B = len(exp), 10000
            rng = np.random.default_rng(42)
            # Bootstrap replicates
            scores = [
                (exp[idx] & shif[idx]).sum() / exp[idx].sum() if exp[idx].sum() > 0 else 0.0
                for idx in rng.integers(0, n, size=(B, n))
            ]
            # Point estimate and 95% CI
            mean_score.append(np.mean(scores))
            ci_lower, ci_upper = np.percentile(scores, [2.5, 97.5])
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)

        return (
            lipid_upreg_xsupertypes_df,
            lipid_downreg_xsupertypes_df,
            lipid_expressed_xsupertypes_df,
            shifted_xsupertypes_df,
            mean_score,
            ci_lowers,
            ci_uppers
        )

    def plot_comodulation_heatmap(
        self,
        shift,
        baseline,
        expressed,
        shifted,
        ddf,
        baseline_condition="naive",
        supertype_col="supertype",
        todrop_supertypes=None,
        todrop_lipids=None,
        k_row=16,
        thresh=0.5,
        threshold=None,  # Add threshold parameter for backward compatibility
        output_filename=None,
        figsize=(16, 10)
    ):
        """
        Create a comprehensive comodulation heatmap showing log2 fold changes with optimal leaf ordering.
        
        Parameters
        ----------
        shift : pd.DataFrame
            DataFrame of shifts
        baseline : pd.DataFrame
            DataFrame of baseline values
        expressed : pd.DataFrame
            DataFrame of expressed values
        shifted : pd.DataFrame
            DataFrame of shifted values
        ddf : pd.DataFrame
            DataFrame with lipid properties
        baseline_condition : str, optional
            Name of baseline condition, by default "naive"
        supertype_col : str, optional
            Name of supertype column, by default "supertype"
        todrop_supertypes : list, optional
            List of supertypes to drop, by default None
        todrop_lipids : list, optional
            List of lipids to drop, by default None
        k_row : int, optional
            Number of row clusters, by default 16
        thresh : float, optional
            Threshold for dense rows/columns, by default 0.5
        threshold : float, optional
            Alias for thresh, by default None
        output_filename : str, optional
            Name of output file, by default None
        figsize : tuple, optional
            Figure size, by default (16, 10)
        """
        # Use threshold if provided, otherwise use thresh
        thresh = threshold if threshold is not None else thresh
        
        if output_filename is None:
            output_filename = f"{self.analysis_name}_overview_CASE_CONTROL_shifts.pdf"
        
        # Calculate log2 fold changes
        shifts = np.log2((shift + baseline) / baseline).fillna(0)
        
        # Apply expressed and shifted filtering
        exp = expressed.astype(bool)
        shif = shifted.astype(bool)
        # Zero out non-expressed or non-shifted values
        shifts[~(exp & shif)] = 0.0
        
        # Apply filtering
        if todrop_supertypes is None:
            todrop_supertypes = []
        if todrop_lipids is None:
            todrop_lipids = []
        
        # Drop specified supertypes and lipids
        shifts = shifts.drop(todrop_supertypes, errors='ignore')
        shifts = shifts.drop(todrop_lipids, axis=1, errors='ignore')
        
        # Get control condition data for normalization
        sub_alldata = self.adata.to_df()
        sub_alldata[self.adata.obs.columns] = self.adata.obs
        
        df = sub_alldata.copy().loc[sub_alldata['Condition'] == baseline_condition, :]
        features = shifts.columns.tolist()
        
        # Clip and normalize features
        lower = df[features].quantile(0.005)
        upper = df[features].quantile(0.995)
        df_clipped = df.copy()
        df_clipped[features] = df_clipped[features].clip(lower=lower, upper=upper, axis=1)
        df_clipped[features] = (df_clipped[features] - lower) / (upper - lower)
        centroids = df_clipped.groupby(supertype_col)[features].mean()
        
        # Reorder with optimal leaf ordering
        (df_opt, row_L, col_L,
         ordered_rows, ordered_cols,
         rows_dense, cols_dense,
         ordered_rows_main) = optimal_reorder_dataframe_cosine_clean(shifts,
                                                                     method='weighted',
                                                                     thresh=thresh)
        
        # Compute clusters on the main block
        clusters_main = sch.fcluster(row_L, t=k_row, criterion='maxclust')
        # order those clusters according to ordered_rows_main
        df_main = shifts.loc[~rows_dense, ~cols_dense]
        clusters_main_ordered = [
            clusters_main[df_main.index.get_loc(lbl)]
            for lbl in ordered_rows_main
        ]
        
        # Build full cluster array, tagging held-aside rows as cluster k_row+1
        n_dense_rows = rows_dense.sum()
        misc_cluster = k_row + 1
        row_clusters_full = np.concatenate([
            clusters_main_ordered,
            np.full(n_dense_rows, misc_cluster, dtype=int)
        ])
        
        # Extract lipid-class colors from provided ddf
        lipid_colors = [ddf.loc[col, 'color'] if col in ddf.index else "#888888"
                        for col in df_opt.columns]
        
        # Create subclass color dictionary from adata
        unique_supertypes = self.adata.obs[supertype_col].unique()
        supertype_to_subclass = {}
        
        # Generate subclass colors
        unique_supertype_list = sorted(unique_supertypes)
        subclass_color_list = generate_distinct_colors(len(unique_supertype_list))
        for i, st in enumerate(unique_supertype_list):
            color_rgba = subclass_color_list[i]
            # Convert to hex
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(color_rgba[0] * 255), 
                int(color_rgba[1] * 255), 
                int(color_rgba[2] * 255)
            )
            supertype_to_subclass[st] = hex_color
        
        # Create color arrays for rows
        subclass_colors = [supertype_to_subclass.get(row_idx, "#888888") for row_idx in df_opt.index]
        
        # Convert hex colors to RGBA
        subclass_rgba = [to_rgba(c) for c in subclass_colors]
        
        # Calculate nonzero counts for each row in the reordered dataframe
        nonzero_counts = (df_opt != 0).sum(axis=1).values
        
        # Plot heatmap + sidebars
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(df_opt.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # Create divider for multiple sidebars
        divider = make_axes_locatable(ax)
        
        # Subclass color sidebar (leftmost)
        cax_subclass = divider.append_axes("left", size="2%", pad=0.05)
        cax_subclass.imshow(np.array(subclass_rgba)[:, None, :], aspect='auto')
        cax_subclass.set_xticks([])
        cax_subclass.set_yticks([])
        cax_subclass.set_ylabel("Subclass", rotation=0, ha='right', va='center')
        
        # Row‐cluster sidebar (second from left, with extra color for dense rows)
        distinct = generate_distinct_colors(k_row + 1)
        distinct[-1] = to_rgba("gray")
        cmap_row = ListedColormap(distinct)
        
        cax_clusters = divider.append_axes("left", size="2%", pad=0.05)
        cax_clusters.imshow(row_clusters_full[:, None], aspect='auto',
                           cmap=cmap_row, vmin=1, vmax=k_row+1)
        cax_clusters.set_xticks([])
        cax_clusters.set_yticks([])
        cax_clusters.set_ylabel("Clusters", rotation=0, ha='right', va='center')
        
        # Lipid‐class sidebar (top)
        rgba = [to_rgba(c) for c in lipid_colors]
        cax_top = divider.append_axes("top", size="2%", pad=0.05)
        cax_top.imshow([rgba], aspect='auto')
        cax_top.set_xticks([])
        cax_top.set_yticks([])
        
        # Nonzero count bar plot (right side)
        cax_bar = divider.append_axes("right", size="15%", pad=0.1)
        y_positions = np.arange(len(nonzero_counts))
        
        # Clip counts at 50
        clipped_counts = np.clip(nonzero_counts, 0, 50)
        
        # Get cluster colors for each bar
        bar_colors = [distinct[cluster - 1] for cluster in row_clusters_full]
        
        cax_bar.barh(y_positions, clipped_counts, height=0.8, color=bar_colors, alpha=0.8)
        cax_bar.set_ylim(-0.5, len(nonzero_counts) - 0.5)
        cax_bar.set_xlim(0, 50)
        cax_bar.invert_yaxis()  # Match heatmap orientation
        cax_bar.set_xlabel("Nonzero Count (max 50)")
        cax_bar.set_yticks([])  # Remove y-axis labels
        cax_bar.grid(True, alpha=0.3, axis='x')
        
        # Draw boundaries only within the main block
        bounds = np.where(np.diff(row_clusters_full[:len(ordered_rows_main)]) != 0)[0] + 0.5
        for b in bounds:
            ax.axhline(b, color='white', linewidth=1.5)
            cax_clusters.axhline(b, color='white', linewidth=1.5)
            # Optionally add boundaries to other sidebars too
            cax_subclass.axhline(b, color='white', linewidth=0.5, alpha=0.7)
            # Add boundaries to bar plot
            cax_bar.axhline(b, color='white', linewidth=0.5, alpha=0.7)
        
        ax.set_xlabel("Lipids (Cosine OLO - colored by class)")
        ax.set_ylabel("Supertypes (Cosine OLO - clustered)")
        ax.set_title(f"Optimal Leaf Ordering - Both Dimensions\n"
                     f"Cosine Distance + Weighted Linkage ({k_row} clusters)")
        
        plt.tight_layout()
        plt.savefig(output_filename)
        plt.show()
        
        # Summary
        print(f"Main‐block rows clustered: {len(ordered_rows_main)}")
        print(f"Held‐aside dense rows: {rows_dense.sum()}")
        print(f"Held‐aside dense columns: {cols_dense.sum()}")
        unique, counts = np.unique(row_clusters_full[:len(ordered_rows_main)], return_counts=True)
        print("Cluster sizes (main block):")
        for u, c in zip(unique, counts):
            print(f" Cluster {u}: {c} rows")
        
        # ===== SPATIAL VISUALIZATION OF COMODULATION CLUSTERS =====
        
        # Alternative method without matplotlib (manual conversion)
        def rgba_to_hex(rgba_array):
            """Convert RGBA array (values 0-1) to hex string"""
            r = int(rgba_array[0] * 255)
            g = int(rgba_array[1] * 255)
            b = int(rgba_array[2] * 255)
            return f"#{r:02x}{g:02x}{b:02x}"

        # Using manual method
        bar_colors_hex_manual = [rgba_to_hex(rgba) for rgba in bar_colors]
        colors = pd.Series(bar_colors_hex_manual, index=df_opt.index)
        clusters_series = pd.Series(row_clusters_full, index=df_opt.index)
        loool = pd.DataFrame((np.abs(shifts) > 0.2).sum(axis=1).sort_values(), columns = ['nmod'])
        loool['linkage'] = loool.index.map(clusters_series)
        loool['color'] = loool.index.map(colors)
        loool['linkage'].value_counts()
        loool.loc[loool['linkage'].isin(loool['linkage'].value_counts().index[loool['linkage'].value_counts() <= 4]), 'linkage'] = misc_cluster
        loool['linkage'].value_counts() # cluster "misc_cluster" is miscellaneous
        clusters_series = loool['linkage'][loool['linkage'] != misc_cluster]
        colors_series = loool['color'][loool['linkage'] != misc_cluster]
        
        # Extract metadata from adata
        metadata = self.adata.obs.copy()
        Linkage_clusters = clusters_series.unique()
        Linkage_colors = colors_series.unique()
        color_df = pd.DataFrame({
            'Linkage Cluster': Linkage_clusters,
            'Color': Linkage_colors
        })
        
        color_df.to_csv(f"{self.analysis_name}_color_df_CASE_CONTROL.csv")
        metadata['cluster_variation'] = metadata[supertype_col].map(clusters_series).fillna("lightgray")
        color_df.index = color_df['Linkage Cluster']
        metadata['cluster_variation_color'] = metadata['cluster_variation'].map(color_df['Color'])
        
        # Create spatial grid visualization
        unique_samples = sorted(metadata['Sample'].unique())
        unique_sections = sorted(metadata['SectionPlot'].unique())
        
        fig2, axes = plt.subplots(6, 6, figsize=(20, 12))
        
        for sample_idx, sample in enumerate(unique_samples[:6]):
            for section_idx, section in enumerate(unique_sections[:6]):
                ax = axes[sample_idx, section_idx]
        
                ddf_spatial = metadata[
                    (metadata['Sample'] == sample) & 
                    (metadata['SectionPlot'] == section)
                ]
        
                ax.scatter(
                    ddf_spatial['y'], 
                    -ddf_spatial['x'], 
                    c=ddf_spatial['cluster_variation_color'].astype(object).fillna('#CCCCCC').tolist(),  
                    s=0.5, 
                    rasterized=True
                )
        
                ax.axis('off')
                ax.set_aspect('equal')
        
                ax.set_title(f'Sample {sample}, Section {section}', fontsize=8)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.savefig(f"{self.analysis_name}_comodulation_clusters.pdf")
        plt.show()
        
        # Add comodulation clusters to adata.obs
        self.adata.obs['comodulation_cluster'] = self.adata.obs[supertype_col].map(clusters_series).fillna(misc_cluster)
        self.adata.obs['comodulation_cluster_color'] = self.adata.obs[supertype_col].map(colors_series).fillna('#CCCCCC')

    def compute_edge_modulation_scores(
        self,
        foldchanges: pd.DataFrame,
        metabolicmodule: pd.DataFrame,
        comodulation_clusters: dict,
        min_component_size: int = 2
    ) -> dict:
        """
        Compute edge modulation scores per cluster, with pre-filtering to avoid NaNs,
        and restrict to connected components of the metabolic network with at least
        `min_component_size` lipids.

        Parameters
        ----------
        foldchanges : pd.DataFrame
            DataFrame of shape (n_supertypes, n_lipids) containing log2FC values.
            Rows are supertypes, columns are lipids.
        metabolicmodule : pd.DataFrame
            Adjacency boolean matrix (lipids x lipids) indicating metabolic edges.
        comodulation_clusters : dict
            Mapping from cluster labels to lists of supertypes in each cluster.
        min_component_size : int
            Minimum number of lipids in a connected component to keep.

        Returns
        -------
        modulation_scores : dict
            Dictionary mapping each cluster label to a DataFrame (lipids x lipids)
            of modulation scores for edges within that cluster.
        """
        # 0. Prefilter: keep only lipids present in both foldchanges and metabolicmodule
        common_lipids = foldchanges.columns.intersection(metabolicmodule.index)
        fc = foldchanges[common_lipids]
        adj = metabolicmodule.loc[common_lipids, common_lipids].astype(int)

        # Drop lipids with zero variance across supertypes to avoid NaNs in Z
        stds = fc.std(axis=0)
        zero_std = stds[stds == 0].index
        if len(zero_std) > 0:
            fc = fc.drop(columns=zero_std)
            adj = adj.drop(index=zero_std, columns=zero_std)

        # Further prefilter: keep only connected components with >= min_component_size
        G0 = nx.from_pandas_adjacency(adj)
        large_comps = [comp for comp in nx.connected_components(G0) if len(comp) >= min_component_size]
        large_lipids = set().union(*large_comps)
        # Restrict dataframes to those lipids
        fc = fc[sorted(large_lipids)]
        adj = adj.loc[sorted(large_lipids), sorted(large_lipids)]

        # 1. Compute Z-scores per lipid across all supertypes
        zscores = fc.sub(fc.mean(axis=0), axis=1).div(fc.std(axis=0), axis=1)

        # 2. Precompute absolute Z-scores and adjacency
        abs_z = zscores.abs().T  # shape = (lipids x supertypes)
        abs_z[fc.T == 0] = 0  # what is zero should stay zero!

        # 3. Compute per-supertype edge scores
        edge_scores = {}
        for supertype in zscores.index:
            arr = abs_z[supertype].values
            edge_vals = arr[:, None]*arr[None, :]  # use the product
            edge_mat = pd.DataFrame(edge_vals, index=abs_z.index, columns=abs_z.index)
            # mask by adjacency
            edge_scores[supertype] = edge_mat * adj

        # 4. Aggregate by cluster: mean of edge scores across supertypes in each cluster
        modulation_scores = {}
        for cluster_label, members in comodulation_clusters.items():
            valid = [m for m in members if m in edge_scores]
            if not valid:
                raise ValueError(f"No valid supertypes in cluster {cluster_label} after filtering.")
            summed = sum(edge_scores[m] for m in valid)
            modulation_scores[cluster_label] = summed / len(valid)

        return modulation_scores

    def plot_modulation_thumbnails(
        self,
        modulation_scores: dict,
        metabolicmodule: pd.DataFrame,
        ddf: pd.DataFrame,
        thresholds=0.1,
        k=0.2,
        palette=None,
        max_width=5,
        figsize_per_plot=(3, 3),
        output_filename=None
    ):
        """
        Draw one thumbnail per cluster:
        - Nodes in fixed layout (no labels)
        - Edges only where score >= threshold for that cluster
        - Edge width ∝ comodulation value
        - Edge color = cluster color
        - Connected nodes colored by ddf dataframe, others gray

        Parameters
        ----------
        modulation_scores : dict[str, pd.DataFrame]
            For each cluster c, a square DataFrame of com-modulation scores.
        metabolicmodule : pd.DataFrame
            Adjacency matrix of the underlying network (indexed by lipid).
        ddf : pd.DataFrame
            DataFrame with 'lipid_name' and 'color' columns for node coloring.
        thresholds : float or dict[str, float]
            If float, same threshold for all clusters; otherwise map cluster→threshold.
        k : float
            spring_layout "k" parameter.
        palette : dict[str, color] or None
            map cluster→hex color; defaults to 8-color palette.
        max_width : float
            Maximum edge width (for the largest score in each thumbnail).
        figsize_per_plot : tuple
            Size (w,h) of each thumbnail; total fig size calculated based on grid.
        output_filename : str
            Output filename for the saved plot. If None, uses analysis_name prefix.
        """
        if output_filename is None:
            output_filename = f"{self.analysis_name}_comodclusters.pdf"
        # 1. Find common lipids & build graph
        common = set.intersection(*(set(df.index) for df in modulation_scores.values()))
        common_lipids = [L for L in metabolicmodule.index if L in common]
        adj = metabolicmodule.loc[common_lipids, common_lipids].astype(int)
        G = nx.from_pandas_adjacency(adj)

        # 2. Compute one layout
        pos = nx.spring_layout(G, seed=42, k=k, iterations=50)

        # 3. Prepare clusters, thresholds, colors
        clusters = list(modulation_scores.keys())
        if isinstance(thresholds, dict):
            thr_map = thresholds
        else:
            thr_map = {c: thresholds for c in clusters}

        if palette is None:
            base = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
            palette = {c: base[i % len(base)] for i, c in enumerate(clusters)}
        else:
            # ensure every cluster has a color
            palette = {c: palette.get(c, '#CCCCCC') for c in clusters}

        # 4. Create lipid to color mapping from ddf
        lipid_to_color = dict(zip(ddf['lipid_name'], ddf['color']))

        # 5. Calculate grid layout (3 columns)
        n = len(clusters)
        ncols = 3
        nrows = (n + ncols - 1) // ncols  # ceiling division
        
        # Scale node and edge sizes based on grid
        node_size = max(4, 12 - nrows)  # smaller nodes for more rows
        scaled_max_width = max(1, max_width * (8 / max(8, nrows)))  # scale edge width

        fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_plot[0]*ncols, figsize_per_plot[1]*nrows))
        
        # Handle case where we have only one row or one subplot
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        # Flatten axes for easier iteration if it's 2D
        if nrows > 1 or ncols > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = axes

        # 6. Draw each thumbnail
        for i, c in enumerate(clusters):
            ax = axes_flat[i]
            
            # select edges above threshold
            thr = thr_map[c]
            scores = modulation_scores[c]
            # only consider existing edges in G
            edges = [(u, v) for u, v in G.edges() if not np.isnan(scores.loc[u, v]) and scores.loc[u, v] >= thr]
            # get corresponding scores
            values = np.array([scores.loc[u, v] for u, v in edges])
            if len(values) > 0:
                # scale widths so max → scaled_max_width
                widths = (values / values.max()) * scaled_max_width
            else:
                widths = []

            # Find connected nodes
            connected_nodes = set()
            for u, v in edges:
                connected_nodes.add(u)
                connected_nodes.add(v)

            # Color nodes: connected nodes get color from ddf, others are gray
            node_colors = []
            node_sizes = []
            for node in G.nodes():
                if node in connected_nodes and node in lipid_to_color:
                    node_colors.append(lipid_to_color[node])
                    node_sizes.append(node_size * 8)  # 4x larger for connected nodes
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(node_size)  # regular size for unconnected nodes

            # draw nodes and edges (edges in black)
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax)
            if edges:
                nx.draw_networkx_edges(G, pos, edgelist=edges, width=widths, edge_color='black', ax=ax)
            
            # Add colored circle around the layout for cluster identification
            # Get the bounding box of the layout
            if pos:
                x_coords = [pos[node][0] for node in pos]
                y_coords = [pos[node][1] for node in pos]
                center_x = (max(x_coords) + min(x_coords)) / 2
                center_y = (max(y_coords) + min(y_coords)) / 2
                radius = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)) / 2 * 1.2
                
                circle = plt.Circle((center_x, center_y), radius, fill=False, 
                                  color=palette[c], linewidth=3, alpha=0.8, clip_on=False)
                ax.add_patch(circle)
            
            ax.set_aspect('equal')    
            ax.set_axis_off()

        # Hide unused subplots
        for i in range(n, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=0.5)
        plt.savefig(output_filename)
        plt.show()

    def differential_lipids(
        self,
        samples_to_keep: list,
        group_col: str,
        group1: str,
        group2: str,
        lipid_props_df: pd.DataFrame,
        min_fc: float = 0.2,
        pthr: float = 0.05,
        show_inline: bool = True,
        output_filename: str = None
    ) -> pd.DataFrame:
        """
        Perform differential testing between two groups for lipids in the AnnData object.
        
        Parameters
        ----------
        samples_to_keep : list
            List of sample names to include in the analysis
        group_col : str
            Column name in adata.obs to use for grouping ('acronym', 'lipizone', or 'level_x')
        group1 : str
            First group label to compare
        group2 : str
            Second group label to compare
        lipid_props_df : pd.DataFrame
            DataFrame containing lipid properties including colors
        min_fc : float, optional
            Minimum fold change threshold, by default 0.2
        pthr : float, optional
            P-value threshold for significance, by default 0.05
        show_inline : bool, optional
            Whether to show the plot inline, by default True
        output_filename : str, optional
            Output filename for the plot, by default None (uses analysis_name prefix)
            
        Returns
        -------
        pd.DataFrame
            Results of differential testing with log2 fold changes and p-values
        """
        import numpy as np
        from scipy.stats import mannwhitneyu
        from statsmodels.stats.multitest import multipletests
        from adjustText import adjust_text
        import matplotlib.pyplot as plt
        
        # Filter data to keep only specified samples
        mask = self.adata.obs['Sample'].isin(samples_to_keep)
        adata_subset = self.adata[mask].copy()
        
        # Create boolean masks for the two groups
        mask1 = adata_subset.obs[group_col] == group1
        mask2 = adata_subset.obs[group_col] == group2
        
        # Extract lipid data
        lipid_data = pd.DataFrame(
            adata_subset.X,
            columns=adata_subset.var_names,
            index=adata_subset.obs_names
        )
        
        # Subset the data into two groups
        groupA = lipid_data.loc[mask1]
        groupB = lipid_data.loc[mask2]
        
        results = []
        for col_name in lipid_data.columns:
            dataA = groupA[col_name].dropna()
            dataB = groupB[col_name].dropna()
            
            # Compute group means and log2 fold change
            meanA = np.mean(dataA) + 1e-11  # avoid division by zero
            meanB = np.mean(dataB) + 1e-11
            log2fc = np.log2(meanB / meanA)
            
            # Mann-Whitney U test
            try:
                _, pval = mannwhitneyu(dataA, dataB, alternative='two-sided')
            except ValueError:
                # Occurs if one group is all identical values, etc.
                pval = np.nan
            
            results.append({
                'lipid': col_name,
                'meanA': meanA,
                'meanB': meanB,
                'log2fold_change': log2fc,
                'p_value': pval
            })
        
        results_df = pd.DataFrame(results)
        
        # Multiple-testing correction
        reject, pvals_corrected, _, _ = multipletests(
            results_df['p_value'].values,
            alpha=pthr,
            method='fdr_bh'
        )
        results_df['p_value_corrected'] = pvals_corrected
        
        # Create visualization
        if output_filename is None:
            output_filename = f"{self.analysis_name}_differential_lipids.pdf"
            
        # Filter significant results and sort
        dfff = results_df.loc[results_df['p_value_corrected'] < pthr].sort_values(by="log2fold_change")
        colors = lipid_props_df.loc[dfff['lipid'], 'color'].fillna("black")
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(dfff)), dfff['log2fold_change'], color=colors)
        
        # Select indices to label
        n_items = len(dfff)
        bottom_5 = list(range(5))
        top_5 = list(range(n_items-5, n_items))
        middle_start = 5
        middle_end = n_items - 5
        middle_5 = list(np.random.choice(range(middle_start, middle_end), 5, replace=False))
        indices_to_label = sorted(bottom_5 + middle_5 + top_5)
        
        # Add labels
        texts = []
        for idx in indices_to_label:
            x = idx
            y = dfff.iloc[idx]['log2fold_change']
            texts.append(plt.text(x, y, dfff.iloc[idx]['lipid'], fontsize=8))
        
        # Adjust text positions to avoid overlap
        adjust_text(texts)
        
        # Clean up plot
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlabel("Sorted lipid species")
        plt.xticks([])
        plt.tight_layout()
        
        if output_filename:
            plt.savefig(output_filename)
        if show_inline:
            plt.show()
        else:
            plt.close()
            
        return results_df

# Helper functions (standalone utilities)
def cosine_clean(u, v):
    """
    Cosine distance computed only on positions where u or v is nonzero.
    - If neither has any nonzero entries: distance = 0.0
    - If one has nonzeros but the other is zero on all those positions: distance = 1.0
    """
    mask = (u != 0) | (v != 0)
    if not mask.any():
        return 0.0
    u2, v2 = u[mask], v[mask]
    nu, nv = np.linalg.norm(u2), np.linalg.norm(v2)
    if nu == 0 or nv == 0:
        return 1.0
    sim = np.dot(u2, v2) / (nu * nv)
    return 1.0 - sim


def optimal_reorder_dataframe_cosine_clean(df, method='weighted', thresh=0.8):
    """
    Reorder a dataframe using optimal leaf ordering on cosine distances.
    Handles empty dataframes and provides better error messages.
    """
    # Check if dataframe is empty
    if df.empty:
        raise ValueError("Input dataframe is empty. Please check your data and filtering criteria.")
    
    # Check if all values are zero or NaN
    if df.isna().all().all() or (df == 0).all().all():
        raise ValueError("All values in the dataframe are zero or NaN. Please check your data and filtering criteria.")
    
    # 1) Identify dense rows and columns
    row_frac = (df != 0).mean(axis=1)
    col_frac = (df != 0).mean(axis=0)
    
    # Print fraction distribution for debugging
    print(f"\nFraction distribution:")
    print(f"Rows: min={row_frac.min():.3f}, 25%={row_frac.quantile(0.25):.3f}, median={row_frac.median():.3f}, 75%={row_frac.quantile(0.75):.3f}, max={row_frac.max():.3f}")
    print(f"Columns: min={col_frac.min():.3f}, 25%={col_frac.quantile(0.25):.3f}, median={col_frac.median():.3f}, 75%={col_frac.quantile(0.75):.3f}, max={col_frac.max():.3f}")
    
    rows_dense = row_frac > thresh
    cols_dense = col_frac > thresh
    
    # Check if we have any non-dense rows/columns
    if not (~rows_dense).any() or not (~cols_dense).any():
        raise ValueError(
            f"No rows/columns have fraction <= {thresh}. "
            f"Current fractions: rows [{row_frac.min():.3f}, {row_frac.max():.3f}], "
            f"columns [{col_frac.min():.3f}, {col_frac.max():.3f}]. "
            "Try increasing the threshold."
        )
    
    # 2) Reorder main block
    df_main = df.loc[~rows_dense, ~cols_dense]
    
    # Check if main block is empty
    if df_main.empty:
        raise ValueError(
            f"After filtering dense rows/columns (threshold={thresh}), no data remains. "
            f"Current fractions: rows [{row_frac.min():.3f}, {row_frac.max():.3f}], "
            f"columns [{col_frac.min():.3f}, {col_frac.max():.3f}]. "
            "Try adjusting the threshold."
        )
    
    # 2a) column clustering on main
    col_d = pdist(df_main.T.values, metric=cosine_clean)
    col_L = sch.linkage(col_d, method=method, optimal_ordering=True)
    col_order = sch.leaves_list(col_L)
    
    # 2b) row clustering on main
    row_d = pdist(df_main.values, metric=cosine_clean)
    row_L = sch.linkage(row_d, method=method, optimal_ordering=True)
    row_order = sch.leaves_list(row_L)
    
    # 3) Reorder everything
    ordered_rows = df_main.index[row_order].tolist()
    ordered_cols = df_main.columns[col_order].tolist()
    
    # Add dense rows/columns at the end
    ordered_rows.extend(df.index[rows_dense])
    ordered_cols.extend(df.columns[cols_dense])
    
    # Reorder the full dataframe
    df_opt = df.loc[ordered_rows, ordered_cols]
    
    return (
        df_opt,
        row_L,
        col_L,
        ordered_rows,
        ordered_cols,
        rows_dense,
        cols_dense,
        ordered_rows[:len(row_order)]
    )


def generate_distinct_colors(n):
    """Generate n visually distinct colors."""
    if n <= 20:
        return plt.cm.tab20(np.linspace(0, 1, min(n, 20)))
    hues = np.linspace(0, 1, n, endpoint=False)
    return [plt.cm.hsv(h) for h in hues]

# def traverse_and_diff(
#     dat,
#     lipid_data,
#     levels,
#     current_level=0,
#     branch_path=None,
#     min_fc=0.2,
#     pthr=0.05,
#     output_dir="diff_results"
# ):
#     """
#     Recursively traverse the hierarchical labels in `dat`, perform differential analysis 
#     (two-group comparison: val vs the rest) at each level, and save results for each split.
    
#     - dat: DataFrame containing hierarchical annotations (columns like 'level_1', 'level_2', ...).
#            Row indices align with samples.
#     - lipid_data: DataFrame with lipid measurements (same rows = samples, columns = lipids).
#     - levels: list of the column names describing the hierarchy.
#     - current_level: integer index into `levels`.
#     - branch_path: keeps track of label choices so far (used for file naming).
#     - min_fc, pthr: thresholds passed to `differential_lipids` (you can incorporate `min_fc` logic as needed).
#     - output_dir: directory where the CSV output is saved.
#     """
#     if branch_path is None:
#         branch_path = []
    
#     # Stop if we've consumed all hierarchical levels
#     if current_level >= len(levels):
#         return
    
#     level_col = levels[current_level]
#     unique_vals = dat[level_col].unique()
    
#     # If there's no real split at this level, just exit
#     if len(unique_vals) < 2:
#         return
    
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     # For each unique group at the current level
#     for val in unique_vals:
#         # labs is a boolean mask for the current subset of `dat`
#         labs = (dat[level_col] == val)
        
#         # 1) Perform differential analysis: val vs. not val
#         diff = differential_lipids(lipid_data, labs, min_fc=min_fc, pthr=pthr)
        
#         # (Optional) sort by log2 fold change, descending
#         diff = diff.sort_values(by="log2fold_change", ascending=False)
        
#         # 2) Construct a filename reflecting the path taken so far
#         path_labels = [
#             f"{lvl_name}={lvl_val}"
#             for lvl_name, lvl_val in zip(levels[:current_level], branch_path)
#         ]
#         path_labels.append(f"{level_col}={val}")
#         filename = "_".join(path_labels) + ".csv"
        
#         # Save differential results
#         out_path = os.path.join(output_dir, filename)
#         diff.to_csv(out_path, index=False)
        
#         # 3) Recurse deeper:
#         #    - subset `dat` to only the rows where labs==True
#         #    - subset `lipid_data` the same way so indexes remain aligned
#         sub_dat = dat.loc[labs]
#         sub_lipid_data = lipid_data.loc[labs]

#         traverse_and_diff(
#             dat=sub_dat,
#             lipid_data=sub_lipid_data,
#             levels=levels,
#             current_level=current_level + 1,
#             branch_path=branch_path + [val],
#             min_fc=min_fc,
#             pthr=pthr,
#             output_dir=output_dir
#         )
