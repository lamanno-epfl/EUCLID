# Cell 1: Import necessary libraries
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

CASECONTROL_DIR = Path(os.getcwd()) / "casecontrol_analysis"
CASECONTROL_DIR.mkdir(exist_ok=True)

# Cell 2: Dashboard for configurable hyperparameters
class LipidAnalysisConfig:
    def __init__(self):
        # Lipids to analyze
        self.lipids_to_analyze = ["PI 38:7"]  # Default lipid
        
        # Model hyperparameters
        self.learning_rate = 0.05
        self.num_epochs = 2000
        self.adaptive_lr = False  # If True, will use learning rate scheduler
        
        # Priors for the model
        self.supertype_prior_std = 1.0
        self.supertype_susceptibility_prior_std = 1.0
        self.sample_prior_std = 1.0
        self.section_prior_std = 5.0
        
        # Data processing
        self.downsampling = 1  # Use every nth point
        self.random_seed = 42
        self.normalize_percentiles = (0.1, 99.9)  # Lower and upper percentiles for normalization
        
        # Guide parameters
        self.guide_supertype_unconst_scale = 0.1
        self.guide_supertype_susceptibility_scale = 0.1
        
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
        print(f"  - Supertype susceptibility: {self.supertype_susceptibility_prior_std}")
        print(f"  - Sample: {self.sample_prior_std}")
        print(f"  - Section: {self.section_prior_std}")
        print("Guide parameters:")
        print(f"  - Supertype unconst scale: {self.guide_supertype_unconst_scale}")
        print(f"  - Supertype susceptibility scale: {self.guide_supertype_susceptibility_scale}")

def cfg_string(cfg):
    return (
        f"lr{cfg.learning_rate}"
        f"_ep{cfg.num_epochs}"
        f"_ds{cfg.downsampling}"
        f"_seed{cfg.random_seed}"
        f"_priorS{cfg.sample_prior_std}"
        f"_priorSec{cfg.section_prior_std}"
        f"_suscPrior{cfg.supertype_susceptibility_prior_std}"
        f"_guideU{cfg.guide_supertype_unconst_scale}"
        f"_guideS{cfg.guide_supertype_susceptibility_scale}"
    )

config = LipidAnalysisConfig()
        
# Cell 3: Load and preprocess data
def load_data():
    """Load the lipid data and perform initial preprocessing."""
    sub_alldata = pd.read_parquet("interpretable_lipids2.parquet")
    # Remove specific samples
    sub_alldata = sub_alldata.loc[~sub_alldata["Sample"].isin(["Male1", "Male2", "Male3"]),:]
    
    # Convert categorical columns
    sub_alldata['Condition'] = sub_alldata['Condition'].astype('category')
    sub_alldata['supertype'] = sub_alldata['supertype'].astype('category')
    sub_alldata['Sample'] = sub_alldata['Sample'].astype('category')
    sub_alldata['SectionID'] = sub_alldata['SectionID'].astype('category')
    
    return sub_alldata

def normalize_lipid_column(df, column, lower_percentile=0.1, upper_percentile=99.9):
    """Normalize a lipid column to the range [0, 1] after clipping outliers."""
    values = df[column].values.astype(np.float32)
    lower_bound = np.percentile(values, lower_percentile)
    upper_bound = np.percentile(values, upper_percentile)
    clipped = np.clip(values, lower_bound, upper_bound)
    normalized = (clipped - lower_bound) / (upper_bound - lower_bound)
    df[column] = normalized
    return df


# Cell 4: Functions for spatial subsampling
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

# Cell 5: Create train and test sets with subsampling
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

# Cell 6: Visualize spatial subsampling
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
    


# Cell 7: Define the hierarchical model for pregnancy
def model_pregnancy_hierarchical(
    condition_code, section_code, supertype_code,
    map_section_to_sample, map_sample_to_condition,
    lipid_x=None
):
    """
    Hierarchical model for pregnancy lipid analysis with:
      a) non-centered sections
      b) per-sample heteroskedasticity
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
        alpha_supertype_susceptibility = numpyro.sample(
            "alpha_supertype_susceptibility",
            dist.Normal(0.0, config.supertype_susceptibility_prior_std)
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
            alpha_supertype_susceptibility[supertype_code],
            0.0
        )
    )
    with numpyro.plate("data", len(section_code)):
        numpyro.sample("obs", dist.Normal(mu, 0.1), obs=lipid_x)


# Cell 8: Define the guide (variational approximation)
def manual_guide(
    condition_code, section_code, supertype_code,
    map_section_to_sample, map_sample_to_condition, 
    lipid_x=None
):
    """
    Manual guide for variational inference.
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
        "alpha_supertype_susceptibility_loc", jnp.zeros((n_supertypes,)))
    alpha_supertype_susc_scale = numpyro.param(
        "alpha_supertype_susceptibility_scale", jnp.full((n_supertypes,), config.guide_supertype_susceptibility_scale),
        constraint=dist.constraints.positive
    )
    
    with numpyro.plate("plate_supertype", n_supertypes):
        numpyro.sample(
            "alpha_supertype_unconst",
            dist.Normal(alpha_supertype_unconst_loc, alpha_supertype_unconst_scale)
        )
        numpyro.sample(
            "alpha_supertype_susceptibility",
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


# Cell 9: Data preparation function
def prepare_data(df, lipid_name):
    """
    Prepare data for model training.
    
    Parameters:
    - df: DataFrame with lipid data
    - lipid_name: Name of the lipid column to use
    
    Returns:
    - Prepared data for model training
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


# Cell 10: Function to run training
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
        model_pregnancy_hierarchical, 
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
        model_pregnancy_hierarchical,
        guide=manual_guide,
        params=final_params,
        num_samples=1000,
        return_sites=[
            "alpha_supertype_susceptibility",
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
    suscmeans      = samples_params["alpha_supertype_susceptibility"].mean(axis=0)

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
    # ── Statistical analysis on susceptibility ─────────────────────────────
    loc   = np.array(samples_params["alpha_supertype_susceptibility"].mean(axis=0))
    scale = np.array(jnp.std(samples_params["alpha_supertype_susceptibility"], axis=0))
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
    df_stats.to_csv(f"{lipid_name}_pregnancy_shifts.csv")

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
    samples = np.array(samples_params["alpha_supertype_susceptibility"])

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

    df_stats.to_csv(f"{lipid_name}_pregnancy_shifts_fdr5_vs0.csv")

    return samples_params, df_stats

# Cell 12: Predictive performance evaluation
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
        model_pregnancy_hierarchical,
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
            alpha_st_susc = float(samples_params["alpha_supertype_susceptibility"].mean(axis=0)[supertypenow])
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
    

# Cell 14: Plot parameter traces and ELBO
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
        


# Cell 15: Main function to run analysis for multiple lipids
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
        np.save(f"{lipid_name}_model_params.npy", final_params)
    
    return results


# Cell 16: Prior predictive check
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
    predictive = Predictive(model_pregnancy_hierarchical, num_samples=25)
    
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
    


# Cell 17: Run the analysis
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

# --- New API function ---
def run_case_control_analysis(
    adata: sc.AnnData,
    lipids_to_analyze,
    learning_rate=0.05,
    num_epochs=2000,
    adaptive_lr=False,
    supertype_prior_std=1.0,
    supertype_susceptibility_prior_std=1.0,
    sample_prior_std=1.0,
    section_prior_std=5.0,
    downsampling=1,
    random_seed=42,
    normalize_percentiles=(0.1, 99.9),
    guide_supertype_unconst_scale=0.1,
    guide_supertype_susceptibility_scale=0.1,
    x_col="x",
    y_col="y",
    sectionid_col="SectionID",
    sample_col="Sample",
    condition_col="Condition",
    supertype_col="supertype",
    verbose=True
):
    """
    Run case-control analysis on an AnnData object and save all outputs to 'casecontrol_analysis'.
    """
    import pandas as pd
    # Extract obs and relevant columns
    obs = adata.obs.copy()
    # Extract lipid intensities (assume adata.var_names matches columns in adata.X)
    lipid_df = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
    # Merge obs and lipid intensities
    df = pd.concat([obs, lipid_df], axis=1)
    # Check required columns
    for col in [x_col, y_col, sectionid_col, sample_col, condition_col, supertype_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in AnnData.obs.")
    # Prepare coords DataFrame for spatial splitting
    coords = df[[x_col, y_col, sectionid_col]].copy()
    coords.columns = ["x", "y", "SectionID"]
    # Prepare config object
    class Config:
        pass
    config = Config()
    config.lipids_to_analyze = lipids_to_analyze
    config.learning_rate = learning_rate
    config.num_epochs = num_epochs
    config.adaptive_lr = adaptive_lr
    config.supertype_prior_std = supertype_prior_std
    config.supertype_susceptibility_prior_std = supertype_susceptibility_prior_std
    config.sample_prior_std = sample_prior_std
    config.section_prior_std = section_prior_std
    config.downsampling = downsampling
    config.random_seed = random_seed
    config.normalize_percentiles = normalize_percentiles
    config.guide_supertype_unconst_scale = guide_supertype_unconst_scale
    config.guide_supertype_susceptibility_scale = guide_supertype_susceptibility_scale
    # Patch global config reference if needed
    global CASECONTROL_DIR
    global PDF_DIR
    PDF_DIR = CASECONTROL_DIR
    # Run main analysis
    results = main(df, coords, config)
    if verbose:
        print(f"All outputs saved to: {CASECONTROL_DIR}")
    return results

def summarize_case_control_results(
    adata,
    lipids_to_analyze,
    supertypes=None,
    model_dir=None,
    normalize_percentiles=(0.1, 99.9),
    output_prefix="pregnancy"
):
    """
    Summarize case-control model results for a set of lipids and supertypes.
    Saves shift, baseline, and foldchange as parquet files in the casecontrol_analysis directory.
    """
    import numpy as np
    import pandas as pd
    from pathlib import Path

    if supertypes is None:
        supertypes = np.sort(adata.obs['supertype'].unique())
    if model_dir is None:
        model_dir = CASECONTROL_DIR
    else:
        model_dir = Path(model_dir)
    aaaa_list, bbbb_list, lipid_names = [], [], []

    for lipid in tqdm(lipids_to_analyze):
        param_path = model_dir / f"{lipid}_model_params.npy"
        if not param_path.exists():
            print(f"Warning: {param_path} not found, skipping.")
            continue
        params = np.load(param_path, allow_pickle=True).item()
        # Baseline: sigmoid of unconst
        baseline = pd.DataFrame(
            sigmoid(params['alpha_supertype_unconst_loc']),
            index=supertypes, columns=[lipid]
        )
        # Shift: susceptibility
        shift = pd.DataFrame(
            params['alpha_supertype_susceptibility_loc'],
            index=supertypes, columns=[lipid]
        )
        aaaa_list.append(shift)
        bbbb_list.append(baseline)
        lipid_names.append(lipid)

    # Concatenate across lipids
    shift_df = pd.concat(aaaa_list, axis=1)
    baseline_df = pd.concat(bbbb_list, axis=1)
    shift_df.columns = lipid_names
    baseline_df.columns = lipid_names

    # Compute fold change
    foldchange_df = shift_df / baseline_df

    # Save results
    shift_df.to_parquet(model_dir / f"shift_{output_prefix}.parquet")
    baseline_df.to_parquet(model_dir / f"baseline_{output_prefix}.parquet")
    foldchange_df.to_parquet(model_dir / f"foldchange_{output_prefix}.parquet")

    print(f"Saved shift, baseline, and foldchange to {model_dir}")
    return shift_df, baseline_df, foldchange_df

def summarize_xsupertypes(
    adata,
    lipids_to_analyze,
    supertypes=None,
    model_dir=None,
    output_prefix="pregnancy",
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

    if supertypes is None:
        supertypes = np.sort(adata.obs['supertype'].unique())
    if model_dir is None:
        model_dir = CASECONTROL_DIR
    else:
        model_dir = Path(model_dir)

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
        loc_susc = params['alpha_supertype_susceptibility_loc']
        scale_susc = params['alpha_supertype_susceptibility_scale']
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
    shifted_xsupertypes_df = lipid_upreg_xsupertypes_df | lipid_downreg_xsupertypes_df

    # Save results
    lipid_upreg_xsupertypes_df.to_parquet(model_dir / f"upreg_xsupertypes_{output_prefix}.parquet")
    lipid_downreg_xsupertypes_df.to_parquet(model_dir / f"downreg_xsupertypes_{output_prefix}.parquet")
    lipid_expressed_xsupertypes_df.to_parquet(model_dir / f"expressed_xsupertypes_{output_prefix}.parquet")
    shifted_xsupertypes_df.to_parquet(model_dir / f"shifted_xsupertypes_{output_prefix}.parquet")

    print(f"Saved upreg, downreg, expressed, and shifted xsupertypes to {model_dir}")

    # --- Downstream: Bootstrap mean and CI for fraction of expressed supertypes that are also shifted ---
    mean_score = []
    ci_lowers = []
    ci_uppers = []
    for yyy in tqdm(range(lipid_expressed_xsupertypes_df.shape[0])):
        exp = lipid_expressed_xsupertypes_df.iloc[yyy, :].values
        shif = shifted_xsupertypes_df.iloc[yyy, :].values
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

# --- Usage example ---
# from euclid_msi.euclid_casecontrol import summarize_xsupertypes
# upreg, downreg, expressed, shifted, mean_score, ci_lowers, ci_uppers = summarize_xsupertypes(
#     adata,
#     lipids_to_analyze=["PC 38:6", "HexCer 40:1;O2"],
#     output_prefix="pregnancy"
# )

