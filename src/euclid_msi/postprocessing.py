"""
Postprocessing module for EUCLID.
This module implements various postprocessing steps on the AnnData object produced by embedding.
The functionalities include:
  - XGBoost‐based imputation (feature restoration)
  - 3D anatomical interpolation
  - (Placeholder) Training a variational autoencoder ("lipimap")
  - Registering an additional modality
  - Comparing parcellations between modalities
  - Running a multiomics factor analysis (MOFA) and subsequent tSNE
  - Neighborhood analysis of clusters
  - UMAP visualization of molecules and lipizones
  - Extraction of spatial modules based on anatomical colocalization

Each BLOCK of code has been integrated into a corresponding method.
"""

import os
import joblib
import pickle
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from tqdm import tqdm
from scipy.stats import pearsonr, mannwhitneyu
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from threadpoolctl import threadpool_limits
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import umap.umap_ as umap
from adjustText import adjust_text
# MOFA
from mofapy2.run.entry_point import entry_point
# For hierarchical clustering in parcellation comparison
import scipy.cluster.hierarchy as sch
from kneed import KneeLocator
from numba import njit
from scipy import ndimage
from euclid_msi.plotting import Plotting

# Set thread limits and suppress warnings
threadpool_limits(limits=8)
os.environ["OMP_NUM_THREADS"] = "6"
warnings.filterwarnings("ignore")


@njit
def fill_array_interpolation(array_annotation, array_slices, divider_radius=5,
                             annot_inside=-0.01, limit_value_inside=-2, structure_guided=True):
    """
    Numba-accelerated interpolation to fill missing voxel values.
    """
    array_interpolated = np.copy(array_slices)
    for x in range(8, array_annotation.shape[0]):
        for y in range(0, array_annotation.shape[1]):
            for z in range(0, array_annotation.shape[2]):
                condition_fulfilled = False
                if array_slices[x, y, z] >= 0:
                    condition_fulfilled = True
                elif limit_value_inside is not None and not condition_fulfilled:
                    if array_annotation[x, y, z] > limit_value_inside:
                        condition_fulfilled = True
                elif (np.abs(array_slices[x, y, z] - annot_inside) < 1e-4) and not condition_fulfilled:
                    condition_fulfilled = True
                if condition_fulfilled:
                    value_voxel = 0.0
                    sum_weights = 0.0
                    size_radius = int(array_annotation.shape[0] / divider_radius)
                    for xt in range(max(0, x - size_radius), min(array_annotation.shape[0], x + size_radius + 1)):
                        for yt in range(max(0, y - size_radius), min(array_annotation.shape[1], y + size_radius + 1)):
                            for zt in range(max(0, z - size_radius), min(array_annotation.shape[2], z + size_radius + 1)):
                                if np.sqrt((x - xt)**2 + (y - yt)**2 + (z - zt)**2) <= size_radius:
                                    if array_slices[xt, yt, zt] >= 0:
                                        if (structure_guided and np.abs(array_annotation[x, y, z] - array_annotation[xt, yt, zt]) < 1e-4) or (not structure_guided):
                                            d = np.sqrt((x - xt)**2 + (y - yt)**2 + (z - zt)**2)
                                            w = np.exp(-d)
                                            value_voxel += w * array_slices[xt, yt, zt]
                                            sum_weights += w
                    if sum_weights > 0:
                        array_interpolated[x, y, z] = value_voxel / sum_weights
    return array_interpolated

def normalize_to_255(a):
    """Normalize array a to 0-255 scale, preserving NaNs."""
    low_percentile_val = np.nanpercentile(a, 10)
    mask = np.logical_or(a < low_percentile_val, np.isnan(a))
    a = np.where(mask, 0, a)
    a = ((a - a.min()) * (255) / (a.max() - a.min()))
    a[mask] = np.nan
    if not np.isnan(a).any():
        a = a.astype(np.uint8)
    return a

class Postprocessing:
    """
    Postprocessing class for EUCLID.
    
    Operates on the AnnData object (and related data) produced by the embedding pipeline.
    """
    def __init__(self, emb, morans, 
                 reference_image=None, annotation_image=None):
        """
        Parameters
        ----------
        adata : sc.AnnData
            The AnnData object from embedding.
        embeddings : pd.DataFrame
            The harmonized NMF embeddings (or similar) stored as a DataFrame.
        morans : pd.DataFrame
            DataFrame with Moran's I values (features x sections).
        alldata : pd.DataFrame
            The full raw data (e.g. from uMAIA) with columns corresponding to features.
        pixels : pd.DataFrame
            Pixel-level data extracted from adata (e.g. with spatial coordinates).
        coordinates : pd.DataFrame
            Spatial coordinates with columns such as ['Section', 'xccf', 'yccf', 'zccf'].
        reference_image : np.ndarray, optional
            3D reference volume (for anatomical interpolation).
        annotation_image : np.ndarray, optional
            3D anatomical annotation image.
        """
        self.adata = emb.adata
        try:
            self.embeddings = emb.adata.obsm['X_Harmonized']
        except:
            self.embeddings = emb.adata.obsm['X_NMF']
        self.morans = morans
        self.alldata = pd.DataFrame(
            self.adata.obsm['peaks'], 
            index=self.adata.obs.index,
            columns=self.adata.uns['peak_names']
        )
        self.pixels = self.adata.obs.copy()
        self.coordinates = self.pixels[['SectionID', 'x', 'y']]
        self.reference_image = reference_image
        self.annotation_image = annotation_image

    def save_msi_dataset(
        self,
        filename="msi_dataset_postprocessing_ops.h5ad"):
        """
        Save the current AnnData object to disk, including any postprocessing results.

        Parameters
        ----------
        filename : str, optional
            File path to save the AnnData object.
        save_embeddings : bool, optional
            Whether to save the harmonized embeddings in the AnnData object.
        save_restored_features : bool, optional
            Whether to save any restored features from XGBoost imputation.
        """
        # Create a copy of the AnnData object to avoid modifying the original
        adata_to_save = self.adata.copy()
        
        # Save the AnnData object
        adata_to_save.write_h5ad(filename)

    # -------------------------------------------------------------------------
    # BLOCK 1: XGBoost Feature Restoration
    # -------------------------------------------------------------------------

    
    def xgboost_feature_restoration(
        self,
        moran_threshold=0.4,
        usage_threshold=0.4,
        usage_top_n=3,
        usage_sum_threshold=2,
        acquisitions=None,
        min_sections_for_training=3,
        train_section_indices=(0, 2),
        val_section_index=1,
        valid_pearson_threshold=0.4,
        output_model_dir="xgbmodels",
        metrics_csv="metrics_imputation_df.csv",
        xgb_params=None
    ):
        import os
        import numpy as np
        import pandas as pd
        from tqdm import tqdm
        from sklearn.metrics import mean_squared_error
        from scipy.stats import pearsonr
        from xgboost import XGBRegressor
        import joblib
        """
        Impute lipid values on sections where measurement failed using XGBoost regression.
        Uses Moran's I and user-defined thresholds to decide which lipids to restore.
    
        This function:
        1. Selects "restorable" features based on `moran_threshold`.
        2. Further filters features via a usage matrix determined by Moran's I on valid sections.
           - Takes the top `usage_top_n` features above `usage_threshold` from the first
             `usage_ncolumns` columns of self.morans, per lipid row.
           - Only keeps rows with sum above `usage_sum_threshold`.
           - Optionally excludes any single feature with `exclude_feature`.
        3. Trains an XGBoost model per feature using the given train/validation split logic.
        4. Saves trained models to `output_model_dir` and a metrics CSV to `metrics_csv`.
        5. Applies all trained models to all embeddings to produce predicted values.
        6. Keeps only those features with validation Pearson > `valid_pearson_threshold`.
    
        Parameters
        ----------
        self : object
            An object that contains:
                - self.morans: pd.DataFrame with Moran's I values, rows are features, columns are sections.
                - self.alldata: pd.DataFrame with raw data (lipids, 'SectionID', etc.).
                - self.embeddings: pd.DataFrame representing features (embeddings) per pixel.
                - self.coordinates: pd.DataFrame with coordinate and 'SectionID' information per pixel.
        moran_threshold : float, default=0.4
            The Moran's I threshold used to decide which features (lipids) are initially restorable.
        usage_threshold : float, default=0.4
            Threshold passed to `top_3_above_threshold` to decide if a value is "good" for usage.
        usage_top_n : int, default=3
            The number of top columns (sections) to pick if more than `usage_top_n` columns pass `usage_threshold`.
        usage_sum_threshold : float, default=2
            Minimum sum (per row) in the usage matrix to keep a lipid feature.
        min_sections_for_training : int, default=3
            Minimum number of sections required to train. If fewer, skip that lipid feature.
        train_section_indices : tuple of int, default=(0, 2)
            The indices of the usage matrix columns (after filtering) to use for training.
        val_section_index : int, default=1
            The index of the usage matrix columns (after filtering) to use for validation.
        valid_pearson_threshold : float, default=0.4
            The Pearson correlation threshold required for a feature to be retained post-deployment.
        output_model_dir : str, default="xgbmodels"
            Directory to which trained XGBoost model files are saved.
        metrics_csv : str, default="metrics_imputation_df.csv"
            CSV file path where per-feature training/validation metrics are recorded.
        xgb_params : dict or None
            Additional parameters passed to the XGBRegressor. If None, default params are used.
    
        Returns
        -------
        anndata.AnnData
            Updated main AnnData object with the restored lipids added in .obsm.
        """
        os.makedirs(output_model_dir, exist_ok=True)

        self.embeddings = pd.DataFrame(self.embeddings, index=self.pixels.index)
    
        # 1) Determine which features (lipids) are restorable based on Moran's I threshold
        isitrestorable = (self.morans > moran_threshold).sum(axis=1).sort_values()
        torestore = isitrestorable[isitrestorable > 3].index ######################################################################################################################
    
        # 2) Adjust alldata columns: ensure first 1400 columns are cast to string numbers (example-specific)
        cols = np.array(self.alldata.columns).astype(float).astype(str)
        self.alldata.columns = cols
    
        # Subset the lipids to restore
        lipids_to_restore = self.alldata.loc[:, self.alldata.columns.isin(torestore.astype(float).astype(str))]

        if acquisitions is not None:
            usage_columns = acquisitions['acqn'].values.astype(str)
            usage_dataframe = self.morans.loc[:, usage_columns].copy()
        else:
            usage_dataframe = self.morans.copy()
        usage_dataframe.columns = self.pixels['SectionID'].unique()##################################################################################
        
        # Define helper: choose top `usage_top_n` above `usage_threshold` per row
        def top_n_above_threshold(row, threshold=usage_threshold, top_n=usage_top_n):
            above_mask = row >= threshold
            if above_mask.sum() >= top_n:
                top_indices = row.nlargest(top_n).index
                result = pd.Series(False, index=row.index)
                result[top_indices] = True
            else:
                result = above_mask
            return result
    
        usage_dataframe = usage_dataframe.apply(
            lambda r: top_n_above_threshold(r, threshold=usage_threshold, top_n=usage_top_n),
            axis=1
        )
        # Keep only rows where the sum is above `usage_sum_threshold`
        usage_dataframe = usage_dataframe.loc[usage_dataframe.sum(axis=1) > usage_sum_threshold, :]
        
        #usage_dataframe = usage_dataframe.iloc[:2,:] ############################
        print(usage_dataframe)
    
        # Select corresponding lipids to restore
        lipids_to_restore = lipids_to_restore.loc[:, lipids_to_restore.columns.isin(usage_dataframe.index.astype(float).astype(str))]
        lipids_to_restore["SectionID"] = self.pixels["SectionID"]
        lipids_to_restore.columns = lipids_to_restore.columns.astype(str)
    
        # Coordinates
        coords = self.coordinates.copy()
        coords["SectionID"] = coords["SectionID"].astype(float).astype(int).astype(str)
    
        metrics_df = pd.DataFrame(columns=["train_pearson_r", "train_rmse", "val_pearson_r", "val_rmse"])

        usage_str_index = usage_dataframe.index.astype(float).astype(str)

        mask = usage_str_index.isin(lipids_to_restore.columns)
        usage_dataframe = usage_dataframe.loc[usage_dataframe.index[mask]]

        # 3) Loop over features in usage_dataframe to train a model for each
        for index, row in tqdm(usage_dataframe.iterrows(), total=usage_dataframe.shape[0], desc="XGB Restoration"):
            # For each lipid feature index, gather sections that are True in usage_dataframe
            train_sections_all = row[row].index.tolist()
            if len(train_sections_all) < min_sections_for_training:
                continue
    
            # Train/val splitting logic
            try:
                val_section = train_sections_all[val_section_index]
                train_sections = [train_sections_all[i] for i in train_section_indices]
            except IndexError:
                # If the user-supplied indices are invalid for the list, skip
                continue
    
            # Gather train data
            train_data = self.embeddings.loc[coords["SectionID"].astype(int).isin(train_sections), :]
            
            column_key = str(float(index))
            y_train = lipids_to_restore.loc[train_data.index, column_key]
           
            # Gather val data
            val_data = self.embeddings.loc[coords["SectionID"].astype(int).isin([val_section]), :] 
            y_val = lipids_to_restore.loc[val_data.index, column_key]
    
            # 4) Train XGBoost model
            model_kwargs = xgb_params if xgb_params is not None else {}
            model = XGBRegressor(**model_kwargs)
            try:
                model.fit(train_data, y_train)
            except Exception as e:
                print(f"Error at index {index}: {e}")
                continue
    
            # Evaluate
            train_pred = model.predict(train_data)
            val_pred = model.predict(val_data)
            train_pearson = pearsonr(y_train, train_pred)[0]
            val_pearson = pearsonr(y_val, val_pred)[0]
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
            metrics_df.loc[index] = {
                "train_pearson_r": train_pearson,
                "train_rmse": train_rmse,
                "val_pearson_r": val_pearson,
                "val_rmse": val_rmse
            }
    
            # Save model
            model_path = os.path.join(output_model_dir, f"{index}_xgb_model.joblib")
            joblib.dump(model, model_path)
    
        # 5) Deploy models on all acquisitions (loop over saved models)
        coords_pred = coords.copy()
        for file in tqdm(os.listdir(output_model_dir), desc="Deploying XGB models"):
            if not file.endswith("joblib"):
                continue
            model_path = os.path.join(output_model_dir, file)
            model = joblib.load(model_path)
            pred = model.predict(self.embeddings)
            coords_pred[file] = pred
    
        # Rename new columns: remove "_xgb_model.joblib"
        new_cols = []
        for i, col in enumerate(coords_pred.columns):
            if i < len(self.coordinates.columns):
                new_cols.append(col)
            else:
                new_cols.append(col.replace("_xgb_model.joblib", ""))
        coords_pred.columns = new_cols
    
        # 6) Keep only lipids whose validation Pearson > `valid_pearson_threshold`
        metrics_df.to_csv(metrics_csv)
        valid_features = metrics_df.loc[metrics_df["val_pearson_r"] > valid_pearson_threshold].index.astype(float).astype(str)
        coords_pred = coords_pred.loc[:, list(coords_pred.columns[:len(self.coordinates.columns)]) + list(valid_features)]

        # Store the restored features in obsm and their names in uns
        self.adata.obsm['X_restored'] = coords_pred.iloc[:, 3:].values
        self.adata.uns['restored_feature_names'] = coords_pred.iloc[:, 3:].columns.tolist()


    # -------------------------------------------------------------------------
    # BLOCK 2: Anatomical Interpolation
    # -------------------------------------------------------------------------
    def anatomical_interpolation(self, lipids, output_dir="3d_interpolated_native", w=50):
        """
        For each lipid (by name) in the given list, perform 3D anatomical interpolation.
        Uses the provided reference_image and annotation_image.
        Saves the interpolated 3D array as a .npy file.
        
        Parameters
        ----------
        lipids : list
            List of lipid names (features) to interpolate.
        output_dir : str, optional
            Directory to save interpolation outputs.
        w : int, optional
            Threshold value used for cleaning.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get lipid data from X
        lipid_data = pd.DataFrame(
            self.adata.X,
            index=self.adata.obs.index,
            columns=self.adata.var_names
        )
        
        # Add spatial coordinates
        lipid_data = pd.concat([
            lipid_data,
            self.adata.obs[["xccf", "yccf", "zccf"]]
        ], axis=1)
        
        for lipid in tqdm(lipids, desc="3D Anatomical Interpolation"):
            try:
                if lipid not in lipid_data.columns:
                    print(f"Warning: {lipid} not found in data columns")
                    continue
                    
                # Get data for this lipid
                data = lipid_data[["xccf", "yccf", "zccf", lipid]].copy()
                
                # Log-transform the intensity
                data[lipid] = np.log(data[lipid])
                
                # Scale spatial coordinates
                data["xccf"] *= 10
                data["yccf"] *= 10
                data["zccf"] *= 10
                
                tensor_shape = self.reference_image.shape
                tensor = np.full(tensor_shape, np.nan)
                intensity_values = defaultdict(list)
                
                for _, row in data.iterrows():
                    x, y, z = int(row["xccf"]) - 1, int(row["yccf"]) - 1, int(row["zccf"]) - 1
                    intensity_values[(x, y, z)].append(row[lipid])
                    
                for coords, values in intensity_values.items():
                    x, y, z = coords
                    if 0 <= x < tensor_shape[0] and 0 <= y < tensor_shape[1] and 0 <= z < tensor_shape[2]:
                        tensor[x, y, z] = np.nanmean(values)
                        
                normalized_tensor = normalize_to_255(tensor)
                non_nan_mask = ~np.isnan(normalized_tensor)
                normalized_tensor[non_nan_mask & (normalized_tensor < w)] = np.nan
                normalized_tensor[self.reference_image < 4] = 0
                
                # Interpolation
                interpolated = fill_array_interpolation(self.annotation_image, normalized_tensor)
                
                # Clean by convolution
                kernel = np.ones((10, 10, 10))
                array_filled = np.where(np.isnan(interpolated), 0, interpolated)
                counts = np.where(np.isnan(interpolated), 0, 1)
                counts = ndimage.convolve(counts, kernel, mode='constant', cval=0.0)
                convolved = ndimage.convolve(array_filled, kernel, mode='constant', cval=0.0)
                avg = np.where(counts > 0, convolved / counts, np.nan)
                filled = np.where(np.isnan(interpolated), avg, interpolated)
                
                np.save(os.path.join(output_dir, f"{lipid}_interpolation_log.npy"), filled)
            except Exception as e:
                print(f"Error processing {lipid}: {e}")
                continue

    # -------------------------------------------------------------------------
    # BLOCK 3: Train Lipimap (Placeholder)
    # -------------------------------------------------------------------------
    def train_lipimap(self):
        """
        Placeholder for training a variational autoencoder to extract lipid programs.
        """
        print("train_lipimap wrapper is not implemented yet.")
        return None

    # -------------------------------------------------------------------------
    # BLOCK 4: Add Modality
    # -------------------------------------------------------------------------
    def add_modality(self, modality_adata, modality_name):
        """
        Register another omic modality onto the main AnnData object.
        
        Parameters
        ----------
        modality_adata : anndata.AnnData
            The AnnData object for the new modality.
        modality_name : str
            Name to assign to this modality.
        
        Returns
        -------
        anndata.AnnData
            Updated main AnnData object with the new modality added in .obsm.
        """
        # Align on shared indices
        common_idx = self.adata.obs_names.intersection(modality_adata.obs_names)
        modality_data = modality_adata[common_idx].X
        self.adata.obsm[modality_name] = modality_data
        return self.adata

    # -------------------------------------------------------------------------
    # BLOCK 5: Compare Parcellations
    # -------------------------------------------------------------------------
    def compare_parcellations(self, parcellation1, parcellation2, substrings=[], M=200, output_pdf="purpleheatmap_acronymlipizones.pdf"):
        """
        Compare two parcellations (e.g. lipizones vs. cell types) via crosstab and heatmap.
        
        Parameters
        ----------
        parcellation1 : pd.Series
            First parcellation labels (indexed by pixel).
        parcellation2 : pd.Series
            Second parcellation labels (indexed by pixel).
        substrings : list, optional
            List of substrings to omit.
        M : int, optional
            Minimum total counts threshold.
        output_pdf : str, optional
            Filename for saving the heatmap.
        
        Returns
        -------
        normalized_df : pd.DataFrame
            The normalized enrichment matrix.
        """
        cmat = pd.crosstab(parcellation1, parcellation2)
        rows_to_keep = ~cmat.index.to_series().str.contains('|'.join(substrings), case=False, na=False)
        cols_to_keep = ~cmat.columns.to_series().str.contains('|'.join(substrings), case=False, na=False)
        cmat = cmat.loc[rows_to_keep, cols_to_keep]
        normalized_df1 = cmat / cmat.sum()
        normalized_df1 = (normalized_df1.T / normalized_df1.T.mean()).T
        cmat2 = pd.crosstab(parcellation1, parcellation2).T
        normalized_df2 = cmat2 / cmat2.sum()
        normalized_df2 = (normalized_df2.T / normalized_df2.T.mean()).T
        normalized_df = normalized_df1 * normalized_df2
        normalized_df[cmat2.T < 20] = 0
        normalized_df = normalized_df.loc[:, normalized_df.sum() > M]
        linkage_matrix = sch.linkage(sch.distance.pdist(normalized_df.T), method='weighted', optimal_ordering=True)
        order = sch.leaves_list(linkage_matrix)
        normalized_df = normalized_df.iloc[:, order]
        order_indices = np.argsort(np.argmax(normalized_df.values, axis=1))
        normalized_df = normalized_df.iloc[order_indices, :]
        plt.figure(figsize=(10, 10))
        sns.heatmap(normalized_df, cmap="Purples", cbar_kws={'label': 'Enrichment'},
                    xticklabels=True, yticklabels=False,
                    vmin=np.percentile(normalized_df, 2), vmax=np.percentile(normalized_df, 98))
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False)
        plt.tight_layout()
        plt.savefig(output_pdf)
        plt.close()
        return normalized_df

    # -------------------------------------------------------------------------
    # BLOCK 6: Run MOFA
    # -------------------------------------------------------------------------
    def run_mofa(self, genes, lipids, factors=100, train_iter=10):
        """
        Run a multiomics factor analysis (MOFA) to integrate two modalities.
        
        Parameters
        ----------
        genes : pd.DataFrame
            Gene expression data (pixels x genes).
        lipids : pd.DataFrame
            Lipidomic data (pixels x lipids); must share the same index as genes.
        factors : int, optional
            Number of factors to learn.
        train_iter : int, optional
            Number of training iterations.
        
        Returns
        -------
        dict
            Dictionary containing MOFA factors, weights, and tSNE coordinates.
        """
        # Downsample genes if necessary.
        gexpr = genes.iloc[::10, :]
        # Remove zero-variance features.
        variance = gexpr.var()
        zero_var = variance[variance < 0.0001].index
        gexpr = gexpr.drop(columns=zero_var)
        lipids = lipids.loc[gexpr.index, :]
        data = [[gexpr], [lipids]]
        ent = entry_point()
        ent.set_data_options(scale_groups=True, scale_views=True)
        ent.set_data_matrix(
            data,
            likelihoods=["gaussian", "gaussian"],
            views_names=["gene_expression", "lipid_profiles"],
            samples_names=[gexpr.index.tolist()]
        )
        ent.set_model_options(factors=factors, spikeslab_weights=True, ard_weights=True)
        ent.set_train_options(iter=train_iter, convergence_mode="fast",
                              startELBO=1, freqELBO=1, dropR2=0.001, verbose=True)
        ent.build()
        ent.run()
        model = ent.model
        expectations = model.getExpectations()
        fac = expectations["Z"]["E"]
        weights_gene = pd.DataFrame(expectations["W"][0]["E"], index=gexpr.columns,
                                     columns=[f"Factor_{i+1}" for i in range(fac.shape[1])])
        weights_lipid = pd.DataFrame(expectations["W"][1]["E"], index=lipids.columns,
                                      columns=[f"Factor_{i+1}" for i in range(fac.shape[1])])
        factors_df = pd.DataFrame(fac, index=gexpr.index,
                                  columns=[f"Factor_{i+1}" for i in range(fac.shape[1])])
        factors_df.to_csv("minimofa_factors.csv")
        weights_gene.to_csv("minimofa_weights_genes.csv")
        weights_lipid.to_csv("minimofa_weights_lipids.csv")
        factors_df.to_hdf("factors_dfMOFA.h5ad", key="table")
        # tSNE on MOFA factors
        scaler = StandardScaler()
        x_train = scaler.fit_transform(factors_df)
        affinities_train = umap.umap_.PerplexityBasedNN(x_train, perplexity=30, metric="euclidean",
                                                         n_jobs=8, random_state=42, verbose=True)
        init_train = x_train[:, [0, 1]]
        tsne_emb = umap.umap_.TSNEEmbedding(init_train, affinities_train,
                                             negative_gradient_method="fft", n_jobs=8, verbose=True)
        tsne_emb_1 = tsne_emb.optimize(n_iter=500, exaggeration=1.2)
        tsne_emb_final = tsne_emb_1.optimize(n_iter=100, exaggeration=2.5)
        tsne_coords = pd.DataFrame(tsne_emb_final.view(np.ndarray), index=factors_df.index, columns=["TSNE1", "TSNE2"])
        np.save("minimofageneslipidsembedding_train_N.npy", tsne_coords.values)
        return {"factors": factors_df, "weights_gene": weights_gene, "weights_lipid": weights_lipid, "tsne": tsne_coords}

    # -------------------------------------------------------------------------
    # BLOCK 7: Neighborhood Analysis
    # -------------------------------------------------------------------------
    def neighborhood(self, metadata):
        """
        Calculate the frequency of clusters surrounding each pixel.
        
        Parameters
        ----------
        metadata : pd.DataFrame
            Must contain columns: 'SectionID', 'x', 'y', and a unique identifier (e.g. 'idd').
        
        Returns
        -------
        proportion_expanded : pd.DataFrame
            DataFrame with proportions of neighbor cluster occurrences.
        """
        metadata['neighbors'] = [[] for _ in range(len(metadata))]
        for section_id, group in metadata.groupby('SectionID'):
            coord_set = set(zip(group['x'], group['y']))
            for idx, row in group.iterrows():
                x0, y0 = row['x'], row['y']
                neighbor_coords = [
                    (x0 - 1, y0 - 1), (x0 - 1, y0), (x0 - 1, y0 + 1),
                    (x0,     y0 - 1),               (x0,     y0 + 1),
                    (x0 + 1, y0 - 1), (x0 + 1, y0), (x0 + 1, y0 + 1)
                ]
                existing = [f"section{section_id}_pixel{nx}_{ny}" for nx, ny in neighbor_coords if (nx, ny) in coord_set]
                metadata.at[idx, 'neighbors'] = existing
        metadata['idd'] = metadata.apply(lambda row: f"section{row.SectionID}_pixel{row.x}_{row.y}", axis=1)
        # Assume metadata has a column 'lipizone_names'
        id_to_lipizone = pd.Series(metadata.lipizone_names.values, index=metadata.idd).to_dict()
        metadata['neighbor_names'] = metadata['neighbors'].apply(lambda lst: [id_to_lipizone.get(nid, None) for nid in lst])
        grouped = metadata.groupby('lipizone_names')['neighbor_names'].apply(lambda lists: [n for sub in lists for n in sub])
        proportion_df = grouped.apply(lambda lst: {k: v/len(lst) for k, v in Counter(lst).items()})
        return proportion_df

    # -------------------------------------------------------------------------
    # BLOCK 8a: UMAP of Molecules
    # -------------------------------------------------------------------------
    def umap_molecules(self, centroidsmolecules, output_pdf="umap_molecules.pdf"):
        """
        Perform UMAP on a subset of user-defined molecules (observations as features).
        Plots labels using adjustText.
        
        Parameters
        ----------
        centroidsmolecules : pd.DataFrame
            DataFrame with lipizone-wise averages (rows: lipizones; columns: molecules).
        output_pdf : str, optional
            Filename for the saved PDF plot.
        
        Returns
        -------
        umap_coords : np.ndarray
            UMAP coordinates.
        """
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.05, n_jobs=1)
        umap_result = reducer.fit_transform(centroidsmolecules.T)
        fig, ax = plt.subplots(figsize=(14, 10))
        texts = []
        scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1],
                             c="gray", edgecolor="w", linewidth=0.5)
        for i, txt in enumerate(centroidsmolecules.columns):
            texts.append(ax.text(umap_result[i, 0], umap_result[i, 1], txt,
                                 fontsize=10, alpha=0.9))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5),
                    expand_points=(1.2, 1.4), force_points=0.2, force_text=0.2, lim=1000)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(output_pdf)
        plt.close()
        return umap_result

    # -------------------------------------------------------------------------
    # BLOCK 8b: UMAP of Lipizones
    # -------------------------------------------------------------------------
    def umap_lipizones(self, centroids, n_neighbors=4, min_dist=0.05):
        """
        Perform UMAP on lipizone centroids.
        
        Parameters
        ----------
        centroids : pd.DataFrame
            DataFrame of lipizone centroids.
        n_neighbors : int, optional
            UMAP parameter.
        min_dist : float, optional
            UMAP parameter.
        
        Returns
        -------
        umap_coords : np.ndarray
            UMAP coordinates.
        """
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_jobs=1)
        umap_coords = reducer.fit_transform(centroids)
        return umap_coords

    # -------------------------------------------------------------------------
    # BLOCK 9: Spatial Modules by Anatomical Colocalization
    # -------------------------------------------------------------------------
    def spatial_modules(self, selected_sections, LA="LA", LB="LB"):
        """
        Identify spatial modules using anatomical colocalization.
        
        Parameters
        ----------
        selected_sections : list or array-like
            Sections (or acronyms) of interest.
        LA, LB : str
            Strings to match anatomical acronyms.
        
        Returns
        -------
        None
            Displays plots and (optionally) saves results.
        """
        # Filter adata.obs (or provided data) by section or anatomical acronym
        focus = self.adata.obs.copy()
        if "Section" in focus.columns:
            focus = focus[focus["Section"].isin(selected_sections)]
        if "acronym" in focus.columns:
            focus = focus[(focus["acronym"].str.endswith(LA, na=False)) | (focus["acronym"].str.endswith(LB, na=False))]
        # Keep only abundant lipizones
        counts = focus["lipizone_color"].value_counts()
        unique_colors = counts.index[counts > 100]
        focus = focus[focus["lipizone_color"].isin(unique_colors)]
        # Compute a crosstab and normalize as in your snippet.
        cmat = pd.crosstab(focus["acronym"], focus["lipizone_color"])
        normalized_df1 = cmat / cmat.sum()
        normalized_df1 = (normalized_df1.T / normalized_df1.T.mean()).T
        cmat2 = pd.crosstab(focus["acronym"], focus["lipizone_color"]).T
        normalized_df2 = cmat2 / cmat2.sum()
        normalized_df2 = (normalized_df2.T / normalized_df2.T.mean())
        normalized_df = normalized_df1 * normalized_df2
        tc = normalized_df.T
        ad = anndata.AnnData(X=tc)
        sc.pp.neighbors(ad, use_rep='X')
        sc.tl.leiden(ad, resolution=2.0)
        cluster_labels = ad.obs['leiden']
        focus["leiden_cluster"] = focus["lipizone_color"].map(cluster_labels.to_dict())
        unique_clusters = sorted(focus["leiden_cluster"].unique())
        # Plot groups
        for cluster in unique_clusters:
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            for ax in axs:
                ax.set_aspect("equal")
                ax.axis("off")
            plt.suptitle(f"Spatial module: {cluster}")
            plt.show()
        return None

    def add_lipid_programs_from_parquet(self, parquet_file: str) -> None:
        """
        Add lipid programs from a parquet file to adata.obsm['X_lipiMap'], matching on Path, x, y.
        The lipid programs file must have columns: Path, x, y, and the program columns.
        Matching logic is the same as preprocessing's add_metadata_from_parquet.
        """
        # Load lipid programs from parquet
        lipid_df = pd.read_parquet(parquet_file)

        # Standardize types before creating composite keys
        self.adata.obs['Path'] = self.adata.obs['Path'].astype(str).str.strip()
        lipid_df['Path'] = lipid_df['Path'].astype(str).str.strip()
        self.adata.obs['x'] = self.adata.obs['x'].astype(int)
        lipid_df['x'] = lipid_df['x'].astype(int)
        self.adata.obs['y'] = self.adata.obs['y'].astype(int)
        lipid_df['y'] = lipid_df['y'].astype(int)

        # Create composite keys
        adata_key = self.adata.obs[['Path', 'x', 'y']].apply(
            lambda row: f"{row['Path']}_{row['x']}_{row['y']}", axis=1
        )
        lipid_key = lipid_df[['Path', 'x', 'y']].apply(
            lambda row: f"{row['Path']}_{row['x']}_{row['y']}", axis=1
        )

        # Find common entries
        common_keys = adata_key[adata_key.isin(lipid_key)]

        # Filter adata to keep only matching observations
        self.adata = self.adata[adata_key.isin(common_keys)].copy()
        adata_key = adata_key[adata_key.isin(common_keys)]

        # Get only program columns (exclude Path, x, y)
        program_cols = [col for col in lipid_df.columns if col not in ['Path', 'x', 'y']]
        # Map composite key to program values (as a row)
        program_map = lipid_df.set_index(lipid_key)[program_cols].to_dict(orient='index')
        # Build matrix for adata order
        X_lipiMap = np.stack([np.array(list(program_map[k].values())) for k in adata_key])
        self.adata.obsm['X_lipiMap'] = X_lipiMap
        self.adata.uns['lipiMap_names'] = program_cols
        print(f"Added lipid programs to adata.obsm['X_lipiMap'] with shape {X_lipiMap.shape}")

    def add_modality_from_parquet(self, parquet_file: str, X_name: str) -> None:
        """
        Add a new modality (e.g., gene expression, proteins) from a parquet file to adata.obsm[X_name], matching on Path, x, y.
        The modality file must have columns: Path, x, y, and the feature columns.
        Unmatched adata entries are filled with NaNs.
        The feature names are stored in adata.uns[X_name + '_names'].
        """
        # Load modality data from parquet
        modality_df = pd.read_parquet(parquet_file)

        # Standardize types before creating composite keys
        self.adata.obs['Path'] = self.adata.obs['Path'].astype(str).str.strip()
        modality_df['Path'] = modality_df['Path'].astype(str).str.strip()
        self.adata.obs['x'] = self.adata.obs['x'].astype(int)
        modality_df['x'] = modality_df['x'].astype(int)
        self.adata.obs['y'] = self.adata.obs['y'].astype(int)
        modality_df['y'] = modality_df['y'].astype(int)

        # Create composite keys
        adata_key = self.adata.obs[['Path', 'x', 'y']].apply(
            lambda row: f"{row['Path']}_{row['x']}_{row['y']}", axis=1
        )
        modality_key = modality_df[['Path', 'x', 'y']].apply(
            lambda row: f"{row['Path']}_{row['x']}_{row['y']}", axis=1
        )

        # Get only feature columns (exclude Path, x, y)
        feature_cols = [col for col in modality_df.columns if col not in ['Path', 'x', 'y']]
        feature_names = feature_cols
        modality_map = modality_df.set_index(modality_key)[feature_cols].to_dict(orient='index')

        # Build matrix for adata order, fill with NaN if not present
        X_modality = np.full((self.adata.n_obs, len(feature_cols)), np.nan)
        for i, k in enumerate(adata_key):
            if k in modality_map:
                X_modality[i, :] = np.array(list(modality_map[k].values()))
        self.adata.obsm[X_name] = X_modality
        self.adata.uns[X_name + '_names'] = feature_names
        print(f"Added modality '{X_name}' to adata.obsm with shape {X_modality.shape}")

    def evaluate_interindividual_variation(self, subject_col: str = 'Sample', subclass_col: str = None):
        """
        Evaluate inter-individual variation (ICC) for each lipid in adata.X.
        Optionally stratifies by a subclass column (e.g., lipid subclass).

        Parameters
        ----------
        subject_col : str, optional
            Column in adata.obs identifying individuals (default 'Sample').
        subclass_col : str, optional
            Column in adata.obs for stratification (e.g., 'subclass').

        Returns
        -------
        pd.DataFrame
            DataFrame of ICC values (lipids × subclasses if stratified, else lipids).
        """
        import pandas as pd
        import numpy as np

        def compute_icc(df, subject_col, measurement_col):
            group_stats = df.groupby(subject_col)[measurement_col].agg(['mean', 'count'])
            means = group_stats['mean']
            counts = group_stats['count']
            grand_mean = df[measurement_col].mean()
            ss_between = (counts * (means - grand_mean)**2).sum()
            N = len(df)
            J = len(means)
            df_between = J - 1
            ms_between = ss_between / df_between if df_between > 0 else np.nan
            ss_within = ((df.groupby(subject_col)[measurement_col].transform('mean') - df[measurement_col])**2).sum()
            df_within = N - J
            ms_within = ss_within / df_within if df_within > 0 else np.nan
            n_bar = counts.mean()
            sigma_b2 = (ms_between - ms_within) / n_bar if n_bar > 0 else np.nan
            sigma_w2 = ms_within
            icc = sigma_b2 / (sigma_b2 + sigma_w2) if (sigma_b2 + sigma_w2) != 0 else np.nan
            return icc

        X = self.adata.X
        obs = self.adata.obs.copy()
        lipid_names = list(self.adata.var_names)

        results = []
        if subclass_col is not None:
            subclasses = obs[subclass_col].unique()
            for subclass in subclasses:
                mask = obs[subclass_col] == subclass
                for i, lipid in enumerate(lipid_names):
                    vals = X[mask.values, i]
                    df = pd.DataFrame({subject_col: obs.loc[mask, subject_col], 'value': vals})
                    df = df.dropna()
                    if df[subject_col].nunique() < 2:
                        icc = np.nan
                    else:
                        icc = compute_icc(df, subject_col, 'value')
                    results.append({'lipid': lipid, 'subclass': subclass, 'icc': icc})
            icc_df = pd.DataFrame(results).pivot(index='lipid', columns='subclass', values='icc')
        else:
            for i, lipid in enumerate(lipid_names):
                vals = X[:, i]
                df = pd.DataFrame({subject_col: obs[subject_col], 'value': vals})
                df = df.dropna()
                if df[subject_col].nunique() < 2:
                    icc = np.nan
                else:
                    icc = compute_icc(df, subject_col, 'value')
                results.append({'lipid': lipid, 'icc': icc})
            icc_df = pd.DataFrame(results).set_index('lipid')['icc']
        print("ICC calculation complete. DataFrame shape:", icc_df.shape)
        return icc_df

    def create_lipidome_matrix(self, final_table, min_score=4.0):
        """
        Create a new obsm slot 'X_lipidome' by combining native and restored features with lipid names.
        
        Parameters
        ----------
        final_table : pd.DataFrame
            DataFrame containing lipid annotations, indexed by m/z values
        min_score : float, optional
            Minimum score required to keep a lipid annotation (default: 4.0)
            
        Returns
        -------
        None
            Updates self.adata in place by adding X_lipidome to obsm
        """
        import pandas as pd
        import numpy as np
        
        # 1. Get restored feature names and their corresponding data
        restored_names = self.adata.uns['restored_feature_names']
        restored_data = self.adata.obsm['X_restored']
        
        # 2. Create a mapping of m/z to lipid names using the same criteria as rename_features
        name_map = {}
        score_map = final_table['Score'].to_dict()
        
        for mz in restored_names:
            if mz in final_table.index:
                # Check score threshold
                if pd.notna(score_map.get(mz)) and score_map[mz] >= min_score:
                    annotation = final_table.loc[mz, 'AnnotationLCMSPrioritized']
                    if pd.notna(annotation) and annotation != '_db':
                        name_map[mz] = str(annotation)
        
        # 3. Filter restored features to keep only those with valid lipid names
        # and not already in var_names
        existing_names = set(self.adata.var_names)
        valid_indices = []
        valid_names = []
        
        for i, mz in enumerate(restored_names):
            if mz in name_map and name_map[mz] not in existing_names:
                valid_indices.append(i)
                valid_names.append(name_map[mz])
        
        # 4. Create the lipidome matrix by concatenating X and filtered X_restored
        X_native = self.adata.X
        X_restored_filtered = restored_data[:, valid_indices]
        
        # Combine matrices
        X_lipidome = np.hstack([X_native, X_restored_filtered])
        
        # Create combined feature names
        lipidome_names = list(self.adata.var_names) + valid_names
        
        # Store in adata
        self.adata.obsm['X_lipidome'] = X_lipidome
        self.adata.uns['lipidome_names'] = lipidome_names
        
        print(f"Created X_lipidome with shape {X_lipidome.shape}")
        print(f"Added {len(valid_names)} new lipid features from restored data")

    def plot_all_annotated_lipids(self, final_table, output_dir="all_annotated_lipids"):
        """
        Create individual PDF plots for each lipid in X_lipidome, showing their spatial distribution
        and metadata including name, source (X or X_restored), m/z value, and quality scores.
        
        Parameters
        ----------
        final_table : pd.DataFrame
            DataFrame containing lipid annotations and scores
        output_dir : str, optional
            Directory to save the PDF plots (default: "all_annotated_lipids")
        """
        import os
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import pandas as pd
        import numpy as np
        import math
        from tqdm import tqdm
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get lipid names and their source (X or X_restored)
        native_names = list(self.adata.var_names)
        restored_names = self.adata.uns['lipidome_names'][len(native_names):]
        
        # Create a mapping of lipid names to their source and m/z values
        lipid_info = {}
        
        # Map native features using old_feature_names
        for i, name in enumerate(native_names):
            mz = self.adata.var['old_feature_names'][i]
            lipid_info[name] = {'source': 'X', 'mz': mz, 'idx': i}
        
        # Map restored features using restored_feature_names
        for i, name in enumerate(restored_names):
            mz = self.adata.uns['restored_feature_names'][i]
            lipid_info[name] = {'source': 'X_restored', 'mz': mz, 'idx': i + len(native_names)}
        
        # Process each lipid with a progress bar
        for lipid_name in tqdm(self.adata.uns['lipidome_names'], desc="Creating lipid plots"):
            # Get data for this lipid
            coords = self.adata.obs[["zccf", "yccf", "xccf", "SectionID", "division", "boundary"]].copy()
            info = lipid_info[lipid_name]
            lipid_idx = info['idx']
            lipid_values = self.adata.obsm['X_lipidome'][:, lipid_idx].flatten()
            df_lipid = pd.DataFrame({lipid_name: lipid_values}, index=coords.index)
            data = pd.concat([coords, df_lipid], axis=1).reset_index(drop=True)

            unique_sections = sorted(data["SectionID"].unique())
            n_sections = len(unique_sections)
            ncols = math.ceil(math.sqrt(n_sections * 4 / 3))
            nrows = math.ceil(n_sections / ncols)

            # Compute percentiles for color scaling
            results = []
            for sec in unique_sections:
                subset = data[data["SectionID"] == sec]
                perc2 = subset[lipid_name].quantile(0.02)
                perc98 = subset[lipid_name].quantile(0.98)
                results.append([sec, perc2, perc98])
            percentile_df = pd.DataFrame(results, columns=["SectionID", "2-perc", "98-perc"])
            med2p = percentile_df["2-perc"].median()
            med98p = percentile_df["98-perc"].median()

            global_min_z = data["zccf"].min()
            global_max_z = data["zccf"].max()
            global_min_y = -data["yccf"].max()
            global_max_y = -data["yccf"].min()

            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            for idx, sec in enumerate(unique_sections):
                ax = axes[idx]
                sec_data = data[data["SectionID"] == sec]
                ax.scatter(
                    sec_data["zccf"], -sec_data["yccf"],
                    c=sec_data[lipid_name], cmap="plasma", s=0.5,
                    alpha=0.8, rasterized=True, vmin=med2p, vmax=med98p
                )
                ax.set_aspect("equal")
                ax.axis("off")
                ax.set_xlim(global_min_z, global_max_z)
                ax.set_ylim(global_min_y, global_max_y)
                ax.set_title(f"SectionID {sec}", fontsize=10)
                if "boundary" in sec_data.columns:
                    contour_data = sec_data[sec_data["boundary"] == 1]
                    ax.scatter(
                        contour_data["zccf"], -contour_data["yccf"],
                        c="black", s=0.25, alpha=0.5, rasterized=True
                    )
            for j in range(idx + 1, len(axes)):
                fig.delaxes(axes[j])

            # Add metadata as a text box in the figure
            mz = info['mz']
            feature_scores = ""
            if 'feature_selection_scores' in self.adata.uns:
                scores = self.adata.uns['feature_selection_scores']
                if mz in scores.index:
                    feature_scores = "\nFeature Selection Scores:\n"
                    for col in scores.columns:
                        feature_scores += f"{col}: {scores.loc[mz, col]:.3f}\n"
            final_table_info = ""
            if mz in final_table.index:
                row = final_table.loc[mz]
                final_table_info = "\nAnnotation Information:\n"
                for col in ['Score', 'AnnotationLCMSPrioritized', 'BestAdduct']:
                    if col in row:
                        final_table_info += f"{col}: {row[col]}\n"
            metadata_text = (
                f"Lipid Name: {lipid_name}\n"
                f"Source: {info['source']}\n"
                f"m/z: {mz}\n"
                f"{feature_scores}"
                f"{final_table_info}"
            )
            # Place the text box in the upper right of the figure
            fig.text(0.98, 0.98, metadata_text, ha='right', va='top', fontsize=10, family='monospace')

            plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave space for the text box
            safe_name = lipid_name.replace(' ', '_').replace(':', '_')
            pdf_path = os.path.join(output_dir, f"{safe_name}.pdf")
            plt.savefig(pdf_path)
            plt.close(fig)
        
        print(f"Created individual PDF plots in directory: {output_dir}")