import os
import pickle
import warnings
import random
import itertools
import json
from datetime import datetime
import cProfile
import pstats

import joblib
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import squidpy as sq
from . import backSPIN
import leidenalg
import networkx as nx
import igraph as ig

from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, NMF
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from xgboost import XGBClassifier
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.distance import squareform, pdist
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu, entropy
from statsmodels.stats.multitest import multipletests

from threadpoolctl import threadpool_limits
from tqdm import tqdm
from kneed import KneeLocator
from PyPDF2 import PdfMerger, PdfReader

# Set thread limits and suppress warnings
threadpool_limits(limits=8)
os.environ['OMP_NUM_THREADS'] = '6'
warnings.filterwarnings('ignore')


# =============================================================================
# Define a Node class for storing the hierarchical clustering tree
# =============================================================================
class Node:
    def __init__(self, level, path=None):
        self.level = level
        self.path = path if path is not None else []
        self.scaler = None
        self.nmf = None
        self.xgb_model = None
        self.feature_importances = None 
        self.children = {}
        self.factors_to_use = None

# =============================================================================
# Clustering class
# =============================================================================
class Clustering:
    """
    Clustering class for EUCLID.
    
    This class encapsulates the entire clustering workflow.
    
    Parameters
    ----------
    emb: a EUCLID Embedding object
    analysis_name: str, optional
        Prefix for all output files. Default is "analysis".
    """
    def __init__(self, emb, analysis_name="analysis"):

        self.analysis_name = analysis_name
        if emb is None:
            self.adata = None

        else:
            
            try:
                self.standardized_embeddings_GLOBAL = pd.DataFrame(StandardScaler().fit_transform(emb.adata.obsm['X_Harmonized']),
                                                            index=emb.adata.obs_names)
            except:
                self.standardized_embeddings_GLOBAL = pd.DataFrame(StandardScaler().fit_transform(emb.adata.obsm['X_NMF']),
                                                            index=emb.adata.obs_names)
            metadata = emb.adata.obs.copy()
            coordinates = metadata[['x','y','SectionID', 'SectionID']]
            coordinates.columns = ["zccf","yccf","Section","xccf"]
            
            self.adata = emb.adata
            self.coordinates = coordinates
            self.reconstructed_data_df = pd.DataFrame(
                emb.adata.obsm['X_approximated'],
                index=emb.adata.obs_names
            )
            self.metadata = metadata
            
            vcnorm = self.coordinates.loc[self.reconstructed_data_df.index, 'Section'] \
                .value_counts()
            vcnorm.index = vcnorm.index.astype(int)
            vcnorm = vcnorm.sort_index()
            
            data = pd.DataFrame(
                self.adata.X,
                index=self.adata.obs_names,
                columns=self.adata.var_names
            )
            rawlips = data.copy()
            data = self.reconstructed_data_df
            
            # do vmin-vmax normalization with the percentiles + clipping for differential lipids testing
            datemp = rawlips.copy() 
            p2 = datemp.quantile(0.02)
            p98 = datemp.quantile(0.98)

            datemp_values = datemp.values
            p2_values = p2.values
            p98_values = p98.values

            normalized_values = (datemp_values - p2_values) / (p98_values - p2_values)

            clipped_values = np.clip(normalized_values, 0, 1)

            normalized_datemp = pd.DataFrame(clipped_values, columns=datemp.columns, index=datemp.index)

            self.adatamaia = sc.AnnData(X=data)
            self.adatamaia.obsm['spatial'] = self.coordinates[['zccf', 'yccf', 'Section']].loc[data.index,:].values

            self.adatamaia.obsm['lipids'] = normalized_datemp

            self.adatamaia.obsm['X_TSNE'] = emb.adata.obsm['X_TSNE']


    def leiden_nmf(self, use_reference_only=True, resolution=1.0, key_added="X_Leiden"):
        """
        Perform conventional Leiden clustering on the harmonized NMF embeddings.
        
        Parameters
        ----------
        use_reference_only : bool, optional
            If True, restrict clustering to reference sections only.
        resolution : float, optional
            Leiden resolution parameter.
        key_added : str, optional
            Key to store the clustering result in adata.obs.
        
        Returns
        -------
        sc.AnnData
            The AnnData object with the added clustering result.
        """
        try:
            sc.pp.neighbors(self.adata, use_rep="X_Harmonized")
        except:
            sc.pp.neighbors(self.adata, use_rep="X_NMF")
        sc.tl.leiden(self.adata, resolution=resolution, key_added=key_added)

    # -------------------------------------------------------------------------
    # Utility functions (internal)
    # -------------------------------------------------------------------------
    def _compute_seeded_NMF(self, data, gamma_min=0.8, gamma_max=1.5, gamma_num=100):
        """
        Private method to compute seeded NMF (as in embedding) on the given data.
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame of pixels x lipids.
        gamma_min : float, optional
            Minimum gamma value for Leiden resolution search. (Default is 0.8)
        gamma_max : float, optional
            Maximum gamma value for Leiden resolution search. (Default is 1.5)
        gamma_num : int, optional
            Number of gamma values to try. (Default is 100)
        Returns
        -------
        nmfdf : pd.DataFrame
            NMF factor matrix (W).
        factor_to_lipid : np.ndarray
            The H matrix (components x lipids).
        N_factors : int
            Number of factors.
        nmf_model : NMF
            Fitted NMF model.
        """
        # 1. Calculate correlation matrix
        corr = np.corrcoef(data.values.T)
        corr_matrix = np.abs(corr)
        np.fill_diagonal(corr_matrix, 0)
        # Build dummy AnnData for neighbors
        adata_dummy = anndata.AnnData(X=np.zeros_like(corr_matrix))
        adata_dummy.obsp['connectivities'] = csr_matrix(corr_matrix)
        adata_dummy.uns['neighbors'] = {
            'connectivities_key': 'connectivities',
            'distances_key': 'distances',
            'params': {'n_neighbors': 10, 'method': 'custom'}
        }
        G = nx.from_numpy_array(corr_matrix)
        gamma_values = np.linspace(gamma_min, gamma_max, num=gamma_num) 
        num_communities = []
        modularity_scores = []
        objective_values = []
        for gamma in gamma_values:
            sc.tl.leiden(adata_dummy, resolution=gamma, key_added=f'leiden_{gamma}')
            clusters = adata_dummy.obs[f'leiden_{gamma}'].astype(int).values
            num_comms = len(np.unique(clusters))
            num_communities.append(num_comms)
            partition = [np.where(clusters == i)[0] for i in range(num_comms)]
            modularity = nx.community.modularity(G, partition)
            modularity_scores.append(modularity)
        epsilon = 1e-10
        alpha = 0.7
        for Q, N_c in zip(modularity_scores, num_communities):
            f_gamma = Q**alpha * np.log(N_c + 1 + epsilon)
            objective_values.append(f_gamma)
        max_index = np.argmax(objective_values)
        best_gamma = gamma_values[max_index]
        best_num_comms = num_communities[max_index]
        sc.tl.leiden(adata_dummy, resolution=best_gamma, key_added='leiden_best')
        clusters = adata_dummy.obs['leiden_best'].astype(int).values
        N_factors = best_num_comms
        # 4. Choose representative lipid per cluster
        dist = 1 - corr_matrix
        np.fill_diagonal(dist, 0)
        dist = np.maximum(dist, dist.T)
        representatives = []
        for i in range(N_factors):
            cluster_members = np.where(clusters == i)[0]
            if len(cluster_members) > 0:
                mean_dist = dist[cluster_members][:, cluster_members].mean(axis=1)
                central_idx = cluster_members[np.argmin(mean_dist)]
                representatives.append(central_idx)
        W_init = data.values[:, representatives]
        H_init = corr[representatives, :]
        H_init[H_init < 0] = 0.
        N_factors = W_init.shape[1]
        nmf = NMF(n_components=N_factors, init='custom', random_state=42)
        data_offset = data.values# - np.min(data) + 1e-7
        data_offset = np.ascontiguousarray(data_offset)
        W_init = np.ascontiguousarray(W_init)
        H_init = np.ascontiguousarray(H_init)
        W = nmf.fit_transform(data_offset, W=W_init, H=H_init)
        nmf_result = nmf.transform(data_offset)
        nmfdf = pd.DataFrame(nmf_result, index=data.index)
        factor_to_lipid = nmf.components_
        return nmfdf, factor_to_lipid, N_factors, nmf

    def _continuity_check(self, spat, kmeans_labels,
                          spat_columns=['zccf','yccf','Section'],
                          min_val_threshold=10,
                          min_nonzero_sections=3,
                          gaussian_sigma=1.8,
                          default_peak_ratio=10):
        """
        Check whether clusters are continuous along the AP axis.

        Parameters
        ----------
        spat : np.array or DataFrame
            Array or DataFrame with spatial coordinates.
        kmeans_labels : array-like of int
            Cluster labels (0 or 1) for each row in `spat`.
        spat_columns : list, optional
            Column names for spatial coords (default ['zccf','yccf','Section']).
        min_val_threshold : int, optional
            Minimum section‐count to keep (default 10).
        min_nonzero_sections : int, optional
            Minimum number of nonzero sections (default 3).
        gaussian_sigma : float, optional
            Sigma for Gaussian smoothing (default 1.8).
        default_peak_ratio : float, optional
            Fallback peak ratio if fewer than 2 peaks (default 10).

        Returns
        -------
        enough0, enough1 : bool
            Continuity flags for cluster 0 and 1.
        peaks0, peaks1 : int
            Number of peaks for cluster 0 and 1.
        ratio0, ratio1 : float
            Top‐peak ratios for cluster 0 and 1.
        """
        import numpy as np
        import pandas as pd
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks

        # Build DataFrame of coords + cluster codes
        dd2 = pd.DataFrame(spat, columns=spat_columns)
        dd2['Section'] = dd2['Section'].astype(int)
        dd2['color'] = pd.Series(kmeans_labels).astype(int)

        # Normalization factor for each section
        vcnorm = dd2['Section'].value_counts()
        vcnorm.index = vcnorm.index.astype(int)
        vcnorm = vcnorm.sort_index()

        enough_flags = []
        peak_counts = []
        peak_ratios = []

        # Loop over each unique cluster code
        for code in np.sort(dd2['color'].unique()):
            sub = dd2[dd2['color'] == code]
            vc = sub['Section'].value_counts()
            vc.index = vc.index.astype(int)
            sorted_counts = vc.sort_index()
            ap = sorted_counts.values.copy()
            ap[ap < min_val_threshold] = 0

            # continuity: >min_nonzero_sections nonzero AND at least one pair of consecutive nonzero
            nonzero_ok = np.sum(ap > 0) > min_nonzero_sections
            consec_ok = any(ap[i] != 0 and ap[i+1] != 0 for i in range(len(ap)-1))
            enough = nonzero_ok and consec_ok

            # normalize by vcnorm, pad, smooth, and detect peaks
            normed = sorted_counts / vcnorm.loc[sorted_counts.index].values
            padded = np.pad(normed.values, pad_width=1, mode='constant', constant_values=0)
            smoothed = gaussian_filter1d(padded, sigma=gaussian_sigma)

            peaks, props = find_peaks(smoothed, height=0)
            n_peaks = len(peaks)

            if n_peaks > 1:
                heights = props['peak_heights']
                top2 = np.sort(heights)[-2:]
                ratio = top2[1] / top2[0]
            else:
                ratio = default_peak_ratio

            enough_flags.append(enough)
            peak_counts.append(n_peaks)
            peak_ratios.append(ratio)

        # Return in the same order as PROTOTYPE (cluster 0 then 1)
        return (enough_flags[0], enough_flags[1],
                peak_counts[0], peak_counts[1],
                peak_ratios[0], peak_ratios[1])



    def _differential_lipids(self, lipidata, kmeans_labels, min_fc=0.2, pthr=0.05):
        """
        Compare two groups (assumed binary) for differential lipids.
        Returns the number of altered lipids and a table of promoted ones.
        """
        results = []
        a = lipidata[kmeans_labels == 0, :]
        b = lipidata[kmeans_labels == 1, :]
        for rrr in range(lipidata.shape[1]):
            groupA = a[:, rrr]
            groupB = b[:, rrr]
            meanA = np.mean(groupA)
            meanB = np.mean(groupB)
            log2fold_change = np.abs(np.log2(meanB/meanA)) if meanA > 0 and meanB > 0 else np.nan
            try:
                _, p_value = mannwhitneyu(groupA, groupB, alternative='two-sided')
            except ValueError:
                p_value = np.nan
            results.append({'lipid': rrr, 'log2fold_change': log2fold_change, 'p_value': p_value})
        results_df = pd.DataFrame(results)
        reject, pvals_corrected, _, _ = multipletests(results_df['p_value'].values, alpha=0.05, method='fdr_bh')
        results_df['p_value_corrected'] = pvals_corrected
        promoted = results_df[(results_df['log2fold_change'] > min_fc) & (results_df['p_value_corrected'] < pthr)]
        alteredlips = np.sum((results_df['log2fold_change'] > min_fc) & (results_df['p_value_corrected'] < pthr))
        return alteredlips, promoted

    def _rank_features_by_combined_score(self, tempadata):
        """
        Rank features by combining variance-of-variances and mean variances.
        """
        sections = tempadata.obsm['spatial'][:, 2]
        unique_sections = np.unique(sections)
        var_of_vars = []
        mean_of_vars = []
        for i in range(tempadata.X.shape[1]):
            feature_values = tempadata.X[:, i]
            section_variances = []
            for sec in unique_sections:
                section_values = feature_values[sections == sec]
                section_variance = np.var(section_values)
                section_variances.append(section_variance)
            var_of_vars.append(np.var(section_variances))
            mean_of_vars.append(np.mean(section_variances))
        var_of_vars = np.array(var_of_vars) / np.mean(var_of_vars)
        mean_of_vars = np.array(mean_of_vars) / np.mean(mean_of_vars)
        combined_score = -var_of_vars/2 + mean_of_vars
        ranked_indices = np.argsort(combined_score)[::-1]
        return ranked_indices

    def _find_elbow_point(self, values):
        """
        Find the elbow point in cumulative absolute loadings.
        """
        sorted_values = np.sort(np.abs(values))[::-1]
        cumulative_variance = np.cumsum(sorted_values) / np.sum(sorted_values)
        kneedle = KneeLocator(range(1, len(cumulative_variance)+1), cumulative_variance, curve='concave', direction='increasing')
        elbow = kneedle.elbow
        return elbow

    def _generate_combinations(self, n, limit=200):
        """
        Generate sorted combinations (of component indices) to try for splitting.
        """
        all_combinations = []
        for r in range(n, 0, -1):
            for comb in itertools.combinations(range(n), r):
                all_combinations.append(comb)
                if len(all_combinations) >= limit:
                    return all_combinations
        return all_combinations

    def _leidenalg_clustering(self, inputdata, Nneigh=40, Niter=5):
        """
        Faster Leiden clustering using leidenalg.
        """
        nn = NearestNeighbors(n_neighbors=Nneigh, n_jobs=4)
        nn.fit(inputdata)
        knn = nn.kneighbors_graph(inputdata)
        G = nx.Graph(knn)
        g = ig.Graph.from_networkx(G)
        partitions = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, n_iterations=Niter, seed=230598)
        labels = np.array(partitions.membership)
        return labels

    def _undersample(self, X, y, sampling_strategy='auto'):
        """
        Under-sample majority class.
        """
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
        return X_res, y_res

    # -------------------------------------------------------------------------
    # BLOCK 2: Learn transferable, locally enhanced clustering
    # -------------------------------------------------------------------------
    def learn_euclid_clustering(self,
                                K=60,
                                min_voxels=150,
                                min_diff_lipids=2,
                                min_fc=0.2,
                                pthr=0.05,
                                thr_signal=1e-10,
                                penalty1=1.5,
                                penalty2=2,
                                ACCTHR=0.6,
                                max_depth=15,
                                ds_factor=1,
                                spat_columns=['zccf','yccf','Section'],
                                min_val_threshold=10,
                                min_nonzero_sections=3,
                                gaussian_sigma=1.8,
                                default_peak_ratio=10,
                                peak_count_threshold=3,
                                peak_ratio_threshold=1.4,
                                combinations=200,
                                xgb_n_estimators=1000,
                                xgb_max_depth=8,
                                xgb_learning_rate=0.02,
                                xgb_subsample=0.8,
                                xgb_colsample_bytree=0.8,
                                xgb_gamma=0.5,
                                xgb_random_state=42,
                                xgb_n_jobs=6,
                                early_stopping_rounds=7,
                                plot_dir=None,
                                do_plotting=False):
        """
        Learn a hierarchical bipartite clustering tree on the dataset.
        This is a self-supervised clustering method that recursively splits the dataset,
        computes local NMFs, reaggregates clusters (e.g., via backSPIN), and trains an XGBoost
        classifier at each split.

        Parameters
        ----------
        K : int, optional
            Number of clusters for initial KMeans clustering. Default is 60.
        min_voxels : int, optional
            Minimum number of voxels required for a valid split. Default is 150.
        min_diff_lipids : int, optional
            Minimum number of differentially expressed lipids required for a valid split. Default is 2.
        min_fc : float, optional
            Minimum fold change threshold for differential lipid analysis. Default is 0.2.
        pthr : float, optional
            P-value threshold for differential lipid analysis. Default is 0.05.
        thr_signal : float, optional
            Signal threshold for filtering low-signal components. Default is 1e-10.
        penalty1 : float, optional
            Penalty factor for previous-level embeddings. Default is 1.5.
        penalty2 : float, optional
            Penalty factor for global embeddings. Default is 2.0.
        ACCTHR : float, optional
            Minimum accuracy threshold for XGBoost classifier. Default is 0.6.
        max_depth : int, optional
            Maximum depth of the clustering tree. Default is 15.
        ds_factor : int, optional
            Downsampling factor for the dataset. Default is 1.
        spat_columns : list, optional
            Column names for spatial coordinates. Default is ['zccf','yccf','Section'].
        min_val_threshold : int, optional
            Minimum section-count threshold for continuity check. Default is 10.
        min_nonzero_sections : int, optional
            Minimum number of nonzero sections required for continuity. Default is 3.
        gaussian_sigma : float, optional
            Sigma parameter for Gaussian smoothing in continuity check. Default is 1.8.
        default_peak_ratio : float, optional
            Default peak ratio if fewer than 2 peaks are found. Default is 10.
        peak_count_threshold : int, optional
            Maximum number of peaks allowed for a valid split. Default is 3.
        peak_ratio_threshold : float, optional
            Minimum ratio between top peaks for a valid split. Default is 1.4.
        combinations : int, optional
            Maximum number of component combinations to try. Default is 200.
        xgb_n_estimators : int, optional
            Number of estimators for XGBoost classifier. Default is 1000.
        xgb_max_depth : int, optional
            Maximum depth for XGBoost classifier. Default is 8.
        xgb_learning_rate : float, optional
            Learning rate for XGBoost classifier. Default is 0.02.
        xgb_subsample : float, optional
            Subsample ratio for XGBoost classifier. Default is 0.8.
        xgb_colsample_bytree : float, optional
            Column sample ratio for XGBoost classifier. Default is 0.8.
        xgb_gamma : float, optional
            Minimum loss reduction for XGBoost classifier. Default is 0.5.
        xgb_random_state : int, optional
            Random state for XGBoost classifier. Default is 42.
        xgb_n_jobs : int, optional
            Number of parallel jobs for XGBoost classifier. Default is 6.
        early_stopping_rounds : int, optional
            Number of rounds for early stopping in XGBoost. Default is 7.
        plot_dir : str, optional
            Directory to save the clustering plots. Default is "clustering_plots".
        do_plotting : bool, optional
            Whether to generate and save plots during clustering. Default is False.

        Returns
        -------
        root_node : Node
            The root of the hierarchical clustering tree.
        clusteringLOG : pd.DataFrame
            A DataFrame recording the split history.
        """
        # Create plot directory if it doesn't exist
        if do_plotting:
            if plot_dir is None:
                plot_dir = f"{self.analysis_name}_clustering_plots"
            os.makedirs(plot_dir, exist_ok=True)

            # Define global plotting parameters
            global_min_z = self.coordinates['zccf'].min()
            global_max_z = self.coordinates['zccf'].max()
            global_min_y = -self.coordinates['yccf'].max()
            global_max_y = -self.coordinates['yccf'].min()

        def plot_spatial_localNMF(spat, nmf_top, path_str, splitlevel):
            figs = []
            for NMF_I in range(nmf_top.shape[1]):
                results = []
                filtered_data = pd.concat([
                    pd.DataFrame(spat, columns=['zccf','yccf','Section']),
                    pd.DataFrame(nmf_top[:,NMF_I], columns=["test"])
                ], axis=1)

                currentNMF = "test"
                for section in filtered_data['Section'].unique():
                    subset = filtered_data[filtered_data['Section'] == section]
                    perc_2 = subset[currentNMF].quantile(0.02)
                    perc_98 = subset[currentNMF].quantile(0.98)
                    results.append([section, perc_2, perc_98])

                percentile_df = pd.DataFrame(results, columns=['Section', '2-perc', '98-perc'])
                med2p = percentile_df['2-perc'].median()
                med98p = percentile_df['98-perc'].median()

                cmap = plt.cm.PuOr
                fig, axes = plt.subplots(4, 8, figsize=(20, 10))
                axes = axes.flatten()

                for section in range(1, 33):
                    ax = axes[section - 1]
                    ddf = filtered_data[(filtered_data['Section'] == section)]
                    ax.scatter(ddf['zccf'], -ddf['yccf'], c=ddf[currentNMF], 
                             cmap="PuOr", s=0.5, rasterized=True, vmin=med2p, vmax=med98p)
                    ax.axis('off')
                    ax.set_aspect('equal')
                    ax.set_xlim(global_min_z, global_max_z)
                    ax.set_ylim(global_min_y, global_max_y)

                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                norm = Normalize(vmin=med2p, vmax=med98p)
                sm = ScalarMappable(norm=norm, cmap=cmap)
                fig.colorbar(sm, cax=cbar_ax)

                plt.tight_layout(rect=[0, 0, 0.9, 1])
                plot_path = os.path.join(plot_dir, f"split_{path_str}_level_{splitlevel}_NMF{NMF_I}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                figs.append(fig)
            return figs

        def plot_spatial_localNMF_kMeans(spat, kmeans_labels, path_str, splitlevel):
            dd2 = pd.DataFrame(spat, columns=['zccf','yccf','Section'])
            dd2['cat_code'] = pd.Series(np.array(kmeans_labels)).astype('category').cat.codes

            color_map = {0: 'purple', 1: 'yellow'}
            dd2['color'] = dd2['cat_code'].map(color_map)

            fig, axes = plt.subplots(4, 8, figsize=(40, 20))
            axes = axes.flatten()
            dot_size = 0.3

            for i, section_num in enumerate(range(1, 33)):
                ax = axes[i]
                xx = dd2[dd2["Section"] == section_num]
                ax.scatter(xx['zccf'], -xx['yccf'],
                         c=np.array(xx['color']), s=dot_size, alpha=1, rasterized=True)
                ax.axis('off')
                ax.set_aspect('equal')
                ax.set_xlim(global_min_z, global_max_z)
                ax.set_ylim(global_min_y, global_max_y)

            for j in range(i+1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plot_path = os.path.join(plot_dir, f"split_{path_str}_level_{splitlevel}_bipartition.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return fig

        def plot_embeddingsNMF(embeddings, kmeans_labels, path_str, splitlevel):
            figs = []
            for i in range(embeddings.shape[1]-1):
                fig = plt.figure(figsize=(10, 8))
                plt.scatter(embeddings[:, i][::10], embeddings[:, (i+1)][::10], 
                          c=kmeans_labels[::10], s=0.005, alpha=0.5, rasterized=True)
                plot_path = os.path.join(plot_dir, f"split_{path_str}_level_{splitlevel}_embedding_{i}_{i+1}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                figs.append(fig)
            return figs

        def plot_tSNE(kmeans_labels, path_str, splitlevel, current_adata):
            fig = plt.figure(figsize=(10, 8))
            # Plot all points in gray
            plt.scatter(self.adata.obsm['X_TSNE'][:,0], self.adata.obsm['X_TSNE'][:,1], 
                       s=0.0005, alpha=0.5, c="gray", rasterized=True)
            # Plot only the points corresponding to current_adata
            plt.scatter(self.adata[current_adata.obs_names].obsm['X_TSNE'][:,0], 
                       self.adata[current_adata.obs_names].obsm['X_TSNE'][:,1], 
                       s=0.005, alpha=1, c=kmeans_labels, rasterized=True)
            plot_path = os.path.join(plot_dir, f"split_{path_str}_level_{splitlevel}_tsne.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return fig

        # 1) SPLIT INTO TRAIN/VAL/TEST BASED ON SECTIONS OR RANDOM
        unique_sections = self.coordinates['Section'].unique()
        num_sections = len(unique_sections)

        if num_sections >= 3:
            # Initialize splits
            if num_sections >= 5:
                valsec = (unique_sections[::5] + 2)[:-1]
                testsec = (unique_sections[::5] + 1)[:-1]
            else:
                valsec = np.array([unique_sections[0]])
                testsec = np.array([unique_sections[1]])

            if len(valsec) == 0:
                valsec = np.array([unique_sections[0]])
            if len(testsec) == 0:
                for sec in unique_sections:
                    if sec not in valsec:
                        testsec = np.array([sec])
                        break

            trainsec = np.setdiff1d(np.setdiff1d(unique_sections, testsec), valsec)
            if len(trainsec) == 0:
                if len(valsec) > len(testsec):
                    trainsec = np.array([valsec[-1]])
                    valsec = valsec[:-1]
                else:
                    trainsec = np.array([testsec[-1]])
                    testsec = testsec[:-1]

            valpoints = self.coordinates.loc[self.coordinates['Section'].isin(valsec), :].index
            testpoints = self.coordinates.loc[self.coordinates['Section'].isin(testsec), :].index
            trainpoints = self.coordinates.loc[self.coordinates['Section'].isin(trainsec), :].index

        else:
            # 60-20-20 random split
            all_indices = self.coordinates.index.values.copy()
            np.random.shuffle(all_indices)
            n_samples = len(all_indices)
            n_train = int(0.6 * n_samples)
            n_val = int(0.2 * n_samples)

            trainpoints = all_indices[:n_train]
            valpoints = all_indices[n_train:n_train + n_val]
            testpoints = all_indices[n_train + n_val:]

        # 3) INITIALIZE CLUSTERING LOG
        column_names = [f"level_{i}" for i in range(1, max_depth + 1)]
        clusteringLOG = pd.DataFrame(
            0,
            index=self.reconstructed_data_df.index,
            columns=column_names
        )[::ds_factor]

        # 4) DEFINE RECURSIVE SPLITTING FUNCTION
        def _dosplit(current_adata, embds, path=None, splitlevel=0):
            if path is None:
                path = []

            # a) STOP CONDITIONS
            if current_adata.X.shape[0] < min_voxels:
                return None
            if splitlevel > max_depth:
                return None

            # b) COMPUTE LOCAL NMF
            nmfdf, loadings, N_factors, nmf_model = self._compute_seeded_NMF(
                pd.DataFrame(current_adata.X, index=current_adata.obs_names)
            )
            nmf_result = nmfdf.values

            # c) FILTER LOW-SIGNAL COMPONENTS
            filter1 = np.abs(nmf_result).mean(axis=0) > thr_signal
            loadings_sel = loadings[filter1, :]
            nmf_result = nmf_result[:, filter1]
            original_nmf_indices = np.arange(N_factors)[filter1]

            tempadata = sc.AnnData(X=nmf_result)
            tempadata.obsm['spatial'] = current_adata.obsm['spatial']

            # d) RANK FEATURES
            goodpcs = self._rank_features_by_combined_score(tempadata)
            goodpcs_indices = original_nmf_indices[goodpcs.astype(int)]
            top_pcs_data = nmf_result[:, goodpcs.astype(int)]
            loadings_sel = loadings_sel[goodpcs.astype(int), :]

            # e) GENERATE COMBINATIONS
            multiplets = self._generate_combinations(len(goodpcs), limit=combinations)
            flag = False
            aaa = 0

            # f) ITERATIVE SEARCH FOR ACCEPTABLE SPLIT
            while (not flag) and (aaa < len(multiplets)):
                bestpcs = multiplets[aaa]
                embeddings_local = top_pcs_data[:, bestpcs]
                loadings_current = loadings_sel[list(bestpcs), :]
                selected_nmf_indices = goodpcs_indices[list(bestpcs)]

                scaler_local = StandardScaler()
                standardized_embeddings = scaler_local.fit_transform(embeddings_local)

                # Combine embeddings: local, previous-level, global
                globembds = self.standardized_embeddings_GLOBAL.loc[current_adata.obs_names].values / penalty2
                embspace = np.concatenate(
                    (standardized_embeddings,
                     embds / penalty1,
                     globembds),
                    axis=1
                )

                # KMeans + backSPIN reaggregation
                kmeans = KMeans(n_clusters=K, random_state=230598)
                kmeans_labels = kmeans.fit_predict(embspace)

                data_for_clustering = pd.DataFrame(
                    current_adata.X,
                    index=current_adata.obs_names,
                    columns=current_adata.var_names
                )
                data_for_clustering['label'] = kmeans_labels

                centroids = data_for_clustering.groupby('label').mean()
                centroids = pd.DataFrame(
                    StandardScaler().fit_transform(centroids),
                    columns=centroids.columns,
                    index=centroids.index
                ).T
                row_ix, columns_ix = backSPIN.SPIN(centroids, widlist=4)
                centroids = centroids.iloc[row_ix, columns_ix]
                _, _, _, gr1, gr2, _, _, _, _ = backSPIN._divide_to_2and_resort(
                    sorted_data=centroids.values,
                    wid=5
                )
                gr1 = np.array(centroids.columns)[gr1]
                gr2 = np.array(centroids.columns)[gr2]

                data_for_clustering['lab'] = 1
                data_for_clustering.loc[
                    data_for_clustering['label'].isin(gr2), 'lab'
                ] = 2

                # Plot embeddings and bipartition
                if do_plotting:
                    path_str = "_".join(map(str, path)) if path else "root"
                    
                    # Plot spatial embeddings
                    plot_spatial_localNMF(current_adata.obsm['spatial'], standardized_embeddings, path_str, splitlevel)
                    
                    # Plot embeddings NMF
                    plot_embeddingsNMF(standardized_embeddings, data_for_clustering['lab'], path_str, splitlevel)
                    
                    # Plot spatial bipartition
                    plot_spatial_localNMF_kMeans(current_adata.obsm['spatial'], data_for_clustering['lab'], path_str, splitlevel)
                    
                    # Plot tSNE
                    plot_tSNE(data_for_clustering['lab'], path_str, splitlevel, current_adata)

                # Continuity check
                (enough_sections0,
                 enough_sections1,
                 num_peaks0,
                 num_peaks1,
                 peak_ratio0,
                 peak_ratio1) = self._continuity_check(
                    current_adata.obsm['spatial'],
                    kmeans_labels=np.array(data_for_clustering['lab']),
                    spat_columns=spat_columns,
                    min_val_threshold=min_val_threshold,
                    min_nonzero_sections=min_nonzero_sections,
                    gaussian_sigma=gaussian_sigma,
                    default_peak_ratio=default_peak_ratio
                )

                # Differential lipids
                alteredlips, promoted = self._differential_lipids(
                    current_adata.obsm['lipids'].values,
                    kmeans_labels,
                    min_fc,
                    pthr
                )

                # Check split criteria
                cond_size = (np.sum(kmeans_labels == 1) > min_voxels) or (np.sum(kmeans_labels == 0) > min_voxels)
                cond_diff = (alteredlips > min_diff_lipids)
                cond_cont0 = ((num_peaks0 < peak_count_threshold) or (peak_ratio0 > peak_ratio_threshold))
                cond_cont1 = ((num_peaks1 < peak_count_threshold) or (peak_ratio1 > peak_ratio_threshold))

                flag = cond_size and cond_diff and enough_sections0 and enough_sections1 and cond_cont0 and cond_cont1

                aaa += 1
                kmeans_labels = data_for_clustering['lab'].astype(int)

            if not flag:
                return None

            # g) TRAIN XGBOOST CLASSIFIER ON EMBEDDINGS
            embeddings_df = pd.DataFrame(embspace, index=current_adata.obs_names)

            X_train = embeddings_df.loc[embeddings_df.index.isin(trainpoints), :]
            X_val = embeddings_df.loc[embeddings_df.index.isin(valpoints), :]
            X_test = embeddings_df.loc[embeddings_df.index.isin(testpoints), :]

            kmeans_labels = kmeans_labels - 1
            y_train = kmeans_labels.loc[X_train.index]
            y_val = kmeans_labels.loc[X_val.index]
            y_test = kmeans_labels.loc[X_test.index]

            X_train_sub, y_train_sub = self._undersample(X_train, y_train)

            xgb_model = XGBClassifier(
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample_bytree,
                gamma=xgb_gamma,
                random_state=xgb_random_state,
                n_jobs=xgb_n_jobs
            )
            xgb_model.fit(
                X_train_sub,
                y_train_sub,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )

            test_pred = xgb_model.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            if test_acc < ACCTHR:
                return None

            # h) OVERWRITE CLUSTER LABELS WITH CLASSIFIER PREDICTIONS
            new_labels = pd.concat([
                pd.Series(xgb_model.predict(X_train), index=X_train.index),
                pd.Series(xgb_model.predict(X_val), index=X_val.index),
                pd.Series(xgb_model.predict(X_test), index=X_test.index)
            ])
            new_labels = new_labels.loc[embeddings_df.index] + 1

            clusteringLOG.loc[new_labels.index, f"level_{splitlevel+1}"] = new_labels.values

            # i) BUILD NODE AND RECURSE
            node = Node(splitlevel, path=path)
            node.scaler = scaler_local
            node.nmf = nmf_model
            node.xgb_model = xgb_model
            node.feature_importances = xgb_model.feature_importances_
            node.factors_to_use = selected_nmf_indices

            idx0 = embeddings_df.index[new_labels == 1]
            idx1 = embeddings_df.index[new_labels == 2]

            adata0 = current_adata[current_adata.obs_names.isin(idx0)]
            adata1 = current_adata[current_adata.obs_names.isin(idx1)]

            emb_local_df = pd.DataFrame(standardized_embeddings,
                                        index=current_adata.obs_names)
            embd0 = emb_local_df.loc[idx0].values
            embd1 = emb_local_df.loc[idx1].values

            child0 = _dosplit(adata0, embd0, path + [0], splitlevel + 1)
            child1 = _dosplit(adata1, embd1, path + [1], splitlevel + 1)

            node.children[0] = child0
            node.children[1] = child1
            return node

        # 5) START RECURSION
        root_node = _dosplit(
            self.adatamaia[::ds_factor],
            self.standardized_embeddings_GLOBAL[::ds_factor].values,
            path=[],
            splitlevel=0
        )

        # 6) SAVE RESULTS
        clusteringLOG.to_parquet(f"{self.analysis_name}_tree_clustering_euclid.parquet")
        with open(f"{self.analysis_name}_rootnode_clustering_euclid.pkl", "wb") as f:
            pickle.dump(root_node, f)

        return root_node, clusteringLOG


    def apply_euclid_clustering(self, root_node, adata_new, standardized_global_new,
                                penalty1=1.5, penalty2=2.0, ds_factor=1):
        """
        Apply a learned Euclid clustering tree to new data, using precomputed standardized global embeddings.

        Parameters
        ----------
        root_node : Node
            The root node of a previously learned hierarchical clustering tree.
        adata_new : sc.AnnData
            New AnnData object to apply the clustering to.
        standardized_global_new : pd.DataFrame
            Precomputed standardized global embeddings for the new data.
        penalty1 : float, optional
            Penalty factor for previous-level embeddings. Default is 1.5.
        penalty2 : float, optional
            Penalty factor for global embeddings. Default is 2.0.
        ds_factor : int, optional
            Downsampling factor for the dataset. Default is 1.

        Returns
        -------
        paths_df : pd.DataFrame
            A DataFrame where each row corresponds to an observation in adata_new
            and each column level_i indicates the cluster assignment at that level.
        """
        standardized_global_new = standardized_global_new.loc[adata_new.obs_names]

        paths = {obs: [] for obs in adata_new.obs_names}

        def traverse_tree(node, current_adata, embds_local, level=0):
            if node is None or not node.children:
                return
            if current_adata.shape[0] == 0:
                return

            # 1) Apply stored NMF
            nmf_model = node.nmf
            X_nmf = nmf_model.transform(current_adata.X)

            # 2) Select factors
            factors_to_use = node.factors_to_use
            X_nmf_sel = X_nmf[:, factors_to_use]

            # 3) Scale with stored scaler
            scaler_local = node.scaler
            X_scaled = scaler_local.transform(X_nmf_sel)

            # 4) Global embeddings for subset
            glob_emb_subset = standardized_global_new.loc[current_adata.obs_names].values / penalty2

            # 5) Combine embeddings
            embspace = np.concatenate(
                (X_scaled,
                 embds_local / penalty1,
                 glob_emb_subset),
                axis=1
            )

            # 6) Predict child labels
            xgb_model = node.xgb_model
            child_labels = xgb_model.predict(embspace)
            child_labels_adjusted = child_labels + 1

            # 7) Record labels
            for i, obs in enumerate(current_adata.obs_names):
                paths[obs].append(int(child_labels_adjusted[i]))

            # 8) Split into children
            mask_child0 = (child_labels == 0)
            mask_child1 = (child_labels == 1)

            obs_child0 = current_adata.obs_names[mask_child0]
            obs_child1 = current_adata.obs_names[mask_child1]

            adata_child0 = current_adata[current_adata.obs_names.isin(obs_child0)]
            adata_child1 = current_adata[current_adata.obs_names.isin(obs_child1)]

            # 9) Prepare embeddings for next
            emb0 = X_scaled[mask_child0]
            emb1 = X_scaled[mask_child1]

            # 10) Recurse
            traverse_tree(node.children[0], adata_child0, emb0, level + 1)
            traverse_tree(node.children[1], adata_child1, emb1, level + 1)

        if ds_factor > 1:
            adata_to_apply = adata_new[::ds_factor]
            global_to_apply = standardized_global_new.loc[adata_to_apply.obs_names]
        else:
            adata_to_apply = adata_new
            global_to_apply = standardized_global_new

        emb_initial = global_to_apply.values
        traverse_tree(root_node, adata_to_apply, emb_initial, level=0)

        max_depth_observed = max(len(p) for p in paths.values()) if paths else 0
        columns = [f"level_{i+1}" for i in range(max_depth_observed)]
        paths_df = pd.DataFrame(index=adata_to_apply.obs_names, columns=columns, dtype=float)

        for obs, path in paths.items():
            padded = path + [np.nan] * (max_depth_observed - len(path))
            paths_df.loc[obs, :] = padded

        for i, col in enumerate(columns):
            adata_new.obs[col] = np.nan
            adata_new.obs.loc[paths_df.index, col] = paths_df[col].values

        print("ciao")
        return paths_df

    def assign_cluster_colors(self, tree):
        """
        Compute a color assignment for lipizones (clusters) based on the hierarchy of splits.
        
        Parameters
        ----------
        tree : pd.DataFrame
            A DataFrame that contains the clustering hierarchy (e.g. split history).

        Returns
        -------
        lipizone_colors : pd.Series
            A Series mapping each observation to a hex color.
        """
        coords = self.coordinates.fillna(0).replace([np.inf, -np.inf], 0) ##########################
        xs = (coords['xccf']*40).astype(int)
        ys = (coords['yccf']*40).astype(int)
        zs = (coords['zccf']*40).astype(int)
        xs.loc[xs>527] = 527
        ys.loc[ys>319] = 319
        zs.loc[zs>455] = 455
        
        # Add standardization step
        data_std = (self.adata.X - np.mean(self.adata.X, axis=1)[:, None]) / (np.std(self.adata.X, axis=1)[:, None] + 1e-8)
        data_std = pd.DataFrame(data_std, index=self.adata.obs_names, columns=self.adata.var_names)
        
        # Normalize the data per pixel
        levels = pd.concat([data_std, self.coordinates.loc[self.adata.obs_names]], axis=1)
        levels = pd.concat([levels, tree], axis=1)
        dd2 = levels.copy()
        
        # For each unique division in a column "class" (built from earlier splits)
        divisions = dd2['class'].unique()
        colormaps = ["RdYlBu", "terrain", "PiYG", "cividis", "plasma", "PuRd", "inferno", "PuOr"]
        dd2['R'] = np.nan
        dd2['G'] = np.nan
        dd2['B'] = np.nan
        
        from tqdm import tqdm
        
        for division, cmap_name in tqdm(zip(divisions, colormaps)):
            if len(dd2.loc[dd2['class'] == division, 'cluster'].unique()) > 1:
                print(division)  # Add debug print
                
                datasub = dd2[dd2['class'] == division]
                clusters = datasub['cluster'].unique()
                lipid_df = pd.DataFrame(columns=self.adata.var_names)
                
                for i in range(len(clusters)):
                    sub = datasub[datasub['cluster'] == clusters[i]]
                    lipid_data = sub[self.adata.var_names].mean(axis=0)
                    lipid_df = pd.concat([lipid_df, pd.DataFrame([lipid_data], columns=self.adata.var_names)], ignore_index=True)
                
                column_means = lipid_df.mean()
                normalized_lipid_df = lipid_df.div(column_means, axis='columns')
                normalized_lipid_df.index = clusters
                normalized_lipid_df = normalized_lipid_df.T
                
                # Compute centroids and distance matrix
                pca_columns = datasub[self.adata.var_names]
                grouped = datasub[['cluster']].join(pca_columns)
                centroids = grouped.groupby('cluster').mean()
                distance_matrix = squareform(pdist(centroids, metric='euclidean'))
                distance_df = pd.DataFrame(distance_matrix, index=centroids.index, columns=centroids.index)
                
                np.fill_diagonal(distance_df.values, np.inf)
                init_idx = np.unravel_index(np.argmin(distance_df.values), distance_df.shape)
                ordered_elements = [distance_df.index[init_idx[0]], distance_df.columns[init_idx[1]]]
                distances = [0, distance_df.iloc[init_idx]]
                
                while len(ordered_elements) < len(distance_df):
                    last = ordered_elements[-1]
                    remaining = distance_df.loc[last, ~distance_df.columns.isin(ordered_elements)]
                    next_elem = remaining.idxmin()
                    ordered_elements.append(next_elem)
                    distances.append(remaining[next_elem])
                
                cumulative = np.cumsum(distances)
                normalized_dist = cumulative / cumulative[-1]
                cmap = plt.get_cmap(cmap_name)
                colors_rgb = [cmap(val) for val in normalized_dist]
                hsv = [mcolors.rgb_to_hsv(rgb[:3]) for rgb in colors_rgb]
                modified_hsv = []
                
                for i, (h, s, v) in enumerate(hsv):
                    if (i+1) % 2 != 0:
                        s = min(1, s + 0.7 * s)
                    modified_hsv.append((h, s, v))
                
                modified_rgb = [mcolors.hsv_to_rgb(hsv_val) for hsv_val in modified_hsv]
                lipocolor = pd.DataFrame(modified_rgb, index=ordered_elements, columns=['R','G','B'])
                lipocolor_reset = lipocolor.reset_index().rename(columns={'index': 'cluster'})
                
                print(lipocolor_reset)  # Add debug print
                
                # Update using the index-based approach (similar to the original function)
                indices = datasub.index
                datasub_rgb = datasub.copy()
                
                # Create a DataFrame with just the RGB columns and the right index
                rgb_df = pd.merge(datasub_rgb[['cluster']], lipocolor_reset, on='cluster', how='left')
                rgb_df.index = datasub_rgb.index
                
                # Update only the RGB columns
                dd2.loc[indices, 'R'] = rgb_df['R']
                dd2.loc[indices, 'G'] = rgb_df['G']
                dd2.loc[indices, 'B'] = rgb_df['B']
            else:
                sub = dd2[dd2['class'] == division]
                sub['R'] = 0; sub['G'] = 0; sub['B'] = 0
                dd2.update(sub[['R','G','B']])
        
        def rgb_to_hex(r, g, b):
            try:
                r, g, b = [int(255*x) for x in [r, g, b]]
                return f'#{r:02x}{g:02x}{b:02x}'
            except:
                return np.nan
        
        dd2['lipizone_color'] = dd2.apply(lambda row: rgb_to_hex(row['R'], row['G'], row['B']), axis=1)
        
        self.adata.obs['lipizone_color'] = dd2['lipizone_color'].values

    def name_lipizones_anatomy(self, acronym_column, lipizone_column):
        """
        Assign anatomical names to clusters based on the cross-tabulation of acronyms and lipizone labels.
        
        Parameters
        ----------
        acronyms : pd.Series
            Anatomical acronyms per pixel.
        lipizones : pd.Series
            Cluster labels per pixel.
        
        Returns
        -------
        mapping_df : pd.DataFrame
            A mapping (and heatmap) of anatomical enrichment per lipizone.
        """
        acronyms = self.adata.obs[acronym_column]
        lipizones = self.adata.obs[lipizone_column]
        acronyms = acronyms[acronyms.isin(acronyms.value_counts().index[acronyms.value_counts() > 50])]
        lipizones = lipizones.loc[acronyms.index]
        cmat = pd.crosstab(acronyms, lipizones)
        normalized_df1 = cmat / cmat.sum()
        normalized_df1 = (normalized_df1.T / normalized_df1.T.mean()).T
        cmat2 = pd.crosstab(lipizones, acronyms)
        normalized_df2 = cmat2 / cmat2.sum()
        normalized_df2 = (normalized_df2.T / normalized_df2.T.mean()).T
        normalized_df = normalized_df2.T * normalized_df1
        # Hierarchically order clusters:
        from scipy.cluster import hierarchy as sch
        linkage_matrix = sch.linkage(sch.distance.pdist(normalized_df.T), method='weighted', optimal_ordering=True)
        order = sch.leaves_list(linkage_matrix)
        normalized_df = normalized_df.iloc[:, order]
        order_indices = np.argmax(normalized_df.values, axis=1)
        order_indices = np.argsort(order_indices)
        normalized_df = normalized_df.iloc[order_indices, :]
        # Plot heatmap (save to PDF)
        plt.figure(figsize=(10, 10))
        import seaborn as sns
        sns.heatmap(normalized_df, cmap="Grays", cbar_kws={'label': 'Enrichment'},
                    xticklabels=True, yticklabels=False,
                    vmin=np.percentile(normalized_df, 2), vmax=np.percentile(normalized_df, 98))
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False)
        plt.tight_layout()
        plt.show()
        
        # Create lipizone names based on enriched acronyms
        lipizone_names = {}
        for lipizone in normalized_df.columns:
            # Find acronyms with enrichment > 300 for this lipizone
            enriched_acronyms = normalized_df[normalized_df[lipizone] > 300].index.tolist()
            # Concatenate with "-" separator, or use "Broad" if no enriched acronyms
            if enriched_acronyms:
                lipizone_name = "-".join(enriched_acronyms)
            else:
                lipizone_name = "Broad"
            lipizone_names[lipizone] = lipizone_name
        
        # Handle duplicate names by adding suffixes
        name_counts = {}
        unique_lipizone_names = {}
        
        for lipizone, name in lipizone_names.items():
            if name in name_counts:
                name_counts[name] += 1
                unique_name = f"{name}_{name_counts[name]}"
            else:
                name_counts[name] = 0
                unique_name = name
            unique_lipizone_names[lipizone] = unique_name
        
        # Add lipizone names to adata.obs
        # First initialize the column
        self.adata.obs['lipizone_names'] = "Unassigned"
        
        # Map the lipizone values to their names
        lipizone_series = self.adata.obs[lipizone_column]
        for lipizone_value, name in unique_lipizone_names.items():
            mask = lipizone_series == lipizone_value
            self.adata.obs.loc[mask, 'lipizone_names'] = name

    def clean_corrupted_pdfs(self, output_folder=None):
        """
        Clean up any corrupted PDF files in the output folder.
        
        Parameters
        ----------
        output_folder : str, optional
            Folder containing the PDF files to check. If None, uses analysis_name prefix.
        """
        if output_folder is None:
            output_folder = f"{self.analysis_name}_lipizones_output"
        if not os.path.exists(output_folder):
            return
            
        corrupted_files = []
        for fname in os.listdir(output_folder):
            if fname.endswith(".pdf"):
                pdf_path = os.path.join(output_folder, fname)
                try:
                    test_reader = PdfReader(pdf_path)
                    # Try to access the first page to ensure file is valid
                    if len(test_reader.pages) > 0:
                        _ = test_reader.pages[0]
                except Exception as e:
                    print(f"Removing corrupted PDF: {fname}")
                    corrupted_files.append(fname)
                    os.remove(pdf_path)
        
        if corrupted_files:
            print(f"Cleaned up {len(corrupted_files)} corrupted PDF files")
        else:
            print("No corrupted PDF files found")

    def clusters_to_pdf(self, output_folder=None, pdf_filename=None):
        """
        Plot each cluster (lipizone) separately into PDF files.
        
        Parameters
        ----------
        lipizone_names : pd.Series
            A Series of cluster labels (or names) per pixel.
        output_folder : str, optional
            Folder in which to save individual PDFs. If None, uses analysis_name prefix.
        pdf_filename : str, optional
            Final merged PDF filename. If None, uses analysis_name prefix.
        """
        if output_folder is None:
            output_folder = f"{self.analysis_name}_lipizones_output"
        if pdf_filename is None:
            pdf_filename = f"{self.analysis_name}_clusters_combined.pdf"
        os.makedirs(output_folder, exist_ok=True)
        # Extract coordinate and lipizone information from adata.obs
        # Get the subset of adata.obs corresponding to reconstructed_data_df
        levels = self.adata.obs.loc[self.reconstructed_data_df.index].copy()
        
        # Ensure we have the necessary columns
        required_cols = ['SectionID', 'xccf', 'yccf', 'zccf', 'lipizone_names']
        for col in required_cols:
            if col not in levels.columns:
                raise KeyError(f"Required column '{col}' not found in adata.obs. Available columns: {list(levels.columns)}")
        dot_size = 0.3
        sections_to_plot = levels['SectionID'].unique()
        global_min_z = levels['zccf'].min()
        global_max_z = levels['zccf'].max()
        global_min_y = -levels['yccf'].max()
        global_max_y = -levels['yccf'].min()
        unique_names = np.sort(levels['lipizone_names'].unique())
        # Plot each cluster to its own PDF
        for uniq in tqdm(unique_names):
            fig, axes = plt.subplots(4, 8, figsize=(40, 20))
            axes = axes.flatten()
            for i, sec in enumerate(sections_to_plot):
                ax = axes[i]
                subset = levels[levels["SectionID"] == sec]
                ax.scatter(subset['zccf'], -subset['yccf'], c=subset['lipizone_names'].astype("category").cat.codes,
                           cmap='Greys', s=dot_size*2, alpha=0.2, rasterized=True)
                highlight = subset[subset['lipizone_names'] == uniq]
                ax.scatter(highlight['zccf'], -highlight['yccf'], c='red', s=dot_size, alpha=1, rasterized=True)
                ax.axis('off')
                ax.set_aspect('equal')
                ax.set_xlim(global_min_z, global_max_z)
                ax.set_ylim(global_min_y, global_max_y)
            for j in range(i+1, len(axes)):
                fig.delaxes(axes[j])
            plt.suptitle(uniq)
            plt.tight_layout()
            # Replace "/" with "_" in filename to avoid directory separator issues
            safe_filename = uniq.replace("/", "_").replace("\\", "_")
            outpath = os.path.join(output_folder, f"{safe_filename}.pdf")
            try:
                plt.savefig(outpath, bbox_inches='tight', dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"Warning: Failed to save PDF for {uniq}: {str(e)}")
                plt.close(fig)
                # Remove potentially corrupted file
                if os.path.exists(outpath):
                    os.remove(outpath)
        # Merge all PDFs into one
        cwd = os.getcwd()
        merger = PdfMerger()
        successful_merges = 0
        failed_files = []
        
        for fname in sorted(os.listdir(output_folder)):
            if fname.endswith(".pdf"):
                pdf_path = os.path.join(output_folder, fname)
                try:
                    # Test if the PDF can be read before merging
                    from PyPDF2 import PdfReader
                    test_reader = PdfReader(pdf_path)
                    # If we can read it, add it to the merger
                    merger.append(pdf_path)
                    successful_merges += 1
                except Exception as e:
                    print(f"Warning: Skipping corrupted PDF {fname}: {str(e)}")
                    failed_files.append(fname)
                    continue
        
        if successful_merges > 0:
            final_pdf = os.path.join(cwd, pdf_filename)
            merger.write(final_pdf)
            merger.close()
            print(f"Merged PDF saved as {final_pdf}")
            print(f"Successfully merged {successful_merges} PDFs")
            if failed_files:
                print(f"Failed to merge {len(failed_files)} PDFs: {failed_files}")
        else:
            merger.close()
            print("Error: No valid PDFs found to merge")

    def compare_leiden_lipizone(self, leiden_key="X_Leiden", lipizone_key="lipizone", output_file=None):
        """
        Compare Leiden clustering results with lipizone annotations.
        
        Parameters
        ----------
        leiden_key : str, optional
            Key in adata.obs containing Leiden clustering results
        lipizone_key : str, optional
            Key in adata.obs containing lipizone annotations
        output_file : str, optional
            Path to save the output visualization. If None, plot is shown but not saved.
            
        Returns
        -------
        pd.DataFrame
            The normalized contingency matrix showing the relationship between
            Leiden clusters and lipizones
        """
        # Create contingency matrix

        print(self.adata.obs[leiden_key])
        print(self.adata.obs[lipizone_key])

        cm = pd.crosstab(self.adata.obs[leiden_key], self.adata.obs[lipizone_key])
        
        # Calculate fractions
        fraction = cm / cm.sum()
        
        # Perform hierarchical clustering
        import scipy.cluster.hierarchy as sch
        
        # Cluster columns (lipizones)
        linkage = sch.linkage(sch.distance.pdist(fraction.T), method='weighted', optimal_ordering=True)
        order = sch.leaves_list(linkage)
        df = fraction.iloc[:, order]
        
        # Cluster rows (Leiden clusters)
        order = np.argmax(df.values, axis=1)
        order = np.argsort(order)
        df = df.iloc[order,:]
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(df.T, cmap="Reds", vmax=0.5)
        plt.colorbar(label='Fraction')
        plt.xlabel('Leiden Clusters')
        plt.ylabel('Lipizones')
        plt.title('Relationship between Leiden Clusters and Lipizones')
        
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
        return df

    def add_clustering_to_adata(self, clusteringLOG):
        """
        Add clustering results from clusteringLOG DataFrame to AnnData object.
        
        Parameters
        ----------
        clusteringLOG : pd.DataFrame
            DataFrame containing clustering results with columns for each level.
            
        Returns
        -------
        sc.AnnData
            The AnnData object with added clustering results in obs.
        """
        # Ensure the index of clusteringLOG matches the observations in adata
        if not all(idx in self.adata.obs_names for idx in clusteringLOG.index):
            raise ValueError("Some indices in clusteringLOG are not present in adata.obs_names")
            
        # Get all level columns
        level_cols = [col for col in clusteringLOG.columns if col.startswith('level_')]
        
        # Add each level to adata.obs
        for col in level_cols:
            self.adata.obs[col] = np.nan
            self.adata.obs.loc[clusteringLOG.index, col] = clusteringLOG[col].values
            
        # Compute class, lipizone, and cluster
        # First ensure all level columns exist
        required_levels = ['level_1', 'level_2', 'level_3', 'level_4', 'level_5', 'level_6']
        for level in required_levels:
            if level not in clusteringLOG.columns:
                clusteringLOG[level] = np.nan

        for lvl in ['level_1','level_2','level_3','level_4','level_5','level_6']:
            clusteringLOG[lvl] = clusteringLOG[lvl].astype(int).fillna(0)
                
        # Compute class (first 3 levels)
        clusteringLOG['class'] = clusteringLOG['level_1'].astype(str) + \
                                clusteringLOG['level_2'].astype(str) + \
                                clusteringLOG['level_3'].astype(str)
                                
        # Compute lipizone (all 6 levels)
        clusteringLOG['lipizone'] = clusteringLOG['level_1'].astype(str) + \
                                   clusteringLOG['level_2'].astype(str) + \
                                   clusteringLOG['level_3'].astype(str) + \
                                   clusteringLOG['level_4'].astype(str) + \
                                   clusteringLOG['level_5'].astype(str) + \
                                   clusteringLOG['level_6'].astype(str) + ".0"
                                   
        # Set cluster equal to lipizone
        clusteringLOG['cluster'] = clusteringLOG['lipizone']
        
        # Add computed columns to adata.obs
        for col in ['class', 'lipizone', 'cluster']:
            self.adata.obs[col] = np.nan
            self.adata.obs.loc[clusteringLOG.index, col] = clusteringLOG[col].values

    def paint_lipizones(self, source_adata):
        """
        Transfer lipizone names and colors from a source AnnData object to this clustering object.
        
        This function maps the 'lipizone_names' and 'lipizone_color' from the source adata
        to the current clustering object's adata based on the 'lipizone' identifier.
        
        Parameters
        ----------
        source_adata : sc.AnnData
            Source AnnData object containing 'lipizone', 'lipizone_names', and 'lipizone_color' 
            in the obs slot.
            
        Raises
        ------
        ValueError
            If required columns are missing from source_adata.obs or if no mapping can be created.
        """
        # Validate that source_adata has the required columns
        required_cols = ['lipizone', 'lipizone_names', 'lipizone_color']
        missing_cols = [col for col in required_cols if col not in source_adata.obs.columns]
        if missing_cols:
            raise ValueError(f"Source adata is missing required columns: {missing_cols}")
            
        mask = source_adata.obs[required_cols].notna().all(axis=1)
        source_adata = source_adata[mask].copy()
        # Validate that self.adata has the lipizone column
        if 'lipizone' not in self.adata.obs.columns:
            raise ValueError("Current adata is missing 'lipizone' column. Run add_clustering_to_adata first.")
            
        # Create mapping from lipizone to lipizone_names and lipizone_color
        source_mapping = source_adata.obs[['lipizone', 'lipizone_names', 'lipizone_color']].drop_duplicates()
        
        # Check for one-to-one mapping
        if source_mapping['lipizone'].duplicated().any():
            print("Warning: Non-unique mapping detected. Using first occurrence for each lipizone.")
            source_mapping = source_mapping.drop_duplicates(subset=['lipizone'], keep='first')
            
        # Create dictionaries for mapping
        lipizone_to_names = dict(zip(source_mapping['lipizone'], source_mapping['lipizone_names']))
        lipizone_to_colors = dict(zip(source_mapping['lipizone'], source_mapping['lipizone_color']))

        print(lipizone_to_colors)
        
        # Map the values to current adata
        self.adata.obs['lipizone_names'] = self.adata.obs['lipizone'].map(lipizone_to_names)
        self.adata.obs['lipizone_color'] = self.adata.obs['lipizone'].map(lipizone_to_colors)
        
        # Report mapping statistics
        mapped_count = self.adata.obs['lipizone_names'].notna().sum()
        total_count = len(self.adata.obs)
        unmapped_lipizones = self.adata.obs.loc[self.adata.obs['lipizone_names'].isna(), 'lipizone'].unique()
        
        print(f"Successfully mapped {mapped_count}/{total_count} observations.")
        if len(unmapped_lipizones) > 0:
            print(f"Unmapped lipizones: {unmapped_lipizones}")


    def plot_distribution_of_pixels_per_lipizone(self, figsize=(12, 6)):
        """
        Plot the distribution of number of pixels per lipizone as a histogram.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size (width, height). Default is (12, 6).
        """
        # Count pixels per lipizone
        pixel_counts = self.adata.obs['lipizone_names'].value_counts()

        # Histogram of pixel counts
        plt.figure(figsize=figsize)
        plt.hist(pixel_counts.values, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Pixels')
        plt.ylabel('Frequency')
        plt.title('Distribution of Pixel Counts per Lipizone')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_histogram_of_lipids_per_lipizone(self, thr1=0.8, thr2=0.0001, figsize=(10, 6)):
        """
        Plot histogram of the number of lipids per lipizone using threshold criteria.
        A lipid is considered present in a lipizone if > thr1 fraction of its pixels have intensity > thr2.
        
        Parameters
        ----------
        thr1 : float, optional
            Proportion threshold (0 < thr1 <= 1). Default is 0.8.
        thr2 : float, optional
            Intensity threshold to count a lipid as present in a pixel. Default is 0.0001.
        figsize : tuple, optional
            Figure size (width, height). Default is (10, 6).
        """
        import numpy as np
        import pandas as pd

        # Extract the lipid data as a DataFrame
        lipid_data = pd.DataFrame(self.adata.X, 
                                index=self.adata.obs_names, 
                                columns=self.adata.var_names)
        lipid_data['lipizone_names'] = self.adata.obs['lipizone_names']

        lipids_count_per_zone = []

        # Loop over each lipizone (skip 'Unassigned')
        for lipizone in self.adata.obs['lipizone_names'].unique():
            if lipizone == 'Unassigned':
                continue

            # Subset to pixels in this lipizone
            mask = lipid_data['lipizone_names'] == lipizone
            sub_df = lipid_data.loc[mask, self.adata.var_names]

            # For each lipid (column), compute fraction of pixels above thr2
            # Count the number of lipids meeting the > thr1 criterion
            frac_above = (sub_df > thr2).sum(axis=0) / sub_df.shape[0]
            n_lipids = (frac_above > thr1).sum()
            lipids_count_per_zone.append(n_lipids)

        # Plot histogram of lipid counts per lipizone
        plt.figure(figsize=figsize)
        plt.hist(lipids_count_per_zone, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Lipids per Lipizone')
        plt.ylabel('Frequency')
        plt.title('Distribution of Lipid Counts per Lipizone')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def save_msi_dataset(self, filename=None):
        """
        Save the current AnnData object to disk.

        Parameters
        ----------
        filename : str, optional
            File path to save the AnnData object. If None, uses analysis_name prefix.
        """
        if filename is None:
            filename = f"{self.analysis_name}_clustering_msi_dataset.h5ad"
        self.adata.write_h5ad(filename)

    def load_msi_dataset(self, filename=None):
        """
        Load an AnnData object from disk and reinitialize the Clustering object.

        Parameters
        ----------
        filename : str, optional
            File path from which to load the AnnData object. If None, uses analysis_name prefix.

        Returns
        -------
        sc.AnnData
            The loaded AnnData object.
        """
        if filename is None:
            filename = f"{self.analysis_name}_clustering_msi_dataset.h5ad"

        class EmbeddingWrapper:
            def __init__(self, adata):
                self.adata = adata

        adata = sc.read_h5ad(filename)
        emb_wrapper = EmbeddingWrapper(adata)
        self.__init__(emb_wrapper, self.analysis_name)


