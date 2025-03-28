"""
Clustering module for EUCLID.
This module performs:
  - Conventional Leiden clustering on harmonized NMF embeddings.
  - A self-supervised, locally enhanced hierarchical (Euclid) clustering.
  - Assignment of colors to clusters.
  - Application of the learnt clustering tree to new data.
  - Anatomical naming of clusters and generation of cluster inspection PDFs.
  
All functions work on an AnnData object produced by the embedding module.
"""

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
import backSPIN
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
from PyPDF2 import PdfMerger

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
        self.feature_importances = None  # feature importances at the split
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
    adata : sc.AnnData
        AnnData object produced by the embedding module.
    coordinates : pd.DataFrame
        Spatial coordinates for each observation. Columns should include
        either ['Section','xccf','yccf','zccf'] or ['Section','z','y','x'].
    reconstructed_data_df : pd.DataFrame
        Approximated dataset (e.g. from harmonized NMF reconstruction).
    standardized_embeddings_GLOBAL : pd.DataFrame
        Global standardized embeddings (learned from the reference dataset).
    metadata : pd.DataFrame, optional
        Additional metadata (e.g., used for merging spatial metadata).
    """
    def __init__(self, adata, coordinates, reconstructed_data_df, standardized_embeddings_GLOBAL, metadata=None):
        self.adata = adata
        self.coordinates = coordinates
        self.reconstructed_data_df = reconstructed_data_df
        self.standardized_embeddings_GLOBAL = standardized_embeddings_GLOBAL
        self.metadata = metadata

    # -------------------------------------------------------------------------
    # BLOCK 1: Conventional Leiden clustering on harmonized NMF embeddings
    # -------------------------------------------------------------------------
    def leiden_nmf(self, use_reference_only=True, resolution=1.0, key_added="leiden_nmf"):
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
        # Here we assume that the harmonized NMF embeddings are stored in adata.obsm['X_NMF_harmonized']
        if "X_NMF_harmonized" not in self.adata.obsm:
            raise ValueError("Harmonized NMF embeddings not found in adata.obsm['X_NMF_harmonized']")
        sc.pp.neighbors(self.adata, use_rep="X_NMF_harmonized")
        sc.tl.leiden(self.adata, resolution=resolution, key_added=key_added)
        return self.adata

    # -------------------------------------------------------------------------
    # Utility functions (internal)
    # -------------------------------------------------------------------------
    def _compute_seeded_NMF(self, data):
        """
        Private method to compute seeded NMF (as in embedding) on the given data.
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame of pixels x lipids.
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
        gamma_values = np.linspace(0.8, 1.5, num=100)
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
        data_offset = data - np.min(data) + 1e-7
        data_offset = np.ascontiguousarray(data_offset)
        W_init = np.ascontiguousarray(W_init)
        H_init = np.ascontiguousarray(H_init)
        W = nmf.fit_transform(data_offset, W=W_init, H=H_init)
        nmf_result = nmf.transform(data_offset)
        nmfdf = pd.DataFrame(nmf_result, index=data.index)
        factor_to_lipid = nmf.components_
        return nmfdf, factor_to_lipid, N_factors, nmf

    def _continuity_check(self, spat, vcnorm):
        """
        Check whether clusters are continuous along the AP axis.
        Parameters
        ----------
        spat : np.array or DataFrame with columns ['zccf','yccf','Section']
        vcnorm : pd.Series
            Normalization factors per Section.
        Returns
        -------
        Tuple of continuity flags and peak info.
        """
        # (Using code from your snippet)
        dd2 = pd.DataFrame(spat, columns=['zccf','yccf','Section'])
        # Convert section to int
        dd2['Section'] = dd2['Section'].astype(int)
        enough_sectionss = []
        number_of_peakss = []
        peak_ratios = []
        # For each unique cluster color we expect two values. In this helper, we return the values for first two clusters.
        # (In practice you might want to loop over clusters.)
        for cluster in [0, 1]:
            test = dd2  # assuming test is subset for a given cluster
            value_counts = test['Section'].value_counts().sort_index()
            ap = value_counts.values.copy()
            ap[ap < 10] = 0
            ap_nonnull = np.sum(ap > 0) > 3
            apflag = any(ap[i] != 0 and ap[i+1] != 0 for i in range(len(ap)-1))
            enough_sections = ap_nonnull and apflag
            ap_norm = value_counts / vcnorm.loc[value_counts.index].values
            ap_norm = np.array(ap_norm)
            zero_padded_ap = np.pad(ap_norm, pad_width=1, mode='constant', constant_values=0)
            smoothed_ap = gaussian_filter1d(zero_padded_ap, sigma=1.8)
            peaks, properties = find_peaks(smoothed_ap, height=0)
            number_of_peaks = len(peaks)
            if number_of_peaks > 1:
                peak_heights = properties['peak_heights']
                top_peaks = np.sort(peak_heights)[-2:]
                peak_ratio = top_peaks[1] / top_peaks[0]
            else:
                peak_ratio = 10
            enough_sectionss.append(enough_sections)
            number_of_peakss.append(number_of_peaks)
            peak_ratios.append(peak_ratio)
        return enough_sectionss[0], enough_sectionss[1], number_of_peakss[0], number_of_peakss[1], peak_ratios[0], peak_ratios[1]

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
                                ds_factor=1):
        """
        Learn a hierarchical bipartite clustering tree on the dataset.
        This is a self-supervised clustering method that recursively splits the dataset,
        computes local NMFs, reaggregates clusters (e.g., via backSPIN), and trains an XGBoost
        classifier at each split.
        
        Parameters
        ----------
        K : int
            Large K for initial kMeans split.
        min_voxels : int
            Minimum number of observations per branch.
        min_diff_lipids : int
            Minimum number of differential lipids to accept a split.
        min_fc : float
            Minimal fold change for differential lipids.
        pthr : float
            p-value threshold after correction.
        thr_signal : float
            Threshold on signal to keep an NMF component.
        penalty1 : float
            Penalty factor for previous split embeddings.
        penalty2 : float
            Penalty factor for global embeddings.
        ACCTHR : float
            Lower bound for test accuracy to accept the classifier.
        max_depth : int
            Maximum recursive depth.
        ds_factor : int
            Downsampling factor.
        
        Returns
        -------
        root_node : Node
            The root of the hierarchical clustering tree.
        clusteringLOG : pd.DataFrame
            A DataFrame recording the split history.
        """
        # For brevity, we assume that the following variables are available from embedding:
        # - data: the reconstructed dataset (self.reconstructed_data_df)
        # - standardized_embeddings_GLOBAL: self.standardized_embeddings_GLOBAL
        # - coordinates: self.coordinates
        # We also assume that a global variable “trainpoints”, “valpoints”, “testpoints” are defined based on coordinates.
        # In a real implementation, these would be passed or computed.
        data = self.reconstructed_data_df.copy()
        # Create a copy of the raw data (for differential testing)
        rawlips = data.copy()
        # Normalize raw data using percentiles (2% and 98%)
        p2 = rawlips.quantile(0.02)
        p98 = rawlips.quantile(0.98)
        normalized_values = (rawlips.values - p2.values) / (p98.values - p2.values)
        clipped_values = np.clip(normalized_values, 0, 1)
        normalized_datemp = pd.DataFrame(clipped_values, columns=rawlips.columns, index=rawlips.index)
        # Prepare a Scanpy object with raw lipids for differential testing.
        adata = sc.AnnData(X=data)
        # Here we assume self.coordinates has columns ['zccf','yccf','Section']
        adata.obsm['spatial'] = self.coordinates[['zccf','yccf','Section']].loc[data.index].values
        adata.obsm['lipids'] = normalized_datemp
        
        # Get global standardized embeddings (assumed computed previously)
        # WARNINGLINE global self_standardized = self.standardized_embeddings_GLOBAL.copy()
        
        # Initialize a log DataFrame for clustering history.
        column_names = [f"level_{i}" for i in range(1, max_depth+1)]
        clusteringLOG = pd.DataFrame(0, index=data.index, columns=column_names)[::ds_factor]
        
        # Define the recursive splitting function.
        def _dosplit(current_adata, embds, path=[], splitlevel=0):
            print("Splitting level:", splitlevel)
            if current_adata.X.shape[0] < min_voxels:
                print("Branch exhausted due to low voxel count.")
                return None
            # Compute a local NMF on current data
            nmfdf, loadings, N_factors, nmf_model = self._compute_seeded_NMF(pd.DataFrame(current_adata.X, index=current_adata.obs_names))
            nmf_result = nmfdf.values
            filter1 = np.abs(nmf_result).mean(axis=0) > thr_signal
            loadings_sel = loadings[filter1, :]
            nmf_result = nmf_result[:, filter1]
            original_nmf_indices = np.arange(N_factors)[filter1]
            tempadata = sc.AnnData(X=nmf_result)
            tempadata.obsm['spatial'] = current_adata.obsm['spatial']
            goodpcs = self._rank_features_by_combined_score(tempadata)
            goodpcs_indices = original_nmf_indices[goodpcs.astype(int)]
            top_pcs_data = nmf_result[:, goodpcs.astype(int)]
            loadings_sel = loadings_sel[goodpcs.astype(int), :]
            multiplets = self._generate_combinations(len(goodpcs))
            flag = False
            aaa = 0
            while (not flag) and (aaa < len(multiplets)):
                bestpcs = multiplets[aaa]
                embeddings_local = top_pcs_data[:, bestpcs]
                loadings_current = loadings_sel[list(bestpcs), :]
                selected_nmf_indices = goodpcs_indices[list(bestpcs)]
                scaler_local = StandardScaler()
                standardized_embeddings = scaler_local.fit_transform(embeddings_local)
                # Combine with previous split and global embeddings
                globembds = self.standardized_embeddings_GLOBAL.loc[current_adata.obs_names].values / penalty2
                embspace = np.concatenate((standardized_embeddings, embds/penalty1, globembds), axis=1)
                kmeans = KMeans(n_clusters=K, random_state=230598)
                kmeans_labels = kmeans.fit_predict(embspace)
                # Reaggregate via backSPIN (using its API)
                data_for_clustering = pd.DataFrame(current_adata.X, index=current_adata.obs_names, columns=current_adata.var_names)
                data_for_clustering['label'] = kmeans_labels
                centroids = data_for_clustering.groupby('label').mean()
                centroids = pd.DataFrame(StandardScaler().fit_transform(centroids), columns=centroids.columns, index=centroids.index).T
                row_ix, columns_ix = backSPIN.SPIN(centroids, widlist=4)
                centroids = centroids.iloc[row_ix, columns_ix]
                _, _, _, gr1, gr2, _, _, _, _ = backSPIN._divide_to_2and_resort(sorted_data=centroids.values, wid=5)
                gr1 = np.array(centroids.columns)[gr1]
                gr2 = np.array(centroids.columns)[gr2]
                data_for_clustering['lab'] = 1
                data_for_clustering.loc[data_for_clustering['label'].isin(gr2), 'lab'] = 2
                # Check continuity along AP axis using self.coordinates and differential lipids in adata.obsm['lipids']
                # For simplicity, assume we have a normalization factor vcnorm from self.coordinates.
                vcnorm = self.coordinates.loc[current_adata.obs_names, 'Section'].value_counts().sort_index()
                enough_sections0, enough_sections1, num_peaks0, num_peaks1, peak_ratio0, peak_ratio1 = self._continuity_check(current_adata.obsm['spatial'], vcnorm)
                alteredlips, promoted = self._differential_lipids(current_adata.obsm['lipids'].values, kmeans_labels, min_fc, pthr)
                flag = ((np.sum(kmeans_labels == 1) > min_voxels or np.sum(kmeans_labels == 0) > min_voxels)
                        and (alteredlips > min_diff_lipids)
                        and enough_sections0 and enough_sections1
                        and ((num_peaks0 < 3) or (peak_ratio0 > 1.4))
                        and ((num_peaks1 < 3) or (peak_ratio1 > 1.4)))
                aaa += 1
                kmeans_labels = data_for_clustering['lab'].astype(int)
            if not flag:
                print("Branch exhausted due to failure of continuity or differential criteria.")
                return None
            # Train an XGB classifier on the embeddings
            embeddings_df = pd.DataFrame(embspace, index=current_adata.obs_names)
            # Here we assume train/val/test splits are defined from self.coordinates (dummy example below)
            all_idx = embeddings_df.index
            train_idx = all_idx[:int(0.6*len(all_idx))]
            val_idx = all_idx[int(0.6*len(all_idx)):int(0.8*len(all_idx))]
            test_idx = all_idx[int(0.8*len(all_idx)):]
            y_train = kmeans_labels[train_idx]
            y_val = kmeans_labels[val_idx]
            y_test = kmeans_labels[test_idx]
            X_train = embeddings_df.loc[train_idx]
            X_val = embeddings_df.loc[val_idx]
            X_test = embeddings_df.loc[test_idx]
            X_train_sub, y_train_sub = self._undersample(X_train, y_train)
            xgb_model = XGBClassifier(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.5,
                random_state=42,
                n_jobs=6
            )
            xgb_model.fit(X_train_sub, y_train_sub,
                          eval_set=[(X_val, y_val)],
                          early_stopping_rounds=7,
                          verbose=False)
            test_pred = xgb_model.predict(X_test)
            test_acc = accuracy_score(y_test, test_pred)
            print(f"Test accuracy: {test_acc}")
            if test_acc < ACCTHR:
                print("Branch exhausted due to poor classifier generalization.")
                return None
            # Overwrite cluster labels with classifier predictions (for consistency)
            new_labels = pd.concat([pd.Series(xgb_model.predict(X_train), index=X_train.index),
                                     pd.Series(xgb_model.predict(X_val), index=X_val.index),
                                     pd.Series(xgb_model.predict(X_test), index=X_test.index)])
            new_labels = new_labels.loc[embeddings_df.index]
            new_labels = new_labels + 1  # adjust if needed
            # Update clustering log
            clusteringLOG.loc[new_labels.index, f"level_{splitlevel+1}"] = new_labels.values
            # Create a Node for this split
            node = Node(splitlevel, path=path)
            node.scaler = scaler_local
            node.nmf = nmf_model
            node.xgb_model = xgb_model
            node.feature_importances = xgb_model.feature_importances_
            node.factors_to_use = selected_nmf_indices
            # Recursively split the two branches
            idx0 = embeddings_df.index[new_labels == 1]
            idx1 = embeddings_df.index[new_labels == 2]
            adata0 = current_adata[current_adata.obs_names.isin(idx0)]
            adata1 = current_adata[current_adata.obs_names.isin(idx1)]
            embd0 = embeddings_df.loc[idx0].values
            embd1 = embeddings_df.loc[idx1].values
            child0 = _dosplit(adata0, embd0, path + [0], splitlevel + 1)
            child1 = _dosplit(adata1, embd1, path + [1], splitlevel + 1)
            node.children[0] = child0
            node.children[1] = child1
            return node
        
        root_node = _dosplit(adata[::ds_factor], self.standardized_embeddings_GLOBAL[::ds_factor].values, path=[], splitlevel=0)
        # Save the clustering log and tree to file.
        clusteringLOG.to_hdf("tree_clustering_down_clean.h5ad", key="table")
        with open("rootnode_clustering_whole_clean.pkl", "wb") as f:
            pickle.dump(root_node, f)
        return root_node, clusteringLOG

    # -------------------------------------------------------------------------
    # BLOCK 3: Assign colors to clusters based on spatial embedding (for visualization)
    # -------------------------------------------------------------------------
    def assign_cluster_colors(self, tree, coordinates, pdf_output="colorzones.h5ad"):
        """
        Compute a color assignment for lipizones (clusters) based on spatial distribution.
        
        Parameters
        ----------
        tree : pd.DataFrame
            A DataFrame that contains the clustering hierarchy (e.g. split history).
        coordinates : pd.DataFrame
            Spatial coordinates with columns including 'zccf','yccf','Section'.
        pdf_output : str, optional
            File path to save the color zones.
        
        Returns
        -------
        lipizone_colors : pd.Series
            A Series mapping each observation to a hex color.
        """
        # In our refactored version, we assume that “tree” is available (e.g. from the clustering history)
        # We use the plotting utilities from the snippet.
        # (For brevity, much of the detailed plotting code is preserved.)
        # Here we simulate the existence of a “contour” array, e.g. eroded annotation.
        conto = np.load("eroded_annot.npy")
        coords = coordinates.fillna(0).replace([np.inf, -np.inf], 0)
        xs = (coords['xccf']*40).astype(int)
        ys = (coords['yccf']*40).astype(int)
        zs = (coords['zccf']*40).astype(int)
        xs.loc[xs>527] = 527
        ys.loc[ys>319] = 319
        zs.loc[zs>455] = 455
        coords['border'] = conto[xs, ys, zs]
        # Normalize the data per pixel
        # Here we assume that self.adata.X exists and is comparable
        # Combine tree with coordinates
        levels = pd.concat([self.adata.obs, coordinates.loc[self.adata.obs_names]], axis=1)
        levels = pd.concat([levels, tree], axis=1)
        dd2 = levels.copy()
        # For each unique division in a column "class" (built from earlier splits)
        divisions = dd2['class'].unique()
        colormaps = ["RdYlBu", "terrain", "PiYG", "cividis", "plasma", "PuRd", "inferno", "PuOr"]
        dd2['R'] = np.nan
        dd2['G'] = np.nan
        dd2['B'] = np.nan
        for division, cmap_name in zip(divisions, colormaps):
            if len(dd2.loc[dd2['class'] == division, 'cluster'].unique()) > 1:
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
                dd2.update(pd.merge(dd2, lipocolor_reset, on='cluster', how='left')[['R','G','B']])
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
        dd2['lipizone_color'].fillna('gray', inplace=True)
        # Save color assignment
        dd2['lipizone_color'].to_hdf(pdf_output, key="table")
        return dd2['lipizone_color']

    # -------------------------------------------------------------------------
    # BLOCK 4: Apply the learnt Euclid clustering tree to the whole dataset.
    # -------------------------------------------------------------------------
    def apply_euclid_clustering(self, tree_file="rootnode_clustering_whole_clean.pkl", ds_factor=1):
        """
        Apply the learnt Euclid clustering tree to new (or full) dataset.
        
        Returns
        -------
        df_paths : pd.DataFrame
            DataFrame with hierarchical cluster labels.
        """
        with open(tree_file, "rb") as f:
            root_node = pickle.load(f)
        # Define a recursive tree traversal.
        def _traverse_tree(node, current_adata, embds, paths, level=0):
            print("Traverse level:", level)
            if node is None or not node.children:
                return
            if current_adata.shape[0] == 0:
                return
            nmf = node.nmf
            X_nmf = nmf.transform(current_adata.X)
            factors = node.factors_to_use
            X_nmf = X_nmf[:, factors]
            scaler_local = node.scaler
            X_scaled = scaler_local.transform(X_nmf)
            globembds = self.standardized_embeddings_GLOBAL.loc[current_adata.obs_names].values / penalty2
            embspace = np.concatenate((X_scaled, embds/penalty1, globembds), axis=1)
            child_labels = node.xgb_model.predict(embspace)
            for i, idx in enumerate(current_adata.obs_names):
                if idx not in paths:
                    paths[idx] = []
                paths[idx].append(child_labels[i])
            idx0 = current_adata.obs_names[child_labels == 0]
            idx1 = current_adata.obs_names[child_labels == 1]
            adata0 = current_adata[current_adata.obs_names.isin(idx0)]
            adata1 = current_adata[current_adata.obs_names.isin(idx1)]
            embd0 = X_scaled[child_labels == 0]
            embd1 = X_scaled[child_labels == 1]
            _traverse_tree(node.children.get(0), adata0, embd0, paths, level+1)
            _traverse_tree(node.children.get(1), adata1, embd1, paths, level+1)
        paths = {}
        embds = self.standardized_embeddings_GLOBAL[::ds_factor].values
        new_adata = sc.AnnData(X=self.reconstructed_data_df)
        new_adata.obsm['spatial'] = self.metadata[['zccf','yccf','Section']].loc[self.reconstructed_data_df.index].values
        _traverse_tree(root_node, new_adata[::ds_factor], embds, paths)
        df_paths = pd.DataFrame.from_dict(paths, orient='index')
        df_paths.columns = [f'level_{i}' for i in range(1, df_paths.shape[1]+1)]
        df_paths = df_paths.fillna(-1).astype(int) + 1
        df_paths.to_hdf("splithistory_allbrains.h5ad", key="table")
        return df_paths

    # -------------------------------------------------------------------------
    # BLOCK 5: Assign a name to each cluster based on anatomical localization.
    # -------------------------------------------------------------------------
    def name_lipizones_anatomy(self, acronyms, lipizones):
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
        sns.heatmap(normalized_df, cmap="Purples", cbar_kws={'label': 'Enrichment'},
                    xticklabels=True, yticklabels=False,
                    vmin=np.percentile(normalized_df, 2), vmax=np.percentile(normalized_df, 98))
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False)
        plt.tight_layout()
        plt.savefig("purpleheatmap_acronymlipizones.pdf")
        plt.close()
        mapping = normalized_df.T
        new_index = [f"{row[:4]}" for row in mapping.index]  # placeholder transformation
        mat = pd.DataFrame({"oldname": mapping.index, "acronym": new_index})
        return mat

    # -------------------------------------------------------------------------
    # BLOCK 6: Plot each cluster to separate PDF files for inspection.
    # -------------------------------------------------------------------------
    def clusters_to_pdf(self, lipizone_names, output_folder="lipizones_output", pdf_filename="clusters_combined.pdf"):
        """
        Plot each cluster (lipizone) separately into PDF files.
        
        Parameters
        ----------
        lipizone_names : pd.Series
            A Series of cluster labels (or names) per pixel.
        output_folder : str, optional
            Folder in which to save individual PDFs.
        pdf_filename : str, optional
            Final merged PDF filename.
        """
        os.makedirs(output_folder, exist_ok=True)
        # Combine coordinates with lipizone names.
        levels = pd.concat([self.coordinates.loc[self.reconstructed_data_df.index],
                            self.adata.obs], axis=1)
        levels['lipizone_names'] = lipizone_names
        levels['Section'] = self.coordinates['Section']
        levels['zccf'] = self.coordinates['zccf']
        levels['yccf'] = self.coordinates['yccf']
        levels['xccf'] = self.coordinates['xccf']
        dot_size = 0.3
        sections_to_plot = range(1, 33)
        global_min_z = levels['xccf'].min()
        global_max_z = levels['xccf'].max()
        global_min_y = -levels['yccf'].max()
        global_max_y = -levels['yccf'].min()
        unique_names = np.sort(levels['lipizone_names'].unique())
        # Plot each cluster to its own PDF
        for uniq in unique_names:
            fig, axes = plt.subplots(4, 8, figsize=(40, 20))
            axes = axes.flatten()
            for i, sec in enumerate(sections_to_plot):
                ax = axes[i]
                subset = levels[levels["Section"] == sec]
                ax.scatter(subset['xccf'], -subset['yccf'], c=subset['lipizone_names'].astype("category").cat.codes,
                           cmap='Greys', s=dot_size*2, alpha=0.2, rasterized=True)
                highlight = subset[subset['lipizone_names'] == uniq]
                ax.scatter(highlight['xccf'], -highlight['yccf'], c='red', s=dot_size, alpha=1, rasterized=True)
                ax.axis('off')
                ax.set_aspect('equal')
                ax.set_xlim(global_min_z, global_max_z)
                ax.set_ylim(global_min_y, global_max_y)
            for j in range(i+1, len(axes)):
                fig.delaxes(axes[j])
            plt.suptitle(uniq)
            plt.tight_layout()
            outpath = os.path.join(output_folder, f"{uniq}.pdf")
            plt.savefig(outpath)
            plt.close(fig)
        # Merge all PDFs into one
        cwd = os.getcwd()
        merger = PdfMerger()
        for fname in sorted(os.listdir(output_folder)):
            if fname.endswith(".pdf"):
                merger.append(os.path.join(output_folder, fname))
        final_pdf = os.path.join(cwd, pdf_filename)
        merger.write(final_pdf)
        merger.close()
        print(f"Merged PDF saved as {final_pdf}")
