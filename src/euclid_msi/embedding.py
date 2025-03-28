import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
from openTSNE import TSNEEmbedding, affinity
from tqdm import tqdm
from collections import deque
import harmonypy as hm
import networkx as nx
from threadpoolctl import threadpool_limits

# Configure thread limits
threadpool_limits(limits=8)
os.environ['OMP_NUM_THREADS'] = '6'


class Embedding:
    """
    A class that encapsulates all embedding steps on an AnnData object produced during preprocessing.
    Each method corresponds to a BLOCK of functionality.
    """

    def learn_seeded_nmf_embeddings(
        self,
        data: pd.DataFrame,
        resolution_range: tuple = (0.8, 1.5),
        num_gamma: int = 100,
        alpha: float = 0.7,
        random_state: int = 42,
    ):
        """
        BLOCK 1:
        Compute seeded NMF embeddings on a reference set.
        
        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame (pixels x lipids) containing the feature-selected data.
        resolution_range : tuple, optional
            Range of gamma values (Leiden resolution) to explore.
        num_gamma : int, optional
            Number of gamma values to test.
        alpha : float, optional
            Weighting factor for modularity vs. number of communities.
        random_state : int, optional
            Random state for NMF initialization.

        Returns
        -------
        nmf_embeddings : pd.DataFrame
            The W matrix of NMF (pixels x N_factors).
        factor_to_lipid : np.ndarray
            The H matrix from NMF (components x lipids).
        N_factors : int
            The number of factors (i.e. communities) chosen.
        nmf_model : NMF
            The fitted NMF model.
        """
        # 1. Compute correlation matrix between lipids (features)
        corr = np.corrcoef(data.values.T)
        corr_matrix = np.abs(corr)
        np.fill_diagonal(corr_matrix, 0)

        # Create a dummy AnnData to store connectivity
        adata = anndata.AnnData(X=np.zeros_like(corr_matrix))
        adata.obsp['connectivities'] = csr_matrix(corr_matrix)
        adata.uns['neighbors'] = {
            'connectivities_key': 'connectivities',
            'distances_key': 'distances',
            'params': {'n_neighbors': 10, 'method': 'custom'}
        }

        # Build a network from correlation matrix
        G = nx.from_numpy_array(corr_matrix)

        gamma_values = np.linspace(resolution_range[0], resolution_range[1], num=num_gamma)
        num_communities = []
        modularity_scores = []
        objective_values = []

        for gamma in gamma_values:
            sc.tl.leiden(adata, resolution=gamma, key_added=f'leiden_{gamma}')
            clusters = adata.obs[f'leiden_{gamma}'].astype(int).values
            num_comms = len(np.unique(clusters))
            num_communities.append(num_comms)
            # Compute modularity over communities
            partition = [np.where(clusters == i)[0] for i in range(num_comms)]
            modularity = nx.community.modularity(G, partition)
            modularity_scores.append(modularity)

        # Compute objective function and choose best gamma
        epsilon = 1e-10
        for Q, N_c in zip(modularity_scores, num_communities):
            f_gamma = Q**alpha * np.log(N_c + 1 + epsilon)
            objective_values.append(f_gamma)

        # Plot for visual inspection (could also be saved)
        plt.figure()
        plt.plot(np.arange(len(objective_values)), objective_values)
        plt.title("Objective function vs Gamma index")
        plt.show()

        max_index = np.argmax(objective_values)
        best_gamma = gamma_values[max_index]
        best_num_comms = num_communities[max_index]
        print(f'Best gamma: {best_gamma}, Number of communities: {best_num_comms}')

        # Run Leiden with best gamma
        sc.tl.leiden(adata, resolution=best_gamma, key_added='leiden_best')
        clusters = adata.obs['leiden_best'].astype(int).values
        N_factors = best_num_comms

        # 4. Choose a representative lipid per cluster
        dist = 1 - corr_matrix
        np.fill_diagonal(dist, 0)
        dist = np.maximum(dist, dist.T)  # enforce symmetry
        dist_condensed = squareform(dist, checks=True)
        representatives = []
        for i in range(N_factors):
            cluster_members = np.where(clusters == i)[0]
            if len(cluster_members) > 0:
                mean_dist = dist[cluster_members][:, cluster_members].mean(axis=1)
                central_idx = cluster_members[np.argmin(mean_dist)]
                representatives.append(central_idx)

        W_init = data.values[:, representatives]

        # 5. Initialize H from the correlation matrix
        H_init = corr[representatives, :]
        H_init[H_init < 0] = 0

        # 6. Compute NMF with custom initialization
        nmf = NMF(n_components=W_init.shape[1], init='custom', random_state=random_state)
        data_offset = data - np.min(data) + 1e-7
        data_offset = np.ascontiguousarray(data_offset)
        W_init = np.ascontiguousarray(W_init)
        H_init = np.ascontiguousarray(H_init)
        W = nmf.fit_transform(data_offset, W=W_init, H=H_init)
        nmf_embeddings = pd.DataFrame(W, index=data.index)
        factor_to_lipid = nmf.components_
        return nmf_embeddings, factor_to_lipid, N_factors, nmf

    def apply_nmf_embeddings(
        self,
        new_data: pd.DataFrame,
        nmf_model: NMF,
        adata: sc.AnnData = None
    ):
        """
        BLOCK 2:
        Apply the learnt NMF model to new data to derive embeddings.
        
        Parameters
        ----------
        new_data : pd.DataFrame
            New data (pixels x lipids) on which to apply the NMF model.
        nmf_model : NMF
            A fitted NMF model from learn_seeded_nmf_embeddings.
        adata : sc.AnnData, optional
            AnnData object to which the embeddings will be added (in obsm['X_NMF']).

        Returns
        -------
        embeddings : pd.DataFrame
            The NMF embeddings for the new data.
        """
        new_data_offset = new_data - np.min(new_data) + 1e-7
        new_data_offset = np.ascontiguousarray(new_data_offset)
        nmf_all = nmf_model.transform(new_data_offset)
        embeddings = pd.DataFrame(nmf_all, index=new_data.index)
        if adata is not None:
            adata.obsm['X_NMF'] = embeddings.values
        return embeddings

    def harmonize_nmf_batches(
        self,
        nmf_embeddings: pd.DataFrame,
        batch_metadata: pd.DataFrame,
        vars_use: list
    ):
        """
        BLOCK 3:
        Correct residual batch effects on the NMF embeddings using Harmony.
        
        Parameters
        ----------
        nmf_embeddings : pd.DataFrame
            DataFrame of NMF embeddings (pixels x factors).
        batch_metadata : pd.DataFrame
            DataFrame of batch covariates (index must match nmf_embeddings).
        vars_use : list
            List of column names in batch_metadata to use for Harmony.
        
        Returns
        -------
        corrected_embeddings : pd.DataFrame
            The Harmony-corrected NMF embeddings.
        """
        ho = hm.run_harmony(nmf_embeddings, batch_metadata, vars_use)
        corrected = pd.DataFrame(ho.Z_corr.T, index=nmf_embeddings.index, columns=nmf_embeddings.columns)
        return corrected

    def approximate_dataset_harmonmf(
        self,
        corrected_embeddings: pd.DataFrame,
        factor_to_lipid: np.ndarray,
        original_feature_names: list
    ):
        """
        BLOCK 4:
        Reconstruct an approximation of the original dataset from the harmonized NMF.
        
        Parameters
        ----------
        corrected_embeddings : pd.DataFrame
            Corrected NMF embeddings (pixels x factors).
        factor_to_lipid : np.ndarray
            The H matrix from NMF (factors x lipids).
        original_feature_names : list
            List of original feature names (lipids) corresponding to columns.
        
        Returns
        -------
        reconstructed_df : pd.DataFrame
            The reconstructed data approximating the original dataset.
        """
        recon = np.dot(corrected_embeddings.values, factor_to_lipid)
        reconstructed_df = pd.DataFrame(recon, index=corrected_embeddings.index, columns=original_feature_names)
        reconstructed_df = reconstructed_df - np.min(reconstructed_df) + 1e-7
        return reconstructed_df

    def tsne(
        self,
        embeddings: pd.DataFrame,
        perplexity: int = 30,
        n_iter1: int = 500,
        exaggeration1: float = 1.2,
        n_iter2: int = 100,
        exaggeration2: float = 2.5,
        init_indices: tuple = (0, 1)
    ):
        """
        BLOCK 5:
        Compute a tSNE visualization of the (corrected) NMF embeddings.
        
        Parameters
        ----------
        embeddings : pd.DataFrame
            The corrected NMF embeddings (pixels x factors).
        perplexity : int, optional
            tSNE perplexity.
        n_iter1 : int, optional
            First stage optimization iterations.
        exaggeration1 : float, optional
            Exaggeration parameter for first stage.
        n_iter2 : int, optional
            Second stage optimization iterations.
        exaggeration2 : float, optional
            Exaggeration parameter for second stage.
        init_indices : tuple, optional
            Indices of two factors to use for initialization.
        
        Returns
        -------
        tsne_coords : pd.DataFrame
            tSNE coordinates (pixels x 2).
        """
        scaler = StandardScaler()
        x_train = scaler.fit_transform(embeddings)
        affinities_train = affinity.PerplexityBasedNN(
            x_train,
            perplexity=perplexity,
            metric="euclidean",
            n_jobs=8,
            random_state=42,
            verbose=True,
        )
        # Use two uncorrelated NMF factors as initialization (default indices or as provided)
        init_train = x_train[:, list(init_indices)]
        tsne_emb = TSNEEmbedding(
            init_train,
            affinities_train,
            negative_gradient_method="fft",
            n_jobs=8,
            verbose=True,
        )
        tsne_emb_1 = tsne_emb.optimize(n_iter=n_iter1, exaggeration=exaggeration1)
        tsne_emb_final = tsne_emb_1.optimize(n_iter=n_iter2, exaggeration=exaggeration2)
        tsne_coords = pd.DataFrame(tsne_emb_final.view(np.ndarray), index=embeddings.index, columns=["TSNE1", "TSNE2"])
        return tsne_coords
