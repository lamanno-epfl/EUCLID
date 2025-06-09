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
from matplotlib.backends.backend_pdf import PdfPages

# Configure thread limits
threadpool_limits(limits=8)
os.environ['OMP_NUM_THREADS'] = '6'


class Embedding():
    
    def __init__(self, prep, analysis_name="analysis"):
        self.analysis_name = analysis_name
        if prep is not None:
            self.adata = prep.adata
            self.data_df = pd.DataFrame(prep.adata.X, 
                                      index=prep.adata.obs_names, 
                                      columns=prep.adata.var_names).fillna(0.0001)
            

    def learn_seeded_nmf_embeddings(
        self,
        resolution_range: tuple = (0.8, 1.5),
        num_gamma: int = 100,
        alpha: float = 0.7,
        random_state: int = 42,
    ):
        """
        Compute seeded NMF embeddings on a reference set.
        
        Parameters
        ----------
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
        Updates itw own state with the NMF embeddings and the factor_to_lipid matrix
        """
        # 1. Compute correlation matrix between lipids (features)
        corr = np.corrcoef(self.data_df.values.T)
        corr_matrix = np.abs(corr)
        np.fill_diagonal(corr_matrix, 0)

        # Create a dummy AnnData to store connectivity for Leiden clustering
        temp_adata = anndata.AnnData(X=np.zeros_like(corr_matrix))
        temp_adata.obsp['connectivities'] = csr_matrix(corr_matrix)
        temp_adata.uns['neighbors'] = {
            'connectivities_key': 'connectivities',
            'distances_key': 'distances',
            'params': {'n_neighbors': 10, 'method': 'custom'}
        }

        G = nx.from_numpy_array(corr_matrix)
        gamma_values = np.linspace(resolution_range[0], resolution_range[1], num=num_gamma)
        num_communities = []
        modularity_scores = []
        objective_values = []

        for gamma in gamma_values:
            sc.tl.leiden(temp_adata, resolution=gamma, key_added=f'leiden_{gamma}')
            clusters = temp_adata.obs[f'leiden_{gamma}'].astype(int).values
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
        #plt.figure()
        #plt.plot(np.arange(len(objective_values)), objective_values)
        #plt.title("Objective function vs Gamma index")
        #plt.show()

        max_index = np.argmax(objective_values)
        best_gamma = gamma_values[max_index]
        best_num_comms = num_communities[max_index]
        print(f'Best gamma: {best_gamma}, Number of communities: {best_num_comms}')

        # Run Leiden with best gamma
        sc.tl.leiden(temp_adata, resolution=best_gamma, key_added='leiden_best')
        clusters = temp_adata.obs['leiden_best'].astype(int).values
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

        W_init = self.data_df.values[:, representatives]

        # 5. Initialize H from the correlation matrix
        H_init = corr[representatives, :]
        H_init[H_init < 0] = 0

        # 6. Compute NMF with custom initialization
        nmf = NMF(n_components=W_init.shape[1], init='custom', random_state=random_state)
        data_offset = self.data_df - np.min(self.data_df) + 1e-7
        data_offset = np.ascontiguousarray(data_offset)
        W_init = np.ascontiguousarray(W_init)
        H_init = np.ascontiguousarray(H_init)
        W = nmf.fit_transform(data_offset, W=W_init, H=H_init)
        self.nmf_embeddings = pd.DataFrame(W, index=self.data_df.index)
        self.factor_to_lipid = nmf.components_
        self.N_factors = N_factors
        self.nmf = nmf

    def apply_nmf_embeddings(
        self,
        new_adata = None
    ):
        """
        Parameters
        ----------
        new_adata : sc.AnnData, optional
            New AnnData object on which to apply the NMF model. If None, uses self.adata.
            Should have the same var_names (features) as the training data.

        Returns
        -------
        embeddings : pd.DataFrame
            The NMF embeddings for the new data.
        """
        
        if new_adata is not None:
            # Verify that the new data has the same features
            if not np.array_equal(new_adata.var_names, self.adata.var_names):
                raise ValueError("New data must have the same features (var_names) as the training data")
            data_to_transform = pd.DataFrame(new_adata.X, 
                                          index=new_adata.obs_names, 
                                          columns=new_adata.var_names)
            target_adata = new_adata
        else:
            data_to_transform = pd.DataFrame(self.adata.X, 
                                          index=self.adata.obs_names, 
                                          columns=self.adata.var_names)
            target_adata = self.adata
            
        data_offset = data_to_transform - np.min(data_to_transform) + 1e-7
        data_offset = np.ascontiguousarray(data_offset)
        nmf_all = self.nmf.transform(data_offset)
        embeddings = pd.DataFrame(nmf_all, index=data_to_transform.index)
        target_adata.obsm['X_NMF'] = embeddings.values
        
        return target_adata

    def harmonize_nmf_batches(
        self,
        covariates: list = None,
    ):
        """
        Correct residual batch effects on the NMF embeddings using Harmony.
        
        Parameters
        ----------
        covariates : list, optional
            List of covariates to use for harmonization.
        
        Returns
        -------
            sets in the adata an X_Harmonized slot
        """
        nmf_embeddings = pd.DataFrame(self.adata.obsm['X_NMF'], index=self.adata.obs_names)
        batches = self.adata.obs[covariates].astype("category")
        batchessub = batches.copy()
        unique_values = sorted(batchessub['SectionID'].unique())
        value_mapping = {old_value: new_index for new_index, old_value in enumerate(unique_values)}
        batchessub['SectionID'] = batchessub['SectionID'].map(value_mapping)
        batchessub['SectionID'] = batchessub['SectionID'].astype("category")
        vars_use=list(batchessub.columns)
        
        ho = hm.run_harmony(nmf_embeddings, batchessub, vars_use, max_iter_harmony=20)
        self.adata.obsm['X_Harmonized'] = ho.Z_corr.T

    def approximate_dataset_harmonmf(
        self
    ):
        """
        Reconstruct an approximation of the original dataset from the harmonized NMF.
        
        Returns
        -------
            sets in the adata an X_approximated slot
        """
        recon = np.dot(self.adata.obsm['X_Harmonized'], self.factor_to_lipid)
        self.adata.obsm['X_approximated'] = recon - np.min(recon) + 1e-7

    def tsne(
        self,
        perplexity: int = 30,
        n_iter1: int = 500,
        exaggeration1: float = 1.2,
        n_iter2: int = 100,
        exaggeration2: float = 2.5,
        init_indices: tuple = (0, 1)
    ):
        """
        Compute a tSNE visualization of the (corrected) NMF embeddings.
        
        Parameters
        ----------
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

        try:
            embeddings = self.adata.obsm["X_Harmonized"]
        except:
            embeddings = self.adata.obsm["X_NMF"]
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
        init_train = x_train[:, list(init_indices)]
        tsne_emb = TSNEEmbedding(
            init_train,
            affinities_train,
            negative_gradient_method="fft",
            n_jobs=8,
            verbose=True,
        )
        tsne_emb_1 = tsne_emb.optimize(n_iter=n_iter1, exaggeration=exaggeration1)
        self.adata.obsm['X_TSNE'] = np.array(tsne_emb_1.optimize(n_iter=n_iter2, exaggeration=exaggeration2))
        
    def save_msi_dataset(
        self,
        filename=None,
        pdf_filename=None,
        plot_embeddings=False
    ):
        """
        Save the current AnnData object to disk and optionally plot spatial embeddings.

        Parameters
        ----------
        filename : str, optional
            File path to save the AnnData object. If None, uses analysis_name prefix.
        pdf_filename : str, optional
            File path to save the spatial embeddings PDF. If None, uses analysis_name prefix.
            Only used if plot_embeddings is True.
        plot_embeddings : bool, optional
            Whether to generate and save spatial embedding plots. Default is False.
        """
        
        # Set default filenames with analysis_name prefix
        if filename is None:
            filename = f"{self.analysis_name}_emb_msi_dataset.h5ad"
        if pdf_filename is None:
            pdf_filename = f"{self.analysis_name}_embeddings_spatial.pdf"
        
        # Save AnnData object
        for k, v in list(self.adata.obsm.items()):
            if isinstance(v, pd.DataFrame):
                self.adata.obsm[k] = v.values
        
        self.adata.write_h5ad(filename)
        
        # Plot spatial embeddings if requested
        if plot_embeddings:
            # Get embeddings to plot (prefer harmonized if available)
            try:
                embeddings = self.adata.obsm["X_Harmonized"]
            except:
                embeddings = self.adata.obsm["X_NMF"]
            
            # Convert embeddings to DataFrame if not already
            if not isinstance(embeddings, pd.DataFrame):
                embeddings_df = pd.DataFrame(embeddings, index=self.adata.obs_names)
            else:
                embeddings_df = embeddings
            
            # Get spatial coordinates
            coords = self.adata.obs[['zccf', 'yccf', 'Section']].copy()
            
            # Get unique samples and their sections
            samples = self.adata.obs['Sample'].unique()
            max_sections = max(len(self.adata.obs[self.adata.obs['Sample'] == sample]['SectionID'].unique()) 
                             for sample in samples)
            
            # Create PDF
            with PdfPages(pdf_filename) as pdf:
                # For each embedding direction
                for emb_idx in tqdm(range(embeddings_df.shape[1]), desc="Plotting embeddings"):
                    fig, axes = plt.subplots(len(samples), max_sections, 
                                          figsize=(max_sections*2, len(samples)*2))
                    
                    # For each sample
                    for sample_idx, sample in enumerate(samples):
                        sample_data = self.adata[self.adata.obs['Sample'] == sample]
                        sample_coords = coords.loc[sample_data.obs_names]
                        sample_emb = embeddings_df.loc[sample_data.obs_names, emb_idx]
                        
                        # Get sections for this sample
                        sections = sorted(sample_data.obs['SectionID'].unique())
                        
                        # For each section
                        for section_idx, section in enumerate(sections):
                            ax = axes[sample_idx, section_idx]
                            
                            # Get data for this section
                            section_mask = sample_data.obs['SectionID'] == section
                            section_coords = sample_coords[section_mask]
                            section_emb = sample_emb[section_mask]
                            
                            # Plot
                            scatter = ax.scatter(section_coords['zccf'], -section_coords['yccf'],
                                              c=section_emb, cmap='PuOr', s=0.5, rasterized=True)
                            
                            # Remove spines and ticks
                            ax.set_axis_off()
                            ax.set_aspect('equal')
                            
                            # Set title for first row
                            if sample_idx == 0:
                                ax.set_title(f'Section {section}')
                            
                            # Set ylabel for first column
                            if section_idx == 0:
                                ax.set_ylabel(sample)
                    
                    # Remove empty subplots
                    for i in range(len(samples)):
                        for j in range(len(sections), max_sections):
                            fig.delaxes(axes[i, j])
                    
                    # Add colorbar
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                    fig.colorbar(scatter, cax=cbar_ax)
                    
                    # Add title
                    plt.suptitle(f'Embedding {emb_idx + 1}', y=0.95)
                    
                    # Adjust layout and save
                    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
                    pdf.savefig(fig)
                    plt.close(fig)

    def load_msi_dataset( # THIS SERVES AS AN ALTERNATIVE INIT
        self,
        filename=None
    ) -> sc.AnnData:
        """
        Load an AnnData object from disk.

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
            filename = f"{self.analysis_name}_emb_msi_dataset.h5ad"
        adata = sc.read_h5ad(filename)
        self.adata = adata
