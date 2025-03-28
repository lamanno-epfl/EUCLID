##################################################################################################################
# BLOCK 1: COMPUTE SEEDED NMF EMBEDDINGS ON REFERENCE SECTIONS -> function learn_seeded_nmf_embeddings
##################################################################################################################

# load the adata file created in the preprocessing phase

import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
from openTSNE import TSNEEmbedding, affinity, initialization
from tqdm import tqdm
from collections import deque
import harmonypy as hm
import networkx as nx
from threadpoolctl import threadpool_limits, threadpool_info

# configure thread limits
threadpool_limits(limits=8)
os.environ['OMP_NUM_THREADS'] = '6'

# extract a dataframe "data" with the feature-selected lipids passing user-defined feature selection and having Moran's I > 0.5

# use NMF to decompose the data into factors

def compute_seeded_NMF(data):  # data is a dataframe pixels x lipids
    # 1. calculate the correlation matrix of this dataset
    corr = np.corrcoef(data.values.T)
    corr_matrix = np.abs(corr)  # anticorrelated lipids convey the same info
    np.fill_diagonal(corr_matrix, 0)
    
    adata = anndata.AnnData(X=np.zeros_like(corr_matrix))
    adata.obsp['connectivities'] = csr_matrix(corr_matrix)
    adata.uns['neighbors'] = {
        'connectivities_key': 'connectivities',
        'distances_key': 'distances',
        'params': {'n_neighbors': 10, 'method': 'custom'}
    }
    
    G = nx.from_numpy_array(corr_matrix)
    
    # span reasonable Leiden resolution parameters
    gamma_values = np.linspace(0.8, 1.5, num=100)
    num_communities = []
    modularity_scores = []
    objective_values = []
    
    for gamma in gamma_values:
        sc.tl.leiden(adata, resolution=gamma, key_added=f'leiden_{gamma}')
        clusters = adata.obs[f'leiden_{gamma}'].astype(int).values
        num_comms = len(np.unique(clusters))
        num_communities.append(num_comms)
        partition = {i: clusters[i] for i in range(len(clusters))}
        modularity = nx.community.modularity(G, [np.where(clusters == i)[0] for i in range(num_comms)])
        modularity_scores.append(modularity)
    
    # 3. pick a number of blocks that is relatively high while preserving good modularity
    epsilon = 1e-10
    alpha = 0.7  # controls the weight of modularity vs pushing higher the number of communities
    for Q, N_c in zip(modularity_scores, num_communities):
        f_gamma = Q**alpha * np.log(N_c + 1 + epsilon)
        objective_values.append(f_gamma)
        
    plt.plot(np.arange(len(objective_values)), objective_values)
    plt.title("obj")
    plt.show()
    
    plt.plot(np.arange(len(modularity_scores)), modularity_scores)
    plt.title("mod")
    plt.show()
        
    plt.plot(np.arange(len(num_communities)), num_communities)
    plt.title("ncomms")
    plt.show()
    
    max_index = np.argmax(objective_values)
    best_gamma = gamma_values[max_index]
    best_modularity = modularity_scores[max_index]
    best_num_comms = num_communities[max_index]
    print(f'Number of communities at best gamma: {best_num_comms}')
    
    sc.tl.leiden(adata, resolution=best_gamma, key_added='leiden_best') # run Leiden one final time with best parameters
    clusters = adata.obs['leiden_best'].astype(int).values
    print(clusters)
    
    N_factors = best_num_comms
    
    # 4. pick a representative lipid from each block, use to initialize W
    dist = 1 - corr_matrix
    np.fill_diagonal(dist, 0)
    dist = np.maximum(dist, dist.T)  # as numerical instability makes it unreasonably asymmetric
    dist_condensed = squareform(dist, checks=True)
    representatives = []
    
    for i in range(0, N_factors):
        cluster_members = np.where(clusters == i)[0]
        print(cluster_members)
        if len(cluster_members) > 0:  # find most central feature in cluster
            mean_dist = dist[cluster_members][:, cluster_members].mean(axis=1)
            central_idx = cluster_members[np.argmin(mean_dist)]
            representatives.append(central_idx)
    
    W_init = data.values[:, representatives]
    
    # 5. initialize H as a subset of the correlation matrix
    H_init = corr[representatives,:]
    H_init[H_init < 0.] = 0.  # only positive correlated can contribute by def in NMF
    
    # 6. compute the NMF with this initialization and rank N
    N_factors = W_init.shape[1]
    nmf = NMF(
        n_components=N_factors,
        init='custom',
        random_state=42
    )
    data_offset = data - np.min(data) + 1e-7
    
    data_offset = np.ascontiguousarray(data_offset)
    W_init = np.ascontiguousarray(W_init)
    H_init = np.ascontiguousarray(H_init)
    
    nmf_result = nmf.fit_transform(
        data_offset,
        W=W_init,
        H=H_init
    )
    nmfdf = pd.DataFrame(nmf_result, index=data.index)
    factor_to_lipid = nmf.components_
    
    return nmfdf, factor_to_lipid, N_factors, nmf


nmfdf, factor_to_lipid, N_factors, nmf = compute_seeded_NMF(data.fillna(0.0001))

# store the right matrix, something like this... store it also in the adata object
np.save("factor_to_lipid.npy", factor_to_lipid)


##################################################################################################################
# BLOCK 2: APPLY THE LEARNT NMF ACROSS ALL ACQUISITIONS TO DERIVE EMBEDDINGS -> function apply_nmf_embeddings
##################################################################################################################
tmp # dataframe from the adata object with the feature-selected features across all acquisitions
nmf_allbrains = nmf.transform(tmp.fillna(0.0001))
nmfall = pd.DataFrame(nmf_allbrains, index=tmp.index)

# add the nmf embeddings as X_NMF slot in the adata object


##################################################################################################################
# BLOCK 3: CORRECT RESIDUAL BATCH EFFECTS ON THE NMF FOR CLUSTERING -> function harmonize_nmf_batches
##################################################################################################################
# load user's batchessub pandas dataframe, which has for rows the pixels of the adata object (i.e., same index) and as columns the covariates

nmfall_mat = nmfall
meta_nmfall = batchessub
vars_use = list(batchessub.columns)

ho = hm.run_harmony(nmfall_mat, meta_nmfall, vars_use)

corrected_nmfall = pd.DataFrame(
    ho.Z_corr.T,
    index=nmfall_mat.index,
    columns=nmfall_mat.columns
)

# store corrected_nmfall as a new slot in the adata, something like X_NMF_harmonized


##################################################################################################################
# BLOCK 4: APPROXIMATE THE DATASET WITH THE HARMONIZED NMF FOR CLUSTERING PROCEDURES -> function approximate_dataset_harmonmf
##################################################################################################################
# here we cluster the atlas with a bipartite, hierarchical splitter strategy
reconstructed_data_df = pd.DataFrame(np.dot(corrected_nmfall.values, factor_to_lipid), index = corrected_nmfall.index, columns = data.columns)

# make sure the data is positive as all downstream computations are NMF-based
reconstructed_data_df = reconstructed_data_df - np.min(reconstructed_data_df) + 1e-7

# save reconstructed_data_df as an additional slot in the adata object



##################################################################################################################
# BLOCK 5: CALCULATE A tSNE TO VISUALIZE THE DATASET -> function tsne
##################################################################################################################

embds = corrected_nmfall#[::N] # do a minimal downsampling if the user desires

scaler = StandardScaler()
x_train = scaler.fit_transform(embds)

affinities_train = affinity.PerplexityBasedNN(
    x_train,
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)

init_train = x_train[:,[N, M]] # initialize with two uncorrelated NMFs, default by selecting automatically or use pair chosen by the user

embedding_train = TSNEEmbedding(
    init_train,
    affinities_train,
    negative_gradient_method="fft",
    n_jobs=8,
    verbose=True,
)

embedding_train_1 = embedding_train.optimize(n_iter=500, exaggeration=1.2)

embedding_train_N = embedding_train_1.optimize(n_iter=100, exaggeration=2.5)

# store the tSNE as X_TSNE in the adata object
