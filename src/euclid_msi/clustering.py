# also extract a dataframe with the spatial coordinates, columns 'Section', 'xccf', 'yccf', 'zccf' or 'Section', 'z', 'y', 'x' (note the correspondence); treat as same from now on, updating code where needed (tbd)

##################################################################################################################
# BLOCK 1: USE A CONVENTIONAL SCANPY LEIDEN TO CLUSTER THE DATA ON THE HARMONIZED NMF EMBEDDINGS -> function leiden_nmf
##################################################################################################################
# just do leiden on the harmonized NMF slot of the adata object. user can choose if to do so with reference sections only or whole dataset


##################################################################################################################
# BLOCK 2: LEARN A TRANSFERRABLE, LOCALLY-ENHANCED CLUSTERING OF THE DATA -> function learn_euclid_clustering
##################################################################################################################
import os
import pickle
import warnings
from collections import deque, Counter
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
from threadpoolctl import threadpool_limits, threadpool_info

from tqdm import tqdm
from kneed import KneeLocator
from PyPDF2 import PdfMerger

# Set thread limits and environment variables
threadpool_limits(limits=8)
os.environ['OMP_NUM_THREADS'] = '6'

# Suppress warnings
warnings.filterwarnings('ignore')

# load the adata object created in the embedding.py script

# extract the coordinates as a pandas dataframe coordinates, the harmonized-nmf reconstructed dataset as reconstructed_data_df, and the harmonized nmf as harmonized_nmf_result

# compute the standardized global embeddings to always use them along with the local ones and the local ones with memory
nmf_result = harmonized_nmf_result.loc[data.index,:]
scalerglobal = StandardScaler()
standardized_embeddings_GLOBAL = pd.DataFrame(scalerglobal.fit_transform(nmf_result), index=nmf_result.index, columns=nmf_result.columns)
standardized_embeddings_GLOBAL

# here i define a bunch of utility functions, hidden from the user, used by the backend to learn the euclid clustering. some could actually be shared with other scripts

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
    
    """
    plt.plot(np.arange(len(objective_values)), objective_values)
    plt.title("obj")
    plt.show()
    
    plt.plot(np.arange(len(modularity_scores)), modularity_scores)
    plt.title("mod")
    plt.show()
        
    plt.plot(np.arange(len(num_communities)), num_communities)
    plt.title("ncomms")
    plt.show()
    """
    
    max_index = np.argmax(objective_values)
    best_gamma = gamma_values[max_index]
    best_modularity = modularity_scores[max_index]
    best_num_comms = num_communities[max_index]
    #print(f'Number of communities at best gamma: {best_num_comms}')
    
    sc.tl.leiden(adata, resolution=best_gamma, key_added='leiden_best') # run Leiden one final time with best parameters
    clusters = adata.obs['leiden_best'].astype(int).values
    #print(clusters)
    
    N_factors = best_num_comms
    
    # 4. pick a representative lipid from each block, use to initialize W
    dist = 1 - corr_matrix
    np.fill_diagonal(dist, 0)
    dist = np.maximum(dist, dist.T)  # as numerical instability makes it unreasonably asymmetric
    dist_condensed = squareform(dist, checks=True)
    representatives = []
    
    for i in range(0, N_factors):
        cluster_members = np.where(clusters == i)[0]
        #print(cluster_members)
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
    MINDATA = np.min(data)
    data_offset = data
    
    data_offset = np.ascontiguousarray(data_offset)
    W_init = np.ascontiguousarray(W_init)
    H_init = np.ascontiguousarray(H_init)
    
    nmf_result = nmf.fit_transform(
        data_offset,
        W=W_init,
        H=H_init
    )
    
    nmf_result = nmf.transform(data_offset)
    
    nmfdf = pd.DataFrame(nmf_result, index=data.index)
    factor_to_lipid = nmf.components_
    
    return nmfdf, factor_to_lipid, N_factors, nmf


# use the harmony-NMF-corrected data from now on for splitting, but the uMAIA output for differential testing with conservative FDR control

rawlips = data.copy()
data = reconstructed_data_df

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
#normalized_datemp.to_hdf("normalized_datemp_raw.h5ad", key="table")

# prepare a scanpy object with also the raw lipids values for differential testing

adata = sc.AnnData(X=data)
adata.obsm['spatial'] = coordinates[['zccf', 'yccf', 'Section']].loc[data.index,:].values

adata.obsm['lipids'] = normalized_datemp

adata

# prepare useful plotting functions

global_min_z = coordinates['zccf'].min()
global_max_z = coordinates['zccf'].max()
global_min_y = -coordinates['yccf'].max() 
global_max_y = -coordinates['yccf'].min()  

def plot_spatial_localPCA(spat, pc_top):
    
    figs = []
    
    for PC_I in range(0, pc_top.shape[1]):
    
        results = []
        filtered_data = pd.concat([pd.DataFrame(spat, columns = ['zccf','yccf','Section']), pd.DataFrame(pc_top[:,PC_I],columns=["test"])], axis=1)
    
        currentPC = "test"
        for section in filtered_data['Section'].unique():
            subset = filtered_data[filtered_data['Section'] == section]
    
            perc_2 = subset[currentPC].quantile(0.02)
            perc_98 = subset[currentPC].quantile(0.98)
    
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
    
            ax.scatter(ddf['zccf'], -ddf['yccf'], c=ddf[currentPC], cmap="PuOr", s=0.5,rasterized=True, vmin=med2p, vmax=med98p) 
            ax.axis('off')
            ax.set_aspect('equal')
            ax.set_xlim(global_min_z, global_max_z)
            ax.set_ylim(global_min_y, global_max_y)
    
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = Normalize(vmin=med2p, vmax=med98p)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, cax=cbar_ax)
    
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        #plt.show()
        
        figs.append(fig)
        
    return figs

def plot_spatial_localPCA_kMeans(dd2, kmeans_labels):

    dd2 = pd.DataFrame(dd2, columns = ['zccf','yccf','Section'])
    dd2['cat_code'] = pd.Series(np.array(kmeans_labels)).astype('category').cat.codes

    color_map = {0: 'purple', 1: 'yellow'}
    dd2['color'] = dd2['cat_code'].map(color_map)
    
    fig, axes = plt.subplots(4, 8, figsize=(40, 20))
    axes = axes.flatten()
    dot_size = 0.3
    
    sections_to_plot = range(1,33)
    
    for i, section_num in enumerate(sections_to_plot):
        ax = axes[i]
        xx = dd2[dd2["Section"] == section_num]
        #print(xx['color'].value_counts())
        sc1 = ax.scatter(xx['zccf'], -xx['yccf'],
                         c=np.array(xx['color']), s=dot_size, alpha=1, rasterized=True)
        ax.axis('off')
        ax.set_aspect('equal')  
        ax.set_xlim(global_min_z, global_max_z)
        ax.set_ylim(global_min_y, global_max_y)
    
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    #plt.show()
    
    return fig
    
def plot_embeddingsPCA(embeddings, kmeans_labels):
    
    figs = []
    
    for i in range(embeddings.shape[1]-1):
        fig = plt.figure()
        
        plt.scatter(embeddings[:, i][::10], embeddings[:, (i+1)][::10], c=kmeans_labels[::10], s=0.005, alpha=0.5, rasterized=True)
        #plt.show()
        
        figs.append(fig)
        
    return figs
        
def plot_tSNE(indexes, kmeans_labels):
    
    kmeans_labels = pd.DataFrame(kmeans_labels, index = indexes)
    tsne_ds_now = tsne_ds.loc[np.intersect1d(indexes, np.array(tsne_ds.index)),:]
    kmeans_labels = kmeans_labels.loc[tsne_ds_now.index,:]
    
    fig = plt.figure()
    
    plt.scatter(tsne_ds.iloc[:,0], tsne_ds.iloc[:,1], s=0.0005, alpha=0.5, c="gray", rasterized=True)
    plt.scatter(tsne_ds_now.iloc[:,0], tsne_ds_now.iloc[:,1], s=0.005, alpha=1, c=np.array(kmeans_labels), rasterized=True)
    #plt.show()
    
    return fig

# a function to assess whether a cluster is continuous along the AP axis

# count the voxels per section as we'll need this as a normalization factor for continuity assessment
vcnorm = coordinates.loc[data.index,'Section'].value_counts()
vcnorm.index = vcnorm.index.astype(int)
vcnorm = vcnorm.sort_index()

def continuity_check(dd2, kmeans_labels):
    dd2 = pd.DataFrame(dd2, columns = ['zccf','yccf','Section'])
    dd2['color'] = pd.Series(kmeans_labels).astype('category').cat.codes

    enough_sectionss = []
    number_of_peakss = []
    peak_ratios = []

    for bbb in np.array(dd2['color'].unique()):
        test = dd2.loc[dd2['color'] == bbb,] 
        value_counts = test['Section'].value_counts()
        value_counts.index = value_counts.index.astype(int)
        aap = value_counts.sort_index()
        ap = np.array(aap)
        ap[ap < 10] = 0

        # check if in ap there are at least two consecutive nonzero entries or 4 total nonzero entries
        ap_nonnull = np.sum(ap > 0)  > 3
        apflag = False
        for boh in range(len(ap) - 1):
            if ap[boh] != 0 and ap[boh + 1] != 0:
                apflag = True
        enough_sections = ap_nonnull & apflag

        # check the number of peaks and if >1 peak the ratio of peak heights to assess continuuty
        ap = aap / vcnorm
        ap = np.array(ap)
        zero_padded_ap = np.pad(ap, pad_width=1, mode='constant', constant_values=0)
        smoothed_ap = gaussian_filter1d(zero_padded_ap, sigma=1.8)
    
        #plt.plot(smoothed_ap)
        #plt.show()
        
        peaks, properties = find_peaks(smoothed_ap, height=0)
        number_of_peaks = len(peaks)
        
        if number_of_peaks > 1:
            peak_heights = properties['peak_heights']
            top_peaks = np.sort(peak_heights)[-2:]  # get the two highest peaks
            peak_ratio = top_peaks[1] / top_peaks[0] 
        else:
            peak_ratio = 10

        enough_sectionss.append(enough_sections)
        number_of_peakss.append(number_of_peaks)
        peak_ratios.append(peak_ratio)

    return enough_sectionss[0], enough_sectionss[1], number_of_peakss[0], number_of_peakss[1], peak_ratios[0], peak_ratios[1]

# a function to check for differential lipids between two groups

def differential_lipids(lipidata, kmeans_labels, min_fc=0.2, pthr=0.05):
    results = []

    a = lipidata[kmeans_labels == 0,:]
    b = lipidata[kmeans_labels == 1,:]
    
    for rrr in range(lipidata.shape[1]):
       
        groupA = a[:,rrr]
        groupB = b[:,rrr]
    
        # log2 fold change
        meanA = np.mean(groupA)
        meanB = np.mean(groupB)
        log2fold_change = np.abs(np.log2(meanB / meanA)) if meanA > 0 and meanB > 0 else np.nan
    
        # Wilcoxon test
        try:
            _, p_value = mannwhitneyu(groupA, groupB, alternative='two-sided')
        except ValueError:
            p_value = np.nan
    
        results.append({'lipid': rrr, 'log2fold_change': log2fold_change, 'p_value': p_value})

    results_df = pd.DataFrame(results)

    # correct for multiple testing
    reject, pvals_corrected, _, _ = multipletests(results_df['p_value'].values, alpha=0.05, method='fdr_bh')
    results_df['p_value_corrected'] = pvals_corrected
    promoted = results_df.loc[(results_df['log2fold_change'] > min_fc) & (results_df['p_value_corrected'] < pthr),:]

    alteredlips = np.sum((results_df['log2fold_change'] > min_fc) & (results_df['p_value_corrected'] < pthr))

    return alteredlips, promoted

# a function to check for ranking PCs based on their spatial informativeness, meaning low variance of section-wise variances (i.e., low batch effect,
# all sections live in the same world, and high mean of variances, i.e., high informativeness, individual sections tend to contain variation)

def rank_features_by_combined_score(tempadata):
    sections = tempadata.obsm['spatial'][:, 2]  
    
    unique_sections = np.unique(sections)

    var_of_vars = []
    mean_of_vars = []

    for i in range(tempadata.X.shape[1]):
        feature_values = tempadata.X[:, i]

        section_variances = []
        for section in unique_sections:
            section_values = feature_values[sections == section]
            section_variance = np.var(section_values)
            section_variances.append(section_variance)

        var_of_vars.append(np.var(section_variances))
        mean_of_vars.append(np.mean(section_variances))

    var_of_vars = np.array(var_of_vars)
    mean_of_vars = np.array(mean_of_vars)

    # put them in a comparable scale
    var_of_vars = var_of_vars / np.mean(var_of_vars)
    mean_of_vars = mean_of_vars / np.mean(mean_of_vars)

    # give a bit more weight to the variability in the section vs the lack of batch effect
    combined_score = -var_of_vars/2 + mean_of_vars

    ranked_indices = np.argsort(combined_score)[::-1]

    return ranked_indices

# a function to find an elbow in cumulative absolute loadings to decide how many and which lipids have at least partially contributed to a split
def find_elbow_point(values):
    sorted_values = np.sort(np.abs(values))[::-1] 
    cumulative_variance = np.cumsum(sorted_values) / np.sum(sorted_values)
    #plt.plot(cumulative_variance)
    kneedle = KneeLocator(range(1, len(cumulative_variance) + 1), cumulative_variance, curve='concave', direction='increasing')
    elbow = kneedle.elbow
    #plt.scatter(elbow, cumulative_variance[elbow], c="red")
    #plt.show()
    return elbow

import itertools

# a function to generate sorted combinations of components to be used to attempt a batch effect-free clustering step
def generate_combinations(n, limit=200):
    all_combinations = []
    for r in range(n, 0, -1):
        for combination in itertools.combinations(range(n), r):
            all_combinations.append(combination)
            if len(all_combinations) == limit:
                return all_combinations
    return all_combinations

# define a function (to be optimized) for faster Leiden clustering
def leidenalg_clustering(inputdata, Nneigh=40, Niter=5):
    nn = NearestNeighbors(n_neighbors=Nneigh, n_jobs=4)
    nn.fit(inputdata)
    knn = nn.kneighbors_graph(inputdata)
    
    G = nx.Graph(knn)
    
    g = ig.Graph.from_networkx(G)
    
    partitions = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, n_iterations=Niter, seed=230598)
    labels = np.array(partitions.membership)
    
    return labels

## Prepare the Node class to create a hierarchical classifier deployable on new data
# this class will store the tree of classifiers, it's the trained object that, given a lipidomic dataset, is able to return lipizones
class Node:
    def __init__(self, level, path=[]):
        self.level = level
        self.path = path
        self.scaler = None
        self.nmf = None
        self.xgb_model = None
        self.feature_importances = None  # store feature importances at each split to establish a minimal palette
        self.children = {}
        self.factors_to_use = None
        
def undersample(X, y, sampling_strategy='auto'):
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled

# define validation and test sections

valsec = (coordinates['Section'].unique()[::5] + 2)[:-1]
testsec = (coordinates['Section'].unique()[::5] + 1)[:-1]
trainsec = np.setdiff1d(np.setdiff1d(coordinates['Section'].unique(), testsec), valsec)

valpoints = coordinates.loc[coordinates['Section'].isin(valsec),:].index
testpoints = coordinates.loc[coordinates['Section'].isin(testsec),:].index
trainpoints = coordinates.loc[coordinates['Section'].isin(trainsec),:].index

## CORE METHOD!!! Hierarchical bipartite splitter with NMF recalculation at each iteration and an XGBC on top of it

## how many splits do we want?
max_depth = 15 # go down where possible! it should automatically stop before everywhere given the biological plausibility criteria!

# parameters
K = 60 # big K for kmeans split before hierarchically reaggregating
min_voxels = 150 # minimum number of observations per lipizone
min_diff_lipids = 2 # minimum number of differential lipids (out of 108 used for clustering) to accept a split
min_fc = 0.2 # minimal fold change required to consider lipids differential
pthr = 0.05 # adjusted pvalue threshold for the wilcoxon test
thr_signal = 0.0000000001 # threshold on signal to keep a component for clustering
penalty1 = 1.5 # extent to penalize the embeddings of the previous split compared to the current one (division factor)
penalty2 = 2 # extent to penalize the global embeddings compared to the current one (division factor)
ACCTHR = 0.6 # lower bound generalization ability for the classifier to relabel and deem a split valid to go deeper
#top_loaders_general = []

def dosplit(current_adata, embds, path = [], splitlevel=0, plotting_active=False, loadings_active=False):
    print("Splitting level: "+str(splitlevel))
    
    # do not split further if the cluster is smaller than min_voxels
    if current_adata.X.shape[0] < min_voxels:
        print("one branch exhausted")
        return

    # do NMF on the current subset of the data
    nmfdf, loadings, N_factors, nmf = compute_seeded_NMF(pd.DataFrame(current_adata.X))
    nmf_result = nmfdf.values
    
    # omit NMFs with very low signal overall
    filter1 = np.abs(nmf_result).mean(axis=0) > thr_signal
    loadings = loadings[filter1, :]
    nmf_result = nmf_result[:, filter1]
    original_nmf_indices = np.arange(N_factors)[filter1]

    # rank the NMFs based on the variances score
    tempadata = sc.AnnData(X=nmf_result)
    tempadata.obsm['spatial'] = current_adata.obsm['spatial']
    goodpcs = rank_features_by_combined_score(tempadata)
    goodpcs_indices = original_nmf_indices[goodpcs.astype(int)]
    top_pcs_data = nmf_result[:, goodpcs.astype(int)]
    loadings = loadings[goodpcs.astype(int), :]

    # attempt the kMeans split with large k, followed by backSPIN reaggregation 
    # start with the top ranking according to the variances criterion and go down until we find something that respects AP continuity criteria!
    flag = False
    aaa = 0
    multiplets = generate_combinations(len(goodpcs))
    
    while (not flag) and (aaa < len(multiplets)):
        bestpcs = multiplets[aaa]
        embeddings = top_pcs_data[:,bestpcs]
        loadings_sel = loadings[bestpcs,:]
        selected_nmf_indices = goodpcs_indices[list(bestpcs)]
        
        # whiten the embeddings to prevent one direction to always dictate all splits - empirically results in better detection of blob-like structures
        scaler = StandardScaler()
        standardized_embeddings = scaler.fit_transform(embeddings)
        standardized_embeddings = scaler.transform(embeddings)
        
        # do the kMeans with a big K
        kmeans = KMeans(n_clusters=K, random_state=230598)
        globembds = standardized_embeddings_GLOBAL.loc[current_adata.obs_names,:].values
        embspace = np.concatenate((standardized_embeddings, embds/penalty1, globembds/penalty2),axis=1)
        kmeans_labels = kmeans.fit_predict(embspace) 
        #kmeans_labels = leidenalg_clustering(embspace)
       
        # reaggregate hierarchically the centroids of the K clusters to get a bipartition using backSPIN
        data_for_clustering = pd.DataFrame(current_adata.X,
                                   index=current_adata.obs_names, 
                                   columns=current_adata.var_names)
        data_for_clustering['label'] = kmeans_labels
        centroids = data_for_clustering.groupby('label').mean()
        centroids = pd.DataFrame(StandardScaler().fit_transform(centroids), columns=centroids.columns, index=centroids.index).T
        row_ix, columns_ix = backSPIN.SPIN(centroids, widlist=4)
        centroids = centroids.iloc[row_ix, columns_ix]
        _, _, _, gr1, gr2, _, _ , _, _= backSPIN._divide_to_2and_resort(sorted_data=centroids.values, wid=5) # wid controls size of neighborhood
        gr1 = np.array(centroids.columns)[gr1]
        gr2 = np.array(centroids.columns)[gr2]
        data_for_clustering['lab'] = 1
        data_for_clustering['lab'][data_for_clustering['label'].isin(gr2)] = 2

        # check the continuity of the resulting clusters along the AP axis
        enough_sections0, enough_sections1, number_of_peaks0, number_of_peaks1, peak_ratio0, peak_ratio1 = continuity_check(current_adata.obsm['spatial'], np.array(data_for_clustering['lab']))

        # check the differential lipids
        alteredlips, promoted = differential_lipids(current_adata.obsm['lipids'].values, kmeans_labels, min_fc, pthr)

        # also check that both branches of the split comprise at least N voxels ("min cells") - we don't try to trust super small clusters
        flag = ((np.sum(kmeans_labels == 1) > min_voxels) | (np.sum(kmeans_labels == 0) > min_voxels)) & (alteredlips > min_diff_lipids) & enough_sections0 and  enough_sections1 and ((number_of_peaks0 <3) or (peak_ratio0 > 1.4)) and ((number_of_peaks1 <3) or (peak_ratio1 > 1.4))
        aaa = aaa+1
        kmeans_labels = data_for_clustering['lab'].astype(int)
        
    # don't split anymore here if no PCs choice can respect AP continuity and differential lipids criteria
    if not flag:
        print("one branch exhausted because embeddings do not respect bland criteria on rostrocaudal axis and diff. lipids")
        return
    
    
    ############# train a classifier on these embeddings to predict the two labels with good generalization capability
    
    # extract the embeddings and the label for train, validation, and test set
    embeddings = pd.DataFrame(embspace, index=current_adata.obs_names) # embeddings
    X_train = embeddings.loc[embeddings.index.isin(trainpoints), :]
    X_val = embeddings.loc[embeddings.index.isin(valpoints), :]
    X_test = embeddings.loc[embeddings.index.isin(testpoints), :]
    
    kmeans_labels = kmeans_labels -1
    y_train = kmeans_labels.loc[X_train.index] # split labels to be able to be recovered by a nonlinear XGB classifier
    y_val = kmeans_labels.loc[X_val.index]
    y_test = kmeans_labels.loc[X_test.index]
    
    # balance the classes, maybe it's not really needed
    X_train_sub_US, y_train_sub_US = undersample(X_train, y_train)
    
    # train an XGB classifierwith good generalization ability on the validation set
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
    xgb_model.fit(
        X_train_sub_US, 
        y_train_sub_US,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=7,
        verbose=False
    )
    print("XGBoosted!")
    print(datetime.now().strftime("%H:%M"))
    
    feature_importances = xgb_model.feature_importances_

    test_pred = xgb_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Test accuracy in this subset: {test_accuracy}")
    val_pred = xgb_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    print(f"Validation accuracy in this subset: {val_accuracy}")
    train_pred = xgb_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"Training accuracy in this subset: {train_accuracy}")
    
    if test_accuracy < ACCTHR:
        print("one branch exhausted due to poor test set accuracy")
        return
    
    test_pred = pd.DataFrame(test_pred, index=X_test.index, columns=["kmean"])
    val_pred = pd.DataFrame(val_pred, index=X_val.index, columns=["kmean"])
    train_pred = pd.DataFrame(train_pred, index=X_train.index, columns=["kmean"])

    # overwrite the cluster labels with those picked by the classifier
    ind = kmeans_labels.index
    kmeans_labels = pd.concat([test_pred, val_pred, train_pred]).loc[ind]
    kmeans_labels = kmeans_labels+1
    kmeans_labels = kmeans_labels['kmean']

    cl1members = kmeans_labels[kmeans_labels == 1].index.values
    cl2members = kmeans_labels[kmeans_labels == 2].index.values

    # create a Node and store its transformations and feature importances
    node = Node(splitlevel, path=path)
    node.scaler = scaler
    node.nmf = nmf
    node.xgb_model = xgb_model
    node.feature_importances = feature_importances
    node.factors_to_use = selected_nmf_indices

    standardized_embeddings = pd.DataFrame(standardized_embeddings, index=current_adata.obs_names)
    cl1members = kmeans_labels[kmeans_labels == 1].index.values
    cl2members = kmeans_labels[kmeans_labels == 2].index.values

    current_adata1 = current_adata[current_adata.obs_names.isin(cl1members)]#.copy()
    current_adata2 = current_adata[current_adata.obs_names.isin(cl2members)]#.copy()
    embd1 = standardized_embeddings.loc[standardized_embeddings.index.isin(cl1members),:].values
    embd2 = standardized_embeddings.loc[standardized_embeddings.index.isin(cl2members),:].values

    splitlev = splitlevel+1

    clusteringLOG.loc[cl1members, "level_"+str(splitlev)] = 1
    clusteringLOG.loc[cl2members, "level_"+str(splitlev)] = 2
    
    
    if plotting_active:
        # show the result in space, and keep track of the lipids which have contributed to splits, counting them as a "diversity metric"
        figs = plot_spatial_localPCA(current_adata.obsm['spatial'], embeddings)
        for fig in figs:
            pdf_pages.savefig(fig)
            plt.close(fig)
        
        figs = plot_embeddingsPCA(standardized_embeddings, kmeans_labels)
        for fig in figs:
            pdf_pages.savefig(fig)
            plt.close(fig)
        
        fig = plot_tSNE(np.array(data_for_clustering.index), kmeans_labels)
        pdf_pages.savefig(fig)
        plt.close(fig)
        
        fig = plot_spatial_localPCA_kMeans(current_adata.obsm['spatial'], kmeans_labels)
        pdf_pages.savefig(fig)
        plt.close(fig)

    if loadings_active:
        # check which lipids are dictating this split and the diversity of lipids that have been used until now
        loadings_sel = pd.DataFrame(loadings_sel, columns = data.columns).T
        loadings_sel = np.abs(loadings_sel)
        top_loaders_ALL = []
        for col in loadings_sel.columns:
            elbow_point = find_elbow_point(loadings_sel[col].values)
            top_loaders = loadings_sel.nlargest(elbow_point, col).index.tolist()
            top_loaders_ALL = top_loaders_ALL+top_loaders
            print("Top 5 drivers PC" + str(col) + ": " + str(top_loaders[:5]))
        #top_loaders_general = top_loaders_general+top_loaders_ALL
        diversity_metric = len(np.unique(top_loaders_ALL))
        print("Diversity metric: " + str(diversity_metric))
        print("Differential lipids between the two branches: ")
        print(np.array(adata.obsm['lipids'].columns)[promoted['lipid'].values].astype(str))

    # do the recursion
    if splitlev < (max_depth+1):
        child_path0 = path + [0]
        child_path1 = path + [1]
        childnode0, childnode1 = dosplit(current_adata1, embd1, child_path0, splitlevel=splitlev), dosplit(current_adata2, embd2, child_path1, splitlevel=splitlev)
        node.children[0] = childnode0
        node.children[1] = childnode1
        return node
    else:
        print("depth reached!")
        return

DSFACT = 1 # downscale for testing if needed
column_names = [f"level_{i}" for i in range(1,max_depth+1)]
clusteringLOG = pd.DataFrame(0, index=data.index, columns=column_names)[::DSFACT]   
root_node = dosplit(adata[::DSFACT], standardized_embeddings_GLOBAL[::DSFACT], splitlevel=0)

# save the hierarchical bipartite tree to file
tree = clusteringLOG
tree['cluster'] = tree['level_1'].astype(str) + tree['level_2'].astype(str) + tree['level_3'].astype(str) + tree['level_4'].astype(str) + tree['level_5'].astype(str) + tree['level_6'].astype(str)  + tree['level_7'].astype(str)  + tree['level_8'].astype(str)  + tree['level_9'].astype(str)  + tree['level_10'].astype(str) + tree['level_11'].astype(str) + tree['level_12'].astype(str) + tree['level_13'].astype(str) + tree['level_14'].astype(str) + tree['level_15'].astype(str)
tree['class'] = tree['level_1'].astype(str) + tree['level_2'].astype(str) + tree['level_3'].astype(str)
tree.to_hdf("tree_clustering_down_clean.h5ad", key="table")

# save the tree to file
import pickle

filename = "rootnode_clustering_whole_clean.pkl"
with open(filename, "wb") as file:
    pickle.dump(root_node, file) 



##################################################################################################################
# BLOCK 3: ASSIGN COLORS TO CLUSTERS 
##################################################################################################################
# represent all clusters in space with nice colors. note, make it a bit more flexible in terms of parametrs
def spatial_plot_all_lipizones_atlas(data, coordinates, tree):
    
    conto = np.load("eroded_annot.npy")

    coordinates = coordinates.fillna(0)
    coordinates = coordinates.replace([np.inf, -np.inf], 0)
    xs,ys,zs = (coordinates['xccf']*40).astype(int), (coordinates['yccf']*40).astype(int), (coordinates['zccf']*40).astype(int)
    xs.loc[xs>527]=527
    ys.loc[ys>319]=319
    zs.loc[zs>455]=455
    coordinates['border'] = conto[xs,ys,zs]
    
    data_std = (data.values - np.mean(data.values, axis=1)[:, None]) / (np.std(data.values, axis=1)[:, None] + 1e-8)
    data_std = pd.DataFrame(data_std, index=data.index, columns= np.array(data.columns)) 
    
    levels = pd.concat([data_std, coordinates.loc[data_std.index,:], tree], axis=1) #_std
    
    dd2 = levels
    
    lipid_columns = np.array(data.columns)
    
    divisions = dd2['class'].unique()
    colormaps = ["RdYlBu",  "terrain", "PiYG", "cividis", "plasma", "PuRd", "inferno", "PuOr"]
        
    dd2['R'] = np.nan
    dd2['G'] = np.nan
    dd2['B'] = np.nan
    
    dfs = []

    for division, colormap_name in tqdm(zip(divisions, colormaps)):
    
        if len(dd2.loc[dd2['class'] == division, 'cluster'].unique()) > 1:
            
            print(division)
            
            datasubaqueo = dd2[dd2['class'] == division]
        
            clusters = datasubaqueo['cluster'].unique()
        
            lipid_df = pd.DataFrame(columns=lipid_columns)
        
            for i in range(len(clusters)):
                datasub = datasubaqueo[datasubaqueo['cluster'] == clusters[i]] 
                lipid_data = datasub.loc[:,lipid_columns].mean(axis=0)
                lipid_row = pd.DataFrame([lipid_data], columns=lipid_columns)
                lipid_df = pd.concat([lipid_df, lipid_row], ignore_index=True)
        
            column_means = lipid_df.mean()
            normalized_lipid_df = lipid_df.div(column_means, axis='columns')
            
            normalized_lipid_df.index = clusters
            normalized_lipid_df = normalized_lipid_df.T
        
            pca_columns = datasubaqueo.loc[:, lipid_columns]
            grouped_data = datasubaqueo[['cluster']].join(pca_columns)
            centroids = grouped_data.groupby('cluster').mean()
        
            distance_matrix = squareform(pdist(centroids, metric='euclidean'))
            distance_df = pd.DataFrame(distance_matrix, index=centroids.index, columns=centroids.index)
        
            np.fill_diagonal(distance_df.values, np.inf)
            initial_min_index = np.unravel_index(np.argmin(distance_df.values), distance_df.shape)
            ordered_elements = [distance_df.index[initial_min_index[0]], distance_df.columns[initial_min_index[1]]]
            distances = [0, distance_df.iloc[initial_min_index]]
        
            while len(ordered_elements) < len(distance_df):
                last_added = ordered_elements[-1]
                remaining_distances = distance_df.loc[last_added, ~distance_df.columns.isin(ordered_elements)]
                next_element = remaining_distances.idxmin()
                ordered_elements.append(next_element)
                distances.append(remaining_distances[next_element])
    
            ordered_elements
        
            leaf_sequence = ordered_elements
        
            sequential_distances = distances
        
            cumulative_sequential_distances = np.cumsum(sequential_distances)
        
            normalized_distances = cumulative_sequential_distances / cumulative_sequential_distances[-1]
            colormap = plt.get_cmap(colormap_name)
            colors = [colormap(value) for value in normalized_distances]
        
            hsv_colors = [mcolors.rgb_to_hsv(rgb[:3]) for rgb in colors] 
        
            modified_hsv_colors = []
            for i, (h, s, v) in enumerate(hsv_colors):
                if (i + 1) % 2 != 0: 
                    s = min(1, s + 0.7 * s)
                modified_hsv_colors.append((h, s, v))
    
            modified_rgb_from_hsv = [mcolors.hsv_to_rgb(hsv) for hsv in modified_hsv_colors]
        
            rgb_list = [list(rgb) for rgb in modified_rgb_from_hsv]
        
            lipocolor = pd.DataFrame(rgb_list, index=leaf_sequence, columns=['R', 'G', 'B'])
        
            lipocolor_reset = lipocolor.reset_index().rename(columns={'index': 'cluster'})
            print(lipocolor_reset)
            indices = datasubaqueo.index
            
            datasubaqueo = datasubaqueo.iloc[:,:-3]
            datasubaqueo = pd.merge(datasubaqueo, lipocolor_reset, on='cluster', how='left')
        
            datasubaqueo.index = indices
        
            dd2.update(datasubaqueo[['R', 'G', 'B']])
    
        else:
            datasubaqueo = dd2[dd2['class'] == division]
            datasubaqueo['R'] = 0
            datasubaqueo['G'] = 0
            datasubaqueo['B'] = 0
            dd2.update(datasubaqueo[['R', 'G', 'B']])

    def rgb_to_hex(r, g, b):
        try:
            """Convert RGB (0-1 range) to hexadecimal color."""
            r, g, b = [int(255 * x) for x in [r, g, b]]  # scale to 0-255
            return f'#{r:02x}{g:02x}{b:02x}'
        except:
            return np.nan
    
    dd2['lipizone_color'] = dd2.apply(lambda row: rgb_to_hex(row['R'], row['G'], row['B']), axis=1)
    
    dd2['lipizone_color'].fillna('gray', inplace=True) 

    fig, axes = plt.subplots(4, 8, figsize=(40, 20))
    axes = axes.flatten()
    dot_size = 0.3
    
    sections_to_plot = range(1,33)
    
    global_min_z = dd2['zccf'].min()
    global_max_z = dd2['zccf'].max()
    global_min_y = -dd2['yccf'].max() 
    global_max_y = -dd2['yccf'].min()  
    
    for i, section_num in enumerate(sections_to_plot):
        ax = axes[i]
        xx = dd2[dd2["Section"] == section_num]
        sc1 = ax.scatter(xx['zccf'], -xx['yccf'],
                         c=np.array(xx['lipizone_color']), s=dot_size, alpha=1, rasterized=True)
        
        cont = coordinates.loc[xx.index,:]
        cont = cont.loc[cont['border'] == 1,:]

        ## add an overlaid contour
        ax.scatter(cont['zccf'], -cont['yccf'],
                         c='black', s=dot_size, alpha=0.2, rasterized=True)
        
        ax.axis('off')
        ax.set_aspect('equal')  
        ax.set_xlim(global_min_z, global_max_z)
        ax.set_ylim(global_min_y, global_max_y)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

    return dd2['lipizone_color']

# make sure here to annotate the colors in the adata object!

lipizone_colors = spatial_plot_all_lipizones_atlas(data, coordinates, tree)
tree.to_hdf("colorzones.h5ad", key="table")


##################################################################################################################
# BLOCK 4: APPLY THE LEARNT EUCLID CLUSTERING TREE TO THE WHOLE DATASET -> function apply_euclid_clustering
##################################################################################################################
scalerglobal # the one learnt on the reference on which the self-supervised euclid was run
harmonized_nmf_result # the whole dataset harmonized nmf embeddings slot
standardized_embeddings_GLOBAL = pd.DataFrame(scalerglobal.transform(harmonized_nmf_result), index=harmonized_nmf_result.index, columns=harmonized_nmf_result.columns)

# extract as for the reference the slot of the approximated dataset
reconstructed_data_df = reconstructed_data_df - np.min(reconstructed_data_df) + 1e-7
data = reconstructed_data_df

# MAKE SURE TO HAVE THE SAME STRUCTURE AND PARAMETERS AS THOSE PROVIDED DURING CLUSTER LEARNING
# the parameters over which the clustering algorithm was built
## how many splits do we want?
max_depth = 15 # go down where possible! it should automatically stop before everywhere given the biological plausibility criteria!

# parameters
K = 60 # big K for kmeans split before hierarchically reaggregating
min_voxels = 150 # minimum number of observations per lipizone
min_diff_lipids = 2 # minimum number of differential lipids (out of 108 used for clustering) to accept a split
min_fc = 0.2 # minimal fold change required to consider lipids differential
pthr = 0.05 # adjusted pvalue threshold for the wilcoxon test
thr_signal = 0.0000000001 # threshold on signal to keep a component for clustering
penalty1 = 1.5 # extent to penalize the embeddings of the previous split compared to the current one (division factor)
penalty2 = 2 # extent to penalize the global embeddings compared to the current one (division factor)
ACCTHR = 0.6 # lower bound generalization ability for the classifier to relabel and deem a split valid to go deeper
#top_loaders_general = []

# this class stores the tree of classifiers, it's the trained object that, given a lipidomic dataset, is able to return lipizones
class Node:
    def __init__(self, level, path=[]):
        self.level = level
        self.path = path
        self.scaler = None
        self.nmf = None
        self.xgb_model = None
        self.feature_importances = None  # store feature importances at each split to establish a minimal palette
        self.children = {}
        self.factors_to_use = None

def traverse_tree(node, current_adata, embds, paths, level=0):
    print(level)
    
    if node is None or not node.children:
        return

    if current_adata.shape[0] == 0:
        print(f"Empty data at level {level}. Returning.")
        return
    
    # apply NMF to the current data subset
    nmf = node.nmf
    X_nmf = nmf.transform(current_adata.X)
    
    # select the factors used at this node
    factors_to_use = node.factors_to_use
    X_nmf = X_nmf[:, factors_to_use]

    # scale (whiten) the NMF-transformed data
    scaler = node.scaler
    X_scaled = scaler.transform(X_nmf)

    globembds = standardized_embeddings_GLOBAL.loc[current_adata.obs_names].values / penalty2
    embspace = np.concatenate((X_scaled, embds / penalty1, globembds), axis=1)

    # predict the child cluster using the stored XGBoost model
    xgb_model = node.xgb_model
    child_labels = xgb_model.predict(embspace)
    
    unique_labels, counts = np.unique(child_labels, return_counts=True)
    
    for i, index in enumerate(current_adata.obs_names):
        if index not in paths:
            paths[index] = []
        paths[index].append(child_labels[i])

    cl0members = current_adata.obs_names[child_labels == 0]
    cl1members = current_adata.obs_names[child_labels == 1]

    current_adata0 = current_adata[current_adata.obs_names.isin(cl0members)]
    current_adata1 = current_adata[current_adata.obs_names.isin(cl1members)]

    if current_adata0.shape[0] == 0 or current_adata1.shape[0] == 0:
        print(f"Warning: One child node has no data at level {level}. Skipping.")
        return

    embd0 = X_scaled[child_labels == 0]
    embd1 = X_scaled[child_labels == 1]

    # recursively traverse the child nodes
    traverse_tree(node.children[0], current_adata0, embd0, paths, level + 1)
    traverse_tree(node.children[1], current_adata1, embd1, paths, level + 1)

# import the root node if needed as we created it before
filename = "rootnode_clustering_whole_clean.pkl"
with open(filename, "rb") as file:
    root_node = pickle.load(file)

# prepare a convenient temporary anndata object
new_adata = sc.AnnData(X=reconstructed_data_df)
new_adata.obsm['spatial'] = metadata[['zccf', 'yccf', 'Section']].loc[reconstructed_data_df.index,:].values

# deploy our tree with NMF and XGBC onto the full multi-dataset dataset
DSFACT = 1
paths = {}
embds = standardized_embeddings_GLOBAL[::DSFACT].values
traverse_tree(root_node, new_adata[::DSFACT], embds, paths)
df_paths = pd.DataFrame.from_dict(paths, orient='index')
df_paths.columns = [f'level_{i}' for i in range(1, 12)]
df_paths = df_paths.fillna(-1)
df_paths = df_paths.astype(int) + 1
df_paths.to_hdf("splithistory_allbrains.h5ad", key="table")
tree = df_paths.copy()

tree['lipizone'] = tree['level_1'].astype(str)
for i in range(2,12):
    tree['lipizone'] = tree['lipizone'].astype(str) + tree['level_'+str(i)].astype(str)

# colorzones is the colors set learnt in the previous block on the reference sections.
colors = pd.read_hdf("colorzones.h5ad", key="table")
mapping = pd.DataFrame({
    'lipizone': tree.loc[colors.index, 'lipizone'],
    'lipizone_color': colors['lipizone_color']
})

modal_mapping = mapping.groupby('lipizone').agg(
    lipizone_color=('lipizone_color', lambda x: x.mode().iloc[0])
).reset_index()

modal_mapping.set_index('lipizone', inplace=True)

tree['lipizone_color'] = tree['lipizone'].map(modal_mapping['lipizone_color'])
modal_mapping

# add df_paths (i.e., the hierarchy of clusters) to the main adata object initially loaded (built throughout the package), and save that adata object to file


##################################################################################################################
# BLOCK 5: ASSIGN A NAME TO EACH CLUSTER BASED ON THEIR ANATOMICAL LOCALIZATION -> function name_lipizones_anatomy
##################################################################################################################




##################################################################################################################
# BLOCK 6: PLOT EACH CLUSTER TO A SEPARATE PDF FILE TO FAVOR VISUAL INSPECTION -> function clusters_to_pdf
##################################################################################################################

# plot the lipizones one by one to file
levels = pd.concat([coordinates.loc[data.index,:], tree], axis=1)
levels['lipizone_names'] = lipizone_names
levels['Section'] = coordinates['Section']
levels['zccf'] = coordinates['zccf']
levels['yccf'] = coordinates['yccf']
levels['xccf'] = coordinates['xccf']

output_folder = "lipizones539"
os.makedirs(output_folder, exist_ok=True)

dot_size = 0.3
sections_to_plot = range(1, 33)
dd2 = levels
global_min_z = dd2['zccf'].min()
global_max_z = dd2['zccf'].max()
global_min_y = -dd2['yccf'].max()
global_max_y = -dd2['yccf'].min()
unique_lev4cols = np.sort(dd2['lipizone_names'].unique())

for unique_val in tqdm(unique_lev4cols):
    print(unique_val)
    fig, axes = plt.subplots(4, 8, figsize=(40, 20))
    axes = axes.flatten()
    for i, section_num in enumerate(sections_to_plot):
        ax = axes[i]
        xx = dd2[dd2["Section"] == section_num]
        sc1 = ax.scatter(xx['zccf'], -xx['yccf'], c=xx['lipizone_names'].astype("category").cat.codes,
                         cmap='Grays', s=dot_size * 2, alpha=0.2, rasterized=True)
        xx_highlight = xx[xx['lipizone_names'] == unique_val]
        sc2 = ax.scatter(xx_highlight['zccf'], -xx_highlight['yccf'],
                         c='red', s=dot_size, alpha=1, rasterized=True)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_xlim(global_min_z, global_max_z)
        ax.set_ylim(global_min_y, global_max_y)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle(unique_val)
    plt.tight_layout()
    
    filename = f"{unique_val}.pdf"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath)
    plt.show()
    plt.close(fig)

subfolder_name = "lipizones589" # note this should be updated after running the XGBC
cwd = os.getcwd()
subfolder_path = os.path.join(cwd, subfolder_name)
merger = PdfMerger()

for filename in tqdm(sorted(os.listdir(subfolder_path))):
    if filename.endswith(".pdf"):
        file_path = os.path.join(subfolder_path, filename)
        merger.append(file_path)

output_filename = "20241125_lipizones_539.pdf"
output_file_path = os.path.join(cwd, output_filename)
merger.write(output_file_path)
merger.close()
