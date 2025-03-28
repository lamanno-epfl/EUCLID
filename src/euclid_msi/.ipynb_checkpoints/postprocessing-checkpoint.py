# load the adata object created in the embedding.py script
# extract the coordinates as a pandas dataframe coordinates, the harmonized-nmf reconstructed dataset as reconstructed_data_df, and the harmonized nmf as harmonized_nmf_result
# for the spatial coordinates, columns 'Section', 'xccf', 'yccf', 'zccf' or 'Section', 'z', 'y', 'x' (note the correspondence); treat as same from now on, updating code where needed (tbd)

##################################################################################################################
# BLOCK 1: IMPUTE LIPIDS ON SECTIONS WHERE THEIR MEASUREMENT FAILED BASED ON THE OTHER LIPIDS -> function xgboost_feature_restoration
##################################################################################################################

import os
import joblib
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from threadpoolctl import threadpool_limits, threadpool_info
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
import scanpy as sc
import umap.umap_ as umap

os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["OMP_NUM_THREADS"] = "6"
threadpool_limits(limits=8)

# extract the embeddings from the harmonized nmf slot of the anndata object

# extract as morans pandas dataframe the moran's i values for the features of the anndata object

isitrestorable = (morans > 0.4).sum(axis=1).sort_values()
torestore = isitrestorable[isitrestorable > 3].index # there must be at least three good sections to train on and one to evaluate on
torestore

cols = np.array(alldata.columns)
cols[:1400]=cols[:1400].astype(float).astype(str)
alldata.columns = cols

lipids_to_restore = alldata.loc[:,torestore.astype(float).astype(str)]
lipids_to_restore = lipids_to_restore.iloc[:-5,:]



## Select the sections to be used to train XGB models for imputation
usage_dataframe = morans.iloc[:,:70].copy() # use the atlases as the basis to impute on

# remove the broken sections
brokenones = alldata[['SectionID', 'BadSection']].drop_duplicates().dropna()
goodones = brokenones.loc[brokenones['BadSection'] == 0,'SectionID'].values
usage_dataframe = usage_dataframe.loc[:, usage_dataframe.columns.astype(float).isin(goodones)]

# choose the best sections to train and validate XGBoost models on
def top_3_above_threshold(row, threshold=0.4):
    
    above_threshold = row >= threshold
    
    if above_threshold.sum() >= 3:
        
        top_3 = row.nlargest(3).index
        result = pd.Series(False, index=row.index)
        result[top_3] = True
    else:
        result = above_threshold
    
    return result

usage_dataframe = usage_dataframe.apply(top_3_above_threshold, axis=1)

usage_dataframe=usage_dataframe.loc[np.array(usage_dataframe.sum(axis=1).index[usage_dataframe.sum(axis=1) > 2]),:]
usage_dataframe = usage_dataframe.loc[usage_dataframe.index.astype(float).astype(str) != '953.120019',:]
usage_dataframe # could be further be optimized by ensuring the 3 training sections are not-so-close-to-each-other

# some data prep
lipids_to_restore = lipids_to_restore.loc[:,usage_dataframe.index.astype(float).astype(str)]
lipids_to_restore['SectionID'] = alldata['SectionID']
coordinates = alldata.loc[embeddings.index, ['SectionID', 'x', 'y']]
coordinates['SectionID'] = coordinates['SectionID'].astype(float).astype(int).astype(str)

metrics_df = pd.DataFrame(
    columns=['train_pearson_r', 'train_rmse', 'val_pearson_r', 'val_rmse']
)

for index, row in tqdm(usage_dataframe.iterrows(), total=usage_dataframe.shape[0]):
    #try:
    train_sections = row[row].index.tolist()  
    val_sections = train_sections[1]
    train_sections = [train_sections[0], train_sections[2]]

    train_data = embeddings.loc[coordinates['SectionID'].isin(train_sections),:]
    y_train = lipids_to_restore.loc[train_data.index, str(index)]

    # take one out and use it for validation: can we trust this XGB model? 
    val_data = embeddings.loc[coordinates['SectionID'] == val_sections,:]
    y_val = lipids_to_restore.loc[val_data.index, str(index)]

    model = XGBRegressor()
    model.fit(train_data, y_train)

    train_pred = model.predict(train_data)
    val_pred = model.predict(val_data)

    train_pearson = pearsonr(y_train, train_pred)[0]
    val_pearson = pearsonr(y_val, val_pred)[0]
    print(val_pearson)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    metrics_df.loc[index] = {
        'train_pearson_r': train_pearson,
        'train_rmse': train_rmse,
        'val_pearson_r': val_pearson,
        'val_rmse': val_rmse
    }

    # save the model
    model_path = os.path.join('xgbmodels_onmnnnmf', str(index)+'_xgb_model.joblib')
    joblib.dump(model, model_path)

    #except Exception as e:
     #   print("exception at index: "+str(index))
      #  continue

## Deploy the trained XGB models across all acquisitions
# loop to import and deploy the models, creating one column at a time. deploy on all sections, also on the training ones, to be in-distribution
coordinates = coordinates[['SectionID',	'x',	'y']]
for file in tqdm(os.listdir('xgbmodels_onmnnnmf')[1:]):
    model_path = os.path.join('xgbmodels_onmnnnmf', file)
    model = joblib.load(model_path)
    coordinates[file] = model.predict(embeddings)
coordinates.columns = [
    col.replace('_xgb_model.joblib', '') if i >= 3 else col 
    for i, col in enumerate(coordinates.columns)
]

# filter with the metrics df to keep only "reliably imputed" lipids
metrics_df.to_csv("metrics_imputation_df.csv")

# keep only the lipids whose generalization Pearson's R is good enough (0.4 threshold)
cols = np.array(coordinates.columns)
cols[3:] = cols[3:].astype(float).astype(str)
coordinates.columns = cols
coordinates = coordinates.loc[:, metrics_df.loc[metrics_df['val_pearson_r'] > 0.4,:].index.astype(float).astype(str)]
coordinates.to_hdf("20241113_xgboost_recovered_lipids.h5ad", key="table")

# add the XGB-restored data as slot to the anndata object as X_lipidome



##################################################################################################################
# BLOCK 2: 3D INTERPOLATION BASED ON A 3D ANATOMICAL REFERENCE -> function anatomical_interpolation
##################################################################################################################

## Define a function to do a radially exponentially decaying weighted neighbors interpolation with anatomical guide

from scipy.ndimage import gaussian_filter
import matplotlib.colors as mcolors
import copy
import os
import cv2
import imageio
import numpy as np
import pandas as pd
import pickle
# import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import *
mpl.rcParams['pdf.fonttype'] = 42
import seaborn as sns
# import scipy.stats
import tqdm
from scipy.stats.mstats import mquantiles
import torch
from numba import njit
import numpy as np
from scipy import ndimage

### adapted from Colas Droin

from numba import njit

@njit
def fill_array_interpolation(
    array_annotation,
    array_slices,
    divider_radius=5,
    annot_inside=-0.01,
    limit_value_inside=-2,
    structure_guided=True,
):
    """This function is used to fill the empty space (unassigned voxels) between the slices with
    interpolated values.

    Args:
        array_annotation (np.ndarray): Three-dimensional array of annotation coming from the Allen
            Brain Atlas.
        array_slices (np.ndarray): Three-dimensional array containing the lipid intensity values
            from the MALDI experiments (with many unassigned voxels).
        divider_radius (int, optional): Divides the radius of the region used for interpolation
            (the bigger, the lower the number of voxels used). Defaults to 5.
        annot_inside (float, optional): Value used to denotate the inside of the brain. Defaults
            to -0.01.
        limit_value_inside (float, optional): Alternative to annot_inside. Values above
            limit_value_inside are considered inside the brain. Defaults to -2.
        structure_guided (bool, optional): If True, the interpolation is done using the annotated
            structures. If False, the interpolation is done blindly.

    Returns:
        (np.ndarray): A three-dimensional array containing the interpolated lipid intensity values.
    """
    array_interpolated = np.copy(array_slices)

    for x in range(8, array_annotation.shape[0]): 
        for y in range(0, array_annotation.shape[1]):
            for z in range(0, array_annotation.shape[2]):
                # if we are in a unfilled region of the brain or just inside the brain
                condition_fulfilled = False
                if array_slices[x, y, z] >= 0:
                    condition_fulfilled = True
                elif limit_value_inside is not None and not condition_fulfilled:
                    if array_annotation[x, y, z] > limit_value_inside:
                        condition_fulfilled = True
                elif (
                    np.abs(array_slices[x, y, z] - annot_inside) < 10**-4
                ) and not condition_fulfilled:
                    condition_fulfilled = True
                if condition_fulfilled:
                    # check all datapoints in the same structure, and do a distance-weighted average
                    value_voxel = 0
                    sum_weights = 0
                    size_radius = int(array_annotation.shape[0] / divider_radius)
                    for xt in range(
                        max(0, x - size_radius), min(array_annotation.shape[0], x + size_radius + 1)
                    ):
                        for yt in range(
                            max(0, y - size_radius),
                            min(array_annotation.shape[1], y + size_radius + 1),
                        ):
                            for zt in range(
                                max(0, z - size_radius),
                                min(array_annotation.shape[2], z + size_radius + 1),
                            ):
                                # if we are inside of the sphere of radius size_radius
                                if (
                                    np.sqrt((x - xt) ** 2 + (y - yt) ** 2 + (z - zt) ** 2)
                                    <= size_radius
                                ):
                                    # the voxel has data
                                    if array_slices[xt, yt, zt] >= 0:
                                        # the structure is identical
                                        if (
                                            structure_guided
                                            and np.abs(
                                                array_annotation[x, y, z]
                                                - array_annotation[xt, yt, zt]
                                            )
                                            < 10**-4
                                        ) or not structure_guided:
                                            d = np.sqrt(
                                                (x - xt) ** 2 + (y - yt) ** 2 + (z - zt) ** 2
                                            )
                                            value_voxel += np.exp(-d) * array_slices[xt, yt, zt]
                                            sum_weights += np.exp(-d)
                    if sum_weights == 0:
                        pass
                    else:
                        value_voxel = value_voxel / sum_weights
                        array_interpolated[x, y, z] = value_voxel

    return array_interpolated

def normalize_to_255(a):
    low_percentile_val = np.nanpercentile(a, 10)

    low_value_mask = a < low_percentile_val
    nan_mask = np.isnan(a)
    mask = np.logical_or(low_value_mask, nan_mask)

    a = np.where(mask, 0, a)

    a = ((a - a.min()) * (255 - 0) / (a.max() - a.min()))

    a[mask] = np.nan

    if not np.isnan(a).any():
        a = a.astype(np.uint8)

    return a

atlas = # load a 3D numpy array reference volume and its 
annotation_image  # load a 3D numpy array reference set of anatomical annotations

# example:
#from bg_atlasapi import BrainGlobeAtlas
#BrainGlobeAtlas("allen_mouse_100um")
#reference_image = atlas.reference
#print(reference_image.shape)
# annotation image
#annotation_image = atlas.annotation
#print(annotation_image.shape)

from tqdm import tqdm
import os
from collections import defaultdict

lipids = # names of the features we want to reconstruct in 3D
# pandas dataframe of processed pixels extracted from the adata object
pixels=pixels.dropna()

for xxx in tqdm(lipids):
    
    try:

        lipid_to_interpolate = pixels[["xccf", "yccf", "zccf", xxx]] 

        # logging the data seems to help rendering beautiful 3Ds
        lipid_to_interpolate.loc[:, xxx] = np.log(lipid_to_interpolate.loc[:, xxx])

        # prepare the data
        lipid_to_interpolate["xccf"] = lipid_to_interpolate["xccf"]*10
        lipid_to_interpolate["yccf"] = lipid_to_interpolate["yccf"]*10
        lipid_to_interpolate["zccf"] = lipid_to_interpolate["zccf"]*10

        tensor_shape = reference_image.shape

        tensor = np.full(tensor_shape, np.nan)

        intensity_col_name = lipid_to_interpolate.columns[3]

        intensity_values = defaultdict(list)

        for _, row in lipid_to_interpolate.iterrows():
            x, y, z = int(row["xccf"]) - 1, int(row["yccf"]) - 1, int(row["zccf"]) - 1
            intensity_values[(x, y, z)].append(row[intensity_col_name])

        for coords, values in intensity_values.items():
            x, y, z = coords
            if 0 <= x < tensor_shape[0] and 0 <= y < tensor_shape[1] and 0 <= z < tensor_shape[2]:
                tensor[x, y, z] = np.nanmean(values)

        not_nan_mask = np.logical_not(np.isnan(tensor))

        indices = np.where(np.any(not_nan_mask, axis=(1, 2)))

        normalized_tensor = normalize_to_255(tensor)

        w = 5

        non_nan_mask = ~np.isnan(normalized_tensor)
        normalized_tensor[non_nan_mask & (normalized_tensor < w)] = np.nan
        normalized_tensor[reference_image < 4] = 0

        # interpolate
        ahaha = fill_array_interpolation(array_annotation = annotation_image, array_slices = normalized_tensor)########, structure_guided=False)

        # clean up by convolution
        k = 10  # kernel size
        kernel = np.ones((k,k,k))
        array = np.where(np.isnan(ahaha), 0, ahaha)

        counts = np.where(np.isnan(ahaha), 0, 1)
        counts = ndimage.convolve(counts, kernel, mode='constant', cval=0.0)

        convolved = ndimage.convolve(array, kernel, mode='constant', cval=0.0)

        avg = np.where(counts > 0, convolved / counts, np.nan)

        filled_ahaha = np.where(np.isnan(ahaha), avg, ahaha)

        np.save(os.getcwd()+"/3d_interpolated_native/"+xxx+'interpolation_log.npy', filled_ahaha)
    
    except:
        continue




##################################################################################################################
# BLOCK 3: TRAIN A VARIATIONAL AUTOENCODER TO EXTRACT ONTOLOGICALLY RELEVANT REPRESENTATIONS (LIPID PROGRAMS) -> function train_lipimap
##################################################################################################################

# train_lipimap will be implemented in the future, maintain a placeholder here for now




##################################################################################################################
# BLOCK 4: REGISTER OTHER OMIC MODALITIES ON THE SAME PIXELS -> function add_modality
##################################################################################################################

# the user passes a second adata object whose indexes in part overlap with the indexes of the core spatial lipidomics anndata object handled by this package. the user also specifies the name of the modality that is added (eg "gene expression") and if it is categorical or continuous-valued (such as cell types vs connectomic streams)
# the function adds a slot for the new modality using the name provided by the user and distinguishing whether it is categorical or continuous


##################################################################################################################
# BLOCK 5: COMPARE PARCELLATIONS DEFINED BY LIPIDS AND OTHER MODALITIES -> function compare_parcellations
##################################################################################################################
import scipy.cluster.hierarchy as sch

# lipizoneZ and celltypeZ in the example, to be given more general names, are two parallel arrays (i.e., aligned!) with two different parcellizations
# which can be extracted as two categorical slots, one defaulting to the lipizones / lipid-derived clusters, of the anndata object

cmat = pd.crosstab(lipizoneZ, celltypeZ)
substrings = # a list passed by the user of parcellation entities to omit from the comparison
rows_to_keep = ~cmat.index.to_series().str.contains('|'.join(substrings), case=False, na=False)
cols_to_keep = ~cmat.columns.to_series().str.contains('|'.join(substrings), case=False, na=False)

# also, require at least N (default = 40) unique entries per category before proceeding

cmat = cmat.loc[:, cols_to_keep]
normalized_df = cmat / cmat.sum() # fraction 
normalized_df = (normalized_df.T / normalized_df.T.mean()).T ## switch to enrichments
normalized_df1 = normalized_df.copy()

cmat = pd.crosstab(lipizoneZ, celltypeZ).T
substrings = # a list passed by the user of parcellation entities to omit from the comparison
rows_to_keep = ~cmat.index.to_series().str.contains('|'.join(substrings), case=False, na=False)
cols_to_keep = ~cmat.columns.to_series().str.contains('|'.join(substrings), case=False, na=False)

# also, require at least N (default = 40) unique entries per category before proceeding
cmat = cmat.loc[rows_to_keep, :]
normalized_df = cmat / cmat.sum() # fraction 
normalized_df = (normalized_df.T / normalized_df.T.mean()) ## switch to enrichments
normalized_df2 = normalized_df.copy()

normalized_df = normalized_df2 * normalized_df1
normalized_df[cmat.T < 20] = 0
normalized_df = normalized_df.loc[:, normalized_df.sum() > M] # M defaults to 200

linkage = sch.linkage(sch.distance.pdist(normalized_df.T), method='weighted', optimal_ordering=True)
order = sch.leaves_list(linkage)
normalized_df = normalized_df.iloc[:, order]

order = np.argmax(normalized_df.values, axis=1)
order = np.argsort(order)
normalized_df = normalized_df.iloc[order,:]

# normalized_df should be saved to file and stored as a property in the adata object

##################################################################################################################
# BLOCK 6: RUN A MULTIOMICS FACTOR ANALYSIS (MOFA) -> function run_mofa
##################################################################################################################

# in this example it's genes and lipids but we should be more general than that
genes = # pandas dataframe pixel x modality1
lipids = # pandas dataframe pixel x modality2 (same index as genes dataframe)

# downsample or MOFA will take 4ever on CPUs
gexpr = genes[::N]

# remove zero-variance features
variance = genes.var()
zero_var_columns = variance[variance < 0.0001].index
print(f"Columns to remove (zero variance): {list(zero_var_columns)}")
genes = genes.drop(columns=zero_var_columns)
lipids = lipids.loc[genes.index,:]

# run a MOFA

import pandas as pd
import numpy as np
from mofapy2.run.entry_point import entry_point

data = [
    [genes],
    [lipids]
]

# Create MOFA object
ent = entry_point()

# Set data options
ent.set_data_options(scale_groups=True, scale_views=True)
ent.set_data_matrix(
    data,
    likelihoods=["gaussian", "gaussian"],  
    views_names=["gene_expression", "lipid_profiles"],
    samples_names=[gexpr.index.tolist()]  # same samples across views
)

# Set model options
ent.set_model_options(
    factors=100,  # number of factors to learn ########################
    spikeslab_weights=True,  # spike and slab sparsity on weights
    ard_weights=True  # Automatic Relevance Determination on weights
)

# Set training options
ent.set_train_options(
    iter=10,######################## tunable
    convergence_mode="fast",
    startELBO=1,
    freqELBO=1,
    dropR2=0.001,
    verbose=True
)

# Build and run the model
ent.build()
ent.run()

# get the model output, extract factors and weights
model = ent.model
expectations = model.getExpectations()
factors = expectations["Z"]["E"] 
weights = [w["E"] for w in expectations["W"]]  

# extract the coordinates of single cells in factors embeddings
factors_df = pd.DataFrame(
    factors,
    index=genes.index,
    columns=[f"Factor_{i+1}" for i in range(factors.shape[1])]
)

# extract the contribution of each gene and lipid to each factor, and the top markers of each factor for both modalities
weights_gene = pd.DataFrame(
    weights[0],  # first view (genes)
    index=genes.columns,
    columns=factors_df.columns
)
weights_lipid = pd.DataFrame(
    weights[1],  # second view (lipids)
    index=lipids.columns,
    columns=factors_df.columns
)

factors_df.to_csv("minimofa_factors.csv")
weights_gene.to_csv("minimofa_weights_genes.csv")
weights_lipid.to_csv("minimofa_weights_lipids.csv")
factors_df.to_hdf("factors_dfMOFA.h5ad", key="table")

# do t-SNE on top of MOFA
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

embds = factors_df

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

init_train = x_train[:,[0, 1]] # initialize with two factors, note this affects results a bit

embedding_train = TSNEEmbedding(
    init_train,
    affinities_train,
    negative_gradient_method="fft",
    n_jobs=8,
    verbose=True,
)

embedding_train_1 = embedding_train.optimize(n_iter=500, exaggeration=1.2)
embedding_train_N = embedding_train_1.optimize(n_iter=100, exaggeration=2.5)
np.save("minimofageneslipidsembedding_train_N.npy", np.array(embedding_train_N))

# store the MOFA factors as an additional slot of the adata object. also store the weights as properties of the features of the adata object and the tSNE in slot X_MOFA_TSNE


##################################################################################################################
# BLOCK 7: NEIGHBORHOOD ANALYSIS, CALCULATES FREQUENCY OF CLUSTERS SURROUNDING A GIVEN CLUSTER -> function neighborhood
##################################################################################################################


metadata['SectionID'] = metadata['SectionID'].astype(int)
metadata['x'] = metadata['x'].astype(int)
metadata['y'] = metadata['y'].astype(int)

metadata['neighbors'] = [[] for _ in range(len(metadata))]

for section_id, group in tqdm(metadata.groupby('SectionID')):
    coord_set = set(zip(group['x'], group['y']))
    
    for idx, row in group.iterrows():
        x0, y0 = row['x'], row['y']
        
        neighbor_coords = [
            (x0 - 1, y0 - 1), (x0 - 1, y0), (x0 - 1, y0 + 1),
            (x0,     y0 - 1),               (x0,     y0 + 1),
            (x0 + 1, y0 - 1), (x0 + 1, y0), (x0 + 1, y0 + 1),
        ]
        
        existing_neighbors = [
            f'section{section_id}_pixel{nx}_{ny}'
            for nx, ny in neighbor_coords
            if (nx, ny) in coord_set
        ]
        
        metadata.at[idx, 'neighbors'] = existing_neighbors

metadata['idd'] = metadata.apply(
    lambda row: f'section{row.SectionID}_pixel{row.x}_{row.y}', axis=1
)
id_to_lipizone = pd.Series(metadata.lipizone_names.values, index=metadata.idd).to_dict()

def map_neighbors_to_names(neighbors):
    return [id_to_lipizone.get(neighbor_id, None) for neighbor_id in neighbors]

metadata['neighbor_names'] = metadata['neighbors'].apply(map_neighbors_to_names)

metadata['class'] = tree.loc[metadata.index, 'class']

id_to_lipizone = pd.Series(metadata['class'].values, index=metadata.idd).to_dict()

def map_neighbors_to_names(neighbors):
    return [id_to_lipizone.get(neighbor_id, None) for neighbor_id in neighbors]

metadata['neighbor_classnames'] = metadata['neighbors'].apply(map_neighbors_to_names)
metadata['neighbor_classnames']

from collections import Counter

grouped = metadata.groupby('lipizone_names')['neighbor_classnames'].apply(lambda lists: [neighbor for sublist in lists for neighbor in sublist])

def calculate_proportions(neighbor_list):
    total = len(neighbor_list)
    counts = Counter(neighbor_list)
    proportions = {classname: count / total for classname, count in counts.items()}
    return proportions

proportion_df = grouped.apply(calculate_proportions).reset_index()
proportion_expanded = proportion_df.set_index('lipizone_names')['neighbor_classnames'].apply(pd.Series).fillna(0)
proportion_expanded



##################################################################################################################
# BLOCK 8: MAKE A UMAP OF A SUBSET OF USER-DEFINED LIPIDS USING OBSERVATIONS AS FEATURES -> function umap_molecules
##################################################################################################################


import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap.umap_ as umap

reducer = umap.UMAP(n_neighbors=5, min_dist=0.05, n_jobs=1)
centroidsmolecules #molecules should be a pandas dataframe extracted from the adata object with pixels as rows and for columns the subset of user-defined molecules (defaulting to all molecules available in the object otherwise, across modalities). centroidsmolecules is another pandas dataframe, the lipizone-wise average of molecules
umap_result = reducer.fit_transform(centroidsmolecules.T) 

df = pd.DataFrame(wm.columns)
df.columns = ["lipid_name"]

# extract the "class" etc from the lipid_name
df["class"] = df["lipid_name"].apply(lambda x: re.split(' |\(', x)[0])
df["carbons"] = df["lipid_name"].apply(lambda x: int(re.search(r'(\d+):', x).group(1)) if re.search(r'(\d+):', x) else np.nan)
df["insaturations"] = df["lipid_name"].apply(lambda x: int(re.search(r':(\d+)', x).group(1)) if re.search(r':(\d+)', x) else np.nan)
df["insaturations_per_Catom"] = df["insaturations"] / df["carbons"]

df["broken"] = df["lipid_name"].str.endswith('_uncertain')
df.loc[df["broken"], 'carbons'] = np.nan
df.loc[df["broken"], 'class'] = np.nan
df.loc[df["broken"], 'insaturations'] = np.nan
df.loc[df["broken"], 'insaturations_per_Catom'] = np.nan

colors = pd.read_hdf("lipidclasscolors.h5ad", key="table")
df['color'] = df['class'].map(colors['classcolors'])
df.loc[df["broken"], 'color'] = "gray"

df.index = df['lipid_name']
df = df.drop_duplicates()
df = df.fillna("gray")

from adjustText import adjust_text
fig, ax = plt.subplots(figsize=(14, 10))

scatter = ax.scatter(umap_result[:, 0], umap_result[:, 1],
                     c=df['color'], edgecolor='w', linewidth=0.5)
texts = []

for i, txt in enumerate(df['lipid_name']):
    texts.append(ax.text(umap_result[i, 0], umap_result[i, 1], txt,
                         fontsize=10, alpha=0.9))

adjust_text(texts, 
            ax=ax,
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5),
            expand_points=(1.2, 1.4),
            force_points=0.2,
            force_text=0.2,
            lim=1000)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.tight_layout()
plt.savefig(# name+".pdf")
plt.show()
# also store this UMAP as a feature property for the features in the anndata object



##################################################################################################################
# BLOCK 8: MAKE A UMAP OF LIPIZONES -> function umap_lipizones
##################################################################################################################


centroids = # get the lipizone centroids (in lipid space)

import umap.umap_ as umap

reducer = umap.UMAP(n_neighbors=4, min_dist=0.05, n_jobs=1)
umap_result = reducer.fit_transform(centroids)

# store it as usual


##################################################################################################################
# BLOCK 9: FIND THE SPATIAL MODULES USING ANATOMICAL COLOCALIZATION -> function spatial_modules
##################################################################################################################

from matplotlib.backends.backend_pdf import PdfPages

# let the user select positions of interest, data is the relevant info from the anndata object
focus = data[data["Section"].isin(SELECTED_SECTIONS)]
# or let the user select anatomical acronyms of interest if available
focus = focus[focus['acronym'].str.endswith(LA, na=False) | focus['acronym'].str.endswith(LB, na=False)]

# keep only abundant lipizones
unique_colors = focus["lipizone_color"].value_counts().index[focus["lipizone_color"].value_counts() > 100]
focus = focus.loc[focus['lipizone_color'].isin(unique_colors),:]
# find clusters of colocalizing lipizones (organizational archetypes)
cmat = pd.crosstab(focus['acronym'], focus['lipizone_color'])
normalized_df1 = cmat / cmat.sum() # fraction 
normalized_df1 = (normalized_df1.T / normalized_df1.T.mean()).T
cmat = pd.crosstab(focus['acronym'], focus['lipizone_color']).T
normalized_df2 = cmat / cmat.sum() # fraction 
normalized_df2 = (normalized_df2.T / normalized_df2.T.mean())
normalized_df = normalized_df1 * normalized_df2
tc = normalized_df.T
adata = anndata.AnnData(X=tc)
sc.pp.neighbors(adata, use_rep='X')
sc.tl.leiden(adata, resolution=2.0)
cluster_labels = adata.obs['leiden']
# plot in groups to eyeball patterns
color_to_cluster = pd.Series(cluster_labels.values, index=cluster_labels.index).to_dict()
focus['leiden_cluster'] = focus['lipizone_color'].map(color_to_cluster)
unique_clusters = sorted(focus['leiden_cluster'].unique())
sections = focus["Section"].unique()

# plot the modules 1 by 1 if desired by the user
for cluster_idx, cluster in enumerate(unique_clusters):
    # Create a new figure for each cluster
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    cluster_colors = focus[focus['leiden_cluster'] == cluster]['lipizone_color'].unique()

    for section_idx, section_value in enumerate(sections):
        section = focus[focus["Section"] == section_value]
        for color in cluster_colors:
            color_section = section[section['lipizone_color'] == color]
            axs[section_idx].scatter(
                color_section['z_index'], 
                -color_section['y_index'],
                c=color, 
                s=10,
                alpha=1, 
                zorder=1, 
                rasterized=True
            )
        filtered_section_contour = section.loc[section['boundary'] == 1, :]
        axs[section_idx].scatter(
            filtered_section_contour['z_index'], 
            -filtered_section_contour['y_index'],
            c='black', 
            s=2, 
            rasterized=True, 
            zorder=2, 
            alpha=0.9
        )
        axs[section_idx].set_aspect('equal')
        axs[section_idx].axis('off')
        colors_str = ', '.join(cluster_colors)

    plt.tight_layout()
    plt.suptitle(cluster)
    plt.show()