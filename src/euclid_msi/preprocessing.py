##################################################################################################################
# BLOCK 1: CALCULATE AND STORE MORAN'S I FOR EACH FEATURE FOR EACH SECTION -> FUNCTION calculate_moran
##################################################################################################################

import zarr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import tqdm
import pandas as pd
import scanpy as sc
import squidpy as sq
import warnings
import datetime
warnings.filterwarnings('ignore')

# the path where uMAIA outputs are saved
PATH_DATA = '/data/luca/lipidatlas/uMAIA_allbrains/021124_ALLBRAINS_normalised.zarr'
root = zarr.open(PATH_DATA, mode='rb')
PATH_MZ = np.sort(list(root.group_keys()))


# list of acquisitions
acquisitions=[
    'BrainAtlas/BRAIN2/20211201_MouseBrain2_S11_306x248_Att30_25um',
 'BrainAtlas/BRAIN2/20211202_MouseBrain2_S12_332x246_Att30_25um',
]

# load the uMAIA tissue masks
masks = [np.load(f'/data/LBA_DATA/{section}/mask.npy') for section in acquisitions]
features = np.sort(list(root.group_keys()))

# calculate for each section = acquisition Moran's I for each lipid
N_acquisitions = len(acquisitions)
morans_by_sec = pd.DataFrame(np.zeros((len(features), N_acquisitions)), index = features)                             

with open("iterations_log.txt", "a") as file:

    for xxx, feat in tqdm(enumerate(features)): # lip

        for j in range(N_acquisitions): # acq

            MASK = masks[int(j)]
            image = root[feat][str(j)][:]

            coords = np.column_stack(np.where(MASK))
            X = image[coords[:, 0], coords[:, 1]]
            adata = sc.AnnData(X=pd.DataFrame(X))
            adata.obsm['spatial'] = coords

            sq.gr.spatial_neighbors(adata, coord_type='grid')
            sq.gr.spatial_autocorr(adata, mode='moran')

            morans_by_sec.loc[feat,str(j)]  = adata.uns['moranI']['I'].values

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Iteration {j + 1}, Time: {current_time}\n")

morans_by_sec.to_csv("morans_by_sec.csv")




##################################################################################################################
# BLOCK 2: STORE DATA IN AN ANNDATA STRUCTURE -> FUNCTION store_exp_data_metadata
##################################################################################################################

datasetsize = N_acquisitions
lipid_native_sections_array = np.full((len(PATH_MZ), datasetsize, 500, 500), np.nan)

for LIPID in tqdm(range(len(PATH_MZ))):
    for SECTION in range(datasetsize):
        img = root[PATH_MZ[LIPID]][SECTION][:]
        img_x, img_y = img.shape
        lipid_native_sections_array[LIPID, SECTION, :img_x, :img_y] = img

print(lipid_native_sections_array.shape)

import pandas as pd

# preparing a pixel x lipid dataframe
lipid_tensor = lipid_native_sections_array

# flatten the tensor
flattened_lipid_tensor = lipid_tensor.reshape(lipid_tensor.shape[0], -1)

# generate temporary lipid names
lipid_names = ["lipid" + str(i+1) for i in range(flattened_lipid_tensor.shape[0])]

flattened_lipid_tensor

# generate pixel names that retain their spatial position as unique identifier
column_names = []
for i in range(lipid_tensor.shape[1]):
    for j in range(lipid_tensor.shape[2]):
        for k in range(lipid_tensor.shape[3]):
            column_names.append(f"section{i+1}_pixel{j+1}_{k+1}")

df = pd.DataFrame(flattened_lipid_tensor, index=lipid_names, columns=column_names)

# removing out-of-brain pixels
df_transposed = df.T
df_transposed = df_transposed.dropna(how='all')

# name columns with peak m/z i.e. features
df_transposed.columns = PATH_MZ

# extract native spatial coordinates
df_index = df_transposed.index.to_series().str.split('_', expand=True)
df_index.columns = ['Section', 'x', 'y']
df_index['Section'] = df_index['Section'].str.replace('section', '')
df_index['x'] = df_index['x'].str.split('pixel').str.get(1)
df_index = df_index.astype(int)
df_transposed = df_transposed.join(df_index)
pixels = df_transposed

import pandas as pd
import zarr
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# exponentiate
Nlipids = len(PATH_MZ)
pixels.iloc[:,:Nlipids] = np.exp(pixels.iloc[:,:Nlipids])

# add metadata on the acquisitions
ind = pixels.index

# pass a metadata CSV file. must have a column "SectionID" to merge on section-wise metadata
metadata = pd.read_csv("acquisitions_metadata.csv")
cols = np.array(pixels.columns)
cols[-3] = "SectionID"
pixels.columns = cols
pixels.index = ind
pixels = pixels.merge(metadata, left_on='SectionID', right_on='SectionID', how='left')
pixels.index = ind

# clean up from background pixels
mask = (pixels.iloc[:, :len(PATH_MZ)] < 0.0001).all(axis=1)
pixels = pixels[~mask]

# concatenate pixel-wise metadata, such as warped coordinates or anatomical regions (eg from the Allen)
## read in parquet file with same-style index identifiers and as much overlap as possible with the entries of "pixels"
## concatenate
## if there is a column called anatomycolor eg anatomycolor = "allencolor", remove the rows having value "#000000" to further polish out-of-tissue pixels

# store also the Moran's for each feature we computed before

# prepare a scanpy-style anndata object to store all our data conveniently



##################################################################################################################
# BLOCK 3: ANNOTATE m/z PEAKS WITH A LIPID NAME -> function annotate_molecules
##################################################################################################################

def map_hmdb_ids(hmdb_ids):
    return np.unique([conversion_dict.get(hmdb_id.strip(), '') for hmdb_id in hmdb_ids.split(',')])

# preprocess LIPIDMAPS
# load SDF file
supplier = Chem.SDMolSupplier('structures.sdf')

# load the first molecule from the supplier
first_molecule = next(iter(supplier))

if first_molecule is not None:
    # get the names of all keys
    keys = first_molecule.GetPropNames()
    print(list(keys))

# initialize empty lists to store data
lm_id_list = []
name_list = []
systematic_name_list = []
category_list = []
main_class_list = []
mass_list = []
abbreviation_list = []
ik_list = []

# iterate over the molecules in the SDF file
for molecule in tqdm(supplier):
    if molecule is not None:
        if molecule.HasProp('LM_ID'):
            lm_id_list.append(molecule.GetProp('LM_ID'))
        else:
            lm_id_list.append(None)
        
        if molecule.HasProp('NAME'):
            name_list.append(molecule.GetProp('NAME'))
        else:
            name_list.append(None)
        
        if molecule.HasProp('SYSTEMATIC_NAME'):
            systematic_name_list.append(molecule.GetProp('SYSTEMATIC_NAME'))
        else:
            systematic_name_list.append(None)
        
        if molecule.HasProp('CATEGORY'):
            category_list.append(molecule.GetProp('CATEGORY'))
        else:
            category_list.append(None)
        
        if molecule.HasProp('MAIN_CLASS'):
            main_class_list.append(molecule.GetProp('MAIN_CLASS'))
        else:
            main_class_list.append(None)
            
        if molecule.HasProp('EXACT_MASS'):
            mass_list.append(molecule.GetProp('EXACT_MASS'))
        else:
            mass_list.append(None)
            
        if molecule.HasProp('ABBREVIATION'):
            abbreviation_list.append(molecule.GetProp('ABBREVIATION'))
        else:
            abbreviation_list.append(None)
            
        if molecule.HasProp('INCHI_KEY'):
            ik_list.append(molecule.GetProp('INCHI_KEY'))
        else:
            ik_list.append(None)

data = {
    'LM_ID': lm_id_list,
    'NAME': name_list,
    'SYSTEMATIC_NAME': systematic_name_list,
    'CATEGORY': category_list,
    'MAIN_CLASS': main_class_list,
    'EXACT_MASS': mass_list,
    'ABBREVIATION': abbreviation_list,
    'INCHY_KEY': ik_list
}

df = pd.DataFrame(data)

lipidmaps = df
lipidmaps_hmdb = lipidmaps.copy()

# prepare using HMDB to match METASPACE and LIPIDMAPS naming conventions

hmdb = pd.read_csv("HMDB_complete.csv", index_col=0)
merged_df = pd.merge(lipidmaps_hmdb, hmdb, left_on='INCHY_KEY', right_on='InchiKey', how='left')

conversionhmdb = merged_df[['DBID', 'ABBREVIATION']].dropna()


## upload a csv file containing the columns:
# PATH_MZ = m/z peak, 6 decimals
# Annotation = the name you want to assign to that peak. for example, from METASPACE. better if paired LC-MS data
# Score = your confidence that this annotation is reasonable
# as an alternative, upload a csv file containing m/z and molecule names. reasonably matched peaks will be automatically annotated:
ppm = 5 # desired ppm to validate an annotation
reference_mz = 800 # scale of our dataset
distance_ab5ppm = ppm / 1e6 * reference_mz

def find_matching_lipids(path_mz, lipid_mz_df):
    lower_bound = path_mz - ppm / 1e6 * path_mz
    upper_bound = path_mz + ppm / 1e6 * path_mz
    matching_lipids = lipid_mz_df[(lipid_mz_df['m/z'] >= lower_bound) & (lipid_mz_df['m/z'] <= upper_bound)]['Lipids']
    return ', '.join(matching_lipids)
# append a "_db" string to feature names that are below the Score threshold passed to the function or come from LIPID MAPS but are not in the user's csv

## TO BE MADE MORE GENERAL
# in case you have access to paired LC-MS data, use molar fractions to disambiguate dubious annotations
quantlcms = pd.read_csv("QuantitativeLCMS.csv", index_col=0)
atlas = quantlcms[['Male',	'Male.1',	'Male.2',	'Male.3']] # use the males as the "reference atlas" we are going to use mostly is a male
ref = atlas.iloc[1:,:].astype(float).mean(axis=1)
annots = pd.read_csv("goslin_output.tsv",sep='\t')
convt = annots[['Original Name', 'Species Name']]
convt.index = convt['Original Name'].astype(str)
refvals = pd.DataFrame(ref.values, index = ref.index, columns=["nmol_fraction_LCMS"])
refvals.index = refvals.index.str.replace('Hex1Cer', 'HexCer')
tmp = pd.read_csv("manuallyannotated_addlcms.csv", index_col=0).dropna()
refvalstmp = refvals.loc[refvals.index.isin(tmp.iloc[:,0]),:]
rvl = np.array(refvals.index)
convl = np.array(convt.index)
annots.index = annots['Original Name']
annots = annots.loc[np.intersect1d(rvl, convl),:]
refvals = refvals.loc[np.intersect1d(rvl, convl),:]
indivannots = annots[['Species Name']]
indivannots = indivannots.groupby('Original Name').first()
refvals['Species Name'] = refvals.index.map(indivannots['Species Name'])
tmp.index = tmp.iloc[:,0]
tmp = tmp.loc[refvalstmp.index,:]
refvalstmp['Species Name'] = tmp['Unnamed: 2']
quantlcms = pd.concat([refvals, refvalstmp], axis=0)
quantlcms.index = quantlcms['Species Name']
quantlcms = quantlcms[['nmol_fraction_LCMS']]
quantlcms = pd.DataFrame(quantlcms['nmol_fraction_LCMS'].groupby(quantlcms.index).sum()) # merge lipids that are distinguished in LCMS but undistinguishable in IMS
THRESHOLD = 0.8 # a lipid to be prioritized should be at least 80% molar fraction
pdf['AnnotationLCMSPrioritized'] = pdf['Annotation']
for i, annot in enumerate(pdf['Annotation']):
    if isinstance(annot, str):
        annot = [annot]
        pdf['Annotation'].iloc[i] = annot
    now = quantlcms.loc[quantlcms.index.isin(annot),:]
    now = now/now.sum()
    if now['nmol_fraction_LCMS'].max() > THRESHOLD:
        pdf['AnnotationLCMSPrioritized'].iloc[i] = now.index[now['nmol_fraction_LCMS'] > THRESHOLD].values[0]
pdf['AnnotationLCMSPrioritized']


# rename the features in the anndata file using the annotations, while storing also the original feature names

##################################################################################################################
# BLOCK 4: SAVE AND RELOAD ANNDATA -> functions save_msi_dataset, load_msi_dataset
##################################################################################################################

# implement save and load functions


##################################################################################################################
# BLOCK 5: PRIORITIZE ADDUCT -> function prioritize_adduct
##################################################################################################################
# we assume here that isotopes have already been filtered out using uMAIA
# use total signal to prioritize, while raising warnings in case of inconsistencies
annotation_to_mz = # create from the annotations dataframe above a dictionary that for each unique Annotation (that was found in the dataset) stores a LIST of peak m/z values
totsig_df = pd.DataFrame(np.zeros((len(features), N_acquisitions)), index=features, columns = np.arange(0, len(acquisitions)).astype(str))
for xxx, feat in tqdm(enumerate(features)): 
    feat_dec = f"{float(feat):.6f}"
    ns = np.array(list(root[feat_dec].keys())).astype(int).astype(str)
    
    for nnn in ns:

        MASK = masks[int(nnn)]
        image = root[feat_dec][nnn][:]

        image[MASK == 0] = 0
        sig = np.mean(image*1e6)
        
        totsig_df.loc[feat, nnn] = sig
        
totsig_df = totsig_df.fillna(0)

featuresum = totsig_df.sum(axis=1)

annotation_to_mz_bestisobar = {}

for annotation, mz_values in annotation_to_mz.items():
    max_featuresum = -float('inf') 
    best_mz = None  
    
    for mz_value in mz_values:
        if mz_value in featuresum.index:
            featuresum_value = featuresum.loc[mz_value]
            if featuresum_value > max_featuresum:
                max_featuresum = featuresum_value
                best_mz = mz_value
        else:
            print(f"m/z value {mz_value} not found in featuresum index.")
            
    if best_mz is not None:
        annotation_to_mz_bestisobar[annotation] = best_mz
    else:
        print(f"No valid m/z values found for annotation {annotation}.")
        
# keep track for each feature whether it is to keep (across m/z peaks with that same name, it has better total signal) or not



##################################################################################################################
# BLOCK 6: FEATURE SELECTION -> function feature_selection
##################################################################################################################

### feature selection 1: threshold on total signal as calculated above (if desired)

### feature selection 2: at least N sections reaching Moran's I > threshold (default threshold = 0.4, default number of sections = 1/10 of total number of acquisitions)

### feature selection 3: combination of scores:

# decide to assess quality metrics on a subset of "reference" sections, such as the first N decided by the user
# moran is the pandas dataframe feature x section that we computed before
moran = moran.iloc[:,:N]

# use a score on variance of section variances and averages
# standardize the data
inputlips = data.iloc[:,:-23]
inputlips[inputlips > 1.] = 0.0001 ### broken values

scaler = StandardScaler()
scaled_data = scaler.fit_transform(inputlips)

inputlips = pd.DataFrame(scaled_data, columns=inputlips.columns, index=inputlips.index)

# use a function to evaluate the variances and the means of section-wise variances
adata = sc.AnnData(X=inputlips)
adata.obsm['spatial'] = data[['zccf', 'yccf', 'Section']].loc[data.index,:].values
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

    combined_score = -var_of_vars/2 + mean_of_vars

    return var_of_vars, mean_of_vars, combined_score
    
var_of_vars, mean_of_vars, combined_score = rank_features_by_combined_score(adata)
ranked_indices = np.argsort(combined_score)[::-1]

scores = pd.DataFrame([np.array(inputlips.columns)[ranked_indices], var_of_vars[ranked_indices], mean_of_vars[ranked_indices], combined_score[ranked_indices]]).T
scores.columns = ["spatvar", "var_of_vars", "mean_of_vars", "combined_score"]
moran_sorted = moran.mean(axis=1).sort_values()[::-1]
scores.index = scores['spatvar'].astype(float).astype(str)
scores = scores.loc[moran_sorted.index.astype(str),:]
scores['combined_score'][scores['combined_score'] < -5] = -5 # bad is bad, control outliers
scores.index = scores.index.astype(float).astype(str)

# a very permissive threshold on Moran's I
scores_good_moran = scores.loc[moran_sorted.index[moran_sorted > 0.4].astype(float).astype(str),:]
scores = scores_good_moran
# a permissive filter over section-wise dropout: too many dropouts => lipids should be excluded for clustering and reimputed
peakmeans = data.iloc[:,:1400].groupby(data['Section']).mean()
missinglipid = np.sum(peakmeans < 0.00015).sort_values()
dropout_acceptable_lipids = missinglipid.loc[missinglipid < 4].index.astype(float).astype(str)
scores = scores.loc[scores.index.isin(dropout_acceptable_lipids),:]

# cluster the space of scores
moran_sorted.index = moran_sorted.index.astype(float).astype(str)
scores['moran'] = moran_sorted.loc[scores.index.astype(float).astype(str)]
missinglipid.index = missinglipid.index.astype(float).astype(str)
scores['missinglipid'] = missinglipid.loc[scores.index.astype(float).astype(str)]
scores = scores.loc[scores['combined_score'] > 0,:]
X = scores[['var_of_vars',	'combined_score',	'moran',	'missinglipid']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# save peak m/z and cluster numbers for each peak to a csv file and plot the clustering annotating the cluster names so the user can then manually decide which clusters to keep and re-call the function for feature selection with a list of good clusters to be kept
plt.scatter(X['combined_score'], X['moran'], c=cluster_labels, s=2, cmap="tab20")
plt.show()
# automatically keep the clusters with the best metrics if user decides so, otherwise pass and wait for call on cluster numbers as described below

### feature selection 4: pass manual list of peak m/z values to be kept or of score clusters as computed above if precomputation has already occurred

# in any case, then annotate the feature selection to the lead anndata object. save it to file specifying it is annotated with lipid names and feature selection
# now use the lipid names and the feature selection to subset-filter the anndata object. specifically, keep features that_
# 1 pass feature selection according to the chosen criterion
# 2 if the user wants (default True), have a lipid name that does not contain the substring "_db" we created to flag the untrustworthy annotations


##################################################################################################################
# BLOCK 7: NORMALIZATION TO 0-1 RANGE -> function min0max1_normalize_clip
##################################################################################################################
datemp # copy the dataframe from the anndata after feature selection, and normalize 0-1 while clipping outliers

p2 = datemp.quantile(0.005) # this is default, user can change the percentiles
p98 = datemp.quantile(0.995)# this is default, user can change the percentiles 

datemp_values = datemp.values
p2_values = p2.values
p98_values = p98.values

normalized_values = (datemp_values - p2_values) / (p98_values - p2_values)

clipped_values = np.clip(normalized_values, 0, 1)

normalized_datemp = pd.DataFrame(clipped_values, columns=datemp.columns, index=datemp.index)
# normalized_datemp should be stored in the adata file as an alternative view of the feature space

