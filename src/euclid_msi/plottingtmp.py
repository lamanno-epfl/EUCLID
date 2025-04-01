# draft code to plot a bunch of early dataset composition statistics [to be strongly updated knowing the adata structure we're working with and avoiding wasted precomputations where available elsewhere in the package)
def extract_class(lipid_name):
    """
    Extract the lipid class from a lipid name.
    Handles cases like "PC O-36:4" where we want to capture "PC O".
    This regex looks for one or more alphanumeric characters followed by an
    optional " O" (with or without a space) and then a space or dash.
    """
    m = re.match(r'^([A-Za-z0-9]+(?:\s?O)?)[\s-]', lipid_name)
    if m:
        return m.group(1)
    else:
        return lipid_name.split()[0]

# Test the extraction with a few examples
test_lipids = ["PC O-36:4", "PC-36:4", "PE O-38:6", "PE-38:6"]
for lip in test_lipids:
    print(f"{lip} -> {extract_class(lip)}")

df["class"] = df["lipid_name"].apply(extract_class)

# Extract number of carbons and insaturations from the lipid name
df["carbons"] = df["lipid_name"].apply(
    lambda x: int(re.search(r'(\d+):', x).group(1)) if re.search(r'(\d+):', x) else np.nan
)
df["insaturations"] = df["lipid_name"].apply(
    lambda x: int(re.search(r':(\d+)', x).group(1)) if re.search(r':(\d+)', x) else np.nan
)
df["insaturations_per_Catom"] = df["insaturations"] / df["carbons"]
df.loc[df['class'] == "HexCer", 'class'] = "Hex1Cer"

# Mark broken entries based on naming convention (e.g., ending with '_uncertain')
df["broken"] = df["lipid_name"].str.endswith('_uncertain')
df.loc[df["broken"], ['carbons', 'class', 'insaturations', 'insaturations_per_Catom']] = np.nan

# Map colors from an external file
colors = pd.read_hdf("lipidclasscolors.h5ad", key="table")
df['color'] = df['class'].map(colors['classcolors'])
df.loc[df["broken"], 'color'] = "gray"

# Set index and remove duplicates
df.index = df['lipid_name']
df = df.drop_duplicates()
df['color'] = df['color'].fillna("black")

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42
classvalues = df['class'].value_counts()
pie_colors = [color_dict.get(cls, 'black') for cls in classvalues.index]

plt.figure(figsize=(8, 8))
classvalues.plot.pie(
    colors=pie_colors, 
    autopct='%1.1f%%', 
    startangle=90, 
    textprops={'fontsize': 10}
)
plt.ylabel('')  # Hide the y-label
plt.title("Lipid class cardinality of species in whole-brain MSI")
plt.savefig("msi_prop.pdf")
plt.show()

meanmoranperpeak = annowmoran.iloc[:, -138:].mean(axis=1)
meanmoranperpeak.index = annowmoran['Annotation']
meanmoranperpeak = meanmoranperpeak.fillna(0)
#meanmoranperpeak = meanmoranperpeak.loc[~pd.isna(meanmoranperpeak)] ##### drop the nans
#meanmoranperpeak = meanmoranperpeak.loc[meanmoranperpeak.index != "_db"]# drop the unannotated

meanmoranperpeak.sort_values()[::-1] # 391 got an annotation and above zero Moran's, also including those with naming that is not confident.

atlas = pd.read_parquet("atlas.parquet")
import re
import numpy as np
import pandas as pd

# Example: assume quantlcms.index contains your lipid names
df = pd.DataFrame(meanmoranperpeak.index).fillna('')
df.columns = ["lipid_name"]
df['Score'] = annowmoran['Score'].values
df['mz'] = annowmoran.index.values
df

def extract_class(lipid_name):
    """
    Extract the lipid class from a lipid name.
    Handles cases like "PC O-36:4" where we want to capture "PC O".
    This regex looks for one or more alphanumeric characters followed by an
    optional " O" (with or without a space) and then a space or dash.
    """
    m = re.match(r'^([A-Za-z0-9]+(?:\s?O)?)[\s-]', lipid_name)
    if m:
        return m.group(1)
    else:
        return lipid_name.split()[0]

# Test the extraction with a few examples
test_lipids = ["PC O-36:4", "PC-36:4", "PE O-38:6", "PE-38:6"]
for lip in test_lipids:
    print(f"{lip} -> {extract_class(lip)}")

df["class"] = df["lipid_name"].apply(extract_class)

# Extract number of carbons and insaturations from the lipid name
df["carbons"] = df["lipid_name"].apply(
    lambda x: int(re.search(r'(\d+):', x).group(1)) if re.search(r'(\d+):', x) else np.nan
)
df["insaturations"] = df["lipid_name"].apply(
    lambda x: int(re.search(r':(\d+)', x).group(1)) if re.search(r':(\d+)', x) else np.nan
)
df["insaturations_per_Catom"] = df["insaturations"] / df["carbons"]
df.loc[df['class'] == "HexCer", 'class'] = "Hex1Cer"

# Mark broken entries based on naming convention (e.g., ending with '_uncertain')
df["broken"] = df["lipid_name"].str.endswith('_uncertain')
df.loc[df["broken"], ['carbons', 'class', 'insaturations', 'insaturations_per_Catom']] = np.nan

# Map colors from an external file
colors = pd.read_hdf("lipidclasscolors.h5ad", key="table")
df['color'] = df['class'].map(colors['classcolors'])
df.loc[df["broken"], 'color'] = "gray"

# Set index and remove duplicates

df['color'] = df['color'].fillna("black")

df

mean_series = meanmoranperpeak.rename('quant')
df['quant'] = mean_series.values
extra_classes = {'CerP', 'LPA', 'PIP O', 'PGP', 'PA', 'CAR', 'ST', 'PA O', 'CoA', 'MG', 'SHexCer', 'LPE O'}

color_dict.update({cls: "gray" for cls in extra_classes})

df.loc[df['class'].isin(['CerP', 'LPA', 'PIP O', 'PGP', 'PA', 'CAR', 'ST', 'PA O', 'CoA', 'MG', 'SHexCer']),'class'] = "others"
df.loc[df['lipid_name'].str.contains("_db"), 'class'] = "others"
color_dict.update({cls: "gray" for cls in ["others"]})

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from adjustText import adjust_text

# Compute the order of classes by ascending median of quant
order = df.groupby('class')['quant'].median().sort_values().index.tolist()

# Create a single figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the boxplot
sns.boxplot(data=df, x='class', y='quant', order=order, 
            palette=color_dict, showfliers=False, ax=ax)

# Overlay the individual data points using a stripplot
sns.stripplot(data=df, x='class', y='quant', order=order, 
              color='black', alpha=0.5, size=3, jitter=True, ax=ax)

# Add a dark red dashed horizontal line at y=0.4
ax.axhline(y=0.4, color='darkred', linestyle='--')

# Set the y-limit to stop at 1
ax.set_ylim(0, 1)

# Remove spines for a cleaner look.
sns.despine(ax=ax)

# Annotate all points with quant > 1 using label repulsion.
texts = []
for idx, row in df[df['quant'] > 1].iterrows():
    # Determine the base x coordinate from the class order:
    cat = row['class']
    try:
        x_center = order.index(cat)
    except ValueError:
        continue  # skip if class not in order (shouldn't happen)
    jitter = np.random.uniform(-0.1, 0.1)
    x = x_center + jitter
    y = row['quant']
    t = ax.text(x, y, row['lipid_name'], fontsize=20, ha='center', va='bottom')
    texts.append(t)

# Final touches: rotate x-axis labels, add axis labels and title.
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_xlabel("Lipid Class")
ax.set_ylabel("Lipid-wise mean Moran's I across sections")
ax.set_title("Spatial evaluation of lipids by average Moran's I")

plt.tight_layout()
plt.savefig("moranbyclass.pdf")
plt.show()





# draft code to plot lipid distributions(ADD optionally with anatomical contours, optionally filtering by metadata - division in the example)
# one could also select by metadata such as division etc (custom) a subset of the organ to display instead of going full-organ, passing a list of divisions/lipizones/selected metadata column unique categorical entries to show
# one could also select which lipizones to plot; layout should be computed dynamically to yield an approx 4/3 tyypical page layout
# the user can also provide a set of x, y, z ranges, in which cases before plotting we crop stuff withing those ranges (not necessarily all the three should be provided, if not provided we keep all along that specific axis)
with PdfPages('ranking_clustering_featsel.pdf') as pdf:
    for currentPC in tqdm(np.array(scores['spatvar'].astype(float).astype(str))):
        results = []
        filtered_data = pd.concat([data[['yccf','zccf','Section']], data.loc[:,str(currentPC)]], axis=1)[::5] #### ds to go faster

        for section in filtered_data['Section'].unique():
            subset = filtered_data[filtered_data['Section'] == section]

            perc_2 = subset[str(currentPC)].quantile(0.02)
            perc_98 = subset[str(currentPC)].quantile(0.98)

            results.append([section, perc_2, perc_98])
        percentile_df = pd.DataFrame(results, columns=['Section', '2-perc', '98-perc'])
        med2p = percentile_df['2-perc'].median()
        med98p = percentile_df['98-perc'].median()

        cmap = plt.cm.inferno

        fig, axes = plt.subplots(4, 8, figsize=(20, 10))
        axes = axes.flatten()

        for section in range(1, 33):
            ax = axes[section - 1]
            ddf = filtered_data[(filtered_data['Section'] == section)]

            ax.scatter(ddf['zccf'], -ddf['yccf'], c=ddf[str(currentPC)], cmap="plasma", s=2.0, alpha=0.8,rasterized=True, vmin=med2p, vmax=med98p) 
            ax.axis('off')
            ax.set_aspect('equal')

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = Normalize(vmin=med2p, vmax=med98p)
        sm = ScalarMappable(norm=norm, cmap="plasma")
        fig.colorbar(sm, cax=cbar_ax)

        plt.tight_layout(rect=[0, 0, 0.9, 1])
        pdf.savefig(fig) 
        plt.close(fig)



# draft code to plot embeddings(ADD optionally with anatomical contours, optionally filtering by metadata - division in the example)
# one could also select by metadata such as division etc (custom) a subset of the organ to display instead of going full-organ
# one could also select which lipizones to plot; layout should be computed dynamically to yield an approx 4/3 tyypical page layout
currentProgram = 'headgroup_with_negative_charge'
for section in data['Section'].unique():
    subset = data[data['Section'] == section]

    perc_2 = subset[currentProgram].quantile(0.02)
    perc_98 = subset[currentProgram].quantile(0.98)

    results.append([section, perc_2, perc_98])
percentile_df = pd.DataFrame(results, columns=['Section', '2-perc', '98-perc'])
med2p = percentile_df['2-perc'].median()
med98p = percentile_df['98-perc'].median()

cmap = plt.cm.PuOr

fig, axes = plt.subplots(4, 8, figsize=(20, 10))
axes = axes.flatten()

for section in range(1, 33):
    ax = axes[section - 1]
    ddf = data[(data['Section'] == section)]

    ax.scatter(ddf['zccf'], -ddf['yccf'], c=ddf[currentProgram], cmap="PuOr", s=0.5,rasterized=True, vmin=med2p, vmax=med98p) 
    ax.axis('off')
    ax.set_aspect('equal')

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
norm = Normalize(vmin=med2p, vmax=med98p)
sm = ScalarMappable(norm=norm, cmap="PuOr")
fig.colorbar(sm, cax=cbar_ax)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()




# draft code to plot lipizones (optionally with anatomical contours, optionally filtering by metadata - division in the example)
# one could also select by metadata such as division etc (custom) a subset of the organ to display instead of going full-organ
# one could also select which lipizones to plot; layout should be computed dynamically to yield an approx 4/3 tyypical page layout
# one could also decide which hierarchical level to plot, like level_1, level_2,... in that case, the color should be for each unique combination of level values down to that level the modal "lipizone_color" value (i.e., most frequent across pixels having that level_'s combination)
unique_sections = data["Section"].unique()

fig, axs = plt.subplots(4, 8, figsize=(32, 16))
axs = axs.flatten()

for i, section_value in enumerate(unique_sections):
    if i >= len(axs):
        break
    ax = axs[i]
    section = data[data["Section"] == section_value]
    filtered_section = section.loc[section['division'].isin(['Isocortex']),:]

    ax.scatter(filtered_section['z_index'], -filtered_section['y_index'],
                    c=filtered_section['lipizone_color'], s=0.05,
                    alpha=1, zorder=1, rasterized=True)  

    filtered_section_contour = section.loc[section['boundary'] == 1,:]
    ax.scatter(filtered_section_contour['z_index'], -filtered_section_contour['y_index'],
                    c='black', s=0.01, rasterized=True, zorder=2, alpha=0.9)
 
    ax.set_aspect('equal')
    
for ax in axs:
    ax.axis('off') 

plt.tight_layout()
plt.show()



# draft code to plot tsne with desired coloring from any metadata both by user and compèuted such as lipizone colors
plt.scatter(tsne[:, 0], tsne[:, 1], c=data.loc[:, 'lipizone_color'], s=0.0001, alpha=0.9, rasterized=True)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()



# draft code to plot a sorted heatmap (should already be properly normalized, if flag normalized is false, the method should make the rows relative so they sum to one
linkage = sch.linkage(sch.distance.pdist(normalized_df.T), method='weighted', optimal_ordering=True)
order = sch.leaves_list(linkage)
normalized_df = normalized_df.iloc[:, order]
order = np.argmax(normalized_df.values, axis=1)
order = np.argsort(order)
normalized_df = normalized_df.iloc[order,:]
# let user decide on xticklabels and yticklabels and on vmin and vmax but defaulting to my values as usual
plt.figure(figsize=(20, 5))
sns.heatmap(normalized_df, cmap="Grays", cbar_kws={'label': 'Enrichment'}, xticklabels=False, yticklabels=False, vmin = np.percentile(normalized_df, 2), vmax = np.percentile(normalized_df, 98))
# add axes labels provided by user
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', left=False, right=False)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()



# draft code to plot few selected lipizones or level_x column values at a time on a lightgray background, optionally with anatomical contours
# (generalize from 1 to many and from lipizones to all level_x according to user's desire, using modal lipizone_color as described before...
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



# draft code to plot a dendrogram from selected sorted list of values (make it more general than it is)
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
# Define the leaf labels in the desired order
labels = [...]
def generate_custom_linkage():
    """
    Generates a custom linkage matrix for 16 leaves arranged in a balanced binary tree,
    with all branches of the same length.
    """
    # Number of original observations (leaves)
    n = 16
    # Initialize an empty linkage matrix with (n-1) rows and 4 columns, dtype=float
    linkage_matrix = np.zeros((n - 1, 4), dtype=float)
    
    # Initialize cluster indices
    current_cluster = n  # Clusters are indexed from n onwards

    # Define pairs to merge at each level
    # Level 1: Merge adjacent leaves
    level1_pairs = [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15)
    ]
    
    # Assign first 8 merges (Level 1)
    for i, (a, b) in enumerate(level1_pairs):
        linkage_matrix[i, 0] = a
        linkage_matrix[i, 1] = b
        linkage_matrix[i, 2] = 1.0  # Distance for Level 1
        linkage_matrix[i, 3] = 2      # Number of samples in the new cluster

    # Level 2: Merge the clusters formed in Level 1
    level2_pairs = [
        (current_cluster, current_cluster + 1),
        (current_cluster + 2, current_cluster + 3),
        (current_cluster + 4, current_cluster + 5),
        (current_cluster + 6, current_cluster + 7)
    ]
    
    for i, (a, b) in enumerate(level2_pairs, start=8):
        linkage_matrix[i, 0] = a
        linkage_matrix[i, 1] = b
        linkage_matrix[i, 2] = 2.0  # Distance for Level 2
        linkage_matrix[i, 3] = 4      # Number of samples

    # Update current_cluster
    current_cluster += 8

    # Level 3: Merge the clusters formed in Level 2
    level3_pairs = [
        (current_cluster, current_cluster + 1),
        (current_cluster + 2, current_cluster + 3)
    ]
    
    for i, (a, b) in enumerate(level3_pairs, start=12):
        linkage_matrix[i, 0] = a
        linkage_matrix[i, 1] = b
        linkage_matrix[i, 2] = 3.0  # Distance for Level 3
        linkage_matrix[i, 3] = 8      # Number of samples

    # Update current_cluster
    current_cluster += 4

    # Level 4: Final merge to form the root
    linkage_matrix[14, 0] = current_cluster
    linkage_matrix[14, 1] = current_cluster + 1
    linkage_matrix[14, 2] = 4.0      # Distance for Level 4
    linkage_matrix[14, 3] = 16         # Number of samples

    return linkage_matrix

# Generate the custom linkage matrix
linkage_matrix = generate_custom_linkage()
# Verify that the linkage matrix contains floats
assert linkage_matrix.dtype == float, "Linkage matrix must be of float type."
# Create the dendrogram plot
plt.figure(figsize=(14, 10))  # Adjust figure size as needed
dendro = dendrogram(
    linkage_matrix,
    orientation='left',
    color_threshold=0,               # All links colored the same
    above_threshold_color='black',   # Color of the links
    labels=labels,                   # Assign the custom labels
    leaf_font_size=10,               # Adjust font size for readability
    show_leaf_counts=False,          # Do not show leaf counts
    no_labels=False,                 # Show labels
    link_color_func=lambda k: 'black'  # All links in black
)
# Style the plot
ax = plt.gca()
# Remove spines for a cleaner look
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(False)
# Remove ticks
ax.tick_params(axis='both', which='both', length=0)
# Remove x-ticks
plt.xticks([])
# Adjust y-ticks font size
plt.yticks(fontsize=10)
plt.xlabel('Distance')  # Optionally add an axis label
plt.tight_layout()
# Display the dendrogram
plt.show()



# draft code to plot a sorted barplot for differential lipids colored by their class
import numpy as np
from adjustText import adjust_text

meansus = coeffmap.mean()
meansus = meansus.sort_values()[:-1] ## the top 1 is clearly the only batch effect - escaped one, it's untrusted
dfff = pd.DataFrame(meansus)
colors = df.loc[dfff.index, 'color'].fillna("black")

plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(dfff)), dfff.iloc[:,0], color=colors)
n_items = len(dfff)
bottom_5 = list(range(5))
top_5 = list(range(n_items-5, n_items))
middle_start = 5
middle_end = n_items - 5
middle_5 = list(np.random.choice(range(middle_start, middle_end), 5, replace=False))
indices_to_label = sorted(bottom_5 + middle_5 + top_5)

texts = []
for idx in indices_to_label:
    x = idx
    y = dfff.iloc[idx, 0]
    label = dfff.index[idx]
    texts.append(plt.text(x, y, label, ha='center', va='bottom'))

adjust_text(texts, 
           arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
           expand_points=(1.5, 1.5))

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylabel('Mean susceptibility across the whole brain')
plt.xlabel("Sorted lipid species")
plt.xticks([])
plt.tight_layout()
plt.savefig("meansusc_pregnancy.pdf")
plt.show()



# plot a scatterplot with two continuous variables stored in the adata object, be them lipids, embeddings, other stored modalities etc, coloring by anything in the adata, be it continuous or categorical color label value




# draft code to plot sample-sample correlation through PCA and heatmap based on normalized lipid expression centroids across a selected level_x (here supertypes, make generic to be passed)
datemp = lips.copy() 
p2 = datemp.quantile(0.005)
p98 = datemp.quantile(0.995)

datemp_values = datemp.values
p2_values = p2.values
p98_values = p98.values

normalized_values = (datemp_values - p2_values) / (p98_values - p2_values)

clipped_values = np.clip(normalized_values, 0, 1)

normalized_datemp = pd.DataFrame(clipped_values, columns=datemp.columns, index=datemp.index)
centroids = normalized_datemp.groupby([metadata['Sample'], splits['supertype']]).mean()
centroids = centroids.unstack()
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(centroids), 
                          index=centroids.index, 
                          columns=centroids.columns)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_data)

var_explained = pca.explained_variance_ratio_ * 100

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(pca_result[:, 0], 
                    pca_result[:, 1], 
                    pca_result[:, 2],
                    c=colors, s=350, alpha=1)

ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}% variance)')
ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}% variance)')
ax.set_zlabel(f'PC3 ({var_explained[2]:.1f}% variance)')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.title('3D PCA of Centroids')
plt.savefig("pca_mfpreg.pdf")
plt.show()
sns.clustermap(centroids.T.corr())
plt.savefig("inter_samplecorrelation.pdf")


# plot 3d renderings using plotly
### keep it as a placeholder

# plot 2d renderings using plotly with zoomability
### keep it as a placeholder

# viz clock
### keep it as a placeholder

# make movies
### keep it as a placeholder