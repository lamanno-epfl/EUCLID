import os
import zarr
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import warnings
import datetime
import matplotlib.pyplot as plt

from tqdm import tqdm
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from functools import reduce
import re

warnings.filterwarnings('ignore')

class Preprocessing:
    """
    A class encapsulating all preprocessing steps (Blocks 1-9) for MSI data.
    Each method corresponds to a specific BLOCK in the original snippets.
    """

    def calculate_moran(
        self,
        path_data,
        acquisitions,
        n_molecules=np.inf,
        log_file="iterations_log.txt",
        morans_csv="morans_by_sec.csv"
    ):
        """
        Calculate and store Moran's I for each feature and each section.

        Parameters
        ----------
        path_data : str
            Path to the uMAIA Zarr dataset.
        acquisitions : list of str, optional
            List of acquisitions/sections to process.
        log_file : str, optional
            Path to the file where iteration logs are appended.
        morans_csv : str, optional
            Path to the CSV file where Moran's I results are saved.

        Returns
        -------
        pd.DataFrame
            DataFrame (feature x acquisition) containing Moran's I values.
        """
        
        acqn = acquisitions['acqn'].values
        acquisitions = acquisitions['acqpath'].values
        
        root = zarr.open(path_data, mode='r')
        features = np.sort(list(root.group_keys()))[:n_molecules]
        masks = [np.load(f'/data/LBA_DATA/{section}/mask.npy') for section in acquisitions]
        
        n_acquisitions = len(acquisitions) # FIXXXX HERE I SHOULD BETTER CALL ACQUISITIONS BY NAME EG PROTOTYPING ON SUBSET
        accqn_num = np.arange(n_acquisitions)
        
        morans_by_sec = pd.DataFrame(
            np.zeros((len(features), n_acquisitions)), 
            index=features, 
            columns=acqn.astype(str)
        )

        with open(log_file, "a") as file:
            for i_feat, feat in tqdm(enumerate(features), desc="Calculating Moran's I"):
                for j, j1 in zip(acqn, accqn_num):
                    mask = masks[j1]
                    
                    image = root[feat][str(j)][:]
                    
                    coords = np.column_stack(np.where(mask))
                    X = image[coords[:, 0], coords[:, 1]]

                    adata = sc.AnnData(X=pd.DataFrame(X))
                    adata.obsm['spatial'] = coords

                    sq.gr.spatial_neighbors(adata, coord_type='grid')
                    sq.gr.spatial_autocorr(adata, mode='moran')

                    morans_by_sec.loc[feat, str(j)] = adata.uns['moranI']['I'].values

                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"Iteration {i_feat + 1}, Time: {current_time}\n")

        morans_by_sec = morans_by_sec.fillna(0)
        morans_by_sec.to_csv(morans_csv)
        return morans_by_sec


    def store_exp_data_metadata(
        self,
        path_data,
        acquisitions=None,
        metadata_csv="acquisitions_metadata.csv",
        output_anndata="msi_preprocessed.h5ad",
        max_dim=(500, 500)
    ):
        """
        Store data in a Scanpy/AnnData structure, incorporating metadata and exponentiating values.

        Parameters
        ----------
        path_data : str, optional
            Path to the uMAIA Zarr dataset.
        acquisitions : list of str, optional
            List of acquisitions/sections to process.
        metadata_csv : str, optional
            Path to the CSV file containing section-wise metadata (must have 'SectionID').
        output_anndata : str, optional
            Path to save the resulting AnnData object.
        max_dim : tuple of int, optional
            Maximum dimensions (x, y) for zero-padding images.

        Returns
        -------
        sc.AnnData
            AnnData object with pixel-wise intensities and metadata.
        """
        
        root = zarr.open(path_data, mode='r')
        features = np.sort(list(root.group_keys()))

        acqn = acquisitions['acqn'].values
        acquisitions = acquisitions['acqpath'].values
        n_acquisitions = len(acquisitions)
        accqn_num = np.arange(n_acquisitions)

        # Prepare a 4D array: (features, acquisitions, x, y)
        lipid_native_sections_array = np.full(
            (len(features), n_acquisitions, max_dim[0], max_dim[1]),
            np.nan
        )

        for i_feat, feat in tqdm(enumerate(features), desc="Storing data into array"):
            for i_sec, i_sec1  in zip(acqn, accqn_num):
                img = root[feat][str(i_sec)][:]
                img_x, img_y = img.shape
                lipid_native_sections_array[i_feat, i_sec1, :img_x, :img_y] = img

        # Flatten along acquisitions * x * y
        flattened_lipid_tensor = lipid_native_sections_array.reshape(lipid_native_sections_array.shape[0], -1)
        lipid_names = features  # Use actual features as row labels

        # Build column names: section{i}_pixel{row}_{col}
        column_names = []
        for i_sec, i_sec1  in zip(acqn, accqn_num):
            for row in range(max_dim[0]):
                for col in range(max_dim[1]):
                    column_names.append(f"section{i_sec1+1}_pixel{row+1}_{col+1}")

        df = pd.DataFrame(flattened_lipid_tensor, index=lipid_names, columns=column_names)
        df_transposed = df.T.dropna(how='all')  # Remove rows that are entirely NaN
        df_transposed.columns = features  # rename columns with actual features

        # Extract spatial coordinates
        df_index = df_transposed.index.to_series().str.split('_', expand=True)
        df_index.columns = ['SectionID', 'x', 'y']
        df_index['SectionID'] = df_index['SectionID'].str.replace('section', '')
        df_index['x'] = df_index['x'].str.split('pixel').str.get(1)
        df_index = df_index.astype(int)
        df_transposed = df_transposed.join(df_index)

        # Exponentiate intensities
        df_transposed.loc[:, features] = np.exp(df_transposed.loc[:, features])

        # Merge with section-wise metadata
        metadata = pd.read_csv(metadata_csv)
        df_transposed = df_transposed.merge(metadata, on='SectionID', how='left')

        # Filter: remove pixels that are entirely < 0.0001
        mask = (df_transposed.loc[:, features] < 0.0001).all(axis=1)
        df_transposed = df_transposed[~mask]

        # Build AnnData, basic approach: everything is in df_transposed. We separate the X (features) from obs (pixels).
        X = df_transposed[features].values
        obs_cols = [c for c in df_transposed.columns if c not in features]
        adata = sc.AnnData(X=X)
        adata.var_names = features
        adata.obs = df_transposed[obs_cols].copy()

        # Save to disk
        adata.write_h5ad(output_anndata)
        return adata


    def annotate_molecules(
        self,
        msipeaks,
        structures_sdf="structures.sdf",
        hmdb_csv="HMDB_complete.csv",
        user_annotation_csv=None,
        ppm=5
    ):
        """
        Annotate m/z peaks with lipid names using external references (LIPID MAPS + HMDB, user's CSV file, ideally from a paired LC-MS dataset).

        Parameters
        ----------
        structures_sdf : str, optional
            Path to the SDF file for LIPID MAPS.
        hmdb_csv : str, optional
            Path to the HMDB reference CSV.
        user_annotation_csv : str, optional
            CSV containing user-provided m/z -> lipid annotations.
        ppm : float, optional
            Parts-per-million tolerance for matching.

        Returns
        -------
        pd.DataFrame
            Combined annotation table with possible matches.
        """
        from rdkit import Chem

        peaks_df = pd.DataFrame(msipeaks, index = msipeaks, columns = ["PATH_MZ"])

        # Load LIPID MAPS from SDF
        supplier = Chem.SDMolSupplier(structures_sdf)
        lm_id_list, name_list, systematic_name_list = [], [], []
        category_list, main_class_list, mass_list = [], [], []
        abbreviation_list, ik_list = [], []

        for molecule in tqdm(supplier, desc="Reading LIPID MAPS SDF"):
            if molecule is not None:
                lm_id_list.append(molecule.GetProp('LM_ID') if molecule.HasProp('LM_ID') else None)
                name_list.append(molecule.GetProp('NAME') if molecule.HasProp('NAME') else None)
                systematic_name_list.append(molecule.GetProp('SYSTEMATIC_NAME') if molecule.HasProp('SYSTEMATIC_NAME') else None)
                category_list.append(molecule.GetProp('CATEGORY') if molecule.HasProp('CATEGORY') else None)
                main_class_list.append(molecule.GetProp('MAIN_CLASS') if molecule.HasProp('MAIN_CLASS') else None)
                mass_list.append(molecule.GetProp('EXACT_MASS') if molecule.HasProp('EXACT_MASS') else None)
                abbreviation_list.append(molecule.GetProp('ABBREVIATION') if molecule.HasProp('ABBREVIATION') else None)
                ik_list.append(molecule.GetProp('INCHI_KEY') if molecule.HasProp('INCHI_KEY') else None)

        lipidmaps = pd.DataFrame({
            'LM_ID': lm_id_list,
            'NAME': name_list,
            'SYSTEMATIC_NAME': systematic_name_list,
            'CATEGORY': category_list,
            'MAIN_CLASS': main_class_list,
            'EXACT_MASS': mass_list,
            'ABBREVIATION': abbreviation_list,
            'INCHY_KEY': ik_list
        })

        # Merge with HMDB if needed to match METASPACE annotations
        hmdb = pd.read_csv(hmdb_csv, index_col=0)
        merged_df = pd.merge(
            lipidmaps, 
            hmdb, 
            left_on='INCHY_KEY', 
            right_on='InchiKey', 
            how='left'
        )
        conversionhmdb = merged_df[['DBID', 'ABBREVIATION']].dropna()

        reference_mz = 800 # scale of our dataset
        distance_ab5ppm = ppm / 1e6 * reference_mz
        
        def _find_closest_abbreviation(mz_list, lipidmaps):
            closest_abbreviations = []
            for mz in mz_list:
                abs_diffs = np.abs(lipidmaps['EXACT_MASS'].astype(float) - float(mz))
                if np.min(abs_diffs) <= distance_ab5ppm:
                    closest_idx = abs_diffs.idxmin()
                    closest_abbreviation = lipidmaps.at[closest_idx, 'ABBREVIATION']
                else:
                    closest_abbreviation = None
                closest_abbreviations.append(closest_abbreviation)
            return closest_abbreviations

        # add LIPIDMAPS annotation
        lipidmaps.loc[lipidmaps['ABBREVIATION'].isna(), 'ABBREVIATION'] = lipidmaps['NAME']
        lipidmaps = lipidmaps[['EXACT_MASS',	'ABBREVIATION']]
        # reconsider all possible adducts
        peaks_df['mz'] = peaks_df.index.astype(float)
        peaks_df['mz'] = [[peaks_df.iloc[i,:]['mz'] - 22.989769, peaks_df.iloc[i,:]['mz'] - 38.963707, peaks_df.iloc[i,:]['mz'] - 1.007825, peaks_df.iloc[i,:]['mz'] - 18.033823] for i in range(0, peaks_df.shape[0])]
        lipidmaps['EXACT_MASS'] = pd.to_numeric(lipidmaps['EXACT_MASS'], errors='coerce')
        peaks_df['LIPIDMAPS'] = _find_closest_abbreviation(peaks_df['PATH_MZ'].values.tolist(), lipidmaps)

        try:
            # Load user annotation CSV containing m/z and Lipid names
            # We assume columns: ["m/z", "Lipids", "Score"]
            user_df = pd.read_csv(user_annotation_csv)
    
            # Example matching function:
            def _find_matching_lipids(path_mz, lipid_mz_df):
                try:
                    lower_bound = path_mz - ppm / 1e6 * path_mz
                    upper_bound = path_mz + ppm / 1e6 * path_mz
                    matching_lipids = lipid_mz_df[(lipid_mz_df['m/z'] >= lower_bound) & (lipid_mz_df['m/z'] <= upper_bound)]['Lipids']
                    return ', '.join(matching_lipids)
                except:
                    return None
    
            peaks_df['Lipid'] = [_find_matching_lipids(i, user_df) for i in peaks_df['PATH_MZ'].astype(float).values.tolist()]
                
        except:
            print("No paired LC-MS or METASPACE annotation dataset provided. Are you sure you want to continue with database search only?")

        return peaks_df


    def save_msi_dataset(
        self,
        adata: sc.AnnData,
        filename="msi_dataset.h5ad"
    ):
        """
        Save the current AnnData object to disk.

        Parameters
        ----------
        adata : sc.AnnData
            The AnnData object containing MSI data.
        filename : str, optional
            File path to save the AnnData object.
        """
        adata.write_h5ad(filename)


    def load_msi_dataset(
        self,
        filename="msi_dataset.h5ad"
    ) -> sc.AnnData:
        """
        Load an AnnData object from disk.

        Parameters
        ----------
        filename : str, optional
            File path from which to load the AnnData object.

        Returns
        -------
        sc.AnnData
            The loaded AnnData object.
        """
        return sc.read_h5ad(filename)


    def prioritize_adduct(
        self,
        path_data="/data/luca/lipidatlas/uMAIA_allbrains/021124_ALLBRAINS_normalised.zarr",
        acquisitions=None,
        annotation_to_mz=None,
        output_csv="prioritized_adducts.csv"
    ):
        """
        Prioritize adducts by total signal across sections.

        Parameters
        ----------
        path_data : str, optional
            Path to the Zarr dataset.
        acquisitions : list of str, optional
            List of acquisitions.
        annotation_to_mz : dict, optional
            Dictionary mapping annotation -> list of candidate m/z values.
        output_csv : str, optional
            Path to save the dictionary of best isobar to CSV.

        Returns
        -------
        dict
            A dictionary mapping each annotation to its best m/z value.
        """
        if acquisitions is None:
            acquisitions = [
                'BrainAtlas/BRAIN2/20211201_MouseBrain2_S11_306x248_Att30_25um',
                'BrainAtlas/BRAIN2/20211202_MouseBrain2_S12_332x246_Att30_25um',
            ]
        if annotation_to_mz is None:
            annotation_to_mz = {
                # Example placeholder
                "SomeAnnotation": ["800.123456", "801.123456"],
            }

        root = zarr.open(path_data, mode='r')
        features = np.sort(list(root.group_keys()))
        masks = [np.load(f'/data/LBA_DATA/{section}/mask.npy') for section in acquisitions]
        n_acquisitions = len(acquisitions)

        # Compute total signal
        totsig_df = pd.DataFrame(
            np.zeros((len(features), n_acquisitions)), 
            index=features, 
            columns=[str(i) for i in range(n_acquisitions)]
        )

        for feat in tqdm(features, desc="Computing total signal"):
            feat_dec = f"{float(feat):.6f}"
            for i_sec in range(n_acquisitions):
                image = root[feat_dec][str(i_sec)][:]
                mask = masks[i_sec]
                image[mask == 0] = 0
                sig = np.mean(image * 1e6)
                totsig_df.loc[feat, str(i_sec)] = sig

        totsig_df = totsig_df.fillna(0)
        featuresum = totsig_df.sum(axis=1)

        annotation_to_mz_bestisobar = {}
        for annotation, mz_values in annotation_to_mz.items():
            max_featuresum = -float('inf')
            best_mz = None

            for mz_value in mz_values:
                if mz_value in featuresum.index:
                    val = featuresum.loc[mz_value]
                    if val > max_featuresum:
                        max_featuresum = val
                        best_mz = mz_value
                else:
                    print(f"m/z value {mz_value} not found in featuresum index.")

            if best_mz is not None:
                annotation_to_mz_bestisobar[annotation] = best_mz
            else:
                print(f"No valid m/z values found for annotation {annotation}.")

        # Optionally save the results
        pd.DataFrame.from_dict(annotation_to_mz_bestisobar, orient='index').to_csv(output_csv)
        return annotation_to_mz_bestisobar


    def feature_selection(
        self,
        adata: sc.AnnData,
        moran: pd.DataFrame,
        n_sections_to_consider=2,
        moran_threshold=0.4,
        cluster_k=10,
        output_csv="feature_scores.csv"
    ):
        """
        Perform feature selection based on Moran's I, variance metrics, and dropout thresholds.

        Parameters
        ----------
        adata : sc.AnnData
            The AnnData object with pixel-wise intensities.
        moran : pd.DataFrame
            DataFrame of Moran's I values (features x sections).
        n_sections_to_consider : int, optional
            Number of sections to consider in the reference set.
        moran_threshold : float, optional
            Minimal Moran's I threshold to keep a feature.
        cluster_k : int, optional
            Number of clusters for grouping features.
        output_csv : str, optional
            File path to save the feature scores.

        Returns
        -------
        pd.DataFrame
            DataFrame of selected features and their scores.
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        # 1) Subset Moran
        sub_moran = moran.iloc[:, :n_sections_to_consider]
        # Example: mean Moran across those sections
        mean_moran = sub_moran.mean(axis=1)

        # 2) Prepare data for variance-based scoring
        df_input = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
        df_input[df_input > 1.0] = 0.0001  # example clamp from your snippet

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_input)
        df_scaled = pd.DataFrame(scaled_data, columns=df_input.columns, index=df_input.index)

        # Build a new AnnData for the scoring step
        temp_adata = sc.AnnData(X=df_scaled)
        if 'Section' in adata.obs.columns:
            temp_adata.obsm['spatial'] = adata.obs[['Section']].values
        else:
            temp_adata.obsm['spatial'] = np.zeros((df_scaled.shape[0], 1))

        # Score features
        var_of_vars, mean_of_vars, combined_score = self._rank_features_by_combined_score(temp_adata)
        features_sorted = df_scaled.columns
        scores_df = pd.DataFrame({
            "feature": features_sorted,
            "var_of_vars": var_of_vars,
            "mean_of_vars": mean_of_vars,
            "combined_score": combined_score
        })
        scores_df.set_index("feature", inplace=True)

        # Attach Moran
        # Align index types
        # Convert both to strings if needed
        mean_moran.index = mean_moran.index.astype(str)
        scores_df.index = scores_df.index.astype(str)
        scores_df['moran'] = mean_moran.reindex(scores_df.index).fillna(0)

        # Filter by Moran threshold
        keep_features = mean_moran[mean_moran > moran_threshold].index
        keep_features = keep_features.intersection(scores_df.index)
        scores_df = scores_df.loc[keep_features]

        # Example dropout threshold logic would go here if needed

        # KMeans clustering
        X = scores_df[['var_of_vars', 'combined_score', 'moran']].fillna(0)
        kmeans = KMeans(n_clusters=cluster_k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        scores_df['cluster'] = cluster_labels

        # Save
        scores_df.to_csv(output_csv)
        return scores_df


    def _rank_features_by_combined_score(self, adata: sc.AnnData):
        """
        Helper method to rank features by a combined score of variance metrics.

        Parameters
        ----------
        adata : sc.AnnData
            An AnnData object with X as scaled features and obsm['spatial'] containing 'Section'.

        Returns
        -------
        tuple of np.ndarrays
            (var_of_vars, mean_of_vars, combined_score)
        """
        sections = adata.obsm['spatial'][:, 0]  # Assuming column 0 is 'Section'
        unique_sections = np.unique(sections)

        var_of_vars = []
        mean_of_vars = []

        # Evaluate each feature
        for i in range(adata.X.shape[1]):
            feature_values = adata.X[:, i]
            section_variances = []
            for sec in unique_sections:
                sec_vals = feature_values[sections == sec]
                section_variances.append(np.var(sec_vals))
            var_of_vars.append(np.var(section_variances))
            mean_of_vars.append(np.mean(section_variances))

        var_of_vars = np.array(var_of_vars)
        mean_of_vars = np.array(mean_of_vars)
        combined_score = -var_of_vars / 2 + mean_of_vars

        return var_of_vars, mean_of_vars, combined_score


    def min0max1_normalize_clip(
        self,
        df_input: pd.DataFrame,
        lower_quantile=0.005,
        upper_quantile=0.995
    ):
        """
        Normalize data by clipping at given quantiles and scaling to [0,1].

        Parameters
        ----------
        df_input : pd.DataFrame
            Input data to normalize.
        lower_quantile : float, optional
            Lower percentile for clipping.
        upper_quantile : float, optional
            Upper percentile for clipping.

        Returns
        -------
        pd.DataFrame
            The normalized DataFrame (values in [0,1]).
        """
        p2 = df_input.quantile(lower_quantile)
        p98 = df_input.quantile(upper_quantile)

        arr = df_input.values
        p2_vals = p2.values
        p98_vals = p98.values

        normalized = (arr - p2_vals) / (p98_vals - p2_vals)
        clipped = np.clip(normalized, 0, 1)

        df_norm = pd.DataFrame(clipped, columns=df_input.columns, index=df_input.index)
        return df_norm


    def lipid_properties(self, lipid_names, color_map_file="lipidclasscolors.h5ad"):
        """
        Extract basic lipid properties from a list of lipid names.

        Parameters
        ----------
        lipid_names : list of str
            List of lipid names from the AnnData object.
        color_map_file : str, optional
            Path to an HDF5 file containing a DataFrame with 'classcolors'.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing columns: [lipid_name, class, carbons, insaturations,
            insaturations_per_Catom, broken, color].
        """
        df = pd.DataFrame(lipid_names, columns=["lipid_name"]).fillna('')
        # Regex extraction
        df["class"] = df["lipid_name"].apply(
            lambda x: re.match(r'^(PE O|PC O|\S+)', x).group(0) if re.match(r'^(PE O|PC O|\S+)', x) else ''
        )
        df["carbons"] = df["lipid_name"].apply(
            lambda x: int(re.search(r'(\d+):', x).group(1)) if re.search(r'(\d+):', x) else np.nan
        )
        df["insaturations"] = df["lipid_name"].apply(
            lambda x: int(re.search(r':(\d+)', x).group(1)) if re.search(r':(\d+)', x) else np.nan
        )
        df["insaturations_per_Catom"] = df["insaturations"] / df["carbons"]
        df["broken"] = df["lipid_name"].str.endswith('_uncertain')

        df.loc[df["broken"], ['carbons','class','insaturations','insaturations_per_Catom']] = np.nan

        # Load color map
        colors = pd.read_hdf(color_map_file, key="table")
        df['color'] = df['class'].map(colors['classcolors'])
        df.loc[df["broken"], 'color'] = "gray"

        df.index = df['lipid_name']
        df = df.drop_duplicates()
        return df


    def reaction_network(self, lipid_props_df, premanannot):
        """
        Extract a reaction network based on lipid classes and transformation rules.

        Parameters
        ----------
        lipid_props_df : pd.DataFrame
            A DataFrame with lipid properties (index=lipid_name, columns=class,carbons,insaturations,...).
        premanannot : pd.DataFrame
            A DataFrame with columns ['reagent','product'] at least, plus user-defined columns.

        Returns
        -------
        pd.DataFrame
            Filtered premanannot that satisfies the transformation rules.
        """
        df = lipid_props_df.copy()
        if df.index.name != 'lipid_name':
            df.set_index('lipid_name', inplace=True)

        # Merge reagent attributes
        premanannot = premanannot.merge(
            df[['class', 'carbons', 'insaturations']],
            left_on='reagent',
            right_index=True,
            how='left',
            suffixes=('', '_reagent')
        ).rename(columns={
            'class': 'reagent_class',
            'carbons': 'reagent_carbons',
            'insaturations': 'reagent_insaturations'
        })

        # Merge product attributes
        premanannot = premanannot.merge(
            df[['class', 'carbons', 'insaturations']],
            left_on='product',
            right_index=True,
            how='left',
            suffixes=('', '_product')
        ).rename(columns={
            'class': 'product_class',
            'carbons': 'product_carbons',
            'insaturations': 'product_insaturations'
        })

        # Extract X
        premanannot['X_reagent'] = premanannot['reagent_class'].apply(self._extract_X)
        premanannot['X_product'] = premanannot['product_class'].apply(self._extract_X)

        # Build conditions
        conditions = self._build_reaction_conditions(premanannot)
        final_condition = reduce(lambda x, y: x | y, conditions)
        filtered = premanannot[final_condition].copy()

        return filtered


    def _extract_X(self, lipid_class):
        """
        Extract the 'X' component from a lipid class, e.g. 'LPX', 'PX', possibly with 'O-'.
        """
        if pd.isna(lipid_class):
            return None
        if 'O-' in lipid_class:
            match = re.match(r'^LP([CSEGIA]) O-|^P([CSEGIA]) O-', lipid_class)
        else:
            match = re.match(r'^LP([CSEGIA])|^P([CSEGIA])', lipid_class)
        if match:
            return match.group(1) if match.group(1) else match.group(2)
        return None


    def _build_reaction_conditions(self, premanannot):
        """
        Define a list of conditions that specify valid transformations.
        Returns a list of boolean Series for OR-combination.
        """
        X_classes = ['C', 'S', 'E', 'G', 'I', 'A']

        c1 = (
            premanannot['reagent_class'].str.startswith('LP') &
            premanannot['product_class'].str.startswith('P') &
            premanannot['X_reagent'].isin(X_classes) &
            premanannot['X_product'].isin(X_classes) &
            (premanannot['X_reagent'] == premanannot['X_product'])
        )

        c2 = (
            premanannot['reagent_class'].str.startswith('P') &
            premanannot['product_class'].str.startswith('LP') &
            premanannot['X_reagent'].isin(X_classes) &
            premanannot['X_product'].isin(X_classes) &
            (premanannot['X_reagent'] == premanannot['X_product'])
        )

        c3a = (
            premanannot['reagent_class'].str.startswith('LP') &
            premanannot['reagent_class'].str.contains('O-') &
            premanannot['product_class'].str.startswith('P') &
            premanannot['product_class'].str.contains('O-') &
            premanannot['X_reagent'].isin(X_classes) &
            premanannot['X_product'].isin(X_classes) &
            (premanannot['X_reagent'] == premanannot['X_product'])
        )

        c3b = (
            premanannot['reagent_class'].str.startswith('P') &
            premanannot['reagent_class'].str.contains('O-') &
            premanannot['product_class'].str.startswith('LP') &
            premanannot['product_class'].str.contains('O-') &
            premanannot['X_reagent'].isin(X_classes) &
            premanannot['X_product'].isin(X_classes) &
            (premanannot['X_reagent'] == premanannot['X_product'])
        )

        c4 = (premanannot['reagent_class'] == 'PC') & (premanannot['product_class'] == 'PA')
        c5 = (
            (premanannot['reagent_class'] == 'LPC') &
            (premanannot['product_class'] == 'LPC') &
            (premanannot['product_carbons'] > premanannot['reagent_carbons'])
        )
        c6 = (premanannot['reagent_class'] == 'PC') & (premanannot['product_class'] == 'DG')
        c7 = (premanannot['reagent_class'] == 'PS') & (premanannot['product_class'] == 'PE')
        c8 = (premanannot['reagent_class'] == 'PE') & (premanannot['product_class'] == 'PS')
        c9 = (premanannot['reagent_class'] == 'PE') & (premanannot['product_class'] == 'PC')
        c10 = (premanannot['reagent_class'] == 'SM') & (premanannot['product_class'] == 'Cer')
        c11 = (premanannot['reagent_class'] == 'Cer') & (premanannot['product_class'] == 'HexCer')
        c12 = (premanannot['reagent_class'] == 'Cer') & (premanannot['product_class'] == 'SM')
        c13 = (premanannot['reagent_class'] == 'HexCer') & (premanannot['product_class'] == 'Cer')
        c14 = (premanannot['reagent_class'] == 'HexCer') & (premanannot['product_class'] == 'Hex2Cer')
        c15 = (premanannot['reagent_class'] == 'Hex2Cer') & (premanannot['product_class'] == 'HexCer')
        c16 = (premanannot['reagent_class'] == 'PG') & (premanannot['product_class'] == 'DG')

        return [c1, c2, c3a, c3b, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16]
