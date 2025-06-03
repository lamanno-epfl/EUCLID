import os
import zarr
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import warnings
import datetime
import matplotlib.pyplot as plt
import itertools
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
        n_molecules=5000000,
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


    def store_exp_data_metadata( # NOTE THAT THIS ALSO SERVES AS AN INIT AS IT CREATES THE SELF.ADATA
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
        df_transposed.index = "ind" + df_transposed.index.astype(str)
        # what broke the adata querying by name was this, which is the correct filter but somehow (no idea how) breaks the adata querying by name
        mask = df_transposed.loc[:, features].mean(axis=1) <= 0.00011
        df_transposed = df_transposed.loc[mask == False, :]
        print(df_transposed.shape)

        # Build AnnData, basic approach: everything is in df_transposed. We separate the X (features) from obs (pixels).
        X = df_transposed[features].values
        obs_cols = [c for c in df_transposed.columns if c not in features]
        adata = sc.AnnData(X=X)
        adata.var_names = features
        adata.obs = df_transposed[obs_cols].copy()

        # Save to disk
        adata.write_h5ad(output_anndata)
        self.adata = adata
        
        return adata

    def add_barcodes_to_adata(self) -> None:
        """
        Generate a unique barcode for each observation in the AnnData object 
        and update the observation index with these barcodes.
        
        The barcode is constructed from the 'SectionID', 'x', and 'y' columns in adata.obs,
        in the format: "section{SectionID}_pixel{x}_{y}".
        
        Returns
        -------
        None
            Updates self.adata.obs.index in-place.
        
        Raises
        ------
        ValueError
            If any of the required columns ('SectionID', 'x', 'y') are missing.
        """
        required_cols = ['SectionID', 'x', 'y']
        if not all(col in self.adata.obs.columns for col in required_cols):
            raise ValueError(f"adata.obs must contain columns: {required_cols}")
        
        # Generate barcode for each observation
        self.adata.obs.index = self.adata.obs.apply(
            lambda row: f"section{int(row['SectionID'])}_pixel{int(row['x'])}_{int(row['y'])}", axis=1
        )

    def add_metadata_from_parquet(self, parquet_file: str) -> None:
        """
        Update the AnnData object by (1) filtering its observations to those present in a parquet metadata file,
        and (2) adding any metadata columns from that file which are not already present in adata.obs.
        
        Parameters
        ----------
        parquet_file : str
            Path to the parquet file containing metadata (with the same index as your barcode/obs index).
        
        Returns
        -------
        None
        """
        # Load metadata from parquet
        metadata_df = pd.read_parquet(parquet_file)
        # Filter adata to keep only observations whose index exists in the metadata
        common_index = self.adata.obs.index.intersection(metadata_df.index)
        self.adata = self.adata[common_index].copy()
        # For each column in the metadata that is not already in adata.obs, add it.
        for col in metadata_df.columns:
            if col not in self.adata.obs.columns:
                # Align by index.
                self.adata.obs[col] = metadata_df.loc[self.adata.obs.index, col]

    def filter_by_metadata(self, column: str, operation: str) -> sc.AnnData:
        """
        Filter the AnnData object based on a condition specified on a metadata column.
        
        The user provides a column name and an operation (as a string) that is applied to that column.
        For example, if the column is "allencolor" and the operation is "!='#000000'", only entries with a 
        different value will be kept. Similarly, for column "x" with operation ">5".
        
        Parameters
        ----------
        column : str
            The name of the metadata column in adata.obs to filter on.
        operation : str
            A string representing a boolean condition (e.g., "!='#000000'", ">5").
        
        Returns
        -------
        sc.AnnData
            A new AnnData object containing only the observations that satisfy the condition.
        
        Example
        -------
        >>> filtered_adata = preprocessing.filter_by_metadata("allencolor", "!='#000000'")
        """
        query_str = f"`{column}` {operation}"
        filtered_obs = self.adata.obs.query(query_str)
        filtered_index = filtered_obs.index
        return self.adata[filtered_index].copy()

    def annotate_molecules(
        self,
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

        msipeaks = self.adata.var_names.tolist()
        peaks_df = pd.DataFrame(msipeaks, columns = ["PATH_MZ"], index = msipeaks)

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
            user_df.index = user_df['m/z'].astype(str)
            peaks_df.index = peaks_df.index.astype(str)
            
                
        except:
            print("No paired LC-MS or METASPACE annotation dataset provided. Are you sure you want to continue with database search only?")
            
        peaks_df['Score'] = 0
        common_indices = peaks_df.index.intersection(user_df.index)
        peaks_df.loc[common_indices, 'Score'] = user_df.loc[common_indices, 'Score']

        return peaks_df


    def save_msi_dataset(
        self,
        filename="prep_msi_dataset.h5ad"
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
        self.adata.write_h5ad(filename)


    def load_msi_dataset( # THIS SERVES AS AN ALTERNATIVE INIT
        self,
        filename="prep_msi_dataset.h5ad"
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
        adata = sc.read_h5ad(filename)
        self.adata = adata

    def prioritize_adducts(
        self,
        path_data,
        acquisitions,
        annotation_to_mz,
        output_csv="prioritized_adducts.csv"
    ):
        """
        Prioritize adducts by total signal across sections.

        Parameters
        ----------
        path_data : str, optional
            Path to the Zarr dataset.
        acquisitions : list of str
            List of acquisitions.
        annotation_to_mz : dict
            Dictionary mapping annotation -> list of candidate m/z values.
        output_csv : str, optional
            Path to save the dictionary of best adduct to CSV.

        Returns
        -------
        dict
            A dictionary mapping each annotation to its best m/z value.
        """


        acqn = acquisitions['acqn'].values
        acquisitions = acquisitions['acqpath'].values
        
        root = zarr.open(path_data, mode='r')
        features = np.sort(list(root.group_keys()))
        masks = [np.load(f'/data/LBA_DATA/{section}/mask.npy') for section in acquisitions]
        
        n_acquisitions = len(acquisitions) # FIXXXX HERE I SHOULD BETTER CALL ACQUISITIONS BY NAME EG PROTOTYPING ON SUBSET
        accqn_num = np.arange(n_acquisitions)
        
        totsig_df = pd.DataFrame(
            np.zeros((len(features), n_acquisitions)), 
            index=features, 
            columns=acqn.astype(str)
        )

        for feat in tqdm(features, desc="Computing total signal"):
            feat_dec = f"{float(feat):.6f}"
            for j, j1 in zip(acqn, accqn_num):
                image = np.exp(root[feat_dec][str(j)][:])
                mask = masks[j1]
                image[mask == 0] = 0
                sig = np.mean(image * 1e6)
                totsig_df.loc[feat, str(j)] = sig

        totsig_df = totsig_df.fillna(0)
        featuresum = totsig_df.sum(axis=1)

        annotation_to_mz_bestadduct = {}
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
                annotation_to_mz_bestadduct[annotation] = best_mz
            else:
                print(f"No valid m/z values found for annotation {annotation}.")

        # Optionally save the results
        pd.DataFrame.from_dict(annotation_to_mz_bestadduct, orient='index').to_csv(output_csv)
        totsig_df.to_csv("totsig_df_" + output_csv)
        return annotation_to_mz_bestadduct, totsig_df

    def feature_selection(
        self,
        moran: pd.DataFrame,
        modality: str = "combined",  # options: "moran", "combined", "manual"
        mz_vals: list = None,  # if provided, these m/z values override all other criteria
        moran_threshold: float = 0.25,
        cluster_k: int = 10,
        output_csv: str = "feature_scores.csv",
        remove_untrustworthy: bool = False
    ):
        """
        Perform feature selection based on one of three modalities.
        
        Parameters
        ----------
        adata : sc.AnnData
            The AnnData object with pixel-wise intensities.
        moran : pd.DataFrame
            DataFrame of Moran's I values (features x sections).
        modality : str, optional
            Which feature selection modality to use. Options are:
              - "moran": select features that have a mean Moran's I above the given threshold.
              - "combined": perform variance-based scoring and clustering and then select
                            clusters with the best combined metrics.
              - "manual": bypass computations and use an explicitly provided list of m/z values.
        mz_vals : list, optional
            A list of m/z values to keep. If provided and non-empty, this list overrides
            any modality-based feature selection.
        moran_threshold : float, optional
            Minimal Moran's I threshold to keep a feature.
        cluster_k : int, optional
            Number of clusters for grouping features in "combined" modality.
        output_csv : str, optional
            File path to save the feature scores.
        remove_untrustworthy : bool, optional
            If True, then features whose lipid names contain '_db' will be removed.
        
        Returns
        -------
        sc.AnnData
            A new AnnData object that is subset to the selected features, with an annotation of the 
            feature selection scores stored in .uns["feature_selection_scores"].
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        import scanpy as sc
        
        # --- Step 0. Manual override: if mz_vals is provided, simply use these features.
        if mz_vals is not None and len(mz_vals) > 0:
            # Convert m/z values to strings (to match adata.var_names)
            selected_features = set(str(x) for x in mz_vals)
            # Create a simple scores table to annotate the decision.
            scores_df = pd.DataFrame(index=sorted(selected_features))
            scores_df["manual_override"] = True
        # --- Modality "moran": use only the Moran I threshold
        elif modality.lower() == "moran":
            # Subset Moran values to the first n_sections_to_consider and compute mean
            sub_moran = moran###.iloc[:, :n_sections_to_consider]
            mean_moran = sub_moran.mean(axis=1)
            # Select features that pass the threshold
            selected_features = set(mean_moran[mean_moran > moran_threshold].index.astype(str))
            # Build a simple scores_df for later annotation
            scores_df = pd.DataFrame({"moran": mean_moran}).loc[[f for f in mean_moran.index.astype(str) if f in selected_features]]
        # --- Modality "combined": compute variance metrics and cluster features
        elif modality.lower() == "combined":
            # Compute mean Moran for the first n_sections_to_consider sections
            sub_moran = moran##.iloc[:, :n_sections_to_consider]
            mean_moran = sub_moran.mean(axis=1)
    
            # Prepare data: create a DataFrame from adata.X (clamping values above 1.0)
            df_input = pd.DataFrame(self.adata.X, columns=self.adata.var_names, index=self.adata.obs_names)
            df_input[df_input > 1.0] = 0.0001  # clamp extreme values
    
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df_input)
            df_scaled = pd.DataFrame(scaled_data, columns=df_input.columns, index=df_input.index)
    
            # Build a temporary AnnData for scoring
            temp_adata = sc.AnnData(X=df_scaled)
            # If spatial section information exists, attach it (using 'SectionID' if available)
            if 'Section' in self.adata.obs.columns:
                if 'SectionID' in self.adata.obs.columns:
                    temp_adata.obsm['spatial'] = self.adata.obs['SectionID'].values
                else:
                    temp_adata.obsm['spatial'] = self.adata.obs['Section'].values
            else:
                temp_adata.obsm['spatial'] = np.zeros((df_scaled.shape[0], 1))
    
            # Use helper function to score features by variance metrics
            var_of_vars, mean_of_vars, combined_score = self._rank_features_by_combined_score(temp_adata)
            features_sorted = df_scaled.columns
            scores_df = pd.DataFrame({
                "var_of_vars": var_of_vars,
                "mean_of_vars": mean_of_vars,
                "combined_score": combined_score
            }, index=features_sorted)
            # Attach the Moran metric (casting indices to str to align)
            scores_df['moran'] = mean_moran.values
            # Keep only features that meet the Moran threshold
            keep_features = set(mean_moran[mean_moran > moran_threshold].index.astype(str))
            scores_df = scores_df.loc[scores_df.index.isin(keep_features)]
            
            # --- Dropout filtering: for each feature, compute the number of sections where the mean is below a threshold.
            section_col = None
            if 'SectionID' in self.adata.obs.columns:
                section_col = 'SectionID'
            elif 'Section' in self.adata.obs.columns:
                section_col = 'Section'
            if section_col is not None:
                peakmeans = df_input.groupby(self.adata.obs[section_col]).mean()
                missinglipid = np.sum(peakmeans < 0.00015)
                dropout_acceptable = set(missinglipid[missinglipid < 4].index.astype(float).astype(str))
                scores_df = scores_df.loc[scores_df.index.isin(dropout_acceptable)]
            # Remove features with nonpositive combined score
            scores_df = scores_df.loc[scores_df['combined_score'] > 0, :]
    
            # --- Clustering features using KMeans on several metrics.
            X = scores_df[['var_of_vars', 'combined_score', 'moran']].copy()
            if section_col is not None:
                # Recalculate missinglipid for the features in scores_df
                missinglipid = np.sum(peakmeans < 0.00015)
                # Align indices as strings
                missinglipid = missinglipid.loc[[str(f) for f in scores_df.index]]
                scores_df['missinglipid'] = missinglipid
                X['missinglipid'] = missinglipid
            # Standardize the clustering features
            scaler2 = StandardScaler()
            X_scaled = scaler2.fit_transform(X.fillna(0))
            kmeans = KMeans(n_clusters=cluster_k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            scores_df['cluster'] = cluster_labels
    
            # --- Plot clustering for visual inspection with a legend
            # Preassign colors for each unique cluster
            import matplotlib.patches as mpatches
            unique_clusters = sorted(scores_df['cluster'].unique())
            n_clusters = len(unique_clusters)
            # Create a color mapping for each cluster using tab20 colormap.
            # We normalize by (n_clusters - 1) to cover the colormap range.
            colors = {cluster: plt.cm.tab20(cluster / (n_clusters - 1)) for cluster in unique_clusters}
            # Plot the scatter using the preassigned colors
            plt.figure()
            plt.scatter(
                scores_df['combined_score'], 
                scores_df['moran'], 
                c=[colors[cluster] for cluster in scores_df['cluster']], 
                s=2
            )
            plt.xlabel("Combined Score")
            plt.ylabel("Moran")
            plt.title("Feature Selection Clustering")
            # Create the legend using the same color mapping
            handles = [mpatches.Patch(color=colors[cluster], label=f"Cluster {cluster}") for cluster in unique_clusters]
            plt.legend(handles=handles, title="Clusters")
            plt.show()
    
            # --- Output CSV file with the cluster assignments for each m/z feature
            cluster_assignments = scores_df[['cluster']]
            cluster_assignments.to_csv("cluster_assignments.csv")
            print("Cluster assignments CSV saved as 'cluster_assignments.csv'.")
    
            # --- Interactive manual cluster selection:
            user_input = input("Enter the cluster numbers you want to keep (comma-separated), "
                               "or press Enter to auto-select top clusters: ")
            if user_input.strip():
                try:
                    keep_clusters = [int(x.strip()) for x in user_input.split(",")]
                except Exception as e:
                    print("Error parsing input, using automatic selection.")
                    keep_clusters = None
            else:
                keep_clusters = None
    
            # --- Cluster selection: use manual selection if provided, else automatic selection.
            if keep_clusters is not None:
                scores_df = scores_df[scores_df['cluster'].isin(keep_clusters)]
            else:
                cluster_means = np.sqrt(scores_df.groupby('cluster')['combined_score'].mean()**2 + 
                                          2 * scores_df.groupby('cluster')['moran'].mean()**2)
                threshold = np.percentile(cluster_means, 50)
                best_clusters = cluster_means[cluster_means >= threshold].index
                scores_df = scores_df[scores_df['cluster'].isin(best_clusters)]
            selected_features = set(scores_df.index.astype(str))
        else:
            raise ValueError("Invalid modality. Choose from 'moran', 'combined', or use mz_vals for manual override.")
    
        # --- Final filtering: remove features flagged as untrustworthy (those whose names contain '_db')
        if remove_untrustworthy:
            selected_features = {f for f in selected_features if "_db" not in f}
    
        # --- Subset the AnnData object to only the selected features.
        # Ensure that the features in the selection actually exist in adata.var_names.
        final_features = [f for f in self.adata.var_names if f in selected_features]
        feature_selected_adata = self.adata[:, final_features].copy()
    
        # Annotate the AnnData object with the feature selection scores table for later reference.
        feature_selected_adata.uns["feature_selection_scores"] = scores_df if 'scores_df' in locals() else None
    
        # Save the scores table to a CSV file.
        scores_df.to_csv(output_csv)
        
        self.adata = feature_selected_adata
    
    
    def _rank_features_by_combined_score(self, temp_adata):
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
        import numpy as np
    
        sections = temp_adata.obsm['spatial']
        unique_sections = np.unique(sections)
    
        var_of_vars = []
        mean_of_vars = []
    
        # Evaluate each feature
        for i in range(temp_adata.X.shape[1]):
            feature_values = temp_adata.X[:, i]
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
        df_input = pd.DataFrame(self.adata.X)
        p2 = df_input.quantile(lower_quantile)
        p98 = df_input.quantile(upper_quantile)

        arr = df_input.values
        p2_vals = p2.values
        p98_vals = p98.values

        normalized = (arr - p2_vals) / (p98_vals - p2_vals)
        clipped = np.clip(normalized, 0, 1)

        df_norm = pd.DataFrame(
            clipped,
            columns=self.adata.var_names,
            index=self.adata.obs_names,
        )
        self.adata.obsm['X_01norm'] = df_norm
        
        
    def rename_features(
        self,
        peaks_df,
        annotation_col='Lipid',
        score_col="Score",
        min_score=None,
        fallback_to_original=True
    ):
        """
        Rename and optionally filter features in self.adata based on an annotation DataFrame.

        Parameters
        ----------
        peaks_df : pd.DataFrame
            DataFrame returned by `annotate_molecules`, indexed by original m/z strings,
            containing at least the annotation_col and (optionally) a score column.
        annotation_col : str, optional
            Column in peaks_df to use for renaming. Default 'LIPIDMAPS'.
        score_col : str, optional
            Column in peaks_df containing a numeric score for the annotation match.
        min_score : float, optional
            Minimum score required to apply a rename. Any feature whose annotation score
            is below this (or missing) will be dropped entirely.
        fallback_to_original : bool, optional
            If True, and a feature has no annotation or score_col isn't provided,
            its original m/z name is retained; otherwise its name becomes None.

        Returns
        -------
        None
            Updates `self.adata` in place: filters out low-scoring features and renames the rest.
        """
        import pandas as pd
        from collections import Counter

        # Validate inputs
        if annotation_col not in peaks_df.columns:
            raise KeyError(f"Column '{annotation_col}' not found in peaks_df")
        if score_col and score_col not in peaks_df.columns:
            raise KeyError(f"Score column '{score_col}' not found in peaks_df")

        # Build mappings
        name_map = peaks_df[annotation_col].to_dict()
        score_map = peaks_df[score_col].to_dict() if score_col else {}

        original_vars = list(self.adata.var_names)
        keep_vars = []
        new_names = []

        for var in original_vars:
            # Determine if we drop based on score threshold
            if score_col and min_score is not None and var in score_map:
                sc = score_map.get(var)
                # drop if score missing or below threshold
                if pd.isna(sc) or sc < min_score:
                    continue

            # Decide new name
            if var in name_map and pd.notna(name_map[var]):
                new_name = str(name_map[var])
            else:
                new_name = var if fallback_to_original else None

            keep_vars.append(var)
            new_names.append(new_name)

        # Subset AnnData to only kept features
        self.adata = self.adata[:, keep_vars].copy()
        self.adata.var['old_feature_names'] = self.adata.var_names
        
        # Ensure unique names: append suffix to duplicates
        counts = Counter(new_names)
        dup_counters = {n:0 for n,c in counts.items() if c>1 and n is not None}
        unique_names = []
        for nm in new_names:
            if nm in dup_counters:
                dup_counters[nm] += 1
                unique_names.append(f"{nm}_{dup_counters[nm]}")
            else:
                unique_names.append(nm)

        # Assign new var_names
        self.adata.var_names = unique_names

    def lipid_properties(self, color_map_file="lipidclasscolors.h5ad"):
        """
        Extract basic lipid properties from a list of lipid names.

        Parameters
        ----------
        adata : sc.AnnData
            The AnnData object with pixel-wise intensities.
        color_map_file : str, optional
            Path to an HDF5 file containing a DataFrame with 'classcolors'.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing columns: [lipid_name, class, carbons, insaturations,
            insaturations_per_Catom, broken, color].
        """

        lipid_names = self.adata.var_names.values
        
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


    def reaction_network(self, lipid_props_df, premanannot=None):
        """
        Extract a reaction network based on lipid classes and transformation rules.

        Parameters
        ----------
        lipid_props_df : pd.DataFrame
            A DataFrame with lipid properties (index=lipid_name, columns=class,carbons,insaturations,...).
        Returns
        -------
        pd.DataFrame
            Filtered premanannot that satisfies the transformation rules.
        """
        if premanannot is None:
            all_pairs = list(itertools.product(lipid_props_df.index, lipid_props_df.index))
            premanannot = pd.DataFrame(all_pairs, columns=['reagent', 'product'])
            
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
