"""
plotting.py – Production-ready plotting module for EUCLID

All plots are saved in a folder named "plots" (created if it does not exist) in PDF format.
Scatter plots are rasterized and other elements remain vectorial.
All adjustable parameters are exposed in the function signatures.
This module integrates your original draft details – for example, median-based vmin/vmax calculations,
dynamic grid layouts for multi-section plots, filtering by metadata, and hierarchical lipizone plotting
(using a user-provided hierarchical level to determine modal "lipizone_color").
  
Expected AnnData structure example:
    obs: 'SectionID', 'x', 'y', 'Path', 'Sample', 'Sex', 'Condition', 'Section', 'BadSection',
         'X_Leiden', 'X_Euclid', 'lipizone_colors', 'allen_division', 'boundary', 'acronym', etc.
    uns: 'feature_selection_scores', 'neighbors'
    obsm: 'X_NMF', 'X_Harmonized', 'X_approximated', 'X_TSNE', 'X_restored'
    obsp: 'distances', 'connectivities'
"""

import os
import re
import math
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, rgb_to_hsv, hsv_to_rgb
from matplotlib.cm import ScalarMappable
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text
from PyPDF2 import PdfMerger

mpl.rcParams['pdf.fonttype'] = 42

# Ensure the "plots" folder exists.
PLOTS_DIR = "plots"
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

class Plotting:
    """
    Provides a suite of plotting functions for EUCLID outputs.

    Parameters
    ----------
    adata : sc.AnnData
        The AnnData object containing processed MSI data.
    coordinates : pd.DataFrame
        DataFrame with spatial coordinates (e.g. 'Section', 'xccf', 'yccf', 'zccf').
    extra_data : dict, optional
        Additional data (e.g. feature selection scores) stored in adata.uns or elsewhere.
    """
    def __init__(self, adata, extra_data=None):
        self.adata = adata
        self.coordinates = self.adata.obs[['Section', 'xccf', 'yccf', 'zccf']]
        self.extra_data = extra_data if extra_data is not None else {}

    @staticmethod
    def extract_class(lipid_name: str) -> str:
        """
        Extract the lipid class from a lipid name.
        
        Parameters
        ----------
        lipid_name : str
            Lipid name (e.g., "PC O-36:4").
        
        Returns
        -------
        str
            Extracted lipid class (e.g., "PC O").
        """
        m = re.match(r'^([A-Za-z0-9]+(?:\s?O)?)[\s-]', lipid_name)
        if m:
            return m.group(1)
        else:
            return lipid_name.split()[0]

    def plot_lipid_class_pie(self, lipid_df,
                             show_inline: bool = True) -> None:
        """
        Plot a pie chart showing the distribution of lipid classes.
        
        Parameters
        ----------
        lipid_df : pd.DataFrame
            DataFrame containing lipid information.
        class_column : str, optional
            Column name for lipid classes.
        show_inline : bool, optional
            If True, display the plot inline.
        """
        lipid_df["color"] = lipid_df["color"].fillna("black")
        class_counts = lipid_df["class"].value_counts()
        color_dict = lipid_df.drop_duplicates("class").set_index("class")["color"].to_dict()
        plt.figure(figsize=(8, 8))
        ax = class_counts.plot.pie(colors=[color_dict.get(cls, "black") for cls in class_counts.index],
                                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        ax.set_ylabel('')
        plt.title("Lipid Class Distribution")
        save_path = os.path.join(PLOTS_DIR, "msi_prop.pdf")
        plt.savefig(save_path)
        if show_inline:
            plt.show()
        else:
            plt.close()

    #def plot_moran_by_class(self, annowmoran: pd.DataFrame, show_inline: bool = True) -> None:
 

    def plot_lipid_distribution(self,
                                  lipid: str,
                                  section_filter: list = None,
                                  metadata_filter: dict = None,
                                  lipizone_filter: dict = None,
                                  x_range: tuple = None,
                                  y_range: tuple = None,
                                  z_range: tuple = None,
                                  layout: tuple = None,
                                  point_size: float = 0.5,
                                  show_contours: bool = True,
                                  contour_column: str = "boundary",
                                  show_inline: bool = True) -> None:
        """
        Plot the spatial distribution of a given lipid with extensive filtering and cropping options.
        
        Options:
          - Filter by sections (section_filter).
          - Filter by metadata (metadata_filter, e.g. {"division": [...]})
          - Filter by lipizone (lipizone_filter, e.g. {"lipizone_names": [...]})
          - Crop by x, y, and/or z ranges.
          - Layout is computed dynamically (aiming for a ~4:3 page ratio) if not provided.
          - Optionally overlay anatomical contours.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with spatial columns ('zccf', 'yccf', 'xccf', 'Section') and lipid values.
        lipid : str
            Column name (feature) to plot.
        section_filter : list, optional
            List of section numbers to include.
        metadata_filter : dict, optional
            Dictionary of {column: [accepted_values]} for filtering.
        lipizone_filter : dict, optional
            Dictionary to filter based on lipizone metadata.
        x_range, y_range, z_range : tuple, optional
            Tuples (min, max) to crop data along the respective axes.
        layout : tuple, optional
            Grid layout (nrows, ncols). If None, computed dynamically.
        point_size : float, optional
            Marker size.
        show_contours : bool, optional
            If True, overlay contours based on contour_column.
        contour_column : str, optional
            Column name indicating anatomical contours.
        show_inline : bool, optional
            If True, display plot inline.
        """
        
        coords = self.adata.obs[["zccf", "yccf", "xccf", "Section", "division", "boundary"]].copy()
        lipid_idx = list(self.adata.var_names).index(lipid)

        lipid_values = self.adata.X[:, lipid_idx].flatten() 
        
        df_lipid = pd.DataFrame({lipid: lipid_values}, index=coords.index)

        data = pd.concat([coords, df_lipid], axis=1).reset_index(drop=True)
        
        df = data.copy()
        if metadata_filter:
            for col, accepted in metadata_filter.items():
                df = df[df[col].isin(accepted)]
        if section_filter is not None:
            df = df[df["Section"].isin(section_filter)]
        if lipizone_filter:
            for col, accepted in lipizone_filter.items():
                df = df[df[col].isin(accepted)]
        if x_range is not None:
            df = df[(df["xccf"] >= x_range[0]) & (df["xccf"] <= x_range[1])]
        if y_range is not None:
            df = df[(df["yccf"] >= y_range[0]) & (df["yccf"] <= y_range[1])]
        if z_range is not None:
            df = df[(df["zccf"] >= z_range[0]) & (df["zccf"] <= z_range[1])]

        unique_sections = sorted(df["Section"].unique())
        n_sections = len(unique_sections)
        if n_sections == 0:
            print("No sections to plot after filtering.")
            return
        # For each section, compute 2nd and 98th percentiles for the given lipid; then take medians.
        results = []
        for sec in unique_sections:
            subset = df[df["Section"] == sec]
            perc2 = subset[lipid].quantile(0.02)
            perc98 = subset[lipid].quantile(0.98)
            results.append([sec, perc2, perc98])
        percentile_df = pd.DataFrame(results, columns=["Section", "2-perc", "98-perc"])
        med2p = percentile_df["2-perc"].median()
        med98p = percentile_df["98-perc"].median()
        # Global axis limits.
        global_min_z = df["zccf"].min()
        global_max_z = df["zccf"].max()
        global_min_y = -df["yccf"].max()
        global_max_y = -df["yccf"].min()
        # Dynamic layout: if not provided, compute a grid aiming for a 4:3 ratio.
        if layout is None:
            ncols = math.ceil(math.sqrt(n_sections * 4 / 3))
            nrows = math.ceil(n_sections / ncols)
        else:
            nrows, ncols = layout
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        for idx, sec in enumerate(unique_sections):
            ax = axes[idx]
            sec_data = df[df["Section"] == sec]
            ax.scatter(sec_data["zccf"], -sec_data["yccf"],
                       c=sec_data[lipid], cmap="plasma", s=point_size,
                       alpha=0.8, rasterized=True, vmin=med2p, vmax=med98p)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_xlim(global_min_z, global_max_z)
            ax.set_ylim(global_min_y, global_max_y)
            ax.set_title(f"Section {sec}", fontsize=10)
            if show_contours and (contour_column in sec_data.columns):
                contour_data = sec_data[sec_data[contour_column] == 1]
                ax.scatter(contour_data["zccf"], -contour_data["yccf"],
                           c="black", s=point_size/2, alpha=0.5, rasterized=True)
        for j in range(idx+1, len(axes)):
            fig.delaxes(axes[j])
            
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(norm=Normalize(vmin=med2p, vmax=med98p), cmap="plasma")
        sm.set_array([])
        fig.colorbar(sm, ax=axes.tolist(), fraction=0.02, pad=0.04)
        
        plt.tight_layout()
        save_path = os.path.join(PLOTS_DIR, f"{lipid}_distribution.pdf")
        plt.savefig(save_path)
        if show_inline:
            plt.show()
        else:
            plt.close()

    def plot_embeddings(self,
                        key: str = "X_Harmonized",
                        currentProgram: int = 0,
                        layout: tuple = (4, 8),
                        point_size: float = 0.5,
                        cmap_name: str = "PuOr",
                        show_inline: bool = True) -> None:
        """
        Plot spatial embeddings from adata.obsm[key], coloring by the chosen embedding‐column index.

        For each section (as found in adata.obs["Section"]), compute that section’s
        2nd and 98th percentile of embedding‐column `currentProgram`. Then take the
        median across all sections to fix vmin/vmax. Plot each section in its own subplot.

        Parameters
        ----------
        key : str, optional
            Key into adata.obsm where the embeddings array lives.
        currentProgram : int, optional
            Integer index of the embedding‐column (i.e. adata.obsm[key][:, currentProgram]).
        layout : tuple, optional
            (nrows, ncols) for the subplot grid. E.g. (1, 3) to get three side-by-side panels.
        point_size : float, optional
            Marker size for the scatter.
        cmap_name : str, optional
            A matplotlib colormap name.
        show_inline : bool, optional
            If True, calls plt.show(); otherwise closes after saving.
        """
        # 1) Sanity checks
        data = self.adata.obs
        if "Section" not in data.columns:
            raise ValueError("`adata.obs` must contain a 'Section' column.")
        if key not in self.adata.obsm:
            raise ValueError(f"Key '{key}' not found in adata.obsm.")
        
        embeddings = self.adata.obsm[key]
        if not isinstance(embeddings, np.ndarray):
            raise ValueError(f"adata.obsm['{key}'] must be a NumPy array (got {type(embeddings)}).")
        
        n_cells, n_dims = embeddings.shape
        if not (0 <= currentProgram < n_dims):
            raise IndexError(f"currentProgram={currentProgram} is out of bounds for embeddings with {n_dims} columns.")
        
        # 2) Decide which sections to plot
        sections = sorted(data["Section"].unique())
        n_plots   = len(sections)
        nrows, ncols = layout
        total_axes = nrows * ncols
        
        if n_plots > total_axes:
            raise ValueError(
                f"You requested layout={layout}, which has {total_axes} subplots, "
                f"but your data contains {n_plots} distinct sections. "
                "Please increase nrows/ncols so that nrows*ncols ≥ number_of_sections."
            )
        
        # 3) Compute 2nd/98th percentile for each section’s embedding‐column
        results = []
        for sec in sections:
            mask = (data["Section"] == sec).to_numpy()
            if np.count_nonzero(mask) < 2:
                # skip if fewer than 2 cells in that section
                continue
            vals = embeddings[mask, currentProgram]
            p2  = np.quantile(vals, 0.02)
            p98 = np.quantile(vals, 0.98)
            results.append((sec, p2, p98))
        
        if len(results) == 0:
            raise RuntimeError("No section had ≥2 cells to compute percentiles.")
        
        pct_df = pd.DataFrame(results, columns=["Section", "2-perc", "98-perc"])
        med2p  = pct_df["2-perc"].median()
        med98p = pct_df["98-perc"].median()
        
        # 4) Create the figure & axes
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 2.5, nrows * 2.5),
                                 squeeze=False)
        axes = axes.flatten()
        cmap = plt.get_cmap(cmap_name)
        
        # 5) Plot each section in its own subplot
        for idx, sec in enumerate(sections):
            ax   = axes[idx]
            mask = (data["Section"] == sec).to_numpy()
            
            # If no cells in this section, just turn off axis
            if not mask.any():
                ax.axis("off")
                continue
            
            coords  = data.loc[mask, ["zccf", "yccf"]]
            emb_vals = embeddings[mask, currentProgram]
            
            ax.scatter(
                coords["zccf"],
                -coords["yccf"],
                c=emb_vals,
                cmap=cmap,
                s=point_size,
                rasterized=True,
                vmin=med2p,
                vmax=med98p
            )
            ax.set_aspect("equal")
            ax.axis("off")
        
        # 6) Turn off any leftover axes (if layout > number of sections)
        for leftover in range(n_plots, total_axes):
            axes[leftover].axis("off")
        
        # 7) Add a colorbar on the right
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = Normalize(vmin=med2p, vmax=med98p)
        sm   = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # Required for colorbar compatibility
        fig.colorbar(sm, cax=cbar_ax)
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        # 8) Save + show/close
        save_path = os.path.join(PLOTS_DIR, f"{currentProgram}_embeddings.pdf")
        plt.savefig(save_path)
        
        if show_inline:
            plt.show()
        else:
            plt.close()

    def plot_lipizones(self, levels: pd.DataFrame,
                       lipizone_col: str = "lipizone_names",
                       section_col: str = "Section",
                       level_filter: str = None,
                       show_inline: bool = True) -> None:
        """
        Plot lipizones (clusters) with flexible options.
        
        Options include:
          - Filtering by metadata (e.g., division) can be applied before.
          - If level_filter is provided (e.g., "level_2"), then for each unique combination
            of levels up to that level the effective color is taken as the mode of 'lipizone_color'.
          - Layout is computed dynamically to yield a roughly 4:3 page.
        
        Parameters
        ----------
        levels : pd.DataFrame
            DataFrame including spatial coordinates ('zccf','yccf'), section (section_col),
            and lipizone labels.
        lipizone_col : str, optional
            Column with lipizone labels.
        section_col : str, optional
            Column with section numbers.
        level_filter : str, optional
            Hierarchical level to use for color determination (e.g., "level_1", "level_2").
        show_inline : bool, optional
            Whether to display the plot inline.
        """
        df = levels.copy()
        # If a level_filter is provided, compute the modal lipizone_color for each unique combination
        if level_filter is not None and level_filter in df.columns:
            grouping_cols = [col for col in df.columns if col.startswith("level_") and int(col.split("_")[1]) <= int(level_filter.split("_")[1])]
            modal = df.groupby(grouping_cols)["lipizone_color"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "gray")
            # Map each observation to its modal color
            def get_modal(row):
                key = tuple(row[col] for col in grouping_cols)
                return modal.get(key, row["lipizone_color"])
            df["plot_color"] = df.apply(get_modal, axis=1)
        else:
            df["plot_color"] = df["lipizone_color"]

        unique_sections = sorted(df[section_col].unique())
        n_sections = len(unique_sections)
        if n_sections == 0:
            print("No sections to plot.")
            return
        global_min_z = df["zccf"].min()
        global_max_z = df["zccf"].max()
        global_min_y = -df["yccf"].max()
        global_max_y = -df["yccf"].min()
        ncols = math.ceil(math.sqrt(n_sections * 4 / 3))
        nrows = math.ceil(n_sections / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = axes.flatten()
        for idx, sec in enumerate(unique_sections):
            ax = axes[idx]
            sec_data = df[df[section_col] == sec]
            ax.scatter(sec_data["zccf"], -sec_data["yccf"],
                       c=sec_data["plot_color"], s=0.5, alpha=0.8, rasterized=True)
            ax.axis("off")
            ax.set_aspect("equal")
            ax.set_xlim(global_min_z, global_max_z)
            ax.set_ylim(global_min_y, global_max_y)
            ax.set_title(f"Section {sec}", fontsize=10)
        for j in range(idx+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        save_path = os.path.join(PLOTS_DIR, "merged_lipizones.pdf")
        # Optionally merge individual PDFs; here we simply save one multi-panel PDF.
        plt.savefig(save_path)
        if show_inline:
            plt.show()
        else:
            plt.close()

    def plot_tsne(self,
                  attribute: str = None,
                  attribute_type: str = "lipid",
                  tsne_key: str = "X_TSNE",
                  program_key: str = None,
                  program_index: int = 0,
                  cmap: str = "plasma",
                  point_size: float = 0.5,
                  show_inline: bool = True):
        """
        t-SNE scatter plot colored by any of:
          - a lipid (continuous; column of adata.X),
          - a peak (continuous; either var_names or a column of an obsm array),
          - a program (continuous; one column of an embedding in adata.obsm),
          - a categorical metadata (discrete; adata.obs[column]),
          - or a direct color column (hex codes or named colors; adata.obs[column], e.g. 'lipizone_color').

        Parameters
        ----------
        attribute : str, optional
            Name of the feature to color by:
              • If attribute_type=="lipid", must be in adata.var_names.
              • If attribute_type=="peak", either in adata.var_names or a key in adata.obsm.
              • If attribute_type=="program", attribute is ignored; use program_key & program_index.
              • If attribute_type=="categorical" or "color", must be a column in adata.obs.
        attribute_type : str, optional
            One of {"lipid", "peak", "program", "categorical", "color"}.  Defaults to "lipid".
        tsne_key : str, optional
            Key in adata.obsm where your 2D t-SNE coordinates live (default "X_TSNE").
        program_key : str, optional
            If attribute_type=="program", this is the obsm key for the embedding (e.g. "X_Harmonized").
        program_index : int, optional
            Column index within the embedding when attribute_type=="program" or ("peak" via obsm).
        cmap : str, optional
            A matplotlib colormap name for continuous or categorical mapping.
        point_size : float, optional
            Marker size for each cell.
        show_inline : bool, optional
            If True, calls plt.show(); otherwise saves and closes.
        """
        # 1) Retrieve t-SNE coordinates
        if tsne_key not in self.adata.obsm:
            raise ValueError(f"`{tsne_key}` not found in adata.obsm; cannot plot t-SNE.")
        tsne_coords = self.adata.obsm[tsne_key]
        if tsne_coords.shape[1] < 2:
            raise ValueError(f"adata.obsm['{tsne_key}'] must have at least 2 dimensions.")
        xs, ys = tsne_coords[:, 0], tsne_coords[:, 1]

        # 2) Build the color array based on attribute_type
        color_array = None
        vmin = vmax = None
        legend_handles = None

        # Continuous: lipid
        if attribute_type == "lipid":
            if attribute not in self.adata.var_names:
                raise ValueError(f"Lipid '{attribute}' not found in adata.var_names.")
            var_idx = list(self.adata.var_names).index(attribute)
            vals = self.adata.X[:, var_idx]
            # If sparse matrix, convert that one column to dense
            values = vals.A.flatten() if hasattr(vals, "A") else vals
            color_array = values
            # Clip at 2nd/98th percentiles
            p2, p98 = np.percentile(color_array, [2, 98])
            vmin, vmax = p2, p98

        # Continuous: peak
        elif attribute_type == "peak":
            # (a) if attribute is in var_names
            if attribute in self.adata.var_names:
                var_idx = list(self.adata.var_names).index(attribute)
                vals = self.adata.X[:, var_idx]
                values = vals.A.flatten() if hasattr(vals, "A") else vals
                color_array = values
                p2, p98 = np.percentile(color_array, [2, 98])
                vmin, vmax = p2, p98
            # (b) else if attribute is a key in obsm
            elif attribute in self.adata.obsm:
                mat = self.adata.obsm[attribute]
                if not isinstance(mat, np.ndarray):
                    raise ValueError(f"adata.obsm['{attribute}'] must be a NumPy array.")
                if program_index < 0 or program_index >= mat.shape[1]:
                    raise IndexError(
                        f"program_index={program_index} out of bounds for obsm['{attribute}'].shape={mat.shape}."
                    )
                color_array = mat[:, program_index]
                p2, p98 = np.percentile(color_array, [2, 98])
                vmin, vmax = p2, p98
            else:
                raise ValueError(
                    f"'{attribute}' not found in adata.var_names or adata.obsm for peaks."
                )

        # Continuous: program
        elif attribute_type == "program":
            if program_key is None:
                raise ValueError("When attribute_type=='program', you must specify program_key.")
            if program_key not in self.adata.obsm:
                raise ValueError(f"'{program_key}' not found in adata.obsm.")
            emb = self.adata.obsm[program_key]
            if not isinstance(emb, np.ndarray):
                raise ValueError(f"adata.obsm['{program_key}'] must be a NumPy array.")
            n_dims = emb.shape[1]
            if program_index < 0 or program_index >= n_dims:
                raise IndexError(f"program_index={program_index} out of bounds for shape {emb.shape}.")
            color_array = emb[:, program_index]
            p2, p98 = np.percentile(color_array, [2, 98])
            vmin, vmax = p2, p98

        # Discrete: categorical metadata
        elif attribute_type == "categorical":
            if attribute not in self.adata.obs.columns:
                raise ValueError(f"Categorical column '{attribute}' not found in adata.obs.")
            series = self.adata.obs[attribute].astype(str)
            # Check if column holds valid color strings (hex or names)
            from matplotlib.colors import is_color_like
            all_colors = series.dropna().unique()
            if all(is_color_like(c) for c in all_colors):
                # Treat as direct color column
                color_array = series.values
                # No legend for this branch (colors are explicit)
            else:
                # Map each category to a numeric index + discrete cmap + legend
                cats = series.values
                uniq = np.unique(cats)
                lut = {u: i for i, u in enumerate(uniq)}
                numeric = np.array([lut[c] for c in cats])
                color_array = numeric
                # Build discrete colormap
                n_colors = len(uniq)
                cmap_obj = mpl.cm.get_cmap(cmap, n_colors)
                # Build legend handles
                handles = []
                for i, u in enumerate(uniq):
                    patch = mpl.patches.Patch(color=cmap_obj(i), label=u)
                    handles.append(patch)
                legend_handles = handles
                cmap = cmap_obj

        # Direct: hex/color column
        elif attribute_type == "color":
            if attribute not in self.adata.obs.columns:
                raise ValueError(f"Color column '{attribute}' not found in adata.obs.")
            series = self.adata.obs[attribute].astype(str)
            from matplotlib.colors import is_color_like
            all_colors = series.dropna().unique()
            if not all(is_color_like(c) for c in all_colors):
                raise ValueError(f"Not all values in '{attribute}' are valid colors.")
            color_array = series.values
            # No colormap, no vmin/vmax

        else:
            raise ValueError("attribute_type must be one of 'lipid', 'peak', 'program', 'categorical', or 'color'.")

        # 3) Scatter plot
        plt.figure(figsize=(6, 6))
        sc_kwargs = {"s": point_size, "alpha": 0.8, "rasterized": True}

        if attribute_type in {"lipid", "peak", "program"}:
            sc_kwargs["c"] = color_array
            sc_kwargs["cmap"] = cmap
            sc_kwargs["vmin"] = vmin
            sc_kwargs["vmax"] = vmax
        else:
            # categorical or color: c is directly the array of strings or numeric + discrete cmap
            sc_kwargs["c"] = color_array
            if legend_handles is None:
                # direct hex/named colors → no cmap
                pass
            else:
                sc_kwargs["cmap"] = cmap

        plt.scatter(xs, ys, **sc_kwargs)
        plt.axis("off")
        plt.gca().set_aspect("equal")

        # 4) Colorbar or legend
        if attribute_type in {"lipid", "peak", "program"}:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            ax = plt.gca()

            # (2) create the ScalarMappable
            sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
            sm.set_array([])

            # (3) attach the colorbar to that Axes
            cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            
            if attribute_type == "lipid":
                cbar.set_label(attribute, rotation=270, labelpad=15)
            elif attribute_type == "peak":
                cbar.set_label(f"Peak '{attribute}'", rotation=270, labelpad=15)
            else:
                cbar.set_label(f"{program_key}[..., {program_index}]", rotation=270, labelpad=15)

        elif legend_handles is not None:
            plt.legend(handles=legend_handles, title=attribute, loc="right", bbox_to_anchor=(1.3, 0.5))

        # 5) Save + show or close
        if attribute_type == "lipid":
            fname = f"tsne_lipid_{attribute.replace(' ', '_')}"
        elif attribute_type == "peak":
            fname = f"tsne_peak_{attribute.replace(' ', '_')}"
        elif attribute_type == "program":
            fname = f"tsne_program_{program_key}_{program_index}"
        elif attribute_type == "categorical":
            fname = f"tsne_cat_{attribute}"
        else:  # color
            fname = f"tsne_colorcol_{attribute}"
        save_path = os.path.join(PLOTS_DIR, f"{fname}.pdf")
        plt.tight_layout()
        plt.savefig(save_path)

        if show_inline:
            plt.show()
        else:
            plt.close()


    def plot_sorted_heatmap(self, data: pd.DataFrame, vmin: float = None, vmax: float = None,
                            xticklabels: bool = False, yticklabels: bool = False,
                            xlabel: str = "", ylabel: str = "", show_inline: bool = False) -> None:
        """
        Plot a sorted heatmap with user-defined parameters.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to plot.
        vmin, vmax : float, optional
            Color scale limits; default to 2nd and 98th percentiles.
        xticklabels, yticklabels : bool, optional
            Whether to show axis tick labels.
        xlabel, ylabel : str, optional
            Axis labels.
        show_inline : bool, optional
            Whether to display inline.
        """
        if vmin is None:
            vmin = np.percentile(data.values, 2)
        if vmax is None:
            vmax = np.percentile(data.values, 98)
        # Sort columns via hierarchical clustering
        L = linkage(squareform(pdist(data.T)), method='weighted', optimal_ordering=True)
        order_cols = leaves_list(L)
        sorted_data = data.iloc[:, order_cols]
        order_rows = np.argsort(np.argmax(sorted_data.values, axis=1))
        sorted_data = sorted_data.iloc[order_rows, :]
        plt.figure(figsize=(20, 5))
        ax = sns.heatmap(sorted_data, cmap="Grays", cbar_kws={'label': 'Enrichment'},
                         xticklabels=xticklabels, yticklabels=yticklabels, vmin=vmin, vmax=vmax)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', left=False, right=False)
        plt.tight_layout()
        save_path = os.path.join(PLOTS_DIR, "sorted_heatmap.pdf")
        plt.savefig(save_path)
        if show_inline:
            plt.show()
        else:
            plt.close()

    def plot_sample_correlation_pca(self, centroids: pd.DataFrame, show_inline: bool = False) -> None:
        """
        Plot 3D PCA of sample centroids and a clustermap of their correlation.
        
        Parameters
        ----------
        centroids : pd.DataFrame
            DataFrame of normalized lipid expression centroids.
        show_inline : bool, optional
            Whether to display the plots inline.
        """
        scaler = StandardScaler()
        scaled = pd.DataFrame(scaler.fit_transform(centroids), index=centroids.index, columns=centroids.columns)
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(scaled)
        var_exp = pca.explained_variance_ratio_ * 100
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
                   c="blue", s=350, alpha=1)
        ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% var)')
        ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% var)')
        ax.set_zlabel(f'PC3 ({var_exp[2]:.1f}% var)')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        plt.title("3D PCA of Centroids")
        save_path = os.path.join(PLOTS_DIR, "pca_centroids.pdf")
        plt.tight_layout()
        plt.savefig(save_path)
        if show_inline:
            plt.show()
        else:
            plt.close()
        cg = sns.clustermap(centroids.T.corr(), cmap="viridis", figsize=(10, 10))
        cg.savefig(os.path.join(PLOTS_DIR, "sample_correlation_heatmap.pdf"))
        plt.close()

    def plot_differential_barplot(self, diff_df: pd.DataFrame, indices_to_label: list,
                                  ylabel: str = "log2 fold change", show_inline: bool = False) -> None:
        """
        Plot a sorted barplot for differential lipids colored by their class.
        
        Parameters
        ----------
        diff_df : pd.DataFrame
            DataFrame of differential values with index as lipid names and a 'color' column.
        indices_to_label : list
            List of positions to annotate.
        ylabel : str, optional
            Y-axis label.
        show_inline : bool, optional
            Whether to display inline.
        """
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(diff_df)), diff_df.iloc[:, 0], color=diff_df["color"])
        texts = []
        for idx in indices_to_label:
            x = idx
            y = diff_df.iloc[idx, 0]
            label = diff_df.index[idx]
            texts.append(plt.text(x, y, label, ha="center", va="bottom"))
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
                    expand_points=(1.5, 1.5))
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.ylabel(ylabel)
        plt.xticks([])
        plt.tight_layout()
        save_path = os.path.join(PLOTS_DIR, "differential_barplot.pdf")
        plt.savefig(save_path)
        if show_inline:
            plt.show()
        else:
            plt.close()

    # Placeholders for Plotly-based rendering, clock visualization, and movie generation.
    def plot_3d_rendering_plotly(self):
        """Placeholder for 3D rendering using Plotly."""
        print("plot_3d_rendering_plotly not implemented yet.")

    def plot_2d_rendering_plotly(self):
        """Placeholder for 2D rendering using Plotly with zoomability."""
        print("plot_2d_rendering_plotly not implemented yet.")

    def plot_treemap(self):
        """Placeholder for lipizones interactive treemap."""
        print("plot_treemap not implemented yet.")

    def make_movie(self):
        """Placeholder for making a movie from the data."""
        print("make_movie not implemented yet.")
