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
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.animation import FuncAnimation
import cv2
import glob
import subprocess
from threadpoolctl import threadpool_limits
from scipy import sparse
from tqdm import tqdm
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
import gc
from skimage.morphology import binary_erosion
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
import ast
import matplotlib.colors as mcolors

# Set PDF font type
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
        DataFrame with spatial coordinates (e.g. 'SectionID', 'xccf', 'yccf', 'zccf').
    extra_data : dict, optional
        Additional data (e.g. feature selection scores) stored in adata.uns or elsewhere.
    analysis_name : str, optional
        Prefix for all output files and plot directories. Default is "analysis".
    """
    def __init__(self, adata, extra_data=None, analysis_name="analysis"):
        self.adata = adata
        self.coordinates = self.adata.obs[['SectionID', 'xccf', 'yccf', 'zccf']]
        self.extra_data = extra_data if extra_data is not None else {}
        self.analysis_name = analysis_name
        self.plots_dir = f"{analysis_name}_plots"
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

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
        save_path = os.path.join(self.plots_dir, "msi_prop.pdf")
        plt.savefig(save_path)
        if show_inline:
            plt.show()
        else:
            plt.close()

    def plot_chainlength_insaturation_hist(self, lipid_props_df: pd.DataFrame, show_inline: bool = True) -> None:
        """
        Plot a nested histogram of chain length and insaturations from a lipid properties DataFrame.

        Parameters
        ----------
        lipid_props_df : pd.DataFrame
            DataFrame containing at least 'carbons' and 'insaturations' columns.
        show_inline : bool, optional
            If True, display the plot inline.
        """
        df = lipid_props_df.copy()
        cmap = plt.cm.Reds
        vmin = df['insaturations'].min()
        vmax = df['insaturations'].max()
        bins = np.linspace(df['carbons'].min(), df['carbons'].max(), 20)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        df['carbon_bins'] = pd.cut(df['carbons'], bins, right=False)
        grouped = df.groupby(['carbon_bins', 'insaturations']).size().unstack(fill_value=0)
        grouped_normalized = grouped  # Optionally normalize if needed
        plt.figure(figsize=(10, 6))
        bottoms = np.zeros(len(bin_centers))
        for insaturation in grouped_normalized.columns:
            color = cmap((insaturation - vmin) / (vmax - vmin))
            plt.bar(
                bin_centers,
                grouped_normalized[insaturation],
                width=np.diff(bins),
                bottom=bottoms,
                color=color,
                label=f"Insaturation {insaturation:.1f}",
                edgecolor="none",
            )
            bottoms += grouped_normalized[insaturation]
        plt.title('Total Chain Length')
        for spine in ['top', 'right', 'left', 'bottom']:
            plt.gca().spines[spine].set_visible(False)
        plt.tick_params(axis='x', which='both', bottom=False, top=False)
        plt.tick_params(axis='y', which='both', left=False, right=False)
        plt.tight_layout()
        plt.legend(title='Insaturations')
        save_path = os.path.join(self.plots_dir, "chainlength_insat_LCMS.pdf")
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
        lipid : str
            The lipid name to plot.
        section_filter : list, optional
            List of section IDs to include.
        metadata_filter : dict, optional
            Dictionary where keys are metadata columns and values are lists of allowed values.
        lipizone_filter : dict, optional
            Dictionary for filtering by lipizone properties.
        x_range : tuple, optional
            (x_min, x_max) for cropping.
        y_range : tuple, optional
            (y_min, y_max) for cropping.
        z_range : tuple, optional
            (z_min, z_max) for cropping.
        layout : tuple, optional
            (rows, cols) layout for subplots.
        point_size : float, optional
            Size of the scatter plot points.
        show_contours : bool, optional
            Whether to overlay anatomical contours.
        contour_column : str, optional
            Column name indicating anatomical contours.
        show_inline : bool, optional
            If True, display plot inline.
        """
        
        coords = self.adata.obs[["zccf", "yccf", "xccf", "SectionID", "division", "boundary", list(lipizone_filter.keys())[0]]].copy()
        lipid_idx = list(self.adata.var_names).index(lipid)

        lipid_values = self.adata.X[:, lipid_idx].flatten() 
        
        df_lipid = pd.DataFrame({lipid: lipid_values}, index=coords.index)

        data = pd.concat([coords, df_lipid], axis=1).reset_index(drop=True)
        
        df = data.copy()
        if metadata_filter:
            for col, accepted in metadata_filter.items():
                df = df[df[col].isin(accepted)]
        if section_filter is not None:
            df = df[df["SectionID"].isin(section_filter)]
        if lipizone_filter:
            for col, accepted in lipizone_filter.items():
                df = df[df[col].isin(accepted)]
        if x_range is not None:
            df = df[(df["xccf"] >= x_range[0]) & (df["xccf"] <= x_range[1])]
        if y_range is not None:
            df = df[(df["yccf"] >= y_range[0]) & (df["yccf"] <= y_range[1])]
        if z_range is not None:
            df = df[(df["zccf"] >= z_range[0]) & (df["zccf"] <= z_range[1])]

        unique_sections = sorted(df["SectionID"].unique())
        n_sections = len(unique_sections)
        if n_sections == 0:
            print("No sections to plot after filtering.")
            return
        # For each section, compute 2nd and 98th percentiles for the given lipid; then take medians.
        results = []
        for sec in unique_sections:
            subset = df[df["SectionID"] == sec]
            perc2 = subset[lipid].quantile(0.02)
            perc98 = subset[lipid].quantile(0.98)
            results.append([sec, perc2, perc98])
        percentile_df = pd.DataFrame(results, columns=["SectionID", "2-perc", "98-perc"])
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
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        for idx, sec in enumerate(unique_sections):
            ax = axes[idx]
            sec_data = df[df["SectionID"] == sec]
            
            # Plot background: entire section in grayscale based on lipizone_names
            # Get all data for this section for background
            background_data = self.adata.obs[self.adata.obs["SectionID"] == sec] if "SectionID" in self.adata.obs.columns else self.adata.obs
            
            if "lipizone_names" in background_data.columns and not background_data.empty:
                # Create a mapping from lipizone_names to gray values
                unique_lipizone_names = background_data["lipizone_names"].unique()
                unique_lipizone_names = unique_lipizone_names[pd.notna(unique_lipizone_names)]  # Remove NaN values
                
                if len(unique_lipizone_names) > 0:
                    # Create grayscale colormap - distribute gray values evenly
                    gray_values = np.linspace(0.2, 0.8, len(unique_lipizone_names))  # Avoid pure black/white
                    lipizone_to_gray = dict(zip(unique_lipizone_names, gray_values))
                    
                    # Map each point to its gray value
                    background_colors = background_data["lipizone_names"].map(lipizone_to_gray)
                    
                    # Plot background points using same coordinate system
                    valid_background = background_data.dropna(subset=["zccf", "yccf", "lipizone_names"])
                    if not valid_background.empty:
                        ax.scatter(
                            valid_background["zccf"], 
                            -valid_background["yccf"], 
                            c=background_colors[valid_background.index], 
                            cmap='gray',
                            s=point_size * 0.5,  # Smaller than foreground points
                            alpha=0.3,
                            rasterized=True
                        )
            
            # Plot the lipid data (foreground)
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
        save_path = os.path.join(self.plots_dir, f"{lipid}_distribution.pdf")
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

        For each section (as found in adata.obs["SectionID"]), compute that section's
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
        if "SectionID" not in data.columns:
            raise ValueError("`adata.obs` must contain a 'SectionID' column.")
        if key not in self.adata.obsm:
            raise ValueError(f"Key '{key}' not found in adata.obsm.")
        
        embeddings = self.adata.obsm[key]
        if not isinstance(embeddings, np.ndarray):
            raise ValueError(f"adata.obsm['{key}'] must be a NumPy array (got {type(embeddings)}).")
        
        n_cells, n_dims = embeddings.shape
        if not (0 <= currentProgram < n_dims):
            raise IndexError(f"currentProgram={currentProgram} is out of bounds for embeddings with {n_dims} columns.")
        
        # 2) Decide which sections to plot
        sections = sorted(data["SectionID"].unique())
        n_plots   = len(sections)
        nrows, ncols = layout
        total_axes = nrows * ncols
        
        if n_plots > total_axes:
            raise ValueError(
                f"You requested layout={layout}, which has {total_axes} subplots, "
                f"but your data contains {n_plots} distinct sections. "
                "Please increase nrows/ncols so that nrows*ncols ≥ number_of_sections."
            )
        
        # 3) Compute 2nd/98th percentile for each section's embedding‐column
        results = []
        for sec in sections:
            mask = (data["SectionID"] == sec).to_numpy()
            if np.count_nonzero(mask) < 2:
                # skip if fewer than 2 cells in that section
                continue
            vals = embeddings[mask, currentProgram]
            p2  = np.quantile(vals, 0.02)
            p98 = np.quantile(vals, 0.98)
            results.append((sec, p2, p98))
        
        if len(results) == 0:
            raise RuntimeError("No section had ≥2 cells to compute percentiles.")
        
        pct_df = pd.DataFrame(results, columns=["SectionID", "2-perc", "98-perc"])
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
            mask = (data["SectionID"] == sec).to_numpy()
            
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
        save_path = os.path.join(self.plots_dir, f"{currentProgram}_embeddings.pdf")
        plt.savefig(save_path)
        
        if show_inline:
            plt.show()
        else:
            plt.close()

    def plot_lipizones(self,
                       lipizone_col: str = "lipizone_names",
                       section_col: str = "SectionID",
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
        lipizone_col : str, optional
            Column with lipizone labels.
        section_col : str, optional
            Column with section numbers.
        level_filter : str, optional
            Hierarchical level to use for color determination (e.g., "level_1", "level_2").
        show_inline : bool, optional
            Whether to display the plot inline.
        """
        df = self.adata.obs.copy()
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
        
        # Fill NaN colors with gray (handle categorical columns)
        if df["plot_color"].dtype.name == 'category':
            # Add "gray" to categories if not already present
            if "gray" not in df["plot_color"].cat.categories:
                df["plot_color"] = df["plot_color"].cat.add_categories(["gray"])
            df["plot_color"] = df["plot_color"].fillna("gray")
        else:
            df["plot_color"] = df["plot_color"].fillna("gray")

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
        save_path = os.path.join(self.plots_dir, "merged_lipizones.pdf")
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
        save_path = os.path.join(self.plots_dir, f"{fname}.pdf")
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
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, "sorted_heatmap.pdf")
        plt.savefig(save_path)
        if show_inline:
            plt.show()
        else:
            plt.close()

    def plot_olosorted_lipid_lipizone(self, show_inline: bool = True) -> None:
        """
        Plot a heatmap of 01-normalized lipid data averaged by lipizone_names,
        with both rows and columns sorted by optimal-leaf-ordering using cosine similarity.
        
        Parameters
        ----------
        show_inline : bool, optional
            Whether to display the plot inline.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import MinMaxScaler
        
        # Extract lipidome data
        if 'X_lipidome' not in self.adata.obsm:
            raise ValueError("'X_lipidome' not found in adata.obsm")
        
        if 'lipidome_names' not in self.adata.uns:
            raise ValueError("'lipidome_names' not found in adata.uns")
        
        if 'lipizone_names' not in self.adata.obs:
            raise ValueError("'lipizone_names' not found in adata.obs")
        
        # Get the data
        lipid_data = self.adata.obsm['X_lipidome']
        lipid_names = self.adata.uns['lipidome_names']
        lipizone_names = self.adata.obs['lipizone_names']
        
        # Normalize each lipid individually to 0-1 range (01-normalization per lipid)
        # Scale each column (lipid) separately so each lipid ranges from 0 to 1
        norm_data = np.zeros_like(lipid_data)
        for i in range(lipid_data.shape[1]):
            col_min = lipid_data[:, i].min()
            col_max = lipid_data[:, i].max()
            if col_max > col_min:  # Avoid division by zero
                norm_data[:, i] = (lipid_data[:, i] - col_min) / (col_max - col_min)
            else:
                norm_data[:, i] = 0  # If all values are the same, set to 0
        
        # Create DataFrame
        df = pd.DataFrame(norm_data, columns=lipid_names, index=self.adata.obs_names)
        df['lipizone_names'] = lipizone_names
        
        # Remove rows with NaN lipizone_names
        df = df.dropna(subset=['lipizone_names'])
        
        # Average by lipizone_names
        averaged_df = df.groupby('lipizone_names').mean()
        
        # Compute cosine similarity and convert to distance for clustering
        # For rows (lipizones)
        cos_sim_rows = cosine_similarity(averaged_df.values)
        cos_dist_rows = 1 - cos_sim_rows
        
        # For columns (lipids)  
        cos_sim_cols = cosine_similarity(averaged_df.values.T)
        cos_dist_cols = 1 - cos_sim_cols
        
        # Convert to condensed distance matrices
        row_distances = squareform(cos_dist_rows, checks=False)
        col_distances = squareform(cos_dist_cols, checks=False)
        
        # Perform hierarchical clustering with optimal leaf ordering
        row_linkage = linkage(row_distances, method='average', optimal_ordering=True)
        col_linkage = linkage(col_distances, method='average', optimal_ordering=True)
        
        # Get the optimal ordering
        row_order = leaves_list(row_linkage)
        col_order = leaves_list(col_linkage)
        
        # Reorder the DataFrame
        sorted_df = averaged_df.iloc[row_order, col_order]
        
        # Create the heatmap
        plt.figure(figsize=(20, 12))
        sns.heatmap(sorted_df, 
                    cmap='Reds', 
                    cbar_kws={'label': '01-normalized intensity'},
                    xticklabels=True, 
                    yticklabels=True)
        
        plt.title('Lipid Expression by Lipizone (Optimal Leaf Ordering)')
        plt.xlabel('Lipids')
        plt.ylabel('Lipizone Names')
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.plots_dir, "olosorted_lipid_lipizone_heatmap.pdf")
        plt.savefig(save_path)
        
        if show_inline:
            plt.show()
        else:
            plt.close()

    def plot_sample_correlation_pca(self, centroids: pd.DataFrame, show_inline: bool = False) -> None: ### UNTESTED YET
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
                   c="blue", s=350, alpha=1, rasterized=True)
        ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% var)')
        ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% var)')
        ax.set_zlabel(f'PC3 ({var_exp[2]:.1f}% var)')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        plt.title("3D PCA of Centroids")
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Sample Correlation PCA')
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, "pca_centroids.pdf")
        
        plt.savefig(save_path)
        if show_inline:
            plt.show()
        else:
            plt.close()
        
        # Also create a correlation heatmap
        cg = sns.clustermap(centroids.T.corr(), cmap="viridis", figsize=(10, 10))
        cg.savefig(os.path.join(self.plots_dir, "sample_correlation_heatmap.pdf"))
        plt.close()

    def plot_differential_barplot(self, diff_df: pd.DataFrame, indices_to_label: list, ### UNTESTED YET
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
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, "differential_barplot.pdf")
        plt.savefig(save_path)
        if show_inline:
            plt.show()
        else:
            plt.close()

    def plot_global_lipidomic_similarity(self, show_inline: bool = True, coloring="subclass") -> None:
        """
        Plot global lipidomic similarity patterns by subclass, coloring each pixel by subclass similarity.
        Subclasses are sorted by similarity, assigned a rainbow colormap, and plotted as in plot_lipizones.
        """
        import numpy as np
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        # 1. Average adata.X by coloring
        if coloring not in self.adata.obs.columns:
            raise ValueError("coloring column not found in adata.obs")
        df = pd.DataFrame(self.adata.X, columns=self.adata.var_names, index=self.adata.obs.index)
        df[coloring] = self.adata.obs[coloring].values
        normalized_df = df.groupby(coloring).mean()
        # 2. Sort coloring by similarity (cumulative distance along the mean vectors)
        X = normalized_df.values
        diffs = np.linalg.norm(X[1:] - X[:-1], axis=1)
        cumdist = np.concatenate([[0.0], np.cumsum(diffs)])
        t = (cumdist - cumdist.min()) / (cumdist.max() - cumdist.min())
        # 3. Assign rainbow colormap
        cmap = cm.get_cmap('rainbow')
        rgba_colors = cmap(t)
        hex_colors = [mcolors.to_hex(c) for c in rgba_colors]
        color_dict = dict(zip(normalized_df.index, hex_colors))
        # 4. Assign color to each pixel by coloring
        coloring_colors = self.adata.obs[coloring].map(color_dict)
        # 5. Prepare DataFrame for plotting (like plot_lipizones)
        coords = self.adata.obs[['zccf', 'yccf', 'xccf', 'SectionID']].copy()
        coords[coloring] = self.adata.obs[coloring]
        coords['plot_color'] = coloring_colors.values
        unique_sections = sorted(coords['SectionID'].unique())
        n_sections = len(unique_sections)
        if n_sections == 0:
            print("No sections to plot.")
            return
        global_min_z = coords['zccf'].min()
        global_max_z = coords['zccf'].max()
        global_min_y = -coords['yccf'].max()
        global_max_y = -coords['yccf'].min()
        ncols = int(np.ceil(np.sqrt(n_sections * 4 / 3)))
        nrows = int(np.ceil(n_sections / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
        axes = axes.flatten()
        for idx, sec in enumerate(unique_sections):
            ax = axes[idx]
            sec_data = coords[coords['SectionID'] == sec]
            ax.scatter(sec_data['zccf'], -sec_data['yccf'],
                       c=sec_data['plot_color'], s=0.5, alpha=0.8, rasterized=True)
            ax.axis('off')
            ax.set_aspect('equal')
            ax.set_xlim(global_min_z, global_max_z)
            ax.set_ylim(global_min_y, global_max_y)
            ax.set_title(f"Section {sec}", fontsize=10)
        for j in range(idx+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, "global_lipidomic_similarity.pdf")
        plt.savefig(save_path)
        if show_inline:
            plt.show()
        else:
            plt.close()

    def plot_lipids_rgb_grid(self, lipid_names, section_ids, group_size=3, output_file="lipid_rgb_grid.png"):
        """
        Plot N lipids in groups of 3 as RGB overlays for each section, in a grid.
        Each row is a group of 3 lipids, each column is a section.
        Uses z_index and y_index from adata.obs for spatial coordinates.

        Parameters
        ----------
        lipid_names : list
            List of lipid names (must be in adata.var_names).
        section_ids : list
            List of section IDs to plot (must match adata.obs['SectionID']).
        group_size : int, optional
            Number of lipids per group (default 3).
        output_file : str, optional
            Output filename (PNG, saved in plots directory).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import random

        # Prepare data
        data = pd.DataFrame(self.adata.X, columns=self.adata.var_names, index=self.adata.obs_names)
        data = data.join(self.adata.obs[['SectionID', 'z_index', 'y_index']])
        
        # Group lipids
        lipid_groups = [lipid_names[i:i+group_size] for i in range(0, len(lipid_names), group_size)]
        
        n_rows = len(lipid_groups)
        n_cols = len(section_ids)
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4*n_cols, 2.5*n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)
        if n_cols == 1:
            axes = np.expand_dims(axes, 1)
        plt.subplots_adjust(wspace=0, hspace=0.01)

        # Helper for random color
        def random_color():
            return np.random.uniform(0.3, 1.0, size=3)

        # Set global axis limits - mirror y-axis limits
        x_min, x_max = data['z_index'].min(), data['z_index'].max()
        y_min, y_max = -data['y_index'].max(), -data['y_index'].min()  # Negate and swap min/max

        # For reproducibility
        np.random.seed(42)
        global_channel_colors = [[random_color() for _ in range(group_size)] for _ in range(n_rows)]

        # Plot
        for col_i, sid in enumerate(section_ids):
            sid_subset = data[data['SectionID'] == sid]
            
            for row_i, lipid_set in enumerate(lipid_groups):
                ax = axes[row_i, col_i]
                ax.set_facecolor('black')
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_aspect('equal')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                
                # Get intensities
                lipid_values = sid_subset[lipid_set].to_numpy()
                
                # Find max channel for each point
                max_indices = np.argmax(lipid_values, axis=1)
                final_colors = np.zeros((lipid_values.shape[0], 3))
                
                for i in range(group_size):
                    mask = max_indices == i
                    if np.sum(mask) > 0:
                        # Normalize intensities using percentile-based normalization
                        values = lipid_values[mask, i]
                        p95 = np.percentile(values, 95)  # Use 95th percentile as max
                        p5 = np.percentile(values, 5)    # Use 5th percentile as min
                        norm_intensity = np.clip((values - p5) / (p95 - p5), 0, 1)
                        final_colors[mask] = norm_intensity[:, None] * global_channel_colors[row_i][i]
                
                # Mirror vertically by negating y-coordinates
                ax.scatter(sid_subset['z_index'], -sid_subset['y_index'], c=final_colors, s=0.4, edgecolors='none', rasterized=True)
                
                # Optionally, add lipid names as text
                if col_i == 0:
                    ax.set_ylabel(f"{', '.join(lipid_set)}", color='white', fontsize=8)
        
        plt.savefig(os.path.join(self.plots_dir, output_file), dpi=300)
        plt.close(fig)
        print(f"Saved lipid RGB grid plot to {os.path.join(self.plots_dir, output_file)}")

    def create_lipids_movie(
        self,
        pc_list=None,
        interp_dir="3d_interpolated_native",
        movie_dir="movie_autostrada",
        final_movie_path="lipids_autostrada.mp4",
        grid_cols=16,
        grid_rows=10,
    ):
        """
        Create a movie showing the distribution of lipids across sections.
        Uses pre-computed 3D interpolated data.

        Parameters
        ----------
        pc_list : list, optional
            List of lipid names to include in the movie. If None, uses a default list.
        interp_dir : str, optional
            Directory containing the 3D interpolated .npy files.
        movie_dir : str, optional
            Directory to store movie frames.
        final_movie_path : str, optional
            Path for the final movie file.
        grid_cols : int, optional
            Number of columns in the grid layout.
        grid_rows : int, optional
            Number of rows in the grid layout.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import cv2
        from tqdm import tqdm
        import glob
        import subprocess
        from threadpoolctl import threadpool_limits

        # Default list of lipids if none provided
        if pc_list is None:
            pc_list = ["HexCer 42:2;O2"]

        # Ensure output directory exists
        os.makedirs(movie_dir, exist_ok=True)

        # Limit thread pools
        threadpool_limits(limits=8)
        os.environ["OMP_NUM_THREADS"] = "6"

        # Walk through files in interp_dir
        for fname in tqdm(os.listdir(interp_dir), desc="Processing 3D .npy files"):
            if fname == "old":
                continue
            if not fname.lower().endswith(".npy"):
                continue

            input_path = os.path.join(interp_dir, fname)
            title = os.path.splitext(fname)[0].split("interpolation")[0].strip()

            # Load the 3D array
            sm = np.load(input_path)
            # Discard first 20 z-slices
            sm = sm[20:, :, :]
            vmin = np.min(sm)
            vmax = np.max(sm)

            # Create a figure for the animation
            fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            ax.axis("off")
            ax.set_facecolor("black")

            # Initial frame
            im = ax.imshow(
                sm[0, :, :], cmap="gray", interpolation="none", vmin=vmin, vmax=vmax
            )
            width = sm.shape[2]
            ax.text(
                width / 2,
                0.5,
                title,
                color="white",
                fontsize=18,
                fontweight="bold",
                ha="center",
                va="top",
            )

            # Update function for FuncAnimation
            def update(frame_idx):
                im.set_data(sm[frame_idx, :, :])
                return [im]

            # Build and save the animation
            ani = FuncAnimation(fig, update, frames=sm.shape[0], interval=50, blit=True)
            out_movie_name = f"movie_autostrada_{fname}.mp4"
            out_movie_path = os.path.join(movie_dir, out_movie_name)
            ani.save(
                out_movie_path,
                writer="ffmpeg",
                fps=20,
                savefig_kwargs={"facecolor": "black", "bbox_inches": "tight", "pad_inches": 0},
            )

            plt.close(fig)

        # Directory containing the individual slice movies
        input_dir = movie_dir
        movie_files = sorted(glob.glob(os.path.join(input_dir, "*.mp4")))[: grid_cols * grid_rows]

        if len(movie_files) < grid_cols * grid_rows:
            print(
                f"Warning: Only {len(movie_files)} movies available for a "
                f"{grid_cols}x{grid_rows} grid ({grid_cols * grid_rows} needed)"
            )
        print(f"Processing {len(movie_files)} movie files for grid assembly...")

        # Temporary directories for frames
        temp_dir = os.path.join(movie_dir, "temp_frames")
        grid_frames_dir = os.path.join(movie_dir, "grid_frames")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(grid_frames_dir, exist_ok=True)

        # Extract dimensions and frame count from the first movie
        sample_frame_path = os.path.join(temp_dir, "sample.png")
        subprocess.call(
            ["ffmpeg", "-i", movie_files[0], "-vframes", "1", sample_frame_path, "-y"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        sample_frame = cv2.imread(sample_frame_path)
        if sample_frame is None:
            raise RuntimeError(f"Could not read sample frame from {movie_files[0]}")
        height, width = sample_frame.shape[:2]

        # Count frames using ffprobe
        result = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                movie_files[0],
            ]
        ).decode("utf-8").strip()
        frame_count = int(result)
        fps = 20  # consistent with earlier

        print(f"Frame dimensions: {width}x{height}")
        print(f"Frames per video: {frame_count}")

        # Determine grid output size (scale down each tile to 25%)
        scale_factor = 0.25
        tile_w = int(width * scale_factor)
        tile_h = int(height * scale_factor)
        grid_width = tile_w * grid_cols
        grid_height = tile_h * grid_rows

        print(f"Final grid dimensions: {grid_width}x{grid_height}")

        # Build every frame of the grid movie
        for frame_idx in tqdm(range(frame_count), desc="Creating grid frames"):
            # Create an empty (black) grid frame
            grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

            # For each individual movie, extract its frame_idx-th frame
            for i, movie_path in enumerate(movie_files):
                frame_png = os.path.join(temp_dir, f"video_{i}_frame_{frame_idx}.png")
                subprocess.call(
                    [
                        "ffmpeg",
                        "-i",
                        movie_path,
                        "-vf",
                        f"select=eq(n\\,{frame_idx})",
                        "-vframes",
                        "1",
                        frame_png,
                        "-y",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                if os.path.exists(frame_png) and os.path.getsize(frame_png) > 0:
                    tile = cv2.imread(frame_png)
                    if tile is not None:
                        # Resize to tile size
                        tile_resized = cv2.resize(tile, (tile_w, tile_h))
                        row_idx = i // grid_cols
                        col_idx = i % grid_cols
                        y_start = row_idx * tile_h
                        y_end = y_start + tile_h
                        x_start = col_idx * tile_w
                        x_end = x_start + tile_w

                        # Place into grid
                        try:
                            grid_frame[y_start:y_end, x_start:x_end] = tile_resized
                        except ValueError as e:
                            print(f"Error placing tile {i} at frame {frame_idx}: {e}")
                            print(
                                f"Expected area: ({y_start}:{y_end}, {x_start}:{x_end}), "
                                f"grid size: {grid_frame.shape}"
                            )
                            print(f"Tile size: {tile_resized.shape}")

                    # Clean up
                    os.remove(frame_png)

            # Save the assembled grid frame
            grid_frame_path = os.path.join(grid_frames_dir, f"grid_frame_{frame_idx:04d}.png")
            cv2.imwrite(grid_frame_path, grid_frame)

        # Encode the grid frames into the final movie
        print("Encoding final grid movie with FFmpeg...")
        subprocess.call(
            [
                "ffmpeg",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(grid_frames_dir, "grid_frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                final_movie_path,
            ]
        )
        print(f"Final grid movie saved to: {final_movie_path}")

        # Cleanup temporary directories
        print("Cleaning up temporary frames...")
        # Remove grid frames
        for f in os.listdir(grid_frames_dir):
            os.remove(os.path.join(grid_frames_dir, f))
        os.rmdir(grid_frames_dir)

        # Remove temp frames
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)

        print("All done!")

    def make_lipizone_rain_movie(self, section_id, output_prefix="scatter_rain_final", total_duration_ms=15000, point_size=2.0):
        """
        Create a movie of lipizones 'raining in' by color for a given SectionID.
        Points appear grouped by their unique color, one color at a time.

        Parameters
        ----------
        section_id : int or str
            The SectionID to plot.
        output_prefix : str, optional
            Prefix for the output mp4 file (saved in plots directory).
        total_duration_ms : int, optional
            Total duration of the movie in milliseconds.
        point_size : float, optional
            Size of scatter points.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.animation as animation
        import os

        data = self.adata.obs.copy()
        if 'SectionID' not in data.columns:
            raise ValueError("'SectionID' not found in adata.obs columns.")
        if 'zccf' not in data.columns or 'yccf' not in data.columns:
            raise ValueError("'zccf' and 'yccf' must be present in adata.obs.")
        if 'lipizone_color' not in data.columns:
            raise ValueError("'lipizone_color' not found in adata.obs columns.")

        ddf = data[data['SectionID'] == section_id]
        if ddf.empty:
            raise ValueError(f"No data found for SectionID {section_id}.")

        x = ddf['zccf'].values
        y = -ddf['yccf'].values
        colors = ddf['lipizone_color'].values

        unique_colors = list(dict.fromkeys(colors))
        n_frames = len(unique_colors)
        interval = total_duration_ms / n_frames

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        ax.set_xlim(x.min() - 1, x.max() + 1)
        ax.set_ylim(y.min() - 1, y.max() + 1)

        scatter = ax.scatter([], [], s=point_size, rasterized=True)

        def init():
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_facecolors([])
            return scatter,

        def animate(frame):
            allowed_colors = unique_colors[:frame + 1]
            mask = np.isin(colors, allowed_colors)
            current_x = x[mask]
            current_y = y[mask]
            current_colors = colors[mask]
            offsets = np.column_stack((current_x, current_y))
            scatter.set_offsets(offsets)
            scatter.set_facecolors(current_colors)
            return scatter,

        # Create animation
        ani = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=interval, blit=True)
        
        # Save animation
        output_path = os.path.join(self.plots_dir, f"{output_prefix}_{section_id}.mp4")
        ani.save(output_path, writer='ffmpeg', fps=30)
        plt.close()
        print(f"Lipizone rain movie saved to {output_path}")

    def make_splitter_movie(self, section_id, level_prefix='level_', n_levels=6, frames_per_level=10, output_file="level_transitions_animation.mp4", point_size=2.0):
        """
        Animate transitions between hierarchical levels for a given SectionID, showing progressive splitting
        with modal colors at each level. Each level splits into two branches, and the animation shows the
        progression from 2 colors to 2^n_levels colors.

        Parameters
        ----------
        section_id : int
            The section ID to visualize
        level_prefix : str, optional
            Prefix for the level columns in adata.obs
        n_levels : int, optional
            Number of hierarchical levels to show (default: 6)
        frames_per_level : int, optional
            Number of frames to show for each level transition (default: 10)
        output_file : str, optional
            Output filename for the animation
        point_size : float, optional
            Size of scatter points
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.animation import FuncAnimation
        from tqdm import tqdm
        import matplotlib.colors as mcolors

        # Get data for the specified section
        section_mask = self.adata.obs['SectionID'] == section_id
        section_data = self.adata.obs[section_mask].copy()
        
        # Get coordinates and rotate 90 degrees counterclockwise
        x = section_data['y'].values  # Swap x and y
        y = -section_data['x'].values  # Negate for counterclockwise rotation

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_facecolor('white')
        ax.axis('off')

        # Set axis limits to match data range
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())

        # Initialize scatter plot
        scatter = ax.scatter(x, y, c='white', s=point_size, alpha=0)
        
        # Function to convert hex color to RGB array
        def hex_to_rgb(hex_color):
            # Remove # if present
            hex_color = hex_color.lstrip('#')
            # Convert to RGB
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return np.array(rgb) / 255.0
        
        # Function to get modal colors for a given level based on full path
        def get_modal_colors(level):
            colors = np.zeros((len(section_data), 3))
            
            # For level 1, just get modal colors for the two initial groups
            if level == 1:
                unique_values = section_data[f"{level_prefix}1"].unique()
                for val in unique_values:
                    mask = section_data[f"{level_prefix}1"] == val
                    modal_color_hex = section_data.loc[mask, 'lipizone_color'].mode().iloc[0]
                    modal_color = hex_to_rgb(modal_color_hex)
                    colors[mask] = modal_color
                return colors
            
            # For subsequent levels, get modal colors based on full path
            # First, get all unique paths up to current level
            path_cols = [f"{level_prefix}{i}" for i in range(1, level + 1)]
            paths = section_data[path_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
            unique_paths = paths.unique()
            
            # For each unique path, get the modal color
            for path in unique_paths:
                mask = paths == path
                modal_color_hex = section_data.loc[mask, 'lipizone_color'].mode().iloc[0]
                modal_color = hex_to_rgb(modal_color_hex)
                colors[mask] = modal_color
            
            return colors

        # Generate all frames
        frames = []
        current_colors = np.zeros((len(section_data), 3))
        
        # For each level
        for level in range(1, n_levels + 1):
            # Get colors for this level
            next_colors = get_modal_colors(level)
            
            # Generate transition frames
            for frame in range(frames_per_level):
                alpha = frame / (frames_per_level - 1)
                frame_colors = current_colors * (1 - alpha) + next_colors * alpha
                frames.append(frame_colors.copy())
            
            current_colors = next_colors

        # Animation update function
        def update(frame):
            scatter.set_color(frames[frame])
            scatter.set_alpha(1)
            return [scatter]

        # Create animation
        anim = FuncAnimation(
            fig, update,
            frames=len(frames),
            interval=200,  # Slower transition (200ms per frame)
            blit=True
        )

        # Save animation
        output_path = os.path.join(self.plots_dir, output_file)
        anim.save(output_path, writer='ffmpeg', fps=5)  # Slower overall speed (5 fps)
        plt.close(fig)
        
        print(f"Splitter movie saved to {output_path}")

    def create_marching_cubes_volumes(self, 
                                      label_col='lipizone_names', 
                                      color_col='lipizone_color', 
                                      x_col='x_index', y_col='y_index', z_col='z_index',
                                      out_dir=None,
                                      grid_shape=(528, 320, 456),
                                      structure_size=12,
                                      keep_top_k=4,
                                      symmetrize=True,
                                      marching_level=0.5,
                                      voxel_spacing=(1,1,1),
                                      origin=(0,0,0),
                                      gaussian_sigma=2.5,
                                      mesh_smoothing_iterations=2,
                                      mesh_smoothing_factor=0.2,
                                      test_cluster=None):
        """
        Create 3D volume renders for lipizones using marching cubes.

        Parameters
        ----------
        label_col : str, optional
            Column containing lipizone labels.
        color_col : str, optional
            Column containing lipizone colors.
        x_col, y_col, z_col : str, optional
            Column names for spatial coordinates.
        out_dir : str, optional
            Output directory for 3D meshes.
            If None, defaults to "{analysis_name}_3d_lipizone_renders".
        grid_shape : tuple, optional
            3D grid shape for voxelization.
        structure_size : int, optional
            Size of morphological structuring element.
        keep_top_k : int, optional
            Number of largest connected components to keep.
        symmetrize : bool, optional
            Whether to symmetrize the volumes.
        marching_level : float, optional
            Iso-level for marching cubes.
        voxel_spacing : tuple, optional
            Spacing between voxels.
        origin : tuple, optional
            Origin coordinates.
        gaussian_sigma : float, optional
            Sigma for Gaussian smoothing.
        mesh_smoothing_iterations : int, optional
            Number of mesh smoothing iterations.
        mesh_smoothing_factor : float, optional
            Mesh smoothing factor.
        test_cluster : str, optional
            If provided, only process this specific cluster for testing.
        """
        
        if out_dir is None:
            out_dir = f"{self.analysis_name}_3d_lipizone_renders"

        import numpy as np
        import os
        from scipy.ndimage import binary_closing, label, gaussian_filter
        from scipy.ndimage import sum as ndi_sum
        from skimage import measure
        from collections import Counter
        from tqdm import tqdm

        print(f"\nStarting 3D volume creation...")
        print(f"Output directory: {out_dir}")

        # Prepare output dir
        os.makedirs(out_dir, exist_ok=True)
        obs = self.adata.obs
        mask = np.isfinite(obs[[x_col, y_col, z_col]].values).all(axis=1)
        pixels = obs.loc[mask].copy()
        points = pixels[[x_col, y_col, z_col]].values
        labels = pixels[label_col].values
        colors_arr = pixels[color_col].values
        unique_labels = np.unique(labels)
        if test_cluster is not None:
            unique_labels = [test_cluster]
            print(f"Testing with single cluster: {test_cluster}")
        color_map = {lbl: colors_arr[labels == lbl][0] for lbl in unique_labels}
        print(f"Found {len(unique_labels)} unique labels")

        # --- Parameters
        GRID_SHAPE = grid_shape
        STRUCTURE_SIZE = structure_size
        KEEP_TOP_K = keep_top_k
        SYMMETRIZE = symmetrize
        MARCHING_LEVEL = marching_level
        VOXEL_SPACING = voxel_spacing
        ORIGIN = origin
        GAUSSIAN_SIGMA = gaussian_sigma

        # --- Helpers
        def voxelize_fixed(points, grid_shape=GRID_SHAPE):
            idx = np.floor(points).astype(int)
            vol = np.zeros(grid_shape, dtype=bool)
            for i in range(idx.shape[0]):
                x, y, z = idx[i]
                if 0 <= x < grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= z < grid_shape[2]:
                    vol[x, y, z] = True
            return vol

        def clean_mask(mask, struct_size=STRUCTURE_SIZE, top_k=KEEP_TOP_K, symmetrize=SYMMETRIZE):
            struct = np.ones((struct_size,)*3, dtype=bool)
            closed = binary_closing(mask, structure=struct)
            labeled, n_comp = label(closed)
            if n_comp == 0:
                return closed.astype(bool)
            vols = ndi_sum(mask, labeled, index=np.arange(1, n_comp+1))
            top_idxs = np.argsort(vols)[-top_k:]
            new_mask = np.zeros_like(mask)
            for ti in top_idxs:
                new_mask |= (labeled == (ti+1))
            if symmetrize:
                new_mask = new_mask | np.flip(new_mask, axis=2)
            return new_mask

        def smooth_mask(mask, sigma=GAUSSIAN_SIGMA):
            float_mask = mask.astype(float)
            smoothed = gaussian_filter(float_mask, sigma=sigma)
            return smoothed

        def mesh_from_binary(mask, level=MARCHING_LEVEL, spacing=VOXEL_SPACING, origin=ORIGIN):
            verts, faces, _, _ = measure.marching_cubes(mask, level=level, spacing=spacing)
            verts = verts + np.array(origin)[None, :]
            return verts, faces

        # --- Main loop
        for lbl in tqdm(unique_labels, desc="Rendering clusters"):
            try:
                print(f"\nProcessing cluster: {lbl}")
                pts = points[labels == lbl]
                print(f"Number of points: {len(pts)}")
                
                raw_mask = voxelize_fixed(pts)
                print(f"Raw mask shape: {raw_mask.shape}")
                
                clean_mask_ = clean_mask(raw_mask)
                sm_fld = smooth_mask(clean_mask_)
                sm_binary = sm_fld >= MARCHING_LEVEL
                
                labeled, n_comp = label(sm_binary)
                print(f"Number of components: {n_comp}")
                
                if n_comp:
                    vols = ndi_sum(sm_binary, labeled, index=np.arange(1, n_comp+1))
                    largest_vol = vols.max()
                    threshold = largest_vol / 4
                    labelz = np.arange(1, n_comp+1)
                    selected_labelz = labelz[vols >= threshold]
                    sm_binary = np.isin(labeled, selected_labelz)
                    print(f"Selected {len(selected_labelz)} components")
                
                if SYMMETRIZE:
                    sm_binary = sm_binary | np.flip(sm_binary, axis=2)
                
                sm_fld = sm_fld * sm_binary.astype(float)
                verts3, faces3 = mesh_from_binary(sm_fld)
                print(f"Generated mesh with {len(verts3)} vertices and {len(faces3)} faces")
                
                fname = str(lbl).replace("/", "_").replace(" ", "_") + "_mesh.npz"
                np.savez_compressed(os.path.join(out_dir, fname), verts=verts3, faces=faces3)
                print(f"Saved mesh to {fname}")
                
            except Exception as e:
                print(f"Failed for {lbl}: {e}")
                continue
        print(f"\nCompleted! Saved all meshes to {out_dir}")

    def plot_marching_cubes_plotly(self, out_dir=None, labels_to_plot=None, color_col='lipizone_color', opacity_mesh=0.8, html_file=None, volume_trace=None, test_cluster=None, vol_factor=4):
        """
        Load meshes from create_marching_cubes_volumes and plot with Plotly.

        Parameters
        ----------
        out_dir : str, optional
            Directory containing .npz mesh files.
            If None, defaults to "{analysis_name}_3d_lipizone_renders".
        labels_to_plot : list, optional
            Specific labels to plot. If None, plots all available.
        color_col : str, optional
            Column for colors.
        opacity_mesh : float, optional
            Mesh opacity.
        html_file : str, optional
            Output HTML file name.
            If None, defaults to "{analysis_name}_lipizone_3d.html".
        volume_trace : np.ndarray, optional
            Additional volume data to include as a volume trace.
        test_cluster : str, optional
            If provided, only plot this specific cluster for testing.
        vol_factor : int, optional
            Factor by which to scale the volume coordinates. Default is 4.
        """
        
        if out_dir is None:
            out_dir = f"{self.analysis_name}_3d_lipizone_renders"
        if html_file is None:
            html_file = f"{self.analysis_name}_lipizone_3d.html"

        import numpy as np
        import os
        import plotly.graph_objects as go
        from tqdm import tqdm

        obs = self.adata.obs
        mask = np.isfinite(obs[['x_index', 'y_index', 'z_index']].values).all(axis=1)
        pixels_clean = obs.loc[mask].copy()
        labels = pixels_clean['lipizone_names'].values
        colors_arr = pixels_clean[color_col].values
        unique_labels = np.unique(labels)
        if test_cluster is not None:
            unique_labels = [test_cluster]
        color_map = {lbl: colors_arr[labels == lbl][0] for lbl in unique_labels}

        if labels_to_plot is None:
            labels_to_plot = unique_labels

        def load_mesh(label):
            fn = os.path.join(out_dir, f"{str(label).replace('/', '_')}_mesh.npz")
            data = np.load(fn)
            return data['verts'], data['faces']

        fig = go.Figure()
        
        if volume_trace is not None:
            nx, ny, nz = volume_trace.shape
            xv = np.arange(nx) * vol_factor
            yv = np.arange(ny) * vol_factor
            zv = np.arange(nz) * vol_factor
            xs, ys, zs = np.meshgrid(xv, yv, zv, indexing='ij')
            
            volume_trace_obj = go.Volume(
                x=xs.flatten(),
                y=ys.flatten(),
                z=zs.flatten(),
                value=volume_trace.flatten(),
                opacity=0.01,
                surface_count=15,
                colorscale='Gray',
                showscale=False,
                caps=dict(
                    x_show=False,
                    y_show=False,
                    z_show=False
                )
            )
            fig.add_trace(volume_trace_obj)
            
        for lbl in labels_to_plot:
            try:
                verts, faces = load_mesh(lbl)
                x, y, z = verts.T
                i, j, k = faces.T
                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    color=color_map.get(lbl, "gray"),
                    opacity=opacity_mesh,
                    name=str(lbl),
                    flatshading=False
                ))
            except Exception as e:
                continue
                
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showgrid=False, zeroline=False, visible=False),
                aspectmode="data"
            )
        )
        fig.write_html(html_file)

    import numpy as np
    import pandas as pd
    import scanpy as sc  # for AnnData aliasing
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.ndimage import gaussian_filter

    def _build_2d_array_from_adata(
        self,
        values,            # full-length array of length n_obs
        section,
        section_key='Section',
        x_key='z_index',
        y_key='y_index',
    ):
        """
        Construct a 2D image for a single section from AnnData, using only the obs in that section.
        """
        df = self.adata.obs
        mask = df[section_key] == section
        if not mask.any():
            return None

        xs = df.loc[mask, x_key].astype(int).values
        ys = df.loc[mask, y_key].astype(int).values
        vals = np.asarray(values)[mask]        # <-- select only this section's values

        max_x = xs.max() + 1
        max_y = ys.max() + 1
        img = np.full((max_y, max_x), np.nan)
        img[ys, xs] = vals
        return img

    def _build_3d_array_from_adata(
        self,
        values,            # full-length array of length n_obs
        section_list=None,
        section_key='Section',
        x_key='z_index',
        y_key='y_index',
        z_key='x_index',
    ):
        """
        Construct a 3D volume from AnnData across all sections (z-index).
        """
        df = self.adata.obs
        if section_list is not None:
            mask = df[section_key].isin(section_list)
        else:
            mask = np.ones(len(df), dtype=bool)

        df_sel = df.loc[mask]
        xs = df_sel[x_key].astype(int).values
        ys = df_sel[y_key].astype(int).values
        zs = df_sel[z_key].astype(int).values
        vals = np.asarray(values)[mask]       # <-- again, select only these obs

        max_x = xs.max() + 1
        max_y = ys.max() + 1
        max_z = zs.max() + 1
        vol = np.full((max_z, max_y, max_x), np.nan)
        vol[zs, ys, xs] = vals
        return vol

    def all_sections_lipid_visualizer(
        self,
        lipid_name,
        section_key='Section',
        x_key='z_index',
        y_key='y_index',
        n_cols=12,
    ):
        """
        Grid of 2D heatmaps for lipid expression across sections in AnnData.
        """
        # Extract intensity vector
        if lipid_name in self.adata.var_names:
            vals = self.adata.X[:, self.adata.var_names.get_loc(lipid_name)].toarray().ravel() if hasattr(self.adata.X, 'toarray') else self.adata.X[:, self.adata.var_names.get_loc(lipid_name)]
        else:
            raise KeyError(f"Lipid '{lipid_name}' not found in adata.var_names")
        sections = self.adata.obs[section_key].unique()
        total = len(sections)
        n_rows = (total + n_cols - 1) // n_cols
        fig = make_subplots(rows=n_rows, cols=n_cols, horizontal_spacing=0.01, vertical_spacing=0.01)

        valid = 0
        max_h, max_w = 0, 0
        # first pass: find max dims
        for section in sections:
            img = self._build_2d_array_from_adata(vals, section, section_key, x_key, y_key)
            if img is None:
                continue
            max_h = max(max_h, img.shape[0])
            max_w = max(max_w, img.shape[1])
        aspect = max_h / max_w if max_w else 1
        cell_w, cell_h = 100, int(100 * aspect)

        # second pass: add traces
        for section in sections:
            img = self._build_2d_array_from_adata(vals, section, section_key, x_key, y_key)
            if img is None:
                continue
            row = valid // n_cols + 1
            col = valid % n_cols + 1
            fig.add_trace(go.Heatmap(z=img, showscale=False, colorscale='viridis'), row=row, col=col)
            fig.add_annotation(text=str(section), x=0.5, y=0.9, xref=f'x{valid+1}', yref=f'y{valid+1}', showarrow=False, font=dict(color='white', size=10))
            valid += 1

        fig.update_layout(
            height=cell_h * n_rows + 100,
            width=cell_w * n_cols + 100,
            showlegend=False,
            margin=dict(t=30, r=10, b=10, l=10),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig.update_xaxes(visible=False, showgrid=False, zeroline=False, scaleanchor='y')
        fig.update_yaxes(visible=False, showgrid=False, zeroline=False, scaleanchor='x')
        return fig

    def all_sections_lipizone_visualizer(
        self,
        section_key='Section',
        x_key='z_index',
        y_key='y_index',
        lipizone_key='lipizone',
        lipizone_color_key='lipizone_color',
        n_cols=12,
    ):
        """
        Grid mosaic of lipizone assignments across sections from AnnData.
        """
        df = self.adata.obs
        if lipizone_key not in df.columns or lipizone_color_key not in df.columns:
            raise KeyError('Lipizone keys not found in adata.obs')
        sections = df[section_key].unique()
        total = len(sections)
        n_rows = (total + n_cols - 1) // n_cols
        fig = make_subplots(rows=n_rows, cols=n_cols,
                            horizontal_spacing=0.01, vertical_spacing=0.01)

        valid = 0
        max_h, max_w = 0, 0
        # First pass: compute integer dims
        for section in sections:
            sel = df[df[section_key] == section]
            if sel.empty:
                continue
            h = int(sel[y_key].max()) + 1
            w = int(sel[x_key].max()) + 1
            max_h = max(max_h, h)
            max_w = max(max_w, w)

        # Guarantee ints
        max_h, max_w = int(max_h), int(max_w)

        aspect = max_h / max_w if max_w else 1
        cell_w, cell_h = 100, int(100 * aspect)

        # Second pass: render each section
        for section in sections:
            sel = df[df[section_key] == section]
            if sel.empty:
                continue

            # Build RGB image
            img = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            img[:] = 0
            for _, row in sel.iterrows():
                x, y = int(row[x_key]), int(row[y_key])
                color = row[lipizone_color_key]
                if isinstance(color, str) and color.startswith('#'):
                    r = int(color[1:3], 16)
                    g = int(color[3:5], 16)
                    b = int(color[5:7], 16)
                    img[y, x] = [r, g, b]

            # Place in subplot
            row_i = valid // n_cols + 1
            col_i = valid % n_cols + 1
            fig.add_trace(go.Image(z=img), row=row_i, col=col_i)
            fig.add_annotation(
                text=str(section),
                x=0.5, y=0.9,
                xref=f'x{valid+1}',
                yref=f'y{valid+1}',
                showarrow=False,
                font=dict(color='white', size=10),
            )
            valid += 1

        fig.update_layout(
            height=cell_h * n_rows + 100,
            width= cell_w * n_cols + 100,
            showlegend=False,
            margin=dict(t=30, r=10, b=10, l=10),
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig

    def treemap_lipizones_adata(
        self,
        path=None,
        lipizone_key='lipizone',
        color_key='lipizone_color',
        maxdepth=2
    ):
        """
        Generate a hierarchical treemap of lipizones directly from AnnData.

        Args:
            path (list of str, optional): Sequence of obs column names defining hierarchy. 
                If None, uses [lipizone_key].
            lipizone_key (str): Column in obs containing lipizone name.
            color_key (str): Column in obs containing hex color for each lipizone.
            maxdepth (int): Maximum depth to display in treemap.

        Returns:
            plotly.graph_objects.Figure: A dark-themed treemap figure.
        """
        df = self.adata.obs.copy()
        # Default single-level hierarchy
        if path is None:
            path = [lipizone_key]
        # Build discrete color map from unique lipizones
        unique = df[[lipizone_key, color_key]].drop_duplicates()
        color_map = dict(zip(unique[lipizone_key], unique[color_key]))

        # Build treemap
        fig = px.treemap(
            df,
            path=path,
            values=None,  # uses count per leaf
            color=lipizone_key,
            color_discrete_map=color_map,
            maxdepth=maxdepth,
        )
        # Dark theme layout
        fig.update_layout(
            template='plotly_dark',
            margin=dict(t=30, r=0, b=10, l=0),
            uniformtext=dict(minsize=12)
        )
        # Remove background grids
        fig.layout.plot_bgcolor = 'rgba(0,0,0,0)'
        fig.layout.paper_bgcolor = 'rgba(0,0,0,0)'
        return fig

    def scatter_lipizone_3d(
        self,
        section_list: list = None,
        lipizone_key: str = 'lipizone',
        section_key: str = 'Section',
        x_key: str = 'z_index',
        y_key: str = 'y_index',
        z_key: str = 'x_index',
    ) -> go.Figure:
        """
        Create a 3D scatter plot of lipizones.

        Parameters
        ----------
        section_list : list, optional
            List of sections to include. If None, uses all sections.
        lipizone_key : str
            Column name for lipizone names
        section_key : str
            Column name for section IDs
        x_key : str
            Column name for x coordinates
        y_key : str
            Column name for y coordinates
        z_key : str
            Column name for z coordinates

        Returns
        -------
        go.Figure
            A Plotly Figure containing the 3D scatter plot
        """
        df = self.adata.obs.copy()
        if section_list is not None:
            df = df[df[section_key].isin(section_list)]

        # Get unique lipizones and their colors
        unique_lipizones = df[lipizone_key].unique()
        color_map = dict(zip(unique_lipizones, px.colors.qualitative.Set3[:len(unique_lipizones)]))

        fig = go.Figure()
        
        for lipizone in unique_lipizones:
            mask = df[lipizone_key] == lipizone
            fig.add_trace(go.Scatter3d(
                x=df.loc[mask, x_key],
                y=df.loc[mask, y_key],
                z=df.loc[mask, z_key],
                mode='markers',
                marker=dict(
                    size=2,
                    color=color_map[lipizone],
                    opacity=0.8
                ),
                name=lipizone
            ))

        fig.update_layout(
            template='plotly_dark',
            scene=dict(
                xaxis=dict(showticklabels=True, showgrid=True, zeroline=False,
                           backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(showticklabels=True, showgrid=True, zeroline=False,
                           backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
                zaxis=dict(showticklabels=True, showgrid=True, zeroline=False,
                           backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
                aspectmode="data",
            ),
            margin=dict(t=0, r=0, b=0, l=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def compute_3d_lipid_figure_from_array(
        self,
        lipid_array: np.ndarray,
        annotation_array: np.ndarray = None,
        set_id_regions: list[int] = None,
        downsample_factor: int = 1,
        opacity: float = 0.4,
        surface_count: int = 15,
        colorscale: str = "Inferno",
        root_decrease_dimensionality_factor: int = 4,
    ) -> go.Figure:
        """
        Render a 3D volume of lipid data from a raw 3D NumPy array, with optional atlas context.

        Parameters
        ----------
        lipid_array : np.ndarray
            3D numpy array of lipid intensities, shape (Z, Y, X).
        annotation_array : np.ndarray, optional
            3D numpy array of atlas annotations (same shape as lipid_array before downsampling).
        set_id_regions : list[int], optional
            List of integer region IDs to mask lipid data to; if None, no masking.
        downsample_factor : int
            Factor by which to downsample both lipid and annotation volumes for speed.
        opacity : float
            Base opacity for the lipid volume.
        surface_count : int
            Number of isosurfaces for the lipid volume.
        colorscale : str
            Plotly colorscale name for the lipid volume.
        root_decrease_dimensionality_factor : int
            If annotation_array is provided, how much to subsample it when building the root volume.

        Returns
        -------
        fig : go.Figure
            A Plotly Figure containing (optionally) the semi‐transparent root volume first,
            then the lipid volume on top.
        """
        data_traces = []

        # --- 1) Atlas "root" border trace, if requested ---
        if annotation_array is not None:
            # subsample atlas
            ann_sub = annotation_array[
                ::root_decrease_dimensionality_factor,
                ::root_decrease_dimensionality_factor,
                ::root_decrease_dimensionality_factor,
            ]
            # simple border extraction: voxels in mask but not in eroded mask
            mask = ann_sub > 0
            eroded = binary_erosion(mask)
            border = (mask & ~eroded).astype(float)

            z0, y0, x0 = np.indices(border.shape)
            root_trace = go.Volume(
                x=x0.flatten(),
                y=y0.flatten(),
                z=z0.flatten(),
                value=border.flatten(),
                isomin=1.0,        # only the border voxels
                isomax=1.0,
                opacity=0.1,
                surface_count=1,
                colorscale="Greys",
                caps=dict(x_show=False, y_show=False, z_show=False),
                showscale=False,
            )
            data_traces.append(root_trace)

        # --- 2) Downsample lipid data ---
        lp = lipid_array[::downsample_factor, ::downsample_factor, ::downsample_factor]

        # --- 3) Region‐masking, if requested ---
        if annotation_array is not None and set_id_regions is not None:
            ann_ds = annotation_array[
                ::downsample_factor, ::downsample_factor, ::downsample_factor
            ]
            mask = np.zeros_like(ann_ds, dtype=bool)
            for rid in set_id_regions:
                mask |= (ann_ds == rid)
            lp_masked = lp.copy()
            lp_masked[~mask] = np.nan
        else:
            lp_masked = lp

        # --- 4) Lipid trace with custom opacityscale ---
        z1, y1, x1 = np.indices(lp_masked.shape)
        opacityscale = [
            [0.0, 0.0],
            [0.3, 0.2],
            [0.7, 0.5],
            [1.0, 0.8],
        ]
        lipid_trace = go.Volume(
            x=x1.flatten(),
            y=y1.flatten(),
            z=z1.flatten(),
            value=lp_masked.flatten(),
            isomin=np.nanmin(lp_masked) if not np.isnan(lp_masked).all() else 0,
            isomax=np.nanmax(lp_masked) if not np.isnan(lp_masked).all() else 1,
            opacityscale=opacityscale,
            surface_count=surface_count,
            colorscale=colorscale,
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
        data_traces.append(lipid_trace)

        # --- 5) Build figure ---
        fig = go.Figure(data=data_traces)
        fig.update_layout(
            margin=dict(t=0, r=0, b=0, l=0),
            scene=dict(
                xaxis=dict(showticklabels=True, showgrid=True, zeroline=False,
                           backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(showticklabels=True, showgrid=True, zeroline=False,
                           backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
                zaxis=dict(showticklabels=True, showgrid=True, zeroline=False,
                           backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.1)"),
                aspectmode="data",
            ),
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        # --- 6) Cleanup ---
        del lp, lp_masked, x1, y1, z1
        gc.collect()

        return fig

    def plot_mosaic(self, section_id, output_file="mosaic.pdf", figsize=(10, 16), point_size=2.0, alpha=1.0):
        """
        Create a mosaic-style visualization of a half of a single section, using lipizone colors.
        The plot uses transformed coordinates (-y, -x) and has a black background.

        Parameters
        ----------
        section_id : float or int
            The section ID to plot
        output_file : str, optional
            Output filename (default: "mosaic.pdf")
        figsize : tuple, optional
            Figure size in inches (width, height) (default: (10, 16))
        point_size : float, optional
            Size of scatter points (default: 2.0)
        alpha : float, optional
            Transparency of points (default: 1.0)
        """
        import matplotlib.pyplot as plt
        import pandas as pd

        # Convert adata to DataFrame for easier manipulation
        data = pd.DataFrame(self.adata.X, columns=self.adata.var_names, index=self.adata.obs_names)
        data = data.join(self.adata.obs[['SectionID', 'x', 'y', 'lipizone_color']])

        # Filter for the specified section and upper half
        ddf = data[data['SectionID'] == section_id]
        ddf = ddf.loc[ddf['y'] < ddf['y'].mean(), :]

        # Create the plot
        plt.figure(figsize=figsize, facecolor='black')
        plt.scatter(-ddf['y'], -ddf['x'], 
                   c=ddf['lipizone_color'],
                   s=point_size, 
                   alpha=alpha)
        plt.axis('off')

        # Save the plot
        plt.savefig(os.path.join(self.plots_dir, output_file))
        print(f"Saved mosaic plot to {os.path.join(self.plots_dir, output_file)}")
        plt.show()