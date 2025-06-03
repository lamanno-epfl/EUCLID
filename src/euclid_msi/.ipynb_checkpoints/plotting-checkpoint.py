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
         'X_Leiden', 'X_Euclid', 'lipizone_colors', 'allen_division', 'border', 'acronym', etc.
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
    def __init__(self, adata, coordinates, extra_data=None):
        self.adata = adata
        self.coordinates = coordinates
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

    def prepare_lipid_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame with a 'lipid_name' column to extract lipid class,
        number of carbons, insaturations, and to map colors from an external file.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that includes a column 'lipid_name'.
        
        Returns
        -------
        pd.DataFrame
            Updated DataFrame with additional columns.
        """
        df = df.copy()
        df["class"] = df["lipid_name"].apply(self.extract_class)
        df["carbons"] = df["lipid_name"].apply(
            lambda x: int(re.search(r'(\d+):', x).group(1)) if re.search(r'(\d+):', x) else np.nan
        )
        df["insaturations"] = df["lipid_name"].apply(
            lambda x: int(re.search(r':(\d+)', x).group(1)) if re.search(r':(\d+)', x) else np.nan
        )
        df["insaturations_per_Catom"] = df["insaturations"] / df["carbons"]
        df.loc[df['class'] == "HexCer", 'class'] = "Hex1Cer"
        df["broken"] = df["lipid_name"].str.endswith('_uncertain')
        df.loc[df["broken"], ['carbons', 'class', 'insaturations', 'insaturations_per_Catom']] = np.nan
        try:
            colors_df = pd.read_hdf("lipidclasscolors.h5ad", key="table")
            color_dict = colors_df["classcolors"].to_dict()
        except Exception:
            color_dict = {}
        df['color'] = df['class'].map(color_dict).fillna("black")
        df.index = df["lipid_name"]
        df = df.drop_duplicates()
        return df

    def plot_lipid_class_pie(self, lipid_df: pd.DataFrame, class_column: str = "class",
                             show_inline: bool = False) -> None:
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
        class_counts = lipid_df[class_column].value_counts()
        palette = sns.color_palette("pastel", len(class_counts))
        color_dict = dict(zip(class_counts.index, palette))
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

    def plot_moran_by_class(self, annowmoran: pd.DataFrame, show_inline: bool = False) -> None:
        """
        Plot a boxplot (with an overlaid stripplot) of mean Moran's I by lipid class.
        
        Parameters
        ----------
        annowmoran : pd.DataFrame
            DataFrame with columns 'lipid_name', 'quant' (mean Moran's I), etc.
        show_inline : bool, optional
            If True, display the plot inline.
        """
        df = annowmoran.copy()
        df["class"] = df["lipid_name"].apply(self.extract_class)
        df["carbons"] = df["lipid_name"].apply(
            lambda x: int(re.search(r'(\d+):', x).group(1)) if re.search(r'(\d+):', x) else np.nan
        )
        df["insaturations"] = df["lipid_name"].apply(
            lambda x: int(re.search(r':(\d+)', x).group(1)) if re.search(r':(\d+)', x) else np.nan
        )
        df["insaturations_per_Catom"] = df["insaturations"] / df["carbons"]
        df.loc[df['class'] == "HexCer", 'class'] = "Hex1Cer"
        df["broken"] = df["lipid_name"].str.endswith('_uncertain')
        df.loc[df["broken"], ['carbons', 'class', 'insaturations', 'insaturations_per_Catom']] = np.nan
        try:
            colors_df = pd.read_hdf("lipidclasscolors.h5ad", key="table")
            color_dict = colors_df["classcolors"].to_dict()
        except Exception:
            color_dict = {}
        df['color'] = df['class'].map(color_dict).fillna("black")
        df.index = df["lipid_name"]
        df = df.drop_duplicates()
        order = df.groupby("class")["quant"].median().sort_values().index.tolist()
        plt.figure(figsize=(12, 8))
        ax = sns.boxplot(data=df, x="class", y="quant", order=order, palette=color_dict, showfliers=False)
        sns.stripplot(data=df, x="class", y="quant", order=order, color="black", alpha=0.5, size=3)
        ax.axhline(y=0.4, color="darkred", linestyle="--")
        ax.set_ylim(0, 1)
        sns.despine(ax=ax)
        ax.set_xlabel("Lipid Class")
        ax.set_ylabel("Mean Moran's I")
        ax.set_title("Spatial Evaluation by Mean Moran's I")
        save_path = os.path.join(PLOTS_DIR, "moranbyclass.pdf")
        plt.tight_layout()
        plt.savefig(save_path)
        if show_inline:
            plt.show()
        else:
            plt.close()

    def plot_lipid_distribution(self,
                                  data: pd.DataFrame,
                                  lipid: str,
                                  section_filter: list = None,
                                  metadata_filter: dict = None,
                                  lipizone_filter: dict = None,
                                  x_range: tuple = None,
                                  y_range: tuple = None,
                                  z_range: tuple = None,
                                  layout: tuple = None,
                                  point_size: float = 5,
                                  show_contours: bool = True,
                                  contour_column: str = "border",
                                  show_inline: bool = False) -> None:
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
        plt.tight_layout()
        save_path = os.path.join(PLOTS_DIR, f"{lipid}_distribution.pdf")
        plt.savefig(save_path)
        if show_inline:
            plt.show()
        else:
            plt.close()

    def plot_embeddings(self,
                        currentProgram: str = 'headgroup_with_negative_charge',
                        layout: tuple = (4, 8),
                        point_size: float = 0.5,
                        cmap_name: str = "PuOr",
                        show_inline: bool = False) -> None:
        """
        Plot spatial embeddings using the same strategy as for lipids.
        
        For each section (1 to 32), compute the 2nd and 98th percentiles for the specified
        program (a column in data), take the median of these percentiles across sections, and plot.
        
        Parameters
        ----------
        currentProgram : str, optional
            The column in self.adata.obs (or provided data) to use for coloring.
        layout : tuple, optional
            Grid layout (nrows, ncols) for the subplots. Default is (4, 8).
        point_size : float, optional
            Marker size.
        cmap_name : str, optional
            Name of the colormap.
        show_inline : bool, optional
            Whether to display the plot inline.
        """
        # Assume that self.adata.obs contains columns 'Section', 'zccf', 'yccf',
        # and a column named currentProgram.
        data = self.adata.obs.copy()
        # Prepare results from each section.
        results = []
        for sec in data["Section"].unique():
            subset = data[data["Section"] == sec]
            perc2 = subset[currentProgram].quantile(0.02)
            perc98 = subset[currentProgram].quantile(0.98)
            results.append([sec, perc2, perc98])
        percentile_df = pd.DataFrame(results, columns=["Section", "2-perc", "98-perc"])
        med2p = percentile_df["2-perc"].median()
        med98p = percentile_df["98-perc"].median()
        cmap = plt.get_cmap(cmap_name)
        n_sections = 32  # as in original (sections 1 to 32)
        nrows, ncols = layout
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
        axes = axes.flatten()
        for sec in range(1, n_sections+1):
            ax = axes[sec - 1]
            ddf = data[data["Section"] == sec]
            ax.scatter(ddf["zccf"], -ddf["yccf"],
                       c=ddf[currentProgram], cmap=cmap, s=point_size,
                       rasterized=True, vmin=med2p, vmax=med98p)
            ax.axis("off")
            ax.set_aspect("equal")
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = Normalize(vmin=med2p, vmax=med98p)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, cax=cbar_ax)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
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
                       show_inline: bool = False) -> None:
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

    def plot_tsne(self, tsne: np.ndarray, color_column: str = "lipizone_color",
                  show_inline: bool = False) -> None:
        """
        Plot a TSNE scatter plot with coloring from a metadata column.
        
        Parameters
        ----------
        tsne : np.ndarray
            2D array of TSNE coordinates.
        color_column : str, optional
            Column in adata.obs to use for color.
        show_inline : bool, optional
            Whether to display inline.
        """
        df = self.adata.obs.copy()
        if color_column not in df.columns:
            print(f"Warning: {color_column} not found in adata.obs; using gray.")
            colors = "gray"
        else:
            colors = df[color_column]
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne[:, 0], tsne[:, 1], c=colors, s=0.5, alpha=0.9, rasterized=True)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labelleft=False)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        save_path = os.path.join(PLOTS_DIR, "tsne.pdf")
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
