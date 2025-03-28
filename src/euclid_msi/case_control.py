# ##################################################################################################################
# # BLOCK 1: CONVENTIONAL DIFFERENTIAL TESTING -> function differential_lipids
# ##################################################################################################################


# from threadpoolctl import threadpool_limits, threadpool_info
# threadpool_limits(limits=8)
# import os
# os.environ['OMP_NUM_THREADS'] = '6'
# dat = pd.read_parquet("atlas.parquet")

# # a function to check for differential lipids between two groups

# from scipy.stats import mannwhitneyu, entropy
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from statsmodels.stats.multitest import multipletests
# from tqdm import tqdm

# import os
# import pandas as pd
# import numpy as np
# from scipy.stats import mannwhitneyu
# from statsmodels.stats.multitest import multipletests

# def differential_lipids(lipid_data, bool_mask, min_fc=0.2, pthr=0.05):
#     """
#     Compare two groups (True vs False in bool_mask) within `lipid_data`.
#     Returns a DataFrame with log2 fold change, p-values, and FDR-corrected p-values
#     for each lipid (column in lipid_data).
#     """
#     results = []
    
#     # Subset the data into two groups
#     groupA = lipid_data.loc[bool_mask]
#     groupB = lipid_data.loc[~bool_mask]

#     for col_name in lipid_data.columns:
#         dataA = groupA[col_name].dropna()
#         dataB = groupB[col_name].dropna()
        
#         # Compute group means and log2 fold change
#         meanA = np.mean(dataA) + 1e-11  # avoid division by zero
#         meanB = np.mean(dataB) + 1e-11
#         log2fc = np.log2(meanB / meanA)
        
#         # Mann-Whitney U test
#         try:
#             _, pval = mannwhitneyu(dataA, dataB, alternative='two-sided')
#         except ValueError:
#             # Occurs if one group is all identical values, etc.
#             pval = np.nan
        
#         results.append({
#             'lipid': col_name,
#             'meanA': meanA,
#             'meanB': meanB,
#             'log2fold_change': log2fc,
#             'p_value': pval
#         })

#     results_df = pd.DataFrame(results)

#     # Multiple-testing correction
#     reject, pvals_corrected, _, _ = multipletests(
#         results_df['p_value'].values,
#         alpha=pthr,
#         method='fdr_bh'
#     )
#     results_df['p_value_corrected'] = pvals_corrected

#     return results_df

# ##################################################################################################################
# # BLOCK 2: CONVENTIONAL DIFFERENTIAL TESTING FOR ALL NESTED HIERARCHICAL LEVELS ON A TREE
# ##################################################################################################################

# def traverse_and_diff(
#     dat,
#     lipid_data,
#     levels,
#     current_level=0,
#     branch_path=None,
#     min_fc=0.2,
#     pthr=0.05,
#     output_dir="diff_results"
# ):
#     """
#     Recursively traverse the hierarchical labels in `dat`, perform differential analysis 
#     (two-group comparison: val vs the rest) at each level, and save results for each split.
    
#     - dat: DataFrame containing hierarchical annotations (columns like 'level_1', 'level_2', ...).
#            Row indices align with samples.
#     - lipid_data: DataFrame with lipid measurements (same rows = samples, columns = lipids).
#     - levels: list of the column names describing the hierarchy.
#     - current_level: integer index into `levels`.
#     - branch_path: keeps track of label choices so far (used for file naming).
#     - min_fc, pthr: thresholds passed to `differential_lipids` (you can incorporate `min_fc` logic as needed).
#     - output_dir: directory where the CSV output is saved.
#     """
#     if branch_path is None:
#         branch_path = []
    
#     # Stop if we've consumed all hierarchical levels
#     if current_level >= len(levels):
#         return
    
#     level_col = levels[current_level]
#     unique_vals = dat[level_col].unique()
    
#     # If there's no real split at this level, just exit
#     if len(unique_vals) < 2:
#         return
    
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
    
#     # For each unique group at the current level
#     for val in unique_vals:
#         # labs is a boolean mask for the current subset of `dat`
#         labs = (dat[level_col] == val)
        
#         # 1) Perform differential analysis: val vs. not val
#         diff = differential_lipids(lipid_data, labs, min_fc=min_fc, pthr=pthr)
        
#         # (Optional) sort by log2 fold change, descending
#         diff = diff.sort_values(by="log2fold_change", ascending=False)
        
#         # 2) Construct a filename reflecting the path taken so far
#         path_labels = [
#             f"{lvl_name}={lvl_val}"
#             for lvl_name, lvl_val in zip(levels[:current_level], branch_path)
#         ]
#         path_labels.append(f"{level_col}={val}")
#         filename = "_".join(path_labels) + ".csv"
        
#         # Save differential results
#         out_path = os.path.join(output_dir, filename)
#         diff.to_csv(out_path, index=False)
        
#         # 3) Recurse deeper:
#         #    - subset `dat` to only the rows where labs==True
#         #    - subset `lipid_data` the same way so indexes remain aligned
#         sub_dat = dat.loc[labs]
#         sub_lipid_data = lipid_data.loc[labs]

#         traverse_and_diff(
#             dat=sub_dat,
#             lipid_data=sub_lipid_data,
#             levels=levels,
#             current_level=current_level + 1,
#             branch_path=branch_path + [val],
#             min_fc=min_fc,
#             pthr=pthr,
#             output_dir=output_dir
#         )
