"""
data_loading.py – Efficient partial loading of AnnData objects

This module provides functionality for loading AnnData objects partially,
which is useful for large datasets where loading everything into memory
is not feasible.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from typing import Optional, List, Union, Dict, Tuple
from pathlib import Path

class PartialAnnData:
    """
    A class for efficient partial loading of AnnData objects.
    
    This class provides methods to load and work with AnnData objects
    in a memory-efficient way by loading only the required parts of the data.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize the PartialAnnData object.
        
        Parameters
        ----------
        filepath : str
            Path to the h5ad file
        """
        self.filepath = filepath
        self._load_metadata()
        
    def _load_metadata(self):
        """Load only the metadata (obs_names, var_names, etc.) from the h5ad file."""
        with anndata.read_h5ad(self.filepath, backed='r') as adata:
            self.obs_names = adata.obs_names
            self.var_names = adata.var_names
            self.n_obs = len(self.obs_names)
            self.n_vars = len(self.var_names)
            self.obs = adata.obs
            self.var = adata.var
            self.uns = dict(adata.uns)
            
    def load_obs_subset(self, obs_indices: Union[List[int], np.ndarray]) -> pd.DataFrame:
        """
        Load a subset of observations.
        
        Parameters
        ----------
        obs_indices : Union[List[int], np.ndarray]
            Indices of observations to load
            
        Returns
        -------
        pd.DataFrame
            Subset of observations
        """
        with anndata.read_h5ad(self.filepath, backed='r') as adata:
            return adata.obs.iloc[obs_indices].copy()
            
    def load_var_subset(self, var_indices: Union[List[int], np.ndarray]) -> pd.DataFrame:
        """
        Load a subset of variables.
        
        Parameters
        ----------
        var_indices : Union[List[int], np.ndarray]
            Indices of variables to load
            
        Returns
        -------
        pd.DataFrame
            Subset of variables
        """
        with anndata.read_h5ad(self.filepath, backed='r') as adata:
            return adata.var.iloc[var_indices].copy()
            
    def load_X_subset(self, 
                     obs_indices: Optional[Union[List[int], np.ndarray]] = None,
                     var_indices: Optional[Union[List[int], np.ndarray]] = None) -> np.ndarray:
        """
        Load a subset of the data matrix X.
        
        Parameters
        ----------
        obs_indices : Optional[Union[List[int], np.ndarray]]
            Indices of observations to load. If None, load all observations.
        var_indices : Optional[Union[List[int], np.ndarray]]
            Indices of variables to load. If None, load all variables.
            
        Returns
        -------
        np.ndarray
            Subset of the data matrix
        """
        with anndata.read_h5ad(self.filepath, backed='r') as adata:
            if obs_indices is None and var_indices is None:
                return adata.X[:]
            elif obs_indices is None:
                return adata.X[:, var_indices]
            elif var_indices is None:
                return adata.X[obs_indices, :]
            else:
                return adata.X[obs_indices, var_indices]
                
    def load_obsm_subset(self, 
                        key: str,
                        obs_indices: Optional[Union[List[int], np.ndarray]] = None) -> np.ndarray:
        """
        Load a subset of an obsm matrix.
        
        Parameters
        ----------
        key : str
            Key of the obsm matrix to load
        obs_indices : Optional[Union[List[int], np.ndarray]]
            Indices of observations to load. If None, load all observations.
            
        Returns
        -------
        np.ndarray
            Subset of the obsm matrix
        """
        with anndata.read_h5ad(self.filepath, backed='r') as adata:
            if key not in adata.obsm:
                raise KeyError(f"Key '{key}' not found in adata.obsm")
            if obs_indices is None:
                return adata.obsm[key][:]
            else:
                return adata.obsm[key][obs_indices]
                
    def load_obsp_subset(self,
                        key: str,
                        obs_indices: Optional[Union[List[int], np.ndarray]] = None) -> np.ndarray:
        """
        Load a subset of an obsp matrix.
        
        Parameters
        ----------
        key : str
            Key of the obsp matrix to load
        obs_indices : Optional[Union[List[int], np.ndarray]]
            Indices of observations to load. If None, load all observations.
            
        Returns
        -------
        np.ndarray
            Subset of the obsp matrix
        """
        with anndata.read_h5ad(self.filepath, backed='r') as adata:
            if key not in adata.obsp:
                raise KeyError(f"Key '{key}' not found in adata.obsp")
            if obs_indices is None:
                return adata.obsp[key][:]
            else:
                return adata.obsp[key][obs_indices][:, obs_indices]
                
    def load_varm_subset(self,
                        key: str,
                        var_indices: Optional[Union[List[int], np.ndarray]] = None) -> np.ndarray:
        """
        Load a subset of a varm matrix.
        
        Parameters
        ----------
        key : str
            Key of the varm matrix to load
        var_indices : Optional[Union[List[int], np.ndarray]]
            Indices of variables to load. If None, load all variables.
            
        Returns
        -------
        np.ndarray
            Subset of the varm matrix
        """
        with anndata.read_h5ad(self.filepath, backed='r') as adata:
            if key not in adata.varm:
                raise KeyError(f"Key '{key}' not found in adata.varm")
            if var_indices is None:
                return adata.varm[key][:]
            else:
                return adata.varm[key][var_indices]
                
    def load_varp_subset(self,
                        key: str,
                        var_indices: Optional[Union[List[int], np.ndarray]] = None) -> np.ndarray:
        """
        Load a subset of a varp matrix.
        
        Parameters
        ----------
        key : str
            Key of the varp matrix to load
        var_indices : Optional[Union[List[int], np.ndarray]]
            Indices of variables to load. If None, load all variables.
            
        Returns
        -------
        np.ndarray
            Subset of the varp matrix
        """
        with anndata.read_h5ad(self.filepath, backed='r') as adata:
            if key not in adata.varp:
                raise KeyError(f"Key '{key}' not found in adata.varp")
            if var_indices is None:
                return adata.varp[key][:]
            else:
                return adata.varp[key][var_indices][:, var_indices]
                
    def load_layers_subset(self,
                          key: str,
                          obs_indices: Optional[Union[List[int], np.ndarray]] = None,
                          var_indices: Optional[Union[List[int], np.ndarray]] = None) -> np.ndarray:
        """
        Load a subset of a layers matrix.
        
        Parameters
        ----------
        key : str
            Key of the layers matrix to load
        obs_indices : Optional[Union[List[int], np.ndarray]]
            Indices of observations to load. If None, load all observations.
        var_indices : Optional[Union[List[int], np.ndarray]]
            Indices of variables to load. If None, load all variables.
            
        Returns
        -------
        np.ndarray
            Subset of the layers matrix
        """
        with anndata.read_h5ad(self.filepath, backed='r') as adata:
            if key not in adata.layers:
                raise KeyError(f"Key '{key}' not found in adata.layers")
            if obs_indices is None and var_indices is None:
                return adata.layers[key][:]
            elif obs_indices is None:
                return adata.layers[key][:, var_indices]
            elif var_indices is None:
                return adata.layers[key][obs_indices, :]
            else:
                return adata.layers[key][obs_indices, var_indices]
                
    def load_full_adata(self) -> anndata.AnnData:
        """
        Load the complete AnnData object into memory.
        
        Returns
        -------
        anndata.AnnData
            The complete AnnData object
        """
        return sc.read_h5ad(self.filepath)
        
    def get_available_keys(self) -> Dict[str, List[str]]:
        """
        Get all available keys in the AnnData object.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary containing lists of available keys for each AnnData attribute
        """
        with anndata.read_h5ad(self.filepath, backed='r') as adata:
            return {
                'obsm': list(adata.obsm.keys()),
                'obsp': list(adata.obsp.keys()),
                'varm': list(adata.varm.keys()),
                'varp': list(adata.varp.keys()),
                'layers': list(adata.layers.keys()),
                'uns': list(adata.uns.keys())
            } 