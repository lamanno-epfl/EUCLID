"""
EUCLID MSI - A package for spatial lipidomics data analysis
"""

from . import preprocessing as prp
from . import embedding as em
from . import clustering as cl
from . import postprocessing as pop
from . import euclid_casecontrol as cc
from . import plotting as pl
from . import data_management

__all__ = ["prp", "em", "cl", "pop", "cc", "pl"]

# Initialize data files on import
data_management.initialize_data()

# Version
__version__ = "0.0.1b"