"""
data_management.py – Management of reference data files for EUCLID MSI

This module handles the downloading and management of reference data files
required for EUCLID MSI analysis.
"""

import os
import sys
import shutil
from pathlib import Path
import requests
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the data directory
DATA_DIR = Path(os.path.expanduser("~")) / ".euclid_msi" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define required files and their expected locations
REQUIRED_FILES = {
    "structures.sdf": DATA_DIR / "structures.sdf",
    "HMDB_complete.csv": DATA_DIR / "HMDB_complete.csv",
    "lipidclasscolors.h5ad": DATA_DIR / "lipidclasscolors.h5ad"
}

# Zenodo record information
ZENODO_RECORD_ID = "15650014"
ZENODO_BASE_URL = f"https://zenodo.org/record/{ZENODO_RECORD_ID}/files"

# File checksums for verification
FILE_CHECKSUMS = {
    "structures.sdf": "e36a2d72c6f863c8047a6067ef2117ef",
    "HMDB_complete.csv": "34641c9fd3c25ea131a56103bfaf060b",
    "lipidclasscolors.h5ad": "d23ea9ea2fa081f9d843670283485e78"
}

def download_file(url: str, destination: Path) -> None:
    """
    Download a file from a URL with progress bar.
    
    Parameters
    ----------
    url : str
        URL to download from
    destination : Path
        Where to save the file
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def verify_file_checksum(filepath: Path, expected_checksum: str) -> bool:
    """
    Verify the MD5 checksum of a downloaded file.
    
    Parameters
    ----------
    filepath : Path
        Path to the file to verify
    expected_checksum : str
        Expected MD5 checksum
        
    Returns
    -------
    bool
        True if checksum matches, False otherwise
    """
    import hashlib
    
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    return md5_hash.hexdigest() == expected_checksum

def check_data_files() -> bool:
    """
    Check if all required data files are present and valid.
    
    Returns
    -------
    bool
        True if all files are present and valid, False otherwise
    """
    for filename, filepath in REQUIRED_FILES.items():
        if not filepath.exists():
            return False
        if not verify_file_checksum(filepath, FILE_CHECKSUMS[filename]):
            logger.warning(f"Checksum verification failed for {filename}. Will re-download.")
            filepath.unlink()  # Delete the invalid file
            return False
    return True

def download_missing_files() -> None:
    """
    Download any missing data files from Zenodo.
    """
    for filename, filepath in REQUIRED_FILES.items():
        if not filepath.exists():
            logger.info(f"Downloading {filename}...")
            url = f"{ZENODO_BASE_URL}/{filename}?download=1"  # Note the ?download=1 parameter
            try:
                download_file(url, filepath)
                if verify_file_checksum(filepath, FILE_CHECKSUMS[filename]):
                    logger.info(f"Successfully downloaded and verified {filename}")
                else:
                    raise ValueError(f"Checksum verification failed for {filename}")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {str(e)}")
                if filepath.exists():
                    filepath.unlink()  # Clean up partial download
                raise

def get_data_path(filename: str) -> Path:
    """
    Get the path to a data file, downloading it if necessary.
    
    Parameters
    ----------
    filename : str
        Name of the required file
        
    Returns
    -------
    Path
        Path to the requested file
        
    Raises
    ------
    FileNotFoundError
        If the file cannot be downloaded
    ValueError
        If the filename is not recognized
    """
    if filename not in REQUIRED_FILES:
        raise ValueError(f"Unknown data file: {filename}")
    
    filepath = REQUIRED_FILES[filename]
    
    if not filepath.exists() or not verify_file_checksum(filepath, FILE_CHECKSUMS[filename]):
        logger.info(f"Required file {filename} not found or invalid. Attempting to download...")
        download_missing_files()
        
    if not filepath.exists():
        raise FileNotFoundError(
            f"Could not find or download {filename}. "
            "Please check your internet connection and try again."
        )
    
    return filepath

def initialize_data() -> None:
    """
    Initialize the data directory and download any missing files.
    This should be called when the package is first imported.
    """
    if not check_data_files():
        logger.info("Some required data files are missing or invalid. Downloading...")
        download_missing_files() 