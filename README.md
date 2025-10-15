# EUCLID

Enhanced uMAIA for CLustering Lipizones, Imputation and Differential Analysis.

This package provides tools for spatial lipidomics data analysis with the following modules:

- **Preprocessing**
- **Embedding**
- **Clustering**
- **Postprocessing**
- **Case-Control Analysis**
- **Plotting**

EUCLID, available as a package, runs downstream of uMAIA (https://github.com/lamanno-epfl/uMAIA). A tutorial illustrating all its functions is available in this repo. EUCLID is still very much work in progress and just partially tested, so we expect corner cases to be all around. We tested EUCLID on Linux and partially Mac, but not Windows. If you try EUCLID, we would love to hear from you!

Contact: luca.fusarbassini@epfl.ch, gioele.lamanno@epfl.ch

EUCLID was developed by Luca Fusar Bassini in the La Manno and D'Angelo Labs at EPFL (2023-2025), for the Lipid Brain Atlas project, as described in our manuscript: https://www.biorxiv.org/cgi/content/short/2025.10.13.682018v1. The name was inspired from the beautiful Sleep Token song: https://www.youtube.com/watch?v=DDdByJYUVeA

## Installation

Install EUCLID v0.0.6 in a fresh conda environment (~ 10 minutes):

```bash
conda create --name EUCLID_ENV python=3.10 -y
conda activate EUCLID_ENV

pip install --upgrade pip

# if you are on a Mac, you also need to run this:
# conda install -c conda-forge proj pyproj shapely fiona rtree geopandas -y

# first install the complex dependencies so conda handles them for you
conda install mamba -c conda-forge -y
mamba install -c conda-forge cxx-compiler pyarrow pyzmq jupyterlab ipykernel -y

pip install euclid-msi==0.0.6 tables

python -m ipykernel install \
  --user \
  --name EUCLID_ENV \
  --display-name "Python (EUCLID_ENV)"
```
To try EUCLID, download from this repo the notebook euclid_pregnancy_tutorial.ipynb - it will autonomously download from Zenodo the uMAIA-normalized data and masks and the metadata needed for the tutorial.
