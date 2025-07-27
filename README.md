# EUCLID

Enhanced uMAIA for CLustering Lipizones, Imputation and Differential Analysis.

This package provides tools for spatial lipidomics data analysis with the following modules:

- **Preprocessing**
- **Embedding**
- **Clustering**
- **Postprocessing**
- **Case-Control Analysis**
- **Plotting**

EUCLID, available as a package, runs downstream of uMAIA (https://github.com/lamanno-epfl/uMAIA). A tutorial illustrating all its functions is available in this repo. EUCLID is still very much work in progress and just partially tested, so we expect corner cases to be all around. If you try EUCLID, we would love to hear from you!

Contact: luca.fusarbassini@epfl.ch, gioele.lamanno@epfl.ch

The files to run the tutorial are available on Zenodo: https://zenodo.org/records/15689279 (they are also downloaded automatically from the tutorial notebook)

EUCLID was developed by Luca Fusar Bassini in the La Manno and D'Angelo Labs at EPFL (2023-2025), for the Lipid Brain Atlas project. The name was inspired from the beautiful Sleep Token song: https://www.youtube.com/watch?v=DDdByJYUVeA

## Installation

Install EUCLID v0.0.5 in a fresh conda environment:

```bash
conda create --name EUCLID_ENV python=3.10 -y
conda activate EUCLID_ENV

pip install --upgrade pip

# if you are on a Mac, you also need to run this:
# conda install -c conda-forge proj pyproj shapely fiona rtree geopandas -y

# first install the complex dependencies so conda handles them for you
conda install -c conda-forge pyarrow pyzmq jupyterlab ipykernel -y

pip install euclid-msi==0.0.5 --no-deps
# for some users, this breaks - you can instead do pip install euclid-msi==0.0.5 and then conda install -c conda-forge pyzmq jupyterlab ipykernel

python -m ipykernel install \
  --user \
  --name EUCLID_ENV \
  --display-name "Python (EUCLID_ENV)"
```

