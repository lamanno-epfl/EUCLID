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

The files to run the tutorial are available on Zenodo: 

EUCLID was developed by Luca Fusar Bassini in the La Manno and D'Angelo Labs at EPFL (2023-2025), for the Lipid Brain Atlas project. The name was inspired from the beautiful Sleep Token song: https://www.youtube.com/watch?v=DDdByJYUVeA

## Installation

Install EUCLID v0.0.4 in a fresh conda environment:

```bash
conda create --name EUCLID_ENV python=3.10 -y
conda activate EUCLID_ENV
pip install --upgrade pip
pip install euclid-msi==0.0.4 jupyterlab ipykernel
python -m ipykernel install \
  --user \
  --name EUCLID_ENV \
  --display-name "Python (EUCLID_ENV)"
```

