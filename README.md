# music-genre-classification
Music genre classification using Free Music Archive dataset and Pytorch.

## Getting Started
- Run `data.ipynb` once to download FMA data and utilities
- Make sure to `pip install -r requirements.txt`
- The `librosa` library requires `libsndfile-dev`, which can be installed with `sudo apt-get install libsndfile1-dev`

## Structure
- `data/` contains FMA data
- `trainer/` contains models and training loop

## Notebooks
- `eda.ipynb` contains data exploration
- `train_local.ipynb` submits training request to GCP AI Platform