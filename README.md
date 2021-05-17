# music-genre-classification
Music genre classification using Free Music Archive dataset, Pytorch, and Google Cloud Platform.

## Getting Started
- Create Google Cloud Bucket to store data and training code.
- Run `data.ipynb` once to download FMA data and utilities
- Make sure to `pip install -r requirements.txt`
    - The `librosa` library requires `libsndfile-dev`, which can be installed with `sudo apt-get install libsndfile1-dev`
- Run `preprocessing.py` to generate spectrograms (as .h5 files) in `data/fma_preprocessed`
- Upload spectrograms to Google Cloud Bucket.
- Edit `cloud_train.sh` to specify parameters (comment out `config.yaml` section to run without hyperparameter tuning) 
- Run `source cloud_train.sh`

## Structure
- `data/` contains FMA data
- `trainer/` contains experiment models, training loop, and Dataset utility classes
- `results/` contains final saved models
- `cloud_train.sh` submits training job to GCP AI Platform
- `preprocessing.py` is run to generate spectrograms
- `setup.py` is for GCP setup
- `utils.py` contains utils to load FMA data

## Notebooks
- `eda.ipynb` contains data exploration
- `data.ipynb` downloads data from FMA
- `cloud_train.ipynb` submits training job to GCP AI Platform (only used for testing code)
