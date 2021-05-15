import numpy as np
import pandas as pd
from PIL import Image
import time
import os

import librosa
import librosa.display

import deepdish as dd
from multiprocessing import Pool, set_start_method

# Suppress this warning: 
# /opt/conda/lib/python3.7/site-packages/librosa/core/audio.py:162: UserWarning: PySoundFile failed. Trying audioread instead.
# warnings.warn("PySoundFile failed. Trying audioread instead.")
import warnings
warnings.filterwarnings("ignore")

import utils

PREPROCESSED_DIR = 'data/fma_preprocessed/'
AUDIO_DIR = 'data/fma_small'
# settings
N = 3 # number of spectrograms to generate
HOP_LENGTH = 512 # number of samples per time-step in spectrogram
N_MELS = 128 # number of bins in spectrogram. Height of image
TIME_STEPS = 127 # number of time-steps. Width of image - 1

# Global Variables
# shared for preprocess_Y
small_tracks = None
small_genres = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']

def print_time(elapsed, measure=''):
    """Utility to print time taken."""
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{}: {:0>2}:{:0>2}:{:05.2f}".format(measure, int(hours),int(minutes),seconds))

def scale_minmax(X, min=0.0, max=1.0):
    """Scale X to be between min and max values"""
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

# https://stackoverflow.com/a/57204349 - save spectrogram as image
def generate_logspectrogram(y, sr, n_mels, hop_length):
    """Generate logspectrogram and return (1, 256, 256) np arr."""
    mels = librosa.feature.melspectrogram(
        y=y, 
        sr=sr,
        n_mels=n_mels,
        n_fft=hop_length*2,
        hop_length=hop_length
    )
    mels = np.log(mels + 1e-9) # add small number to avoid log(0)
    img = scale_minmax(mels, 0, 255).astype(np.uint8) # min-max scale to fit inside 8-bit range
    img = np.flip(img, axis=0) # put low frequencies at the bottom in image
    img = 255-img # invert. make black==more energy
    img = img[np.newaxis, ...] 
    
    return img

def generate_logspectrogram2(y, sr, n_mels, hop_length):
    """Different method of generating log spectrograms.
    """
    S = librosa.stft(y, n_fft=hop_length*2, hop_length=hop_length, win_length=hop_length*2)
    mel_basis = librosa.filters.mel(sr, n_fft=hop_length*2, n_mels=n_mels)
    img = np.dot(mel_basis, np.abs(S))
    img = np.log10(1+10*img)
    img = img[np.newaxis, ...]
    
    return img

def load_audio(index, audio_dir=AUDIO_DIR):
    """Load audio frome index
    Returns y, sr if successful, otherwise returns None
    """
    try:
        filename = utils.get_audio_path(AUDIO_DIR, index)
        y, sr = librosa.load(filename, sr=None, mono=True)
        return y, sr
    except Exception as e:
        print(f'Exception at index {index}, {split} split.')
        return None

def preprocess_X(index):
    """Load audio files at index and convert into N spectrograms
    Returns:
        - s - list of spectrograms of shape (1, 256, 256)
    """
    arr = []
    obj = load_audio(index)
    
    # If error loading audio then return empty list
    if obj is None:
        return []
    
    y, sr = obj
    
    for i in range(N):
        start = int(i / N * y.size)
        end = start + (TIME_STEPS * HOP_LENGTH)
        window = y[start:end]

        img = generate_logspectrogram2(window, sr, N_MELS, HOP_LENGTH)

        # If not correct shape, skip
        if img.shape != (1, N_MELS, N_MELS):
            print(f'Incorrect shape {img.shape} for index {index} in split {split}. Skipping...')                
            continue
        
        # append img of size (1, 1, 256, 256)
        # for ease of np.concatenate
        arr.append(img[np.newaxis, ...])
    
    return arr

def preprocess_Y(index, n):
    """Return 1-hot-encoded genre of audio file at index for n spectrograms
    
    The 8 fma_small genres are encoded in alphabetical order: 
    ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']
    
    Returns:
        - <1-hot vector> * n - list of np.ndarrays
    """
    
    global small_tracks
    global small_genres
    
    s = small_tracks.loc[index]['track', 'genre_top']
    s_arr = [int(s == genre) for genre in small_genres]
    s_np = np.asarray(s_arr)
    s_np = s_np[np.newaxis, ...] # Change to (8,) -> (1, 8) for ease of np.concatenate
    
    return [s_np] * n

def preprocess_XY(index):
    """Return a tuple of (spectrograms, labels)
    
    spectrograms - N-length list of spectrograms
    labels - N-length list of 1-hot vector of genres
    """
    spectrograms = preprocess_X(index)
    
    # If error generating spectrograms,
    # return empty lists
    n = len(spectrograms)
    if n == 0:
        return [], []
    
    labels = preprocess_Y(index, n)
    
    return spectrograms, labels

if __name__=='__main__':
    # Load metadata
    print('Loading metadata...')
    tracks = utils.load('data/fma_metadata/tracks.csv')
    genres = utils.load('data/fma_metadata/genres.csv')
    
    # Get small tracks
    small_tracks = tracks[tracks['set', 'subset'] <= 'small']
    
    # Initialize objects
    X = { 
        'training': np.ndarray((0, 1, N_MELS, N_MELS)), 
        'validation': np.ndarray((0, 1, N_MELS, N_MELS)), 
        'test': np.ndarray((0, 1, N_MELS, N_MELS)) 
    }
    Y = {
        'training': np.ndarray((0, 8)),
        'validation': np.ndarray((0, 8)),
        'test': np.ndarray((0, 8))
    }
    
    for split in ['test', 'validation', 'training']:
        print(f'Processing on {split} split...')
        # Get indices corresponding to subset
        indices = tracks.loc[(tracks['set', 'subset'] <= 'small') & (tracks['set', 'split'] == split)].index
        
        X_arr, Y_arr = [], []
        
        # Apply preprocess_XY to indices
        print(f'Generating {split} spectrogram, label pairs...')
        s1 = time.time()
        with Pool(4) as p:
            tuple_arr = p.map(preprocess_XY, indices)
        print_time(time.time() - s1, measure='Elapsed')
        
        # Get X_arr and Y_arr from tuple_arr
        # tuple_arr is a list of tuples, e.g.
        # [(list of spectrograms, list of labels)]
        for tup in tuple_arr:
            X_arr.extend(tup[0])
            Y_arr.extend(tup[1])
        
        # Concatenate to dictionaries
        print(f'Concatenating {split} spectrograms and labels...')
        s2 = time.time()
        X[split] = np.concatenate(X_arr, axis=0)
        Y[split] = np.concatenate(Y_arr, axis=0)
        print_time(time.time() - s2, measure='Elapsed')
    
    print(f'Saving objects...')
    s5 = time.time()
    XPATH = os.path.join(PREPROCESSED_DIR, 'X_3.h5')
    dd.io.save(XPATH, X)
    YPATH = os.path.join(PREPROCESSED_DIR, 'Y_3.h5')
    dd.io.save(YPATH, Y)
    print_time(time.time() - s5, measure='Elapsed')
    
    print(f'Done!')
    print_time(time.time() - s1, measure='Total Elapsed')
    
        
        
    