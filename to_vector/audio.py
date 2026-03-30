import librosa
import numpy as np


def load_audio(filename, start=0.0, end=None):
    '''Load audio from disk at 16 kHz.'''
    if end is None:
        duration = None
    else:
        if end < start:
            raise ValueError('end must be greater than or equal to start')
        duration = end - start
    audio, sr = librosa.load(
        filename,
        sr=16000,
        offset=start,
        duration=duration)
    return audio


def standardize_audio(x):
    '''Standardize audio to zero mean and unit variance.'''
    x = np.asarray(x)
    return (x - x.mean()) / (x.std() + 1e-8)
