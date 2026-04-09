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
    audio, sr = librosa.load(filename, sr=16000, offset=start,
        duration=duration)
    return audio

def load_audio_milliseconds(filename, start=0, end=None):
    if isinstance(start, float) or isinstance(end, float):
        m = 'start and end must be integers representing milliseconds'
        raise TypeError(m)
    start_sec = start / 1000
    end_sec = None if end is None else end / 1000
    audio = load_audio(filename, start_sec, end_sec)
    if end_sec is None: return audio
    expected_length = int((end_sec - start_sec) * 16000)
    diff = abs(len(audio) - expected_length)
    if diff > 16000 * 0.01:
        m = f'Loaded audio length {len(audio)} differs from expected length '
        m += f'{expected_length} by more than 1%. Check the start and end '
        m += f'times and the sampling rate of the audio file.'
        raise ValueError(m)
    return audio


def standardize_audio(x):
    '''Standardize audio to zero mean and unit variance.'''
    x = np.asarray(x)
    return (x - x.mean()) / (x.std() + 1e-8)
