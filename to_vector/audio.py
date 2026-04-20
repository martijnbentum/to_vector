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
    audio, _ = librosa.load(filename, sr=16000, offset=start, duration=duration)
    return audio

def load_audio_batch(filenames, starts=None, ends=None):
    '''Load multiple audio segments from disk at 16 kHz.
    filenames:   sequence of audio file paths
    starts:      optional sequence of segment starts in seconds
    ends:        optional sequence of segment ends in seconds
    '''
    filenames = list(filenames)
    if not filenames:
        raise ValueError('filenames must contain at least one filename')
    starts = _resolve_batch_times(starts, len(filenames), 0.0, 'starts')
    ends = _resolve_batch_times(ends, len(filenames), None, 'ends')
    audios = []
    for filename, start, end in zip(filenames, starts, ends):
        audio = load_audio(filename, start, end)
        audios.append(audio)
    return audios

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


def load_audio_batch_milliseconds(filenames, starts=None, ends=None):
    '''Load multiple audio segments using millisecond time inputs.
    filenames:   sequence of audio file paths
    starts:      optional sequence of segment starts in milliseconds
    ends:        optional sequence of segment ends in milliseconds
    '''
    filenames = list(filenames)
    if not filenames:
        raise ValueError('filenames must contain at least one filename')
    starts_ms = _resolve_batch_times(starts, len(filenames), 0, 'starts')
    ends_ms = _resolve_batch_times(ends, len(filenames), None, 'ends')
    _validate_millisecond_batch_times(starts_ms, ends_ms)
    starts_sec = [start / 1000 for start in starts_ms]
    ends_sec = [None if end is None else end / 1000 for end in ends_ms]
    return load_audio_batch(filenames, starts_sec, ends_sec)


def standardize_audio(x):
    '''Standardize audio to zero mean and unit variance.'''
    x = np.asarray(x)
    return (x - x.mean()) / (x.std() + 1e-8)


def _resolve_batch_times(values, expected_length, default, name):
    if values is None:
        return [default] * expected_length
    values = list(values)
    if len(values) != expected_length:
        m = f'{name} must have the same length as filenames'
        raise ValueError(m)
    return values


def _validate_millisecond_batch_times(starts, ends):
    for start, end in zip(starts, ends):
        if isinstance(start, float) or isinstance(end, float):
            m = 'start and end must be integers representing milliseconds'
            raise TypeError(m)
