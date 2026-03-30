import librosa


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
