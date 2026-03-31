from pathlib import Path

import numpy as np
import torch

from . import _spidr_util
from . import audio


def filename_to_codebook_indices(audio_filename, start=0.0, end=None,
    model=None, gpu=False):
    '''Convert an audio file to SpidR codebook indices.'''
    audio_filename = Path(audio_filename).resolve()
    array = audio.load_audio(audio_filename, start, end)
    return audio_to_codebook_indices(
        array,
        model=model,
        gpu=gpu)


def filename_to_codevectors(audio_filename, start=0.0, end=None,
    model=None, gpu=False):
    '''Convert an audio file to SpidR codevectors.'''
    audio_filename = Path(audio_filename).resolve()
    array = audio.load_audio(audio_filename, start, end)
    return audio_to_codevectors(
        array,
        model=model,
        gpu=gpu)


def audio_to_codebook_probabilities(audio_array, model=None, gpu=False):
    '''Convert audio to SpidR codebook probabilities.'''
    model = _spidr_util.prepare_model(model, gpu)
    x = _spidr_util.prepare_waveform(audio_array, model)
    with torch.no_grad():
        predictions = model.get_codebooks(x)
    probabilities = []
    for prediction in predictions:
        if prediction is None: continue
        probabilities.append(normalize_probability_shape(
            prediction.detach().cpu().numpy()))
    return probabilities


def audio_to_codebook_indices(audio_array, model=None, gpu=False):
    '''Convert audio to SpidR codebook indices.'''
    probabilities = audio_to_codebook_probabilities(
        audio_array,
        model=model,
        gpu=gpu)
    return probabilities_to_codebook_indices(probabilities)


def audio_to_codevectors(audio_array, model=None, gpu=False):
    '''Convert audio to SpidR codevectors.'''
    model = _spidr_util.prepare_model(model, gpu)
    codebook_indices = audio_to_codebook_indices(
        audio_array,
        model=model,
        gpu=False)
    codebooks = load_codebooks(model)
    return codebook_indices_to_codevectors(codebook_indices, codebooks)


def load_codebooks(model):
    '''Load SpidR codebook vectors from the model.'''
    return [codebook.codebook.detach().cpu().numpy()
        for codebook in model.codebooks]


def probabilities_to_codebook_indices(probabilities):
    '''Convert SpidR codebook probabilities to codebook indices.'''
    indices = []
    for probability in probabilities:
        indices.append(np.argmax(probability, axis=-1))
    return indices


def normalize_probability_shape(probability):
    '''Normalize SpidR probability outputs to (frames, codebook_size).'''
    probability = np.asarray(probability)
    if probability.ndim == 1:
        return probability[np.newaxis, :]
    if probability.ndim == 2:
        return probability
    if probability.ndim == 3 and probability.shape[0] == 1:
        return probability[0]
    raise ValueError(
        f'unsupported SpidR codebook probability shape: {probability.shape}')


def codebook_indices_to_codevectors(codebook_indices, codebooks):
    '''Map SpidR codebook indices to codevectors.'''
    codevectors = []
    for indices, codebook in zip(codebook_indices, codebooks, strict=True):
        codevectors.append(codebook[np.asarray(indices)])
    return codevectors
