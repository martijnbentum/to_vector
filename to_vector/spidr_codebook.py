from pathlib import Path

import numpy as np
import torch

from . import _spidr_util
from . import audio
from . import batch_helper
from . import spidr_batch_helper


def filename_to_codebook_indices(audio_filename, start=0.0, end=None,
    model=None, gpu=False):
    '''Convert an audio file to SpidR codebook indices.'''
    audio_filename = Path(audio_filename).resolve()
    array = audio.load_audio(audio_filename, start, end)
    return audio_to_codebook_indices(array, model=model, gpu=gpu)


def filename_to_codebook_probabilities(audio_filename, start=0.0, end=None,
    model=None, gpu=False):
    '''Convert an audio file to SpidR codebook probabilities.'''
    audio_filename = Path(audio_filename).resolve()
    array = audio.load_audio(audio_filename, start, end)
    return audio_to_codebook_probabilities(array, model=model, gpu=gpu)


def filename_batch_to_codebook_probabilities(audio_filenames, starts=None,
    ends=None, model=None, gpu=False, batch_minutes=None):
    '''Convert multiple audio files to SpidR codebook probabilities.'''
    audio_filenames = [Path(filename).resolve() for filename in audio_filenames]
    if not audio_filenames:
        raise ValueError('audio_filenames must contain at least one filename')
    starts = _resolve_batch_values(starts, len(audio_filenames), 0.0,
        'starts')
    ends = _resolve_batch_values(ends, len(audio_filenames), None, 'ends')
    arrays = audio.load_audio_batch(audio_filenames, starts, ends)
    return audio_batch_to_codebook_probabilities(arrays, model=model,
        gpu=gpu, batch_minutes=batch_minutes)


def filename_batch_to_codebook_indices(audio_filenames, starts=None,
    ends=None, model=None, gpu=False, batch_minutes=None):
    '''Convert multiple audio files to SpidR codebook indices.'''
    audio_filenames = [Path(filename).resolve() for filename in audio_filenames]
    if not audio_filenames:
        raise ValueError('audio_filenames must contain at least one filename')
    starts = _resolve_batch_values(starts, len(audio_filenames), 0.0,
        'starts')
    ends = _resolve_batch_values(ends, len(audio_filenames), None, 'ends')
    arrays = audio.load_audio_batch(audio_filenames, starts, ends)
    return audio_batch_to_codebook_indices(arrays, model=model, gpu=gpu,
        batch_minutes=batch_minutes)


def filename_to_codevectors(audio_filename, start=0.0, end=None,
    model=None, gpu=False):
    '''Convert an audio file to SpidR codevectors.'''
    audio_filename = Path(audio_filename).resolve()
    array = audio.load_audio(audio_filename, start, end)
    return audio_to_codevectors(array, model=model, gpu=gpu)


def filename_batch_to_codevectors(audio_filenames, starts=None, ends=None,
    model=None, gpu=False, batch_minutes=None):
    '''Convert multiple audio files to SpidR codevectors.'''
    audio_filenames = [Path(filename).resolve() for filename in audio_filenames]
    if not audio_filenames:
        raise ValueError('audio_filenames must contain at least one filename')
    starts = _resolve_batch_values(starts, len(audio_filenames), 0.0,
        'starts')
    ends = _resolve_batch_values(ends, len(audio_filenames), None, 'ends')
    arrays = audio.load_audio_batch(audio_filenames, starts, ends)
    return audio_batch_to_codevectors(arrays, model=model, gpu=gpu,
        batch_minutes=batch_minutes)


def audio_to_codebook_probabilities(audio_array, model=None, gpu=False):
    '''Convert audio to SpidR codebook probabilities.'''
    return audio_batch_to_codebook_probabilities([audio_array], model=model,
        gpu=gpu)[0]


def audio_batch_to_codebook_probabilities(audio_arrays, model=None,
    gpu=False, batch_minutes=None):
    '''Convert multiple audio arrays to SpidR codebook probabilities.'''
    audio_arrays = _require_audio_batch(audio_arrays)
    model = _spidr_util.prepare_model(model, gpu)
    batch_minutes = batch_helper.resolve_batch_minutes(model, batch_minutes)
    max_samples = batch_helper.batch_minutes_to_samples(batch_minutes)
    items = []
    for batch_audio_arrays in batch_helper.split_audio_arrays(audio_arrays,
        max_samples):
        items.extend(_single_batch_to_probabilities(batch_audio_arrays, model))
    return items


def audio_to_codebook_indices(audio_array, model=None, gpu=False):
    '''Convert audio to SpidR codebook indices.'''
    return audio_batch_to_codebook_indices([audio_array], model=model,
        gpu=gpu)[0]


def audio_batch_to_codebook_indices(audio_arrays, model=None, gpu=False,
    batch_minutes=None):
    '''Convert multiple audio arrays to SpidR codebook indices.'''
    probabilities = audio_batch_to_codebook_probabilities(audio_arrays,
        model=model, gpu=gpu, batch_minutes=batch_minutes)
    return [probabilities_to_codebook_indices(item) for item in probabilities]


def audio_to_codevectors(audio_array, model=None, gpu=False):
    '''Convert audio to SpidR codevectors.'''
    return audio_batch_to_codevectors([audio_array], model=model, gpu=gpu)[0]


def audio_batch_to_codevectors(audio_arrays, model=None, gpu=False,
    batch_minutes=None):
    '''Convert multiple audio arrays to SpidR codevectors.'''
    model = _spidr_util.prepare_model(model, gpu)
    codebook_indices = audio_batch_to_codebook_indices(audio_arrays,
        model=model, gpu=False, batch_minutes=batch_minutes)
    codebooks = load_codebooks(model)
    return [codebook_indices_to_codevectors(item, codebooks)
        for item in codebook_indices]


def load_codebooks(model):
    '''Load SpidR codebook vectors from the model.'''
    return [codebook.codebook.detach().cpu().numpy()
        for codebook in model.codebooks]


def probabilities_to_codebook_indices(probabilities):
    '''Convert SpidR codebook probabilities to codebook indices.'''
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 3:
        m = 'SpidR codebook probabilities must have shape '
        m += '(frames, codebooks, codebook_size)'
        raise ValueError(m)
    return np.argmax(probabilities, axis=-1)


def normalize_probability_shape(probability):
    '''Normalize SpidR probability outputs to (frames, codebook_size).'''
    probability = np.asarray(probability)
    if probability.ndim == 1:
        return probability[np.newaxis, :]
    if probability.ndim == 2:
        return probability
    if probability.ndim == 3 and probability.shape[0] == 1:
        return probability[0]
    m = 'unsupported SpidR codebook probability shape: '
    m += f'{probability.shape}'
    raise ValueError(m)


def codebook_indices_to_codevectors(codebook_indices, codebooks):
    '''Map SpidR codebook indices to codevectors.'''
    codebook_indices = np.asarray(codebook_indices, dtype=int)
    if codebook_indices.ndim != 2:
        raise ValueError(
            'SpidR codebook indices must have shape (frames, codebooks)')
    if codebook_indices.shape[1] != len(codebooks):
        raise ValueError('number of SpidR codebooks does not match indices')
    codevectors = []
    for index, codebook in enumerate(codebooks):
        codevectors.append(codebook[codebook_indices[:, index]])
    return np.stack(codevectors, axis=1)


def _single_batch_to_probabilities(audio_arrays, model):
    waveforms, output_lengths, attention_mask = (
        spidr_batch_helper.prepare_waveform_batch(audio_arrays, model))
    with torch.no_grad():
        predictions = model.get_codebooks(waveforms,
            attention_mask=attention_mask)
    return _slice_batched_probabilities(predictions, output_lengths)


def _slice_batched_probabilities(predictions, output_lengths):
    batch_size = len(output_lengths)
    normalized = []
    for prediction in predictions:
        if prediction is None:
            continue
        probability = prediction.detach().cpu().numpy()
        normalized.append(normalize_batched_probability_shape(probability,
            batch_size))
    if not normalized:
        raise ValueError('SpidR model returned no codebook predictions')
    items = []
    for index, output_length in enumerate(output_lengths):
        item = [probability[index, :output_length] for probability in normalized]
        items.append(np.stack(item, axis=1))
    return items


def normalize_batched_probability_shape(probability, batch_size):
    '''Normalize SpidR probability outputs to batch-first arrays.'''
    probability = np.asarray(probability)
    if probability.ndim == 3:
        return probability
    if probability.ndim == 2:
        if batch_size == 1:
            return probability[np.newaxis, :, :]
        return probability[:, np.newaxis, :]
    if probability.ndim == 1 and batch_size == 1:
        return probability[np.newaxis, np.newaxis, :]
    m = 'unsupported batched SpidR codebook probability shape: '
    m += f'{probability.shape}'
    raise ValueError(m)


def _require_audio_batch(audio_arrays):
    audio_arrays = list(audio_arrays)
    if not audio_arrays:
        raise ValueError('audio_arrays must contain at least one audio array')
    return audio_arrays


def _resolve_batch_values(values, expected_length, default, name):
    if values is None:
        return [default] * expected_length
    values = list(values)
    if len(values) != expected_length:
        m = f'{name} must have the same length as audio_filenames'
        raise ValueError(m)
    return values
