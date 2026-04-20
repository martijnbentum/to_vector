'''Normalized codebook artifacts for storage-oriented callers.'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _spidr_util
from . import audio
from . import load
from . import model_registry
from . import spidr_codebook
from . import wav2vec2_codebook


@dataclass(frozen=True)
class CodebookArtifacts:
    '''Storage-oriented normalized codebook payloads.'''

    indices: np.ndarray
    codebook_matrix: np.ndarray
    model_architecture: str


def filename_to_codebook_artifacts(audio_filename, start=0.0, end=None,
    model=None, gpu=False):
    '''Load audio from disk and return normalized codebook artifacts.'''
    array = audio.load_audio(Path(audio_filename).resolve(), start, end)
    return audio_to_codebook_artifacts(array, model=model, gpu=gpu)


def audio_to_codebook_artifacts(audio_array, model=None, gpu=False):
    '''Return normalized codebook indices and codebook matrices.

    wav2vec2:
        indices shape: (frames, 2)
        codebook_matrix shape: (codebook_size, codevector_dim_half)

    spidr:
        indices shape: (frames, heads)
        codebook_matrix shape: (heads, codebook_size, codevector_dim)
    '''
    model_architecture = _infer_model_architecture(model)
    if model_architecture == 'wav2vec2':
        return _audio_to_wav2vec2_codebook_artifacts(audio_array, model, gpu)
    if model_architecture == 'spidr':
        return _audio_to_spidr_codebook_artifacts(audio_array, model, gpu)
    raise ValueError(
        f'unsupported codebook architecture: {model_architecture!r}')


def _audio_to_wav2vec2_codebook_artifacts(audio_array, model, gpu):
    model_pt = _prepare_wav2vec2_model(model, gpu)
    indices = wav2vec2_codebook.audio_to_codebook_indices(audio_array,
        model_pt=model_pt, gpu=False)
    indices = np.asarray(indices, dtype=int)
    if indices.ndim == 1:
        indices = indices[np.newaxis, :]
    codebook_matrix = np.asarray(
        wav2vec2_codebook.load_codebook(model_pt))
    return CodebookArtifacts(indices=indices,
        codebook_matrix=codebook_matrix,
        model_architecture='wav2vec2')


def _audio_to_spidr_codebook_artifacts(audio_array, model, gpu):
    resolved_model = _spidr_util.prepare_model(model, gpu)
    indices = spidr_codebook.audio_to_codebook_indices(audio_array,
        model=resolved_model, gpu=False)
    indices = np.asarray(indices, dtype=int)
    codebook_matrix = np.stack([
        np.asarray(matrix) for matrix in spidr_codebook.load_codebooks(
            resolved_model)
    ], axis=0)
    return CodebookArtifacts(indices=indices,
        codebook_matrix=codebook_matrix,
        model_architecture='spidr')


def _prepare_wav2vec2_model(model, gpu):
    if model is None:
        return load.load_model_pt(gpu=gpu)
    if isinstance(model, (str, Path)):
        return load.load_model_pt(checkpoint=str(model), gpu=gpu)
    return model


def _infer_model_architecture(model):
    if model is None:
        return 'wav2vec2'
    if isinstance(model, (str, Path)):
        model_type = model_registry.filename_model_type(str(model))
        if model_type == 'spidr':
            return 'spidr'
        return 'wav2vec2'
    model_type = model_registry.model_to_type(model)
    if model_type == 'spidr':
        return 'spidr'
    if model_type in {'wav2vec2', 'wav2vec2-pretraining'}:
        return 'wav2vec2'
    raise ValueError(
        'codebook artifacts currently support only wav2vec2 and spidr')
