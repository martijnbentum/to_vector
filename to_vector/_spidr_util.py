from pathlib import Path

import torch

from . import audio
from . import load
from . import model_registry


def prepare_model(model, gpu=False):
    '''Prepare and validate a SpidR model.'''
    if model is None or isinstance(model, (str, Path)):
        model = load.prepare_model(model, gpu)
    if model_registry.model_to_type(model) != 'spidr':
        raise ValueError('SpidR helpers require a SpidR model')
    return model


def prepare_waveform(audio_array, model):
    '''Standardize, tensorize, and place audio on the model device.'''
    x = audio.standardize_audio(audio_array)
    x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
    if load.model_is_on_gpu(model): x = x.to('cuda')
    return x
