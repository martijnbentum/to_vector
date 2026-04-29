from pathlib import Path

import torch
from transformers.modeling_outputs import BaseModelOutput

from . import _spidr_util
from . import audio
from . import batch_helper
from . import load
from . import model_registry


def filename_to_vector(audio_filename, start=0.0, end=None, model=None,
    gpu=False, numpify_output=True):
    '''Convert an audio file to a vector using a pretrained model.
    audio_filename:  path to the audio file
    start:           segment start time in seconds
    end:             segment end time in seconds
    model:           pretrained model instance or model name
    numpify_output:  whether to convert outputs to numpy
    '''
    audio_filename = Path(audio_filename).resolve()
    array = audio.load_audio(audio_filename, start, end)
    outputs = audio_to_vector(array, model, gpu, numpify_output)
    return outputs

def filename_batch_to_vector(audio_filenames, starts=None, ends=None,
    model=None, gpu=False, numpify_output=True, batch_size=None):
    '''Convert multiple audio files to embeddings.
    audio_filenames: sequence of audio file paths
    starts:          optional sequence of segment starts in seconds
    ends:            optional sequence of segment ends in seconds
    model:           pretrained model instance or model name
    gpu:             whether to request CUDA
    numpify_output:  whether to convert outputs to numpy
    batch_size:      optional item count per batch
    '''
    outputs = list(iter_filename_batch_to_vector(audio_filenames, starts=starts,
        ends=ends, model=model, gpu=gpu, numpify_output=numpify_output,
        batch_size=batch_size))
    if len(audio_filenames) != len(outputs):
        m = f'iter_filename_batch_to_vector() returned {len(outputs)} outputs '
        m += f', but expected {len(audio_filenames)} outputs'
        raise ValueError(m)
    return outputs


def iter_filename_batch_to_vector(audio_filenames, starts=None, ends=None,
    model=None, gpu=False, numpify_output=True, batch_size=None):
    '''Yield embeddings for multiple audio files in input order.
    audio_filenames: sequence of audio file paths
    starts:          optional sequence of segment starts in seconds
    ends:            optional sequence of segment ends in seconds
    model:           pretrained model instance or model name
    gpu:             whether to request CUDA
    numpify_output:  whether to convert outputs to numpy
    batch_size:      optional item count per batch
    '''
    yield from batch_helper.iter_handle_batching(audio_filenames, starts, ends,
        model, gpu, numpify_output, batch_size)


def filename_to_cnn(audio_filename, start=0.0, end=None, model=None, gpu=False):
    '''Convert an audio file to features using a pretrained model.
    audio_filename:  path to the audio file
    start:           segment start time in seconds
    end:             segment end time in seconds
    model:           pretrained model instance or model name
    gpu:             whether to request CUDA
    '''
    audio_filename = Path(audio_filename).resolve()
    array = audio.load_audio(audio_filename, start, end)
    outputs = audio_to_cnn(array, model, gpu) 
    o = BaseModelOutput(hidden_states=None)
    o.extract_features = outputs
    return o


def audio_to_vector(audio_array, model=None, gpu=False, numpify_output=True):
    '''Convert an audio array to a vector using a pretrained model.
    audio_array:     1D audio samples
    model:           pretrained model instance or model name
    gpu:             whether to request CUDA
    numpify_output:  whether to convert outputs to numpy
    '''
    model = load.prepare_model(model, gpu)
    model_type = model_registry.model_to_type(model)
    if model_type == 'spidr':
        return _spidr_audio_to_vector(audio_array, model, numpify_output)
    if model_type in ('wav2vec2', 'wavlm', 'hubert'):
        return _huggingface_audio_to_vector(audio_array, model, model_type,
            numpify_output)
    return _huggingface_audio_to_vector(audio_array, model, model_type,
        numpify_output)


def _huggingface_audio_to_vector(audio_array, model, model_type,
    numpify_output=True):
    '''Convert an audio array with a Hugging Face feature extractor.'''
    feature_extractor = load.prepare_feature_extractor(model)
    gpu = load.model_is_on_gpu(model)
    inputs = feature_extractor(audio_array, sampling_rate=16_000,
        return_tensors='pt', padding=True)
    if gpu: inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    if hasattr(outputs, 'last_hidden_state'):
        outputs.last_hidden_state = None
    outputs.model_type = model_type
    if not hasattr(outputs, 'extract_features'):
        if model_type == 'hubert':
            o = audio_to_cnn(audio_array, model, gpu)
            outputs.extract_features = o
    if numpify_output: return batch_helper.numpify(outputs)
    return outputs


def _spidr_audio_to_vector(audio_array, model, numpify_output=True):
    '''Convert an audio array with SpidR-specific frontend logic.'''
    x = _spidr_util.prepare_waveform(audio_array, model)
    with torch.no_grad():
        extract_features = model.feature_extractor(x)
        extract_features = model.feature_projection(extract_features)
        hidden_states = model.student.get_intermediate_outputs(
            extract_features)
    outputs = BaseModelOutput(
        hidden_states=tuple(hidden_states))
    outputs.extract_features = extract_features
    outputs.model_type = 'spidr'
    if numpify_output: return batch_helper.numpify(outputs)
    return outputs

def audio_to_cnn(audio, model=None, gpu=False):
    '''Convert an audio array to features using a pretrained model.
    audio:         1D audio samples
    model:         pretrained model instance or model name
    gpu:           whether to request CUDA
    '''
    model = load.prepare_model(model, gpu)
    if model_registry.model_to_type(model) == 'spidr':
        m = 'audio_to_cnn() is not implemented for SpidR models yet. '
        m += 'Check whether the convolutional frontend can be called '
        m += 'directly on the SpidR model.'
        raise ValueError(m)
    feature_extractor = load.prepare_feature_extractor(model)
    gpu = load.model_is_on_gpu(model)
    array = audio
    inputs = feature_extractor(array, sampling_rate=16_000, return_tensors='pt',
        padding=True)
    if gpu: inputs = inputs.to('cuda')
    with torch.no_grad():
        input_values = inputs['input_values']
        if 'ForPreTraining' in str(type(model)):
            outputs = model.wav2vec2.feature_extractor(input_values)
        else:
            outputs = model.feature_extractor(input_values)
    outputs = outputs.transpose(1, 2).detach().cpu().numpy()
    return outputs
