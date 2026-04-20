from pathlib import Path

import torch
from transformers.modeling_outputs import BaseModelOutput

from . import _spidr_util
from . import audio
from . import hf_batch_helper
from . import load
from . import model_registry
from . import spidr_batch_helper


def filename_to_vector(audio_filename, start=0.0, end=None, model=None,
    gpu=False, identifier='', name='', numpify_output=True):
    '''Convert an audio file to a vector using a pretrained model.
    audio_filename:  path to the audio file
    start:           segment start time in seconds
    end:             segment end time in seconds
    model:           pretrained model instance or model name
    identifier:      optional identifier
    name:            optional name
    numpify_output:  whether to convert outputs to numpy
    '''
    audio_filename = Path(audio_filename).resolve()
    array = audio.load_audio(audio_filename, start, end)
    outputs = audio_to_vector(array, model, gpu, numpify_output)
    outputs = add_info(outputs, audio_filename, start, end, identifier, name)
    return outputs


def filename_batch_to_vector(audio_filenames, starts=None, ends=None,
    model=None, gpu=False, identifiers=None, names=None,
    numpify_output=True):
    '''Convert multiple audio files to embeddings.
    audio_filenames: sequence of audio file paths
    starts:          optional sequence of segment starts in seconds
    ends:            optional sequence of segment ends in seconds
    model:           pretrained model instance or model name
    gpu:             whether to request CUDA
    identifiers:     optional sequence of identifiers
    names:           optional sequence of names
    numpify_output:  whether to convert outputs to numpy
    '''
    audio_filenames = [Path(filename).resolve() for filename in audio_filenames]
    if not audio_filenames:
        raise ValueError('audio_filenames must contain at least one filename')
    starts = _resolve_batch_values(starts, len(audio_filenames), 0.0, 'starts')
    ends = _resolve_batch_values(ends, len(audio_filenames), None, 'ends')
    identifiers = _resolve_batch_values(identifiers, len(audio_filenames), '',
        'identifiers')
    names = _resolve_batch_values(names, len(audio_filenames), '', 'names')
    arrays = audio.load_audio_batch(audio_filenames, starts, ends)
    outputs = audio_batch_to_vector(arrays, model, gpu, numpify_output)
    items = []
    for output, audio_filename, start, end, identifier, name in zip(
        outputs, audio_filenames, starts, ends, identifiers, names):
        item = add_info(output, audio_filename, start, end, identifier, name)
        items.append(item)
    return items


def filename_to_cnn(audio_filename, start=0.0, end=None, model=None,
    gpu=False, identifier='', name=''):
    '''Convert an audio file to features using a pretrained model.
    audio_filename:  path to the audio file
    start:           segment start time in seconds
    end:             segment end time in seconds
    model:           pretrained model instance or model name
    gpu:             whether to request CUDA
    identifier:      optional identifier
    name:            optional name
    '''
    audio_filename = Path(audio_filename).resolve()
    array = audio.load_audio(audio_filename, start, end)
    outputs = audio_to_cnn(array, model, gpu, identifier, name)
    o = BaseModelOutput(hidden_states=None)
    o.extract_features = outputs
    outputs = add_info(o, audio_filename, start, end, identifier, name)
    return outputs


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


def audio_batch_to_vector(audio_arrays, model=None, gpu=False,
    numpify_output=True):
    '''Convert multiple audio arrays to embeddings.
    audio_arrays:    sequence of 1D audio sample arrays
    model:           pretrained model instance or model name
    gpu:             whether to request CUDA
    numpify_output:  whether to convert outputs to numpy
    '''
    audio_arrays = _require_audio_batch(audio_arrays)
    model = load.prepare_model(model, gpu)
    model_type = model_registry.model_to_type(model)
    if model_type == 'spidr':
        return _spidr_audio_batch_to_vector(audio_arrays, model,
            numpify_output)
    return _huggingface_audio_batch_to_vector(audio_arrays, model, model_type,
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
    if numpify_output: return numpify(outputs)
    return outputs


def _huggingface_audio_batch_to_vector(audio_arrays, model, model_type,
    numpify_output=True):
    '''Convert multiple audio arrays with one Hugging Face forward pass.'''
    items = hf_batch_helper.audio_batch_to_outputs(audio_arrays, model,
        model_type)
    if numpify_output:
        return [numpify(item) for item in items]
    return items


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
    if numpify_output: return numpify(outputs)
    return outputs


def _spidr_audio_batch_to_vector(audio_arrays, model, numpify_output=True):
    '''Convert multiple audio arrays with one SpidR frontend pass.'''
    items = spidr_batch_helper.audio_batch_to_outputs(audio_arrays, model)
    if numpify_output:
        return [numpify(item) for item in items]
    return items


def add_info(outputs, audio_filename, start, end, identifier, name):
    '''Add information about the audio file to the output object.
    outputs:         output object to update
    audio_filename:  path to the audio file
    start:           segment start time
    end:             segment end time
    identifier:      optional identifier
    name:            optional name
    '''
    audio_filename = str(audio_filename)
    outputs.audio_filename = audio_filename
    outputs.start_time = start
    outputs.end_time = end
    outputs.identifier = identifier
    outputs.name = name
    return outputs


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


def numpify(outputs):
    '''Convert the outputs of a model to numpy arrays.
    outputs            The output object from the model.
    '''
    if hasattr(outputs, 'extract_features'):
        if type(outputs.extract_features) == torch.Tensor:
            outputs.extract_features = outputs.extract_features.cpu().numpy()
    hs = []
    for hidden_state in outputs.hidden_states:
        hs.append(hidden_state.cpu().numpy())
    outputs.hidden_states = hs
    return outputs


def audio_to_cnn(audio, model=None, gpu=False, identifier='', name=''):
    '''Convert an audio array to features using a pretrained model.
    audio:         1D audio samples
    model:         pretrained model instance or model name
    gpu:           whether to request CUDA
    identifier:    optional identifier
    name:          optional name
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
