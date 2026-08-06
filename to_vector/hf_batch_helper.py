import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

from . import load

def audio_batch_to_outputs(audio_arrays, model, model_type):
    '''Convert a batch of audio arrays to per-item Hugging Face outputs.'''
    feature_extractor = load.prepare_feature_extractor(model)
    gpu = load.model_is_on_gpu(model)
    arrays = [np.asarray(audio_array) for audio_array in audio_arrays]
    inputs = feature_extractor(arrays, sampling_rate=16_000,
        return_tensors='pt', padding=True)
    if gpu: inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    if hasattr(outputs, 'last_hidden_state'):
        outputs.last_hidden_state = None
    extract_features = getattr(outputs, 'extract_features', None)
    if extract_features is None and model_type == 'hubert':
        extract_features = inputs_to_cnn(inputs, model)
    output_lengths = compute_output_lengths(inputs, outputs, model)
    items = []
    for index, output_length in enumerate(output_lengths):
        item = slice_outputs(outputs, index, output_length, extract_features,
            model_type)
        items.append(item)
    return items

def audio_batch_to_cnn(audio_arrays, model, model_type):
    '''Convert a batch of audio arrays to per-item CNN-only outputs.'''
    feature_extractor = load.prepare_feature_extractor(model)
    gpu = load.model_is_on_gpu(model)
    arrays = [np.asarray(audio_array) for audio_array in audio_arrays]
    inputs = feature_extractor(arrays, sampling_rate=16_000,
        return_tensors='pt', padding=True)
    if gpu: inputs = inputs.to('cuda')
    extract_features = inputs_to_cnn(inputs, model)
    output_lengths = compute_cnn_output_lengths(inputs, extract_features, model)
    items = []
    for index, output_length in enumerate(output_lengths):
        item = BaseModelOutput(hidden_states=None)
        item.extract_features = extract_features[index:index + 1,
            :output_length].detach().cpu().numpy()
        item.model_type = model_type
        items.append(item)
    return items


def compute_output_lengths(inputs, outputs, model):
    '''Resolve per-item output lengths from padded Hugging Face inputs.'''
    hidden_states = getattr(outputs, 'hidden_states', None)
    if hidden_states is None:
        raise ValueError('model outputs did not contain hidden_states')
    if 'attention_mask' not in inputs:
        default_length = int(hidden_states[0].shape[1])
        return [default_length] * hidden_states[0].shape[0]
    return _attention_mask_output_lengths(inputs, model)


def compute_cnn_output_lengths(inputs, extract_features, model):
    '''Resolve per-item output lengths for CNN-only (extract_features) outputs.'''
    if 'attention_mask' not in inputs:
        default_length = int(extract_features.shape[1])
        return [default_length] * extract_features.shape[0]
    return _attention_mask_output_lengths(inputs, model)


def _attention_mask_output_lengths(inputs, model):
    '''Resolve per-item output lengths from an attention mask.'''
    attention_mask = inputs['attention_mask']
    input_lengths = attention_mask.sum(dim=-1).to('cpu')
    if hasattr(model, '_get_feat_extract_output_lengths'):
        output_lengths = model._get_feat_extract_output_lengths(input_lengths)
        return [int(length) for length in output_lengths]
    else:
        print('Warning: model does not have _get_feat_extract_output_lengths')
        print('using stride 320 and window size 400 to estimate output lengths')
        output_lengths = [n_frames(length) for length in input_lengths]
    return output_lengths


def slice_outputs(outputs, index, output_length, extract_features, model_type):
    '''Split a batched Hugging Face output object into one per-item output.'''
    hidden_states = tuple(
        hidden_state[index:index + 1, :output_length].detach()
        for hidden_state in outputs.hidden_states
    )
    item = BaseModelOutput(hidden_states=hidden_states)
    if extract_features is not None:
        item.extract_features = extract_features[index:index + 1,
            :output_length].detach()
    item.model_type = model_type
    item.last_hidden_state = None
    return item


def inputs_to_cnn(inputs, model):
    '''Map prepared Hugging Face batch inputs to CNN features.'''
    with torch.no_grad():
        input_values = inputs['input_values']
        if 'ForPreTraining' in str(type(model)):
            outputs = model.wav2vec2.feature_extractor(input_values)
        else:
            outputs = model.feature_extractor(input_values)

    return outputs.transpose(1, 2).detach()

def n_frames(n_samples, stride=320, window_size=400):
    '''map number of input samples to number of valid frames.
    drops frames whose window does not fully fit
    '''
    if n_samples < window_size:
        return 0
    return 1 + (n_samples - window_size) // stride
