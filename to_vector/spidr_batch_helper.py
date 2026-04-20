import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import BaseModelOutput

from . import audio
from . import load


def audio_batch_to_outputs(audio_arrays, model):
    '''Convert a batch of audio arrays to per-item SpidR outputs.'''
    waveforms, output_lengths, attention_mask = prepare_waveform_batch(
        audio_arrays, model)
    with torch.no_grad():
        extract_features = model.feature_extractor(waveforms)
        extract_features = model.feature_projection(extract_features)
        hidden_states = model.student.get_intermediate_outputs(
            extract_features, attention_mask=attention_mask)
    outputs = BaseModelOutput(hidden_states=tuple(hidden_states))
    outputs.extract_features = extract_features
    return [
        slice_outputs(outputs, index, output_length)
        for index, output_length in enumerate(output_lengths)
    ]


def prepare_waveform_batch(audio_arrays, model):
    '''Standardize, pad, and place a batch of audio on the model device.'''
    tensors = []
    lengths = []
    for audio_array in audio_arrays:
        array = audio.standardize_audio(np.asarray(audio_array))
        tensor = torch.as_tensor(array, dtype=torch.float32)
        tensors.append(tensor)
        lengths.append(int(tensor.shape[0]))
    waveforms = pad_sequence(tensors, batch_first=True)
    if load.model_is_on_gpu(model):
        waveforms = waveforms.to('cuda')
        input_lengths = torch.tensor(lengths, device='cuda')
    else:
        input_lengths = torch.tensor(lengths)
    output_lengths = resolve_output_lengths(model, input_lengths)
    attention_mask = make_attention_mask(output_lengths)
    return waveforms, output_lengths.tolist(), attention_mask


def resolve_output_lengths(model, input_lengths):
    '''Map waveform lengths to SpidR frontend frame lengths.'''
    output_lengths = input_lengths.clone()
    for kernel_size, stride in extractor_shapes(model):
        output_lengths = torch.div(output_lengths - kernel_size, stride,
            rounding_mode='floor') + 1
        output_lengths = torch.maximum(output_lengths,
            torch.zeros_like(output_lengths))
    return output_lengths


def extractor_shapes(model):
    '''Read convolution kernel and stride values from the SpidR frontend.'''
    conv_layers = model.feature_extractor.conv_layers
    shapes = []
    for layer in conv_layers:
        shapes.append((int(layer.kernel_size), int(layer.stride)))
    return shapes


def make_attention_mask(output_lengths):
    '''Create the SpidR batch attention mask for padded frame sequences.'''
    batch_size = int(output_lengths.shape[0])
    max_length = int(output_lengths.max())
    padding_mask = torch.arange(max_length,
        device=output_lengths.device).expand(batch_size, max_length)
    padding_mask = padding_mask >= output_lengths[:, None]
    return ~padding_mask[:, None, None, :].expand(batch_size, 1, max_length,
        max_length)


def slice_outputs(outputs, index, output_length):
    '''Split a batched SpidR output object into one per-item output.'''
    hidden_states = tuple(
        hidden_state[index:index + 1, :output_length].detach()
        for hidden_state in outputs.hidden_states
    )
    item = BaseModelOutput(hidden_states=hidden_states)
    item.extract_features = outputs.extract_features[index:index + 1,
        :output_length].detach()
    item.model_type = 'spidr'
    item.last_hidden_state = None
    return item
