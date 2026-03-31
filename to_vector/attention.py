from pathlib import Path

import torch
from transformers.modeling_outputs import BaseModelOutput

from . import _spidr_attention
from . import audio
from . import load
from . import model_registry
from .to_embeddings import add_info


def audio_to_attention(audio_array, model, gpu=False, numpify_output=True,
    layer=None, head=None, average_heads=False):
    '''Extract attention weights from an audio array.
    audio_array:     1D audio samples
    model:           pretrained model or model name
    gpu:             whether to request CUDA
    numpify_output:  whether to convert attention to numpy
    layer:           optional layer index
    head:            optional head index
    average_heads:   whether to average heads
    '''
    model = load.prepare_model(model, gpu, for_attention_extraction=True)
    model_type = model_registry.model_to_type(model)
    if model_type == 'spidr':
        return _spidr_audio_to_attention(audio_array, model, numpify_output,
            layer, head, average_heads)
    return _huggingface_audio_to_attention(audio_array, model, model_type,
        numpify_output, layer, head, average_heads)


def _huggingface_audio_to_attention(audio_array, model, model_type,
    numpify_output=True, layer=None, head=None, average_heads=False):
    '''Extract attention weights with a Hugging Face model.'''
    feature_extractor = load.prepare_feature_extractor(model)
    gpu = load.model_is_on_gpu(model)
    inputs = feature_extractor(audio_array, sampling_rate=16_000,
        return_tensors='pt', padding=True)
    if gpu: inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attention = outputs_to_attention(outputs, layer, head, average_heads,
        numpify_output=False)
    return pack_attention_outputs(attention, model_type, numpify_output)


def _spidr_audio_to_attention(audio_array, model, numpify_output=True,
    layer=None, head=None, average_heads=False):
    '''Extract attention weights with local SpidR logic.'''
    outputs = _spidr_attention.audio_to_attention_outputs(audio_array, model)
    attention = outputs_to_attention(outputs, layer, head, average_heads,
        numpify_output=False)
    return pack_attention_outputs(attention, 'spidr', numpify_output)


def pack_attention_outputs(attention, model_type, numpify_output=True):
    '''Create a compact attention output object.'''
    if numpify_output: attention = attention.detach().cpu().numpy()
    outputs = BaseModelOutput(hidden_states=None)
    outputs.extract_features = None
    outputs.attentions = attention
    outputs.model_type = model_type
    return outputs


def filename_to_attention(audio_filename, start=0.0, end=None, model=None,
    gpu=False, numpify_output=True, layer=None, head=None,
    average_heads=False):
    '''Extract attention weights from an audio file.
    audio_filename:  path to the audio file
    start:           segment start time in seconds
    end:             segment end time in seconds
    model:           pretrained model or model name
    gpu:             whether to request CUDA
    numpify_output:  whether to convert attention to numpy
    layer:           optional layer index
    head:            optional head index
    average_heads:   whether to average heads
    '''
    audio_filename = Path(audio_filename).resolve()
    array = audio.load_audio(audio_filename, start, end)
    outputs = audio_to_attention(array, model, gpu, numpify_output, layer, head,
        average_heads)
    return add_info(outputs, audio_filename, start, end, '', '')


def attention_to_tensor(attention):
    '''Convert attention output to a tensor.
    attention               A tensor or array with attention values.
    '''
    if type(attention) == torch.Tensor:
        return attention
    return torch.as_tensor(attention)


def stack_attentions(attentions):
    '''Stack attention tensors.
    attentions              Sequence of layer attention tensors.
    '''
    layers = []
    for attention in attentions:
        layers.append(attention_to_tensor(attention))
    stacked = torch.stack(layers)
    if stacked.ndim == 5:
        stacked = stacked.permute(1,0,2,3,4)
        if stacked.shape[0] == 1:
            stacked = stacked[0]
    return stacked


def select_attention(attention, layer=None, head=None, average_heads=False):
    '''Select layers or heads from attention.
    attention               Attention tensor with shape (L,H,T,T) or
                            (B,L,H,T,T).
    layer                   Optional layer index.
    head                    Optional head index.
    average_heads           If True, average across heads.
    '''
    if layer is not None:
        if attention.ndim == 4: attention = attention[layer]
        else: attention = attention[:, layer]
    if average_heads:
        if attention.ndim == 4: attention = attention.mean(dim=1)
        elif attention.ndim == 5: attention = attention.mean(dim=2)
        elif attention.ndim == 3: attention = attention.mean(dim=0)
        else: attention = attention.mean(dim=1)
    elif head is not None:
        if attention.ndim == 4: attention = attention[:, head]
        elif attention.ndim == 5: attention = attention[:, :, head]
        elif attention.ndim == 3: attention = attention[head]
        else: attention = attention[:, head]
    return attention


def outputs_to_attention(outputs, layer=None, head=None,
    average_heads=False, numpify_output=True):
    '''Convert model outputs to attention arrays.
    outputs                 Model outputs with attention tensors.
    layer                   Optional layer index.
    head                    Optional head index.
    average_heads           If True, average across heads.
    '''
    if outputs.attentions is None:
        raise ValueError('model outputs do not contain attentions')
    attention = stack_attentions(outputs.attentions)
    attention = select_attention(attention, layer, head, average_heads)
    if numpify_output: attention = attention.detach().cpu().numpy()
    return attention
