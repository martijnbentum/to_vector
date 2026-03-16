from pathlib import Path
import torch
from . import load
from transformers.modeling_outputs import BaseModelOutput


def audio_to_attention(audio_array, model, feature_extractor = None,
    gpu = False, numpify_output = True, layer = None, head = None,
    average_heads = False):
    '''Extract attention weights from an audio array.
    audio_array             A 1D array containing audio samples.
    model                   A pretrained model or model name.
    feature_extractor       A feature extractor. If None, the default feature
                            extractor will be loaded based on the model.
    gpu                     If True, the model and inputs will be moved to GPU.
    numpify_output          If True, convert the attention to numpy arrays.
    layer                   Optional layer index.
    head                    Optional head index.
    average_heads           If True, average attention across heads.
    '''
    model, feature_extractor, gpu = load.handle_model_feature_extractor(model,
        feature_extractor, gpu, for_attention_extraction = True)
    inputs = feature_extractor(audio_array, sampling_rate = 16_000,
        return_tensors = 'pt', padding = True)
    if gpu: inputs = inputs.to('cuda')
    with torch.no_grad():
        outputs = model(**inputs, output_attentions = True)
    attention = outputs_to_attention(outputs, layer, head, average_heads,
        numpify_output = False)
    if numpify_output: attention = attention.detach().cpu().numpy()
    outputs = BaseModelOutput(hidden_states = None)
    outputs.extract_features = None
    outputs.attentions = attention
    return outputs


def filename_to_attention(audio_filename, start = 0.0, end = None,
    model = None, feature_extractor = None, gpu = False,
    numpify_output = True, layer = None, head = None,
    average_heads = False):
    '''Extract attention weights from an audio file.
    audio_filename          Path to the audio file.
    start                   Start time in seconds. Default is 0.0.
    end                     End time in seconds. Default is None.
    model                   A pretrained model or model name.
    feature_extractor       A feature extractor. If None, the default feature
                            extractor will be loaded based on the model.
    gpu                     If True, the model and inputs will be moved to GPU.
    numpify_output          If True, convert the attention to numpy arrays.
    layer                   Optional layer index.
    head                    Optional head index.
    average_heads           If True, average attention across heads.
    '''
    audio_filename = Path(audio_filename).resolve()
    array = load.load_audio(audio_filename, start, end)
    return audio_to_attention(array, model, feature_extractor, gpu,
        numpify_output, layer, head, average_heads)


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


def select_attention(attention, layer = None, head = None,
    average_heads = False):
    '''Select layers or heads from attention.
    attention               Attention tensor with shape (L,H,T,T) or
                            (B,L,H,T,T).
    layer                   Optional layer index.
    head                    Optional head index.
    average_heads           If True, average across heads.
    '''
    if layer is not None:
        if attention.ndim == 4: attention = attention[layer]
        else: attention = attention[:,layer]
    if average_heads:
        if attention.ndim == 4: attention = attention.mean(dim = 1)
        elif attention.ndim == 5: attention = attention.mean(dim = 2)
        elif attention.ndim == 3: attention = attention.mean(dim = 0)
        else: attention = attention.mean(dim = 1)
    elif head is not None:
        if attention.ndim == 4: attention = attention[:,head]
        elif attention.ndim == 5: attention = attention[:,:,head]
        elif attention.ndim == 3: attention = attention[head]
        else: attention = attention[:,head]
    return attention


def outputs_to_attention(outputs, layer = None, head = None,
    average_heads = False, numpify_output = True):
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
