import math

import torch
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutput

from . import audio
from . import load


def audio_to_attention_outputs(audio_array, model):
    '''Extract student attentions from a SpidR model.'''
    x = audio.standardize_audio(audio_array)
    x = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
    if load.model_is_on_gpu(model): x = x.to('cuda')
    with torch.no_grad():
        features = model.feature_extractor(x)
        features = model.feature_projection(features)
        attentions = student_attentions(model.student, features)
    outputs = BaseModelOutput(hidden_states=None)
    outputs.extract_features = None
    outputs.attentions = tuple(attentions)
    outputs.model_type = 'spidr'
    return outputs


def student_attentions(student, x, attention_mask=None):
    '''Extract per-layer student attention tensors.'''
    attentions = []
    x = x + student.pos_conv_embed(x, attention_mask)
    if student.layer_norm_first:
        x = student.layer_norm(x)
    x = student.dropout(x)
    for layer in student.layers:
        x, attention = layer_output_and_attention(layer, x, attention_mask)
        attentions.append(attention)
    return attentions


def layer_output_and_attention(layer, x, attention_mask=None):
    '''Run a transformer layer and return its output and attention map.'''
    residual = x
    if layer.layer_norm_first:
        attention_input = layer.layer_norm(x)
    else:
        attention_input = x
    attention_output, attention = self_attention_with_weights(
        layer.attention,
        attention_input,
        attention_mask)
    attention_output = layer.dropout(attention_output)
    x = residual + attention_output
    if layer.layer_norm_first:
        residual = x
        layer_result = layer.feed_forward(layer.final_layer_norm(x))
        x = layer.dropout(layer_result)
        x = residual + x
    else:
        x = layer.layer_norm(x)
        residual = x
        layer_result = layer.feed_forward(x)
        x = layer.dropout(layer_result)
        x = layer.final_layer_norm(residual + x)
    return x, attention


def self_attention_with_weights(attention, x, attention_mask=None):
    '''Compute self-attention output and attention weights locally.'''
    batch, seq, dim = x.shape
    head_dim = dim // attention.num_heads
    qkv = attention.qkv(x)
    q, k, v = qkv.split(attention.embed_dim, dim=2)
    k = k.view(batch, seq, attention.num_heads, head_dim).transpose(1, 2)
    q = q.view(batch, seq, attention.num_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq, attention.num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
    scores = apply_attention_mask(scores, attention_mask)
    attention_weights = torch.softmax(scores, dim=-1)
    dropout_p = attention.dropout if attention.training else 0.0
    if dropout_p:
        attention_output_weights = F.dropout(
            attention_weights,
            p=dropout_p,
            training=True)
    else:
        attention_output_weights = attention_weights
    output = torch.matmul(attention_output_weights, v)
    output = output.transpose(1, 2).contiguous().view(batch, seq, dim)
    output = attention.proj(output)
    return output, attention_weights


def apply_attention_mask(scores, attention_mask):
    '''Apply an attention mask to raw attention scores.'''
    if attention_mask is None: return scores
    if attention_mask.dtype == torch.bool:
        mask = attention_mask
        while mask.ndim < scores.ndim:
            mask = mask.unsqueeze(1)
        return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return scores + attention_mask
