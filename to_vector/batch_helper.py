import torch

from . import hf_batch_helper
from . import load
from . import model_registry
from . import spidr_batch_helper

sample_rate = 16_000
default_gpu_batch_minutes = 10.0
default_cpu_batch_minutes = 200.0


def handle_batch(audio_arrays, model=None, gpu=False, numpify_output=True,
    batch_minutes=None):
    '''Run batched embedding extraction with multi-batch coordination.'''
    model = load.prepare_model(model, gpu)
    model_type = model_registry.model_to_type(model)
    batch_minutes = resolve_batch_minutes(model, batch_minutes)
    max_samples = batch_minutes_to_samples(batch_minutes)
    items = []
    for batch_audio_arrays in split_audio_arrays(audio_arrays, max_samples):
        outputs = single_batch_to_outputs(batch_audio_arrays, model, model_type)
        if numpify_output:
            outputs = [numpify(item) for item in outputs]
        items.extend(outputs)
    return items


def resolve_batch_minutes(model, batch_minutes):
    '''Resolve the batch budget in minutes from input or device defaults.'''
    if batch_minutes is None:
        if load.model_is_on_gpu(model):
            return default_gpu_batch_minutes
        return default_cpu_batch_minutes
    if batch_minutes <= 0:
        raise ValueError('batch_minutes must be greater than zero')
    return float(batch_minutes)


def batch_minutes_to_samples(batch_minutes):
    '''Convert batch minutes to a sample budget at 16 kHz.'''
    return int(batch_minutes * 60 * sample_rate)


def split_audio_arrays(audio_arrays, max_samples):
    '''Split audio arrays into batches by total sample count.'''
    batch = []
    batch_total = 0
    for audio_array in audio_arrays:
        item_size = int(len(audio_array))
        if batch and batch_total + item_size > max_samples:
            yield batch
            batch = []
            batch_total = 0
        batch.append(audio_array)
        batch_total += item_size
    if batch:
        yield batch


def single_batch_to_outputs(audio_arrays, model, model_type):
    '''Dispatch one prepared batch to the correct backend helper.'''
    if model_type == 'spidr':
        return spidr_batch_helper.audio_batch_to_outputs(audio_arrays, model)
    return hf_batch_helper.audio_batch_to_outputs(audio_arrays, model,
        model_type)


def numpify(outputs):
    '''Convert model outputs to numpy arrays.'''
    if hasattr(outputs, 'extract_features'):
        if type(outputs.extract_features) == torch.Tensor:
            outputs.extract_features = outputs.extract_features.cpu().numpy()
    hs = []
    for hidden_state in outputs.hidden_states:
        hs.append(hidden_state.cpu().numpy())
    outputs.hidden_states = hs
    return outputs
