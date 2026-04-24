import torch

from . import hf_batch_helper
from . import load
from . import model_registry
from . import spidr_batch_helper

sample_rate = 16_000
reserved_gpu_gb = 1.0
free_gpu_gb = 1.0
estimated_embedding_mb_per_second = 2.0
embedding_safety_factor = 4.0


def handle_batch(audio_arrays, model=None, gpu=False, numpify_output=True,
    batch_size=None):
    '''Run batched embedding extraction with multi-batch coordination.'''
    model = load.prepare_model(model, gpu)
    model_type = model_registry.model_to_type(model)
    batches = split_audio_arrays(audio_arrays, batch_size=batch_size)
    items = []
    for batch_audio_arrays in batches:
        outputs = single_batch_to_outputs(batch_audio_arrays, model, model_type)
        if numpify_output:
            outputs = [numpify(item) for item in outputs]
        items.extend(outputs)
    return items

def compute_embedding_batch_size(n_items, length_seconds, gpu_size_gb,
    batch_size=None):
    '''Resolve a defensive embedding batch size from coarse GPU limits.'''
    n_items = int(n_items)
    if n_items <= 0:
        raise ValueError('n_items must be greater than zero')
    length_seconds = float(length_seconds)
    if length_seconds <= 0:
        raise ValueError('length_seconds must be greater than zero')
    gpu_size_gb = float(gpu_size_gb)
    if gpu_size_gb <= 0:
        raise ValueError('gpu_size_gb must be greater than zero')
    if batch_size is not None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError('batch_size must be greater than zero')
        batch_size = min(batch_size, n_items)
        print(f'embedding batch size: {batch_size} items (explicit)')
        return batch_size
    usable_gb = max(1.0, gpu_size_gb - reserved_gpu_gb - free_gpu_gb)
    usable_bytes = usable_gb * (1024 ** 3)
    item_bytes = (
        length_seconds * estimated_embedding_mb_per_second *
        embedding_safety_factor * (1024 ** 2)
    )
    item_count = int(usable_bytes // item_bytes)
    item_count = max(1, min(n_items, item_count))
    m = f'embedding batch size: {item_count} items '
    m += f'(estimated from {gpu_size_gb:g} GB GPU, '
    m += f'{length_seconds:.2f}s/item)'
    print(m)
    return item_count


def split_audio_arrays(audio_arrays, batch_size=None):
    '''Split audio arrays into fixed-size batches.'''
    if batch_size is None:
        yield list(audio_arrays)
        return
    yield from split_audio_arrays_by_count(audio_arrays, batch_size)


def split_audio_arrays_by_count(audio_arrays, batch_size):
    '''Split audio arrays into fixed-size batches.'''
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError('batch_size must be greater than zero')
    batch = []
    for audio_array in audio_arrays:
        batch.append(audio_array)
        if len(batch) == batch_size:
            yield batch
            batch = []
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
