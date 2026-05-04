import math
import threading
from pathlib import Path
from queue import Queue

from progressbar import progressbar
import torch

from . import hf_batch_helper
from . import load
from . import model_registry
from . import spidr_batch_helper


sample_rate = 16_000
estimated_embedding_mb_per_second = 2.0
embedding_safety_factor = 4.0

def make_audio_queue(batches):
    prefetch_queue = Queue(maxsize=2)  # buffer at most one batch ahead

    def producer():
        try:
            for batch in batches:
                audio_arrays = [load.load_audio(fn, s, e) for fn, s, e in batch]
                prefetch_queue.put(('audio', audio_arrays))
        except Exception as exc:
            prefetch_queue.put(('error', exc))
        finally:
            prefetch_queue.put(('done', None))

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    return prefetch_queue


def handle_batching(filenames, starts = None, ends = None, model=None, gpu=False, 
    numpify_output=True, batch_size=None):
    '''Run batched embedding extraction with multi-batch coordination.'''
    return list(iter_handle_batching(filenames, starts=starts, ends=ends,
        model=model, gpu=gpu, numpify_output=numpify_output,
        batch_size=batch_size))


def iter_handle_batching(filenames, starts = None, ends = None, model=None,
    gpu=False, numpify_output=True, batch_size=None):
    '''Yield embedding outputs in input order while batching internally.'''
    if gpu is True and ends is None and batch_size is None: 
        m = 'ends must be provided when gpu is True to compute batch size'
        raise ValueError(m)
    fn = [Path(filename).resolve() for filename in filenames]
    if len(filenames) == 0: return
    if starts is None and ends is None: pass
    elif ends is None and len(fn) == len(starts): pass
    elif starts is None and len(fn) == len(ends): pass
    elif len(fn) == len(starts) == len(ends): pass
    else: 
        raise ValueError('filenames, starts, and ends must have the same length')
    starts = _check_batch_values(starts, len(filenames), 0.0, 'starts')
    ends = _check_batch_values(ends, len(filenames), None, 'ends')
    model = load.prepare_model(model, gpu)
    model_type = model_registry.model_to_type(model)
    if batch_size is None and gpu is True:
        durations = _compute_durations(starts, ends)
        batch_size = compute_embedding_batch_size(durations, model)
    batch_size = _check_batch_size(batch_size)
    if not batch_size is None: print(f'batch size: {batch_size}, gpu: {gpu}') 
    else: print(f'no batching, gpu: {gpu}')
    input_items = zip(filenames, starts, ends)
    batches = split(input_items, batch_size=batch_size)
    prefetch_queue = make_audio_queue(batches)
    max_value = math.ceil(len(filenames) / batch_size) if batch_size else 1
    print(f'processing batches: {max_value}')
    for _ in progressbar(range(max_value)):
        queue_type, queue_value = prefetch_queue.get()
        if queue_type == 'done': break
        if queue_type == 'error': raise queue_value
        audio_arrays = queue_value
        outputs = single_batch_to_outputs(audio_arrays, model, model_type)
        if numpify_output: outputs = [numpify(item) for item in outputs]
        if gpu: torch.cuda.empty_cache()
        yield from outputs

def compute_embedding_batch_size(durations, model):
    '''compute a defensive embedding batch size from coarse GPU limits.'''
    n_items = len(durations)
    length_seconds = max(durations) 
    if length_seconds <= 0: 
        raise ValueError('segment durations must be greater than zero')
    free_gb = model_gpu_free_gb(model)
    usable_gb = free_gb * 0.9
    usable_bytes = usable_gb * (1024 ** 3)
    item_bytes = compute_item_bytes(length_seconds)
    item_mb = item_bytes / (1024 ** 2)
    item_count = int(usable_bytes // item_bytes)
    item_count = max(1, min(n_items, item_count))
    m = f'embedding batch size: {item_count} items '
    m += f'(estimated from {free_gb:g} GB GPU, '
    m += f'{length_seconds:.2f}s/item, '
    m += f'{item_mb:.2f} MB/item)'
    print(m)
    return item_count

def compute_item_bytes(length_seconds):
    item_bytes = length_seconds * estimated_embedding_mb_per_second
    item_bytes *= embedding_safety_factor
    item_bytes *= (1024 ** 2)
    return item_bytes

def model_gpu_free_gb(model):
    '''Return the free memory of the model GPU in gigabytes.'''
    device = load.model_device(model)
    if device.type != 'cuda': raise ValueError('model must be on GPU')
    device_index = device.index
    if device_index is None:device_index = torch.cuda.current_device()
    free_bytes, _ = torch.cuda.mem_get_info(device_index)
    return free_bytes / (1024 ** 3)


def split(input_items, batch_size=None):
    '''Split input_items into fixed-size batches.'''
    if batch_size is None:
        yield list(input_items)
        return
    yield from split_by_count(input_items, batch_size)


def split_by_count(input_items, batch_size):
    '''Split input items into fixed-size batches.'''
    batch_size = _check_batch_size(batch_size)
    batch = []
    for input_item in input_items:
        batch.append(input_item)
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
        del hidden_state
    outputs.hidden_states = hs
    return outputs

def _check_batch_values(values, expected_length, default, name):
    if values is None:
        return [default] * expected_length
    values = list(values)
    if len(values) != expected_length:
        m = f'{name} must have the same length as audio_filenames'
        raise ValueError(m)
    return values

def _check_batch_size(batch_size):
    if batch_size is None: return None
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError('batch_size must be greater than zero')
    return batch_size

def _compute_durations(starts, ends):
    durations = []
    for start, end in zip(starts, ends):
        if end is None: continue
        duration = end - start
        if duration < 0:
            raise ValueError('end time must be greater than start time')
        durations.append(duration)
    if len(durations) == 0:
        raise ValueError('at least one duration must be computable') 
    return durations
