'''Run to-vector work in a child process so the GPU is freed on exit.

A CUDA context stays allocated until its process exits, so GPU memory
cannot be fully released from a long-lived IPython session. Running the
work in a spawned child process that loads the model itself releases
everything once the child exits.
'''
import inspect
import multiprocessing
from pathlib import Path
import queue
import traceback

queue_size = 8
poll_seconds = 1.0


def should_spawn(model, gpu, spawn, auto=False):
    '''Decide whether a call runs in a child process.
    model:     model name, path, None, or loaded model instance
    gpu:       whether the call requests CUDA
    spawn:     None for automatic, True to force, False to disable
    auto:      whether the function spawns automatically on GPU
    '''
    is_reference = model is None or isinstance(model, (str, Path))
    if spawn is None: return auto and gpu and is_reference
    if spawn and not is_reference:
        m = 'spawn=True needs a model name or path, not a loaded model'
        raise ValueError(m)
    return bool(spawn)


def run_in_child(function, *args, **kwargs):
    '''Run function in a child process and return its result.'''
    return next(iter_in_child(function, *args, **kwargs))


def iter_in_child(function, *args, **kwargs):
    '''Run function in a child process and yield results as they arrive.
    A generator function streams each item; other functions yield once.
    '''
    context = multiprocessing.get_context('spawn')
    results = context.Queue(maxsize=queue_size)
    process = context.Process(target=_child,
        args=(function, args, kwargs, results))
    process.start()
    finished = False
    try:
        while True:
            kind, value = _get(results, process)
            if kind == 'done':
                finished = True
                return
            if kind == 'error':
                raise RuntimeError(f'child process failed:\n{value}')
            yield value
    finally:
        if not finished: process.kill()
        process.join()


def _child(function, args, kwargs, results):
    try:
        output = function(*args, **kwargs)
        if inspect.isgenerator(output):
            for item in output: results.put(('item', item))
        else: results.put(('item', output))
        results.put(('done', None))
    except Exception:
        results.put(('error', traceback.format_exc()))


def _get(results, process):
    '''Get the next message, raising if the child died without sending one.'''
    while True:
        alive = process.is_alive()
        try: return results.get(timeout=poll_seconds)
        except queue.Empty:
            if alive: continue
            m = f'child process exited with code {process.exitcode}'
            raise RuntimeError(m)
