'''Run to-vector work in a child process so the GPU is freed on exit.

A CUDA context stays allocated until its process exits, so GPU memory
cannot be fully released from a long-lived IPython session. Running the
work in a spawned child process that loads the model itself releases
everything once the child exits.
'''
import inspect
import multiprocessing
from pathlib import Path
import pickle
import queue
import traceback

queue_size = 8
poll_seconds = 1.0


def should_spawn(model, gpu, spawn, auto=False):
    '''Decide whether a call runs in a child process and print why.
    Only spawns when gpu is True. Silent for spawn=False and for functions
    that do not spawn automatically (auto=False) when spawn is None.
    model:     model name, path, None, or loaded model instance
    gpu:       whether the call requests CUDA
    spawn:     None for automatic, True to request, False to disable
    auto:      whether the function spawns automatically on GPU
    '''
    is_reference = model is None or isinstance(model, (str, Path))
    if spawn and not is_reference:
        m = 'spawn=True needs a model name or path, not a loaded model'
        raise ValueError(m)
    if spawn is False or (spawn is None and not auto): return False
    if not gpu: reason = 'gpu=False'
    elif not is_reference: reason = 'model is already loaded in this process'
    else:
        if spawn: reason = 'spawn=True'
        else: reason = 'gpu=True and model is a name or path'
        print(f'running in a child process ({reason}); '
            'the GPU is freed when it exits')
        return True
    print(f'running in this process ({reason})')
    return False


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
                error, text = value
                cause = RuntimeError(f'child process traceback:\n{text}')
                raise error from cause
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
    except Exception as error:
        text = traceback.format_exc()
        # an unpicklable exception would be dropped by the queue
        try: pickle.dumps(error)
        except Exception: error = RuntimeError(text)
        results.put(('error', (error, text)))


def _get(results, process):
    '''Get the next message, raising if the child died without sending one.'''
    while True:
        alive = process.is_alive()
        try: return results.get(timeout=poll_seconds)
        except queue.Empty:
            if alive: continue
            m = f'child process exited with code {process.exitcode}'
            raise RuntimeError(m)
