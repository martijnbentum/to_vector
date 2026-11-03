import os
from pathlib import Path
import pickle
import time
import unittest
from unittest import mock

import numpy as np
from transformers.modeling_outputs import BaseModelOutput

import to_vector
from to_vector import child_process

from tests.test_helpers import DummyModel


def count_to(n):
    for i in range(n): yield i


def add(a, b):
    return a + b


def fail():
    raise ValueError('boom')


def die():
    os._exit(3)


def endless():
    while True:
        yield 1
        time.sleep(0.01)


class ShouldSpawnTests(unittest.TestCase):
    def test_automatic_spawns_only_on_gpu_with_model_reference(self):
        self.assertTrue(child_process.should_spawn('repo/model', True, None,
            auto=True))
        self.assertTrue(child_process.should_spawn(None, True, None,
            auto=True))
        self.assertTrue(child_process.should_spawn(Path('model'), True, None,
            auto=True))
        self.assertFalse(child_process.should_spawn('repo/model', False, None,
            auto=True))
        self.assertFalse(child_process.should_spawn(DummyModel(), True, None,
            auto=True))
        self.assertFalse(child_process.should_spawn('repo/model', True, None))

    def test_explicit_spawn_overrides_automatic(self):
        self.assertTrue(child_process.should_spawn('repo/model', False, True))
        self.assertFalse(child_process.should_spawn('repo/model', True, False,
            auto=True))

    def test_explicit_spawn_rejects_loaded_model(self):
        with self.assertRaises(ValueError):
            child_process.should_spawn(DummyModel(), True, True)


class ChildProcessTests(unittest.TestCase):
    def test_iter_in_child_streams_generator_items_in_order(self):
        items = list(child_process.iter_in_child(count_to, 20))
        self.assertEqual(items, list(range(20)))

    def test_run_in_child_returns_value(self):
        self.assertEqual(child_process.run_in_child(add, 2, b=3), 5)

    def test_child_exception_is_raised_with_traceback(self):
        with self.assertRaisesRegex(RuntimeError, 'ValueError: boom'):
            child_process.run_in_child(fail)

    def test_child_that_dies_raises_instead_of_hanging(self):
        with self.assertRaisesRegex(RuntimeError, 'exited with code 3'):
            child_process.run_in_child(die)

    def test_closing_generator_stops_child(self):
        items = child_process.iter_in_child(endless)
        self.assertEqual(next(items), 1)
        items.close()

    def test_numpified_output_survives_pickling(self):
        outputs = BaseModelOutput(hidden_states=[np.ones((1, 2, 3))])
        outputs.extract_features = np.zeros((1, 2, 4))
        outputs.model_type = 'wav2vec2'
        restored = pickle.loads(pickle.dumps(outputs))
        np.testing.assert_array_equal(restored.hidden_states[0],
            outputs.hidden_states[0])
        np.testing.assert_array_equal(restored.extract_features,
            outputs.extract_features)
        self.assertEqual(restored.model_type, 'wav2vec2')


class SpawnRoutingTests(unittest.TestCase):
    @mock.patch('to_vector.to_embeddings.child_process.iter_in_child')
    def test_batch_with_model_name_on_gpu_runs_in_child(self, iter_in_child):
        iter_in_child.return_value = iter(['a', 'b'])
        outputs = to_vector.filename_batch_to_vector(['x.wav', 'y.wav'],
            ends=[1.0, 1.0], model='repo/model', gpu=True)
        self.assertEqual(outputs, ['a', 'b'])
        args, kwargs = iter_in_child.call_args
        self.assertIs(args[0], to_vector.iter_filename_batch_to_vector)
        self.assertEqual(kwargs, {'spawn': False})

    @mock.patch('to_vector.to_embeddings.batch_helper.iter_handle_batching')
    @mock.patch('to_vector.to_embeddings.child_process.iter_in_child')
    def test_batch_with_loaded_model_runs_in_process(self, iter_in_child,
        iter_handle_batching):
        iter_handle_batching.return_value = iter(['a'])
        outputs = to_vector.filename_batch_to_vector(['x.wav'], ends=[1.0],
            model=DummyModel(), gpu=True)
        self.assertEqual(outputs, ['a'])
        iter_in_child.assert_not_called()

    @mock.patch('to_vector.to_embeddings.audio_to_vector')
    @mock.patch('to_vector.to_embeddings.audio.load_audio')
    @mock.patch('to_vector.to_embeddings.child_process.run_in_child')
    def test_single_file_spawns_only_when_requested(self, run_in_child,
        load_audio, audio_to_vector):
        to_vector.filename_to_vector('x.wav', model='repo/model', gpu=True)
        run_in_child.assert_not_called()
        to_vector.filename_to_vector('x.wav', model='repo/model', gpu=True,
            spawn=True)
        args, kwargs = run_in_child.call_args
        self.assertIs(args[0], to_vector.filename_to_vector)
        self.assertEqual(kwargs, {'spawn': False})


if __name__ == '__main__':
    unittest.main()
