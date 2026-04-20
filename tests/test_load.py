import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import to_vector
from to_vector import load

from tests.test_helpers import DeviceModel


class LoadTests(unittest.TestCase):
    @mock.patch('to_vector.load.AutoModel.from_pretrained')
    @mock.patch('to_vector.load.torch.cuda.is_available', return_value=False)
    def test_load_model_does_not_force_cuda_when_unavailable(
        self, mock_cuda, mock_from_pretrained
    ):
        model = DeviceModel(device_type='cpu')
        mock_from_pretrained.return_value = model

        loaded = load.load_model('repo/model', gpu=True)

        self.assertIs(loaded, model)
        self.assertEqual(model.moves, [])

    @mock.patch('to_vector.load.load_feature_extractor')
    def test_prepare_feature_extractor_uses_model_name_or_path(
        self, mock_load_feature_extractor
    ):
        feature_extractor = mock.Mock()
        mock_load_feature_extractor.return_value = feature_extractor
        model = DeviceModel()
        model.name_or_path = 'custom/repo'

        resolved_feature_extractor = load.prepare_feature_extractor(model)

        self.assertIs(resolved_feature_extractor, feature_extractor)
        mock_load_feature_extractor.assert_called_once_with('custom/repo')

    @mock.patch('to_vector.load.load_spidr_model')
    def test_load_model_routes_local_spidr_configs(self, mock_load_spidr_model):
        mock_load_spidr_model.return_value = DeviceModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / 'model.pt'
            checkpoint.write_bytes(b'checkpoint')
            config_filename = Path(tmpdir) / 'config.json'
            config_filename.write_text(json.dumps({'model_type': 'spidr'}))

            loaded = load.load_model(str(checkpoint))

        self.assertIs(loaded, mock_load_spidr_model.return_value)
        mock_load_spidr_model.assert_called_once_with(
            str(checkpoint),
            gpu=False,
            config_filename=None,
            strict=True,
        )

    def test_load_audio_allows_zero_length_slice(self):
        mock_librosa = SimpleNamespace(
            load=mock.Mock(return_value=(np.array([0.1]), 16000))
        )
        with mock.patch('to_vector.audio.librosa', mock_librosa):
            audio = load.load_audio('sample.wav', start=0.0, end=0.0)

        np.testing.assert_array_equal(audio, np.array([0.1]))
        mock_librosa.load.assert_called_once_with(
            'sample.wav', sr=0.0 + 16000, offset=0.0, duration=0.0
        )

    def test_load_audio_rejects_end_before_start(self):
        with self.assertRaisesRegex(
            ValueError, 'end must be greater than or equal to start'
        ):
            load.load_audio('sample.wav', start=2.0, end=1.0)

    @mock.patch('to_vector.audio.load_audio')
    def test_load_audio_batch_loads_each_segment(self, mock_load_audio):
        mock_load_audio.side_effect = [
            np.array([1.0, 2.0]),
            np.array([3.0]),
        ]

        result = to_vector.load_audio_batch(
            ['a.wav', 'b.wav'],
            starts=[0.0, 1.5],
            ends=[1.0, None],
        )

        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(result[0], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result[1], np.array([3.0]))
        mock_load_audio.assert_has_calls([
            mock.call('a.wav', 0.0, 1.0),
            mock.call('b.wav', 1.5, None),
        ])

    @mock.patch('to_vector.audio.load_audio')
    def test_load_audio_batch_defaults_times(self, mock_load_audio):
        mock_load_audio.side_effect = [np.array([1.0]), np.array([2.0])]

        to_vector.load_audio_batch(['a.wav', 'b.wav'])

        mock_load_audio.assert_has_calls([
            mock.call('a.wav', 0.0, None),
            mock.call('b.wav', 0.0, None),
        ])

    def test_load_audio_batch_rejects_empty_filenames(self):
        with self.assertRaisesRegex(
            ValueError, 'filenames must contain at least one filename'
        ):
            to_vector.load_audio_batch([])

    def test_load_audio_batch_rejects_mismatched_time_lengths(self):
        with self.assertRaisesRegex(
            ValueError, 'starts must have the same length as filenames'
        ):
            to_vector.load_audio_batch(['a.wav', 'b.wav'], starts=[0.0])

        with self.assertRaisesRegex(
            ValueError, 'ends must have the same length as filenames'
        ):
            to_vector.load_audio_batch(['a.wav', 'b.wav'], ends=[1.0])

    @mock.patch('to_vector.audio.load_audio_batch')
    def test_load_audio_batch_milliseconds_converts_and_delegates(
        self, mock_load_audio_batch
    ):
        mock_load_audio_batch.return_value = [np.array([1.0])]

        result = to_vector.load_audio_batch_milliseconds(
            ['a.wav'],
            starts=[250],
            ends=[1250],
        )

        self.assertEqual(len(result), 1)
        mock_load_audio_batch.assert_called_once_with(
            ['a.wav'],
            [0.25],
            [1.25],
        )

    @mock.patch('to_vector.audio.load_audio_batch')
    def test_load_audio_batch_milliseconds_defaults_times(
        self, mock_load_audio_batch
    ):
        mock_load_audio_batch.return_value = [np.array([1.0]), np.array([2.0])]

        to_vector.load_audio_batch_milliseconds(['a.wav', 'b.wav'])

        mock_load_audio_batch.assert_called_once_with(
            ['a.wav', 'b.wav'],
            [0.0, 0.0],
            [None, None],
        )

    def test_load_audio_batch_milliseconds_rejects_float_times(self):
        with self.assertRaisesRegex(
            TypeError, 'start and end must be integers representing milliseconds'
        ):
            to_vector.load_audio_batch_milliseconds(
                ['a.wav'],
                starts=[0.5],
            )

    def test_standardize_audio_returns_zero_mean_unit_variance(self):
        audio = to_vector.standardize_audio(np.array([1.0, 2.0, 3.0]))

        self.assertAlmostEqual(float(audio.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(audio.std()), 1.0, places=6)

    def test_prepare_feature_extractor_returns_none_for_spidr(self):
        model = mock.Mock()

        with mock.patch('to_vector.load.model_registry.model_to_type',
            return_value='spidr'):
            feature_extractor = load.prepare_feature_extractor(model)

        self.assertIsNone(feature_extractor)

    def test_spidr_config_to_kwargs_returns_nested_model_section(self):
        config_dict = {'model': {'n_layers': 12}, 'run': {'seed': 1}}

        kwargs = load.spidr_config_to_kwargs(config_dict)

        self.assertEqual(kwargs, {'n_layers': 12})


if __name__ == '__main__':
    unittest.main()
