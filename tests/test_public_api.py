import json
from pathlib import Path
import unittest
from unittest import mock
from types import SimpleNamespace
import tempfile

import numpy as np
import torch

import to_vector
from to_vector import load, wav2vec2_codebook


class DummyModel:
    def __init__(self, name_or_path='repo/model'):
        self.name_or_path = name_or_path
        self.base_model_prefix = 'dummy'
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        return self

    def parameters(self):
        return iter([np.zeros(1)])


class DummyParameter:
    def __init__(self, device_type='cpu'):
        self.device = mock.Mock(type=device_type)


class DeviceModel(DummyModel):
    def __init__(self, device_type='cpu'):
        super().__init__()
        self._parameter = DummyParameter(device_type)

    def parameters(self):
        return iter([self._parameter])


class FakeSpidrModule:
    def __init__(self, outputs):
        self.outputs = outputs

    def get_intermediate_outputs(self, features):
        return self.outputs


class FakeSpidrModel(DeviceModel):
    def __init__(self):
        super().__init__(device_type='cpu')
        self.base_model_prefix = 'spidr'
        self.feature_extractor = mock.Mock(
            return_value=torch.tensor([[[1.0, 2.0]]]))
        self.feature_projection = mock.Mock(
            side_effect=lambda value: value + 1.0)
        self.student = FakeSpidrModule([
            torch.tensor([[[3.0, 4.0]]]),
            torch.tensor([[[5.0, 6.0]]]),
        ])
        self.teacher = FakeSpidrModule([
            torch.tensor([[[7.0, 8.0]]]),
        ])
        self.get_codebooks = mock.Mock(
            return_value=[torch.tensor([[[9.0, 10.0]]])])


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
            'sample.wav', sr=16000, offset=0.0, duration=0.0
        )

    def test_load_audio_rejects_end_before_start(self):
        with self.assertRaisesRegex(ValueError, 'end must be greater than or equal to start'):
            load.load_audio('sample.wav', start=2.0, end=1.0)

    def test_standardize_audio_returns_zero_mean_unit_variance(self):
        audio = to_vector.standardize_audio(np.array([1.0, 2.0, 3.0]))

        self.assertAlmostEqual(float(audio.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(audio.std()), 1.0, places=6)


class CodebookTests(unittest.TestCase):
    def test_get_row_index_of_vector_in_matrix_uses_tolerant_match(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        vector = np.array([3.0 + 1e-7, 4.0 - 1e-7])

        index = wav2vec2_codebook.get_row_index_of_vector_in_matrix(
            vector, matrix)

        self.assertEqual(index, 1)

    def test_get_row_index_of_vector_in_matrix_raises_clear_error(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

        with self.assertRaisesRegex(ValueError, 'vector was not found in the codebook'):
            wav2vec2_codebook.get_row_index_of_vector_in_matrix(
                np.array([9.0, 9.0]), matrix)


class EntryPointTests(unittest.TestCase):
    @mock.patch('to_vector.to_embeddings.load.prepare_feature_extractor')
    @mock.patch('to_vector.to_embeddings.audio.standardize_audio')
    @mock.patch('to_vector.to_embeddings.load.get_model_type',
        return_value='spidr')
    @mock.patch('to_vector.to_embeddings.load.prepare_model')
    def test_audio_to_vector_routes_spidr_without_feature_extractor(
        self, mock_prepare_model, mock_get_model_type,
        mock_standardize_audio, mock_prepare_feature_extractor
    ):
        model = FakeSpidrModel()
        mock_prepare_model.return_value = model
        mock_standardize_audio.side_effect = lambda value: value

        outputs = to_vector.audio_to_vector(np.array([1.0, 2.0, 3.0]),
            model='checkpoint.pt')

        mock_prepare_model.assert_called_once_with('checkpoint.pt', False)
        mock_get_model_type.assert_called_once_with(model)
        mock_standardize_audio.assert_called_once()
        mock_prepare_feature_extractor.assert_not_called()
        self.assertEqual(len(outputs.hidden_states), 2)
        self.assertEqual(outputs.hidden_states[-1].shape, (1, 1, 2))
        self.assertEqual(outputs.extract_features.shape, (1, 1, 2))
        self.assertEqual(len(outputs.teacher_hidden_states), 1)
        self.assertEqual(len(outputs.codebook_predictions), 1)

    def test_public_api_exports_main_helpers(self):
        for name in [
            'audio_to_vector',
            'filename_to_vector',
            'audio_to_attention',
            'filename_to_attention',
            'audio_to_codebook_indices',
            'filename_to_codebook_indices',
            'load_audio',
            'load_feature_extractor',
            'load_model',
            'load_model_pt',
            'load_spidr_model',
            'standardize_audio',
        ]:
            self.assertTrue(hasattr(to_vector, name), name)

    def test_public_api_excludes_helper_functions(self):
        for name in [
            'outputs_to_attention',
            'outputs_to_codebook_indices',
            'select_attention',
            'stack_attentions',
            'move_model',
        ]:
            self.assertFalse(hasattr(to_vector, name), name)


if __name__ == '__main__':
    unittest.main()
