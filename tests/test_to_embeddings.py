import unittest
from unittest import mock

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

import to_vector

from tests.test_helpers import FakeHuggingFaceModel
from tests.test_helpers import FakeSpidrModel


class ToEmbeddingsTests(unittest.TestCase):
    def test_audio_to_cnn_raises_clear_error_for_spidr(self):
        with mock.patch('to_vector.to_embeddings.load.prepare_model',
            return_value=FakeSpidrModel()):
            with mock.patch('to_vector.to_embeddings.model_registry.model_to_type',
                return_value='spidr'):
                with self.assertRaisesRegex(
                    ValueError, 'audio_to_cnn\\(\\) is not implemented for SpidR'):
                    to_vector.audio_to_cnn(np.array([1.0, 2.0, 3.0]),
                        model='checkpoint.pt')

    @mock.patch('to_vector.to_embeddings.load.prepare_feature_extractor')
    @mock.patch('to_vector.to_embeddings.model_registry.model_to_type',
        return_value='wav2vec2')
    @mock.patch('to_vector.to_embeddings.load.prepare_model')
    def test_audio_to_vector_removes_huggingface_last_hidden_state(
        self, mock_prepare_model, mock_get_model_type,
        mock_prepare_feature_extractor
    ):
        outputs = BaseModelOutput(
            last_hidden_state=torch.tensor([[[1.0]]]),
            hidden_states=(torch.tensor([[[2.0]]]),))
        model = FakeHuggingFaceModel(outputs)
        feature_extractor = mock.Mock(
            return_value={'input_values': torch.tensor([[1.0]])})
        mock_prepare_model.return_value = model
        mock_prepare_feature_extractor.return_value = feature_extractor

        result = to_vector.audio_to_vector(np.array([1.0, 2.0, 3.0]),
            model='repo/model', numpify_output=False)

        self.assertIsNone(result.last_hidden_state)
        self.assertEqual(result.model_type, 'wav2vec2')

    @mock.patch('to_vector.to_embeddings.load.prepare_feature_extractor')
    @mock.patch('to_vector.to_embeddings.audio.standardize_audio')
    @mock.patch('to_vector.to_embeddings.model_registry.model_to_type',
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
        self.assertEqual(outputs.model_type, 'spidr')


if __name__ == '__main__':
    unittest.main()
