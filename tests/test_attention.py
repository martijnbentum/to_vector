import unittest
from unittest import mock

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

import to_vector
from to_vector import _spidr_attention
from to_vector import attention

from tests.test_helpers import FakeHuggingFaceModel
from tests.test_helpers import FakeSpidrAttentionModel


class SpidrAttentionHelperTests(unittest.TestCase):
    def test_spidr_attention_helper_returns_layer_attention_tensors(self):
        model = FakeSpidrAttentionModel()

        outputs = _spidr_attention.audio_to_attention_outputs(
            np.array([1.0, 0.0, -1.0]), model)

        self.assertEqual(outputs.model_type, 'spidr')
        self.assertEqual(len(outputs.attentions), 1)
        self.assertEqual(tuple(outputs.attentions[0].shape), (1, 2, 3, 3))
        sums = outputs.attentions[0].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)))


class AttentionHelperTests(unittest.TestCase):
    def test_stack_attentions_squeezes_single_batch(self):
        attentions = (
            torch.ones(1, 2, 3, 3),
            torch.zeros(1, 2, 3, 3),
        )

        stacked = attention.stack_attentions(attentions)

        self.assertEqual(tuple(stacked.shape), (2, 2, 3, 3))

    def test_select_attention_can_choose_layer_and_average_heads(self):
        attention_tensor = torch.arange(2 * 3 * 2 * 2,
            dtype=torch.float32).view(2, 3, 2, 2)

        selected = attention.select_attention(attention_tensor, layer=1,
            average_heads=True)

        expected = attention_tensor[1].mean(dim=0)
        self.assertTrue(torch.equal(selected, expected))

    def test_outputs_to_attention_raises_without_attentions(self):
        outputs = BaseModelOutput(hidden_states=None)
        outputs.attentions = None

        with self.assertRaisesRegex(
            ValueError, 'model outputs do not contain attentions'):
            attention.outputs_to_attention(outputs)


class AttentionEntryPointTests(unittest.TestCase):
    @mock.patch('to_vector.attention.load.prepare_feature_extractor')
    @mock.patch('to_vector.attention.model_registry.model_to_type',
        return_value='wav2vec2')
    @mock.patch('to_vector.attention.load.prepare_model')
    def test_audio_to_attention_sets_huggingface_model_type(
        self, mock_prepare_model, mock_get_model_type,
        mock_prepare_feature_extractor
    ):
        outputs = BaseModelOutput(attentions=(torch.ones(1, 2, 3, 3),))
        model = FakeHuggingFaceModel(outputs)
        feature_extractor = mock.Mock(
            return_value={'input_values': torch.tensor([[1.0]])})
        mock_prepare_model.return_value = model
        mock_prepare_feature_extractor.return_value = feature_extractor

        result = to_vector.audio_to_attention(np.array([1.0, 2.0, 3.0]),
            model='repo/model', numpify_output=False)

        self.assertEqual(result.model_type, 'wav2vec2')

    @mock.patch('to_vector.attention.load.prepare_feature_extractor')
    @mock.patch('to_vector.attention.model_registry.model_to_type',
        return_value='spidr')
    @mock.patch('to_vector.attention.load.prepare_model')
    def test_audio_to_attention_routes_spidr_without_feature_extractor(
        self, mock_prepare_model, mock_get_model_type,
        mock_prepare_feature_extractor
    ):
        model = FakeSpidrAttentionModel()
        mock_prepare_model.return_value = model

        outputs = to_vector.audio_to_attention(np.array([1.0, 0.0, -1.0]),
            model='checkpoint.pt', numpify_output=False)

        mock_prepare_model.assert_called_once_with(
            'checkpoint.pt', False, for_attention_extraction=True)
        mock_get_model_type.assert_any_call(model)
        mock_prepare_feature_extractor.assert_not_called()
        self.assertEqual(outputs.model_type, 'spidr')
        self.assertEqual(tuple(outputs.attentions.shape), (1, 2, 3, 3))

    @mock.patch('to_vector.attention.audio_to_attention')
    @mock.patch('to_vector.attention.audio.load_audio')
    def test_filename_to_attention_adds_file_metadata(
        self, mock_load_audio, mock_audio_to_attention
    ):
        output = BaseModelOutput(hidden_states=None)
        output.attentions = np.zeros((2, 2))
        mock_load_audio.return_value = np.array([0.1, 0.2])
        mock_audio_to_attention.return_value = output

        result = attention.filename_to_attention('sample.wav', start=0.5,
            end=1.5)

        self.assertTrue(result.audio_filename.endswith('sample.wav'))
        self.assertEqual(result.start_time, 0.5)
        self.assertEqual(result.end_time, 1.5)


if __name__ == '__main__':
    unittest.main()
