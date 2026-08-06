import unittest
from unittest import mock

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

import to_vector
import to_vector.batch_helper as batch_helper
import to_vector.hf_batch_helper as hf_batch_helper
import to_vector.spidr_batch_helper as spidr_batch_helper

from tests.test_helpers import FakeHuggingFaceModel
from tests.test_helpers import FakeSpidrModel


class ToEmbeddingsTests(unittest.TestCase):
    @mock.patch('to_vector.hf_batch_helper.load.prepare_feature_extractor')
    def test_hf_batch_helper_splits_huggingface_batch_outputs(
        self, mock_prepare_feature_extractor
    ):
        outputs = BaseModelOutput(
            last_hidden_state=torch.tensor([
                [[1.0], [2.0], [3.0]],
                [[4.0], [5.0], [6.0]],
            ]),
            hidden_states=(
                torch.tensor([
                    [[10.0], [11.0], [12.0]],
                    [[20.0], [21.0], [22.0]],
                ]),
            ),
        )
        outputs.extract_features = torch.tensor([
            [[30.0], [31.0], [32.0]],
            [[40.0], [41.0], [42.0]],
        ])
        model = FakeHuggingFaceModel(outputs)
        model._get_feat_extract_output_lengths = mock.Mock(
            return_value=torch.tensor([2, 3]))
        feature_extractor = mock.Mock(return_value={
            'input_values': torch.tensor([
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 5.0],
            ]),
            'attention_mask': torch.tensor([
                [1, 1, 0],
                [1, 1, 1],
            ]),
        })
        mock_prepare_feature_extractor.return_value = feature_extractor

        result = hf_batch_helper.audio_batch_to_outputs([
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0, 5.0]),
        ], model, 'wav2vec2')

        self.assertEqual(len(result), 2)
        feature_extractor.assert_called_once()
        self.assertIsNone(result[0].last_hidden_state)
        self.assertEqual(result[0].model_type, 'wav2vec2')
        self.assertEqual(tuple(result[0].hidden_states[0].shape), (1, 2, 1))
        self.assertEqual(tuple(result[1].hidden_states[0].shape), (1, 3, 1))
        self.assertEqual(tuple(result[0].extract_features.shape), (1, 2, 1))
        self.assertEqual(tuple(result[1].extract_features.shape), (1, 3, 1))

    @mock.patch('to_vector.hf_batch_helper.load.prepare_feature_extractor')
    def test_hf_batch_helper_audio_batch_to_cnn_splits_batch_without_full_forward(
        self, mock_prepare_feature_extractor
    ):
        outputs = BaseModelOutput(hidden_states=None)
        model = FakeHuggingFaceModel(outputs)
        model.feature_extractor = mock.Mock(return_value=torch.tensor([
            [[10.0, 11.0, 12.0]],
            [[20.0, 21.0, 22.0]],
        ]))
        model._get_feat_extract_output_lengths = mock.Mock(
            return_value=torch.tensor([2, 3]))
        feature_extractor = mock.Mock(return_value={
            'input_values': torch.tensor([
                [1.0, 2.0, 0.0],
                [3.0, 4.0, 5.0],
            ]),
            'attention_mask': torch.tensor([
                [1, 1, 0],
                [1, 1, 1],
            ]),
        })
        mock_prepare_feature_extractor.return_value = feature_extractor

        with mock.patch.object(FakeHuggingFaceModel, '__call__') as mock_call:
            result = hf_batch_helper.audio_batch_to_cnn([
                np.array([1.0, 2.0]),
                np.array([3.0, 4.0, 5.0]),
            ], model, 'wav2vec2')
            mock_call.assert_not_called()

        self.assertEqual(len(result), 2)
        self.assertIsNone(result[0].hidden_states)
        self.assertIsNone(result[1].hidden_states)
        self.assertEqual(result[0].model_type, 'wav2vec2')
        self.assertEqual(result[1].model_type, 'wav2vec2')
        self.assertEqual(result[0].extract_features.shape, (1, 2, 1))
        self.assertEqual(result[1].extract_features.shape, (1, 3, 1))
        np.testing.assert_allclose(result[0].extract_features,
            np.array([[[10.0], [11.0]]]))
        np.testing.assert_allclose(result[1].extract_features,
            np.array([[[20.0], [21.0], [22.0]]]))

    def test_compute_cnn_output_lengths_defaults_to_extract_features_length(self):
        extract_features = torch.zeros((3, 5, 2))
        result = hf_batch_helper.compute_cnn_output_lengths({}, extract_features,
            None)
        self.assertEqual(result, [5, 5, 5])

    def test_single_batch_to_cnn_outputs_raises_clear_error_for_spidr(self):
        with self.assertRaisesRegex(
            ValueError, 'audio_batch_to_cnn\\(\\) is not implemented for SpidR'):
            batch_helper.single_batch_to_cnn_outputs([np.array([1.0])],
                FakeSpidrModel(), 'spidr')

    @mock.patch('to_vector.spidr_batch_helper.audio.standardize_audio')
    def test_spidr_batch_helper_batches_spidr_outputs(
        self, mock_standardize_audio
    ):
        model = FakeSpidrModel()
        mock_standardize_audio.side_effect = lambda value: value

        result = spidr_batch_helper.audio_batch_to_outputs([
            np.array([1.0, 2.0]),
            np.array([4.0, 5.0, 6.0]),
        ], model)

        self.assertEqual(len(model.student.calls), 1)
        attention_mask = model.student.calls[0]['attention_mask']
        self.assertEqual(tuple(attention_mask.shape), (2, 1, 2, 2))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].model_type, 'spidr')
        self.assertEqual(result[1].model_type, 'spidr')
        self.assertEqual(tuple(result[0].hidden_states[0].shape), (1, 1, 2))
        self.assertEqual(tuple(result[1].hidden_states[0].shape), (1, 2, 2))
        self.assertEqual(tuple(result[0].extract_features.shape), (1, 1, 2))
        self.assertEqual(tuple(result[1].extract_features.shape), (1, 2, 2))

    def test_audio_to_cnn_raises_clear_error_for_spidr(self):
        with mock.patch('to_vector.to_embeddings.load.prepare_model',
            return_value=FakeSpidrModel()):
            with mock.patch('to_vector.to_embeddings.model_registry.model_to_type',
                return_value='spidr'):
                with self.assertRaisesRegex(
                    ValueError, 'audio_to_cnn\\(\\) is not implemented for SpidR'):
                    to_vector.audio_to_cnn(np.array([1.0, 2.0, 3.0]),
                        model='checkpoint.pt')

    @mock.patch('to_vector.batch_helper.single_batch_to_outputs')
    @mock.patch('to_vector.batch_helper.load.load_audio')
    @mock.patch('to_vector.batch_helper.model_registry.model_to_type',
        return_value='wav2vec2')
    @mock.patch('to_vector.batch_helper.load.prepare_model')
    def test_iter_handle_batching_uses_prefetched_audio_arrays(
        self, mock_prepare_model, mock_get_model_type, mock_load_audio,
        mock_single_batch_to_outputs
    ):
        model = object()
        audio_arrays = [np.array([1.0]), np.array([2.0])]
        mock_prepare_model.return_value = model
        mock_load_audio.side_effect = audio_arrays
        mock_single_batch_to_outputs.return_value = ['output']

        result = list(batch_helper.iter_handle_batching(
            ['a.wav', 'b.wav'], model='stub', batch_size=2,
            numpify_output=False))

        self.assertEqual(result, ['output'])
        mock_single_batch_to_outputs.assert_called_once()
        args = mock_single_batch_to_outputs.call_args.args
        self.assertEqual(args[0], audio_arrays)
        self.assertIs(args[1], model)
        self.assertEqual(args[2], 'wav2vec2')

    @mock.patch('to_vector.batch_helper.load.load_audio',
        side_effect=RuntimeError('bad audio'))
    @mock.patch('to_vector.batch_helper.model_registry.model_to_type',
        return_value='wav2vec2')
    @mock.patch('to_vector.batch_helper.load.prepare_model')
    def test_iter_handle_batching_reraises_prefetch_errors(
        self, mock_prepare_model, mock_get_model_type, mock_load_audio
    ):
        mock_prepare_model.return_value = object()

        with self.assertRaisesRegex(RuntimeError, 'bad audio'):
            list(batch_helper.iter_handle_batching(['a.wav'], model='stub',
                batch_size=1, numpify_output=False))

    @mock.patch('to_vector.batch_helper.make_audio_queue')
    @mock.patch('to_vector.batch_helper.model_registry.model_to_type',
        return_value='wav2vec2')
    @mock.patch('to_vector.batch_helper.load.prepare_model')
    def test_iter_handle_batching_rejects_invalid_batch_size_before_thread(
        self, mock_prepare_model, mock_get_model_type, mock_make_audio_queue
    ):
        mock_prepare_model.return_value = object()

        with self.assertRaisesRegex(ValueError,
            'batch_size must be greater than zero'):
            list(batch_helper.iter_handle_batching(['a.wav'], model='stub',
                batch_size=0, numpify_output=False))

        mock_make_audio_queue.assert_not_called()

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
        self.assertEqual(outputs.hidden_states[-1].shape, (1, 3, 2))
        self.assertEqual(outputs.extract_features.shape, (1, 3, 2))
        self.assertEqual(outputs.model_type, 'spidr')


if __name__ == '__main__':
    unittest.main()
