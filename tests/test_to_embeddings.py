import unittest
from unittest import mock

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

import to_vector

from tests.test_helpers import FakeHuggingFaceModel
from tests.test_helpers import FakeSpidrModel


class ToEmbeddingsTests(unittest.TestCase):
    @mock.patch('to_vector.to_embeddings.audio_batch_to_vector')
    @mock.patch('to_vector.to_embeddings.audio.load_audio_batch')
    def test_filename_batch_to_vector_composes_batch_helpers(
        self, mock_load_audio_batch, mock_audio_batch_to_vector
    ):
        first = BaseModelOutput(hidden_states=[np.array([[[1.0]]])])
        second = BaseModelOutput(hidden_states=[np.array([[[2.0]]])])
        mock_load_audio_batch.return_value = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]
        mock_audio_batch_to_vector.return_value = [first, second]

        outputs = to_vector.filename_batch_to_vector(
            ['a.wav', 'b.wav'],
            starts=[0.0, 1.5],
            ends=[0.5, None],
            model='repo/model',
            gpu=True,
            identifiers=['id-a', 'id-b'],
            names=['name-a', 'name-b'],
        )

        self.assertEqual(len(outputs), 2)
        mock_load_audio_batch.assert_called_once()
        filenames, starts, ends = mock_load_audio_batch.call_args[0]
        self.assertTrue(str(filenames[0]).endswith('/a.wav'))
        self.assertTrue(str(filenames[1]).endswith('/b.wav'))
        self.assertEqual(starts, [0.0, 1.5])
        self.assertEqual(ends, [0.5, None])
        mock_audio_batch_to_vector.assert_called_once_with(
            mock_load_audio_batch.return_value, 'repo/model', True, True)
        self.assertTrue(outputs[0].audio_filename.endswith('/a.wav'))
        self.assertEqual(outputs[0].start_time, 0.0)
        self.assertEqual(outputs[0].end_time, 0.5)
        self.assertEqual(outputs[0].identifier, 'id-a')
        self.assertEqual(outputs[0].name, 'name-a')
        self.assertTrue(outputs[1].audio_filename.endswith('/b.wav'))
        self.assertEqual(outputs[1].start_time, 1.5)
        self.assertIsNone(outputs[1].end_time)
        self.assertEqual(outputs[1].identifier, 'id-b')
        self.assertEqual(outputs[1].name, 'name-b')

    def test_filename_batch_to_vector_rejects_mismatched_metadata_lengths(self):
        with self.assertRaisesRegex(
            ValueError, 'identifiers must have the same length as audio_filenames'
        ):
            to_vector.filename_batch_to_vector(
                ['a.wav', 'b.wav'],
                identifiers=['id-a'],
            )

    @mock.patch('to_vector.to_embeddings.load.prepare_feature_extractor')
    @mock.patch('to_vector.to_embeddings.model_registry.model_to_type',
        return_value='wav2vec2')
    @mock.patch('to_vector.to_embeddings.load.prepare_model')
    def test_audio_batch_to_vector_splits_huggingface_batch_outputs(
        self, mock_prepare_model, mock_get_model_type,
        mock_prepare_feature_extractor
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
        mock_prepare_model.return_value = model
        mock_prepare_feature_extractor.return_value = feature_extractor

        result = to_vector.audio_batch_to_vector([
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0, 5.0]),
        ], model='repo/model', numpify_output=False)

        self.assertEqual(len(result), 2)
        feature_extractor.assert_called_once()
        self.assertIsNone(result[0].last_hidden_state)
        self.assertEqual(result[0].model_type, 'wav2vec2')
        self.assertEqual(tuple(result[0].hidden_states[0].shape), (1, 2, 1))
        self.assertEqual(tuple(result[1].hidden_states[0].shape), (1, 3, 1))
        self.assertEqual(tuple(result[0].extract_features.shape), (1, 2, 1))
        self.assertEqual(tuple(result[1].extract_features.shape), (1, 3, 1))

    @mock.patch('to_vector.to_embeddings.audio.standardize_audio')
    @mock.patch('to_vector.to_embeddings.model_registry.model_to_type',
        return_value='spidr')
    @mock.patch('to_vector.to_embeddings.load.prepare_model')
    def test_audio_batch_to_vector_batches_spidr_outputs(
        self, mock_prepare_model, mock_get_model_type, mock_standardize_audio
    ):
        model = FakeSpidrModel()
        mock_prepare_model.return_value = model
        mock_standardize_audio.side_effect = lambda value: value

        result = to_vector.audio_batch_to_vector([
            np.array([1.0, 2.0]),
            np.array([4.0, 5.0, 6.0]),
        ], model='checkpoint.pt')

        mock_prepare_model.assert_called_once_with('checkpoint.pt', False)
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
