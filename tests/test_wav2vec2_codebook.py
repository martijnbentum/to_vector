import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from to_vector import wav2vec2_codebook


class FakeQuantizer:
    def __init__(self, codevectors):
        self.codevectors = torch.tensor(np.asarray([codevectors]),
            dtype=torch.float32)

    def __call__(self, cnn_output):
        codevectors = cnn_output.repeat(1, 1, 2)
        return codevectors, None


class FakePretrainingModel:
    def __init__(self, codevectors):
        self.quantizer = FakeQuantizer(codevectors)


class FakeBatchPretrainingModel(nn.Module):
    def __init__(self, codevectors):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(1))
        self.quantizer = FakeQuantizer(codevectors)
        self.wav2vec2 = SimpleNamespace(feature_extractor=mock.Mock(
            side_effect=self._extract_features))

    def _extract_features(self, input_values):
        return input_values.unsqueeze(1)

    def _get_feat_extract_output_lengths(self, input_lengths):
        return input_lengths


class Wav2Vec2CodebookTests(unittest.TestCase):
    def test_get_row_index_of_vector_in_matrix_returns_match(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

        index = wav2vec2_codebook.get_row_index_of_vector_in_matrix(
            np.array([3.0, 4.0]), matrix)

        self.assertEqual(index, 1)

    def test_get_row_index_of_vector_in_matrix_raises_when_missing(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

        with self.assertRaisesRegex(
            ValueError, 'vector was not found in the codebook'):
            wav2vec2_codebook.get_row_index_of_vector_in_matrix(
                np.array([5.0, 6.0]), matrix)

    def test_codebook_indices_to_codevector_concatenates_both_halves(self):
        codebook = np.array([[1.0, 2.0], [3.0, 4.0]])

        codevector = wav2vec2_codebook.codebook_indices_to_codevector(
            (1, 0), codebook)

        np.testing.assert_array_equal(codevector, np.array([3.0, 4.0, 1.0,
            2.0]))

    def test_cnn_output_to_codevectors_accepts_2d_input(self):
        model_pt = FakePretrainingModel(np.array([[1.0, 2.0], [3.0, 4.0]]))
        cnn_output = np.array([[1.0, 2.0], [3.0, 4.0]])

        codevectors = wav2vec2_codebook.cnn_output_to_codevectors(cnn_output,
            model_pt)

        self.assertEqual(tuple(codevectors.shape), (1, 2, 4))

    def test_outputs_to_codebook_indices_maps_extract_features(self):
        codebook = np.array([[1.0, 2.0], [3.0, 4.0]])
        model_pt = FakePretrainingModel(codebook)
        outputs = BaseModelOutput(hidden_states=None)
        outputs.extract_features = np.array([[[1.0, 2.0], [3.0, 4.0]]])

        indices = wav2vec2_codebook.outputs_to_codebook_indices(outputs,
            model_pt)

        self.assertEqual(indices, [(0, 0), (1, 1)])

    @mock.patch('to_vector.wav2vec2_codebook.load.prepare_feature_extractor')
    @mock.patch('to_vector.wav2vec2_codebook.audio.load_audio')
    def test_filename_batch_to_codevectors_batches_and_trims_outputs(
        self, mock_load_audio, mock_prepare_feature_extractor
    ):
        model_pt = FakeBatchPretrainingModel(np.array([
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]))
        mock_load_audio.side_effect = [
            np.array([1.0, 2.0]),
            np.array([3.0, 0.0, 0.0]),
        ]
        mock_prepare_feature_extractor.return_value = self._feature_extractor

        result = wav2vec2_codebook.filename_batch_to_codevectors([
            'first.wav',
            'second.wav',
        ], ends=[2.0, 3.0], model_pt=model_pt)

        self.assertEqual(len(result), 2)
        self.assertEqual(mock_prepare_feature_extractor.call_count, 1)
        self.assertEqual(model_pt.wav2vec2.feature_extractor.call_count, 1)
        self.assertEqual(tuple(result[0].shape), (2, 2))
        self.assertEqual(tuple(result[1].shape), (3, 2))
        np.testing.assert_array_equal(result[0], np.array([
            [1.0, 1.0],
            [2.0, 2.0],
        ]))
        np.testing.assert_array_equal(result[1], np.array([
            [3.0, 3.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]))

    @mock.patch('to_vector.wav2vec2_codebook.load.prepare_feature_extractor')
    @mock.patch('to_vector.wav2vec2_codebook.audio.load_audio')
    def test_iter_filename_batch_to_codebook_indices_preserves_order(
        self, mock_load_audio, mock_prepare_feature_extractor
    ):
        model_pt = FakeBatchPretrainingModel(np.array([
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]))
        mock_load_audio.side_effect = [
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
        ]
        mock_prepare_feature_extractor.return_value = self._feature_extractor

        result = list(
            wav2vec2_codebook.iter_filename_batch_to_codebook_indices([
                'one.wav',
                'two.wav',
                'three.wav',
            ], model_pt=model_pt, batch_size=2))

        self.assertEqual(len(result), 3)
        self.assertEqual(mock_prepare_feature_extractor.call_count, 2)
        self.assertEqual(model_pt.wav2vec2.feature_extractor.call_count, 2)
        self.assertEqual(result[0], [(1, 1)])
        self.assertEqual(result[1], [(2, 2)])
        self.assertEqual(result[2], [(3, 3)])

    def test_filename_batch_to_codevectors_validates_metadata_lengths(self):
        with self.assertRaisesRegex(
            ValueError, 'filenames, starts, and ends must have the same length'
        ):
            wav2vec2_codebook.filename_batch_to_codevectors(
                ['a.wav', 'b.wav'], starts=[0.0],
                model_pt=FakeBatchPretrainingModel(np.array([[0.0]])))

    @mock.patch('to_vector.wav2vec2_codebook.torch.cuda.empty_cache')
    @mock.patch('to_vector.wav2vec2_codebook._audio_batch_to_codevectors')
    @mock.patch('to_vector.wav2vec2_codebook.audio.load_audio')
    @mock.patch('to_vector.wav2vec2_codebook.load.model_is_on_gpu',
        return_value=True)
    def test_iter_filename_batch_to_codevectors_empties_cache_after_gpu_batch(
        self, mock_model_is_on_gpu, mock_load_audio,
        mock_audio_batch_to_codevectors, mock_empty_cache
    ):
        model_pt = FakeBatchPretrainingModel(np.array([[0.0]]))
        mock_load_audio.side_effect = [
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
        ]
        mock_audio_batch_to_codevectors.side_effect = [
            [np.array([[1.0, 1.0]])],
            [np.array([[2.0, 2.0]])],
            [np.array([[3.0, 3.0]])],
        ]

        result = list(wav2vec2_codebook.iter_filename_batch_to_codevectors([
            'one.wav',
            'two.wav',
            'three.wav',
        ], model_pt=model_pt, batch_size=1))

        self.assertEqual(len(result), 3)
        self.assertEqual(mock_empty_cache.call_count, 3)
        mock_model_is_on_gpu.assert_called_once_with(model_pt)

    def _feature_extractor(self, arrays, sampling_rate, return_tensors,
        padding):
        max_length = max(len(array) for array in arrays)
        input_values = []
        attention_mask = []
        for array in arrays:
            padded = np.pad(array, (0, max_length - len(array)))
            mask = np.concatenate([
                np.ones(len(array), dtype=int),
                np.zeros(max_length - len(array), dtype=int),
            ])
            input_values.append(padded)
            attention_mask.append(mask)
        return {
            'input_values': torch.tensor(np.asarray(input_values),
                dtype=torch.float32),
            'attention_mask': torch.tensor(np.asarray(attention_mask)),
        }


if __name__ == '__main__':
    unittest.main()
