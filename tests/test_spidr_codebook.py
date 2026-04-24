import unittest
from unittest import mock

import numpy as np
import torch

from to_vector import spidr_codebook

from tests.test_helpers import FakeSpidrModel


class SpidrCodebookTests(unittest.TestCase):
    def test_probabilities_to_codebook_indices_uses_argmax(self):
        probabilities = np.array([
            [[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]],
            [[0.9, 0.05, 0.05], [0.2, 0.3, 0.5]],
        ])

        indices = spidr_codebook.probabilities_to_codebook_indices(
            probabilities)

        np.testing.assert_array_equal(indices, np.array([
            [1, 0],
            [0, 2],
        ]))

    def test_normalize_probability_shape_restores_time_axis(self):
        probability = np.array([0.2, 0.3, 0.5])

        normalized = spidr_codebook.normalize_probability_shape(probability)

        self.assertEqual(normalized.shape, (1, 3))

    def test_normalize_batched_probability_shape_restores_batch_axis(self):
        probability = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])

        normalized = spidr_codebook.normalize_batched_probability_shape(
            probability, batch_size=2)

        self.assertEqual(normalized.shape, (2, 1, 3))

    def test_codebook_indices_to_codevectors_indexes_each_codebook(self):
        indices = np.array([
            [1, 0],
            [0, 1],
        ])
        codebooks = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ]

        codevectors = spidr_codebook.codebook_indices_to_codevectors(
            indices, codebooks)

        np.testing.assert_array_equal(codevectors, np.array([
            [[3.0, 4.0], [5.0, 6.0]],
            [[1.0, 2.0], [7.0, 8.0]],
        ]))

    @mock.patch('to_vector.spidr_codebook.spidr_batch_helper.prepare_waveform_batch')
    def test_audio_to_codebook_probabilities_normalizes_single_item_output(
        self, mock_prepare_waveform_batch
    ):
        model = FakeSpidrModel()
        mock_prepare_waveform_batch.return_value = (
            torch.zeros((1, 2)),
            [1],
            torch.ones((1, 1, 1, 1), dtype=torch.bool),
        )
        model.get_codebooks = mock.Mock(return_value=[
            None,
            torch.tensor([0.1, 0.9]),
            torch.tensor([0.8, 0.2]),
        ])

        probabilities = spidr_codebook.audio_to_codebook_probabilities(
            np.array([1.0, 2.0]), model=model)

        self.assertEqual(probabilities.shape, (1, 2, 2))
        np.testing.assert_allclose(probabilities, np.array([
            [[0.1, 0.9], [0.8, 0.2]],
        ]))

    @mock.patch('to_vector.spidr_codebook.spidr_batch_helper.prepare_waveform_batch')
    def test_audio_batch_to_codebook_indices_trims_padding_and_preserves_order(
        self, mock_prepare_waveform_batch
    ):
        model = FakeSpidrModel()
        attention_mask = torch.ones((2, 1, 2, 2), dtype=torch.bool)
        mock_prepare_waveform_batch.return_value = (
            torch.zeros((2, 3)),
            [1, 2],
            attention_mask,
        )
        model.get_codebooks = mock.Mock(return_value=[
            None,
            torch.tensor([
                [[0.1, 0.9], [0.8, 0.2]],
                [[0.7, 0.3], [0.4, 0.6]],
            ]),
            torch.tensor([
                [[0.9, 0.1], [0.2, 0.8]],
                [[0.3, 0.7], [0.6, 0.4]],
            ]),
        ])

        indices = spidr_codebook.audio_batch_to_codebook_indices([
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0, 5.0]),
        ], model=model)

        self.assertEqual(model.get_codebooks.call_count, 1)
        self.assertIs(model.get_codebooks.call_args.kwargs['attention_mask'],
            attention_mask)
        self.assertEqual(len(indices), 2)
        np.testing.assert_array_equal(indices[0], np.array([[1, 0]]))
        np.testing.assert_array_equal(indices[1], np.array([
            [0, 1],
            [1, 0],
        ]))

    @mock.patch('to_vector.spidr_codebook._single_batch_to_probabilities')
    def test_audio_batch_to_codebook_probabilities_splits_multiple_batches(
        self, mock_single_batch_to_probabilities
    ):
        model = FakeSpidrModel()
        mock_single_batch_to_probabilities.side_effect = [
            [np.array([[[0.1, 0.9]]]), np.array([[[0.8, 0.2]]])],
            [np.array([[[0.3, 0.7]]])],
        ]

        result = spidr_codebook.audio_batch_to_codebook_probabilities([
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
        ], model=model, batch_size=2)

        self.assertEqual(len(result), 3)
        self.assertEqual(mock_single_batch_to_probabilities.call_count, 2)
        first_batch = mock_single_batch_to_probabilities.call_args_list[0].args[0]
        second_batch = mock_single_batch_to_probabilities.call_args_list[1].args[0]
        self.assertEqual([len(item) for item in first_batch], [2, 2])
        self.assertEqual([len(item) for item in second_batch], [2])

    @mock.patch('to_vector.spidr_codebook.audio.load_audio_batch')
    def test_filename_batch_to_codebook_indices_validates_metadata_lengths(
        self, mock_load_audio_batch
    ):
        mock_load_audio_batch.return_value = [np.array([1.0])]

        with self.assertRaisesRegex(
            ValueError, 'starts must have the same length as audio_filenames'
        ):
            spidr_codebook.filename_batch_to_codebook_indices(
                ['a.wav', 'b.wav'], starts=[0.0], model=FakeSpidrModel())

    @mock.patch('to_vector.spidr_codebook.load_codebooks')
    @mock.patch('to_vector.spidr_codebook.audio_batch_to_codebook_indices')
    def test_audio_batch_to_codevectors_is_thin_wrapper(
        self, mock_audio_batch_to_codebook_indices, mock_load_codebooks
    ):
        model = FakeSpidrModel()
        mock_audio_batch_to_codebook_indices.return_value = [
            np.array([[1, 0], [0, 1]]),
        ]
        mock_load_codebooks.return_value = [
            np.array([[1.0], [2.0]]),
            np.array([[3.0], [4.0]]),
        ]

        codevectors = spidr_codebook.audio_batch_to_codevectors(
            [np.array([1.0, 2.0])], model=model)

        self.assertEqual(len(codevectors), 1)
        np.testing.assert_array_equal(codevectors[0], np.array([
            [[2.0], [3.0]],
            [[1.0], [4.0]],
        ]))


if __name__ == '__main__':
    unittest.main()
