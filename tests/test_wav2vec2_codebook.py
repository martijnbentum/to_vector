import unittest

import numpy as np
import torch
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


if __name__ == '__main__':
    unittest.main()
