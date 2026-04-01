import unittest

import numpy as np

from to_vector import spidr_codebook


class SpidrCodebookTests(unittest.TestCase):
    def test_probabilities_to_codebook_indices_uses_argmax(self):
        probabilities = [
            np.array([[0.1, 0.7, 0.2], [0.9, 0.05, 0.05]]),
            np.array([[0.2, 0.3, 0.5]]),
        ]

        indices = spidr_codebook.probabilities_to_codebook_indices(
            probabilities)

        np.testing.assert_array_equal(indices[0], np.array([1, 0]))
        np.testing.assert_array_equal(indices[1], np.array([2]))

    def test_normalize_probability_shape_restores_time_axis(self):
        probability = np.array([0.2, 0.3, 0.5])

        normalized = spidr_codebook.normalize_probability_shape(probability)

        self.assertEqual(normalized.shape, (1, 3))

    def test_codebook_indices_to_codevectors_indexes_each_codebook(self):
        indices = [np.array([1, 0]), np.array([0, 1])]
        codebooks = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ]

        codevectors = spidr_codebook.codebook_indices_to_codevectors(
            indices, codebooks)

        np.testing.assert_array_equal(
            codevectors[0], np.array([[3.0, 4.0], [1.0, 2.0]]))
        np.testing.assert_array_equal(
            codevectors[1], np.array([[5.0, 6.0], [7.0, 8.0]]))


if __name__ == '__main__':
    unittest.main()
