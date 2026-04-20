import unittest
from unittest import mock

import numpy as np

from to_vector import codebook_artifacts


class CodebookArtifactsTests(unittest.TestCase):
    def test_wav2vec2_artifacts_normalize_to_frame_pairs(self):
        fake_model = object()
        with mock.patch.object(codebook_artifacts, '_prepare_wav2vec2_model',
            return_value=fake_model), mock.patch.object(
            codebook_artifacts.wav2vec2_codebook, 'audio_to_codebook_indices',
            return_value=[(0, 3), (1, 2)]), mock.patch.object(
            codebook_artifacts.wav2vec2_codebook, 'load_codebook',
            return_value=np.array([[1.0], [2.0], [3.0], [4.0]])):
            result = codebook_artifacts.audio_to_codebook_artifacts(
                np.zeros(10), model=None)

        self.assertEqual(result.model_architecture, 'wav2vec2')
        self.assertEqual(result.indices.shape, (2, 2))
        np.testing.assert_array_equal(result.indices, np.array([
            [0, 3],
            [1, 2],
        ]))

    def test_spidr_artifacts_normalize_to_frame_head_matrix(self):
        fake_model = object()
        with mock.patch.object(codebook_artifacts._spidr_util,
            'prepare_model', return_value=fake_model), mock.patch.object(
            codebook_artifacts.spidr_codebook, 'audio_to_codebook_indices',
            return_value=np.array([
                [1, 0],
                [0, 1],
            ])), mock.patch.object(
            codebook_artifacts.spidr_codebook, 'load_codebooks',
            return_value=[
                np.array([[1.0], [2.0]]),
                np.array([[3.0], [4.0]]),
            ]), mock.patch.object(
            codebook_artifacts.model_registry, 'filename_model_type',
            return_value='spidr'):
            result = codebook_artifacts.audio_to_codebook_artifacts(
                np.zeros(10), model='checkpoint.pt')

        self.assertEqual(result.model_architecture, 'spidr')
        self.assertEqual(result.indices.shape, (2, 2))
        np.testing.assert_array_equal(result.indices, np.array([
            [1, 0],
            [0, 1],
        ]))
        self.assertEqual(result.codebook_matrix.shape, (2, 2, 1))

    def test_unknown_instance_type_raises(self):
        with mock.patch.object(codebook_artifacts.model_registry,
            'model_to_type', return_value='unknown'):
            with self.assertRaisesRegex(
                ValueError, 'support only wav2vec2 and spidr'):
                codebook_artifacts.audio_to_codebook_artifacts(
                    np.zeros(10), model=object())


if __name__ == '__main__':
    unittest.main()
