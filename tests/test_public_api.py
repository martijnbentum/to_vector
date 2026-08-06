import unittest
from unittest.mock import patch

import to_vector


class PublicApiTests(unittest.TestCase):
    def test_public_api_exports_main_helpers(self):
        for name in [
            'audio_to_vector',
            'filename_batch_to_cnn',
            'iter_filename_batch_to_cnn',
            'filename_batch_to_vector',
            'iter_filename_batch_to_vector',
            'filename_to_vector',
            'audio_to_attention',
            'audio_to_codebook_artifacts',
            'filename_to_attention',
            'audio_to_codebook_indices',
            'filename_to_codebook_indices',
            'filename_to_codebook_artifacts',
            'load_audio',
            'load_feature_extractor',
            'load_model',
            'load_model_pt',
            'load_spidr_model',
            'standardize_audio',
            'CodebookArtifacts',
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

    def test_iter_filename_batch_to_vector_yields_outputs(self):
        outputs = ['first', 'second']
        with patch('to_vector.to_embeddings.batch_helper.iter_handle_batching',
                return_value=iter(outputs)) as iter_handle_batching:
            result = list(to_vector.iter_filename_batch_to_vector(
                ['a.wav', 'b.wav'], starts=[0.0, 1.0], ends=[1.0, 2.0],
                model='stub', gpu=True, numpify_output=False, batch_size=2))

        self.assertEqual(result, outputs)
        iter_handle_batching.assert_called_once_with(
            ['a.wav', 'b.wav'], [0.0, 1.0], [1.0, 2.0], 'stub', True, False, 2)

    def test_iter_filename_batch_to_cnn_yields_outputs(self):
        outputs = ['first', 'second']
        with patch(
                'to_vector.to_embeddings.batch_helper.iter_handle_cnn_batching',
                return_value=iter(outputs)) as iter_handle_cnn_batching:
            result = list(to_vector.iter_filename_batch_to_cnn(
                ['a.wav', 'b.wav'], starts=[0.0, 1.0], ends=[1.0, 2.0],
                model='stub', gpu=True, batch_size=2))

        self.assertEqual(result, outputs)
        iter_handle_cnn_batching.assert_called_once_with(
            ['a.wav', 'b.wav'], [0.0, 1.0], [1.0, 2.0], 'stub', True, 2)


if __name__ == '__main__':
    unittest.main()
