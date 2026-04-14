import unittest

import to_vector


class PublicApiTests(unittest.TestCase):
    def test_public_api_exports_main_helpers(self):
        for name in [
            'audio_to_vector',
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


if __name__ == '__main__':
    unittest.main()
