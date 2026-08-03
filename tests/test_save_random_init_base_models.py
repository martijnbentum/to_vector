import unittest

from to_vector import scripts
from to_vector.scripts import save_random_init_base_models


class SaveRandomInitBaseModelsTests(unittest.TestCase):
    def test_scripts_package_exports_save_functions(self):
        self.assertIs(scripts.save_random_wav2vec2_base,
            save_random_init_base_models.save_random_wav2vec2_base)
        self.assertIs(scripts.save_random_hubert_base,
            save_random_init_base_models.save_random_hubert_base)


if __name__ == '__main__':
    unittest.main()
