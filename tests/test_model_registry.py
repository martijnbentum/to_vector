import json
import tempfile
from pathlib import Path
import unittest

from to_vector import model_registry

from tests.test_helpers import DeviceModel
from tests.test_helpers import FakeSpidrAttentionModel


class ModelRegistryTests(unittest.TestCase):
    def test_model_to_type_recognizes_spidr_models(self):
        model = FakeSpidrAttentionModel()

        model_type = model_registry.model_to_type(model)

        self.assertEqual(model_type, 'spidr')

    def test_model_to_type_returns_unknown_for_unsupported_model(self):
        model = DeviceModel()

        model_type = model_registry.model_to_type(model)

        self.assertEqual(model_type, 'unknown')

    def test_filename_model_type_reads_local_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_filename = Path(tmpdir) / 'config.json'
            config_filename.write_text(json.dumps({'model_type': 'spidr'}))

            model_type = model_registry.filename_model_type(tmpdir)

        self.assertEqual(model_type, 'spidr')

    def test_filename_model_type_reads_nested_run_model_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_filename = Path(tmpdir) / 'config.json'
            config_filename.write_text(
                json.dumps({'run': {'model_type': 'spidr'}}))

            model_type = model_registry.filename_model_type(tmpdir)

        self.assertEqual(model_type, 'spidr')

    def test_infer_config_filename_handles_checkpoint_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / 'model.pt'
            checkpoint.write_bytes(b'checkpoint')

            config_filename = model_registry.infer_config_filename(checkpoint)

        self.assertEqual(config_filename, checkpoint.parent / 'config.json')

    def test_load_config_dict_returns_none_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / 'missing.json'

            config_dict = model_registry.load_config_dict(missing)

        self.assertIsNone(config_dict)


if __name__ == '__main__':
    unittest.main()
