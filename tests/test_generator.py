import os
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.generator import DailyPhonkGenerator, _load_config, StyleConfig

class TestDailyPhonkGenerator(unittest.TestCase):
    def setUp(self):
        # Create a dummy config for testing
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.test_dir.name, "test_config.yaml")
        
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write(f"""
backend: "dsp"
duration_seconds: 2
sample_rate: 44100
bit_depth: 16
neural:
  model_name: "facebook/musicgen-medium"
  device: "cpu"
styles:
  - id: "phonk"
    probability: 1.0
    prompt: "test prompt"
output:
  root_dir: "{self.test_dir.name}/outputs"
  filename_prefix: "test_output"
""")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_load_config(self):
        config = _load_config(self.config_path)
        self.assertEqual(config.backend, "dsp")
        self.assertEqual(config.duration_seconds, 2)
        self.assertEqual(len(config.styles), 1)

    def test_dsp_generation(self):
        # This acts as a smoke test for the DSP generation path
        gen = DailyPhonkGenerator(config_path=self.config_path)
        out_dict = gen.generate_once(seed=42)
        out_path = out_dict["path"]
        
        self.assertTrue(os.path.exists(out_path))
        self.assertTrue(out_path.endswith(".wav"))
        
        # Test stats logging
        from src.stats import log_generation
        log_generation({"stats": {"db_path": os.path.join(self.test_dir.name, "test_stats.db")}}, out_dict)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir.name, "test_stats.db")))
        
        # Test metadata generation
        from src.metadata import generate_publishing_assets
        generate_publishing_assets({"publishing": {"artist_name": "Test Artist"}}, out_dict)
        base_name = os.path.splitext(out_path)[0]
        self.assertTrue(os.path.exists(f"{base_name}_meta.json"))
        self.assertTrue(os.path.exists(f"{base_name}_promo.txt"))

    def test_style_fallback_when_probs_zero(self):
        # Test edge case where probabilities are 0
        from src.generator import _choose_style
        styles = [
            StyleConfig(id="1", probability=0.0, prompt=""),
            StyleConfig(id="2", probability=0.0, prompt=""),
        ]
        chosen = _choose_style(styles)
        self.assertIn(chosen.id, ["1", "2"])

if __name__ == "__main__":
    unittest.main()
