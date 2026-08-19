import json
import os
import tempfile
import unittest

from src.config.manager import ConfigManager
from src.config.schema import AppConfig, dataclass_to_dict


class TaskCompletionSoundConfigTests(unittest.TestCase):
    def test_default_config_disables_task_completion_sound(self):
        config = dataclass_to_dict(AppConfig())

        self.assertIn("task_completion_sound_enabled", config["general"])
        self.assertFalse(config["general"]["task_completion_sound_enabled"])

    def test_config_manager_backfills_task_completion_sound_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump({"general": {"language": "zh"}}, handle)

            manager = ConfigManager(config_path)

            self.assertFalse(manager.get("general", "task_completion_sound_enabled", True))
            self.assertFalse(manager.get_typed().general.task_completion_sound_enabled)

            with open(config_path, "r", encoding="utf-8") as handle:
                saved_config = json.load(handle)

            self.assertIn("task_completion_sound_enabled", saved_config["general"])
            self.assertFalse(saved_config["general"]["task_completion_sound_enabled"])

    def test_config_example_includes_task_completion_sound_setting(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        config_example_path = os.path.join(repo_root, "config.example.json")

        with open(config_example_path, "r", encoding="utf-8-sig") as handle:
            config_example = json.load(handle)

        self.assertIn("task_completion_sound_enabled", config_example["general"])
        self.assertFalse(config_example["general"]["task_completion_sound_enabled"])

    def test_style_profile_defaults_to_auto_and_is_in_example(self):
        config = dataclass_to_dict(AppConfig())
        self.assertEqual("auto", config["general"]["style_profile"])

        repo_root = os.path.dirname(os.path.dirname(__file__))
        config_example_path = os.path.join(repo_root, "config.example.json")
        with open(config_example_path, "r", encoding="utf-8-sig") as handle:
            config_example = json.load(handle)

        self.assertEqual("auto", config_example["general"]["style_profile"])

    def test_long_text_chunk_target_default_does_not_exceed_threshold(self):
        config = dataclass_to_dict(AppConfig())

        self.assertLessEqual(
            config["general"]["long_text_chunk_target_chars"],
            config["general"]["long_text_chunk_threshold_chars"],
        )

    def test_config_example_long_text_chunk_target_does_not_exceed_threshold(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        config_example_path = os.path.join(repo_root, "config.example.json")

        with open(config_example_path, "r", encoding="utf-8-sig") as handle:
            config_example = json.load(handle)

        self.assertLessEqual(
            config_example["general"]["long_text_chunk_target_chars"],
            config_example["general"]["long_text_chunk_threshold_chars"],
        )

    def test_prompt_cache_warmup_is_enabled_by_default_and_in_example(self):
        config = dataclass_to_dict(AppConfig())
        self.assertTrue(config["general"]["prompt_cache_warmup_enabled"])

        repo_root = os.path.dirname(os.path.dirname(__file__))
        config_example_path = os.path.join(repo_root, "config.example.json")
        with open(config_example_path, "r", encoding="utf-8-sig") as handle:
            config_example = json.load(handle)

        self.assertTrue(config_example["general"]["prompt_cache_warmup_enabled"])

    def test_llm_examples_expose_deepseek_reasoning_effort(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        config_example_path = os.path.join(repo_root, "config.example.json")
        with open(config_example_path, "r", encoding="utf-8-sig") as handle:
            config_example = json.load(handle)

        for section in ("llm", "llm_search", "llm_search_fallback"):
            with self.subTest(section=section):
                self.assertIn("reasoning_effort", config_example[section]["parameters"])


if __name__ == "__main__":
    unittest.main()
