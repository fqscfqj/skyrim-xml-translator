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


if __name__ == "__main__":
    unittest.main()