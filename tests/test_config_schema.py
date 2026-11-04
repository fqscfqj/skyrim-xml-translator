import glob
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.config.manager import ConfigManager
from src.config.schema import AppConfig, dataclass_to_dict, validate_config


class TaskCompletionSoundConfigTests(unittest.TestCase):
    def test_llm_api_mode_defaults_to_chat_completions_and_is_in_example(self):
        config = dataclass_to_dict(AppConfig())
        repo_root = os.path.dirname(os.path.dirname(__file__))
        config_example_path = os.path.join(repo_root, "config.example.json")
        with open(config_example_path, "r", encoding="utf-8-sig") as handle:
            config_example = json.load(handle)

        for section in ("llm", "llm_search", "llm_search_fallback"):
            with self.subTest(section=section):
                self.assertEqual(config[section]["api_mode"], "chat_completions")
                self.assertEqual(
                    config_example[section]["api_mode"], "chat_completions"
                )

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

    def test_llm_examples_expose_provider_aware_reasoning_controls(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        config_example_path = os.path.join(repo_root, "config.example.json")
        with open(config_example_path, "r", encoding="utf-8-sig") as handle:
            config_example = json.load(handle)

        for section in ("llm", "llm_search", "llm_search_fallback"):
            with self.subTest(section=section):
                parameters = config_example[section]["parameters"]
                self.assertEqual(parameters["reasoning_protocol"], "auto")
                self.assertIn("enable_thinking", parameters)
                self.assertIn("reasoning_effort", parameters)
                self.assertNotIn("reasoning_budget_tokens", parameters)


class ConfigSaveAndCoercionTests(unittest.TestCase):
    def test_save_config_returns_true_on_success_and_false_on_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            manager = ConfigManager(config_path)
            manager.set("general", "language", "zh", save=False)

            self.assertTrue(manager.save_config())

            with patch(
                "src.config.manager.os.replace",
                side_effect=OSError("denied"),
            ):
                self.assertFalse(manager.save_config())

            self.assertEqual(glob.glob(config_path + ".tmp*"), [])

    def test_get_accepts_float_value_for_int_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump({"cache": {"cache_ttl_hours": 12.5}}, handle)

            manager = ConfigManager(config_path)

            self.assertEqual(
                manager.get("cache", "cache_ttl_hours", 0), 12.5
            )
            self.assertEqual(
                manager.get("cache", "cache_ttl_hours", 0.0), 12.5
            )

    def test_get_accepts_int_value_for_float_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump({"cache": {"cache_ttl_hours": 3}}, handle)

            manager = ConfigManager(config_path)

            value = manager.get("cache", "cache_ttl_hours", 0.0)
            self.assertEqual(value, 3.0)
            self.assertIsInstance(value, float)

    def test_missing_runtime_embedding_and_rag_keys_are_schema_backfilled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.json")
            manager = ConfigManager(config_path)
            typed = manager.get_typed()

            self.assertEqual(typed.embedding.max_retries, 2)
            self.assertEqual(typed.embedding.backoff_base, 0.5)
            self.assertEqual(typed.embedding.retry_total_timeout, 60.0)
            self.assertEqual(typed.rag.batch_prompt_max_chars, 12000)
            self.assertEqual(typed.rag.glossary_entry_max_chars, 240)
            self.assertEqual(
                typed.general.short_text_batch_circuit_timeout_seconds, 300
            )
            self.assertEqual(manager.validate(), [])
            self.assertTrue(manager.save_config())

            with open(config_path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertIn("max_retries", saved["embedding"])
            self.assertIn("retry_total_timeout", saved["embedding"])
            self.assertIn("batch_prompt_max_chars", saved["rag"])
            self.assertIn(
                "short_text_batch_circuit_timeout_seconds", saved["general"]
            )

    def test_new_runtime_keys_validate_within_range(self):
        config = dataclass_to_dict(AppConfig())
        self.assertEqual(validate_config(config), [])

        config["embedding"]["retry_total_timeout"] = -1
        self.assertTrue(any(
            "embedding.retry_total_timeout" in error
            for error in validate_config(config)
        ))


if __name__ == "__main__":
    unittest.main()
