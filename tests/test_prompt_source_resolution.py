"""Prompt source resolution: external prompts next to the exe win over embedded."""

import os
import sys
import tempfile
import unittest
from unittest import mock

from src.prompt.prompt_manager import PromptManager


class FrozenPromptSourceTests(unittest.TestCase):
    def _resolve(self, exe_dir, prompts_layout, meipass_dir):
        prompts_dir = os.path.join(exe_dir, "prompts")
        if prompts_layout is not None:
            os.makedirs(prompts_dir, exist_ok=True)
            for name in prompts_layout:
                path = os.path.join(prompts_dir, name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("{}")
        exe_path = os.path.join(exe_dir, "SkyrimXMLTranslator.exe")
        with mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.object(sys, "executable", exe_path), \
                mock.patch.object(sys, "_MEIPASS", meipass_dir, create=True):
            return PromptManager._resolve_frozen_base_path()

    def test_external_prompts_with_json_win_over_embedded(self):
        with tempfile.TemporaryDirectory() as exe_dir, \
                tempfile.TemporaryDirectory() as meipass:
            resolved = self._resolve(
                exe_dir, ["translator/system_prompts/nsfw.json"], meipass)
            self.assertEqual(os.path.abspath(exe_dir), os.path.abspath(resolved))

    def test_empty_external_prompts_fall_back_to_embedded(self):
        with tempfile.TemporaryDirectory() as exe_dir, \
                tempfile.TemporaryDirectory() as meipass:
            resolved = self._resolve(exe_dir, [], meipass)
            self.assertEqual(os.path.abspath(meipass), os.path.abspath(resolved))

    def test_missing_external_prompts_fall_back_to_embedded(self):
        with tempfile.TemporaryDirectory() as exe_dir, \
                tempfile.TemporaryDirectory() as meipass:
            resolved = self._resolve(exe_dir, None, meipass)
            self.assertEqual(os.path.abspath(meipass), os.path.abspath(resolved))

    def test_missing_meipass_falls_back_to_module_root(self):
        with tempfile.TemporaryDirectory() as exe_dir:
            exe_path = os.path.join(exe_dir, "SkyrimXMLTranslator.exe")
            with mock.patch.object(sys, "frozen", True, create=True), \
                    mock.patch.object(sys, "executable", exe_path), \
                    mock.patch.object(sys, "_MEIPASS", "", create=True):
                resolved = PromptManager._resolve_frozen_base_path()
            self.assertNotEqual(os.path.abspath(exe_dir), os.path.abspath(resolved))
            self.assertTrue(os.path.isdir(os.path.join(resolved, "prompts")))

    def test_unfrozen_uses_repo_prompts(self):
        with mock.patch.object(sys, "frozen", False, create=True):
            manager = PromptManager()
        self.assertTrue(os.path.isdir(manager.prompts_dir))
        self.assertTrue(
            os.path.isfile(os.path.join(
                manager.prompts_dir, "translator", "system_prompts", "nsfw.json")))


if __name__ == "__main__":
    unittest.main()
