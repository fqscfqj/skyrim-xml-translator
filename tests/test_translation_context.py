import unittest

from src.gui_main import Worker
from src.translation.prompt_builder import PromptBuilder


class TranslationContextSignatureTests(unittest.TestCase):
    def test_structured_record_fields_isolate_dedupe_and_batch_context(self):
        source = "Iron Sword"
        weapon = {
            "file_type": "esp_xml",
            "record_type": "WEAP",
            "field_type": "FULL",
            "style_profile": "item_name",
            "content_mode": "default",
        }
        dialogue = {
            "file_type": "esp_xml",
            "record_type": "DIAL",
            "field_type": "NAM1",
            "style_profile": "dialogue",
            "content_mode": "default",
        }

        self.assertNotEqual(
            Worker._translation_context_signature(source, weapon),
            Worker._translation_context_signature(source, dialogue),
        )
        self.assertNotEqual(
            Worker._translation_dedupe_key(source, weapon),
            Worker._translation_dedupe_key(source, dialogue),
        )

    def test_previous_source_only_participates_for_dialogue(self):
        dialogue = PromptBuilder.context_fingerprint_data(
            {"text_kind": "dialogue", "previous_source": "Who goes there?"},
            "Wait here.",
        )
        document = PromptBuilder.context_fingerprint_data(
            {"text_kind": "document", "previous_source": "Who goes there?"},
            "Wait here.",
        )
        self.assertIn("previous_source", dialogue)
        self.assertNotIn("previous_source", document)

    def test_compact_short_label_drops_entry_id(self):
        compact = PromptBuilder.context_fingerprint_data(
            {"text_kind": "ui", "entry_id": "HEALTH_OPTION"},
            "Health",
        )
        sentence = PromptBuilder.context_fingerprint_data(
            {"text_kind": "quest", "entry_id": "MQ101Objective"},
            "Find the Dragon Stone.",
        )
        self.assertNotIn("entry_id", compact)
        self.assertIn("entry_id", sentence)

    def test_compact_identifier_labels_keep_passthrough_boundary(self):
        matching = {"text_kind": "generic", "entry_id": "SKSE64"}
        different = {"text_kind": "generic", "entry_id": "OtherEntry"}
        self.assertNotEqual(
            Worker._translation_dedupe_key("SKSE64", matching),
            Worker._translation_dedupe_key("SKSE64", different),
        )

    def test_whitespace_policies_isolate_worker_reuse(self):
        self.assertNotEqual(
            Worker._translation_dedupe_key(
                "Wait here.", {"text_kind": "dialogue", "whitespace_policy": "strict"}),
            Worker._translation_dedupe_key(
                "Wait here.", {"text_kind": "dialogue", "whitespace_policy": "relaxed_spaces"}),
        )

    def test_compact_label_keeps_inferred_style_and_control_semantics(self):
        for first, second in (
            ({"record_type": "BOOK", "entry_id": "BookDiary"},
             {"record_type": "BOOK", "entry_id": "BookHistory"}),
            ({"text_kind": "ui", "entry_id": "$FOO_TT_BAR"},
             {"text_kind": "ui", "entry_id": "$FOO_HEADER"}),
        ):
            with self.subTest(first=first):
                self.assertNotEqual(
                    Worker._translation_dedupe_key("Read", first),
                    Worker._translation_dedupe_key("Read", second),
                )

    def test_other_identifiers_are_not_shadowed_by_entry_id(self):
        self.assertNotEqual(
            Worker._translation_dedupe_key(
                "Read this book.", {"entry_id": "same", "edid": "BookDiary"}),
            Worker._translation_dedupe_key(
                "Read this book.", {"entry_id": "same", "edid": "BookHistory"}),
        )

    def test_missing_or_non_dict_context_is_safe(self):
        self.assertEqual({}, PromptBuilder.context_fingerprint_data(None, "Health"))
        self.assertEqual({}, PromptBuilder.context_fingerprint_data([], "Health"))


if __name__ == "__main__":
    unittest.main()
