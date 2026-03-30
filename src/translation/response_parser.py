"""Parse LLM translation responses, extract JSON, handle malformed output."""

import ast
import json
import re
from typing import Callable, Optional

from src.logging_helper import emit as log_emit


class ResponseParser:
    _JSON_EXTRACT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
    _MARKDOWN_CODE_RE = re.compile(r"```(?:json)?")
    _TRANSLATION_KV_RE = re.compile(
        r"""["']?translation["']?\s*[:：]\s*(?P<q>["'])(?P<value>.*?)(?P=q)(?=\s*(?:[,}]|$))""",
        flags=re.DOTALL | re.IGNORECASE,
    )
    _BARE_TRANSLATION_RE = re.compile(
        r"""translation\s*[:：]\s*(.+)""",
        flags=re.IGNORECASE,
    )

    def __init__(self, config_manager=None):
        self.config = config_manager

    def parse(self, response: str, original_text: str, messages: list,
              llm_client=None, log_callback: Optional[Callable] = None) -> str:
        """Parse translation from LLM response. Handles JSON, plain text, and recovery."""
        clean_response = self._MARKDOWN_CODE_RE.sub("", response).strip()

        if self._looks_like_broken_json_fragment(clean_response):
            log_emit(log_callback, self.config, "WARNING",
                     f"Discarding broken JSON fragment response: {clean_response[:120]}",
                     module="response_parser", func="parse")
            return str(original_text)

        # Try direct JSON parse
        try:
            data = json.loads(clean_response)
            found, translation = self._extract_translation_value(data)
            if found:
                return translation
        except json.JSONDecodeError:
            pass

        # Accept valid leading JSON even when extra explanatory text is appended.
        data = self._try_parse_first_json_object(clean_response)
        found, translation = self._extract_translation_value(data)
        if found:
            return translation

        # Try extracting JSON substring
        data = self._try_parse_first_json_object(response)
        found, translation = self._extract_translation_value(data)
        if found:
            return translation

        log_emit(log_callback, self.config, "WARNING",
                 f"JSON Parse Error. Response: {response}",
                 module="response_parser", func="parse")

        # Try relaxed JSON extraction (trailing commas, single quotes, bare KV)
        result = self._try_relaxed_json_extract(response, log_callback)
        if result is not None:
            return result

        # Try asking LLM to reformat as JSON
        if llm_client:
            result = self._try_followup_reformat(response, messages, llm_client, log_callback)
            if result is not None:
                return result

        # Plain text fallback
        result = self._try_plain_text_fallback(response)
        if result is not None:
            return result

        return str(response.strip())

    @staticmethod
    def _looks_like_broken_json_fragment(text: str) -> bool:
        if not text:
            return True
        compact = text.strip()
        if compact in {"{", "}", "[", "]", "\"", "'", "{\"", "\"}"}:
            return True
        if compact.startswith("{") and "translation" not in compact.lower() and len(compact) < 8:
            return True
        if compact.count("{") > compact.count("}") and len(compact) < 32:
            return True
        return False

    @staticmethod
    def _try_parse_first_json_object(text: str) -> Optional[dict]:
        """Parse the first valid JSON object from text, ignoring trailing content."""
        if not text:
            return None

        decoder = json.JSONDecoder()

        # Fast path: string starts with JSON object.
        compact = text.lstrip()
        if compact.startswith("{"):
            try:
                parsed, _ = decoder.raw_decode(compact)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # Fallback: find the first decodable object anywhere in the text.
        for match in re.finditer(r"\{", text):
            try:
                parsed, _ = decoder.raw_decode(text[match.start():])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        return None

    @staticmethod
    def _extract_translation_value(data: object) -> tuple[bool, str]:
        if not isinstance(data, dict) or "translation" not in data:
            return False, ""

        value = data.get("translation")
        if value is None:
            return True, ""

        text = value if isinstance(value, str) else str(value)
        return True, text if text.strip() else ""

    def _try_relaxed_json_extract(self, response: str,
                                  log_callback: Optional[Callable] = None) -> Optional[str]:
        """Try to extract translation from malformed JSON-like responses."""
        clean = response.strip()

        # Fix trailing commas: {"translation": "text",} -> {"translation": "text"}
        fixed = re.sub(r",\s*}", "}", clean)
        fixed = re.sub(r",\s*]", "]", fixed)
        try:
            m = self._JSON_EXTRACT_RE.search(fixed)
            if m:
                data = json.loads(m.group(0))
                found, result = self._extract_translation_value(data)
                if found:
                    log_emit(log_callback, self.config, "DEBUG",
                             "Recovered translation via trailing-comma fix",
                             module="response_parser", func="_try_relaxed_json_extract")
                    return result
        except Exception:
            pass

        # Handle Python-like dicts with single quotes.
        if fixed.startswith("{") and fixed.endswith("}") and "'" in fixed:
            try:
                data = ast.literal_eval(fixed)
                found, result = self._extract_translation_value(data)
                if found:
                    log_emit(log_callback, self.config, "DEBUG",
                             "Recovered translation via literal-eval fix",
                             module="response_parser", func="_try_relaxed_json_extract")
                    return result
            except Exception:
                pass

        # Match "translation": "value" or 'translation': 'value' pattern
        m = self._TRANSLATION_KV_RE.search(clean)
        if m:
            log_emit(log_callback, self.config, "DEBUG",
                     "Recovered translation via KV regex",
                     module="response_parser", func="_try_relaxed_json_extract")
            return m.group("value")

        # Match bare translation: value (no quotes)
        m = self._BARE_TRANSLATION_RE.search(clean)
        if m:
            result = m.group(1).strip().strip("\"").strip("'").strip()
            if result:
                log_emit(log_callback, self.config, "DEBUG",
                         "Recovered translation via bare-value regex",
                         module="response_parser", func="_try_relaxed_json_extract")
                return result

        return None

    def _try_followup_reformat(self, response: str, messages: list,
                               llm_client, log_callback) -> Optional[str]:
        """Ask LLM to reformat a non-JSON response into JSON."""
        followup_response = None
        try:
            original_input = None
            for msg in messages:
                if msg.get("role") == "user":
                    original_input = msg.get("content", "")
                    break

            followup_msg = [
                {"role": "system", "content": "你是 JSON 格式化器，只输出合法 JSON。"},
                {"role": "user", "content": (
                    f"原任务输入：{original_input}\n\n"
                    f"模型回复：{response}\n\n"
                    "请提取其中最终译文，并按以下格式返回："
                    "{\"translation\":\"...\"}\n"
                    "只输出合法 JSON，不要输出其他内容。"
                )}
            ]
            followup_response = llm_client.chat_completion(followup_msg, log_callback=log_callback)
            clean_followup = self._MARKDOWN_CODE_RE.sub("", followup_response).strip()
            data = json.loads(clean_followup)
            found, result = self._extract_translation_value(data)
            if not found:
                return None

            # Safety check for prompt leakage
            prompt_patterns = [
                "reformat", "Respond only", "Output only", "Extract the",
                "格式化器", "原任务输入", "模型回复", "只输出合法 JSON",
                "{\"translation\"",
            ]
            if result and not any(pattern in result for pattern in prompt_patterns):
                return result
            if result:
                log_emit(log_callback, self.config, "WARNING",
                         f"Followup response may contain prompt leakage: {result[:100]}...",
                         module="response_parser", func="_try_followup_reformat")
        except json.JSONDecodeError:
            if followup_response:
                log_emit(log_callback, self.config, "WARNING",
                         f"Followup JSON Parse Error. Response: {followup_response}",
                         module="response_parser", func="_try_followup_reformat")
        except Exception:
            pass
        return None

    def _try_plain_text_fallback(self, response: str) -> Optional[str]:
        """If response looks like plain translation text (not JSON/meta output), use it directly."""
        clean = response.strip()
        if not clean or clean.startswith("{"):
            return None
        if self._BARE_TRANSLATION_RE.match(clean):
            return None
        return clean
