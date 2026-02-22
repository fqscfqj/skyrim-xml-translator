"""Parse LLM translation responses, extract JSON, handle malformed output."""

import json
import re
from typing import Optional, Callable

from src.logging_helper import emit as log_emit


class ResponseParser:
    _JSON_EXTRACT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)
    _MARKDOWN_CODE_RE = re.compile(r'```(?:json)?')
    _CJK_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')

    def __init__(self, config_manager=None):
        self.config = config_manager

    def parse(self, response: str, original_text: str, messages: list,
              llm_client=None, log_callback: Optional[Callable] = None) -> str:
        """Parse translation from LLM response. Handles JSON, plain text, and recovery."""
        clean_response = self._MARKDOWN_CODE_RE.sub('', response).strip()

        # Try direct JSON parse
        try:
            data = json.loads(clean_response)
            return str(data.get("translation", original_text))
        except json.JSONDecodeError:
            pass

        # Try extracting JSON substring
        log_emit(log_callback, self.config, "WARNING",
                 f"JSON Parse Error. Response: {response}",
                 module="response_parser", func="parse")

        try:
            m = self._JSON_EXTRACT_RE.search(response)
            if m:
                data = json.loads(m.group(0))
                return str(data.get("translation", response.strip()))
        except Exception:
            pass

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
            clean_followup = self._MARKDOWN_CODE_RE.sub('', followup_response).strip()
            data = json.loads(clean_followup)
            result = str(data.get("translation", ""))

            # Safety check for prompt leakage
            prompt_patterns = [
                "reformat", "Respond only", "Output only", "Extract the",
                "格式化器", "原任务输入", "模型回复", "只输出合法 JSON",
                "{\"translation\"", "translation", "JSON", "json"
            ]
            if result and not any(pattern in result for pattern in prompt_patterns if len(pattern) > 5):
                return result
            elif result:
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
        """If response looks like valid Chinese translation (not JSON), use it directly."""
        clean = response.strip()
        chinese_chars = len(self._CJK_CHAR_RE.findall(clean))
        if chinese_chars > 0 and not clean.startswith('{'):
            return clean
        return None
