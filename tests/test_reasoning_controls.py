import unittest

from src.llm.reasoning import apply_reasoning_controls, detect_reasoning_protocol


class ReasoningProtocolDetectionTests(unittest.TestCase):
    def test_endpoint_takes_priority_over_model_name(self):
        self.assertEqual(
            detect_reasoning_protocol(
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "deepseek-v4-flash",
            ),
            "qwen",
        )

    def test_recognizes_common_openai_compatible_providers(self):
        cases = (
            ("https://openrouter.ai/api/v1", "anthropic/claude-sonnet-5", "openrouter"),
            ("https://api.deepseek.com/v1", "deepseek-v4-pro", "deepseek"),
            ("https://api.meta.ai/v1", "muse-spark-1.2", "standard"),
            ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.6-flash", "gemini"),
            ("https://proxy.example/v1", "claude-sonnet-4-6", "anthropic_adaptive"),
            ("https://proxy.example/v1", "claude-opus-4-5", "anthropic_adaptive"),
            ("https://proxy.example/v1", "claude-sonnet-4-5", "standard"),
            ("https://proxy.example/v1", "qwen3.8-max", "qwen"),
        )
        for base_url, model, expected in cases:
            with self.subTest(model=model):
                self.assertEqual(detect_reasoning_protocol(base_url, model), expected)


class ReasoningControlMappingTests(unittest.TestCase):
    def test_standard_protocol_uses_top_level_effort_and_none_for_off(self):
        params = {
            "reasoning_protocol": "standard",
            "enable_thinking": False,
            "reasoning_effort": "high",
        }
        extra = {}

        applied = apply_reasoning_controls(
            params, extra, base_url="https://api.meta.ai/v1", model="muse-spark-1.2"
        )

        self.assertTrue(applied.applied)
        self.assertEqual(params["reasoning_effort"], "none")
        self.assertEqual(extra, {})

        enabled_params = {
            "reasoning_protocol": "standard",
            "enable_thinking": True,
        }
        apply_reasoning_controls(
            enabled_params, {}, base_url="https://api.meta.ai/v1", model="muse-spark-1.2"
        )
        self.assertEqual(enabled_params["reasoning_effort"], "medium")

    def test_standard_max_maps_to_each_provider_effective_maximum(self):
        cases = (
            ("https://api.meta.ai/v1", "muse-spark-1.2", "high"),
            ("https://api.openai.com/v1", "gpt-5.4", "xhigh"),
        )
        for base_url, model, expected in cases:
            with self.subTest(model=model):
                params = {
                    "reasoning_protocol": "standard",
                    "reasoning_effort": "max",
                }
                apply_reasoning_controls(
                    params, {}, base_url=base_url, model=model
                )
                self.assertEqual(params["reasoning_effort"], expected)

    def test_deepseek_uses_thinking_object_and_strips_sampling(self):
        params = {
            "reasoning_protocol": "auto",
            "enable_thinking": True,
            "reasoning_effort": "high",
            "temperature": 0.2,
            "top_p": 0.8,
        }
        extra = {}

        apply_reasoning_controls(
            params, extra, base_url="https://api.deepseek.com/v1", model="deepseek-v4-pro"
        )

        self.assertEqual(extra, {"thinking": {"type": "enabled"}})
        self.assertEqual(params["reasoning_effort"], "high")
        self.assertNotIn("temperature", params)
        self.assertNotIn("top_p", params)

    def test_deepseek_maps_neutral_effort_to_native_v4_levels(self):
        cases = (
            ("minimal", "high"),
            ("low", "high"),
            ("medium", "high"),
            ("high", "high"),
            ("xhigh", "max"),
            ("max", "max"),
        )
        for effort, expected in cases:
            with self.subTest(effort=effort):
                params = {
                    "reasoning_protocol": "deepseek",
                    "reasoning_effort": effort,
                }
                apply_reasoning_controls(params, {}, base_url="", model="")
                self.assertEqual(params["reasoning_effort"], expected)

    def test_qwen_uses_switch_and_top_level_effort(self):
        params = {
            "reasoning_protocol": "qwen",
            "enable_thinking": True,
            "reasoning_effort": "medium",
        }
        extra = {}

        apply_reasoning_controls(params, extra, base_url="", model="qwen3.8-max")

        self.assertEqual(extra, {"enable_thinking": True})
        self.assertEqual(params["reasoning_effort"], "medium")

    def test_openrouter_emits_unified_reasoning_object(self):
        params = {
            "reasoning_protocol": "openrouter",
            "enable_thinking": True,
            "reasoning_effort": "high",
        }
        extra = {}

        apply_reasoning_controls(params, extra, base_url="", model="")

        self.assertEqual(extra, {"reasoning": {"effort": "high"}})
        self.assertNotIn("reasoning_effort", params)

    def test_anthropic_adaptive_uses_output_config_effort(self):
        params = {
            "reasoning_protocol": "anthropic_adaptive",
            "enable_thinking": True,
            "reasoning_effort": "medium",
        }
        extra = {}

        apply_reasoning_controls(params, extra, base_url="", model="")

        self.assertEqual(extra, {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium"},
        })

    def test_anthropic_effort_does_not_force_adaptive_thinking(self):
        params = {
            "reasoning_protocol": "anthropic_adaptive",
            "reasoning_effort": "minimal",
        }
        extra = {}

        apply_reasoning_controls(
            params, extra, base_url="", model="claude-opus-4-5"
        )

        self.assertEqual(extra, {"output_config": {"effort": "low"}})

    def test_mandatory_anthropic_thinking_maps_off_to_lowest_adaptive_effort(self):
        params = {
            "reasoning_protocol": "auto",
            "enable_thinking": False,
        }
        extra = {}

        apply_reasoning_controls(
            params, extra, base_url="https://proxy.example/v1", model="claude-opus-4-8"
        )

        self.assertEqual(extra, {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
        })

    def test_gemini_maps_unsupported_max_effort_to_high(self):
        params = {
            "reasoning_protocol": "gemini",
            "reasoning_effort": "max",
        }
        extra = {}

        apply_reasoning_controls(params, extra, base_url="", model="")

        self.assertEqual(extra, {})
        self.assertEqual(params["reasoning_effort"], "high")

    def test_removed_reasoning_budget_is_ignored(self):
        params = {
            "reasoning_protocol": "qwen",
            "reasoning_effort": "high",
            "reasoning_budget_tokens": 1024,
        }
        extra = {}

        apply_reasoning_controls(params, extra, base_url="", model="")

        self.assertNotIn("reasoning_budget_tokens", params)
        self.assertNotIn("thinking_budget", extra)
        self.assertEqual(params["reasoning_effort"], "high")

    def test_mandatory_gemini_thinking_maps_off_to_lowest_supported_effort(self):
        params = {
            "reasoning_protocol": "gemini",
            "enable_thinking": False,
        }

        apply_reasoning_controls(
            params, {}, base_url="", model="gemini-3.1-pro-preview"
        )

        self.assertEqual(params["reasoning_effort"], "low")


if __name__ == "__main__":
    unittest.main()
