import unittest

from app import OllamaError, default_settings, default_two_pass_prompts, ollama_chat_content


class OllamaChatContentTests(unittest.TestCase):
    def test_returns_final_content(self):
        response = {
            "message": {"content": "Réponse finale", "thinking": "raisonnement"},
            "done_reason": "stop",
        }

        self.assertEqual(ollama_chat_content(response), "Réponse finale")

    def test_explains_thinking_budget_exhaustion(self):
        response = {
            "message": {"content": "", "thinking": "raisonnement inachevé"},
            "done_reason": "length",
        }

        with self.assertRaisesRegex(OllamaError, "num_predict.*done_reason=length"):
            ollama_chat_content(response)

    def test_explains_empty_response_without_thinking(self):
        with self.assertRaisesRegex(OllamaError, "sans texte.*done_reason=stop"):
            ollama_chat_content({"message": {"content": ""}, "done_reason": "stop"})


class TwoPassPromptTests(unittest.TestCase):
    def test_both_two_pass_prompts_are_exposed_with_common_context(self):
        settings = default_settings()
        prompts = default_two_pass_prompts(settings)

        self.assertEqual(set(prompts), {"analysis_prompt_d1", "analysis_prompt_d3"})
        self.assertIn(settings["analysis_objective"], prompts["analysis_prompt_d1"])
        self.assertIn(settings["analysis_objective"], prompts["analysis_prompt_d3"])
        self.assertIn("selected_keyframes", prompts["analysis_prompt_d1"])


if __name__ == "__main__":
    unittest.main()
