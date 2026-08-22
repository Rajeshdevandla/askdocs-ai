import unittest

from core.llm_client import DemoLLMClient


class DemoLLMClientTests(unittest.TestCase):
    def test_returns_labeled_context_without_external_call(self):
        prompt = "Document context:\n[Page 2]: Policy renews yearly.\n\nQuestion: When?"
        answer = DemoLLMClient().generate(prompt)

        self.assertIn("Demo mode", answer)
        self.assertIn("[Page 2]: Policy renews yearly.", answer)


if __name__ == "__main__":
    unittest.main()
