import unittest

from core.prompting import build_rag_prompt


class PromptingTests(unittest.TestCase):
    def test_includes_latest_three_turns_and_current_question(self):
        history = [
            {"question": f"question-{i}", "answer": f"answer-{i}"}
            for i in range(4)
        ]

        prompt = build_rag_prompt("[Page 1]: context", "follow-up", history)

        self.assertNotIn("question-0", prompt)
        for i in range(1, 4):
            self.assertIn(f"question-{i}", prompt)
            self.assertIn(f"answer-{i}", prompt)
        self.assertIn("Question: follow-up", prompt)

    def test_marks_empty_history(self):
        prompt = build_rag_prompt("context", "question", [])
        self.assertIn("Previous conversation:\nNone", prompt)


if __name__ == "__main__":
    unittest.main()
