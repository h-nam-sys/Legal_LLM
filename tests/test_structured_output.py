"""Model literals remain data; schema validation stays with each caller."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.structured_output import parse_model_value


class StructuredOutputTests(unittest.TestCase):
    def test_json_and_single_quoted_lists_preserve_vietnamese(self):
        for text in ('["mượn sách", "14 ngày"]', "['mượn sách', '14 ngày']"):
            self.assertEqual(parse_model_value(text), ['mượn sách', '14 ngày'])

    def test_apostrophes_inside_json_strings_are_unchanged(self):
        self.assertEqual(parse_model_value('["student\'s card"]'), ["student's card"])

    def test_fenced_arrays_are_accepted(self):
        self.assertEqual(parse_model_value('```json\n["fee"]\n```'), ['fee'])

    def test_expressions_and_code_are_rejected(self):
        for text in ('[open("secret")]', '[x for x in []]', '["fee"] + ["description"]',
                     '__import__("os").system("whoami")', '("fee",)', '{"fee"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_model_value(text)

    def test_prose_and_oversized_text_are_rejected(self):
        for text in ('The answer is ["fee"]', 'x' * 65537, None):
            with self.assertRaises(ValueError):
                parse_model_value(text)


if __name__ == '__main__':
    unittest.main()
