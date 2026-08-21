import unittest

from core.text_utils import split_into_chunks


class SplitIntoChunksTests(unittest.TestCase):
    def test_splits_text_with_overlap(self):
        self.assertEqual(
            split_into_chunks("ABCDEFGHIJKLMN", chunk_size=10, overlap=3),
            ["ABCDEFGHIJ", "HIJKLMN"],
        )

    def test_skips_whitespace_only_chunks(self):
        self.assertEqual(split_into_chunks("   ", chunk_size=10, overlap=0), [])

    def test_rejects_non_positive_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            split_into_chunks("text", chunk_size=0, overlap=0)

    def test_rejects_negative_overlap(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            split_into_chunks("text", chunk_size=10, overlap=-1)

    def test_rejects_overlap_equal_to_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "smaller than chunk_size"):
            split_into_chunks("text", chunk_size=10, overlap=10)

    def test_rejects_overlap_larger_than_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "smaller than chunk_size"):
            split_into_chunks("text", chunk_size=10, overlap=11)


if __name__ == "__main__":
    unittest.main()
