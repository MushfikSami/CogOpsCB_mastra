"""
Unit tests for cogops.pipeline.normalize
"""

import unittest

from cogops.pipeline.normalize import normalize_sub_query, normalize_sub_queries


class TestNormalizeSubQuery(unittest.TestCase):
    def test_plane_to_biman(self):
        raw = "আচ্ছা প্লেনের টিকেট কিভাবে কাটবো?"
        out = normalize_sub_query(raw)
        self.assertIn("বিমান", out)
        self.assertNotIn("প্লেন", out)
        self.assertNotIn("আচ্ছা", out)

    def test_train_to_rail(self):
        raw = "ট্রেনের টিকেট কোথায় পাব?"
        out = normalize_sub_query(raw)
        self.assertIn("রেল", out)
        self.assertNotIn("ট্রেন", out)

    def test_ticket_to_tikit(self):
        raw = "টিকেট কাটার নিয়ম কী?"
        out = normalize_sub_query(raw)
        self.assertIn("টিকিট", out)
        self.assertNotIn("টিকেট", out)

    def test_filler_removal(self):
        raw = "আচ্ছা ভাই, দেখেন শুনুন বলুন তো একটু কিন্তু তাই তাহলে"
        out = normalize_sub_query(raw)
        self.assertEqual(out, "")

    def test_no_false_positive_in_middle_of_word(self):
        # "ভাই" should not be stripped from "ভাইস" or similar
        raw = "ভাইস চেয়ারম্যান কে?"
        out = normalize_sub_query(raw)
        # "ভাই" is a standalone word here, but "ভাইস" contains it.
        # Our regex uses word boundaries, so "ভাই" inside "ভাইস" should NOT match.
        self.assertIn("ভাইস", out)

    def test_whitespace_collapse(self):
        raw = "পাসপোর্ট    ফি   কত?"
        out = normalize_sub_query(raw)
        self.assertNotIn("  ", out)
        self.assertEqual(out, "পাসপোর্ট ফি কত?")

    def test_passthrough_already_formal(self):
        raw = "বিমানের টিকিট কাটার নিয়ম কী?"
        out = normalize_sub_query(raw)
        self.assertEqual(out, raw)


class TestNormalizeSubQueries(unittest.TestCase):
    def test_drops_empty_after_cleanup(self):
        queries = ["আচ্ছা ভাই", "পাসপোর্ট ফি কত?"]
        out = normalize_sub_queries(queries)
        self.assertEqual(out, ["পাসপোর্ট ফি কত?"])

    def test_fallback_when_all_empty(self):
        queries = ["আচ্ছা", "ভাই"]
        out = normalize_sub_queries(queries)
        # fallback to original list if everything is stripped
        self.assertEqual(out, queries)


if __name__ == "__main__":
    unittest.main()
