"""
Unit tests for cogops.pipeline.query_expand
"""

import unittest

from cogops.pipeline.query_expand import (
    check_document_type_match,
    expand_sub_query,
    extract_document_type,
)


class TestExpandSubQuery(unittest.TestCase):
    def test_marriage_certificate_expansion(self):
        q = "বিয়ের সার্টিফিকেটে নাম পরিবর্তন"
        out = expand_sub_query(q)
        self.assertIn("বিবাহ সনদ", out)
        self.assertIn("বিবাহিত প্রত্যয়ন", out)
        self.assertTrue(out.startswith(q))

    def test_nid_expansion(self):
        q = "এনআইডি কার্ড হারিয়ে গেলে কী করব?"
        out = expand_sub_query(q)
        self.assertIn("জাতীয় পরিচয়পত্র", out)
        self.assertIn("স্মার্ট কার্ড", out)

    def test_no_expansion_when_no_doc_type(self):
        q = "সরকারি চাকরির আবেদন কীভাবে করব?"
        out = expand_sub_query(q)
        self.assertEqual(out, q)

    def test_passthrough_already_formal(self):
        q = "বিবাহ সনদে নাম সংশোধন"
        out = expand_sub_query(q)
        # Should not duplicate terms already present
        self.assertEqual(out.count("বিবাহ সনদ"), 1)


class TestExtractDocumentType(unittest.TestCase):
    def test_marriage_certificate(self):
        self.assertEqual(extract_document_type("বিয়ের সার্টিফিকেট"), "বিবাহ সনদ")
        self.assertEqual(extract_document_type("বিবাহ সনদ"), "বিবাহ সনদ")
        self.assertEqual(extract_document_type("বিবাহিত প্রত্যয়ন"), "বিবাহ সনদ")

    def test_nid(self):
        self.assertEqual(extract_document_type("এনআইডি"), "এনআইডি")
        self.assertEqual(extract_document_type("স্মার্ট কার্ড"), "এনআইডি")

    def test_passport(self):
        self.assertEqual(extract_document_type("পাসপোর্ট"), "পাসপোর্ট")

    def test_no_doc_type(self):
        self.assertIsNone(extract_document_type("সরকারি চাকরির আবেদন"))


class TestCheckDocumentTypeMatch(unittest.TestCase):
    def test_match_found(self):
        source_map = {
            "S1": {
                "category": "জরুরি প্রত্যয়ন ও সনদ",
                "sub_category": "জরুরি প্রত্যয়ন",
                "service": "বিবাহিত প্রত্যয়ন: আবেদন পদ্ধতি",
                "topic": "বিবাহিত প্রত্যয়নপত্রের জন্য আবেদন যেভাবে করতে হবে",
                "text": "some text",
            }
        }
        self.assertTrue(check_document_type_match("বিয়ের সার্টিফিকেট", source_map))

    def test_no_match_nid_vs_marriage(self):
        source_map = {
            "S1": {
                "category": "স্মার্ট কার্ড ও জাতীয়পরিচয়পত্র",
                "sub_category": "স্মার্ট কার্ড",
                "service": "এনআইডি সংশোধন",
                "topic": "নাম পরিবর্তন",
                "text": "বিয়ের পর স্বামীর নাম NID কার্ডে",
            }
        }
        self.assertFalse(check_document_type_match("বিয়ের সার্টিফিকেট", source_map))

    def test_no_doc_type_skips_check(self):
        source_map = {
            "S1": {
                "category": "Random",
                "text": "hello",
            }
        }
        self.assertTrue(check_document_type_match("সাধারণ প্রশ্ন", source_map))

    def test_english_category_match(self):
        source_map = {
            "S1": {
                "category": "NID",
                "sub_category": "",
                "service": "",
                "topic": "",
                "text": "some text",
            }
        }
        self.assertTrue(check_document_type_match("এনআইডি কার্ড", source_map))


if __name__ == "__main__":
    unittest.main()
