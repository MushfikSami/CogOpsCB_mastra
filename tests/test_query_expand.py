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
    def test_passthrough_no_hardcoded_map(self):
        """expand_sub_query is a pure passthrough — formalization is LLM-only."""
        q = "বিয়ের সার্টিফিকেটে নাম পরিবর্তন"
        out = expand_sub_query(q)
        self.assertEqual(out, q)

    def test_passthrough_nid(self):
        q = "এনআইডি কার্ড হারিয়ে গেলে কী করব?"
        out = expand_sub_query(q)
        self.assertEqual(out, q)

    def test_passthrough_passport(self):
        q = "পাসপোর্ট করার নিয়ম কি?"
        out = expand_sub_query(q)
        self.assertEqual(out, q)

    def test_passthrough_no_doc_type(self):
        q = "সরকারি চাকরির আবেদন কীভাবে করব?"
        out = expand_sub_query(q)
        self.assertEqual(out, q)

    def test_passthrough_already_formal(self):
        q = "বিবাহ সনদে নাম সংশোধন"
        out = expand_sub_query(q)
        self.assertEqual(out, q)

    def test_passthrough_metro(self):
        q = "মেট্রোরেলের ভাড়া কত?"
        out = expand_sub_query(q)
        self.assertEqual(out, q)

    def test_passthrough_utility_gas(self):
        q = "গ্যাস সংযোগ নিতে চাই"
        out = expand_sub_query(q)
        self.assertEqual(out, q)

    def test_passthrough_utility_wasa(self):
        q = "ওয়াসার বিল কত?"
        out = expand_sub_query(q)
        self.assertEqual(out, q)

    def test_empty_query(self):
        out = expand_sub_query("")
        self.assertEqual(out, "")

    def test_passthrough_generic_question(self):
        q = "বাংলাদেশের প্রধানমন্ত্রী কে?"
        out = expand_sub_query(q)
        self.assertEqual(out, q)


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
        self.assertEqual(extract_document_type("ই-পাসপোর্ট"), "পাসপোর্ট")

    def test_no_doc_type(self):
        self.assertIsNone(extract_document_type("সরকারি চাকরির আবেদন"))
        self.assertIsNone(extract_document_type(""))


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
