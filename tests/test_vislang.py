"""Tests for the vislang package."""

import os
import json
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from vislang.scraper import HumanitarianReportScraper, DUMMY_REPORTS
from vislang.ocr import OCRPipeline
from vislang.alignment import AlignmentBuilder
from vislang.exporter import DatasetExporter
from vislang.stats import DatasetStats


class TestHumanitarianReportScraper(unittest.TestCase):

    def setUp(self):
        self.scraper = HumanitarianReportScraper()

    def test_fetch_reports_returns_list(self):
        with patch("requests.get", side_effect=Exception("network error")):
            result = self.scraper.fetch_reports()
        self.assertIsInstance(result, list)

    def test_fetch_reports_fallback_has_two_entries(self):
        with patch("requests.get", side_effect=Exception("network error")):
            result = self.scraper.fetch_reports()
        self.assertEqual(len(result), 2)

    def test_fetch_reports_fallback_required_keys(self):
        required = {"title", "url", "date", "language", "summary"}
        with patch("requests.get", side_effect=Exception("network error")):
            result = self.scraper.fetch_reports()
        for item in result:
            self.assertTrue(required.issubset(item.keys()), f"Missing keys in {item}")

    def test_fetch_reports_success_returns_list(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "data": [
                {
                    "fields": {
                        "title": "Test Report",
                        "url_alias": "https://reliefweb.int/report/test",
                        "date": {"created": "2024-01-01"},
                        "language": [{"code": "en"}],
                        "body": "Test body text.",
                    }
                }
            ]
        }
        with patch("requests.get", return_value=mock_resp):
            result = self.scraper.fetch_reports()
        self.assertIsInstance(result, list)
        self.assertGreaterEqual(len(result), 1)

    def test_fetch_images_returns_list_on_failure(self):
        with patch("requests.get", side_effect=Exception("network error")):
            result = self.scraper.fetch_images_from_report("https://example.com/report")
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    def test_fetch_images_extracts_urls(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.text = '<img src="https://example.com/img1.jpg"><img src="https://example.com/img2.png">'
        with patch("requests.get", return_value=mock_resp):
            result = self.scraper.fetch_images_from_report("https://example.com/report")
        self.assertEqual(len(result), 2)
        self.assertIn("https://example.com/img1.jpg", result)


class TestOCRPipeline(unittest.TestCase):

    def setUp(self):
        self.ocr = OCRPipeline()

    def test_is_available_returns_bool(self):
        result = self.ocr.is_available()
        self.assertIsInstance(result, bool)

    def test_extract_text_fallback_returns_str(self):
        """extract_text returns str even when pytesseract is not available."""
        ocr = OCRPipeline(fallback_text="[OCR unavailable]")
        # Force pytesseract to be None/missing to trigger fallback
        saved = sys.modules.get("pytesseract", "NOTSET")
        sys.modules["pytesseract"] = None
        try:
            result = ocr.extract_text("dummy_image.jpg")
            self.assertIsInstance(result, str)
        finally:
            if saved == "NOTSET":
                sys.modules.pop("pytesseract", None)
            else:
                sys.modules["pytesseract"] = saved

    def test_extract_text_fallback_contains_prefix(self):
        """extract_text fallback contains the fallback_text prefix."""
        ocr = OCRPipeline(fallback_text="[OCR unavailable]")
        saved = sys.modules.get("pytesseract", "NOTSET")
        sys.modules["pytesseract"] = None
        try:
            result = ocr.extract_text("test_image.png")
            self.assertIn("[OCR unavailable]", result)
        finally:
            if saved == "NOTSET":
                sys.modules.pop("pytesseract", None)
            else:
                sys.modules["pytesseract"] = saved

    def test_extract_batch_returns_list(self):
        images = ["img1.jpg", "img2.jpg", "img3.jpg"]
        results = self.ocr.extract_batch(images)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, str)

    def test_extract_batch_empty(self):
        result = self.ocr.extract_batch([])
        self.assertEqual(result, [])


class TestAlignmentBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = AlignmentBuilder()

    def test_split_sentences_basic(self):
        text = "Hello world. This is a test. Another sentence."
        sentences = self.builder.split_sentences(text)
        self.assertIsInstance(sentences, list)
        self.assertGreaterEqual(len(sentences), 2)
        self.assertIn("Hello world", sentences)

    def test_split_sentences_exclamation(self):
        text = "Wow! Amazing! Great"
        sentences = self.builder.split_sentences(text)
        self.assertGreaterEqual(len(sentences), 2)

    def test_split_sentences_question(self):
        text = "Is this working? Yes it is."
        sentences = self.builder.split_sentences(text)
        self.assertGreaterEqual(len(sentences), 2)

    def test_align_returns_list_of_dicts(self):
        images = ["http://example.com/img1.jpg", "http://example.com/img2.jpg"]
        text = "First sentence. Second sentence. Third sentence."
        pairs = self.builder.align(images, text)
        self.assertIsInstance(pairs, list)
        required = {"image_url", "sentence", "score"}
        for p in pairs:
            self.assertTrue(required.issubset(p.keys()), f"Missing keys: {p}")

    def test_align_round_robin(self):
        images = ["img1.jpg"]
        text = "Sentence one. Sentence two."
        pairs = self.builder.align(images, text)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["image_url"], "img1.jpg")
        self.assertEqual(pairs[1]["image_url"], "img1.jpg")

    def test_align_score_range(self):
        images = ["img1.jpg"]
        text = "Short. This is a longer sentence here."
        pairs = self.builder.align(images, text)
        for p in pairs:
            self.assertGreaterEqual(p["score"], 0.0)
            self.assertLessEqual(p["score"], 1.0)

    def test_align_empty_inputs(self):
        self.assertEqual(self.builder.align([], "Some text."), [])
        self.assertEqual(self.builder.align(["img.jpg"], ""), [])

    def test_filter_short_removes_short_pairs(self):
        pairs = [
            {"image_url": "img1.jpg", "sentence": "Hi", "score": 0.1},
            {"image_url": "img2.jpg", "sentence": "This is a longer sentence", "score": 0.9},
            {"image_url": "img3.jpg", "sentence": "Short", "score": 0.2},
        ]
        filtered = self.builder.filter_short(pairs, min_len=10)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["sentence"], "This is a longer sentence")

    def test_filter_short_default(self):
        pairs = [
            {"image_url": "img.jpg", "sentence": "Hello", "score": 0.1},
            {"image_url": "img.jpg", "sentence": "Hello world!", "score": 0.8},
        ]
        filtered = self.builder.filter_short(pairs)
        self.assertEqual(len(filtered), 1)


class TestDatasetExporter(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.exporter = DatasetExporter(self.tmpdir)

    def test_export_jsonl_creates_file(self):
        pairs = [{"image_url": "img1.jpg", "sentence": "Caption one", "language": "en", "score": 0.8}]
        path = self.exporter.export_jsonl(pairs)
        self.assertTrue(os.path.exists(path))

    def test_export_jsonl_returns_path(self):
        path = self.exporter.export_jsonl([])
        self.assertIsInstance(path, str)

    def test_load_jsonl_round_trip(self):
        pairs = [
            {"image_url": "img1.jpg", "sentence": "Caption one", "language": "en", "score": 0.8},
            {"image_url": "img2.jpg", "sentence": "Caption two", "language": "fr", "score": 0.5},
        ]
        path = self.exporter.export_jsonl(pairs)
        loaded = self.exporter.load_jsonl(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["image_url"], "img1.jpg")
        self.assertEqual(loaded[0]["text"], "Caption one")
        self.assertEqual(loaded[1]["language"], "fr")

    def test_export_metadata_creates_file(self):
        stats = {"n_pairs": 5, "vocab_size": 100}
        path = self.exporter.export_metadata(stats)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["n_pairs"], 5)

    def test_load_jsonl_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, dir=self.tmpdir) as f:
            path = f.name
        result = self.exporter.load_jsonl(path)
        self.assertEqual(result, [])


class TestDatasetStats(unittest.TestCase):

    def setUp(self):
        self.stats = DatasetStats()
        self.sample_pairs = [
            {"image_url": "img1.jpg", "text": "flooding in somalia", "language": "en", "score": 0.9},
            {"image_url": "img2.jpg", "text": "aid workers arrived", "language": "en", "score": 0.7},
            {"image_url": "img3.jpg", "text": "rapport sur la crise", "language": "fr", "score": 0.6},
        ]

    def test_compute_returns_dict(self):
        result = self.stats.compute(self.sample_pairs)
        self.assertIsInstance(result, dict)

    def test_compute_has_required_keys(self):
        required = {"n_pairs", "vocab_size", "image_count", "language_distribution", "avg_text_len"}
        result = self.stats.compute(self.sample_pairs)
        self.assertTrue(required.issubset(result.keys()), f"Missing: {required - result.keys()}")

    def test_compute_n_pairs(self):
        result = self.stats.compute(self.sample_pairs)
        self.assertEqual(result["n_pairs"], 3)

    def test_compute_vocab_size_positive(self):
        result = self.stats.compute(self.sample_pairs)
        self.assertGreater(result["vocab_size"], 0)

    def test_compute_language_distribution_is_dict(self):
        result = self.stats.compute(self.sample_pairs)
        self.assertIsInstance(result["language_distribution"], dict)

    def test_compute_language_distribution_counts(self):
        result = self.stats.compute(self.sample_pairs)
        dist = result["language_distribution"]
        self.assertEqual(dist.get("en", 0), 2)
        self.assertEqual(dist.get("fr", 0), 1)

    def test_compute_empty_pairs(self):
        result = self.stats.compute([])
        self.assertEqual(result["n_pairs"], 0)
        self.assertEqual(result["vocab_size"], 0)
        self.assertEqual(result["avg_text_len"], 0.0)

    def test_build_vocab_returns_set(self):
        texts = ["hello world", "hello python"]
        vocab = self.stats.build_vocab(texts)
        self.assertIsInstance(vocab, set)
        self.assertIn("hello", vocab)
        self.assertIn("world", vocab)

    def test_build_vocab_lowercase(self):
        vocab = self.stats.build_vocab(["Hello WORLD"])
        self.assertIn("hello", vocab)
        self.assertNotIn("Hello", vocab)

    def test_print_report_runs(self):
        result = self.stats.compute(self.sample_pairs)
        self.stats.print_report(result)  # should not raise


if __name__ == "__main__":
    unittest.main()
