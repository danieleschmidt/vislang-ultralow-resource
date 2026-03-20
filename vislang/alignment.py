"""AlignmentBuilder: pair image regions with caption sentences."""

import re


class AlignmentBuilder:
    """Pair image URLs with caption sentences."""

    def __init__(self, min_sentence_len=5):
        self.min_sentence_len = min_sentence_len

    def split_sentences(self, text):
        """Split text into sentences on . ! ? delimiters."""
        parts = re.split(r"[.!?]", text)
        return [s.strip() for s in parts if s.strip()]

    def align(self, image_urls, caption_text):
        """Align image URLs to caption sentences (round-robin).

        Returns list of dicts: {image_url, sentence, score}.
        Score = len(sentence) / max_sentence_len.
        """
        sentences = self.split_sentences(caption_text)
        if not sentences or not image_urls:
            return []

        max_len = max(len(s) for s in sentences)
        pairs = []
        for i, sentence in enumerate(sentences):
            image_url = image_urls[i % len(image_urls)]
            score = len(sentence) / max_len if max_len > 0 else 0.0
            pairs.append({"image_url": image_url, "sentence": sentence, "score": score})
        return pairs

    def filter_short(self, pairs, min_len=10):
        """Filter out pairs where sentence length < min_len."""
        return [p for p in pairs if len(p.get("sentence", "")) >= min_len]
