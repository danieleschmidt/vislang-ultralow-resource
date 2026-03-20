"""DatasetStats: compute and report dataset statistics."""


class DatasetStats:
    """Compute and report statistics for a visual-language dataset."""

    def compute(self, pairs, language_field="language"):
        """Compute stats over pairs. Returns dict with required keys."""
        n_pairs = len(pairs)
        texts = []
        image_urls = set()
        lang_dist = {}

        for pair in pairs:
            text = pair.get("text", pair.get("sentence", pair.get("summary", ""))) or ""
            texts.append(text)
            img = pair.get("image_url", pair.get("url", ""))
            if img:
                image_urls.add(img)
            lang = pair.get(language_field, "unknown") or "unknown"
            lang_dist[lang] = lang_dist.get(lang, 0) + 1

        vocab = self.build_vocab(texts)
        avg_text_len = sum(len(t) for t in texts) / len(texts) if texts else 0.0

        return {
            "n_pairs": n_pairs,
            "vocab_size": len(vocab),
            "image_count": len(image_urls),
            "language_distribution": lang_dist,
            "avg_text_len": avg_text_len,
        }

    def build_vocab(self, texts):
        """Build vocabulary set from texts (lowercase, whitespace-split)."""
        vocab = set()
        for text in texts:
            vocab.update(text.lower().split())
        return vocab

    def print_report(self, stats):
        """Print a formatted statistics summary."""
        print("=" * 40)
        print("Dataset Statistics Report")
        print("=" * 40)
        print(f"  Total pairs:     {stats.get('n_pairs', 0)}")
        print(f"  Vocabulary size: {stats.get('vocab_size', 0)}")
        print(f"  Unique images:   {stats.get('image_count', 0)}")
        print(f"  Avg text length: {stats.get('avg_text_len', 0):.1f} chars")
        print("  Language distribution:")
        for lang, count in stats.get("language_distribution", {}).items():
            print(f"    {lang}: {count}")
        print("=" * 40)
