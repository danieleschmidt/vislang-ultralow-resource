"""OCRPipeline: pytesseract wrapper with graceful fallback."""


class OCRPipeline:
    """Text extraction pipeline with pytesseract fallback."""

    def __init__(self, lang="eng", fallback_text="[OCR unavailable]"):
        self.lang = lang
        self.fallback_text = fallback_text

    def is_available(self):
        """Return True if pytesseract is importable."""
        try:
            import pytesseract  # noqa: F401
            return True
        except (ImportError, Exception):
            return False

    def extract_text(self, image_path_or_array):
        """Extract text from an image. Falls back to fallback_text on failure."""
        try:
            import pytesseract
            return pytesseract.image_to_string(image_path_or_array, lang=self.lang)
        except (ImportError, Exception):
            return self.fallback_text + " " + str(image_path_or_array)[:50]

    def extract_batch(self, image_list):
        """Extract text from a list of images. Returns list of strings."""
        return [self.extract_text(img) for img in image_list]
