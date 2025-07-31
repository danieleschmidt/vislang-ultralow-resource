"""Test version and basic imports."""

import vislang_ultralow


def test_version():
    """Test that version is defined."""
    assert vislang_ultralow.__version__ == "0.1.0"


def test_imports():
    """Test that main classes can be imported."""
    from vislang_ultralow import DatasetBuilder, HumanitarianScraper, VisionLanguageTrainer

    assert DatasetBuilder is not None
    assert HumanitarianScraper is not None
    assert VisionLanguageTrainer is not None