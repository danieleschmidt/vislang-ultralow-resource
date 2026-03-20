# VisLang Ultra-Low Resource

Dataset builder and training framework for visual-language models in ultra-low-resource languages.

## Features
- HumanitarianReportScraper: Fetch UN OCHA humanitarian reports via ReliefWeb API
- OCRPipeline: pytesseract-based text extraction with graceful fallback
- AlignmentBuilder: Pair image regions with caption sentences via word overlap
- DatasetExporter: Export datasets in HuggingFace-compatible JSONL format
- DatasetStats: Vocabulary, image count, and language distribution statistics

## Install
```
pip install numpy requests
# Optional: pip install pytesseract Pillow langdetect
```

## Usage
```python
from vislang.scraper import HumanitarianReportScraper
from vislang.stats import DatasetStats

scraper = HumanitarianReportScraper()
reports = scraper.fetch_reports(query="flood response", limit=5)
stats = DatasetStats()
result = stats.compute([{"caption": r["body"], "image_path": ""} for r in reports])
print(stats.report(result))
```

## Run Tests
```
~/anaconda3/bin/python3 -m pytest tests/ -v
```
