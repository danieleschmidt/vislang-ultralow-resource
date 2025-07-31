# VisLang-UltraLow-Resource

Dataset builder and training framework for visual-language models in ultra-low-resource languages. Scrapes humanitarian reports, OCRs infographics, and creates aligned vision-language instruction datasets following the Masakhane visual-instruction corpus approach (2025).

## Overview

VisLang-UltraLow-Resource addresses the critical need for vision-language AI in underserved languages by automating the creation of multimodal datasets from humanitarian and development organization reports. The framework handles complex document layouts, multilingual OCR, and cross-lingual alignment to build instruction-following datasets for languages with limited digital resources.

## Key Features

- **Humanitarian Report Processing**: Automated extraction from UN, WHO, NGO reports
- **Multilingual OCR**: Support for 100+ languages including low-resource scripts
- **Infographic Understanding**: Extract and align data from charts, maps, and diagrams
- **Cross-Lingual Alignment**: English-to-target language instruction generation
- **Quality Assurance**: Automated and human-in-the-loop verification
- **Model Training**: Fine-tune vision-language models for low-resource languages

## Installation

```bash
# Basic installation
pip install vislang-ultralow-resource

# With all OCR engines
pip install vislang-ultralow-resource[ocr]

# With model training support
pip install vislang-ultralow-resource[training]

# Development installation
git clone https://github.com/yourusername/vislang-ultralow-resource
cd vislang-ultralow-resource
pip install -e ".[dev]"
```

## Quick Start

### Basic Dataset Creation

```python
from vislang_ultralow import DatasetBuilder
from vislang_ultralow.sources import HumanitarianScraper

# Initialize scraper for humanitarian reports
scraper = HumanitarianScraper(
    sources=["unhcr", "who", "unicef", "wfp"],
    languages=["sw", "am", "ha"],  # Swahili, Amharic, Hausa
    date_range=("2020-01-01", "2025-01-01")
)

# Build dataset
builder = DatasetBuilder(
    target_languages=["sw", "am", "ha"],
    source_language="en",
    min_quality_score=0.8
)

# Create visual-language dataset
dataset = builder.build(
    scraper=scraper,
    include_infographics=True,
    include_maps=True,
    include_charts=True,
    output_format="hf_dataset"
)

print(f"Created dataset with {len(dataset)} examples")
```

### Training Vision-Language Model

```python
from vislang_ultralow import VisionLanguageTrainer
from transformers import AutoProcessor, AutoModelForVision2Seq

# Load multilingual base model
model = AutoModelForVision2Seq.from_pretrained("facebook/mblip-mt0-xl")
processor = AutoProcessor.from_pretrained("facebook/mblip-mt0-xl")

# Initialize trainer
trainer = VisionLanguageTrainer(
    model=model,
    processor=processor,
    languages=["sw", "am", "ha"],
    instruction_style="natural"
)

# Fine-tune on created dataset
trainer.train(
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    num_epochs=10,
    learning_rate=5e-5,
    warmup_steps=
