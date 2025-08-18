"""Test data fixtures and sample data for comprehensive testing."""

import json
import base64
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image
import pandas as pd

# Sample text data in multiple languages
SAMPLE_TEXTS = {
    'en': [
        "The World Health Organization reports a significant increase in humanitarian needs.",
        "Emergency response teams have been deployed to affected regions.",
        "Access to clean water remains a critical challenge in refugee camps.",
        "Educational programs continue despite funding constraints.",
        "Healthcare infrastructure requires immediate attention and support."
    ],
    'sw': [
        "Shirikisho la Afya Duniani linarihi ongezeko kubwa la mahitaji ya kibinadamu.",
        "Timu za majibu ya dharura zimepelekwa katika maeneo yaliyoathiriwa.",
        "Upatikanaji wa maji safi unabaki changamoto muhimu katika makambi ya wakimbizi.",
        "Mipango ya elimu inaendelea licha ya vikwazo vya fedha.",
        "Miundombinu ya huduma za afya inahitaji msaada wa haraka."
    ],
    'am': [
        "የዓለም ጤና ድርጅት የሰብአዊ ፍላጎቶች ከፍተኛ ጭማሪ እንዳለ አስታውቋል።",
        "የአደጋ ምላሽ ቡድኖች ወደተጎዱ ቦታዎች ተሰማርተዋል።",
        "ንጹህ ውሃ ማግኘት በስደተኞች ካምፖች ውስጥ ትልቅ ፈተና ሆኖ ቀጥሏል።",
        "የትምህርት መርሃ ግብሮች የገንዘብ እጥረት ቢኖርም እንደቀጠሉ ነው።",
        "የጤና አገልግሎት መሠረተ ልማት አስቸኳይ ትኩረትና ድጋፍ ይፈልጋል።"
    ],
    'ha': [
        "Kungiyar Lafiya ta Duniya ta ba da rahoton karuwar bukatar jin kai.",
        "An tura kungiyoyin amsa gaggawa zuwa yankunan da abin ya shafa.",
        "Samun ruwa mai tsabta ya rage babban kalubale a sansanonin 'yan gudun hijira.",
        "Shirye-shiryen ilimi na ci gaba duk da karancin kudade.",
        "Kayayyakin lafiya na bukatar kulawa da tallafi cikin gaggawa."
    ],
    'ar': [
        "تفيد منظمة الصحة العالمية بزيادة كبيرة في الاحتياجات الإنسانية.",
        "تم نشر فرق الاستجابة الطارئة في المناطق المتضررة.",
        "لا يزال الحصول على المياه النظيفة يشكل تحدياً بالغاً في مخيمات اللاجئين.",
        "تستمر البرامج التعليمية رغم القيود المالية.",
        "تتطلب البنية التحتية للرعاية الصحية اهتماماً ودعماً فورياً."
    ]
}

# Sample OCR results with varying confidence levels
SAMPLE_OCR_RESULTS = [
    {
        "text": "HUMANITARIAN SITUATION REPORT",
        "confidence": 0.98,
        "language": "en",
        "bounding_boxes": [
            {"text": "HUMANITARIAN", "bbox": [100, 50, 350, 80], "confidence": 0.99},
            {"text": "SITUATION", "bbox": [360, 50, 520, 80], "confidence": 0.98},
            {"text": "REPORT", "bbox": [530, 50, 650, 80], "confidence": 0.97}
        ]
    },
    {
        "text": "Emergency Response Team Deployment",
        "confidence": 0.92,
        "language": "en",
        "bounding_boxes": [
            {"text": "Emergency", "bbox": [50, 100, 180, 130], "confidence": 0.94},
            {"text": "Response", "bbox": [190, 100, 300, 130], "confidence": 0.93},
            {"text": "Team", "bbox": [310, 100, 380, 130], "confidence": 0.91},
            {"text": "Deployment", "bbox": [390, 100, 550, 130], "confidence": 0.89}
        ]
    },
    {
        "text": "Makambi ya Wakimbizi",
        "confidence": 0.87,
        "language": "sw",
        "bounding_boxes": [
            {"text": "Makambi", "bbox": [75, 150, 200, 180], "confidence": 0.89},
            {"text": "ya", "bbox": [210, 150, 240, 180], "confidence": 0.92},
            {"text": "Wakimbizi", "bbox": [250, 150, 400, 180], "confidence": 0.84}
        ]
    }
]

# Sample document metadata
SAMPLE_DOCUMENTS = [
    {
        "url": "https://reliefweb.int/report/global/humanitarian-needs-2024",
        "title": "Global Humanitarian Needs Overview 2024",
        "source": "unhcr",
        "language": "en",
        "content": SAMPLE_TEXTS['en'][0] + " " + SAMPLE_TEXTS['en'][1],
        "content_type": "pdf",
        "word_count": 156,
        "quality_score": 0.94,
        "date_published": "2024-01-15",
        "metadata": {
            "author": "UNHCR Global Team",
            "pages": 45,
            "file_size": "2.3MB",
            "classification": "public",
            "region": "global",
            "theme": "humanitarian_overview"
        },
        "images": [
            {
                "src": "https://reliefweb.int/sites/default/files/chart1.png",
                "alt": "Global displacement statistics by region",
                "width": 800,
                "height": 600,
                "page": 3,
                "type": "chart",
                "ocr_result": SAMPLE_OCR_RESULTS[0]
            }
        ]
    },
    {
        "url": "https://www.who.int/emergencies/disease-outbreak-news",
        "title": "WHO Disease Outbreak Response Report",
        "source": "who",
        "language": "en",
        "content": SAMPLE_TEXTS['en'][2] + " " + SAMPLE_TEXTS['en'][4],
        "content_type": "pdf",
        "word_count": 89,
        "quality_score": 0.91,
        "date_published": "2024-01-20",
        "metadata": {
            "author": "WHO Emergency Response Team",
            "pages": 23,
            "file_size": "1.8MB",
            "classification": "public",
            "region": "africa",
            "theme": "health_emergency"
        },
        "images": [
            {
                "src": "https://www.who.int/images/emergency-map.png",
                "alt": "Disease outbreak geographical distribution",
                "width": 1000,
                "height": 700,
                "page": 5,
                "type": "map",
                "ocr_result": SAMPLE_OCR_RESULTS[1]
            }
        ]
    },
    {
        "url": "https://data.unicef.org/resources/education-emergency-report",
        "title": "Education in Emergency Situations - East Africa",
        "source": "unicef",
        "language": "en",
        "content": SAMPLE_TEXTS['en'][3],
        "content_type": "pdf",
        "word_count": 67,
        "quality_score": 0.88,
        "date_published": "2024-01-25",
        "metadata": {
            "author": "UNICEF Education Team",
            "pages": 18,
            "file_size": "1.2MB",
            "classification": "public",
            "region": "east_africa",
            "theme": "education"
        },
        "images": [
            {
                "src": "https://data.unicef.org/images/education-infographic.png",
                "alt": "Education access statistics infographic",
                "width": 800,
                "height": 1200,
                "page": 7,
                "type": "infographic",
                "ocr_result": SAMPLE_OCR_RESULTS[2]
            }
        ]
    }
]

# Sample multilingual dataset
SAMPLE_MULTILINGUAL_DATASET = [
    {
        "image_path": "/data/images/unhcr_chart_001.jpg",
        "text": SAMPLE_TEXTS['en'][0],
        "language": "en",
        "source": "unhcr",
        "quality_score": 0.94,
        "metadata": {
            "title": "Global Displacement Chart",
            "date": "2024-01-15",
            "region": "global",
            "chart_type": "bar"
        }
    },
    {
        "image_path": "/data/images/who_map_002.jpg",
        "text": SAMPLE_TEXTS['en'][2],
        "language": "en",
        "source": "who",
        "quality_score": 0.91,
        "metadata": {
            "title": "Health Emergency Map",
            "date": "2024-01-20",
            "region": "africa",
            "map_type": "choropleth"
        }
    },
    {
        "image_path": "/data/images/unicef_infographic_003.jpg",
        "text": SAMPLE_TEXTS['sw'][3],
        "language": "sw",
        "source": "unicef",
        "quality_score": 0.88,
        "metadata": {
            "title": "Takwimu za Elimu",
            "date": "2024-01-25",
            "region": "east_africa",
            "infographic_type": "education_stats"
        }
    },
    {
        "image_path": "/data/images/wfp_chart_004.jpg",
        "text": SAMPLE_TEXTS['am'][0],
        "language": "am",
        "source": "wfp",
        "quality_score": 0.86,
        "metadata": {
            "title": "የምግብ እርዳታ ሪፖርት",
            "date": "2024-01-30",
            "region": "horn_of_africa",
            "chart_type": "pie"
        }
    }
]

# Training examples with instructions
SAMPLE_TRAINING_EXAMPLES = [
    {
        "instruction": "Describe what you see in this humanitarian report chart.",
        "image_description": "A bar chart showing refugee numbers by country",
        "response": "This chart displays refugee population statistics across different countries. The highest numbers are shown for Syria, Afghanistan, and South Sudan, with Syria having approximately 6.8 million refugees."
    },
    {
        "instruction": "What information does this health emergency map convey?",
        "image_description": "A map showing disease outbreak locations with color coding",
        "response": "This map indicates the geographical distribution of disease outbreaks across African regions. Red areas show active outbreak zones, while yellow areas indicate areas under surveillance."
    },
    {
        "instruction": "Eleza kile unachoona katika chati hii ya elimu.",
        "image_description": "An infographic about education statistics in East Africa",
        "response": "Chati hii inaonyesha takwimu za elimu katika Afrika Mashariki. Inaonyesha idadi ya watoto wanaosoma shule, vipimo vya uongozi wa wasichana, na changamoto za miundombinu."
    }
]

# Quality control examples
QUALITY_CONTROL_EXAMPLES = [
    {
        "type": "high_quality",
        "score": 0.95,
        "text": "The World Health Organization has confirmed a significant increase in malnutrition rates among children under five in the affected regions.",
        "issues": [],
        "reasons": ["clear_text", "relevant_content", "good_ocr_confidence", "proper_language_detection"]
    },
    {
        "type": "medium_quality",
        "score": 0.78,
        "text": "Emergency respnse teams have been deployd to provide assistnce in the region.",
        "issues": ["spelling_errors"],
        "reasons": ["minor_ocr_errors", "content_relevant", "language_correct"]
    },
    {
        "type": "low_quality",
        "score": 0.45,
        "text": "Th3 r3p0rt sh0ws th@t m@ny p30pl3 n33d h3lp 1n th3 @r3@.",
        "issues": ["severe_ocr_errors", "unclear_content"],
        "reasons": ["poor_image_quality", "text_corruption", "low_confidence_scores"]
    },
    {
        "type": "rejected",
        "score": 0.25,
        "text": "asdfkjh aslkdfj alskdfj alskdfj",
        "issues": ["no_meaningful_content", "language_detection_failed"],
        "reasons": ["corrupted_text", "no_humanitarian_content", "unintelligible"]
    }
]

# Performance benchmark data
PERFORMANCE_BENCHMARKS = {
    "image_processing": {
        "small_image": {"size": (224, 224), "expected_time": 0.5, "max_memory": 50},
        "medium_image": {"size": (512, 512), "expected_time": 1.2, "max_memory": 100},
        "large_image": {"size": (1024, 1024), "expected_time": 3.0, "max_memory": 200}
    },
    "text_processing": {
        "short_text": {"words": 50, "expected_time": 0.1, "max_memory": 10},
        "medium_text": {"words": 500, "expected_time": 0.5, "max_memory": 25},
        "long_text": {"words": 2000, "expected_time": 1.5, "max_memory": 75}
    },
    "model_inference": {
        "single_inference": {"expected_time": 0.3, "max_memory": 150},
        "batch_inference": {"batch_size": 8, "expected_time": 1.5, "max_memory": 400}
    }
}

# Error scenarios for testing
ERROR_SCENARIOS = [
    {
        "name": "network_timeout",
        "type": "network_error",
        "description": "Simulates network timeout during document download",
        "exception": "requests.exceptions.Timeout",
        "recovery": "retry_with_backoff"
    },
    {
        "name": "invalid_pdf",
        "type": "parsing_error",
        "description": "Simulates corrupted PDF file",
        "exception": "PyPDF2.utils.PdfReadError",
        "recovery": "skip_document"
    },
    {
        "name": "ocr_failure",
        "type": "processing_error",
        "description": "Simulates OCR engine failure",
        "exception": "RuntimeError",
        "recovery": "fallback_ocr_engine"
    },
    {
        "name": "model_oom",
        "type": "memory_error",
        "description": "Simulates out-of-memory during model inference",
        "exception": "torch.cuda.OutOfMemoryError",
        "recovery": "reduce_batch_size"
    },
    {
        "name": "storage_full",
        "type": "storage_error",
        "description": "Simulates storage space exhaustion",
        "exception": "OSError",
        "recovery": "cleanup_temporary_files"
    }
]


class TestDataGenerator:
    """Generate test data for various testing scenarios."""
    
    @staticmethod
    def create_test_image_with_text(text: str, width: int = 400, height: int = 200) -> Image.Image:
        """Create a test image with embedded text for OCR testing."""
        # Create white background
        img = Image.new('RGB', (width, height), color='white')
        
        # Add some visual elements to make it look like a document
        img_array = np.array(img)
        
        # Add text regions (simplified as white rectangles)
        text_lines = text.split('. ')
        y_offset = 30
        for line in text_lines[:3]:  # Max 3 lines
            if y_offset < height - 20:
                img_array[y_offset:y_offset+20, 20:width-20] = [240, 240, 240]  # Light gray for text
                y_offset += 40
        
        return Image.fromarray(img_array)
    
    @staticmethod
    def create_test_pdf_bytes(content: str = "Test PDF content") -> bytes:
        """Create minimal PDF bytes for testing."""
        # Simplified PDF structure
        pdf_content = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj

2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources << /Font << /F1 5 0 R >> >>
>>
endobj

4 0 obj
<< /Length {len(content) + 20} >>
stream
BT
/F1 12 Tf
72 720 Td
({content}) Tj
ET
endstream
endobj

5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
456
%%EOF"""
        return pdf_content.encode('utf-8')
    
    @staticmethod
    def create_mock_api_responses() -> Dict[str, Any]:
        """Create mock API responses for different humanitarian sources."""
        return {
            "unhcr": {
                "documents": [
                    {
                        "id": "unhcr_001",
                        "title": "Global Trends Report 2024",
                        "url": "https://www.unhcr.org/reports/global-trends-2024.pdf",
                        "date": "2024-06-20",
                        "language": "en",
                        "summary": "Annual report on forced displacement worldwide"
                    }
                ]
            },
            "who": {
                "articles": [
                    {
                        "id": "who_001",
                        "headline": "Health Emergency Update",
                        "url": "https://www.who.int/emergencies/update-2024.pdf",
                        "date": "2024-05-15",
                        "language": "en",
                        "body": "Latest updates on global health emergencies"
                    }
                ]
            }
        }


def get_sample_data(data_type: str) -> Any:
    """Get sample data by type."""
    data_map = {
        'texts': SAMPLE_TEXTS,
        'ocr_results': SAMPLE_OCR_RESULTS,
        'documents': SAMPLE_DOCUMENTS,
        'dataset': SAMPLE_MULTILINGUAL_DATASET,
        'training_examples': SAMPLE_TRAINING_EXAMPLES,
        'quality_examples': QUALITY_CONTROL_EXAMPLES,
        'benchmarks': PERFORMANCE_BENCHMARKS,
        'error_scenarios': ERROR_SCENARIOS
    }
    
    return data_map.get(data_type, {})


def save_test_fixtures(output_dir: Path):
    """Save all test fixtures to files for reuse."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fixtures = {
        'sample_texts.json': SAMPLE_TEXTS,
        'sample_ocr_results.json': SAMPLE_OCR_RESULTS,
        'sample_documents.json': SAMPLE_DOCUMENTS,
        'sample_dataset.json': SAMPLE_MULTILINGUAL_DATASET,
        'training_examples.json': SAMPLE_TRAINING_EXAMPLES,
        'quality_examples.json': QUALITY_CONTROL_EXAMPLES,
        'performance_benchmarks.json': PERFORMANCE_BENCHMARKS,
        'error_scenarios.json': ERROR_SCENARIOS
    }
    
    for filename, data in fixtures.items():
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Save fixtures when run as script
    save_test_fixtures(Path(__file__).parent / "data")
    print("Test fixtures saved to tests/fixtures/data/")