"""Sample data fixtures for testing VisLang-UltraLow-Resource."""

import json
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
import numpy as np


def create_sample_humanitarian_documents() -> List[Dict[str, Any]]:
    """Create sample humanitarian document data for testing."""
    return [
        {
            "id": "unhcr_report_001",
            "title": "Global Trends: Forced Displacement in 2023",
            "url": "https://www.unhcr.org/reports/global-trends-2023.pdf",
            "source": "unhcr",
            "date": "2023-06-15",
            "language": "en",
            "pages": 120,
            "topics": ["displacement", "refugees", "statistics"],
            "regions": ["global"],
            "content_type": "application/pdf",
            "file_size": "15.2MB",
            "abstract": "Comprehensive report on global forced displacement trends and statistics for 2023."
        },
        {
            "id": "who_health_002",
            "title": "Health Emergency Response: East Africa Crisis",
            "url": "https://www.who.int/emergencies/east-africa-health-response.pdf",
            "source": "who",
            "date": "2023-08-22",
            "language": "en",
            "pages": 45,
            "topics": ["health", "emergency_response", "malnutrition"],
            "regions": ["east_africa", "somalia", "ethiopia", "kenya"],
            "content_type": "application/pdf",
            "file_size": "8.7MB",
            "abstract": "Health situation analysis and response plan for the East Africa humanitarian crisis."
        },
        {
            "id": "unicef_education_003",
            "title": "Education in Emergency: Children's Learning Continuity",
            "url": "https://www.unicef.org/education-emergency-2023.pdf",
            "source": "unicef",
            "date": "2023-09-10",
            "language": "en",
            "pages": 67,
            "topics": ["education", "children", "emergency_response"],
            "regions": ["syria", "yemen", "afghanistan"],
            "content_type": "application/pdf",
            "file_size": "12.3MB",
            "abstract": "Analysis of educational needs and interventions for children in humanitarian crises."
        },
        {
            "id": "wfp_food_004",
            "title": "Food Security Monitoring Bulletin",
            "url": "https://www.wfp.org/food-security-bulletin-q3-2023.pdf",
            "source": "wfp",
            "date": "2023-10-05",
            "language": "en",
            "pages": 28,
            "topics": ["food_security", "nutrition", "monitoring"],
            "regions": ["sahel", "horn_of_africa", "central_america"],
            "content_type": "application/pdf",
            "file_size": "5.1MB",
            "abstract": "Quarterly monitoring report on food security indicators in crisis-affected regions."
        }
    ]


def create_sample_multilingual_content() -> List[Dict[str, Any]]:
    """Create sample multilingual content for testing."""
    return [
        {
            "text_en": "Emergency health services are being provided to refugee populations.",
            "text_sw": "Huduma za afya za dharura zinapatikana kwa wakimbizi.",
            "text_am": "የአደጋ ጊዜ የጤና አገልግሎቶች ለስደተኞች እየተሰጡ ናቸው።",
            "text_ar": "يتم تقديم خدمات الصحة الطارئة للاجئين.",
            "topic": "health_services",
            "context": "refugee_assistance"
        },
        {
            "text_en": "Educational materials are distributed to children in displacement camps.",
            "text_sw": "Vifaa vya elimu vinagatwa kwa watoto katika kambi za wakimbizi.",
            "text_am": "የትምህርት ቁሳቁሶች በስደተኛ ካምፖች ውስጥ ላሉ ልጆች እየተሰጡ ናቸው።",
            "text_ar": "يتم توزيع المواد التعليمية على الأطفال في مخيمات النزوح.",
            "topic": "education",
            "context": "child_protection"
        },
        {
            "text_en": "Food assistance reaches 10,000 families this month.",
            "text_sw": "Msaada wa chakula unafikia familia 10,000 mwezi huu.",
            "text_am": "የምግብ እርዳታ በዚህ ወር 10,000 ቤተሰቦችን ደርሷል።",
            "text_ar": "تصل المساعدات الغذائية إلى 10,000 أسرة هذا الشهر.",
            "topic": "food_assistance",
            "context": "humanitarian_aid"
        }
    ]


def create_sample_ocr_results() -> List[Dict[str, Any]]:
    """Create sample OCR results for testing."""
    return [
        {
            "image_id": "chart_001",
            "ocr_results": {
                "tesseract": {
                    "text": "Refugee Population by Region\nEast Africa: 4.2M\nWest Africa: 2.1M\nCentral Africa: 1.8M",
                    "confidence": 0.87,
                    "language": "en",
                    "processing_time": 1.2
                },
                "easyocr": {
                    "text": "Refugee Population by Region\nEast Africa: 4.2M\nWest Africa: 2.1M\nCentral Africa: 1.8M",
                    "confidence": 0.94,
                    "language": "en",
                    "processing_time": 2.1
                },
                "paddleocr": {
                    "text": "Refugee Population by Region\nEast Africa: 4.2M\nWest Africa: 2.1M\nCentral Africa: 1.8M",
                    "confidence": 0.91,
                    "language": "en",
                    "processing_time": 1.8
                }
            },
            "consensus_result": {
                "text": "Refugee Population by Region\nEast Africa: 4.2M\nWest Africa: 2.1M\nCentral Africa: 1.8M",
                "confidence": 0.91,
                "language": "en",
                "agreement_score": 0.95
            }
        },
        {
            "image_id": "infographic_002",
            "ocr_results": {
                "tesseract": {
                    "text": "Health Indicators\nMalnutrition Rate: 15%\nVaccination Coverage: 78%\nWater Access: 65%",
                    "confidence": 0.82,
                    "language": "en",
                    "processing_time": 1.5
                },
                "easyocr": {
                    "text": "Health Indicators\nMalnutrition Rate: 15%\nVaccination Coverage: 78%\nWater Access: 65%",
                    "confidence": 0.89,
                    "language": "en",
                    "processing_time": 2.3
                }
            },
            "consensus_result": {
                "text": "Health Indicators\nMalnutrition Rate: 15%\nVaccination Coverage: 78%\nWater Access: 65%",
                "confidence": 0.86,
                "language": "en",
                "agreement_score": 0.92
            }
        }
    ]


def create_sample_instruction_templates() -> List[Dict[str, Any]]:
    """Create sample instruction templates for testing."""
    return [
        {
            "template_id": "describe_image",
            "instruction": "Describe what you see in this image.",
            "category": "general_description",
            "languages": ["en", "sw", "am", "ar"],
            "translations": {
                "sw": "Eleza unachoona katika picha hii.",
                "am": "በዚህ ምስል ውስጥ የሚታየውን ይግለጹ።",
                "ar": "صف ما تراه في هذه الصورة."
            }
        },
        {
            "template_id": "extract_statistics",
            "instruction": "What statistics or numbers are shown in this document?",
            "category": "data_extraction",
            "languages": ["en", "sw", "am", "ar"],
            "translations": {
                "sw": "Ni takwimu au nambari gani zinaonyeshwa katika hati hii?",
                "am": "በዚህ ሰነድ ውስጥ የተሳሉት ስታቲስቲክስ ወይም ቁጥሮች ምንድናቸው?",
                "ar": "ما هي الإحصائيات أو الأرقام الموضحة في هذه الوثيقة؟"
            }
        },
        {
            "template_id": "identify_topic",
            "instruction": "What is the main topic or theme of this humanitarian document?",
            "category": "topic_identification",
            "languages": ["en", "sw", "am", "ar"],
            "translations": {
                "sw": "Ni mada gani kuu ya hati hii ya kibinadamu?",
                "am": "የዚህ የሰብአዊ ድርጅት ሰነድ ዋና ርዕሰ ጉዳይ ምንድን ነው?",
                "ar": "ما هو الموضوع الرئيسي لهذه الوثيقة الإنسانية؟"
            }
        }
    ]


def create_sample_quality_metrics() -> Dict[str, Any]:
    """Create sample quality assessment metrics."""
    return {
        "text_quality": {
            "readability_score": 0.85,
            "grammar_score": 0.92,
            "coherence_score": 0.88,
            "completeness_score": 0.91
        },
        "ocr_quality": {
            "confidence_score": 0.89,
            "character_accuracy": 0.94,
            "word_accuracy": 0.91,
            "layout_preservation": 0.87
        },
        "image_quality": {
            "resolution_score": 0.92,
            "clarity_score": 0.88,
            "contrast_score": 0.85,
            "noise_level": 0.12
        },
        "multilingual_quality": {
            "translation_accuracy": 0.87,
            "cultural_appropriateness": 0.90,
            "terminology_consistency": 0.89,
            "script_handling": 0.93
        },
        "overall_quality": 0.89
    }


def create_sample_training_data() -> List[Dict[str, Any]]:
    """Create sample training data for vision-language models."""
    return [
        {
            "image_id": "humanitarian_chart_001",
            "instruction": "What does this chart show about refugee populations?",
            "response": "This chart displays refugee population statistics by region, showing East Africa has the highest number at 4.2 million, followed by West Africa at 2.1 million and Central Africa at 1.8 million refugees.",
            "language": "en",
            "source": "unhcr",
            "topic": "displacement_statistics",
            "quality_score": 0.94
        },
        {
            "image_id": "health_infographic_002",
            "instruction": "Describe the health indicators shown in this infographic.",
            "response": "The infographic presents key health indicators including a malnutrition rate of 15%, vaccination coverage at 78%, and water access at 65% for the affected population.",
            "language": "en",
            "source": "who",
            "topic": "health_indicators",
            "quality_score": 0.91
        },
        {
            "image_id": "education_map_003",
            "instruction": "What information does this map convey about education access?",
            "response": "This map illustrates education access levels across different regions, with color coding indicating areas where children have limited or no access to schooling due to conflict and displacement.",
            "language": "en",
            "source": "unicef",
            "topic": "education_access",
            "quality_score": 0.87
        }
    ]


def create_sample_evaluation_metrics() -> Dict[str, Any]:
    """Create sample evaluation metrics for model performance."""
    return {
        "bleu_scores": {
            "bleu_1": 0.65,
            "bleu_2": 0.58,
            "bleu_3": 0.52,
            "bleu_4": 0.47
        },
        "rouge_scores": {
            "rouge_1": 0.71,
            "rouge_2": 0.63,
            "rouge_l": 0.68
        },
        "bert_score": {
            "precision": 0.82,
            "recall": 0.79,
            "f1": 0.80
        },
        "custom_metrics": {
            "humanitarian_domain_accuracy": 0.86,
            "multilingual_consistency": 0.83,
            "factual_accuracy": 0.88,
            "cultural_sensitivity": 0.91
        },
        "per_language_performance": {
            "en": {"bleu_4": 0.52, "rouge_l": 0.71},
            "sw": {"bleu_4": 0.43, "rouge_l": 0.65},
            "am": {"bleu_4": 0.38, "rouge_l": 0.61},
            "ar": {"bleu_4": 0.41, "rouge_l": 0.63}
        }
    }


def save_sample_data_to_files(output_dir: Path) -> None:
    """Save all sample data to JSON files for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all sample data
    data_files = {
        "humanitarian_documents.json": create_sample_humanitarian_documents(),
        "multilingual_content.json": create_sample_multilingual_content(),
        "ocr_results.json": create_sample_ocr_results(),
        "instruction_templates.json": create_sample_instruction_templates(),
        "quality_metrics.json": create_sample_quality_metrics(),
        "training_data.json": create_sample_training_data(),
        "evaluation_metrics.json": create_sample_evaluation_metrics()
    }
    
    for filename, data in data_files.items():
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def create_sample_images(output_dir: Path) -> List[Path]:
    """Create sample test images with realistic humanitarian document content."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    
    # Create different types of sample images
    image_configs = [
        {
            "name": "refugee_statistics_chart.jpg",
            "size": (800, 600),
            "content": "chart"
        },
        {
            "name": "health_infographic.jpg",
            "size": (600, 800),
            "content": "infographic"
        },
        {
            "name": "education_map.jpg",
            "size": (1024, 768),
            "content": "map"
        },
        {
            "name": "document_page.jpg",
            "size": (612, 792),  # Standard letter size
            "content": "document"
        }
    ]
    
    for config in image_configs:
        # Create a simple but realistic-looking image
        img_array = np.ones((*config["size"][::-1], 3), dtype=np.uint8) * 255
        
        # Add some structure based on content type
        if config["content"] == "chart":
            # Add chart-like elements
            img_array[50:100, 50:750] = [200, 200, 255]  # Light blue header
            img_array[150:200, 100:300] = [255, 200, 200]  # Red bar
            img_array[150:250, 350:550] = [200, 255, 200]  # Green bar
            
        elif config["content"] == "infographic":
            # Add infographic-like elements
            img_array[100:150, 50:550] = [255, 255, 200]  # Yellow section
            img_array[200:250, 50:550] = [200, 255, 255]  # Cyan section
            
        elif config["content"] == "document":
            # Add document-like text lines
            for i in range(5, 25):
                y_pos = 50 + i * 30
                if y_pos < config["size"][1] - 50:
                    img_array[y_pos:y_pos+20, 50:500] = [240, 240, 240]
        
        # Convert to PIL Image and save
        image = Image.fromarray(img_array)
        image_path = output_dir / config["name"]
        image.save(image_path)
        image_paths.append(image_path)
    
    return image_paths


if __name__ == "__main__":
    # Create sample data when run directly
    fixtures_dir = Path(__file__).parent
    data_dir = fixtures_dir / "data"
    images_dir = fixtures_dir / "images"
    
    save_sample_data_to_files(data_dir)
    create_sample_images(images_dir)
    
    print(f"Sample data created in {fixtures_dir}")
    print(f"Data files: {data_dir}")
    print(f"Sample images: {images_dir}")