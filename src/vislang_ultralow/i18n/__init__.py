"""Internationalization and localization support."""

import os
import json
import logging
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported languages for humanitarian AI
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'fr': 'Français', 
    'es': 'Español',
    'ar': 'العربية',
    'pt': 'Português',
    'de': 'Deutsch',
    'it': 'Italiano',
    'ru': 'Русский',
    'zh': '中文',
    'sw': 'Kiswahili',
    'am': 'አማርኛ',
    'ha': 'Hausa',
    'yo': 'Yorùbá',
    'ig': 'Igbo', 
    'zu': 'isiZulu',
    'so': 'Soomaali',
    'ti': 'ትግርኛ',
    'om': 'Afaan Oromoo',
    'hi': 'हिन्दी',
    'ur': 'اردو',
    'bn': 'বাংলা',
    'ta': 'தமிழ்',
    'te': 'తెలుగు',
    'ml': 'മലയാളം',
    'kn': 'ಕನ್ನಡ',
    'gu': 'ગુજરાતી',
    'pa': 'ਪੰਜਾਬੀ',
    'fa': 'فارسی',
    'ps': 'پښتو',
    'ku': 'Kurdî',
    'tr': 'Türkçe',
    'az': 'Azərbaycan',
    'uz': 'Oʻzbek',
    'kk': 'Қазақ',
    'ky': 'Кыргыз',
    'tg': 'Тоҷикӣ',
    'my': 'မြန်မာ',
    'th': 'ไทย',
    'vi': 'Tiếng Việt',
    'km': 'ខ្មែរ',
    'lo': 'ລາວ',
    'si': 'සිංහල',
    'ne': 'नेपाली',
    'dz': 'རྫོང་ཁ',
    'bo': 'བོད་ཡིག'
}

# Default translations
DEFAULT_TRANSLATIONS = {
    'en': {
        'app_name': 'VisLang-UltraLow-Resource',
        'processing': 'Processing...',
        'error': 'Error',
        'success': 'Success',
        'scraping_documents': 'Scraping humanitarian documents',
        'building_dataset': 'Building vision-language dataset',
        'training_model': 'Training vision-language model',
        'completed': 'Completed successfully',
        'failed': 'Operation failed',
        'invalid_language': 'Invalid language code',
        'quality_check': 'Quality check',
        'high_quality': 'High quality',
        'medium_quality': 'Medium quality', 
        'low_quality': 'Low quality',
        'documents_found': '{count} documents found',
        'processing_images': 'Processing {count} images',
        'ocr_confidence': 'OCR confidence: {confidence:.1%}',
        'model_accuracy': 'Model accuracy: {accuracy:.1%}',
    },
    'fr': {
        'app_name': 'VisLang-Ressources-Ultra-Faibles',
        'processing': 'Traitement en cours...',
        'error': 'Erreur',
        'success': 'Succès',
        'scraping_documents': 'Extraction de documents humanitaires',
        'building_dataset': 'Construction du jeu de données vision-langage',
        'training_model': 'Entraînement du modèle vision-langage',
        'completed': 'Terminé avec succès',
        'failed': 'Opération échouée',
        'invalid_language': 'Code de langue invalide',
        'quality_check': 'Contrôle qualité',
        'high_quality': 'Haute qualité',
        'medium_quality': 'Qualité moyenne',
        'low_quality': 'Faible qualité',
        'documents_found': '{count} documents trouvés',
        'processing_images': 'Traitement de {count} images',
        'ocr_confidence': 'Confiance OCR: {confidence:.1%}',
        'model_accuracy': 'Précision du modèle: {accuracy:.1%}',
    },
    'es': {
        'app_name': 'VisLang-Recursos-Ultra-Bajos',
        'processing': 'Procesando...',
        'error': 'Error',
        'success': 'Éxito',
        'scraping_documents': 'Extrayendo documentos humanitarios',
        'building_dataset': 'Construyendo conjunto de datos visión-lenguaje',
        'training_model': 'Entrenando modelo visión-lenguaje',
        'completed': 'Completado exitosamente',
        'failed': 'Operación fallida',
        'invalid_language': 'Código de idioma inválido',
        'quality_check': 'Control de calidad',
        'high_quality': 'Alta calidad',
        'medium_quality': 'Calidad media',
        'low_quality': 'Baja calidad',
        'documents_found': '{count} documentos encontrados',
        'processing_images': 'Procesando {count} imágenes',
        'ocr_confidence': 'Confianza OCR: {confidence:.1%}',
        'model_accuracy': 'Precisión del modelo: {accuracy:.1%}',
    },
    'ar': {
        'app_name': 'VisLang-موارد-فائقة-الندرة',
        'processing': 'جاري المعالجة...',
        'error': 'خطأ',
        'success': 'نجح',
        'scraping_documents': 'استخراج الوثائق الإنسانية',
        'building_dataset': 'بناء مجموعة بيانات الرؤية واللغة',
        'training_model': 'تدريب نموذج الرؤية واللغة',
        'completed': 'اكتمل بنجاح',
        'failed': 'فشلت العملية',
        'invalid_language': 'رمز لغة غير صالح',
        'quality_check': 'فحص الجودة',
        'high_quality': 'جودة عالية',
        'medium_quality': 'جودة متوسطة',
        'low_quality': 'جودة منخفضة',
        'documents_found': 'تم العثور على {count} وثيقة',
        'processing_images': 'معالجة {count} صورة',
        'ocr_confidence': 'ثقة OCR: {confidence:.1%}',
        'model_accuracy': 'دقة النموذج: {accuracy:.1%}',
    }
}


class Translator:
    """Simple translator for humanitarian AI applications."""
    
    def __init__(self, language: str = 'en', translations_dir: Optional[Path] = None):
        """Initialize translator.
        
        Args:
            language: Target language code
            translations_dir: Directory containing translation files
        """
        self.language = language
        self.translations_dir = translations_dir
        self.translations = {}
        
        # Load default translations
        self.translations.update(DEFAULT_TRANSLATIONS)
        
        # Load custom translations if directory provided
        if translations_dir and translations_dir.exists():
            self._load_custom_translations()
        
        # Validate language
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Language '{language}' not in supported languages, falling back to English")
            self.language = 'en'
    
    def _load_custom_translations(self):
        """Load custom translations from directory."""
        try:
            for lang_file in self.translations_dir.glob("*.json"):
                lang_code = lang_file.stem
                with open(lang_file, 'r', encoding='utf-8') as f:
                    custom_translations = json.load(f)
                
                if lang_code not in self.translations:
                    self.translations[lang_code] = {}
                
                self.translations[lang_code].update(custom_translations)
                
            logger.info(f"Loaded custom translations for {len(self.translations)} languages")
            
        except Exception as e:
            logger.error(f"Error loading custom translations: {e}")
    
    def t(self, key: str, **kwargs) -> str:
        """Translate a key with optional formatting.
        
        Args:
            key: Translation key
            **kwargs: Format arguments
            
        Returns:
            Translated string
        """
        # Get translation for current language
        lang_translations = self.translations.get(self.language, {})
        
        # Fallback to English if not found
        if key not in lang_translations:
            lang_translations = self.translations.get('en', {})
        
        # Get translation or use key as fallback
        translation = lang_translations.get(key, key)
        
        # Apply formatting if kwargs provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Translation formatting error for key '{key}': {e}")
        
        return translation
    
    def set_language(self, language: str):
        """Change current language.
        
        Args:
            language: New language code
        """
        if language in SUPPORTED_LANGUAGES:
            self.language = language
            logger.info(f"Language changed to {SUPPORTED_LANGUAGES[language]} ({language})")
        else:
            logger.warning(f"Language '{language}' not supported")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get all supported languages.
        
        Returns:
            Dictionary of language codes to names
        """
        return SUPPORTED_LANGUAGES.copy()
    
    def get_available_translations(self) -> List[str]:
        """Get languages with available translations.
        
        Returns:
            List of language codes with translations
        """
        return list(self.translations.keys())


# Global translator instance
_translator = None


def init_i18n(language: str = 'en', translations_dir: Optional[Path] = None):
    """Initialize global translator.
    
    Args:
        language: Default language
        translations_dir: Custom translations directory
    """
    global _translator
    _translator = Translator(language, translations_dir)
    logger.info(f"Internationalization initialized with language: {language}")


def get_translator() -> Translator:
    """Get global translator instance.
    
    Returns:
        Global translator instance
    """
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator


def t(key: str, **kwargs) -> str:
    """Translate using global translator.
    
    Args:
        key: Translation key
        **kwargs: Format arguments
        
    Returns:
        Translated string
    """
    return get_translator().t(key, **kwargs)


def set_language(language: str):
    """Set global language.
    
    Args:
        language: Language code
    """
    get_translator().set_language(language)


# Auto-detect system language
def detect_system_language() -> str:
    """Detect system language from environment.
    
    Returns:
        Detected language code or 'en' as fallback
    """
    # Check environment variables
    for env_var in ['LANG', 'LANGUAGE', 'LC_ALL', 'LC_MESSAGES']:
        lang = os.environ.get(env_var)
        if lang:
            # Extract language code (e.g., 'en_US.UTF-8' -> 'en')
            lang_code = lang.split('_')[0].split('.')[0].lower()
            if lang_code in SUPPORTED_LANGUAGES:
                return lang_code
            break
    
    return 'en'  # Default fallback


# Initialize with detected language on import
if _translator is None:
    detected_lang = detect_system_language()
    init_i18n(detected_lang)