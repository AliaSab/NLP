"""
Скрипт для предварительной загрузки модели перевода для back translation.
Модель загружается один раз и затем кэшируется локально.
"""

import logging
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_translation_model(language: str = 'ru', model_name: str = None):
    """
    Загрузка модели перевода для back translation.
    
    Args:
        language: Язык ('ru' для русского)
        model_name: Имя модели (если None, используется по умолчанию)
    
    Returns:
        Загруженный pipeline
    """
    if model_name is None:
        if language == 'ru':
            model_name = "Helsinki-NLP/opus-mt-ru-en"
        else:
            model_name = f"Helsinki-NLP/opus-mt-{language}-en"
    
    logger.info(f"Загрузка модели перевода: {model_name}")
    logger.info("Это может занять несколько минут при первом запуске...")
    logger.info("Модель будет сохранена локально для последующего использования")
    
    try:
        # Загрузка модели (автоматически кэшируется в ~/.cache/huggingface/)
        translation_pipeline = pipeline(
            "translation",
            model=model_name,
            device=-1  # CPU
        )
        
        logger.info("✓ Модель успешно загружена и сохранена!")
        logger.info(f"Модель сохранена в: ~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
        
        return translation_pipeline
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        logger.info("\nАльтернативные варианты:")
        logger.info("1. Используйте другие методы аугментации (synonym_replacement, EDA)")
        logger.info("2. Загрузите модель вручную через huggingface-cli:")
        logger.info(f"   huggingface-cli download {model_name}")
        logger.info("3. Используйте VPN или другое интернет-соединение")
        raise


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Загрузка модели перевода для back translation')
    parser.add_argument('--language', type=str, default='ru', help='Язык (ru, en, etc.)')
    parser.add_argument('--model', type=str, default=None, help='Имя модели (опционально)')
    
    args = parser.parse_args()
    
    try:
        pipeline = download_translation_model(args.language, args.model)
        print("\n✓ Модель готова к использованию!")
        print("Теперь back translation будет работать без задержек.")
    except Exception as e:
        print(f"\n✗ Не удалось загрузить модель: {e}")
        print("Back translation будет недоступен, но другие методы аугментации работают.")

















