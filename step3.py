#!/usr/bin/env python3
"""
Этап 3: Проектирование универсального модуля предобработки
Замена числительных, URL, email на токены, расшифровка сокращений
"""
import os
import sys
import json
import logging
from universal_preprocessor import UniversalPreprocessor, PreprocessingConfig

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_articles(filename: str = "kommersant_articles_cleaned.jsonl"):
    """Загрузка очищенных статей"""
    articles = []
    
    if not os.path.exists(filename):
        logger.error(f"Файл {filename} не найден!")
        return []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                article = json.loads(line.strip())
                articles.append(article)
            except json.JSONDecodeError as e:
                logger.warning(f"Ошибка парсинга JSON: {e}")
                continue
    
    logger.info(f"Загружено {len(articles)} статей из {filename}")
    return articles

def save_articles(articles, filename: str):
    """Сохранение обработанных статей"""
    with open(filename, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')
    
    logger.info(f"Сохранено {len(articles)} статей в {filename}")

def main():
    """Этап 3: Универсальная предобработка"""
    print("="*60)
    print("ЭТАП 3: ПРОЕКТИРОВАНИЕ УНИВЕРСАЛЬНОГО МОДУЛЯ ПРЕДОБРАБОТКИ")
    print("="*60)
    print("Функции:")
    print("- Замена числительных на <NUM>")
    print("- Замена URL на <URL>")
    print("- Замена email на <EMAIL>")
    print("- Замена телефонов на <PHONE>")
    print("- Замена дат на <DATE>")
    print("- Замена времени на <TIME>")
    print("- Замена валют на <CURRENCY>")
    print("- Расшифровка сокращений")
    print()
    
    # Загрузка очищенных статей
    articles = load_articles("kommersant_articles_cleaned.jsonl")
    if not articles:
        logger.error("Не удалось загрузить очищенные статьи!")
        return False
    
    print(f"📁 Загружено {len(articles)} очищенных статей")
    
    # Создание конфигурации
    config = PreprocessingConfig(
        replace_numbers=True,
        replace_urls=True,
        replace_emails=True,
        replace_phones=True,
        replace_dates=True,
        replace_times=True,
        replace_currencies=True,
        normalize_punctuation=True,
        normalize_quotes=True,
        normalize_dashes=True,
        normalize_spaces=True,
        expand_abbreviations=True,
        expand_contractions=True,
        to_lowercase=False,
        remove_extra_punctuation=True,
        preserve_sentence_structure=True
    )
    
    # Создание предпроцессора
    preprocessor = UniversalPreprocessor(config, language='russian')
    
    # Предобработка статей
    logger.info("Начинаем универсальную предобработку...")
    processed_articles = preprocessor.batch_preprocess(articles)
    
    if not processed_articles:
        logger.error("Не удалось предобработать статьи!")
        return False
    
    # Сохранение обработанных статей
    save_articles(processed_articles, "kommersant_articles_processed.jsonl")
    
    # Сохранение конфигурации
    preprocessor.save_config("preprocessing_config.json")
    
    # Статистика
    original_words = sum(len(article['text'].split()) for article in articles)
    processed_words = sum(len(article['text'].split()) for article in processed_articles)
    
    # Подсчет токенов
    token_counts = {
        '<NUM>': 0,
        '<URL>': 0,
        '<EMAIL>': 0,
        '<PHONE>': 0,
        '<DATE>': 0,
        '<TIME>': 0,
        '<CURRENCY>': 0
    }
    
    for article in processed_articles:
        text = article['text']
        for token in token_counts.keys():
            token_counts[token] += text.count(token)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ЭТАПА 3")
    print("="*60)
    print(f"✅ Обработано статей: {len(processed_articles)}")
    print(f"✅ Исходное количество слов: {original_words:,}")
    print(f"✅ После предобработки слов: {processed_words:,}")
    
    print(f"\n🔢 Статистика замен:")
    for token, count in token_counts.items():
        if count > 0:
            print(f"   {token}: {count:,} замен")
    
    # Примеры предобработки
    print(f"\n📝 Примеры предобработки:")
    for i, (original, processed) in enumerate(zip(articles[:3], processed_articles[:3])):
        print(f"\nСтатья {i+1}:")
        print(f"Исходный текст: {original['text'][:200]}...")
        print(f"Обработанный текст: {processed['text'][:200]}...")
    
    print(f"\n📁 Обработанные статьи сохранены в: kommersant_articles_processed.jsonl")
    print(f"📁 Конфигурация сохранена в: preprocessing_config.json")
    print(f"\n🎉 Этап 3 завершен успешно!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Этап 3 выполнен успешно!")
        print("📝 Следующий шаг: python step4.py")
    else:
        print("\n❌ Этап 3 завершился с ошибкой!")
        sys.exit(1)

