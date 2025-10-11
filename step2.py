#!/usr/bin/env python3
"""
Этап 2: Предварительная обработка и очистка текста
Очистка HTML, удаление служебных символов, нормализация
"""
import os
import sys
import json
import logging
from text_cleaner import TextCleaner

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_articles(filename: str = "kommersant_articles.jsonl"):
    """Загрузка статей из JSONL файла"""
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
    """Сохранение статей в JSONL файл"""
    with open(filename, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')
    
    logger.info(f"Сохранено {len(articles)} статей в {filename}")

def main():
    """Этап 2: Очистка и предобработка текстов"""
    print("="*60)
    print("ЭТАП 2: ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА И ОЧИСТКА ТЕКСТА")
    print("="*60)
    print("Функции:")
    print("- Удаление HTML-разметки")
    print("- Удаление служебных символов")
    print("- Стандартизация пробельных символов")
    print("- Фильтрация стоп-слов")
    print()
    
    # Загрузка статей
    articles = load_articles("kommersant_articles.jsonl")
    if not articles:
        logger.error("Не удалось загрузить статьи!")
        return False
    
    print(f"📁 Загружено {len(articles)} статей")
    
    # Создание очистителя
    cleaner = TextCleaner(remove_stopwords=True, language='russian')
    
    # Очистка статей
    logger.info("Начинаем очистку статей...")
    cleaned_articles = cleaner.batch_clean(
        articles,
        clean_title=True,
        clean_text=True,
        remove_html=True,
        remove_urls=True,
        remove_phones=True,
        remove_dates=True,
        remove_numbers=False,  # Сохраняем числа для анализа
        normalize_whitespace=True,
        normalize_punctuation=True,
        to_lowercase=False,  # Сохраняем регистр
        remove_stopwords=True
    )
    
    if not cleaned_articles:
        logger.error("Не удалось очистить статьи!")
        return False
    
    # Сохранение очищенных статей
    save_articles(cleaned_articles, "kommersant_articles_cleaned.jsonl")
    
    # Статистика
    original_words = sum(len(article['text'].split()) for article in articles)
    cleaned_words = sum(len(article['text'].split()) for article in cleaned_articles)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ЭТАПА 2")
    print("="*60)
    print(f"✅ Обработано статей: {len(cleaned_articles)}")
    print(f"✅ Исходное количество слов: {original_words:,}")
    print(f"✅ После очистки слов: {cleaned_words:,}")
    print(f"✅ Сжатие: {((original_words - cleaned_words) / original_words * 100):.1f}%")
    
    # Примеры очистки
    print(f"\n📝 Примеры очистки:")
    for i, (original, cleaned) in enumerate(zip(articles[:3], cleaned_articles[:3])):
        print(f"\nСтатья {i+1}:")
        print(f"Исходный заголовок: {original['title'][:100]}...")
        print(f"Очищенный заголовок: {cleaned['title'][:100]}...")
        print(f"Исходный текст: {original['text'][:150]}...")
        print(f"Очищенный текст: {cleaned['text'][:150]}...")
    
    print(f"\n📁 Очищенные статьи сохранены в: kommersant_articles_cleaned.jsonl")
    print(f"\n🎉 Этап 2 завершен успешно!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Этап 2 выполнен успешно!")
        print("📝 Следующий шаг: python step3.py")
    else:
        print("\n❌ Этап 2 завершился с ошибкой!")
        sys.exit(1)

