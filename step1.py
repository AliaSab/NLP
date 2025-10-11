#!/usr/bin/env python3
"""
Этап 1: Формирование экспериментального корпуса текстов
Парсинг статей с сайта Коммерсантъ (ID 8060000-8069999)
"""
import os
import sys
import json
import logging
from kommersant_parser import KommersantParser

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Этап 1: Парсинг новостных статей"""
    print("="*60)
    print("ЭТАП 1: ФОРМИРОВАНИЕ ЭКСПЕРИМЕНТАЛЬНОГО КОРПУСА ТЕКСТОВ")
    print("="*60)
    print("Источник: kommersant.ru")
    print("Диапазон ID: 8060000 - 8069999")
    print("Цель: не менее 50,000 слов")
    print()
    
    # Создание парсера
    parser = KommersantParser(delay=1.0)
    
    # Парсинг статей
    logger.info("Начинаем парсинг статей Коммерсанта...")
    articles = parser.parse_article_range(8060000, 8069999, max_articles=2000)
    
    if not articles:
        logger.error("Не удалось спарсить ни одной статьи!")
        return False
    
    # Сохранение результатов
    filepath = parser.save_articles(articles, "kommersant_articles.jsonl")
    
    # Статистика
    stats = parser.get_statistics(articles)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ЭТАПА 1")
    print("="*60)
    print(f"✅ Всего статей: {stats['total_articles']}")
    print(f"✅ Всего слов: {stats['total_words']:,}")
    print(f"✅ Среднее количество слов на статью: {stats['avg_words_per_article']:,}")
    
    # Проверка требований
    if stats['total_words'] >= 50000:
        print(f"✅ Требование выполнено: {stats['total_words']:,} >= 50,000 слов")
    else:
        print(f"⚠️ Требование не выполнено: {stats['total_words']:,} < 50,000 слов")
    
    print(f"\n📁 Файл сохранен: {filepath}")
    
    # Статистика по категориям
    if stats['categories']:
        print(f"\n📊 Статьи по категориям:")
        for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {category}: {count} статей")
    
    # Топ авторов
    if stats['top_authors']:
        print(f"\n👥 Топ авторов:")
        for author, count in list(stats['top_authors'].items())[:5]:
            print(f"   {author}: {count} статей")
    
    print(f"\n🎉 Этап 1 завершен успешно!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Этап 1 выполнен успешно!")
        print("📝 Следующий шаг: python step2.py")
    else:
        print("\n❌ Этап 1 завершился с ошибкой!")
        sys.exit(1)

