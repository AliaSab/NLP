#!/usr/bin/env python3
"""
Этап 4: Сравнительный анализ методов токенизации и нормализации
Анализ различных методов токенизации и нормализации текста
"""
import os
import sys
import json
import logging
from tokenization_analysis import TokenizationAnalyzer

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_articles(filename: str = "kommersant_articles_processed.jsonl"):
    """Загрузка обработанных статей"""
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

def main():
    """Этап 4: Анализ токенизации"""
    print("="*60)
    print("ЭТАП 4: СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТОДОВ ТОКЕНИЗАЦИИ И НОРМАЛИЗАЦИИ")
    print("="*60)
    print("Методы токенизации:")
    print("- Наивная (по пробелам)")
    print("- Регулярные выражения")
    print("- NLTK")
    print("- spaCy")
    print("- razdel")
    print()
    print("Методы нормализации:")
    print("- Porter Stemmer")
    print("- Snowball Stemmer")
    print("- spaCy лемматизация")
    print("- pymorphy2 лемматизация")
    print()
    
    # Загрузка обработанных статей
    articles = load_articles("kommersant_articles_processed.jsonl")
    if not articles:
        logger.error("Не удалось загрузить обработанные статьи!")
        return False
    
    print(f"📁 Загружено {len(articles)} обработанных статей")
    
    # Извлечение текстов
    texts = []
    for article in articles:
        if 'text' in article and article['text'].strip():
            texts.append(article['text'])
    
    if not texts:
        logger.error("Не найдено текстов для анализа!")
        return False
    
    print(f"📝 Извлечено {len(texts)} текстов для анализа")
    
    # Создание анализатора
    analyzer = TokenizationAnalyzer(language='russian')
    
    # Анализ корпуса
    logger.info("Начинаем анализ токенизации...")
    results = analyzer.analyze_corpus(texts, test_size=0.2)
    
    if not results:
        logger.error("Не удалось выполнить анализ!")
        return False
    
    # Сохранение результатов
    df = analyzer.save_results(results, "tokenization_analysis_results.json")
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ЭТАПА 4")
    print("="*60)
    
    # Вывод сравнительной таблицы
    print("\n📊 Сравнительная таблица методов токенизации:")
    print(df.to_string(index=False))
    
    # Анализ лучших методов
    best_oov = df.loc[df['oov_rate'].idxmin()]
    best_semantic = df.loc[df['semantic_similarity'].idxmax()]
    fastest = df.loc[df['train_processing_time'].idxmin()]
    
    print(f"\n🏆 Лучшие результаты:")
    print(f"   Минимальный OOV rate: {best_oov['method']} ({best_oov['oov_rate']:.4f})")
    print(f"   Максимальное семантическое сходство: {best_semantic['method']} ({best_semantic['semantic_similarity']:.4f})")
    print(f"   Самая быстрая обработка: {fastest['method']} ({fastest['train_processing_time']:.4f}с)")
    
    # Анализ нормализации на примере
    print(f"\n📝 Пример анализа нормализации:")
    sample_text = "Это пример текста для анализа различных методов нормализации слов."
    tokens = analyzer.naive_tokenization(sample_text)
    normalization_results = analyzer.analyze_normalization(tokens)
    
    for method, result in normalization_results.items():
        if 'error' not in result:
            print(f"\n{method}:")
            print(f"  Токенов: {result['token_count']}")
            print(f"  Уникальных: {result['unique_tokens']}")
            print(f"  Время обработки: {result['processing_time']:.4f}с")
            print(f"  Коэффициент сжатия: {result['compression_ratio']:.2f}")
            print(f"  Токены: {result['tokens'][:10]}...")
    
    print(f"\n📁 Результаты сохранены в:")
    print(f"   - tokenization_analysis_results.json")
    print(f"   - tokenization_analysis_results.csv")
    print(f"\n🎉 Этап 4 завершен успешно!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Этап 4 выполнен успешно!")
        print("📝 Следующий шаг: python step5.py")
    else:
        print("\n❌ Этап 4 завершился с ошибкой!")
        sys.exit(1)

