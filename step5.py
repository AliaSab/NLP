#!/usr/bin/env python3
"""
Этап 5: Обучение подсловных моделей токенизации
Обучение BPE, WordPiece, Unigram и SentencePiece моделей
"""
import os
import sys
import json
import logging
from subword_models import SubwordModelTrainer

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
    """Этап 5: Обучение подсловных моделей"""
    print("="*60)
    print("ЭТАП 5: ОБУЧЕНИЕ ПОДСЛОВНЫХ МОДЕЛЕЙ ТОКЕНИЗАЦИИ")
    print("="*60)
    print("Модели:")
    print("- Byte Pair Encoding (BPE)")
    print("- WordPiece")
    print("- Unigram Language Model")
    print("- SentencePiece (BPE и Unigram)")
    print()
    print("Размеры словаря: 8,000, 16,000, 32,000 токенов")
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
        logger.error("Не найдено текстов для обучения!")
        return False
    
    print(f"📝 Извлечено {len(texts)} текстов для обучения")
    
    # Создание тренера
    trainer = SubwordModelTrainer(language='russian')
    
    # Обучение всех моделей
    logger.info("Начинаем обучение подсловных моделей...")
    results = trainer.train_all_models(texts, vocab_sizes=[8000, 16000, 32000])
    
    if not results:
        logger.error("Не удалось обучить модели!")
        return False
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ЭТАПА 5")
    print("="*60)
    
    # Вывод результатов обучения
    if 'training_results' in results:
        print(f"\n📚 Результаты обучения:")
        for model_name, metrics in results['training_results'].items():
            print(f"\n{model_name}:")
            print(f"  Тип модели: {metrics['model_type']}")
            print(f"  Размер словаря: {metrics['vocab_size']:,}")
            print(f"  Время обучения: {metrics['training_time']:.2f}с")
            print(f"  Тестовых токенов: {metrics['test_token_count']}")
            print(f"  Пример токенов: {metrics['test_tokens'][:10]}...")
    
    # Вывод результатов оценки
    if 'evaluation_results' in results:
        print(f"\n📊 Результаты оценки моделей:")
        for model_name, metrics in results['evaluation_results'].items():
            print(f"\n{model_name}:")
            print(f"  Коэффициент фрагментации: {metrics['fragmentation_rate']:.4f}")
            print(f"  Коэффициент сжатия: {metrics['compression_ratio']:.4f}")
            print(f"  Среднее время обработки: {metrics['avg_processing_time']:.4f}с")
            print(f"  Токенов в секунду: {metrics['tokens_per_second']:.2f}")
            print(f"  Всего слов: {metrics['total_words']:,}")
            print(f"  Всего токенов: {metrics['total_tokens']:,}")
    
    # Анализ лучших моделей
    if 'evaluation_results' in results:
        eval_results = results['evaluation_results']
        
        # Лучшая модель по фрагментации (меньше лучше)
        best_fragmentation = min(eval_results.items(), key=lambda x: x[1]['fragmentation_rate'])
        
        # Лучшая модель по сжатию (больше лучше)
        best_compression = max(eval_results.items(), key=lambda x: x[1]['compression_ratio'])
        
        # Самая быстрая модель
        fastest_model = min(eval_results.items(), key=lambda x: x[1]['avg_processing_time'])
        
        print(f"\n🏆 Лучшие результаты:")
        print(f"   Минимальная фрагментация: {best_fragmentation[0]} ({best_fragmentation[1]['fragmentation_rate']:.4f})")
        print(f"   Максимальное сжатие: {best_compression[0]} ({best_compression[1]['compression_ratio']:.4f})")
        print(f"   Самая быстрая: {fastest_model[0]} ({fastest_model[1]['avg_processing_time']:.4f}с)")
    
    print(f"\n📁 Результаты сохранены в:")
    print(f"   - subword_models_results.json")
    print(f"   - subword_models_comparison.csv")
    print(f"   - corpus.txt (обучающий корпус)")
    print(f"   - Модели: *.json, *.model, *.vocab")
    print(f"\n🎉 Этап 5 завершен успешно!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Этап 5 выполнен успешно!")
        print("📝 Следующий шаг: python step6.py")
    else:
        print("\n❌ Этап 5 завершился с ошибкой!")
        sys.exit(1)

