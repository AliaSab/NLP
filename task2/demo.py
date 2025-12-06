"""
Демонстрационный скрипт для быстрого тестирования основных функций.

Этот скрипт показывает основные возможности системы анализа векторных представлений
без полного запуска всех этапов.
"""

import os
import sys
import json
import logging
from classical_vectorizers import ClassicalVectorizer
from dimensionality_reduction import DimensionalityReducer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_classical_vectorization():
    """Демонстрация классической векторизации."""
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ: КЛАССИЧЕСКАЯ ВЕКТОРИЗАЦИЯ")
    print("="*50)
    
    vectorizer = ClassicalVectorizer()
    
    # Загрузка данных
    data_path = "C:/Users/alia2/OneDrive/Desktop/NLP_ALYA/kommersant_articles_processed.jsonl"
    if not os.path.exists(data_path):
        print(f"Файл данных не найден: {data_path}")
        return
    
    texts = vectorizer.load_data(data_path)
    if not texts:
        print("Не удалось загрузить данные")
        return
    
    # Ограничиваем для демонстрации
    texts = texts[:100]
    print(f"Обработка {len(texts)} текстов")
    
    # TF-IDF векторизация
    print("\n1. TF-IDF векторизация:")
    tfidf_result = vectorizer.tfidf_vectorization(texts, ngram_range=(1, 2))
    print(f"   Размерность: {tfidf_result['n_features']} признаков")
    print(f"   Разреженность: {tfidf_result['sparsity']:.3f}")
    print(f"   Время обработки: {tfidf_result['processing_time']:.2f} сек")
    
    # One-Hot Encoding
    print("\n2. One-Hot Encoding:")
    onehot_result = vectorizer.one_hot_encoding(texts, ngram_range=(1, 1))
    print(f"   Размерность: {onehot_result['n_features']} признаков")
    print(f"   Разреженность: {onehot_result['sparsity']:.3f}")
    print(f"   Время обработки: {onehot_result['processing_time']:.2f} сек")
    
    # Bag of Words
    print("\n3. Bag of Words:")
    bow_result = vectorizer.bag_of_words(texts, ngram_range=(1, 1), weighting='count')
    print(f"   Размерность: {bow_result['n_features']} признаков")
    print(f"   Разреженность: {bow_result['sparsity']:.3f}")
    print(f"   Время обработки: {bow_result['processing_time']:.2f} сек")
    
    return tfidf_result


def demo_dimensionality_reduction(matrix, feature_names):
    """Демонстрация снижения размерности."""
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ: СНИЖЕНИЕ РАЗМЕРНОСТИ")
    print("="*50)
    
    reducer = DimensionalityReducer()
    
    # SVD анализ
    print("\n1. SVD анализ:")
    svd_results = reducer.svd_analysis(matrix, n_components_range=[10, 50, 100])
    
    for n_components, result in svd_results.items():
        cumulative_variance = result['cumulative_variance'][-1]
        print(f"   {n_components} компонент: объясненная дисперсия {cumulative_variance:.3f}")
    
    # Анализ объясненной дисперсии
    print("\n2. Анализ объясненной дисперсии:")
    variance_df = reducer.analyze_variance_explained(svd_results)
    print(variance_df.to_string(index=False))
    
    return svd_results


def demo_vector_arithmetic():
    """Демонстрация векторной арифметики (если есть обученная модель)."""
    print("\n" + "="*50)
    print("ДЕМОНСТРАЦИЯ: ВЕКТОРНАЯ АРИФМЕТИКА")
    print("="*50)
    
    # Поиск обученных моделей
    model_files = [f for f in os.listdir('.') if f.endswith('.model')]
    
    if not model_files:
        print("Обученные модели не найдены.")
        print("Для демонстрации векторной арифметики сначала обучите модель:")
        print("python distributed_representations.py")
        return
    
    print(f"Найдены модели: {model_files}")
    print("Для полной демонстрации векторной арифметики запустите:")
    print("python vector_arithmetic.py")


def main():
    """Основная функция демонстрации."""
    print("ДЕМОНСТРАЦИЯ СИСТЕМЫ АНАЛИЗА ВЕКТОРНЫХ ПРЕДСТАВЛЕНИЙ")
    print("="*60)
    
    try:
        # Демонстрация классической векторизации
        tfidf_result = demo_classical_vectorization()
        
        if tfidf_result:
            # Демонстрация снижения размерности
            demo_dimensionality_reduction(
                tfidf_result['matrix'], 
                tfidf_result['feature_names']
            )
        
        # Демонстрация векторной арифметики
        demo_vector_arithmetic()
        
        print("\n" + "="*60)
        print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
        print("="*60)
        
        print("\nДля полного анализа запустите:")
        print("python run_all_steps.py")
        
        print("\nДля веб-интерфейса запустите:")
        print("streamlit run web_interface.py")
        
    except Exception as e:
        print(f"Ошибка в демонстрации: {e}")
        logger.error(f"Ошибка в демонстрации: {e}")


if __name__ == "__main__":
    main()

