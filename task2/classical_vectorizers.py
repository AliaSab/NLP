"""
Модуль классической векторизации текста с поддержкой n-грамм.

Реализует:
- One-Hot Encoding для слов и n-грамм
- Bag of Words с различными схемами взвешивания
- TF-IDF с настройкой параметров
- Поддержку n-грамм (1-3) и их комбинаций
- Анализ разреженности и размерности матриц
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import json
import time
from typing import List, Dict, Tuple, Any, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassicalVectorizer:
    """Класс для классической векторизации текста."""
    
    def __init__(self):
        self.vectorizers = {}
        self.results = {}
        
    def load_data(self, file_path: str) -> List[str]:
        """Загрузка данных из JSONL файла."""
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if 'text' in data and data['text']:
                        texts.append(data['text'])
            logger.info(f"Загружено {len(texts)} текстов из {file_path}")
            return texts
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return []
    
    def one_hot_encoding(self, texts: List[str], ngram_range: Tuple[int, int] = (1, 1)) -> Dict[str, Any]:
        """One-Hot Encoding для слов и n-грамм."""
        logger.info(f"Выполнение One-Hot Encoding с ngram_range={ngram_range}")
        
        start_time = time.time()
        
        # Создание векторизатора
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            binary=True,  # One-Hot Encoding
            min_df=2,  # Минимальная частота документа
            max_features=10000,  # Ограничение количества признаков
            token_pattern=r'\b\w+\b',  # Паттерн для токенов
            lowercase=True,  # Приведение к нижнему регистру
            stop_words=None  # Не используем стоп-слова для демонстрации
        )
        
        # Обучение и трансформация
        matrix = vectorizer.fit_transform(texts)
        
        # Анализ результатов
        n_docs, n_features = matrix.shape
        sparsity = 1 - (matrix.nnz / (n_docs * n_features))
        
        processing_time = time.time() - start_time
        
        result = {
            'matrix': matrix,
            'vectorizer': vectorizer,
            'feature_names': vectorizer.get_feature_names_out(),
            'n_documents': n_docs,
            'n_features': n_features,
            'sparsity': sparsity,
            'processing_time': processing_time,
            'ngram_range': ngram_range,
            'method': 'One-Hot Encoding'
        }
        
        logger.info(f"One-Hot Encoding завершен: {n_features} признаков, разреженность {sparsity:.3f}")
        return result
    
    def bag_of_words(self, texts: List[str], ngram_range: Tuple[int, int] = (1, 1), 
                    weighting: str = 'count') -> Dict[str, Any]:
        """Bag of Words с различными схемами взвешивания."""
        logger.info(f"Выполнение Bag of Words с ngram_range={ngram_range}, weighting={weighting}")
        
        start_time = time.time()
        
        # Создание векторизатора
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            min_df=2,
            max_features=10000,
            token_pattern=r'\b\w+\b',
            lowercase=True,
            stop_words=None
        )
        
        # Обучение и трансформация
        matrix = vectorizer.fit_transform(texts)
        
        # Применение различных схем взвешивания
        if weighting == 'binary':
            matrix = (matrix > 0).astype(int)
        elif weighting == 'log':
            matrix.data = np.log(matrix.data + 1)
        elif weighting == 'sqrt':
            matrix.data = np.sqrt(matrix.data)
        elif weighting == 'normalized':
            # Нормализация по длине документа
            from sklearn.preprocessing import normalize
            matrix = normalize(matrix, norm='l2')
        
        # Анализ результатов
        n_docs, n_features = matrix.shape
        sparsity = 1 - (matrix.nnz / (n_docs * n_features))
        
        processing_time = time.time() - start_time
        
        result = {
            'matrix': matrix,
            'vectorizer': vectorizer,
            'feature_names': vectorizer.get_feature_names_out(),
            'n_documents': n_docs,
            'n_features': n_features,
            'sparsity': sparsity,
            'processing_time': processing_time,
            'ngram_range': ngram_range,
            'weighting': weighting,
            'method': f'Bag of Words ({weighting})'
        }
        
        logger.info(f"Bag of Words завершен: {n_features} признаков, разреженность {sparsity:.3f}")
        return result
    
    def tfidf_vectorization(self, texts: List[str], ngram_range: Tuple[int, int] = (1, 1),
                          smooth_idf: bool = True, sublinear_tf: bool = False) -> Dict[str, Any]:
        """TF-IDF векторизация с настройкой параметров."""
        logger.info(f"Выполнение TF-IDF с ngram_range={ngram_range}, smooth_idf={smooth_idf}, sublinear_tf={sublinear_tf}")
        
        start_time = time.time()
        
        # Создание векторизатора
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=2,
            max_features=10000,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            token_pattern=r'\b\w+\b',
            lowercase=True,
            stop_words=None,
            norm='l2'  # L2 нормализация
        )
        
        # Обучение и трансформация
        matrix = vectorizer.fit_transform(texts)
        
        # Анализ результатов
        n_docs, n_features = matrix.shape
        sparsity = 1 - (matrix.nnz / (n_docs * n_features))
        
        processing_time = time.time() - start_time
        
        result = {
            'matrix': matrix,
            'vectorizer': vectorizer,
            'feature_names': vectorizer.get_feature_names_out(),
            'n_documents': n_docs,
            'n_features': n_features,
            'sparsity': sparsity,
            'processing_time': processing_time,
            'ngram_range': ngram_range,
            'smooth_idf': smooth_idf,
            'sublinear_tf': sublinear_tf,
            'method': f'TF-IDF (smooth_idf={smooth_idf}, sublinear_tf={sublinear_tf})'
        }
        
        logger.info(f"TF-IDF завершен: {n_features} признаков, разреженность {sparsity:.3f}")
        return result
    
    def analyze_sparsity_and_dimensions(self, matrix: csr_matrix, method_name: str) -> Dict[str, Any]:
        """Анализ разреженности и размерности матрицы."""
        n_docs, n_features = matrix.shape
        nnz = matrix.nnz
        sparsity = 1 - (nnz / (n_docs * n_features))
        
        # Статистика по ненулевым элементам
        non_zero_counts = np.array(matrix.sum(axis=1)).flatten()
        avg_non_zero = np.mean(non_zero_counts)
        std_non_zero = np.std(non_zero_counts)
        
        # Статистика по частоте признаков
        feature_counts = np.array(matrix.sum(axis=0)).flatten()
        avg_feature_freq = np.mean(feature_counts)
        std_feature_freq = np.std(feature_counts)
        
        analysis = {
            'method': method_name,
            'n_documents': n_docs,
            'n_features': n_features,
            'total_elements': n_docs * n_features,
            'non_zero_elements': nnz,
            'sparsity': sparsity,
            'avg_non_zero_per_doc': avg_non_zero,
            'std_non_zero_per_doc': std_non_zero,
            'avg_feature_frequency': avg_feature_freq,
            'std_feature_frequency': std_feature_freq,
            'memory_usage_mb': matrix.data.nbytes / (1024 * 1024)
        }
        
        return analysis
    
    def semantic_consistency_analysis(self, matrix: csr_matrix, texts: List[str], 
                                    method_name: str) -> Dict[str, Any]:
        """Анализ семантической согласованности - косинусное сходство между документами одной темы."""
        logger.info(f"Анализ семантической согласованности для {method_name}")
        
        try:
            # Простая эвристика для группировки документов по темам
            # Используем первые несколько слов как индикатор темы
            topic_groups = {}
            for i, text in enumerate(texts):
                # Берем первые 3 слова как индикатор темы
                topic_key = ' '.join(text.split()[:3]).lower()
                if topic_key not in topic_groups:
                    topic_groups[topic_key] = []
                topic_groups[topic_key].append(i)
            
            # Фильтруем группы с минимум 2 документами
            valid_groups = {k: v for k, v in topic_groups.items() if len(v) >= 2}
            
            if not valid_groups:
                return {'error': 'Недостаточно документов для анализа семантической согласованности'}
            
            similarities = []
            
            for topic, doc_indices in valid_groups.items():
                if len(doc_indices) >= 2:
                    # Вычисляем косинусное сходство между документами в группе
                    group_matrix = matrix[doc_indices]
                    
                    # Вычисляем попарные сходства
                    for i in range(len(doc_indices)):
                        for j in range(i + 1, len(doc_indices)):
                            try:
                                # Косинусное сходство
                                from sklearn.metrics.pairwise import cosine_similarity
                                sim = cosine_similarity(group_matrix[i:i+1], group_matrix[j:j+1])[0][0]
                                similarities.append(sim)
                            except:
                                continue
            
            if similarities:
                analysis = {
                    'method': method_name,
                    'semantic_consistency_mean': np.mean(similarities),
                    'semantic_consistency_std': np.std(similarities),
                    'semantic_consistency_min': np.min(similarities),
                    'semantic_consistency_max': np.max(similarities),
                    'semantic_consistency_median': np.median(similarities),
                    'topic_groups_analyzed': len(valid_groups),
                    'total_similarities': len(similarities)
                }
            else:
                analysis = {'error': 'Не удалось вычислить семантическую согласованность'}
            
            return analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа семантической согласованности: {e}")
            return {'error': f'Ошибка анализа: {str(e)}'}
    
    def compare_ngram_models(self, texts: List[str]) -> Dict[str, Any]:
        """Сравнение различных n-граммных моделей."""
        logger.info("Сравнение n-граммных моделей")
        
        ngram_configs = [
            (1, 1, "Unigrams"),
            (2, 2, "Bigrams"),
            (3, 3, "Trigrams"),
            (1, 2, "Unigrams + Bigrams"),
            (1, 3, "Unigrams + Bigrams + Trigrams"),
            (2, 3, "Bigrams + Trigrams")
        ]
        
        results = {}
        
        for ngram_min, ngram_max, name in ngram_configs:
            logger.info(f"Обработка {name}")
            
            # TF-IDF с различными n-граммами
            result = self.tfidf_vectorization(
                texts, 
                ngram_range=(ngram_min, ngram_max)
            )
            
            # Анализ разреженности
            analysis = self.analyze_sparsity_and_dimensions(
                result['matrix'], 
                f"TF-IDF {name}"
            )
            
            # Анализ семантической согласованности
            semantic_analysis = self.semantic_consistency_analysis(
                result['matrix'], 
                texts, 
                f"TF-IDF {name}"
            )
            
            results[name] = {
                'result': result,
                'analysis': analysis,
                'semantic_analysis': semantic_analysis
            }
        
        return results
    
    def run_all_experiments(self, texts: List[str]) -> Dict[str, Any]:
        """Запуск всех экспериментов по классической векторизации."""
        logger.info("Запуск всех экспериментов по классической векторизации")
        
        experiments = {}
        
        # One-Hot Encoding эксперименты
        logger.info("Эксперименты с One-Hot Encoding")
        experiments['one_hot'] = {}
        for ngram_range in [(1, 1), (1, 2), (1, 3)]:
            name = f"One-Hot {ngram_range[0]}-{ngram_range[1]}gram"
            result = self.one_hot_encoding(texts, ngram_range)
            analysis = self.analyze_sparsity_and_dimensions(result['matrix'], name)
            semantic_analysis = self.semantic_consistency_analysis(result['matrix'], texts, name)
            experiments['one_hot'][name] = {
                'result': result, 
                'analysis': analysis,
                'semantic_analysis': semantic_analysis
            }
        
        # Bag of Words эксперименты
        logger.info("Эксперименты с Bag of Words")
        experiments['bag_of_words'] = {}
        weightings = ['count', 'binary', 'log', 'sqrt', 'normalized']
        for weighting in weightings:
            for ngram_range in [(1, 1), (1, 2)]:
                name = f"BoW {weighting} {ngram_range[0]}-{ngram_range[1]}gram"
                result = self.bag_of_words(texts, ngram_range, weighting)
                analysis = self.analyze_sparsity_and_dimensions(result['matrix'], name)
                semantic_analysis = self.semantic_consistency_analysis(result['matrix'], texts, name)
                experiments['bag_of_words'][name] = {
                    'result': result, 
                    'analysis': analysis,
                    'semantic_analysis': semantic_analysis
                }
        
        # TF-IDF эксперименты
        logger.info("Эксперименты с TF-IDF")
        experiments['tfidf'] = {}
        configs = [
            (True, False, "smooth_idf=True"),
            (False, False, "smooth_idf=False"),
            (True, True, "smooth_idf=True, sublinear_tf=True"),
            (False, True, "smooth_idf=False, sublinear_tf=True")
        ]
        
        for smooth_idf, sublinear_tf, config_name in configs:
            for ngram_range in [(1, 1), (1, 2), (1, 3)]:
                name = f"TF-IDF {config_name} {ngram_range[0]}-{ngram_range[1]}gram"
                result = self.tfidf_vectorization(texts, ngram_range, smooth_idf, sublinear_tf)
                analysis = self.analyze_sparsity_and_dimensions(result['matrix'], name)
                semantic_analysis = self.semantic_consistency_analysis(result['matrix'], texts, name)
                experiments['tfidf'][name] = {
                    'result': result, 
                    'analysis': analysis,
                    'semantic_analysis': semantic_analysis
                }
        
        # N-граммное сравнение
        logger.info("Сравнение n-граммных моделей")
        experiments['ngram_comparison'] = self.compare_ngram_models(texts)
        
        self.results = experiments
        return experiments
    
    def save_results(self, output_file: str = "classical_vectorization_results.json"):
        """Сохранение результатов экспериментов."""
        logger.info(f"Сохранение результатов в {output_file}")
        
        # Подготовка данных для сохранения (без матриц)
        save_data = {}
        
        for category, methods in self.results.items():
            save_data[category] = {}
            for method_name, data in methods.items():
                save_data[category][method_name] = {
                    'analysis': data['analysis'],
                    'feature_names': data['result']['feature_names'].tolist() if 'feature_names' in data['result'] else [],
                    'n_documents': data['result']['n_documents'],
                    'n_features': data['result']['n_features'],
                    'sparsity': data['result']['sparsity'],
                    'processing_time': data['result']['processing_time']
                }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info("Результаты сохранены")
    
    def create_summary_table(self) -> pd.DataFrame:
        """Создание сводной таблицы результатов."""
        summary_data = []
        
        for category, methods in self.results.items():
            for method_name, data in methods.items():
                analysis = data['analysis']
                semantic_analysis = data.get('semantic_analysis', {})
                
                summary_data.append({
                    'Method': method_name,
                    'Category': category,
                    'Documents': analysis['n_documents'],
                    'Features': analysis['n_features'],
                    'Sparsity': analysis['sparsity'],
                    'Avg_NonZero_Per_Doc': analysis['avg_non_zero_per_doc'],
                    'Memory_MB': analysis['memory_usage_mb'],
                    'Processing_Time': data['result']['processing_time'],
                    'Semantic_Consistency_Mean': semantic_analysis.get('semantic_consistency_mean', 0.0),
                    'Semantic_Consistency_Std': semantic_analysis.get('semantic_consistency_std', 0.0),
                    'Topic_Groups_Analyzed': semantic_analysis.get('topic_groups_analyzed', 0)
                })
        
        return pd.DataFrame(summary_data)


def main():
    """Основная функция для демонстрации работы модуля."""
    vectorizer = ClassicalVectorizer()
    
    # Загрузка данных
    texts = vectorizer.load_data("../Task1/kommersant_articles_processed.jsonl")
    
    if not texts:
        logger.error("Не удалось загрузить данные")
        return
    
    # Ограничиваем количество текстов для демонстрации
    texts = texts[:1000]  # Первые 1000 текстов
    
    logger.info(f"Обработка {len(texts)} текстов")
    
    # Запуск всех экспериментов
    results = vectorizer.run_all_experiments(texts)
    
    # Создание сводной таблицы
    summary_df = vectorizer.create_summary_table()
    summary_df.to_csv("vectorization_metrics.csv", index=False, encoding='utf-8')
    
    # Сохранение результатов
    vectorizer.save_results("classical_vectorization_results.json")
    
    # Вывод сводной статистики
    print("\n=== СВОДНАЯ СТАТИСТИКА ===")
    print(summary_df.to_string(index=False))
    
    print(f"\nРезультаты сохранены в:")
    print("- vectorization_metrics.csv")
    print("- classical_vectorization_results.json")


if __name__ == "__main__":
    main()

