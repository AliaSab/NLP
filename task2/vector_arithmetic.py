"""
Модуль экспериментов с векторной арифметикой и семантическими операциями.

Реализует:
- Косинусное расстояние и семантическое сходство
- Векторную арифметику и word analogies
- Анализ семантических осей
- Качественный анализ ближайших соседей
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import time
from typing import List, Dict, Tuple, Any, Optional
import logging
from gensim.models import Word2Vec, FastText, Doc2Vec
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class VectorArithmeticAnalyzer:
    """Класс для анализа векторной арифметики и семантических операций."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_model(self, model_path: str, model_type: str = 'word2vec') -> Any:
        """Загрузка обученной модели."""
        logger.info(f"Загрузка модели {model_type} из {model_path}")
        
        try:
            if model_type == 'word2vec':
                model = Word2Vec.load(model_path)
            elif model_type == 'fasttext':
                model = FastText.load(model_path)
            elif model_type == 'doc2vec':
                model = Doc2Vec.load(model_path)
            elif model_type == 'glove':
                # Попытка загрузки GloVe
                try:
                    from glove import Glove
                    model = Glove.load(model_path)
                except ImportError:
                    try:
                        from glove_python import Glove
                        model = Glove.load(model_path)
                    except ImportError:
                        logger.error("GloVe не установлен")
                        return None
            else:
                raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
            
            # Определение количества слов в модели
            if hasattr(model, 'wv'):
                vocab_size = len(model.wv)
            elif hasattr(model, 'dictionary'):
                vocab_size = len(model.dictionary)
            else:
                vocab_size = "неизвестно"
            
            logger.info(f"Модель загружена: {vocab_size} слов")
            return model
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return None
    
    def cosine_similarity_analysis(self, model, word_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Анализ косинусного сходства между словами."""
        logger.info("Анализ косинусного сходства")
        
        similarities = []
        valid_pairs = []
        
        for word1, word2 in word_pairs:
            try:
                if word1 in model.wv and word2 in model.wv:
                    similarity = model.wv.similarity(word1, word2)
                    similarities.append(similarity)
                    valid_pairs.append((word1, word2))
            except:
                continue
        
        if not similarities:
            return {'error': 'Не удалось вычислить сходства'}
        
        analysis = {
            'similarities': similarities,
            'valid_pairs': valid_pairs,
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'median_similarity': np.median(similarities)
        }
        
        return analysis
    
    def distance_distribution_analysis(self, model, sample_words: List[str] = None) -> Dict[str, Any]:
        """Анализ распределения расстояний в векторном пространстве."""
        logger.info("Анализ распределения расстояний")
        
        if sample_words is None:
            # Выбираем случайную выборку слов
            vocab = list(model.wv.key_to_index.keys())
            sample_words = np.random.choice(vocab, min(1000, len(vocab)), replace=False)
        
        # Получение векторов
        vectors = np.array([model.wv[word] for word in sample_words if word in model.wv])
        
        if len(vectors) < 2:
            return {'error': 'Недостаточно векторов для анализа'}
        
        # Вычисление матрицы расстояний
        distances = cosine_distances(vectors)
        
        # Анализ распределения
        # Исключаем диагональные элементы (расстояние до самого себя)
        mask = np.ones_like(distances, dtype=bool)
        np.fill_diagonal(mask, False)
        distances_flat = distances[mask]
        
        analysis = {
            'mean_distance': np.mean(distances_flat),
            'std_distance': np.std(distances_flat),
            'min_distance': np.min(distances_flat),
            'max_distance': np.max(distances_flat),
            'median_distance': np.median(distances_flat),
            'percentile_25': np.percentile(distances_flat, 25),
            'percentile_75': np.percentile(distances_flat, 75),
            'distances': distances_flat.tolist()
        }
        
        return analysis
    
    def word_analogy_experiments(self, model, analogies: List[Tuple[str, str, str, str]]) -> Dict[str, Any]:
        """Эксперименты с векторной арифметикой и аналогиями."""
        logger.info("Проведение экспериментов с аналогиями")
        
        results = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'analogy_results': [],
            'category_results': {}
        }
        
        # Категории аналогий
        categories = {
            'semantic': [],
            'syntactic': [],
            'morphological': [],
            'professional': []
        }
        
        for word1, word2, word3, expected in analogies:
            try:
                if all(word in model.wv for word in [word1, word2, word3]):
                    # Векторная арифметика: word2 - word1 + word3
                    result_vector = model.wv[word2] - model.wv[word1] + model.wv[word3]
                    
                    # Поиск ближайших соседей
                    similar_words = model.wv.similar_by_vector(result_vector, topn=5)
                    
                    predicted = similar_words[0][0] if similar_words else None
                    is_correct = predicted == expected
                    
                    if is_correct:
                        results['correct_predictions'] += 1
                    
                    results['total_predictions'] += 1
                    
                    analogy_result = {
                        'word1': word1,
                        'word2': word2,
                        'word3': word3,
                        'expected': expected,
                        'predicted': predicted,
                        'is_correct': is_correct,
                        'similar_words': similar_words
                    }
                    
                    results['analogy_results'].append(analogy_result)
                    
                    # Определение категории (упрощённо)
                    if any(word in ['москва', 'париж', 'россия', 'франция'] for word in [word1, word2, word3, expected]):
                        categories['semantic'].append(analogy_result)
                    elif any(word in ['делать', 'сделал', 'писать', 'написал'] for word in [word1, word2, word3, expected]):
                        categories['morphological'].append(analogy_result)
                    elif any(word in ['учитель', 'школа', 'врач', 'больница'] for word in [word1, word2, word3, expected]):
                        categories['professional'].append(analogy_result)
                    else:
                        categories['syntactic'].append(analogy_result)
                        
            except Exception as e:
                logger.warning(f"Ошибка при обработке аналогии {word1}-{word2}+{word3}={expected}: {e}")
                continue
        
        # Вычисление точности по категориям
        for category, category_results in categories.items():
            if category_results:
                correct = sum(1 for result in category_results if result['is_correct'])
                total = len(category_results)
                results['category_results'][category] = {
                    'accuracy': correct / total,
                    'correct': correct,
                    'total': total
                }
        
        results['overall_accuracy'] = results['correct_predictions'] / results['total_predictions'] if results['total_predictions'] > 0 else 0
        
        return results
    
    def semantic_axis_analysis(self, model, axis_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Анализ семантических осей в векторном пространстве."""
        logger.info("Анализ семантических осей")
        
        axis_results = {}
        
        for axis_name, (word1, word2) in axis_pairs:
            try:
                # Проверяем наличие слов в модели
                if hasattr(model, 'wv'):
                    # Word2Vec, FastText, Doc2Vec
                    if word1 in model.wv and word2 in model.wv:
                        # Вычисление оси
                        axis_vector = model.wv[word2] - model.wv[word1]
                        axis_vector = axis_vector / np.linalg.norm(axis_vector)  # Нормализация
                        
                        # Проекция слов на ось
                        vocab = list(model.wv.key_to_index.keys())
                        projections = {}
                        
                        for word in vocab[:1000]:  # Ограничиваем для производительности
                            if word in model.wv:
                                projection = np.dot(model.wv[word], axis_vector)
                                projections[word] = projection
                        
                        # Сортировка по проекции
                        sorted_projections = sorted(projections.items(), key=lambda x: x[1])
                        
                        axis_results[axis_name] = {
                            'axis_vector': axis_vector.tolist(),
                            'word1': word1,
                            'word2': word2,
                            'projections': dict(sorted_projections),
                            'most_negative': sorted_projections[:10],
                            'most_positive': sorted_projections[-10:]
                        }
                        
                elif hasattr(model, 'dictionary'):
                    # GloVe
                    if word1 in model.dictionary and word2 in model.dictionary:
                        # Вычисление оси
                        axis_vector = model.word_vectors[model.dictionary[word2]] - model.word_vectors[model.dictionary[word1]]
                        axis_vector = axis_vector / np.linalg.norm(axis_vector)  # Нормализация
                        
                        # Проекция слов на ось
                        projections = {}
                        
                        for word, idx in list(model.dictionary.items())[:1000]:  # Ограничиваем для производительности
                            projection = np.dot(model.word_vectors[idx], axis_vector)
                            projections[word] = projection
                        
                        # Сортировка по проекции
                        sorted_projections = sorted(projections.items(), key=lambda x: x[1])
                        
                        axis_results[axis_name] = {
                            'axis_vector': axis_vector.tolist(),
                            'word1': word1,
                            'word2': word2,
                            'projections': dict(sorted_projections),
                            'most_negative': sorted_projections[:10],
                            'most_positive': sorted_projections[-10:]
                        }
                    
            except Exception as e:
                logger.warning(f"Ошибка при анализе оси {axis_name}: {e}")
                continue
        
        return axis_results
    
    def nearest_neighbors_analysis(self, model, test_words: List[str], top_k: int = 10) -> Dict[str, Any]:
        """Анализ ближайших соседей для тестовых слов."""
        logger.info("Анализ ближайших соседей")
        
        neighbors_results = {}
        
        for word in test_words:
            try:
                if hasattr(model, 'wv'):
                    # Word2Vec, FastText, Doc2Vec
                    if word in model.wv:
                        # Поиск ближайших соседей
                        similar_words = model.wv.most_similar(word, topn=top_k)
                        
                        neighbors_results[word] = {
                            'neighbors': similar_words,
                            'neighbor_words': [neighbor[0] for neighbor in similar_words],
                            'neighbor_similarities': [neighbor[1] for neighbor in similar_words]
                        }
                        
                elif hasattr(model, 'most_similar'):
                    # GloVe
                    if word in model.dictionary:
                        # Поиск ближайших соседей
                        similar_words = model.most_similar(word, topn=top_k)
                        
                        neighbors_results[word] = {
                            'neighbors': similar_words,
                            'neighbor_words': [neighbor[0] for neighbor in similar_words],
                            'neighbor_similarities': [neighbor[1] for neighbor in similar_words]
                        }
                    
            except Exception as e:
                logger.warning(f"Ошибка при поиске соседей для {word}: {e}")
                continue
        
        return neighbors_results
    
    def visualize_semantic_space(self, model, words: List[str], method: str = 'tsne') -> None:
        """Визуализация семантического пространства."""
        logger.info(f"Создание визуализации семантического пространства методом {method}")
        
        # Получение векторов слов
        vectors = []
        valid_words = []
        
        for word in words:
            try:
                if hasattr(model, 'wv'):
                    # Word2Vec, FastText, Doc2Vec
                    if word in model.wv:
                        vectors.append(model.wv[word])
                        valid_words.append(word)
                elif hasattr(model, 'dictionary'):
                    # GloVe
                    if word in model.dictionary:
                        vectors.append(model.word_vectors[model.dictionary[word]])
                        valid_words.append(word)
            except Exception as e:
                logger.debug(f"Ошибка при получении вектора для {word}: {e}")
                continue
        
        if len(vectors) < 2:
            logger.warning("Недостаточно слов для визуализации")
            return
        
        vectors = np.array(vectors)
        
        # Снижение размерности
        if method == 'tsne':
            try:
                perplexity = min(30, len(vectors) - 1)
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            except Exception as e:
                logger.warning(f"Ошибка с t-SNE: {e}, используем PCA")
                reducer = PCA(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            logger.error(f"Неподдерживаемый метод: {method}")
            return
        
        try:
            vectors_2d = reducer.fit_transform(vectors)
        except Exception as e:
            logger.error(f"Ошибка снижения размерности: {e}")
            return
        
        # Создание scatter plot
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7, s=50)
        
        # Добавление подписей к точкам
        for i, word in enumerate(valid_words):
            plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title(f'Семантическое пространство ({method.upper()})')
        plt.xlabel('Компонента 1')
        plt.ylabel('Компонента 2')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'semantic_space_{method}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_test_data(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str, str]], List[str]]:
        """Создание тестовых данных для экспериментов."""
        
        # Пары слов для анализа сходства
        similarity_pairs = [
            # Синонимы
            ("дом", "жилище"),
            ("автомобиль", "машина"),
            ("работа", "труд"),
            ("красивый", "прекрасный"),
            
            # Антонимы
            ("большой", "маленький"),
            ("хороший", "плохой"),
            ("горячий", "холодный"),
            ("быстрый", "медленный"),
            
            # Тематически близкие
            ("кошка", "собака"),
            ("яблоко", "груша"),
            ("солнце", "луна"),
            ("вода", "огонь"),
        ]
        
        # Аналогии
        analogies = [
            # Семантические аналогии
            ("москва", "россия", "франция", "париж"),
            ("мужчина", "женщина", "король", "королева"),
            ("хороший", "лучше", "плохой", "хуже"),
            ("большой", "больше", "маленький", "меньше"),
            
            # Синтаксические аналогии
            ("делать", "сделал", "писать", "написал"),
            ("читать", "читал", "думать", "думал"),
            ("говорить", "говорил", "слушать", "слушал"),
            
            # Профессиональные аналогии
            ("учитель", "школа", "врач", "больница"),
            ("повар", "кухня", "водитель", "машина"),
        ]
        
        # Тестовые слова для анализа соседей
        test_words = [
            "дом", "работа", "человек", "время", "жизнь",
            "мир", "страна", "город", "улица", "дорога",
            "книга", "слово", "мысль", "чувство", "любовь"
        ]
        
        return similarity_pairs, analogies, test_words
    
    def run_comprehensive_analysis(self, model_path: str, model_type: str = 'word2vec') -> Dict[str, Any]:
        """Запуск комплексного анализа векторной арифметики."""
        logger.info("Запуск комплексного анализа векторной арифметики")
        
        # Загрузка модели
        model = self.load_model(model_path, model_type)
        if model is None:
            return {}
        
        # Создание тестовых данных
        similarity_pairs, analogies, test_words = self.create_test_data()
        
        results = {}
        
        # Анализ косинусного сходства
        logger.info("Анализ косинусного сходства")
        results['cosine_similarity'] = self.cosine_similarity_analysis(model, similarity_pairs)
        
        # Анализ распределения расстояний
        logger.info("Анализ распределения расстояний")
        results['distance_distribution'] = self.distance_distribution_analysis(model)
        
        # Эксперименты с аналогиями
        logger.info("Эксперименты с аналогиями")
        results['word_analogies'] = self.word_analogy_experiments(model, analogies)
        
        # Анализ семантических осей
        logger.info("Анализ семантических осей")
        axis_pairs = [
            ("мужчина", "женщина"),
            ("большой", "маленький"),
            ("хороший", "плохой"),
            ("горячий", "холодный")
        ]
        results['semantic_axes'] = self.semantic_axis_analysis(model, axis_pairs)
        
        # Анализ ближайших соседей
        logger.info("Анализ ближайших соседей")
        results['nearest_neighbors'] = self.nearest_neighbors_analysis(model, test_words)
        
        # Визуализация семантического пространства
        logger.info("Создание визуализаций")
        self.visualize_semantic_space(model, test_words, 'tsne')
        self.visualize_semantic_space(model, test_words, 'pca')
        
        # Сохранение результатов
        self.save_analysis_results(results, model_path)
        
        return results
    
    def save_analysis_results(self, results: Dict[str, Any], model_path: str) -> None:
        """Сохранение результатов анализа."""
        logger.info("Сохранение результатов анализа")
        
        # Подготовка данных для сохранения
        save_data = {
            'model_path': model_path,
            'cosine_similarity': results.get('cosine_similarity', {}),
            'distance_distribution': results.get('distance_distribution', {}),
            'word_analogies': {
                'overall_accuracy': results.get('word_analogies', {}).get('overall_accuracy', 0),
                'total_predictions': results.get('word_analogies', {}).get('total_predictions', 0),
                'correct_predictions': results.get('word_analogies', {}).get('correct_predictions', 0),
                'category_results': results.get('word_analogies', {}).get('category_results', {})
            },
            'semantic_axes_summary': {
                axis_name: {
                    'word1': data['word1'],
                    'word2': data['word2'],
                    'most_negative': data['most_negative'],
                    'most_positive': data['most_positive']
                }
                for axis_name, data in results.get('semantic_axes', {}).items()
            },
            'nearest_neighbors_summary': {
                word: {
                    'neighbors': data['neighbors'][:5]  # Топ-5 соседей
                }
                for word, data in results.get('nearest_neighbors', {}).items()
            }
        }
        
        # Сохранение в JSON
        with open('vector_arithmetic_results.json', 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        # Создание сводной таблицы по аналогиям
        analogy_data = []
        for result in results.get('word_analogies', {}).get('analogy_results', []):
            analogy_data.append({
                'Word1': result['word1'],
                'Word2': result['word2'],
                'Word3': result['word3'],
                'Expected': result['expected'],
                'Predicted': result['predicted'],
                'Correct': result['is_correct']
            })
        
        if analogy_data:
            analogy_df = pd.DataFrame(analogy_data)
            analogy_df.to_csv('word_analogies_results.csv', index=False, encoding='utf-8')
        
        logger.info("Результаты сохранены в:")
        logger.info("- vector_arithmetic_results.json")
        logger.info("- word_analogies_results.csv")


def main():
    """Основная функция для демонстрации работы модуля."""
    analyzer = VectorArithmeticAnalyzer()
    
    # Пример использования с обученной моделью
    # Замените путь на реальный путь к вашей модели
    model_path = "word2vec_skip-gram_size200_win5_min5.model"
    
    try:
        results = analyzer.run_comprehensive_analysis(model_path, 'word2vec')
        
        if results:
            print("\n=== АНАЛИЗ ВЕКТОРНОЙ АРИФМЕТИКИ ЗАВЕРШЕН ===")
            print(f"Точность аналогий: {results.get('word_analogies', {}).get('overall_accuracy', 0):.3f}")
            print(f"Среднее косинусное сходство: {results.get('cosine_similarity', {}).get('mean_similarity', 0):.3f}")
            print("Результаты сохранены в файлы JSON и CSV")
        else:
            print("Не удалось провести анализ")
            
    except FileNotFoundError:
        print(f"Модель {model_path} не найдена. Сначала обучите модель с помощью distributed_representations.py")


if __name__ == "__main__":
    main()

