"""
Модуль обучения моделей распределённых представлений слов и документов.

Реализует:
- Word2Vec (Skip-gram и CBOW)
- FastText (skipgram и cbow режимы)
- Doc2Vec (PV-DM и PV-DBOW)
- GloVe (при наличии ресурсов)
- Сравнительный анализ моделей
"""

import numpy as np
import pandas as pd
import json
import time
import logging
from typing import List, Dict, Tuple, Any, Optional
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile
import warnings
warnings.filterwarnings('ignore')

# Попытка импорта GloVe
try:
    from glove import Glove
    GLOVE_AVAILABLE = True
except ImportError:
    try:
        from glove_python import Glove
        GLOVE_AVAILABLE = True
    except ImportError:
        GLOVE_AVAILABLE = False
        print("GloVe не установлен. Обучение GloVe будет пропущено.")

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedRepresentationsTrainer:
    """Класс для обучения моделей распределённых представлений."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_and_preprocess_data(self, file_path: str) -> Tuple[List[List[str]], List[str]]:
        """Загрузка и предобработка данных."""
        logger.info(f"Загрузка данных из {file_path}")
        
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if 'text' in data and data['text']:
                        texts.append(data['text'])
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return [], []
        
        # Токенизация текстов
        logger.info("Токенизация текстов")
        tokenized_texts = []
        for text in texts:
            # Простая токенизация по пробелам
            tokens = text.lower().split()
            # Фильтрация коротких токенов
            tokens = [token for token in tokens if len(token) > 2]
            if len(tokens) > 5:  # Минимальная длина документа
                tokenized_texts.append(tokens)
        
        logger.info(f"Обработано {len(tokenized_texts)} документов")
        return tokenized_texts, texts
    
    def train_word2vec(self, tokenized_texts: List[List[str]], 
                      vector_sizes: List[int] = [100, 200, 300],
                      window_sizes: List[int] = [5, 8, 10],
                      min_counts: List[int] = [5, 10]) -> Dict[str, Any]:
        """Обучение моделей Word2Vec."""
        logger.info("Обучение моделей Word2Vec")
        
        results = {}
        
        for vector_size in vector_sizes:
            for window_size in window_sizes:
                for min_count in min_counts:
                    for sg in [0, 1]:  # 0 = CBOW, 1 = Skip-gram
                        architecture = "CBOW" if sg == 0 else "Skip-gram"
                        
                        logger.info(f"Обучение Word2Vec: {architecture}, size={vector_size}, window={window_size}, min_count={min_count}")
                        
                        start_time = time.time()
                        
                        model = Word2Vec(
                            sentences=tokenized_texts,
                            vector_size=vector_size,
                            window=window_size,
                            min_count=min_count,
                            sg=sg,
                            workers=4,
                            epochs=10,
                            seed=42
                        )
                        
                        training_time = time.time() - start_time
                        
                        # Сохранение модели
                        model_name = f"word2vec_{architecture.lower()}_size{vector_size}_win{window_size}_min{min_count}"
                        model_path = f"{model_name}.model"
                        model.save(model_path)
                        
                        # Анализ модели
                        vocab_size = len(model.wv)
                        total_words = sum(len(doc) for doc in tokenized_texts)
                        
                        results[model_name] = {
                            'model': model,
                            'model_path': model_path,
                            'architecture': architecture,
                            'vector_size': vector_size,
                            'window_size': window_size,
                            'min_count': min_count,
                            'vocab_size': vocab_size,
                            'total_words': total_words,
                            'training_time': training_time,
                            'coverage': vocab_size / len(set(word for doc in tokenized_texts for word in doc))
                        }
                        
                        logger.info(f"Word2Vec {architecture}: {vocab_size} слов, время обучения {training_time:.2f}с")
        
        return results
    
    def train_fasttext(self, tokenized_texts: List[List[str]],
                      vector_sizes: List[int] = [100, 200, 300],
                      window_sizes: List[int] = [5, 8, 10],
                      min_counts: List[int] = [5, 10]) -> Dict[str, Any]:
        """Обучение моделей FastText."""
        logger.info("Обучение моделей FastText")
        
        results = {}
        
        for vector_size in vector_sizes:
            for window_size in window_sizes:
                for min_count in min_counts:
                    for sg in [0, 1]:  # 0 = CBOW, 1 = Skip-gram
                        architecture = "CBOW" if sg == 0 else "Skip-gram"
                        
                        logger.info(f"Обучение FastText: {architecture}, size={vector_size}, window={window_size}, min_count={min_count}")
                        
                        start_time = time.time()
                        
                        model = FastText(
                            sentences=tokenized_texts,
                            vector_size=vector_size,
                            window=window_size,
                            min_count=min_count,
                            sg=sg,
                            workers=4,
                            epochs=10,
                            seed=42
                        )
                        
                        training_time = time.time() - start_time
                        
                        # Сохранение модели
                        model_name = f"fasttext_{architecture.lower()}_size{vector_size}_win{window_size}_min{min_count}"
                        model_path = f"{model_name}.model"
                        model.save(model_path)
                        
                        # Анализ модели
                        vocab_size = len(model.wv)
                        total_words = sum(len(doc) for doc in tokenized_texts)
                        
                        results[model_name] = {
                            'model': model,
                            'model_path': model_path,
                            'architecture': architecture,
                            'vector_size': vector_size,
                            'window_size': window_size,
                            'min_count': min_count,
                            'vocab_size': vocab_size,
                            'total_words': total_words,
                            'training_time': training_time,
                            'coverage': vocab_size / len(set(word for doc in tokenized_texts for word in doc))
                        }
                        
                        logger.info(f"FastText {architecture}: {vocab_size} слов, время обучения {training_time:.2f}с")
        
        return results
    
    def train_glove(self, tokenized_texts: List[List[str]],
                   vector_sizes: List[int] = [100, 200, 300],
                   window_sizes: List[int] = [5, 8, 10],
                   min_counts: List[int] = [5, 10]) -> Dict[str, Any]:
        """Обучение моделей GloVe."""
        if not GLOVE_AVAILABLE:
            logger.warning("GloVe не доступен, пропускаем обучение GloVe")
            return {}
        
        logger.info("Обучение моделей GloVe")
        
        results = {}
        
        for vector_size in vector_sizes:
            for window_size in window_sizes:
                for min_count in min_counts:
                    logger.info(f"Обучение GloVe: size={vector_size}, window={window_size}, min_count={min_count}")
                    
                    start_time = time.time()
                    
                    try:
                        # Создание модели GloVe
                        model = Glove(vector_size=vector_size, window=window_size, min_count=min_count)
                        
                        # Обучение модели
                        model.fit(tokenized_texts)
                        
                        training_time = time.time() - start_time
                        
                        # Сохранение модели
                        model_name = f"glove_size{vector_size}_win{window_size}_min{min_count}"
                        model_path = f"{model_name}.model"
                        model.save(model_path)
                        
                        # Анализ модели
                        vocab_size = len(model.wv)
                        total_words = sum(len(doc) for doc in tokenized_texts)
                        
                        results[model_name] = {
                            'model': model,
                            'model_path': model_path,
                            'architecture': 'GloVe',
                            'vector_size': vector_size,
                            'window_size': window_size,
                            'min_count': min_count,
                            'vocab_size': vocab_size,
                            'total_words': total_words,
                            'training_time': training_time,
                            'coverage': vocab_size / len(set(word for doc in tokenized_texts for word in doc))
                        }
                        
                        logger.info(f"GloVe: {vocab_size} слов, время обучения {training_time:.2f}с")
                        
                    except Exception as e:
                        logger.error(f"Ошибка обучения GloVe: {e}")
                        continue
        
        return results
    
    def train_doc2vec(self, tokenized_texts: List[List[str]], texts: List[str],
                     vector_sizes: List[int] = [100, 200, 300],
                     window_sizes: List[int] = [5, 8, 10],
                     min_counts: List[int] = [5, 10]) -> Dict[str, Any]:
        """Обучение моделей Doc2Vec."""
        logger.info("Обучение моделей Doc2Vec")
        
        # Подготовка документов для Doc2Vec
        tagged_docs = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(tokenized_texts)]
        
        results = {}
        
        for vector_size in vector_sizes:
            for window_size in window_sizes:
                for min_count in min_counts:
                    for dm in [0, 1]:  # 0 = PV-DBOW, 1 = PV-DM
                        architecture = "PV-DM" if dm == 1 else "PV-DBOW"
                        
                        logger.info(f"Обучение Doc2Vec: {architecture}, size={vector_size}, window={window_size}, min_count={min_count}")
                        
                        start_time = time.time()
                        
                        model = Doc2Vec(
                            documents=tagged_docs,
                            vector_size=vector_size,
                            window=window_size,
                            min_count=min_count,
                            dm=dm,
                            workers=4,
                            epochs=10,
                            seed=42
                        )
                        
                        training_time = time.time() - start_time
                        
                        # Сохранение модели
                        model_name = f"doc2vec_{architecture.lower()}_size{vector_size}_win{window_size}_min{min_count}"
                        model_path = f"{model_name}.model"
                        model.save(model_path)
                        
                        # Анализ модели
                        vocab_size = len(model.wv)
                        doc_count = len(model.dv)
                        
                        results[model_name] = {
                            'model': model,
                            'model_path': model_path,
                            'architecture': architecture,
                            'vector_size': vector_size,
                            'window_size': window_size,
                            'min_count': min_count,
                            'vocab_size': vocab_size,
                            'doc_count': doc_count,
                            'training_time': training_time,
                            'coverage': vocab_size / len(set(word for doc in tokenized_texts for word in doc))
                        }
                        
                        logger.info(f"Doc2Vec {architecture}: {vocab_size} слов, {doc_count} документов, время обучения {training_time:.2f}с")
        
        return results
    
    def evaluate_word_analogies(self, model, analogies: List[Tuple[str, str, str, str]]) -> float:
        """Оценка точности аналогий для модели слов."""
        correct = 0
        total = 0
        
        for word1, word2, word3, expected in analogies:
            try:
                # Проверяем наличие слов в модели
                if hasattr(model, 'wv'):
                    # Word2Vec, FastText, Doc2Vec
                    if all(word in model.wv for word in [word1, word2, word3]):
                        predicted = model.wv.most_similar(positive=[word2, word3], negative=[word1], topn=1)[0][0]
                        if predicted == expected:
                            correct += 1
                        total += 1
                elif hasattr(model, 'most_similar'):
                    # GloVe
                    if all(word in model.dictionary for word in [word1, word2, word3]):
                        predicted = model.most_similar(positive=[word2, word3], negative=[word1], topn=1)[0][0]
                        if predicted == expected:
                            correct += 1
                        total += 1
            except Exception as e:
                logger.debug(f"Ошибка при оценке аналогии {word1}-{word2}+{word3}={expected}: {e}")
                continue
        
        return correct / total if total > 0 else 0.0
    
    def evaluate_semantic_similarity(self, model, word_pairs: List[Tuple[str, str]]) -> float:
        """Оценка семантического сходства."""
        similarities = []
        
        for word1, word2 in word_pairs:
            try:
                if hasattr(model, 'wv'):
                    # Word2Vec, FastText, Doc2Vec
                    if word1 in model.wv and word2 in model.wv:
                        similarity = model.wv.similarity(word1, word2)
                        similarities.append(similarity)
                elif hasattr(model, 'similarity'):
                    # GloVe
                    if word1 in model.dictionary and word2 in model.dictionary:
                        similarity = model.similarity(word1, word2)
                        similarities.append(similarity)
            except Exception as e:
                logger.debug(f"Ошибка при оценке сходства {word1}-{word2}: {e}")
                continue
        
        return np.mean(similarities) if similarities else 0.0
    
    def evaluate_document_clustering(self, doc2vec_model, n_clusters: int = 10) -> Dict[str, float]:
        """Оценка качества кластеризации документов."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, adjusted_rand_score
        
        # Получение векторов документов
        doc_vectors = np.array([doc2vec_model.dv[str(i)] for i in range(len(doc2vec_model.dv))])
        
        # K-means кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(doc_vectors)
        
        # Оценка качества
        silhouette_avg = silhouette_score(doc_vectors, cluster_labels)
        inertia = kmeans.inertia_
        
        return {
            'silhouette_score': silhouette_avg,
            'inertia': inertia,
            'n_clusters': n_clusters
        }
    
    def create_test_analogies(self) -> List[Tuple[str, str, str, str]]:
        """Создание тестовых аналогий для русского языка."""
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
        
        return analogies
    
    def create_similarity_pairs(self) -> List[Tuple[str, str]]:
        """Создание пар слов для оценки семантического сходства."""
        pairs = [
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
        
        return pairs
    
    def comprehensive_evaluation(self, word_models: Dict[str, Any], 
                               doc_models: Dict[str, Any]) -> Dict[str, Any]:
        """Комплексная оценка всех моделей."""
        logger.info("Проведение комплексной оценки моделей")
        
        # Подготовка тестовых данных
        analogies = self.create_test_analogies()
        similarity_pairs = self.create_similarity_pairs()
        
        evaluation_results = {
            'word_models': {},
            'doc_models': {}
        }
        
        # Оценка word models
        for model_name, model_data in word_models.items():
            logger.info(f"Оценка word model: {model_name}")
            
            model = model_data['model']
            
            # Оценка аналогий
            analogy_accuracy = self.evaluate_word_analogies(model, analogies)
            
            # Оценка семантического сходства
            semantic_similarity = self.evaluate_semantic_similarity(model, similarity_pairs)
            
            # Анализ покрытия словаря
            vocab_coverage = model_data['coverage']
            
            evaluation_results['word_models'][model_name] = {
                'analogy_accuracy': analogy_accuracy,
                'semantic_similarity': semantic_similarity,
                'vocab_coverage': vocab_coverage,
                'vocab_size': model_data['vocab_size'],
                'training_time': model_data['training_time'],
                'vector_size': model_data['vector_size'],
                'architecture': model_data['architecture']
            }
        
        # Оценка doc models
        for model_name, model_data in doc_models.items():
            logger.info(f"Оценка doc model: {model_name}")
            
            model = model_data['model']
            
            # Оценка кластеризации
            clustering_metrics = self.evaluate_document_clustering(model)
            
            evaluation_results['doc_models'][model_name] = {
                'clustering_silhouette': clustering_metrics['silhouette_score'],
                'clustering_inertia': clustering_metrics['inertia'],
                'vocab_size': model_data['vocab_size'],
                'doc_count': model_data['doc_count'],
                'training_time': model_data['training_time'],
                'vector_size': model_data['vector_size'],
                'architecture': model_data['architecture']
            }
        
        return evaluation_results
    
    def save_results(self, word_models: Dict[str, Any], doc_models: Dict[str, Any],
                    evaluation_results: Dict[str, Any]) -> None:
        """Сохранение результатов обучения и оценки."""
        logger.info("Сохранение результатов")
        
        # Подготовка данных для сохранения
        save_data = {
            'word_models_summary': {},
            'doc_models_summary': {},
            'evaluation_results': evaluation_results
        }
        
        # Сводка по word models
        for model_name, model_data in word_models.items():
            save_data['word_models_summary'][model_name] = {
                'architecture': model_data['architecture'],
                'vector_size': model_data['vector_size'],
                'vocab_size': model_data['vocab_size'],
                'training_time': model_data['training_time'],
                'coverage': model_data['coverage']
            }
        
        # Сводка по doc models
        for model_name, model_data in doc_models.items():
            save_data['doc_models_summary'][model_name] = {
                'architecture': model_data['architecture'],
                'vector_size': model_data['vector_size'],
                'vocab_size': model_data['vocab_size'],
                'doc_count': model_data['doc_count'],
                'training_time': model_data['training_time'],
                'coverage': model_data['coverage']
            }
        
        # Сохранение в JSON
        with open('distributed_representations_results.json', 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        # Создание сводных таблиц
        word_df = pd.DataFrame([
            {
                'Model': name,
                'Architecture': data['architecture'],
                'Vector_Size': data['vector_size'],
                'Vocab_Size': data['vocab_size'],
                'Training_Time': data['training_time'],
                'Coverage': data['coverage']
            }
            for name, data in word_models.items()
        ])
        
        doc_df = pd.DataFrame([
            {
                'Model': name,
                'Architecture': data['architecture'],
                'Vector_Size': data['vector_size'],
                'Vocab_Size': data['vocab_size'],
                'Doc_Count': data['doc_count'],
                'Training_Time': data['training_time'],
                'Coverage': data['coverage']
            }
            for name, data in doc_models.items()
        ])
        
        word_df.to_csv('word_models_summary.csv', index=False, encoding='utf-8')
        doc_df.to_csv('doc_models_summary.csv', index=False, encoding='utf-8')
        
        logger.info("Результаты сохранены в:")
        logger.info("- distributed_representations_results.json")
        logger.info("- word_models_summary.csv")
        logger.info("- doc_models_summary.csv")
    
    def run_training_pipeline(self, file_path: str) -> Dict[str, Any]:
        """Запуск полного пайплайна обучения."""
        logger.info("Запуск пайплайна обучения распределённых представлений")
        
        # Загрузка и предобработка данных
        tokenized_texts, texts = self.load_and_preprocess_data(file_path)
        
        if not tokenized_texts:
            logger.error("Не удалось загрузить данные")
            return {}
        
        # Ограничиваем количество текстов для демонстрации
        tokenized_texts = tokenized_texts[:500]
        texts = texts[:500]
        
        logger.info(f"Обучение на {len(tokenized_texts)} документах")
        
        # Обучение моделей
        logger.info("Обучение Word2Vec моделей")
        word2vec_results = self.train_word2vec(tokenized_texts)
        
        logger.info("Обучение FastText моделей")
        fasttext_results = self.train_fasttext(tokenized_texts)
        
        logger.info("Обучение GloVe моделей")
        glove_results = self.train_glove(tokenized_texts)
        
        logger.info("Обучение Doc2Vec моделей")
        doc2vec_results = self.train_doc2vec(tokenized_texts, texts)
        
        # Объединение результатов
        word_models = {**word2vec_results, **fasttext_results, **glove_results}
        doc_models = doc2vec_results
        
        # Комплексная оценка
        evaluation_results = self.comprehensive_evaluation(word_models, doc_models)
        
        # Сохранение результатов
        self.save_results(word_models, doc_models, evaluation_results)
        
        return {
            'word_models': word_models,
            'doc_models': doc_models,
            'evaluation_results': evaluation_results
        }


def main():
    """Основная функция для демонстрации работы модуля."""
    trainer = DistributedRepresentationsTrainer()
    
    # Запуск пайплайна обучения
    results = trainer.run_training_pipeline("../Task1/kommersant_articles_processed.jsonl")
    
    if results:
        print("\n=== ОБУЧЕНИЕ РАСПРЕДЕЛЁННЫХ ПРЕДСТАВЛЕНИЙ ЗАВЕРШЕНО ===")
        print(f"Обучено word models: {len(results['word_models'])}")
        print(f"Обучено doc models: {len(results['doc_models'])}")
        print("Результаты сохранены в файлы CSV и JSON")


if __name__ == "__main__":
    main()

