"""
Основной скрипт для запуска полного пайплайна кластеризации.
"""

import os
os.environ['TCL_LIBRARY'] = "C:/Program Files/Python313/tcl/tcl8.6"
os.environ['TK_LIBRARY'] = "C:/Program Files/Python313/tcl/tk8.6"
import sys
import json
import argparse
import logging
import numpy as np
from typing import List, Dict, Any, Optional

# Добавляем пути
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Task1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Task2'))

from text_preprocessing import TextPreprocessor, load_articles_from_jsonl, preprocess_articles
from text_to_vector import TextVectorizer, find_embedding_models
from clustering import ClusteringPipeline, tune_kmeans
from evaluation import ClusteringEvaluator
from interpretation import ClusteringInterpreter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_clustering_pipeline(
    input_file: str,
    output_dir: str = "results",
    tokenization_method: str = 'whitespace',
    vectorization_method: str = 'tfidf',
    clustering_method: str = 'kmeans',
    n_clusters: int = 5,
    use_lemmatization: bool = True,
    vocab_size: Optional[int] = None,
    embedding_model_path: Optional[str] = None,
    max_articles: Optional[int] = None
):
    """Запуск полного пайплайна кластеризации."""
    
    # Создаем директорию для результатов
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*70)
    logger.info("ЗАПУСК ПАЙПЛАЙНА КЛАСТЕРИЗАЦИИ")
    logger.info("="*70)
    
    # 1. Загрузка данных
    logger.info("\n[1/6] Загрузка данных...")
    articles = load_articles_from_jsonl(input_file)
    
    if max_articles:
        articles = articles[:max_articles]
        logger.info(f"Ограничено до {max_articles} статей")
    
    logger.info(f"Загружено {len(articles)} статей")
    
    # 2. Предобработка
    logger.info("\n[2/6] Предобработка...")
    preprocessor = TextPreprocessor(use_spacy=True, use_pymorphy=True)
    
    # Загрузка BPE/SentencePiece моделей
    if tokenization_method in ['bpe', 'sentencepiece']:
        task1_path = os.path.join(os.path.dirname(__file__), '..', 'Task1')
        if tokenization_method == 'bpe' and vocab_size:
            bpe_path = os.path.join(task1_path, f'bpe_model_{vocab_size}.json')
            if os.path.exists(bpe_path):
                preprocessor.load_bpe_model(bpe_path, vocab_size)
        elif tokenization_method == 'sentencepiece' and vocab_size:
            sp_path = os.path.join(task1_path, f'sp_bpe_{vocab_size}.model')
            if os.path.exists(sp_path):
                preprocessor.load_sentencepiece_model(sp_path, vocab_size)
    
    tokenized_texts = preprocess_articles(
        articles,
        preprocessor,
        tokenization_method=tokenization_method,
        lemmatize=use_lemmatization,
        vocab_size=vocab_size
    )
    logger.info(f"Предобработано {len(tokenized_texts)} статей")
    
    # 3. Векторизация
    logger.info("\n[3/6] Векторизация...")
    vectorizer = TextVectorizer(normalize_vectors=True)
    
    if vectorization_method == 'tfidf':
        vectorizer.fit_tfidf(tokenized_texts)
        vectors = vectorizer.transform_tfidf(tokenized_texts)
    elif vectorization_method == 'bm25':
        vectorizer.fit_bm25(tokenized_texts)
        vectors = vectorizer.transform_bm25(tokenized_texts)
    elif vectorization_method in ['word2vec', 'fasttext', 'glove']:
        if embedding_model_path and os.path.exists(embedding_model_path):
            if vectorization_method == 'word2vec':
                vectorizer.load_word2vec(embedding_model_path)
            elif vectorization_method == 'fasttext':
                vectorizer.load_fasttext(embedding_model_path)
            elif vectorization_method == 'glove':
                vectorizer.load_glove(embedding_model_path)
            
            vectors = vectorizer.transform_embeddings(
                tokenized_texts,
                model_type=vectorization_method
            )
        else:
            logger.error(f"Модель {vectorization_method} не найдена: {embedding_model_path}")
            return
    else:
        logger.error(f"Неизвестный метод векторизации: {vectorization_method}")
        return
    
    logger.info(f"Векторизация завершена. Размерность: {vectors.shape}")
    
    # 4. Кластеризация
    logger.info("\n[4/6] Кластеризация...")
    clustering = ClusteringPipeline()
    
    if clustering_method == 'kmeans':
        labels = clustering.kmeans(vectors, n_clusters=n_clusters)
    elif clustering_method == 'dbscan':
        labels = clustering.dbscan(vectors, eps=0.5, min_samples=5)
    elif clustering_method == 'hierarchical':
        labels = clustering.hierarchical(vectors, n_clusters=n_clusters)
    elif clustering_method == 'gmm':
        labels = clustering.gmm(vectors, n_components=n_clusters)
    elif clustering_method == 'spectral':
        labels = clustering.spectral(vectors, n_clusters=n_clusters)
    else:
        logger.error(f"Неизвестный метод кластеризации: {clustering_method}")
        return
    
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info(f"Кластеризация завершена. Найдено кластеров: {n_clusters_found}")
    
    # 5. Оценка
    logger.info("\n[5/6] Оценка качества...")
    evaluator = ClusteringEvaluator()
    metrics = evaluator.evaluate_clustering(vectors, labels)
    evaluator.print_metrics(metrics)
    
    # Сохранение метрик
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Метрики сохранены: {metrics_path}")
    
    # 6. Интерпретация
    logger.info("\n[6/6] Интерпретация результатов...")
    interpreter = ClusteringInterpreter(vectorizer)
    interpreter.set_data(tokenized_texts, vectors, labels)
    
    # Топ слова
    top_words = interpreter.get_top_words_per_cluster_frequency(n_words=10)
    
    # Размеры кластеров
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique_labels, counts))
    
    # Сохранение интерпретации
    interpretation = {
        'top_words': {str(k): v for k, v in top_words.items()},
        'cluster_sizes': {str(k): int(v) for k, v in cluster_sizes.items()}
    }
    
    interpretation_path = os.path.join(output_dir, 'interpretation.json')
    with open(interpretation_path, 'w', encoding='utf-8') as f:
        json.dump(interpretation, f, ensure_ascii=False, indent=2)
    logger.info(f"Интерпретация сохранена: {interpretation_path}")
    
    # Визуализация
    logger.info("Построение визуализации...")
    viz_path = os.path.join(output_dir, 'clusters_umap.png')
    interpreter.visualize_clusters_umap(save_path=viz_path)
    
    # Сохранение результатов кластеризации
    results = {
        'articles': [
            {
                'title': article.get('title', ''),
                'text': article.get('text', '')[:200] + '...' if len(article.get('text', '')) > 200 else article.get('text', ''),
                'cluster': int(labels[i])
            }
            for i, article in enumerate(articles)
        ],
        'labels': labels.tolist(),
        'n_clusters': n_clusters_found
    }
    
    results_path = os.path.join(output_dir, 'clustering_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Результаты сохранены: {results_path}")
    
    logger.info("\n" + "="*70)
    logger.info("ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО")
    logger.info("="*70)
    
    return {
        'articles': articles,
        'tokenized_texts': tokenized_texts,
        'vectors': vectors,
        'labels': labels,
        'metrics': metrics,
        'top_words': top_words
    }


def main():
    parser = argparse.ArgumentParser(description='Пайплайн кластеризации текстов')
    
    
    default_input = os.path.join(
        os.path.dirname(__file__), '..', 'Task4', 'kommersant_articles_processed.jsonl'
    )
    default_input = os.path.normpath(default_input)
    
    parser.add_argument('--input', type=str, default=default_input,
                       help=f'Путь к входному JSONL файлу (по умолчанию: {default_input})')
    parser.add_argument('--output', type=str, default='results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--tokenization', type=str, default='whitespace',
                       choices=['whitespace', 'regex', 'bpe', 'sentencepiece'],
                       help='Метод токенизации')
    parser.add_argument('--vectorization', type=str, default='tfidf',
                       choices=['tfidf', 'bm25', 'word2vec', 'fasttext', 'glove'],
                       help='Метод векторизации')
    parser.add_argument('--clustering', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan', 'hierarchical', 'gmm', 'spectral'],
                       help='Метод кластеризации')
    parser.add_argument('--n-clusters', type=int, default=5,
                       help='Количество кластеров')
    parser.add_argument('--no-lemmatization', action='store_true',
                       help='Отключить лемматизацию')
    parser.add_argument('--vocab-size', type=int, default=None,
                       help='Размер словаря для BPE/SentencePiece')
    parser.add_argument('--embedding-model', type=str, default=None,
                       help='Путь к модели эмбеддингов')
    parser.add_argument('--max-articles', type=int, default=None,
                       help='Максимальное количество статей для обработки')
    
    args = parser.parse_args()
    
    # Проверяем существование входного файла
    if not os.path.exists(args.input):
        logger.error(f"Входной файл не найден: {args.input}")
        logger.info(f"Проверьте путь или укажите файл через --input")
        return
    
    logger.info(f"Используется входной файл: {args.input}")
    
    run_clustering_pipeline(
        input_file=args.input,
        output_dir=args.output,
        tokenization_method=args.tokenization,
        vectorization_method=args.vectorization,
        clustering_method=args.clustering,
        n_clusters=args.n_clusters,
        use_lemmatization=not args.no_lemmatization,
        vocab_size=args.vocab_size,
        embedding_model_path=args.embedding_model,
        max_articles=args.max_articles
    )


if __name__ == "__main__":
    main()

