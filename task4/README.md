# Задача 4: Кластеризация текстов

Реализация полного пайплайна кластеризации текстовых данных с использованием различных методов векторизации и алгоритмов кластеризации.

## Структура проекта

- `text_preprocessing.py` - предобработка текстов (очистка, лемматизация, токенизация)
- `text_to_vector.py` - векторизация текстов (TF-IDF, BM25, Word2Vec, FastText, GloVe)
- `clustering.py` - алгоритмы кластеризации (K-Means, DBSCAN, Hierarchical, GMM, Spectral)
- `evaluation.py` - оценка качества кластеризации (внутренние и внешние метрики)
- `interpretation.py` - интерпретация результатов (топ слова, визуализация)
- `web_interface.py` - веб-интерфейс на Streamlit
- `run_pipeline.py` - основной скрипт для запуска пайплайна

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Установите русскую модель spaCy:
```bash
python -m spacy download ru_core_news_sm
```

3. Убедитесь, что в папке `Task1` есть BPE модели и в папке `Task2` есть модели эмбеддингов.

## Использование

### Веб-интерфейс

Запустите веб-интерфейс:
```bash
streamlit run web_interface.py
```

Интерфейс позволяет:
- Загрузить данные из JSONL файла
- Выбрать метод токенизации и векторизации
- Выполнить кластеризацию
- Просмотреть результаты, метрики и интерпретацию

### Программный интерфейс

Пример использования:

```python
from text_preprocessing import TextPreprocessor, load_articles_from_jsonl
from text_to_vector import TextVectorizer
from clustering import ClusteringPipeline
from evaluation import ClusteringEvaluator
from interpretation import ClusteringInterpreter

# Загрузка данных
articles = load_articles_from_jsonl('kommersant_articles_processed.jsonl')

# Предобработка
preprocessor = TextPreprocessor()
tokenized_texts = preprocess_articles(articles, preprocessor, tokenization_method='whitespace')

# Векторизация
vectorizer = TextVectorizer()
vectorizer.fit_tfidf(tokenized_texts)
vectors = vectorizer.transform_tfidf(tokenized_texts)

# Кластеризация
clustering = ClusteringPipeline()
labels = clustering.kmeans(vectors, n_clusters=5)

# Оценка
evaluator = ClusteringEvaluator()
metrics = evaluator.compute_internal_metrics(vectors, labels)
print(metrics)

# Интерпретация
interpreter = ClusteringInterpreter(vectorizer)
interpreter.set_data(tokenized_texts, vectors, labels)
top_words = interpreter.get_top_words_per_cluster_frequency(n_words=10)
```

## Методы токенизации

- **whitespace** - разделение по пробелам
- **regex** - регулярные выражения
- **bpe** - Byte Pair Encoding (из Task1)
- **sentencepiece** - SentencePiece (из Task1)

## Методы векторизации

- **TF-IDF** - Term Frequency-Inverse Document Frequency
- **BM25** - Best Matching 25
- **Word2Vec** - модели из Task2
- **FastText** - модели из Task2
- **GloVe** - модели из Task2 (если доступны)

## Алгоритмы кластеризации

- **K-Means** - кластеризация с заданным числом кластеров
- **DBSCAN** - плотностная кластеризация
- **Hierarchical** - иерархическая кластеризация
- **GMM** - Gaussian Mixture Model
- **Spectral** - спектральная кластеризация

## Метрики оценки

### Внутренние метрики:
- **Silhouette Score** - чем выше, тем лучше (диапазон [-1, 1])
- **Calinski-Harabasz Index** - чем выше, тем лучше
- **Davies-Bouldin Index** - чем ниже, тем лучше

### Внешние метрики (если есть разметка):
- **ARI** - Adjusted Rand Index
- **NMI** - Normalized Mutual Information
- **V-measure** - гармоническое среднее однородности и полноты

## Интерпретация

- Топ-N слов по кластерам (на основе TF-IDF или частоты)
- Ближайшие слова к центроидам (для эмбеддингов)
- Визуализация кластеров (UMAP, PCA)

## Примечания

- Для работы с BPE/SentencePiece необходимо наличие обученных моделей в Task1
- Для работы с Word2Vec/FastText/GloVe необходимо наличие обученных моделей в Task2
- GloVe может быть недоступен, если библиотека не установлена
- BM25 требует установки `rank-bm25`

## Автор

Создано для выполнения лабораторной работы по NLP.


