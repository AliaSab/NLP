"""
Модуль для векторизации текстов.
Поддерживает TF-IDF, BM25, Word2Vec, FastText, GloVe.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Добавляем путь к Task1 и Task2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Task1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Task2'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 не установлен. BM25 будет недоступен.")

# Gensim для эмбеддингов
try:
    from gensim.models import Word2Vec, FastText
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# GloVe
try:
    from glove import Glove
    GLOVE_AVAILABLE = True
except ImportError:
    try:
        from glove_python import Glove
        GLOVE_AVAILABLE = True
    except ImportError:
        GLOVE_AVAILABLE = False


class TextVectorizer:
    """Класс для векторизации текстов."""
    
    def __init__(self, normalize_vectors: bool = True):
        self.normalize_vectors = normalize_vectors
        
        # Векторизаторы
        self.tfidf_vectorizer = None
        self.bm25 = None
        
        # Модели эмбеддингов
        self.word2vec_model = None
        self.fasttext_model = None
        self.glove_model = None
        
        # Параметры моделей
        self.embedding_dim = None
    
    def fit_tfidf(self, tokenized_texts: List[List[str]], 
                  max_features: int = 10000,
                  min_df: int = 2,
                  max_df: float = 0.95) -> None:
        """Обучение TF-IDF векторизатора."""
        # Преобразуем токенизированные тексты в строки
        texts = [' '.join(tokens) for tokens in tokenized_texts]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            tokenizer=lambda x: x.split(),
            token_pattern=None
        )
        
        self.tfidf_vectorizer.fit(texts)
        logger.info(f"TF-IDF обучен: {len(self.tfidf_vectorizer.vocabulary_)} признаков")
    
    def transform_tfidf(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """Преобразование текстов в TF-IDF векторы."""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF векторизатор не обучен. Вызовите fit_tfidf() сначала.")
        
        texts = [' '.join(tokens) for tokens in tokenized_texts]
        vectors = self.tfidf_vectorizer.transform(texts).toarray()
        
        if self.normalize_vectors:
            vectors = normalize(vectors, norm='l2', axis=1)
        
        return vectors
    
    def fit_bm25(self, tokenized_texts: List[List[str]]) -> None:
        """Обучение BM25."""
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 не установлен. Установите: pip install rank-bm25")
        
        self.bm25 = BM25Okapi(tokenized_texts)
        logger.info("BM25 обучен")
    
    def transform_bm25(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """Преобразование текстов в BM25 векторы."""
        if self.bm25 is None:
            raise ValueError("BM25 не обучен. Вызовите fit_bm25() сначала.")
        
        # BM25 возвращает матрицу релевантности для каждого документа
        # Преобразуем в векторы, используя средние значения по корпусу
        vectors = []
        for tokens in tokenized_texts:
            scores = self.bm25.get_scores(tokens)
            vectors.append(scores)
        
        vectors = np.array(vectors)
        
        if self.normalize_vectors:
            vectors = normalize(vectors, norm='l2', axis=1)
        
        return vectors
    
    def load_word2vec(self, model_path: str) -> bool:
        """Загрузка модели Word2Vec из Task2."""
        if not GENSIM_AVAILABLE:
            logger.error("gensim не установлен")
            return False
        
        try:
            self.word2vec_model = Word2Vec.load(model_path)
            self.embedding_dim = self.word2vec_model.wv.vector_size
            logger.info(f"Word2Vec модель загружена: размерность {self.embedding_dim}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки Word2Vec модели: {e}")
            return False
    
    def load_fasttext(self, model_path: str) -> bool:
        """Загрузка модели FastText из Task2."""
        if not GENSIM_AVAILABLE:
            logger.error("gensim не установлен")
            return False
        
        try:
            self.fasttext_model = FastText.load(model_path)
            self.embedding_dim = self.fasttext_model.wv.vector_size
            logger.info(f"FastText модель загружена: размерность {self.embedding_dim}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки FastText модели: {e}")
            return False
    
    def load_glove(self, model_path: str) -> bool:
        """Загрузка модели GloVe."""
        if not GLOVE_AVAILABLE:
            logger.error("GloVe не установлен")
            return False
        
        try:
            self.glove_model = Glove.load(model_path)
            # GloVe использует другую структуру
            if hasattr(self.glove_model, 'word_vectors'):
                self.embedding_dim = self.glove_model.word_vectors.shape[1]
            logger.info(f"GloVe модель загружена: размерность {self.embedding_dim}")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки GloVe модели: {e}")
            return False
    
    def _get_word_embedding(self, word: str, model_type: str = 'word2vec') -> Optional[np.ndarray]:
        """Получение эмбеддинга слова."""
        if model_type == 'word2vec' and self.word2vec_model:
            try:
                return self.word2vec_model.wv[word]
            except KeyError:
                return None
        elif model_type == 'fasttext' and self.fasttext_model:
            try:
                return self.fasttext_model.wv[word]
            except KeyError:
                return None
        elif model_type == 'glove' and self.glove_model:
            try:
                word_id = self.glove_model.dictionary[word]
                return self.glove_model.word_vectors[word_id]
            except (KeyError, AttributeError):
                return None
        
        return None
    
    def get_document_embedding(self, tokens: List[str], 
                              model_type: str = 'word2vec',
                              method: str = 'mean') -> np.ndarray:
        """
        Получение эмбеддинга документа из эмбеддингов слов.
        
        Args:
            tokens: Список токенов
            model_type: Тип модели ('word2vec', 'fasttext', 'glove')
            method: Метод агрегации ('mean', 'max', 'sum')
        
        Returns:
            Вектор документа
        """
        embeddings = []
        
        for token in tokens:
            emb = self._get_word_embedding(token, model_type)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            # Если не найдено ни одного слова, возвращаем нулевой вектор
            if self.embedding_dim:
                return np.zeros(self.embedding_dim)
            else:
                return np.zeros(100)  # По умолчанию
        
        embeddings = np.array(embeddings)
        
        if method == 'mean':
            doc_embedding = np.mean(embeddings, axis=0)
        elif method == 'max':
            doc_embedding = np.max(embeddings, axis=0)
        elif method == 'sum':
            doc_embedding = np.sum(embeddings, axis=0)
        else:
            raise ValueError(f"Неизвестный метод агрегации: {method}")
        
        if self.normalize_vectors:
            norm = np.linalg.norm(doc_embedding)
            if norm > 0:
                doc_embedding = doc_embedding / norm
        
        return doc_embedding
    
    def transform_embeddings(self, tokenized_texts: List[List[str]],
                            model_type: str = 'word2vec',
                            method: str = 'mean') -> np.ndarray:
        """Преобразование списка текстов в эмбеддинги."""
        embeddings = []
        
        for tokens in tokenized_texts:
            doc_emb = self.get_document_embedding(tokens, model_type, method)
            embeddings.append(doc_emb)
        
        return np.array(embeddings)
    
    def find_similar_words(self, word: str, top_n: int = 10, 
                          model_type: str = 'word2vec') -> List[Tuple[str, float]]:
        """Поиск ближайших слов к заданному слову."""
        if model_type == 'word2vec' and self.word2vec_model:
            try:
                return self.word2vec_model.wv.most_similar(word, topn=top_n)
            except KeyError:
                return []
        elif model_type == 'fasttext' and self.fasttext_model:
            try:
                return self.fasttext_model.wv.most_similar(word, topn=top_n)
            except KeyError:
                return []
        elif model_type == 'glove' and self.glove_model:
            # GloVe не имеет встроенного метода most_similar
            # Нужно вычислять вручную через косинусное расстояние
            try:
                word_id = self.glove_model.dictionary[word]
                word_vec = self.glove_model.word_vectors[word_id]
                
                # Вычисляем косинусное сходство со всеми словами
                similarities = []
                for other_word, other_id in self.glove_model.dictionary.items():
                    if other_word != word:
                        other_vec = self.glove_model.word_vectors[other_id]
                        similarity = np.dot(word_vec, other_vec) / (
                            np.linalg.norm(word_vec) * np.linalg.norm(other_vec)
                        )
                        similarities.append((other_word, float(similarity)))
                
                # Сортируем и возвращаем топ-N
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:top_n]
            except (KeyError, AttributeError):
                return []
        
        return []


def find_embedding_models(task2_path: str) -> Dict[str, List[str]]:
    """Поиск моделей эмбеддингов в Task2."""
    models = {
        'word2vec': [],
        'fasttext': [],
        'glove': []
    }
    
    if not os.path.exists(task2_path):
        return models
    
    for filename in os.listdir(task2_path):
        if filename.endswith('.model'):
            if filename.startswith('word2vec_'):
                models['word2vec'].append(os.path.join(task2_path, filename))
            elif filename.startswith('fasttext_'):
                models['fasttext'].append(os.path.join(task2_path, filename))
            elif filename.startswith('glove_'):
                models['glove'].append(os.path.join(task2_path, filename))
    
    return models


if __name__ == "__main__":
    # Пример использования
    vectorizer = TextVectorizer()
    
    # Тестовые данные
    tokenized_texts = [
        ['это', 'пример', 'текста'],
        ['другой', 'пример', 'документа'],
        ['текст', 'содержит', 'слова']
    ]
    
    # TF-IDF
    vectorizer.fit_tfidf(tokenized_texts)
    tfidf_vectors = vectorizer.transform_tfidf(tokenized_texts)
    print(f"TF-IDF векторы: {tfidf_vectors.shape}")
    
    # BM25
    if BM25_AVAILABLE:
        vectorizer.fit_bm25(tokenized_texts)
        bm25_vectors = vectorizer.transform_bm25(tokenized_texts)
        print(f"BM25 векторы: {bm25_vectors.shape}")

