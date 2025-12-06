"""
Модуль для предобработки текстовых данных и извлечения признаков.
Этап 2: Подготовка данных для классификации.
"""

import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import logging

# NLP библиотеки
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK не установлен.")

# Векторизация
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

# Эмбеддинги
try:
    from gensim.models import Word2Vec, FastText
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Класс для предобработки текстов."""
    
    def __init__(self, language='ru', use_spacy=True, use_stopwords=True):
        self.language = language
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_stopwords = use_stopwords
        
        # Инициализация spaCy
        if self.use_spacy:
            try:
                self.nlp = spacy.load('ru_core_news_sm')
            except OSError:
                logger.warning("Русская модель spaCy не найдена. Используется упрощенная токенизация.")
                self.use_spacy = False
        
        # Стоп-слова
        if self.use_stopwords and NLTK_AVAILABLE:
            try:
                if language == 'ru':
                    # Русские стоп-слова
                    self.stop_words = set(['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'])
                else:
                    self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = set()
        else:
            self.stop_words = set()
    
    def clean_text(self, text: str) -> str:
        """Очистка текста от HTML-тегов, URL, специальных символов."""
        if not text:
            return ""
        
        # Удаление HTML-тегов
        text = re.sub(r'<[^>]+>', '', text)
        
        # Удаление URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Удаление email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Замена эмодзи на текстовое описание (упрощенная версия)
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(' [эмодзи] ', text)
        
        # Удаление специальных символов, оставляем только буквы, цифры и пробелы
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Удаление множественных пробелов
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def to_lowercase(self, text: str) -> str:
        """Приведение к нижнему регистру."""
        return text.lower()
    
    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста."""
        if self.use_spacy:
            doc = self.nlp(text)
            return [token.text for token in doc]
        elif NLTK_AVAILABLE:
            return word_tokenize(text)
        else:
            return text.split()
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Лемматизация токенов."""
        if self.use_spacy:
            doc = self.nlp(' '.join(tokens))
            return [token.lemma_ for token in doc]
        elif NLTK_AVAILABLE:
            lemmatizer = WordNetLemmatizer()
            return [lemmatizer.lemmatize(token) for token in tokens]
        else:
            return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Удаление стоп-слов."""
        if self.use_stopwords:
            return [token for token in tokens if token.lower() not in self.stop_words]
        return tokens
    
    def preprocess(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Полная предобработка текста."""
        text = self.clean_text(text)
        text = self.to_lowercase(text)
        tokens = self.tokenize(text)
        tokens = self.lemmatize(tokens)
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        # Удаляем пустые токены
        tokens = [t for t in tokens if t.strip()]
        return tokens


class FeatureExtractor:
    """Класс для извлечения признаков из текстов."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def extract_statistical_features(self, text: str) -> Dict[str, float]:
        """Извлечение статистических признаков."""
        tokens = self.preprocessor.preprocess(text, remove_stopwords=False)
        
        if not tokens:
            return {
                'text_length': 0,
                'word_count': 0,
                'avg_word_length': 0,
                'unique_words': 0,
                'punctuation_ratio': 0,
                'uppercase_ratio': 0,
                'digit_ratio': 0
            }
        
        text_lower = text.lower()
        total_chars = len(text)
        
        return {
            'text_length': len(text),
            'word_count': len(tokens),
            'avg_word_length': np.mean([len(t) for t in tokens]) if tokens else 0,
            'unique_words': len(set(tokens)),
            'punctuation_ratio': len(re.findall(r'[^\w\s]', text)) / total_chars if total_chars > 0 else 0,
            'uppercase_ratio': len(re.findall(r'[А-ЯA-Z]', text)) / total_chars if total_chars > 0 else 0,
            'digit_ratio': len(re.findall(r'\d', text)) / total_chars if total_chars > 0 else 0
        }
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Извлечение лингвистических признаков (сложность текста)."""
        tokens = self.preprocessor.preprocess(text, remove_stopwords=False)
        
        if not tokens:
            return {'flesch_kincaid': 0.0}
        
        # Упрощенная версия Flesch-Kincaid для русского языка
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {'flesch_kincaid': 0.0}
        
        avg_sentence_length = len(tokens) / len(sentences) if sentences else 0
        avg_word_length = np.mean([len(t) for t in tokens]) if tokens else 0
        
        # Адаптированная формула сложности
        complexity = avg_sentence_length * 0.5 + avg_word_length * 0.3
        
        return {'flesch_kincaid': complexity}


class Vectorizer:
    """Класс для векторизации текстов."""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.bow_vectorizer = None
        self.word2vec_model = None
        self.fasttext_model = None
        self.bert_tokenizer = None
        self.bert_model = None
    
    def fit_tfidf(self, texts: List[str], max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        """Обучение TF-IDF векторизатора."""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        self.tfidf_vectorizer.fit(texts)
        logger.info(f"TF-IDF обучен: {len(self.tfidf_vectorizer.vocabulary_)} признаков")
    
    def transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """Преобразование текстов в TF-IDF векторы."""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF векторизатор не обучен. Вызовите fit_tfidf() сначала.")
        return self.tfidf_vectorizer.transform(texts).toarray()
    
    def fit_bow(self, texts: List[str], max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        """Обучение Bag of Words векторизатора."""
        self.bow_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
        self.bow_vectorizer.fit(texts)
        logger.info(f"BoW обучен: {len(self.bow_vectorizer.vocabulary_)} признаков")
    
    def transform_bow(self, texts: List[str]) -> np.ndarray:
        """Преобразование текстов в BoW векторы."""
        if self.bow_vectorizer is None:
            raise ValueError("BoW векторизатор не обучен. Вызовите fit_bow() сначала.")
        return self.bow_vectorizer.transform(texts).toarray()
    
    def load_word2vec(self, model_path: str):
        """Загрузка модели Word2Vec."""
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim не установлен")
        self.word2vec_model = Word2Vec.load(model_path)
        logger.info(f"Word2Vec модель загружена: размерность {self.word2vec_model.wv.vector_size}")
    
    def load_fasttext(self, model_path: str):
        """Загрузка модели FastText."""
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim не установлен")
        self.fasttext_model = FastText.load(model_path)
        logger.info(f"FastText модель загружена: размерность {self.fasttext_model.wv.vector_size}")
    
    def get_word_embeddings(self, tokens: List[str], model_type: str = 'word2vec') -> Optional[np.ndarray]:
        """Получение эмбеддингов слов."""
        if model_type == 'word2vec' and self.word2vec_model:
            model = self.word2vec_model
        elif model_type == 'fasttext' and self.fasttext_model:
            model = self.fasttext_model
        else:
            return None
        
        embeddings = []
        for token in tokens:
            try:
                embeddings.append(model.wv[token])
            except KeyError:
                continue
        
        return np.array(embeddings) if embeddings else None
    
    def get_document_embedding(self, tokens: List[str], model_type: str = 'word2vec', method: str = 'mean') -> Optional[np.ndarray]:
        """Получение эмбеддинга документа из эмбеддингов слов."""
        word_embeddings = self.get_word_embeddings(tokens, model_type)
        
        if word_embeddings is None or len(word_embeddings) == 0:
            return None
        
        if method == 'mean':
            return np.mean(word_embeddings, axis=0)
        elif method == 'max':
            return np.max(word_embeddings, axis=0)
        elif method == 'sum':
            return np.sum(word_embeddings, axis=0)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
    
    def load_bert(self, model_name: str = 'DeepPavlov/rubert-base-cased'):
        """Загрузка BERT модели для русского языка."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers не установлен")
        
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.bert_model.eval()
        logger.info(f"BERT модель загружена: {model_name}")
    
    def get_bert_embeddings(self, texts: List[str], method: str = 'mean') -> np.ndarray:
        """Получение BERT эмбеддингов для текстов."""
        if self.bert_tokenizer is None or self.bert_model is None:
            raise ValueError("BERT модель не загружена. Вызовите load_bert() сначала.")
        
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.bert_tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                outputs = self.bert_model(**inputs)
                
                if method == 'mean':
                    # Усреднение по токенам (исключая padding)
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    masked_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
                    summed = torch.sum(masked_embeddings, dim=1)
                    counts = torch.sum(attention_mask, dim=1, keepdim=True)
                    mean_pooled = summed / counts
                    embeddings.append(mean_pooled.squeeze().numpy())
                elif method == 'cls':
                    # Использование [CLS] токена
                    embeddings.append(outputs.last_hidden_state[0, 0, :].numpy())
                else:
                    raise ValueError(f"Неизвестный метод: {method}")
        
        return np.array(embeddings)


def load_jsonl(file_path: str) -> List[Dict]:
    """Загрузка данных из JSONL файла."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_features(data: List[Dict], preprocessor: TextPreprocessor, 
                     feature_extractor: FeatureExtractor, vectorizer: Vectorizer,
                     use_tfidf: bool = True, use_embeddings: bool = False,
                     embedding_model_path: Optional[str] = None,
                     fit_vectorizer: bool = True) -> Tuple[np.ndarray, List]:
    """
    Подготовка признаков для классификации.
    
    Args:
        fit_vectorizer: Если True, обучает векторизатор (для train), если False - только трансформирует (для test)
    
    Returns:
        Tuple[features, labels]
    """
    texts = []
    labels = []
    
    # Определяем тип задачи
    if 'sentiment' in data[0]:
        task_type = 'binary'
        label_key = 'sentiment'
    elif 'category' in data[0]:
        task_type = 'multiclass'
        label_key = 'category'
    elif 'categories' in data[0]:
        task_type = 'multilabel'
        label_key = 'categories'
    else:
        raise ValueError("Неизвестный тип задачи")
    
    # Извлечение текстов и меток
    for item in data:
        full_text = f"{item.get('title', '')} {item.get('text', '')}".strip()
        texts.append(full_text)
        
        if task_type == 'multilabel':
            labels.append(item[label_key])
        else:
            labels.append(item[label_key])
    
    # Векторизация
    feature_vectors = []
    
    if use_tfidf:
        if fit_vectorizer:
            vectorizer.fit_tfidf(texts)
        tfidf_features = vectorizer.transform_tfidf(texts)
        feature_vectors.append(tfidf_features)
    
    if use_embeddings and embedding_model_path:
        # Загрузка модели эмбеддингов
        if 'word2vec' in embedding_model_path.lower():
            vectorizer.load_word2vec(embedding_model_path)
            embedding_type = 'word2vec'
        elif 'fasttext' in embedding_model_path.lower():
            vectorizer.load_fasttext(embedding_model_path)
            embedding_type = 'fasttext'
        else:
            embedding_type = 'word2vec'
        
        # Получение эмбеддингов документов
        doc_embeddings = []
        for text in texts:
            tokens = preprocessor.preprocess(text)
            embedding = vectorizer.get_document_embedding(tokens, embedding_type, method='mean')
            if embedding is not None:
                doc_embeddings.append(embedding)
            else:
                # Если не удалось получить эмбеддинг, используем нулевой вектор
                if embedding_type == 'word2vec' and vectorizer.word2vec_model:
                    dim = vectorizer.word2vec_model.wv.vector_size
                elif embedding_type == 'fasttext' and vectorizer.fasttext_model:
                    dim = vectorizer.fasttext_model.wv.vector_size
                else:
                    dim = 100
                doc_embeddings.append(np.zeros(dim))
        
        feature_vectors.append(np.array(doc_embeddings))
    
    # Статистические признаки
    stat_features = []
    for text in texts:
        stat_feat = feature_extractor.extract_statistical_features(text)
        stat_features.append(list(stat_feat.values()))
    
    feature_vectors.append(np.array(stat_features))
    
    # Объединение всех признаков
    if len(feature_vectors) > 1:
        features = np.hstack(feature_vectors)
    else:
        features = feature_vectors[0]
    
    return features, labels

