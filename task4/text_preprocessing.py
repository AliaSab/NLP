"""
Модуль для предобработки текстовых данных.
Поддерживает очистку, лемматизацию и токенизацию (whitespace, regex, BPE).
"""

import re
import json
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import logging

# Добавляем путь к Task1 для использования BPE моделей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Task1'))

# NLP библиотеки
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import pymorphy2
    PYMORPHY_AVAILABLE = True
except ImportError:
    PYMORPHY_AVAILABLE = False

# BPE токенизация
try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

try:
    import sentencepiece as sp
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Класс для предобработки текстов."""
    
    def __init__(self, language: str = 'ru', use_spacy: bool = True, use_pymorphy: bool = True):
        self.language = language
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_pymorphy = use_pymorphy and PYMORPHY_AVAILABLE
        
        # Инициализация spaCy
        if self.use_spacy:
            try:
                self.nlp = spacy.load('ru_core_news_sm')
                logger.info("spaCy модель загружена")
            except OSError:
                logger.warning("Русская модель spaCy не найдена. Используется pymorphy2.")
                self.use_spacy = False
        
        # Инициализация pymorphy2
        if self.use_pymorphy:
            try:
                self.morph = pymorphy2.MorphAnalyzer()
                logger.info("pymorphy2 инициализирован")
            except Exception as e:
                logger.warning(f"Ошибка инициализации pymorphy2: {e}")
                self.use_pymorphy = False
        
        # BPE модели
        self.bpe_tokenizers = {}
        self.sp_tokenizers = {}
    
    def clean_text(self, text: str) -> str:
        """Очистка текста от HTML, URL, специальных символов."""
        if not text:
            return ""
        
        # Удаление HTML тегов
        text = re.sub(r'<[^>]+>', '', text)
        
        # Удаление URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Удаление email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Нормализация пробелов
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Лемматизация токенов."""
        lemmatized = []
        
        if self.use_spacy:
            # Используем spaCy
            doc = self.nlp(' '.join(tokens))
            lemmatized = [token.lemma_ for token in doc if token.lemma_.strip()]
        elif self.use_pymorphy:
            # Используем pymorphy2
            for token in tokens:
                parsed = self.morph.parse(token)[0]
                lemmatized.append(parsed.normal_form)
        else:
            # Без лемматизации
            lemmatized = tokens
        
        return [t for t in lemmatized if t.strip()]
    
    def tokenize_whitespace(self, text: str) -> List[str]:
        """Токенизация по пробелам."""
        return text.split()
    
    def tokenize_regex(self, text: str, pattern: str = r'\w+') -> List[str]:
        """Токенизация с использованием регулярных выражений."""
        return re.findall(pattern, text)
    
    def load_bpe_model(self, model_path: str, vocab_size: Optional[int] = None) -> bool:
        """Загрузка BPE модели из Task1."""
        try:
            if TOKENIZERS_AVAILABLE:
                tokenizer = Tokenizer.from_file(model_path)
                key = f'bpe_{vocab_size}' if vocab_size else 'bpe_default'
                self.bpe_tokenizers[key] = tokenizer
                logger.info(f"BPE модель загружена: {model_path}")
                return True
        except Exception as e:
            logger.warning(f"Не удалось загрузить BPE модель {model_path}: {e}")
        
        return False
    
    def load_sentencepiece_model(self, model_path: str, vocab_size: Optional[int] = None) -> bool:
        """Загрузка SentencePiece модели из Task1."""
        try:
            if SENTENCEPIECE_AVAILABLE:
                sp_model = sp.SentencePieceProcessor()
                sp_model.load(model_path)
                key = f'sp_{vocab_size}' if vocab_size else 'sp_default'
                self.sp_tokenizers[key] = sp_model
                logger.info(f"SentencePiece модель загружена: {model_path}")
                return True
        except Exception as e:
            logger.warning(f"Не удалось загрузить SentencePiece модель {model_path}: {e}")
        
        return False
    
    def tokenize_bpe(self, text: str, vocab_size: Optional[int] = None) -> List[str]:
        """Токенизация с использованием BPE."""
        key = f'bpe_{vocab_size}' if vocab_size else 'bpe_default'
        
        if key in self.bpe_tokenizers:
            tokenizer = self.bpe_tokenizers[key]
            encoding = tokenizer.encode(text)
            return encoding.tokens
        elif 'bpe_default' in self.bpe_tokenizers:
            tokenizer = self.bpe_tokenizers['bpe_default']
            encoding = tokenizer.encode(text)
            return encoding.tokens
        else:
            logger.warning("BPE модель не загружена. Используется whitespace токенизация.")
            return self.tokenize_whitespace(text)
    
    def tokenize_sentencepiece(self, text: str, vocab_size: Optional[int] = None) -> List[str]:
        """Токенизация с использованием SentencePiece."""
        key = f'sp_{vocab_size}' if vocab_size else 'sp_default'
        
        if key in self.sp_tokenizers:
            sp_model = self.sp_tokenizers[key]
            return sp_model.encode(text, out_type=str)
        elif 'sp_default' in self.sp_tokenizers:
            sp_model = self.sp_tokenizers['sp_default']
            return sp_model.encode(text, out_type=str)
        else:
            logger.warning("SentencePiece модель не загружена. Используется whitespace токенизация.")
            return self.tokenize_whitespace(text)
    
    def preprocess(self, text: str, 
                   tokenization_method: str = 'whitespace',
                   lemmatize: bool = True,
                   clean: bool = True,
                   vocab_size: Optional[int] = None) -> List[str]:
        """
        Полная предобработка текста.
        
        Args:
            text: Исходный текст
            tokenization_method: Метод токенизации ('whitespace', 'regex', 'bpe', 'sentencepiece')
            lemmatize: Применять ли лемматизацию
            clean: Применять ли очистку
            vocab_size: Размер словаря для BPE/SentencePiece
        
        Returns:
            Список токенов
        """
        if not text:
            return []
        
        # Очистка
        if clean:
            text = self.clean_text(text)
        
        # Токенизация
        if tokenization_method == 'whitespace':
            tokens = self.tokenize_whitespace(text)
        elif tokenization_method == 'regex':
            tokens = self.tokenize_regex(text)
        elif tokenization_method == 'bpe':
            tokens = self.tokenize_bpe(text, vocab_size)
        elif tokenization_method == 'sentencepiece':
            tokens = self.tokenize_sentencepiece(text, vocab_size)
        else:
            logger.warning(f"Неизвестный метод токенизации: {tokenization_method}. Используется whitespace.")
            tokens = self.tokenize_whitespace(text)
        
        # Лемматизация
        if lemmatize and tokens:
            tokens = self.lemmatize(tokens)
        
        return [t for t in tokens if t.strip()]


def load_articles_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Загрузка статей из JSONL файла."""
    articles = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    articles.append(json.loads(line))
        logger.info(f"Загружено {len(articles)} статей из {file_path}")
    except Exception as e:
        logger.error(f"Ошибка загрузки файла {file_path}: {e}")
    
    return articles


def preprocess_articles(articles: List[Dict[str, Any]], 
                       preprocessor: TextPreprocessor,
                       tokenization_method: str = 'whitespace',
                       lemmatize: bool = True,
                       vocab_size: Optional[int] = None) -> List[List[str]]:
    """Предобработка списка статей."""
    processed = []
    
    for i, article in enumerate(articles):
        # Объединяем заголовок и текст
        text = f"{article.get('title', '')} {article.get('text', '')}".strip()
        
        tokens = preprocessor.preprocess(
            text,
            tokenization_method=tokenization_method,
            lemmatize=lemmatize,
            vocab_size=vocab_size
        )
        
        processed.append(tokens)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Обработано {i + 1}/{len(articles)} статей")
    
    return processed


if __name__ == "__main__":
    # Пример использования
    preprocessor = TextPreprocessor()
    
    # Попытка загрузить BPE модели из Task1
    task1_path = os.path.join(os.path.dirname(__file__), '..', 'Task1')
    bpe_8000_path = os.path.join(task1_path, 'bpe_model_8000.json')
    sp_8000_path = os.path.join(task1_path, 'sp_bpe_8000.model')
    
    if os.path.exists(bpe_8000_path):
        preprocessor.load_bpe_model(bpe_8000_path, vocab_size=8000)
    
    if os.path.exists(sp_8000_path):
        preprocessor.load_sentencepiece_model(sp_8000_path, vocab_size=8000)
    
    # Тестовый текст
    test_text = "Это пример текста для тестирования предобработки. Текст содержит различные слова и фразы."
    
    print("Исходный текст:", test_text)
    print("\nТокенизация whitespace:", preprocessor.preprocess(test_text, tokenization_method='whitespace'))
    print("\nТокенизация regex:", preprocessor.preprocess(test_text, tokenization_method='regex'))
    print("\nТокенизация BPE:", preprocessor.preprocess(test_text, tokenization_method='bpe', vocab_size=8000))
    print("\nТокенизация SentencePiece:", preprocessor.preprocess(test_text, tokenization_method='sentencepiece', vocab_size=8000))


